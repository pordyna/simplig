import scipy.constants as cs
import numpy as np
import openpmd_api as io
from .. import ureg
import numba
from warnings import warn, filterwarnings
from tqdm import TqdmExperimentalWarning
import dask.array as da

filterwarnings("ignore", category=TqdmExperimentalWarning)

from typing import Mapping, Union, Optional
from numpy.typing import ArrayLike

from tqdm.autonotebook import tqdm
from copy import copy, deepcopy

from ..data import FieldMetaData
from ..data import rotate
from ..data import DescribedField

# type defs:
Domain = tuple[Union[ureg.Quantity, None], Union[ureg.Quantity, None]]

if io.variants["mpi"]:
    try:
        from mpi4py import MPI

        HAVE_MPI = True
    except ImportError:
        warn(
            """openPMD-api was built with support for MPI,
            but mpi4py Python package was not found.
            Will continue in serial mode.""",
            ImportWarning,
        )
        HAVE_MPI = False
else:
    HAVE_MPI = False


class FallbackMPICommunicator:
    def __init__(self):
        self.size = 1
        self.rank = 0


# just for better static code analysis and hints
if HAVE_MPI:
    CommType = MPI.Comm
else:
    CommType = FallbackMPICommunicator


def get_chunk_1d(offset_total: int, stop_total: int, chunk_idx: int, n_chunks: int):
    extent_total = stop_total - offset_total
    stride = extent_total // n_chunks
    rest = extent_total % n_chunks

    # local function f computes the offset of a rank
    # for more equal balancing, we want the start index
    # at the upper gaussian bracket of (N/n*rank)
    # where N the size of the dataset in dimension dim
    # and n the MPI size
    # for avoiding integer overflow, this is the same as:
    # (N div n)*rank + round((N%n)/n*rank)
    def f(rank):
        res = stride * rank
        pad_divident = rest * rank
        pad = pad_divident // n_chunks
        if pad * n_chunks < pad_divident:
            pad += 1
        return res + pad

    offset = f(chunk_idx)
    extent = extent_total
    if chunk_idx >= n_chunks - 1:
        extent -= offset
    else:
        extent = f(chunk_idx + 1) - offset
    offset += offset_total
    return offset, extent


@numba.jit(nopython=True, parallel=True)
def _process_loaded_iteration_data_0(
    tmp_sum, out_volume, beam_index_arr, prop_axis_pos_arr, interpolation_coeff_arr, prop_min
):
    # distribute values
    for ii in numba.prange(len(beam_index_arr)):
        beam_idx = beam_index_arr[ii]
        prop_pos = prop_axis_pos_arr[ii]
        coeff_local = interpolation_coeff_arr[ii]
        out_volume[beam_idx, :, :] += tmp_sum[prop_pos - prop_min, :, :] * coeff_local


@numba.jit(nopython=True, parallel=True)
def _process_loaded_iteration_data_1(
    tmp_sum, out_volume, beam_index_arr, prop_axis_pos_arr, interpolation_coeff_arr, prop_min
):
    # distribute values
    for ii in numba.prange(len(beam_index_arr)):
        beam_idx = beam_index_arr[ii]
        prop_pos = prop_axis_pos_arr[ii]
        coeff_local = interpolation_coeff_arr[ii]
        out_volume[beam_idx, :, :] += tmp_sum[:, prop_pos - prop_min, :] * coeff_local


@numba.jit(nopython=True, parallel=True)
def _process_loaded_iteration_data_2(
    tmp_sum, out_volume, beam_index_arr, prop_axis_pos_arr, interpolation_coeff_arr, prop_min
):
    # distribute values
    for ii in numba.prange(len(beam_index_arr)):
        beam_idx = beam_index_arr[ii]
        prop_pos = prop_axis_pos_arr[ii]
        coeff_local = interpolation_coeff_arr[ii]
        out_volume[beam_idx, :, :] += tmp_sum[:, :, prop_pos - prop_min] * coeff_local


class SAXSPropagator:
    def __init__(
        self,
        series_path,
        density_fields,
        sim_write_interval,
        t_0=0.0 * ureg.second,
        first_avail_iteration=None,
        last_avail_iteration=None,
        linear_read=False,
        rotation_axis=None,
        rotation_angle=None,
        read_options="{}",
        axis_labels=None,
        checkpoint_series_path=None,
        checkpoint_options="{}",
        checkpoint_interval=None,
        force_mpi=False,
    ):
        self.prop_axis_str = None
        self.chunking_axis = None
        self.prop_end = None
        self.prop_start = None

        # private:
        self._unit_length_int = None
        self._interpolation_coeff = None
        self._used_iterations = None
        self._alpha = None
        self._times = None
        self._it_max = None
        self._it_min = None
        self._detection_duration_int_time = None
        self.prop_axis = None
        self._last_cell_in_prop = None
        self._first_cell_in_prop = None
        self._volume = None
        self._checkpoint_series = None
        self._last_iteration_to_process = None
        self._last_processed_iteration = -1
        self._restarted_from_iteration = None

        if HAVE_MPI:
            self._comm: CommType = MPI.COMM_WORLD
        else:
            assert not force_mpi, "not mpi support, but explicitly requested!"
            self._comm: CommType = FallbackMPICommunicator()

        if checkpoint_series_path is not None:
            try:
                if HAVE_MPI:
                    self._checkpoint_series = io.Series(
                        checkpoint_series_path,
                        io.Access_Type.read_write,
                        self._comm,
                        checkpoint_options,
                    )
                else:
                    self._checkpoint_series = io.Series(
                        checkpoint_series_path, io.Access_Type.read_write, checkpoint_options
                    )
            except:
                if HAVE_MPI:
                    self._checkpoint_series = io.Series(
                        checkpoint_series_path,
                        io.Access_Type.create,
                        self._comm,
                        checkpoint_options,
                    )
                else:
                    self._checkpoint_series = io.Series(
                        checkpoint_series_path, io.Access_Type.create, checkpoint_options
                    )
            assert (
                self._checkpoint_series.iteration_encoding == io.Iteration_Encoding.file_based
            ), "Checkpointing supports only file-based iteration encoding."

        self.linear_read = linear_read

        if linear_read:
            access_mode = io.Access_Type.read_linear
            assert first_avail_iteration is not None
            assert last_avail_iteration is not None
        else:
            access_mode = io.Access_Type.read_only

        if HAVE_MPI:
            self._series = io.Series(str(series_path), access_mode, self._comm, read_options)
        else:
            self._series = io.Series(str(series_path), access_mode, read_options)

        if first_avail_iteration is None:
            self._first_avail_iteration = list(self._series.iterations)[0]

        else:
            self._first_avail_iteration = first_avail_iteration
        if last_avail_iteration is None:
            self._last_avail_iteration = list(self._series.iterations)[-1]
        else:
            self._last_avail_iteration = last_avail_iteration

        if linear_read:
            read_iterations = self._series.read_iterations()
            first_iteration = next(iter(read_iterations))
        else:
            first_iteration = self._series.iterations[self._first_avail_iteration]
        first_iteration.open()
        self.simulation_step_duration = (
            first_iteration.dt * first_iteration.time_unit_SI * ureg.second
        )

        example_mr: io.Mesh = first_iteration.meshes[density_fields[0]]
        example_mrc: io.Mesh_Record_Component = example_mr[io.Mesh_Record_Component.SCALAR]

        for field in density_fields[1:]:
            mr: io.Mesh = first_iteration.meshes[field]
            mrc: io.Mesh_Record_Component = mr[io.Mesh_Record_Component.SCALAR]
            assert np.allclose(
                np.array(mr.grid_spacing) * mr.grid_unit_SI,
                np.array(example_mr.grid_spacing) * example_mr.grid_unit_SI,
            )
            assert mr.axis_labels == example_mr.axis_labels
            assert mrc.shape == example_mrc.shape
            assert np.allclose(
                np.array(example_mr.grid_global_offset) * example_mr.grid_unit_SI,
                np.array(mr.grid_global_offset) * mr.grid_unit_SI,
            )
            assert np.allclose(example_mrc.position, mrc.position)
            assert np.allclose(example_mrc.unit_SI, mrc.unit_SI)
            assert example_mrc.dtype is mrc.dtype

        self._in_unit_SI = example_mrc.unit_SI
        if axis_labels is None:
            axis_labels = example_mr.axis_labels
        axis_map = {key: value for value, key in enumerate(axis_labels)}
        self._field_position = example_mrc.position
        self._read_dtype = example_mrc.dtype

        self.sim_write_interval = sim_write_interval
        self.t_0 = t_0

        self.density_fields = density_fields

        # internal units:
        self._unit_time_int = self.sim_write_interval * self.simulation_step_duration

        self.rotation_axis_idx: Union[None, int] = None
        self.rotation_angle_rad: Union[None, float] = None
        self.rotation_axis_str = rotation_axis
        if rotation_axis is not None:
            self.rotation_axis_idx = axis_map[rotation_axis]
            self.rotation_angle_rad = rotation_angle.to(ureg.radian).magnitude

        pre_rot_meta = FieldMetaData(
            ndim=3,
            axis_labels=axis_labels,
            first_cell_positions=np.array(example_mr.grid_global_offset)
            * example_mr.grid_unit_SI
            * ureg.meter,
            shape=example_mrc.shape,
            cell_size=np.array(example_mr.grid_spacing) * example_mr.grid_unit_SI * ureg.meter,
            in_cell_position=self._field_position,
            value_unit=1 / ureg.meter**3,
            time=0 * ureg.second,
            field_description="",
        )
        if rotation_axis is not None:
            past_rot_meta = rotate(
                DescribedField(None, pre_rot_meta),
                self.rotation_angle_rad,
                rotation_axis_idx=self.rotation_axis_idx,
            ).meta
        else:
            past_rot_meta = pre_rot_meta

        self._pre_rot_sim_shape = pre_rot_meta.shape
        # the following block needs changing when applying rotation:
        # celll size doesn't change buuut the cells have to cubes.
        self._cell_sizes = past_rot_meta.cell_size
        self._simulation_shape = past_rot_meta.shape
        self._grid_offset = past_rot_meta.first_cell_positions
        # x - > x' when using rotation
        self._axis_map = {key: value for value, key in enumerate(past_rot_meta.axis_labels)}
        self._in_axis_labels = past_rot_meta.axis_labels

        # may need to change _read_dtype

        self._total_offset = [0, 0, 0]
        self._total_extent = list(self._simulation_shape)

        self._chunk_offset = None
        self._chunk_extent = None

        self.output_series = None

    def _position_to_idx(self, position: ureg.Quantity, axis: str):
        axis_idx = self._axis_map[axis]
        dx = self._cell_sizes[axis_idx].to_base_units()
        offset = self._grid_offset[axis_idx].to_base_units()
        offset += self._field_position[axis_idx] * dx

        position = position.to_base_units()
        cell_idx = (position - offset) / dx
        cell_idx.ito_reduced_units()
        cell_idx = int(round(cell_idx))
        n_cells = self._simulation_shape[axis_idx]
        assert cell_idx >= 0
        assert cell_idx <= n_cells
        if cell_idx == n_cells:
            cell_idx = n_cells - 1
        return cell_idx

    def _idx_to_position(self, idx: int, axis: str):
        axis_idx = self._axis_map[axis]
        dx = self._cell_sizes[axis_idx].to_base_units()
        offset = self._grid_offset[axis_idx].to_base_units()
        offset += self._field_position[axis_idx] * dx
        position = dx * idx + offset
        return position

    def _domain_to_offset_extent(self, domain: Domain, axis: str):
        start_pos = domain[0]
        end_pos = domain[1]
        if start_pos is not None:
            start_pos = self._position_to_idx(start_pos, axis)
        else:
            start_pos = 0
        if end_pos is not None:
            end_pos = self._position_to_idx(end_pos, axis)
        else:
            end_pos = self._simulation_shape[self._axis_map[axis]]
        return start_pos, end_pos - start_pos

    def setup_transverse_domain(self, domain_dict: Mapping[str, Domain]):
        for key in domain_dict:
            assert key in self._axis_map

        for axis, domain in domain_dict.items():
            idx = self._axis_map[axis]
            offset, extent = self._domain_to_offset_extent(domain, axis)
            self._total_offset[idx] = offset
            self._total_extent[idx] = extent

    def set_chunking(self, chunking_axis: str):
        self.chunking_axis = chunking_axis
        chunking_axis_idx = self._axis_map[chunking_axis]
        assert chunking_axis_idx != self.prop_axis

        start = self._total_offset[chunking_axis_idx]
        stop = self._total_extent[chunking_axis_idx] + start
        chunk_idx = self._comm.rank
        n_chunks = self._comm.size
        offset, extent = get_chunk_1d(start, stop, chunk_idx, n_chunks)
        self._chunk_offset = deepcopy(self._total_offset)
        self._chunk_extent = deepcopy(self._total_extent)
        self._chunk_offset[chunking_axis_idx] = offset
        self._chunk_extent[chunking_axis_idx] = extent

    def setup_propagation(self, prop_domain: Domain, start_time, axis, detection_duration):
        self.prop_axis_str = axis
        self.prop_axis = self._axis_map[axis]
        self._unit_length_int = self._cell_sizes[self.prop_axis]

        prop_offset, prop_extent = self._domain_to_offset_extent(prop_domain, axis)
        self._first_cell_in_prop = prop_offset
        self._last_cell_in_prop = prop_extent + prop_offset

        self.prop_start = self._idx_to_position(self._first_cell_in_prop, axis)
        self.prop_end = self._idx_to_position(self._last_cell_in_prop, axis)

        speed_of_light = cs.c * ureg.meter / ureg.second
        speed_of_light_int = speed_of_light / self._unit_length_int * self._unit_time_int
        speed_of_light_int.ito_reduced_units()
        propagation_duration = (self.prop_end - self.prop_start) / speed_of_light
        propagation_duration.ito_reduced_units()
        self._detection_duration_int_time = detection_duration / self._unit_time_int
        self._detection_duration_int_time.ito_reduced_units()
        self._detection_duration_int_time = int(round(self._detection_duration_int_time))

        start_time += self.t_0
        start_time_int = start_time / self._unit_time_int
        start_time_int.ito_reduced_units()

        cells_in_prop = self._last_cell_in_prop - self._first_cell_in_prop
        n_m = self._detection_duration_int_time
        n_z = cells_in_prop
        times = np.fromfunction(
            lambda m, z: m + z / speed_of_light_int, (n_m, n_z), dtype=np.float32
        )
        times = times.magnitude
        times += start_time_int.magnitude
        times_min = np.floor(times).astype(np.uint32)
        times_max = times_min + 1
        self._it_min = times_min * self.sim_write_interval
        self._it_max = times_max * self.sim_write_interval
        self._interpolation_coeff = times - times_min

        self._used_iterations = np.sort(np.unique(np.concatenate((self._it_min, self._it_max))))
        if not self.linear_read:
            iterations = self._series.iterations
            is_in = np.isin(self._used_iterations, iterations)
            assert np.all(is_in), f"missing iterations: {np.unique(self._used_iterations[~is_in])}"
        else:
            assert self._used_iterations[0] >= self._first_avail_iteration
            assert self._used_iterations[-1] <= self._last_avail_iteration
        self._last_iteration_to_process = self._used_iterations[-1]

    def gather_results(self):
        size = self._comm.size
        if size == 1:
            return self._volume
        rank = self._comm.rank

        # send sizes
        local_chunk_shape = np.array(self._volume.shape, dtype="i")
        ndim = len(self._simulation_shape)
        chunk_shapes = None
        if rank == 0:
            chunk_shapes = np.empty((size, ndim), dtype="i")
        self._comm.Gather(sendbuf=local_chunk_shape, recvbuf=chunk_shapes, root=0)
        dtype = self._volume.dtype
        recvbuffer = None
        counts = None
        displacements = None
        if rank == 0:
            counts = np.prod(chunk_shapes, axis=1)
            displacements = np.cumsum(counts) - counts
            recvbuffer = np.empty(np.sum(counts), dtype=dtype)
        self._comm.Gatherv(
            sendbuf=self._volume, recvbuf=(recvbuffer, (counts, displacements)), root=0
        )

        if rank == 0:
            chunking_axis_idx = self._axis_map[self.chunking_axis]
            full_shape = np.copy(chunk_shapes[0, :])
            full_shape[chunking_axis_idx + 1] = np.sum(chunk_shapes[:, chunking_axis_idx + 1])
            result = np.empty(full_shape, dtype=dtype)
            slicing = [slice(None)] * len(full_shape)
            offsets = (
                np.cumsum(chunk_shapes[:, chunking_axis_idx + 1])
                - chunk_shapes[:, chunking_axis_idx + 1]
            )
            for i in range(size):
                slicing_local = slicing
                slicing_local[chunking_axis_idx + 1] = slice(
                    offsets[i], offsets[i] + chunk_shapes[i, chunking_axis_idx + 1]
                )
                result[tuple(slicing_local)] = recvbuffer[
                    displacements[i] : displacements[i] + counts[i]
                ].reshape(chunk_shapes[i])
            return result

    def _open_output_series(self, out_series_path, options):
        if HAVE_MPI:
            self.output_series = io.Series(
                str(out_series_path), io.Access_Type.create, self._comm, options=options
            )
        else:
            self.output_series = io.Series(
                str(out_series_path), io.Access_Type.create, options=options
            )
        self.output_series.set_software("simplig")

    def close_output_series(self):
        self.output_series.close()
        del self.output_series
        self.output_series = None

    def _write_iteration(self, it, iteration_idx):
        it.open()
        mesh: io.Mesh = it.meshes["integrated_density"]
        mrc: io.Mesh_Record_Component = mesh[io.Mesh_Record_Component.SCALAR]

        t_0 = self.t_0.to("fs")
        t_start = (self.simulation_step_duration * self._used_iterations[0]).to("fs") - t_0
        it.set_time(t_start.magnitude)
        it.set_dt(self.simulation_step_duration.to("fs").magnitude * iteration_idx)
        it.set_time_unit_SI(1.0e-15)
        it.set_attribute("t_0", t_0.magnitude)

        unit_length = self._unit_length_int.to("meter").magnitude

        global_extent = deepcopy(self._total_extent)
        global_extent.pop(self.prop_axis)
        global_extent.insert(0, self._detection_duration_int_time)

        global_offset = deepcopy(self._total_offset)
        global_offset = list(global_offset + self._grid_offset.to("meter").magnitude / unit_length)
        global_offset.pop(self.prop_axis)
        global_offset.insert(0, 0)

        local_offset = deepcopy(self._chunk_offset)
        local_extent = deepcopy(self._chunk_extent)
        for dd in range(len(local_offset)):
            local_offset[dd] -= self._total_offset[dd]
        local_offset.pop(self.prop_axis)
        local_offset.insert(0, 0)

        local_extent.pop(self.prop_axis)
        local_extent.insert(0, self._detection_duration_int_time)

        grid_spacing = deepcopy(list(self._cell_sizes.magnitude / unit_length))
        grid_spacing.pop(self.prop_axis)
        time_axis_spacing = (self._unit_time_int.to(ureg.second).magnitude * cs.c) / unit_length
        grid_spacing.insert(0, time_axis_spacing)

        axis_labels = deepcopy(list(self._in_axis_labels))
        axis_labels.pop(self.prop_axis)
        axis_labels.insert(0, "ct")

        position = deepcopy(list(self._field_position))
        position.pop(self.prop_axis)
        position.insert(0, 0.0)
        if self.rotation_axis_str is not None:
            mesh.set_attribute("initialRotationAxis", self.rotation_axis_str)
            mesh.set_attribute("rotationAngleRad", self.rotation_angle_rad)

        mesh.set_attribute("propagationAxis", self.prop_axis_str)
        mesh.set_attribute(
            "propagationDomain",
            [
                self.prop_start.to("m").magnitude,
                self.prop_end.to("m").magnitude
                + self._cell_sizes[self.prop_axis].to("m").magnitude,
            ],
        )
        mesh.set_attribute("propagationDomainUnitSI", 1.0)
        mesh.set_grid_global_offset(global_offset)
        mesh.set_grid_unit_SI(unit_length)
        mesh.set_grid_spacing(grid_spacing)
        mesh.set_axis_labels(axis_labels)
        mesh.set_unit_dimension({io.Unit_Dimension.L: -2})
        mrc.set_unit_SI(self._in_unit_SI * unit_length)
        mrc.set_attribute("position", position)

        dataset = io.Dataset(self._volume.dtype, global_extent)
        mrc.reset_dataset(dataset)
        mrc.store_chunk(self._volume, offset=local_offset, extent=local_extent)
        it.close()

    def write_to_openpmd(self, out_series_path, options="{}", iteration_idx=None, finalize=True):
        if iteration_idx is None:
            iteration_idx = self._last_processed_iteration
        if self.output_series is None:
            self._open_output_series(out_series_path, options)
        it: io.Iteration = self.output_series.iterations[iteration_idx]
        self._write_iteration(it, iteration_idx)
        if finalize:
            self.close_output_series()

    def _fill_read_buffer(self, iteration, mrc_list, offset, extent):
        read_buffer = np.empty((len(mrc_list), *extent), dtype=self._read_dtype)
        for ii, mrc in enumerate(mrc_list):
            mrc.load_chunk(read_buffer[ii], offset=offset, extent=extent)
        self._series.flush()
        iteration.close()
        return np.sum(read_buffer, axis=0)

    def _load_data_no_rotation(self, iteration, mrc_list, offset, extent):
        read_buffer = self._fill_read_buffer(iteration, mrc_list, offset, extent)
        read_buffer = read_buffer.astype(np.float64, copy=False)
        return read_buffer

    def _load_data_with_rotation(self, iteration, mrc_list, offset, extent):
        # It is ensured that the rotation axis is the same as the chunking axis.
        # We need the whole simulation domain in the rotation plane, but we can
        # restrict the load to chunk along the rotation angle. The chunking also
        # doesn't change since the axis will remain unchanged.

        pre_rot_offset = [0, 0, 0]
        pre_rot_extent = list(copy(self._pre_rot_sim_shape))
        pre_rot_offset[self.rotation_axis_idx] = offset[self.rotation_axis_idx]
        pre_rot_extent[self.rotation_axis_idx] = extent[self.rotation_axis_idx]

        read_buffer = self._fill_read_buffer(iteration, mrc_list, pre_rot_offset, pre_rot_extent)
        read_buffer = rotate(
            read_buffer, self.rotation_angle_rad, rotation_axis_idx=self.rotation_axis_idx
        )
        slicing = [None, None, None]
        for dd in range(3):
            slicing[dd] = slice(offset[dd], offset[dd] + extent[dd])
        slicing[self.rotation_axis_idx] = slice(None)
        slicing = tuple(slicing)
        read_buffer = read_buffer[slicing]
        return read_buffer.astype(np.float64, copy=False)

    def _load_data(self, iteration, mrc_list, offset, extent):
        if self.rotation_axis_idx is not None:
            return self._load_data_with_rotation(iteration, mrc_list, offset, extent)
        else:
            return self._load_data_no_rotation(iteration, mrc_list, offset, extent)

    def _restart_from_checkpoint(self, checkpoint_iteration):
        assert self._checkpoint_series is not None, "Provide checkpoint series!"
        it = self._checkpoint_series[checkpoint_iteration]
        it.open()
        mrc = it.meshes["integrated_density"][io.Mesh_Record_Component.SCALAR]
        self._volume = mrc[:]
        self._checkpoint_series.flush()
        if self._comm.rank == 0:
            print(f"Restarting from{checkpoint_iteration}", flush=True)
        self._restarted_from_iteration = checkpoint_iteration
        self._last_processed_iteration = checkpoint_iteration

    def _write_checkpoint(self):
        if self._comm.rank == 0:
            print(f"Writing checkpoint {self._last_processed_iteration}", flush=True)
        assert self._checkpoint_series is not None, "Provide checkpoint series!"
        it = self._checkpoint_series.iterations[self._last_processed_iteration]
        self._write_iteration(it, self._last_processed_iteration)
        if self._comm.rank == 0:
            print(f"Finished writing checkpoint {self._last_processed_iteration}", flush=True)

    def __call__(
        self,
        disable_progress=None,
        tqdm_kwargs=None,
        dump_every_step=False,
        openpmd_kwargs=None,
        restart_iteration=None,
        try_restart=False,
    ):
        if self.prop_axis == 0:
            _process_loaded_iteration_data = _process_loaded_iteration_data_0
        elif self.prop_axis == 1:
            _process_loaded_iteration_data = _process_loaded_iteration_data_1
        elif self.prop_axis == 2:
            _process_loaded_iteration_data = _process_loaded_iteration_data_2
        else:
            raise NotImplementedError("This code only works with 3Dim data.")
        if tqdm_kwargs is None:
            tqdm_kwargs = {}
        if openpmd_kwargs is None:
            openpmd_kwargs = {}
        shape = deepcopy(self._chunk_extent)
        shape.pop(self.prop_axis)
        shape.insert(0, self._detection_duration_int_time)
        shape = tuple(shape)
        if restart_iteration is not None:
            self._restart_from_checkpoint(restart_iteration)
        elif try_restart and len(self._checkpoint_series.iterations) > 0:
            self._restart_from_checkpoint(self._checkpoint_series.iterations[-1])
        else:
            if self._comm.rank == 0:
                print("No restart", flush=True)
            self._volume = np.zeros(shape, dtype=np.float64)

        def iteration_loop(iteration_l: io.Iteration, iteration_idx_l: int):
            # Check if we have been here already (can happen with checkpointing)
            if iteration_idx_l <= self._last_processed_iteration:
                return
            # Find slices needed from this iteration
            where_min = np.where(self._it_min == iteration_idx_l)
            where_max = np.where(self._it_max == iteration_idx_l)
            if where_min[0].size == 0 and where_max[0].size == 0:
                return
            iteration_l.open()
            # indices of beam slices that are in the integration volume at this time step
            beam_index_arr = np.concatenate((where_min[0], where_max[0]))
            # density slices positions along the propagation direction
            prop_axis_pos_arr = np.concatenate((where_min[1], where_max[1]))
            interpolation_coeff_arr = np.concatenate(
                (self._interpolation_coeff[where_min], (1.0 - self._interpolation_coeff[where_max]))
            )
            prop_min = np.min(prop_axis_pos_arr)
            prop_max = np.max(prop_axis_pos_arr)

            mrc_list: list[io.Mesh_Record_Component] = [
                iteration_l.meshes[field][io.Mesh_Record_Component.SCALAR]
                for field in self.density_fields
            ]

            start, stop = (
                self._first_cell_in_prop + prop_min,
                self._first_cell_in_prop + prop_max + 1,
            )
            offset = deepcopy(list(self._chunk_offset))
            extent = deepcopy(list(self._chunk_extent))
            offset[self.prop_axis] = start
            extent[self.prop_axis] = stop - start

            tmp_sum = self._load_data(iteration_l, mrc_list, offset, extent)
            _process_loaded_iteration_data(
                tmp_sum,
                self._volume,
                beam_index_arr,
                prop_axis_pos_arr,
                interpolation_coeff_arr,
                prop_min,
            )
            if dump_every_step:
                self.write_to_openpmd(
                    **openpmd_kwargs, finalize=False, iteration_idx=iteration_idx_l
                )
            self._last_processed_iteration = iteration_idx_l
            pbar.update(1)

        def _tqdm():
            if self._restarted_from_iteration is None:
                initial = 0
            else:
                initial = np.searchsorted(
                    self._used_iterations, self._restarted_from_iteration, side="right"
                )
            return tqdm(
                position=self._comm.rank,
                desc=f"MPI rank {self._comm.rank}: ",
                disable=disable_progress,
                initial=initial,
                **tqdm_kwargs,
                total=len(self._used_iterations),
            )

        if self.linear_read:
            with _tqdm() as pbar:
                for iteration in self._series.read_iterations():
                    if iteration.iteration_index > self._last_iteration_to_process:
                        break
                    iteration_loop(iteration, iteration.iteration_index)
        else:
            with _tqdm() as pbar:
                for iteration_idx in self._used_iterations:
                    iteration = self._series.iterations[iteration_idx]
                    iteration_loop(iteration, iteration_idx)
        # Finished processing iterations
        finished = self._last_processed_iteration >= self._last_iteration_to_process
        if not finished and self._checkpoint_series is not None:
            self._write_checkpoint()
            self._checkpoint_series.close()
        return finished


def to_intensity(
    field: Union[DescribedField, ArrayLike],
    photons_in_pulse: int,
    wavelength,
    pulse_shape: Optional[ArrayLike],
    pulse_profile: Optional[ArrayLike],
    dask_fft=False,
):
    volume = field.data
    volume_metadata = field.meta

    volume = volume * np.sqrt(pulse_profile[None, ...])
    if dask_fft:
        volume = da.from_array(volume, chunks={0: "auto", 1: -1, 2: -1})
        intensity = da.abs((da.fft.fft2(volume))) ** 2
        intensity = intensity.compute()
    else:
        intensity = np.abs(np.fft.fft2(volume)) ** 2
    intensity = np.fft.fftshift(intensity, axes=(-1, -2))
    cell_size = list(volume_metadata.cell_size)
    intensity *= (cell_size[1] * cell_size[2]).to(1 / volume_metadata.value_unit)

    axis_labels = list(volume_metadata.axis_labels)
    axis_labels[0] = "t"
    for i, label in enumerate(axis_labels[1:]):
        axis_labels[i + 1] = "q_" + label
    intensity *= pulse_shape[:, None, None]

    cell_size[0] = (cell_size[0] / (cs.c * ureg.meter / ureg.second)).to("fs")
    intensity *= photons_in_pulse
    intensity *= wavelength**2
    intensity *= cs.physical_constants["classical electron radius"][0] ** 2
    intensity *= cell_size[0].to(ureg.second).magnitude

    ndim = volume_metadata.ndim
    axis_labels = list(volume_metadata.axis_labels)
    axis_labels[0] = "t"
    for i, label in enumerate(axis_labels[1:]):
        axis_labels[i + 1] = "q_" + label
    first_cell_positions = list(volume_metadata.first_cell_positions)
    first_cell_positions[0] = (first_cell_positions[0] / (cs.c * ureg.meter / ureg.second)).to("fs")
    q1 = np.fft.fftshift(
        np.fft.fftfreq(volume.shape[1], d=volume_metadata.cell_size[1].magnitude / (2 * np.pi))
    )
    q2 = np.fft.fftshift(
        np.fft.fftfreq(volume.shape[2], d=volume_metadata.cell_size[2].magnitude / (2 * np.pi))
    )
    first_cell_positions[1] = q1[0] / volume_metadata.cell_size[1].units
    first_cell_positions[2] = q2[0] / volume_metadata.cell_size[2].units
    shape = intensity.shape
    cell_size[1] = (q1[1] - q1[0]) / volume_metadata.cell_size[1].units
    cell_size[2] = (q2[1] - q2[0]) / volume_metadata.cell_size[2].units
    in_cell_position = volume_metadata.in_cell_position
    value_unit = ureg.Quantity(1).units
    time = volume_metadata.time
    field_description = "Instantaneous intensity"
    value_symbol = "I"

    metadata = FieldMetaData(
        ndim=ndim,
        axis_labels=axis_labels,
        first_cell_positions=first_cell_positions,
        shape=shape,
        cell_size=cell_size,
        in_cell_position=in_cell_position,
        value_unit=value_unit,
        time=time,
        field_description=field_description,
        value_symbol=value_symbol,
    )
    return DescribedField(intensity, metadata)
