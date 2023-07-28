import scipy.constants as cs
import numpy as np
import openpmd_api as io
from .. import ureg
import numba
from warnings import warn, filterwarnings
from tqdm import TqdmExperimentalWarning

filterwarnings("ignore", category=TqdmExperimentalWarning)

from typing import Mapping, Union, Optional
from numpy.typing import ArrayLike

from tqdm.autonotebook import tqdm
from copy import deepcopy

from ..data import FieldMetaData

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


@numba.jit(nopython=True, parallel=False)
def _process_loaded_iteration_data_0(
    read_buffer, out_volume, beam_index_arr, prop_axis_pos_arr, interpolation_coeff_arr, prop_min
):
    tmp_sum = np.sum(read_buffer, axis=0)
    # distribute values
    for ii in numba.prange(len(beam_index_arr)):
        beam_idx = beam_index_arr[ii]
        prop_pos = prop_axis_pos_arr[ii]
        coeff_local = interpolation_coeff_arr[ii]
        out_volume[beam_idx, :, :] += tmp_sum[prop_pos - prop_min, :, :] * coeff_local


@numba.jit(nopython=True, parallel=False)
def _process_loaded_iteration_data_1(
    read_buffer, out_volume, beam_index_arr, prop_axis_pos_arr, interpolation_coeff_arr, prop_min
):
    tmp_sum = np.sum(read_buffer, axis=0)
    # distribute values
    for ii in numba.prange(len(beam_index_arr)):
        beam_idx = beam_index_arr[ii]
        prop_pos = prop_axis_pos_arr[ii]
        coeff_local = interpolation_coeff_arr[ii]
        out_volume[beam_idx, :, :] += tmp_sum[:, prop_pos - prop_min, :] * coeff_local


@numba.jit(nopython=True, parallel=False)
def _process_loaded_iteration_data_2(
    read_buffer, out_volume, beam_index_arr, prop_axis_pos_arr, interpolation_coeff_arr, prop_min
):
    tmp_sum = np.sum(read_buffer, axis=0)
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

        if HAVE_MPI:
            self._comm: CommType = MPI.COMM_WORLD
        else:
            self._comm: CommType = FallbackMPICommunicator()

        if HAVE_MPI:
            self._series = io.Series(str(series_path), io.Access_Type.read_only, self._comm)
        else:
            self._series = io.Series(str(series_path), io.Access_Type.read_only)

        if first_avail_iteration is None:
            self._first_avail_iteration = list(self._series.iterations)[0]
        else:
            self._first_avail_iteration = first_avail_iteration
        if last_avail_iteration is None:
            self._last_avail_iteration = list(self._series.iterations)[-1]
        else:
            self._last_avail_iteration = last_avail_iteration

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
            assert np.isclose(
                np.array(mr.grid_spacing) * mr.grid_unit_SI,
                np.array(example_mr.grid_spacing) * example_mr.grid_unit_SI,
            )
            assert mr.axis_labels == example_mr.axis_labels
            assert mrc.shape == example_mrc.shape
            assert np.isclose(
                np.array(example_mr.grid_global_offset) * example_mr.grid_unit_SI,
                np.array(mr.grid_global_offset) * mr.grid_unit_SI,
            )
            assert np.isclose(example_mrc.position, mrc.position)
            assert np.isclose(example_mrc.unit_SI, mrc.unit_SI)
            assert example_mrc.dtype is mrc.dtype

        self._in_unit_SI = example_mrc.unit_SI

        self._cell_sizes = np.array(example_mr.grid_spacing) * example_mr.grid_unit_SI * ureg.meter
        self._simulation_shape = example_mrc.shape
        axis_labels = example_mr.axis_labels
        self._in_axis_labels = axis_labels
        self._axis_map = {key: value for value, key in enumerate(axis_labels)}
        self._grid_offset = (
            np.array(example_mr.grid_global_offset) * example_mr.grid_unit_SI * ureg.meter
        )
        self._field_position = example_mrc.position
        self._read_dtype = example_mrc.dtype

        self.sim_write_interval = sim_write_interval
        self.t_0 = t_0

        self.density_fields = density_fields

        # internal units:
        self._unit_time_int = self.sim_write_interval * self.simulation_step_duration

        self._total_offset = [0, 0, 0]
        self._total_extent = list(self._simulation_shape)

        self._chunk_offset = None
        self._chunk_extent = None

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

        iterations = self._series.iterations
        self._used_iterations = np.sort(np.unique(np.concatenate((self._it_min, self._it_max))))
        is_in = np.isin(self._used_iterations, iterations)

        assert np.all(is_in), f"missing iterations: {np.unique(self._used_iterations[~is_in])}"

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

    def write_to_openpmd(self, out_series_path, options):
        if HAVE_MPI:
            out_series = io.Series(
                str(out_series_path), io.Access_Type.create, self._comm, options=options
            )
        else:
            out_series = io.Series(str(out_series_path), io.Access_Type.create, options=options)
        out_series.set_software("plasma_hed_xray")
        it: io.Iteration = out_series.iterations[0]
        it.open()
        mesh: io.Mesh = it.meshes["integrated_density"]
        mrc: io.Mesh_Record_Component = mesh[io.Mesh_Record_Component.SCALAR]

        t_0 = self.t_0.to("fs")
        t_start = (self.simulation_step_duration * self._used_iterations[0]).to("fs") - t_0
        it.set_time(t_start.magnitude)
        it.set_dt(0.0)
        it.set_time_unit_SI(1e-15)
        it.set_attribute("t_0", t_0.magnitude)

        global_extent = deepcopy(self._total_extent)
        global_extent.pop(self.prop_axis)
        global_extent.insert(0, self._detection_duration_int_time)
        global_offset = self._total_offset
        global_offset.pop(self.prop_axis)
        global_offset.insert(0, 0)

        local_offset = deepcopy(self._chunk_offset)
        local_offset.pop(self.prop_axis)
        local_offset.insert(0, 0)
        local_extent = self._chunk_extent
        local_extent.pop(self.prop_axis)
        local_extent.insert(0, self._detection_duration_int_time)

        unit_length = self._unit_length_int.to("meter").magnitude

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
        out_series.close()

    def __call__(self, disable_progress=None, tqdm_kwargs=None):
        if self.prop_axis == 0:
            _process_loaded_iteration_data = _process_loaded_iteration_data_0
        if self.prop_axis == 1:
            _process_loaded_iteration_data = _process_loaded_iteration_data_1
        if self.prop_axis == 2:
            _process_loaded_iteration_data = _process_loaded_iteration_data_2
        else:
            raise NotImplementedError("This code only works with 3Dim data.")
        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        shape = deepcopy(self._chunk_extent)
        shape.pop(self.prop_axis)
        shape.insert(0, self._detection_duration_int_time)
        shape = tuple(shape)
        self._volume = np.zeros(shape, dtype=np.float64)

        for iteration_idx in tqdm(
            self._used_iterations,
            position=self._comm.rank,
            desc=f"MPI rank {self._comm.rank}: ",
            disable=disable_progress,
            **tqdm_kwargs,
        ):
            iteration = self._series.iterations[iteration_idx]

            # Find slices needed from this iteration
            where_min = np.where(self._it_min == iteration_idx)
            where_max = np.where(self._it_max == iteration_idx)
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
                iteration.meshes[field][io.Mesh_Record_Component.SCALAR]
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
            read_buffer = np.empty((len(self.density_fields), *extent), dtype=self._read_dtype)

            for ii, mrc in enumerate(mrc_list):
                mrc.load_chunk(read_buffer[ii], offset=offset, extent=extent)

            self._series.flush()
            iteration.close()
            read_buffer = read_buffer.astype(np.float64, copy=False)

            _process_loaded_iteration_data(
                read_buffer,
                self._volume,
                beam_index_arr,
                prop_axis_pos_arr,
                interpolation_coeff_arr,
                prop_min,
            )


def to_intensity(
    volume: ArrayLike,
    photons_in_pulse: int,
    pulse_duration: ureg.Quantity,
    pulse_crosssection: ureg.Quantity,
    volume_metadata: Optional[FieldMetaData] = None,
    pulse_shape: Optional[ArrayLike] = None,
    pulse_profile: Optional[ArrayLike] = None,
):
    volume = np.array(volume, copy=True)

    if pulse_profile is not None:
        pulse_profile = np.array(volume, copy=True)
        volume = volume * np.sqrt(pulse_profile[None, ...])
    intensity = np.abs(np.fft.fft2(volume, norm="backward")) ** 2
    intensity = np.fft.fftshift(intensity, axes=(-1, -2))
    if pulse_shape is not None:
        intensity *= pulse_shape[:, None, None]
    intensity *= photons_in_pulse
    intensity /= pulse_crosssection.to(ureg.meter ** (2)).magnitude
    intensity /= pulse_duration.to(ureg.second).magnitude
    intensity *= cs.physical_constants["classical electron radius"][0] ** 2

    if volume_metadata is None:
        return intensity
    else:
        ndim = volume_metadata.ndim
        axis_labels = list(volume_metadata.axis_labels)
        for i, label in enumerate(axis_labels[1:]):
            axis_labels[i + 1] = "q_" + label
        first_cell_positions = list(volume_metadata.first_cell_positions)
        q1 = np.fft.fftshift(
            np.fft.fftfreq(volume.shape[1], d=volume_metadata.cell_size[1].magnitude / (2 * np.pi))
        )
        q2 = np.fft.fftshift(
            np.fft.fftfreq(volume.shape[2], d=volume_metadata.cell_size[2].magnitude / (2 * np.pi))
        )
        first_cell_positions[1] = q1[0] / volume_metadata.cell_size[1].units
        first_cell_positions[2] = q2[0] / volume_metadata.cell_size[2].units
        shape = intensity.shape
        cell_size = list(volume_metadata.cell_size)
        cell_size[1] = (q1[1] - q1[0]) / volume_metadata.cell_size[1].units
        cell_size[2] = (q2[1] - q2[0]) / volume_metadata.cell_size[2].units
        in_cell_position = volume_metadata.in_cell_position
        value_unit = (1 * volume_metadata.value_unit).to_base_units().units
        value_unit *= ureg.meter * ureg.meter / ureg.second
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
        return intensity, metadata