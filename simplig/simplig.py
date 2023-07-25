import numpy as np
import openpmd_api as io
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

from . import ureg

from dataclasses import dataclass, asdict
from typing import Sequence
from typing import Mapping, Optional, AnyStr
from textwrap import wrap


@dataclass(frozen=True)
class FieldMetaData:
    # dimensionality:
    ndim: int
    # dataclass fields:
    # axes
    axis_labels: Sequence[AnyStr]
    first_cell_positions: ureg.Quantity
    shape: Sequence[int]
    cell_size: ureg.Quantity
    in_cell_position: Sequence[float]
    # field value
    value_unit: ureg.Unit
    time: ureg.Quantity
    field_description: AnyStr
    value_symbol: AnyStr = ""
    # extra info about slicing and averaging
    slicing_positions: Optional[Mapping[AnyStr, ureg.Quantity]] = None
    averaging_region: Optional[Mapping[AnyStr, ureg.Quantity]] = None

    def __post_init__(self):
        dim_condition = (
            len(self.axis_labels)
            == len(self.first_cell_positions)
            == len(self.shape)
            == len(self.cell_size)
            == len(self.in_cell_position)
            == self.ndim
        )
        if not dim_condition:
            raise ValueError(
                f"axis_labels: len is {len(self.axis_labels)}"
                f", first_cell_positions: len is {len(self.first_cell_positions)}"
                f", shape: len is {len(self.shape)}"
                f", cell_size: len is {len(self.cell_size)}"
                f", in_cell_position: len is {len(self.in_cell_position)}"
                f" have to have length equal ndim = {self.ndim}"
            )

    def get_modified(self, **kwargs):
        dict_repr = asdict(self)
        dict_repr.update(kwargs)
        return FieldMetaData(**dict_repr)

    @property
    def last_cell_positions(self):
        return np.add(self.first_cell_positions, np.multiply(self.shape, self.cell_size))

    @property
    def img_meta(self):
        return {"simplig_" + k: f"{str(v)}" for k, v in asdict(self).items()}

    @property
    def plot_title(self):
        slicing_description = ""
        if self.slicing_positions is not None:
            for axis, pos in self.slicing_positions.items():
                if pos is not None:
                    if slicing_description:
                        slicing_description += " and"
                    slicing_description += f" at {pos.to_compact():.6g~P} along {axis}"
        averaging_description = ""
        if self.averaging_region is not None:
            for axis, reg in self.averaging_region.items():
                if reg is not None:
                    if averaging_description:
                        averaging_description += " and"
                    averaging_description += (
                        f" from {reg[0].to_compact():.6g~P} "
                        f"to {reg[1].to_compact():.6g~P} along {axis}"
                    )
        title = f"{self.field_description} at {self.time.to_compact() :.6g~P}"
        if slicing_description:
            title += f" sliced{slicing_description}"
        if averaging_description:
            title += f" averaged{averaging_description}"
        return title

    def get_imshow_extent(self, unit=None):
        assert self.ndim == 2, "imshow extent is defined only for 2D datasets"
        extent = [
            self.first_cell_positions[1] + (self.in_cell_position[1] - 0.5) * self.cell_size[1],
            self.last_cell_positions[1] + (0.5 + self.in_cell_position[1]) * self.cell_size[1],
            self.first_cell_positions[0] + (self.in_cell_position[0] - 0.5) * self.cell_size[0],
            self.last_cell_positions[0] + (0.5 + self.in_cell_position[0]) * self.cell_size[0],
        ]
        extent = ureg.Quantity.from_list(extent)
        if unit is not None:
            extent = extent.to(unit)
        else:
            extent = extent.to(np.max(np.abs(extent)).to_compact().units)
        return extent.magnitude, extent.units

    def get_positions(self, axis, unit=None):
        # assert axis>= 0 and axis < self.ndim
        positions = (
            np.arange(self.shape[axis]) * self.cell_size[axis] + self.first_cell_positions[axis]
        )
        if unit is not None:
            positions = positions.to(unit)
        else:
            positions = positions.to(np.max(positions).to_compact().units)
        return positions


def plot_2d_field(data, meta_data, ax=None, log_scale=False, unit=None, **imshow_kwargs):
    assert meta_data.ndim == 2
    if ax is None:
        f, ax = plt.subplots(1)
    else:
        f = ax.get_figure()
    norm = None
    if log_scale:
        if np.any(data < 0):
            norm = SymLogNorm(1.0)
        else:
            norm = LogNorm()
    kwargs = {"norm": norm, "interpolation": "none"}
    kwargs.update(imshow_kwargs)
    data *= meta_data.value_unit
    if unit is not None:
        data = data.to(unit)
    extent, extent_unit = meta_data.get_imshow_extent()
    img = ax.imshow(data.magnitude, extent=extent, origin="lower", **kwargs)

    ax.set_xlabel(meta_data.axis_labels[1] + f" [{extent_unit:~P}]")
    ax.set_ylabel(meta_data.axis_labels[0] + f" [{extent_unit:~P}]")
    ax.set_title(meta_data.plot_title)
    cax = ax.inset_axes([1.01, 0.0, 0.05, 1])
    f.colorbar(img, ax=ax, cax=cax, label=rf"{meta_data.value_symbol}" + rf"$\left[{data.units:~L}\right]$")
    plt.tight_layout()

def wrap_text(text, length):
    return "\n".join(wrap(text, length))


def plot_1d_field(
    data, meta_data, ax=None, log_scale=False, unit=None, title_fontsize=12, scatter=False, **plot_kwargs
):
    assert meta_data.ndim == 1
    if ax is None:
        _, ax = plt.subplots(1)
    if log_scale:
        ax.set_yscale("log")
    x = meta_data.get_positions(0)
    if unit is not None:
        data = ((data * meta_data.value_unit).to(unit)).magnitude
    else:
        unit = meta_data.value_unit
    if scatter:
        ax.scatter(x.magnitude, data, **plot_kwargs)
    else:
        ax.plot(x.magnitude, data, **plot_kwargs)
    ax.set_xlabel(meta_data.axis_labels[0] + f" [{x.units:~P}]")
    ax.set_ylabel(rf"{meta_data.value_symbol}" + f"[{unit:~P}]")
    title_len = int(round(ax.bbox.width / 500 * 12 / title_fontsize * 60))
    ax.set_title(wrap_text(meta_data.plot_title, title_len), fontsize=title_fontsize)
    plt.tight_layout()


def plot_field(data, meta_data, ax=None, log_scale=False, unit=None, **plot_func_kwargs):
    if meta_data.ndim == 0:
        if unit is not None:
            quantity = (data * meta_data.value_unit).to(unit)
        else:
            quantity = data * meta_data.value_unit
        print(meta_data.plot_title + f" is {quantity:.6g~P}")

    elif meta_data.ndim == 1:
        plot_1d_field(data, meta_data, ax, log_scale, unit, **plot_func_kwargs)
    elif meta_data.ndim == 2:
        plot_2d_field(data, meta_data, ax, log_scale, unit, **plot_func_kwargs)
    else:
        raise Exception("Wrong dimensionality, meta_data.ndim must be 0, 1, or 2!")


def get_first_cell_position(mr, axis, slicing=None):
    global_offset = mr.get_attribute("gridGlobalOffset")
    if slicing is not None:
        if type(slicing[axis]) is slice:
            local_start = slicing[axis].start
            if local_start is None:
                local_start = 0
        else:
            local_start = slicing[axis]
    else:
        local_start = 0
    start = global_offset[axis]
    start += local_start * mr.grid_spacing[axis]
    start *= mr.grid_unit_SI
    return start


def unit_dimension_to_pint(unit_dimension):
    # unit dimension description from the openPMD standard:
    # powers of the 7 base measures characterizing the record's unit in SI
    # (length L, mass M, time T, electric current I, thermodynamic temperature theta,
    # amount of substance N, luminous intensity J)
    base_units = (
        ureg.metre,
        ureg.kilogram,
        ureg.second,
        ureg.ampere,
        ureg.kelvin,
        ureg.mole,
        ureg.candela,
    )
    unit = ureg.Quantity(1)
    for dim, base_unit in zip(unit_dimension, base_units):
        if dim != 0:
            unit *= base_unit**dim
    return unit.units


def slice_index_to_position(index, mr, mrc, axis):
    return (
        mr.grid_spacing[axis] * (index + mrc.position[axis]) + mr.grid_global_offset[axis]
    ) * mr.grid_unit_SI


def _get_density_name(species):
    return species + "_density"


def _get_energy_density_name(species):
    return species + "_energyDensity"


class OpenPMDDataLoader:
    def __init__(self, series_path, *args, **kwargs):
        self.series = io.Series(str(series_path), io.Access.read_only, *args, **kwargs)

    def get_field(
        self,
        iteration,
        field,
        component=io.Mesh_Record_Component.SCALAR,
        slicing=None,
        axes_to_average=None,
        ret_meta=True,
        unit=None
    ):
        # verify correct and supported slicing
        if slicing is not None:
            for sc in slicing:
                if type(sc) is int:
                    assert sc >= 0, "No support for negative indexing yet"
                if type(sc) is slice:
                    if sc.start is not None:
                        assert sc.start >= 0, "No support for negative indexing in slice start"
                    if sc.step is not None:
                        assert sc.step == 1, "No support for striding"

        it = self.series.iterations[iteration]
        it.open()
        mr = it.meshes[field]
        mrc = mr[component]
        if slicing is None:
            slicing = tuple([slice(None)] * mrc.ndim)
        data = mrc[slicing]
        self.series.flush()
        shape_before_average = data.shape
        if axes_to_average is not None:
            data = np.squeeze(np.average(data, axis=axes_to_average))

        data *= mrc.unit_SI
        unit_dataset = unit_dimension_to_pint(mr.unit_dimension)
        if unit is not None:
            data = data * unit_dataset
            data = data.to(unit)
            unit_dataset = data.units
            data = data.magnitude
        if ret_meta:
            slice_axes = []
            slicing_positions = dict.fromkeys(mr.axis_labels, None)
            for i, sc in enumerate(slicing):
                if type(sc) is int:
                    slice_axes.append(i)
                    slicing_positions[mr.axis_labels[i]] = (
                        slice_index_to_position(sc, mr, mrc, i) * ureg.meter
                    )

            axis_labels = mr.axis_labels
            first_cell_positions = [
                get_first_cell_position(mr, axis, slicing) for axis in range(mrc.ndim)
            ]
            first_cell_positions = np.array(first_cell_positions) * ureg.meter
            cell_size = np.array(mr.grid_spacing) * mr.grid_unit_SI * ureg.meter

            axis_labels = np.delete(axis_labels, slice_axes)
            first_cell_positions = np.delete(first_cell_positions, slice_axes)
            cell_size = np.delete(cell_size, slice_axes)
            in_cell_position = np.delete(mrc.position, slice_axes)

            averaging_region = None
            if axes_to_average is not None:
                averaging_region = {}
                if type(axes_to_average) is int:
                    axes_to_average = [
                        axes_to_average,
                    ]
                for axis in axes_to_average:
                    start = first_cell_positions[axis]
                    end = start + shape_before_average[axis] * cell_size[axis]
                    label = axis_labels[axis]
                    averaging_region[label] = ureg.Quantity.from_list([start, end])

                axis_labels = np.delete(axis_labels, axes_to_average)
                first_cell_positions = np.delete(first_cell_positions, axes_to_average)
                cell_size = np.delete(cell_size, axes_to_average)
                in_cell_position = np.delete(in_cell_position, axes_to_average)

            component_str = component
            if component == io.Mesh_Record_Component.SCALAR:
                component_str = ""
            time = (iteration * it.dt + mr.time_offset) * it.time_unit_SI

            meta_data = FieldMetaData(
                ndim=data.ndim,
                axis_labels=axis_labels,
                first_cell_positions=first_cell_positions,
                shape=data.shape,
                cell_size=cell_size,
                in_cell_position=in_cell_position,
                value_unit=unit_dataset,
                field_description=f"{field}{component_str}",
                time=time * ureg.second,
                slicing_positions=slicing_positions,
                averaging_region=averaging_region,
            )
            return data, meta_data
        else:
            return data

    def get_temp(
        self,
        iteration,
        species,
        slicing=None,
        axes_to_average=None,
        ret_meta=True,
        unit=ureg.eV,
        get_density_name=_get_density_name,
        get_energy_density_name=_get_energy_density_name,
    ):
        """

        :param series:
        :param iteration:
        :param species:
        :param slicing:
        :param axes_to_average:
        :param ret_meta:
        :param unit:
        :param get_density_name:
        :param get_energy_density_name:
        :return:
        """
        density_field = get_density_name(species)
        energy_density_field = get_energy_density_name(species)
        density, density_meta = self.get_field(
            iteration,
            density_field,
            slicing=slicing,
            axes_to_average=axes_to_average,
            ret_meta=True,
        )
        energy_density, energy_density_meta = self.get_field(
            iteration,
            energy_density_field,
            slicing=slicing,
            axes_to_average=axes_to_average,
            ret_meta=True,
        )
        if density_meta.ndim == 0:
            temp = 0
            if density > 0:
                temp = (2 / 3) * energy_density / density
        else:
            temp = np.zeros_like(density)
            mask = density > 0
            temp[mask] = (2 / 3) * energy_density[mask] / density[mask]
        temp *= energy_density_meta.value_unit / density_meta.value_unit
        temp = temp.to(unit)
        if ret_meta:
            field_description = f"mean kinetic energy of {species}"
            value_symbol = r"$2/3\left<E_\mathrm{kin}\right>$"
            value_unit = unit
            temp_meta = energy_density_meta.get_modified(
                field_description=field_description,
                value_symbol=value_symbol,
                value_unit=value_unit,
            )
            return temp.magnitude, temp_meta
        return temp.magnitude
