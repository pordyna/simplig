import numpy as np
import openpmd_api as io
from .. import ureg
from ..data import FieldMetaData
from .DescribedField import DescribedField


def _get_first_cell_position(mr, axis, slicing=None):
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


def _unit_dimension_to_pint(unit_dimension):
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


def _slice_index_to_position(index, mr, mrc, axis):
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
        unit=None,
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
        unit_dataset = _unit_dimension_to_pint(mr.unit_dimension)
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
                        _slice_index_to_position(sc, mr, mrc, i) * ureg.meter
                    )

            axis_labels = mr.axis_labels
            first_cell_positions = [
                _get_first_cell_position(mr, axis, slicing) for axis in range(mrc.ndim)
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
            return DescribedField(data, meta_data)
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
            return DescribedField(temp.magnitude, temp_meta)
        return temp.magnitude
