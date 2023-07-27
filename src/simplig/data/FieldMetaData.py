import numpy as np
from dataclasses import dataclass, asdict
from typing import Sequence
from typing import Mapping, Optional, AnyStr

from .. import ureg


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
