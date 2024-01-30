from dataclasses import dataclass
import numpy as np
from .FieldMetaData import FieldMetaData
from typing import Union
from copy import copy, deepcopy
from .. import ureg


@dataclass
class DescribedField:
    data: Union[np.ndarray, None]
    meta: FieldMetaData

    def __iter__(self):
        for member in [self.data, self.meta]:
            yield member

    @property
    def T(self):
        if self.meta.ndim > 1:
            new_values = dict(
                axis_labels=self.meta.axis_labels[::-1],
                first_cell_positions=self.meta.first_cell_positions[::-1],
                shape=self.meta.shape[::-1],
                cell_size=self.meta.cell_size[::-1],
                in_cell_position=self.meta.in_cell_position[::-1],
            )

            new_meta = self.meta.get_modified(**new_values)
            return DescribedField(self.data.T, new_meta)
        else:
            return self

    def __post_init__(self):
        if self.data is not None:
            assert self.data.ndim == self.meta.ndim
            assert self.data.shape == self.meta.shape

    def _slice_index_to_position(self, index, axis):
        dx = self.meta.cell_size[axis].to_base_units()
        in_cell_position = self.meta.in_cell_position[axis]
        offset = self.meta.first_cell_positions[axis].to_base_units()
        return offset + dx * (index + in_cell_position)

    def _get_first_cell_position(self, axis, slicing=None):
        if slicing is not None:
            if type(slicing[axis]) is slice:
                local_start = slicing[axis].start
                if local_start is None:
                    local_start = 0
            else:
                local_start = slicing[axis]
        else:
            local_start = 0
        start = self.meta.first_cell_positions[axis].to_base_units()
        start += local_start * self.meta.cell_size[axis]
        return start

    def __getitem__(self, slicing):
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
        # handle proper slices (integer indexing)>.
        slice_axes = []
        slicing_positions = dict.fromkeys(self.meta.axis_labels, None)
        for i, sc in enumerate(slicing):
            if type(sc) is int:
                slice_axes.append(i)
                slicing_positions[self.meta.axis_labels[i]] = self._slice_index_to_position(sc, i)
        axis_labels = self.meta.axis_labels
        if self.meta.slicing_positions is not None:
            slicing_positions = dict(self.meta.slicing_positions) | slicing_positions
        first_cell_positions = [
            self._get_first_cell_position(axis, slicing) for axis in range(self.meta.ndim)
        ]
        #first_cell_positions = ureg.Quantity.from_list(first_cell_positions)
        def _delete(sequence, idxs):
            try:
                sequence = np.delete(sequence, idxs)
            except ValueError:
                sequence = list(sequence)
                for idx in sorted(idxs, reverse=True):
                    del sequence[idx]
            return sequence

        cell_size = self.meta.cell_size
        in_cell_position = self.meta.in_cell_position
        axis_labels = np.delete(axis_labels, slice_axes)
        first_cell_positions = _delete(first_cell_positions, slice_axes)
        cell_size = _delete(cell_size, slice_axes)
        in_cell_position = np.delete(in_cell_position, slice_axes)
        new_data = self.data[slicing]
        new_meta = self.meta.get_modified(
            cell_size=cell_size,
            in_cell_position=in_cell_position,
            axis_labels=axis_labels,
            first_cell_positions=first_cell_positions,
            ndim=new_data.ndim,
            shape=new_data.shape,
            slicing_positions=slicing_positions,
        )
        return DescribedField(new_data, new_meta)
