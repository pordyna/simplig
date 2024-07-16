from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from ..data import DescribedField
from affine_transform import transform
from mgen import rotation_around_x, rotation_around_y, rotation_around_z, rotation_from_angle
from .. import ureg


def get_extents(corners):
    arr = [[np.min(corners[i, :]), np.max(corners[i, :])] for i in range(corners.shape[0])]
    return np.array(arr)


def rotate(field: Union[ArrayLike, DescribedField], rotation_angle, rotation_axis_idx=None):
    if type(field) is DescribedField:
        # Note: rotation like that works with square or cubic cells.
        # For other cells one may need to use sth like scipy.ndimage.map_coordinates.
        assert np.allclose(
            field.meta.cell_size, field.meta.cell_size
        ), "Different cell side lengths are not supported"
        original = field.data
        shape = field.meta.shape
        ndim = field.meta.ndim
        do_rot = original is not None
        do_meta = True
    else:
        original = np.asarray(field)
        shape = original.shape
        ndim = original.ndim
        do_rot = True
        do_meta = False

    if ndim == 3:
        if rotation_axis_idx == 0:
            matrix_factory = rotation_around_x
        elif rotation_axis_idx == 1:
            matrix_factory = rotation_around_y
        elif rotation_axis_idx == 2:
            matrix_factory = rotation_around_z
        else:
            raise IndexError(
                f"Input parameter rotation_axis_idx={rotation_axis_idx} out of range for a 3D array"
            )
        corners_original = np.empty((3, 8), dtype=np.float64)
        corners_original[:, :4] = [
            [0, 0, shape[0], shape[0]],
            [0, shape[1], 0, shape[1]],
            [0, 0, 0, 0],
        ]
        corners_original[:, 4:] = corners_original[:, :4]
        corners_original[2, 4:] = shape[2]

    elif ndim == 2:
        matrix_factory = rotation_from_angle
        corners_original = np.asarray(
            [[0, 0, shape[0], shape[0]], [0, shape[1], 0, shape[1]]], dtype=np.float64
        )
    else:
        raise NotImplementedError("This code only works with 2D or 3D data.")

    rot_matrix = matrix_factory(rotation_angle)
    origin = np.array([0, 0, 0], dtype=float)

    new_corners = rot_matrix.dot(corners_original - origin[:, None]) + origin[:, None]
    new_extents = get_extents(new_corners)
    new_shape = tuple(np.around(np.squeeze(np.diff(new_extents, axis=1))).astype(int))
    offset = -new_extents[:, 0]
    new_meta = None
    if do_meta:
        old_axis_labels = field.meta.axis_labels
        new_axis_labels = [label + "'" for label in old_axis_labels]
        if ndim == 3:
            new_axis_labels[rotation_axis_idx] = old_axis_labels[rotation_axis_idx]
            rot_axis_label = old_axis_labels[rotation_axis_idx]
        else:
            rot_axis_label = f"{old_axis_labels[0]} x {old_axis_labels[1]}"
        applied_rotation = (rot_axis_label, rotation_angle * ureg.radian)
        applied_rotations = field.meta.applied_rotations
        if applied_rotations is None:
            applied_rotations = [
                applied_rotation,
            ]
        else:
            applied_rotations.append(applied_rotation)
        cell_size = field.meta.cell_size[0]
        old_first_cell_positions = field.meta.first_cell_positions
        origin_with_units = (origin * cell_size).to(old_first_cell_positions.units)
        old_first_cell_positions = (
            rot_matrix.dot(old_first_cell_positions.magnitude - origin_with_units.magnitude)
            * old_first_cell_positions.units
        )
        offset_with_unit = offset * cell_size
        corner_with_unit = new_corners[:, 0] * cell_size
        new_first_cell_positions = [
            -offset_with_unit[dd].to(pos.units) - corner_with_unit[dd].to(pos.units) + pos
            for dd, pos in enumerate(old_first_cell_positions)
        ]
        new_first_cell_positions = ureg.Quantity.from_list(new_first_cell_positions)
        new_meta = field.meta.get_modified(
            shape=new_shape,
            applied_rotations=applied_rotations,
            axis_labels=new_axis_labels,
            first_cell_positions=new_first_cell_positions,
        )

    if do_rot:
        transformed = np.empty(new_shape, dtype=original.dtype)
        transform(
            original, rot_matrix, offset, origin=origin, output_image=transformed, order="linear"
        )
        if new_meta is not None:
            return DescribedField(transformed, new_meta)
        else:
            return transformed
    else:
        return DescribedField(None, new_meta)
