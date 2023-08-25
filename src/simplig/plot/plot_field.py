import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

# from .. import ureg

from textwrap import wrap


def _plot_2d_field(
    data, meta_data, ax=None, log_scale=False, unit=None, title_fontsize=12, **imshow_kwargs
):
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
    extent, (extent_unit_y, extent_unit_x) = meta_data.get_imshow_extent()
    img = ax.imshow(data.magnitude, extent=extent, origin="lower", **kwargs)

    ax.set_xlabel(meta_data.axis_labels[1] + f" [{extent_unit_x:~P}]")
    ax.set_ylabel(meta_data.axis_labels[0] + f" [{extent_unit_y:~P}]")
    title_len = int(round(ax.bbox.width / 500 * 12 / title_fontsize * 60))
    ax.set_title(wrap_text(meta_data.plot_title, title_len), fontsize=title_fontsize)
    cax = ax.inset_axes([1.01, 0.0, 0.05, 1])
    f.colorbar(
        img, ax=ax, cax=cax, label=rf"{meta_data.value_symbol}" + rf"$\left[{data.units:~L}\right]$"
    )
    plt.tight_layout()


def wrap_text(text, length):
    return "\n".join(wrap(text, length))


def _plot_1d_field(
    data,
    meta_data,
    ax=None,
    log_scale=False,
    unit=None,
    title_fontsize=12,
    scatter=False,
    **plot_kwargs,
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


def plot_field(field, ax=None, log_scale=False, unit=None, **plot_func_kwargs):
    data = field.data
    meta_data = field.meta
    if meta_data.ndim == 0:
        data = field.data
        meta_data = field.meta
        if unit is not None:
            quantity = (data * meta_data.value_unit).to(unit)
        else:
            quantity = data * meta_data.value_unit
        print(meta_data.plot_title + f" is {quantity:.6g~P}")

    elif meta_data.ndim == 1:
        _plot_1d_field(data, meta_data, ax, log_scale, unit, **plot_func_kwargs)
    elif meta_data.ndim == 2:
        _plot_2d_field(data, meta_data, ax, log_scale, unit, **plot_func_kwargs)
    else:
        raise Exception("Wrong dimensionality, meta_data.ndim must be 0, 1, or 2!")
