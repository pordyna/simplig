import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

# from .. import ureg

from textwrap import wrap


def _plot_2d_field(
    data,
    meta_data,
    ax=None,
    log_scale=False,
    unit=None,
    title_fontsize=12,
    colorbar=True,
    tight_layout=True,
    vmin=None,
    vmax=None,
    min_max_sig_figs=4,
    linthresh=1,
    disable_title=False,
    **imshow_kwargs,
):
    data *= meta_data.value_unit
    if unit is not None:
        data = data.to(unit)
    assert meta_data.ndim == 2

    if (
        log_scale
        and (vmin is None and vmax is None and np.min(data.data) == np.max(data.data))
        or np.max(data.data == vmin)
        or np.min(data.data == vmax)
    ):
        log_scale = False

    if ax is None:
        f, ax = plt.subplots(1)
    else:
        f = ax.get_figure()
    norm = None
    cmap = None
    if log_scale:
        if np.any(data.magnitude < 0):
            if vmin is None and vmax is None:
                vmin = np.min(data.magnitude)
                vmax = np.max(data.magnitude)
                vmax = max(np.abs(vmin), np.abs(vmax))
                vmin = -vmax
            norm = SymLogNorm(linthresh, vmin=vmin, vmax=vmax)
        else:
            norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        if (
            vmin is None
            and vmax is None
            and np.any(data.magnitude < 0)
            and np.any(data.magnitude > 0)
        ):
            vmin = np.min(data.magnitude)
            vmax = np.max(data.magnitude)
            vmax = max(np.abs(vmin), np.abs(vmax))
            vmin = -vmax
    if vmin is not None and vmax is not None and np.sign(vmin) * np.sign(vmax) == -1:
        cmap = "RdBu_r"
    if not log_scale:
        imshow_kwargs = imshow_kwargs | dict(vmax=vmax, vmin=vmin)
    kwargs = {"norm": norm, "interpolation": "none", "cmap": cmap}
    kwargs.update(imshow_kwargs)

    extent, (extent_unit_y, extent_unit_x) = meta_data.get_imshow_extent()

    img = ax.imshow(data.magnitude, extent=extent, origin="lower", **kwargs)
    cax = None
    if colorbar:
        cax = ax.inset_axes([1.01, 0.0, 0.05, 1])
        f.colorbar(
            img,
            ax=ax,
            cax=cax,
            label=rf"{meta_data.value_symbol}"
            + rf"$\left[{data.units:~L}\right]$"
            + f"    min={np.min(data).magnitude:.{min_max_sig_figs}g}, max={np.max(data).magnitude:.{min_max_sig_figs}g}",
        )
    ax.set_xlabel(meta_data.axis_labels[1] + f" [{extent_unit_x:~P}]")
    ax.set_ylabel(meta_data.axis_labels[0] + f" [{extent_unit_y:~P}]")
    title_len = int(round(ax.bbox.width / 500 * 12 / title_fontsize * 60))
    # ax.set_title(wrap_text(meta_data.plot_title, title_len), fontsize=title_fontsize)
    if not disable_title:
        ax.set_title(meta_data.plot_title, fontsize=title_fontsize, wrap=True)

    if tight_layout:
        plt.tight_layout()

    return dict(ax=ax, img=img, cax=cax)


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
    tight_layout=True,
    disable_title=False,
    **plot_kwargs,
):
    assert meta_data.ndim == 1
    if ax is None:
        _, ax = plt.subplots(1)
    if log_scale:
        if np.any(data < 0):
            ax.set_yscale("symlog")
        else:
            ax.set_yscale("log")

    x = meta_data.get_positions(0)
    if unit is not None:
        data = ((data * meta_data.value_unit).to(unit)).magnitude
    else:
        unit = meta_data.value_unit
    if scatter:
        plot = ax.scatter(x.magnitude, data, **plot_kwargs)
    else:
        plot = ax.plot(x.magnitude, data, **plot_kwargs)
    ax.set_xlabel(meta_data.axis_labels[0] + f" [{x.units:~P}]")
    ax.set_ylabel(rf"{meta_data.value_symbol}" + f"[{unit:~P}]")
    # title_len = int(round(ax.bbox.width / 500 * 12 / title_fontsize * 60))
    if not disable_title:
        ax.set_title(meta_data.plot_title, fontsize=title_fontsize, wrap=True)
    if tight_layout:
        plt.tight_layout()

    return dict(ax=ax, plot=plot)


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
        return _plot_1d_field(data, meta_data, ax, log_scale, unit, **plot_func_kwargs)
    elif meta_data.ndim == 2:
        return _plot_2d_field(data, meta_data, ax, log_scale, unit, **plot_func_kwargs)
    else:
        raise Exception("Wrong dimensionality, meta_data.ndim must be 0, 1, or 2!")
