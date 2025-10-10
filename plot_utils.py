from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_panels_with_ci(
    summary: pd.DataFrame,
    *,
    series_order: Sequence[str],
    label_map: Dict[str, str],
    style_for: Callable[[str], Tuple[str, str]],
    savepath: str,
    facet_col: str = "T",
    x_col: str = "H",
    series_col: str = "algo",
    y_mean: str = "mean",
    y_lo: str = "lo",
    y_hi: str = "hi",
    inset: bool = False,
    y_label: str = "Clustering error rate",
    inset_xlim: Tuple[int, int] | None = None,
    inset_xticks: Sequence[int] | None = None,
    legend_order: Sequence[str] | None = None,
    legend_ncol: int | None = None,
    extra_xticks: Sequence[int] | None = None,
    legend_bbox: Tuple[float, float] | None = None,
    top_adjust: float | None = None,
    title_fontsize: int | None = None,
    label_fontsize: int | None = None,
    tick_fontsize: int | None = None,
    legend_fontsize: int | None = None,
    x_tick_formatter: Callable[[int], str] | None = None,
    facet_title_map: Dict[str, str] | None = None,
    x_tick_formatter_per_facet: Dict[str, Callable[[int], str]] | None = None,
) -> None:
    """
    Render a grid of panels faceted by `facet_col`, x-axis=`x_col`, plotting
    mean lines with confidence bands per series in `series_order`.

    The visual style (colors/linestyles) and legend labels are provided via
    `style_for` and `label_map` to keep experiment-specific styling local to
    the caller modules.
    """

    unique_facets = sorted(summary[facet_col].unique())
    ncols = min(3, len(unique_facets))
    nrows = int(np.ceil(len(unique_facets) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6.6 * ncols, 4.6 * nrows),
        constrained_layout=True,
    )
    axes = np.array(axes, ndmin=1).ravel()

    # Font sizes (overridable)
    title_fs = title_fontsize if title_fontsize is not None else 16
    label_fs = label_fontsize if label_fontsize is not None else 14
    tick_fs = tick_fontsize if tick_fontsize is not None else 12
    legend_fs = legend_fontsize if legend_fontsize is not None else 14

    # Shared y-lims across panels based on CI bounds
    y_min = summary[y_lo].min()
    y_max = summary[y_hi].max()
    pad = 0.02 * (y_max - y_min + 1e-12)
    ylims = (y_min - pad, y_max + pad)

    handles: Dict[str, plt.Line2D] = {}

    for i, facet in enumerate(unique_facets):
        ax = axes[i]
        sub = summary[summary[facet_col] == facet].sort_values(x_col)
        xs = sorted(sub[x_col].unique())

        for series in series_order:
            d = sub[sub[series_col] == series].sort_values(x_col)
            color, ls = style_for(series)
            (line,) = ax.plot(
                d[x_col],
                d[y_mean],
                ls=ls,
                marker="o",
                lw=1.8,
                ms=4,
                color=color,
                label=label_map.get(series, series),
            )
            ax.fill_between(d[x_col], d[y_lo], d[y_hi], alpha=0.18, linewidth=0, color=color)

            handles[label_map.get(series, series)] = line

        # Use custom title from facet_title_map if provided, otherwise use default format
        title = facet_title_map.get(facet, f"{facet_col}={facet}") if facet_title_map else f"{facet_col}={facet}"
        ax.set_title(title, fontsize=title_fs)
        ax.set_xlabel(x_col, fontsize=label_fs)
        # Only leftmost column shows the y-label
        if i % ncols == 0:
            ax.set_ylabel(y_label, fontsize=label_fs)
        else:
            ax.set_ylabel("")
        ax.set_ylim(*ylims)
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
        if len(xs) > 0:
            ticks = xs[::2]
            if extra_xticks is not None:
                keep = [x for x in extra_xticks if x in xs]
                ticks = sorted(set(ticks).union(keep))
            ax.set_xticks(ticks)
            # Use per-facet formatter if provided, otherwise use global formatter
            formatter_to_use = None
            if x_tick_formatter_per_facet is not None and facet in x_tick_formatter_per_facet:
                formatter_to_use = x_tick_formatter_per_facet[facet]
            elif x_tick_formatter is not None:
                formatter_to_use = x_tick_formatter
            if formatter_to_use is not None:
                ax.set_xticklabels([formatter_to_use(x) for x in ticks])
        ax.tick_params(labelsize=tick_fs)

        if inset and len(xs) > 1:
            axins = inset_axes(ax, width="42%", height="42%", loc="upper right", borderpad=1)
            xmin, xmax = min(xs), max(xs)
            if inset_xlim is not None:
                zlo, zhi = inset_xlim
            else:
                zlo, zhi = int(xmin + 0.3 * (xmax - xmin)), xmax
            for series in series_order:
                d = sub[(sub[series_col] == series) & (sub[x_col].between(zlo, zhi))].sort_values(x_col)
                color, ls = style_for(series)
                axins.plot(d[x_col], d[y_mean], ls=ls, marker="o", lw=1.3, ms=3, color=color)
            axins.set_xlim(zlo, zhi)
            if inset_xticks is not None:
                axins.set_xticks(list(inset_xticks))
                if x_tick_formatter is not None:
                    axins.set_xticklabels([x_tick_formatter(x) for x in inset_xticks])
            axins.set_ylim(*ylims)
            axins.grid(alpha=0.25, linestyle="--", linewidth=0.5)
            axins.tick_params(labelsize=max(8, tick_fs - 2))

    # Hide unused axes
    for j in range(len(unique_facets), len(axes)):
        axes[j].axis("off")

    # Shared legend
    # Order legend according to legend_order if provided
    if legend_order is not None:
        # Build ordered handles, supporting explicit spacers: "<spacer>" or ""
        ordered_labels: List[str] = []
        ordered_handles: List[plt.Line2D] = []
        for key in legend_order:
            if key in ("<spacer>", ""):
                spacer_label = " "  # non-empty so legend reserves column space
                ordered_labels.append(spacer_label)
                ordered_handles.append(Line2D([], [], lw=0, alpha=0))
                continue
            lbl = label_map.get(key, key)
            if lbl not in handles:
                handles[lbl] = Line2D([], [], lw=0, alpha=0)
            ordered_labels.append(lbl)
            ordered_handles.append(handles[lbl])
    else:
        ordered_labels = list(handles.keys())
        ordered_handles = list(handles.values())

    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        ncol=(legend_ncol if legend_ncol is not None else min(6, len(ordered_handles))),
        bbox_to_anchor=(legend_bbox if legend_bbox is not None else (0.5, 1.10)),
        frameon=False,
        fontsize=legend_fs,
    )
    fig.subplots_adjust(top=(top_adjust if top_adjust is not None else 0.88))
    fig.savefig(savepath, dpi=300, bbox_inches="tight")


