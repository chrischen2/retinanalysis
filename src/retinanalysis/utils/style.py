"""Publication-grade color/style conventions for retinanalysis plots.

Single-source-of-truth for colors and line widths so every figure in the
package looks consistent and reproducible. Two design choices:

1. **Categorical palette: Okabe-Ito** (Wong, Nature Methods 2011) — eight
   colors chosen to be distinguishable for the most common kinds of
   color-vision deficiency. Used directly for cell types and any other
   categorical split.

2. **Sequential / condition palette: matplotlib ``cividis``** — perceptually
   uniform, monotonic in luminance, and colorblind-safe. Used for ordinal
   stimulus conditions (e.g. ``backgroundScale`` going from dim → bright).

Cell-type → color is deterministic so a given type plots the same color
in every figure. Other types fall back to Okabe-Ito in order.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# --- Okabe-Ito (Wong 2011) ---------------------------------------------------
OKABE_ITO: List[str] = [
    '#000000',  # black
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#F0E442',  # yellow
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7',  # reddish purple
]

# Slate gray (#666666) is a useful fallback for "other / unclassified".
NEUTRAL_GRAY = '#666666'

# Canonical cell-type → color. ON cells get warm colors, OFF cells cool —
# the convention used in most primate-retina figures. Anything not listed
# gets the next Okabe-Ito color (see :func:`color_for_celltype`).
CELLTYPE_COLORS: Dict[str, str] = {
    'OnP':  '#D55E00',  # vermillion
    'OffP': '#0072B2',  # blue
    'OnM':  '#E69F00',  # orange
    'OffM': '#56B4E9',  # sky blue
    'OnS':  '#CC79A7',  # reddish purple
    'OffS': '#009E73',  # bluish green
    'SBC':  '#F0E442',  # yellow
    'A1':   '#000000',  # black
    'BT':   '#999999',  # gray
}

# Canonical "main" cell types — the publication-ready subset for mosaic
# and cell-type-grouped plots. Override per-call when you need more.
MAIN_CELL_TYPES: List[str] = ['OnP', 'OffP', 'OnM', 'OffM']


def color_for_celltype(cell_type: str, fallback_index: int = 0) -> str:
    """Return the canonical hex color for ``cell_type``.

    Falls back to Okabe-Ito[``fallback_index`` mod 8] for types not in the
    canonical map (e.g. ``OnMystery``).
    """
    if cell_type in CELLTYPE_COLORS:
        return CELLTYPE_COLORS[cell_type]
    return OKABE_ITO[fallback_index % len(OKABE_ITO)]


# Colors for types outside the canonical map — Amacrine, OnMystery, whatever a
# new typing file introduces. Every Okabe-Ito slot is already spoken for by
# CELLTYPE_COLORS, so without these the fallback below is empty and *every*
# uncanonical type renders the same gray, which is not an identity at all.
#
# These two are checked, not chosen by eye: against the four main types they
# clear the chroma floor, the normal-vision floor (worst adjacent ΔE 18.3) and
# the CVD separation band. NEUTRAL_GRAY fails both chroma and normal-vision
# against blue, which is why it is the last resort rather than the first.
EXTENSION_COLORS: List[str] = [
    '#7F3C8D',  # purple
    '#DC267F',  # magenta
]


def colors_for_celltypes(cell_types: Sequence[str]) -> Dict[str, str]:
    """Build a ``{cell_type: hex}`` map honoring the canonical assignments.

    Types outside :data:`CELLTYPE_COLORS` draw from unused Okabe-Ito slots
    first, then :data:`EXTENSION_COLORS`, and only fall back to gray once
    those run out. Resolve a whole set through this rather than calling
    :func:`color_for_celltype` per type — that returns the same fallback color
    for every uncanonical type, so they come out indistinguishable.
    """
    out: Dict[str, str] = {}
    pool = [c for c in OKABE_ITO if c not in CELLTYPE_COLORS.values()]
    pool += [c for c in EXTENSION_COLORS if c not in CELLTYPE_COLORS.values()]
    fallback_iter = iter(pool)
    for ct in cell_types:
        if ct in CELLTYPE_COLORS:
            out[ct] = CELLTYPE_COLORS[ct]
        else:
            out[ct] = next(fallback_iter, NEUTRAL_GRAY)
    return out


# --- Sequential palette for ordinal conditions -------------------------------

def colors_for_conditions(
    conditions: Sequence,
    cmap_name: str = 'cividis',
    lo: float = 0.15,
    hi: float = 0.85,
) -> Dict:
    """Sample a sequential colormap evenly across ``conditions``.

    Skips the deep-shadow / pale-highlight ends of the colormap by default
    (``lo`` / ``hi``) so neighboring conditions stay visually distinct on
    white backgrounds.

    Returns a dict ``{condition: hex_string}`` ordered as the input sequence.
    """
    n = len(conditions)
    if n == 0:
        return {}
    if n == 1:
        # Single condition → mid-luminance color of the cmap.
        cmap = plt.get_cmap(cmap_name)
        c = mpl.colors.to_hex(cmap(0.5))
        return {conditions[0]: c}
    cmap = plt.get_cmap(cmap_name)
    samples = np.linspace(lo, hi, n)
    return {cond: mpl.colors.to_hex(cmap(s)) for cond, s in zip(conditions, samples)}


# --- Global rc params --------------------------------------------------------

def apply_publication_style(
        font_size: float = 10,
        axis_linewidth: float = 0.5,
        major_tick_length: float = 2,
        minor_tick_length: float = 2,
        line_width: float = 1,
        tick_label_pad: float = -1.5,
        legend_frame: bool = True,
        rc: Optional[dict] = None,
        ):
    """Apply the package-wide Igor-style defaults to future figures.

    This is the Matplotlib counterpart of the lab's Igor Pro 7
    ``FormatGraph`` macro. Figure width/height and Igor's display-only
    ``expand`` value are intentionally omitted. Pass individual keyword
    arguments, or additional Matplotlib settings through ``rc``, to adjust a
    specific default. Existing figures can be updated with
    :func:`format_figure`.
    """
    font_size = float(font_size)
    mpl.rcParams.update({
        'figure.dpi': 110,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'font.family': ['Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif'],
        'font.weight': 'light',
        'font.size': font_size,
        'axes.titlesize': font_size,
        'axes.titleweight': 'light',
        'axes.labelsize': font_size,
        'axes.labelweight': 'light',
        'axes.linewidth': float(axis_linewidth),
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': mpl.cycler(color=OKABE_ITO),
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': float(major_tick_length),
        'ytick.major.size': float(major_tick_length),
        'xtick.minor.size': float(minor_tick_length),
        'ytick.minor.size': float(minor_tick_length),
        'xtick.major.width': float(axis_linewidth),
        'ytick.major.width': float(axis_linewidth),
        'xtick.minor.width': float(axis_linewidth),
        'ytick.minor.width': float(axis_linewidth),
        'xtick.major.pad': float(tick_label_pad),
        'ytick.major.pad': float(tick_label_pad),
        'legend.fontsize': font_size,
        'legend.frameon': bool(legend_frame),
        'legend.fancybox': False,
        'legend.edgecolor': 'black',
        'legend.framealpha': 1.0,
        'lines.linewidth': float(line_width),
        'lines.solid_capstyle': 'butt',
        'image.cmap': 'cividis',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    if rc:
        mpl.rcParams.update(rc)


def _axes_list(fig, axes=None):
    """Normalize one axis, an array of axes, or ``None`` to a flat list."""
    from matplotlib.axes import Axes

    if axes is None:
        return list(fig.axes)
    if isinstance(axes, Axes):
        return [axes]
    return [axis for axis in np.asarray(axes, dtype=object).ravel()
            if isinstance(axis, Axes)]


def format_figure(
        fig=None,
        axes=None,
        *,
        font_size: float = 10,
        font_family: Union[str, Sequence[str]] = (
            'Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif'),
        font_weight: str = 'light',
        nticks: Optional[int] = 3,
        axis_enable: Optional[Tuple[float, float]] = (0.05, 1.0),
        margins_pt: Optional[Tuple[float, float, float, float]] = (35, 27, 21, 21),
        axis_linewidth: float = 0.5,
        major_tick_length: float = 2,
        minor_tick_length: float = 2,
        tick_label_pad: float = -1.5,
        line_width: Optional[float] = 1,
        legend: Optional[bool] = None,
        legend_frame: bool = True,
        legend_kwargs: Optional[dict] = None,
        ):
    """Format an existing Matplotlib figure like Igor Pro ``FormatGraph``.

    Parameters are independently adjustable. ``axes`` can be one axis or any
    array/sequence of axes; by default every axis in ``fig`` is formatted.
    ``nticks`` only replaces automatic numeric/log locators, preserving fixed
    categorical ticks. ``axis_enable=(0.05, 1)`` shortens the left and bottom
    axes to the Igor fractional range. ``margins_pt`` is ordered left, bottom,
    top, right. ``legend=None`` formats existing legends, ``True`` also creates
    missing legends from labeled artists, and ``False`` removes legends.

    Returns the formatted figure so calls can be chained or assigned.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.transforms import Bbox
    from matplotlib.ticker import AutoLocator, LogLocator, MaxNLocator

    if fig is None:
        if axes is not None:
            candidates = _axes_list(plt.gcf(), axes)
            if not candidates:
                raise ValueError('axes did not contain a Matplotlib Axes object')
            fig = candidates[0].figure
        else:
            fig = plt.gcf()
    formatted_axes = _axes_list(fig, axes)
    families = [font_family] if isinstance(font_family, str) else list(font_family)
    if axis_enable is not None:
        start, stop = axis_enable
        if not (0 <= start < stop <= 1):
            raise ValueError('axis_enable must satisfy 0 <= start < stop <= 1.')

    for ax in formatted_axes:
        for spine in ax.spines.values():
            spine.set_linewidth(axis_linewidth)
        ax.tick_params(
            axis='both', which='major', direction='in', width=axis_linewidth,
            length=major_tick_length, pad=tick_label_pad)
        ax.tick_params(
            axis='both', which='minor', direction='in', width=axis_linewidth,
            length=minor_tick_length)

        if nticks is not None:
            if int(nticks) < 1:
                raise ValueError('nticks must be at least 1 or None.')
            for axis in (ax.xaxis, ax.yaxis):
                locator = axis.get_major_locator()
                if isinstance(locator, (AutoLocator, MaxNLocator)):
                    axis.set_major_locator(MaxNLocator(nbins=int(nticks)))
                elif isinstance(locator, LogLocator):
                    axis.set_major_locator(LogLocator(numticks=int(nticks)))

        texts = [ax.title, ax.xaxis.label, ax.yaxis.label]
        texts.extend(ax.get_xticklabels())
        texts.extend(ax.get_yticklabels())
        for text in texts:
            text.set_fontfamily(families)
            text.set_fontweight(font_weight)
            text.set_fontsize(font_size)

        if line_width is not None:
            for line in ax.lines:
                line.set_linewidth(line_width)
            for collection in ax.collections:
                if isinstance(collection, LineCollection):
                    collection.set_linewidth(line_width)

        current_legend = ax.get_legend()
        if legend is False and current_legend is not None:
            current_legend.remove()
            current_legend = None
        elif legend is True and current_legend is None:
            handles, labels = ax.get_legend_handles_labels()
            keep = [(handle, label) for handle, label in zip(handles, labels)
                    if label and not label.startswith('_')]
            if keep:
                current_legend = ax.legend(
                    *zip(*keep), **dict(legend_kwargs or {}))
        if current_legend is not None:
            _format_legend(
                current_legend, families, font_weight, font_size,
                legend_frame, axis_linewidth)

    if legend is False:
        for figure_legend in list(fig.legends):
            figure_legend.remove()
    else:
        for figure_legend in fig.legends:
            _format_legend(
                figure_legend, families, font_weight, font_size,
                legend_frame, axis_linewidth)

    for text in fig.texts:
        text.set_fontfamily(families)
        text.set_fontweight(font_weight)
        text.set_fontsize(font_size)

    if margins_pt is not None:
        if len(margins_pt) != 4 or any(float(value) < 0 for value in margins_pt):
            raise ValueError('margins_pt must contain four non-negative values.')
        left, bottom, top, right = (float(value) for value in margins_pt)
        width_pt, height_pt = fig.get_size_inches() * 72
        if left + right >= width_pt or bottom + top >= height_pt:
            raise ValueError('margins_pt must leave a positive plotting area.')
        fig.subplots_adjust(
            left=left / width_pt, bottom=bottom / height_pt,
            right=1 - right / width_pt, top=1 - top / height_pt)

    if axis_enable is not None:
        start, stop = axis_enable
        for ax in formatted_axes:
            subplotspec = ax.get_subplotspec()
            if subplotspec is not None:
                base = subplotspec.get_position(fig)
            else:
                base = getattr(ax, '_ra_format_base_position', ax.get_position()).frozen()
                ax._ra_format_base_position = base
            ax.set_position(Bbox.from_extents(
                base.x0 + start * base.width,
                base.y0 + start * base.height,
                base.x0 + stop * base.width,
                base.y0 + stop * base.height))

    return fig


def _format_legend(legend, families, font_weight, font_size,
                   frame_on, axis_linewidth):
    """Apply shared typography and border settings to one legend."""
    legend.set_frame_on(frame_on)
    frame = legend.get_frame()
    frame.set_linewidth(axis_linewidth)
    frame.set_edgecolor('black')
    frame.set_alpha(1.0)
    for text in legend.get_texts():
        text.set_fontfamily(families)
        text.set_fontweight(font_weight)
        text.set_fontsize(font_size)


def diverging_cmap(name: str = 'ra_diverging'):
    """Blue ↔ neutral gray ↔ vermillion, for values with a meaningful midpoint.

    Use when zero (or 1.0, or "no change") means something and the two
    directions mean opposite things — a cell firing more than usual versus
    less. Sequential ramps are for magnitude alone and can't show that.

    The two poles are the Okabe-Ito blue and vermillion already used for OFF
    and ON parasols, so the figure stays inside one palette, and they read as
    opposite because one is cool and one warm. The midpoint is neutral gray,
    not a third hue: a hue there would read as its own category rather than as
    "nothing happening".
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        name, ['#0072B2', '#8fb8d0', '#e8e6e3', '#e0a179', '#D55E00'])
