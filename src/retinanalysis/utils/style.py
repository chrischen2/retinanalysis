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

from typing import Dict, Iterable, List, Sequence

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


def colors_for_celltypes(cell_types: Sequence[str]) -> Dict[str, str]:
    """Build a ``{cell_type: hex}`` map honoring the canonical assignments."""
    out: Dict[str, str] = {}
    fallback_iter = (c for c in OKABE_ITO if c not in CELLTYPE_COLORS.values())
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

def apply_publication_style():
    """Apply a minimal, readable rc preset across the package.

    Conservative on font size and line widths so figures stay legible at
    typical journal column widths without re-tuning per figure. Idempotent.
    """
    mpl.rcParams.update({
        'figure.dpi': 110,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': mpl.cycler(color=OKABE_ITO),
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'lines.linewidth': 1.2,
        'lines.solid_capstyle': 'round',
        'image.cmap': 'cividis',
    })
