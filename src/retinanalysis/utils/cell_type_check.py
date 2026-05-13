"""Cell-type verification panels: mosaic + temporal filter + ISI/ACF.

For each cell type, the canonical three-up panel:

* **Mosaic**: 1.6 σ RF ellipses (the spatial fingerprint).
* **Temporal filter**: per-cell green-channel STA timecourse plus the
  mean ± SEM (the temporal fingerprint).
* **ISI**: inter-spike-interval histogram (or auto-correlation) from the
  Vision ``.params`` file (refractory period + bursting fingerprint).

If any three are inconsistent across cells of a "type", classification is
suspect. This is the workflow people use to vet the typing file before
analysis.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse

from .vision_utils import get_ells


def _cells_of_type(analysis_chunk, cell_type: str,
                   typing_file: Optional[str] = None) -> List[int]:
    typing_file = typing_file or (analysis_chunk.typing_files[0]
                                  if analysis_chunk.typing_files else None)
    if typing_file is None:
        return []
    idx = analysis_chunk.typing_files.index(typing_file)
    df = analysis_chunk.df_cell_params
    return df.query(f'typing_file_{idx} == @cell_type')['cell_id'].tolist()


def _isi_bin_centers(analysis_chunk) -> np.ndarray:
    edges = np.asarray(analysis_chunk.isi_bin_edges)
    return 0.5 * (edges[:-1] + edges[1:])


def plot_cell_type_check(
    analysis_chunk,
    cell_type: str,
    typing_file: Optional[str] = None,
    cell_ids: Optional[Iterable[int]] = None,
    std_scaling: float = 1.6,
    axes: Optional[Sequence[Axes]] = None,
    color: Optional[str] = None,
    isi_xlim_ms: float = 200.0,
) -> Sequence[Axes]:
    """Three-panel verification: mosaic | mean temporal filter | mean ISI.

    Parameters
    ----------
    analysis_chunk : AnalysisChunk
    cell_type : str
        Class label as it appears in the typing file (already-normalized
        via ``ra.map_cell_type`` upstream — e.g. ``'OnP'``).
    cell_ids : iterable[int], optional
        Restrict to specific cells. Default: all of this type.
    axes : (mosaic_ax, timecourse_ax, isi_ax), optional
        Pre-built axes. New 1x3 figure if None.
    color : str, optional
        Override per-type color (default = ``f'C{hash(cell_type)%10}'``).
    isi_xlim_ms : float
        Right edge of the ISI plot in ms (default 200 — most of the
        interesting refractory + post-spike structure).
    """
    if cell_ids is None:
        ids = _cells_of_type(analysis_chunk, cell_type, typing_file)
    else:
        ids = list(int(c) for c in cell_ids)
    if not ids:
        raise ValueError(f'No cells of type {cell_type!r} in '
                         f'{analysis_chunk.exp_name}/{analysis_chunk.chunk_name}')

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    ax_m, ax_t, ax_i = axes

    if color is None:
        # Stable color per type name
        color = f'C{abs(hash(cell_type)) % 10}'

    # --- (1) Mosaic ---
    d_by_type = {cell_type: ids}
    d_ells, _ = get_ells(analysis_chunk, d_by_type,
                         std_scaling=std_scaling, units='pixels')
    canvas_w, canvas_h = analysis_chunk.canvas_size
    for cid, ell in d_ells[cell_type].items():
        patch = Ellipse(xy=ell.center, width=ell.width, height=ell.height,
                        angle=ell.angle, facecolor='none', edgecolor=color,
                        alpha=0.7, linewidth=0.9)
        ax_m.add_patch(patch)
    ax_m.set_xlim(0, canvas_w); ax_m.set_ylim(canvas_h, 0)
    ax_m.set_aspect('equal')
    ax_m.set_title(f'{cell_type}  (n={len(ids)})')
    ax_m.set_xlabel('canvas x (pix)'); ax_m.set_ylabel('canvas y (pix)')

    # --- (2) Temporal filter (green channel by default) ---
    # Average per-cell timecourses; show individual traces translucent.
    tcs = []
    for cid in ids:
        tc = analysis_chunk.d_timecourses.get(cid)
        if tc is None:
            continue
        # Use green for monochrome; for chromatic stim could plot RGB separately.
        tcs.append(tc['green'])
    if tcs:
        # Pad to common length (timecourses are usually all the same length).
        L = min(len(t) for t in tcs)
        tc_mat = np.stack([t[:L] for t in tcs])  # (n_cells, L)
        t_axis = np.arange(L)  # frames
        for row in tc_mat:
            ax_t.plot(t_axis, row, color=color, alpha=0.15, linewidth=0.6)
        mean = tc_mat.mean(axis=0)
        sem = tc_mat.std(axis=0) / np.sqrt(max(tc_mat.shape[0], 1))
        ax_t.plot(t_axis, mean, color=color, linewidth=1.8, label='mean')
        ax_t.fill_between(t_axis, mean - sem, mean + sem,
                          color=color, alpha=0.25, linewidth=0)
        ax_t.axhline(0, color='gray', lw=0.4, alpha=0.5)
        ax_t.set_xlabel('STA frame')
        ax_t.set_ylabel('contrast')
        ax_t.set_title('mean temporal filter')
    else:
        ax_t.text(0.5, 0.5, '(no timecourses)', transform=ax_t.transAxes,
                  ha='center', va='center')

    # --- (3) ISI distribution ---
    centers = _isi_bin_centers(analysis_chunk)
    isi_mat = []
    for cid in ids:
        isi = analysis_chunk.d_ISIs.get(cid)
        if isi is None:
            continue
        h = np.asarray(isi, dtype=float)
        # Some Vision builds save counts; normalize each cell to a density
        # so cells with very different spike counts don't dominate the mean.
        s = h.sum()
        isi_mat.append(h / s if s > 0 else h)
    if isi_mat:
        isi_mat = np.stack(isi_mat)  # (n_cells, n_bins)
        for row in isi_mat:
            ax_i.plot(centers, row, color=color, alpha=0.15, linewidth=0.6)
        mean = isi_mat.mean(axis=0)
        sem = isi_mat.std(axis=0) / np.sqrt(max(isi_mat.shape[0], 1))
        ax_i.plot(centers, mean, color=color, linewidth=1.8, label='mean')
        ax_i.fill_between(centers, mean - sem, mean + sem,
                          color=color, alpha=0.25, linewidth=0)
        ax_i.set_xlim(0, isi_xlim_ms)
        ax_i.set_xlabel('ISI (ms)')
        ax_i.set_ylabel('density')
        ax_i.set_title('mean ISI distribution')
    else:
        ax_i.text(0.5, 0.5, '(no ISI data)', transform=ax_i.transAxes,
                  ha='center', va='center')

    return ax_m, ax_t, ax_i


def plot_cell_type_grid(
    analysis_chunk,
    cell_types: Iterable[str],
    typing_file: Optional[str] = None,
    minimum_n: int = 3,
    std_scaling: float = 1.6,
    isi_xlim_ms: float = 200.0,
) -> plt.Figure:
    """Stack one :func:`plot_cell_type_check` row per cell type."""
    # Filter to types with enough cells
    types = []
    for ct in cell_types:
        ids = _cells_of_type(analysis_chunk, ct, typing_file)
        if len(ids) >= minimum_n:
            types.append((ct, ids))
    if not types:
        raise ValueError('No cell types with enough cells.')

    fig, axes = plt.subplots(len(types), 3, figsize=(13, 3.5 * len(types)),
                             squeeze=False)
    for r, (ct, ids) in enumerate(types):
        plot_cell_type_check(
            analysis_chunk, ct, typing_file=typing_file, cell_ids=ids,
            std_scaling=std_scaling, axes=axes[r],
            color=f'C{r % 10}', isi_xlim_ms=isi_xlim_ms,
        )
    fig.tight_layout()
    return fig
