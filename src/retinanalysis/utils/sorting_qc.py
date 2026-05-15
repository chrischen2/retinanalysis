"""Sampled spike-sorting QC — raw traces with multi-cell asterisks.

PSTH/raster QC tells you whether spike *times* line up with the stimulus;
it doesn't tell you whether the sorter actually assigned the spike to
the right cell. This module samples a handful of cells (top-firing
visual-QC-``good``) and plots the raw voltage on each one's primary
electrode for a few entire epochs, with color-coded asterisks marking
the spikes assigned to the target cell *and* every other cell whose EI
peaks on the same electrode. A clean sort shows red asterisks tracking
real spikes in the trace; co-occurring colored asterisks from a neighbor
suggest that unit is being merged in.

Public entry point: :func:`sample_and_plot_sorting_qc`. Loads QC + visual-QC
CSVs from disk, builds the candidate pool, samples cells, and produces
one figure per cell with one panel per epoch. Designed to be callable
directly from a notebook cell without any extra plumbing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..config.settings import OUTPUT_DIR


__all__ = ['sample_sorting_qc_cells', 'sample_and_plot_sorting_qc']


def _resolve_protocol_subdir(response_block, protocol_subdir, datafile_name,
                              append_datafile_to_subdir) -> str:
    """Mirror the logic in ``analyze_experiment`` for the default subdir."""
    from .cell_plot_archive import protocol_short_name
    base = protocol_short_name(response_block.protocol_name)
    if protocol_subdir is not None:
        return protocol_subdir
    if append_datafile_to_subdir:
        df_name = datafile_name or getattr(response_block, 'datafile_name', None)
        if df_name:
            return f'{base}_{df_name}'
    return base


def sample_sorting_qc_cells(
    response_block,
    *,
    exp_name: Optional[str] = None,
    protocol_subdir: Optional[str] = None,
    append_datafile_to_subdir: bool = False,
    output_root: Optional[str] = None,
    n_cells: int = 4,
    per_cell_types: bool = True,
    rate_col: str = 'mean_rate_hz',
    require_visual_qc_good: bool = True,
) -> pd.DataFrame:
    """Pick ``n_cells`` for sorting QC.

    Reads ``<OUTPUT>/<exp>/<protocol_subdir>/qc.csv`` and (when present)
    ``visual_qc.csv`` written by ``analyze_experiment``. Builds the
    candidate pool as **QC-pass ∩ visual-QC ``'good'``** (or just
    QC-pass if no visual QC has been done and ``require_visual_qc_good``
    is False), sorts by ``rate_col`` descending, and returns the top
    ``n_cells`` — one per ``cell_type`` when ``per_cell_types=True``,
    otherwise top-N across all types.

    Returns a DataFrame with at least ``cell_id, cell_type, mean_rate_hz``.
    Empty when nothing passes the filters.
    """
    exp = exp_name or getattr(response_block, 'exp_name', None)
    if exp is None:
        raise ValueError('Cannot resolve exp_name; pass it explicitly.')
    short = _resolve_protocol_subdir(
        response_block, protocol_subdir,
        getattr(response_block, 'datafile_name', None),
        append_datafile_to_subdir,
    )
    root = Path(output_root) if output_root else Path(OUTPUT_DIR)
    proto_root = root / exp / short
    qc_path = proto_root / 'qc.csv'
    vqc_path = proto_root / 'visual_qc.csv'

    if not qc_path.exists():
        raise FileNotFoundError(
            f'No qc.csv at {qc_path}. Run §17 (analyze_experiment) first.')
    qc_df = pd.read_csv(qc_path)

    pool = qc_df.loc[qc_df['passes']].copy() if 'passes' in qc_df.columns else qc_df.copy()

    if vqc_path.exists():
        vqc_df = pd.read_csv(vqc_path)
        good = set(vqc_df.loc[vqc_df['tag'] == 'good', 'cell_id'].astype(int))
        pool = pool.loc[pool['cell_id'].astype(int).isin(good)]
    elif require_visual_qc_good:
        # visual_qc.csv is optional; missing one is fine on first-time runs.
        pass

    if rate_col in pool.columns:
        pool = pool.sort_values(rate_col, ascending=False)

    if per_cell_types and 'cell_type' in pool.columns:
        sample = pool.groupby('cell_type', as_index=False).head(1).head(n_cells)
    else:
        sample = pool.head(n_cells)

    keep = [c for c in ('cell_id', 'cell_type', rate_col, 'n_epochs')
            if c in sample.columns]
    return sample[keep].reset_index(drop=True)


def sample_and_plot_sorting_qc(
    response_block,
    *,
    exp_name: Optional[str] = None,
    protocol_subdir: Optional[str] = None,
    append_datafile_to_subdir: bool = False,
    output_root: Optional[str] = None,
    n_cells: int = 4,
    n_epochs: int = 2,
    per_cell_types: bool = True,
    max_other_cells: int = 6,
    require_visual_qc_good: bool = True,
    figsize_per_epoch: Tuple[float, float] = (6.5, 2.8),
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[plt.Figure]]:
    """Sample top-firing cells and plot full-epoch raw traces for each.

    One figure per cell; one subplot per epoch (full duration of that
    epoch — no time-window slicing). The cell's spikes appear as red
    asterisks at the top; spikes of other cells whose primary electrode
    matches appear as color-coded asterisks slightly below. The
    neighbor search is restricted to the QC-pass set so noise units
    don't clutter the panel.

    Parameters
    ----------
    response_block : MEAResponseBlock
        Built by ``ra.create_mea_pipeline`` (cell §3).
    exp_name : str, optional
        Defaults to ``response_block.exp_name``.
    protocol_subdir : str, optional
        Subdir under ``<OUTPUT>/<exp>/`` where qc.csv / visual_qc.csv
        live. Defaults to the protocol short-name (so the resolution
        matches ``analyze_experiment``). Override to disambiguate when
        the date has multiple datafiles of the same protocol.
    append_datafile_to_subdir : bool
        If True and ``protocol_subdir is None``, auto-append the
        datafile name (matches the §17 flag of the same name).
    n_cells : int
        Cells to sample. Top by ``mean_rate_hz``.
    n_epochs : int
        Epochs to show per cell. Always the first ``n_epochs``.
    per_cell_types : bool
        True (default): top-1 firing rate per cell type, up to n_cells.
        False: top-N across all types.
    max_other_cells : int
        Cap on neighbor cells annotated per panel.
    require_visual_qc_good : bool
        When True (default) and visual_qc.csv exists, restrict to cells
        tagged 'good'. When False, the visual-QC filter is skipped
        even if the file exists.
    figsize_per_epoch : (w, h)
        Per-subplot figure size; passed to plt.subplots.

    Returns
    -------
    (sample_df, figures)
        ``sample_df`` is the DataFrame of cells chosen.
        ``figures`` is a list of matplotlib Figures (one per cell).
    """
    from ..classes.raw import RawTraces, plot_sorting_qc
    from .style import apply_publication_style

    apply_publication_style()

    # 1. Pick cells
    sample_df = sample_sorting_qc_cells(
        response_block,
        exp_name=exp_name, protocol_subdir=protocol_subdir,
        append_datafile_to_subdir=append_datafile_to_subdir,
        output_root=output_root, n_cells=n_cells,
        per_cell_types=per_cell_types,
        require_visual_qc_good=require_visual_qc_good,
    )
    if verbose:
        print(f'sampled {len(sample_df)} cells:')
        print(sample_df.to_string(index=False))
    if sample_df.empty:
        return sample_df, []

    # 2. Resolve candidate pool for neighbor lookup (all QC-passing cells —
    # so the neighbor list isn't polluted by noise units).
    exp = exp_name or response_block.exp_name
    short = _resolve_protocol_subdir(
        response_block, protocol_subdir,
        getattr(response_block, 'datafile_name', None),
        append_datafile_to_subdir,
    )
    root = Path(output_root) if output_root else Path(OUTPUT_DIR)
    qc_df = pd.read_csv(root / exp / short / 'qc.csv')
    candidate_pool = qc_df.loc[qc_df['passes'], 'cell_id'].astype(int).tolist()

    # 3. Build raw-trace loader.
    rt = RawTraces(response_block)

    # 4. Build figures. Outer loop on epochs so each epoch's raw block
    # is loaded exactly once and reused across all sampled cells.
    n_ep = min(n_epochs, response_block.n_epochs)
    epoch_idxs = list(range(n_ep))
    # Cell-major layout: one figure per cell with n_epochs columns.
    figures: List[plt.Figure] = []
    cell_ids = sample_df['cell_id'].astype(int).tolist()
    cell_types = sample_df['cell_type'].tolist() if 'cell_type' in sample_df.columns \
                 else [''] * len(cell_ids)

    for cid, ctype in zip(cell_ids, cell_types):
        fig, axes = plt.subplots(
            1, len(epoch_idxs),
            figsize=(figsize_per_epoch[0] * len(epoch_idxs), figsize_per_epoch[1]),
            squeeze=False,
        )
        for j, ep in enumerate(epoch_idxs):
            ax = axes[0, j]
            try:
                plot_sorting_qc(
                    rt, response_block, cid, ep,
                    start_time=0.0,
                    end_time=None,             # → full epoch
                    candidate_cell_ids=candidate_pool,
                    max_other_cells=max_other_cells,
                    ax=ax,
                    show_legend=(j == 0),
                )
            except Exception as exc:
                ax.set_title(f'cell {cid}, epoch {ep}: {exc!r}', fontsize=8)
                ax.axis('off')
        fig.suptitle(
            f'{exp} / {getattr(response_block, "datafile_name", "?")} — '
            f'cell {cid} ({ctype}): sorting QC',
            fontsize=10, y=1.001,
        )
        fig.tight_layout()
        figures.append(fig)

    return sample_df, figures
