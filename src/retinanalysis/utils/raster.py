"""Raster + PSTH plots organized by cell type.

A raster shows one row per ``(cell, epoch)`` pair: spike times as vertical
ticks. The PSTH below is the mean instantaneous firing rate across all
rows (epochs averaged per cell, then averaged across cells).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .psth import epoch_spikes_to_psth, psth_time_axis


def _gather_cell_spike_times(
    response_block, cell_type: str, cell_ids: Optional[Iterable[int]] = None,
) -> Tuple[List[int], List[Sequence[np.ndarray]]]:
    """Return ``(cell_ids, [per-epoch spike-time lists])`` for a cell type.

    Honors an optional ``cell_ids`` filter (intersected with the type).
    """
    df = response_block.df_spike_times.query('cell_type == @cell_type')
    if cell_ids is not None:
        wanted = set(int(c) for c in cell_ids)
        df = df[df['cell_id'].isin(wanted)]
    ids = df['cell_id'].tolist()
    spikes = df['spike_times'].tolist()  # list of (n_epochs,) lists of ms arrays
    return ids, spikes


def plot_raster_with_psth(
    response_block,
    cell_type: str,
    t_end_ms: Optional[float] = None,
    t_start_ms: float = 0.0,
    cell_ids: Optional[Iterable[int]] = None,
    psth_sigma_ms: float = 10.0,
    sample_rate_hz: float = 1000.0,
    axes: Optional[Sequence[Axes]] = None,
    raster_color: str = 'k',
    psth_color: Optional[str] = None,
    pre_time_ms: Optional[float] = None,
    stim_time_ms: Optional[float] = None,
    title: Optional[str] = None,
) -> Tuple[Axes, Axes]:
    """Draw a raster + averaged PSTH for one cell type.

    Layout: two stacked axes — raster on top (one row per ``(cell, epoch)``
    pair), Gaussian-smoothed PSTH on bottom (Hz, averaged across rows).

    Parameters
    ----------
    response_block : MEAResponseBlock
        Spike-time source (``df_spike_times`` filtered by ``cell_type``).
    cell_type : str
        Which type to plot. Cells with other labels are ignored.
    t_end_ms, t_start_ms : float
        Time window to plot, in milliseconds from epoch onset. If
        ``t_end_ms`` is None, defaults to the max spike-time across all
        plotted epochs.
    cell_ids : iterable[int], optional
        Restrict to specific cells. Default: all cells of this type.
    psth_sigma_ms : float
        Gaussian-kernel sigma for the bottom PSTH (default 10 ms,
        matching ``spikeTimeToPSTH.m``).
    sample_rate_hz : float
        Internal sample rate for the PSTH (default 1 kHz).
    axes : (raster_ax, psth_ax), optional
        Pre-built axes. If None, a new ``(2,1)`` figure is created.
    raster_color, psth_color : str, optional
        Override colors. ``psth_color`` defaults to ``raster_color``.
    pre_time_ms, stim_time_ms : float, optional
        If both given, dashed verticals are drawn at ``pre`` and
        ``pre+stim`` to mark stim onset/offset.

    Returns
    -------
    (raster_ax, psth_ax)
    """
    ids, spikes_per_cell = _gather_cell_spike_times(
        response_block, cell_type, cell_ids,
    )
    if not ids:
        raise ValueError(f'No cells found for cell_type={cell_type!r}')

    # Default t_end_ms = max spike across all epochs across all cells
    if t_end_ms is None:
        max_t = 0.0
        for cell_epochs in spikes_per_cell:
            for arr in cell_epochs:
                if len(arr):
                    max_t = max(max_t, float(np.asarray(arr).max()))
        t_end_ms = max_t

    if axes is None:
        fig, (ax_r, ax_p) = plt.subplots(
            2, 1, figsize=(10, 5),
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True,
        )
    else:
        ax_r, ax_p = axes

    # --- Raster: scatter / vlines, one row per (cell, epoch) ---
    row = 0
    cell_band_centers = []  # for y-axis labels (one per cell)
    for cell_idx, (cid, cell_epochs) in enumerate(zip(ids, spikes_per_cell)):
        band_start = row
        for epoch_idx, arr in enumerate(cell_epochs):
            arr = np.asarray(arr, dtype=float)
            mask = (arr >= t_start_ms) & (arr <= t_end_ms)
            if mask.any():
                ax_r.vlines(arr[mask], row + 0.05, row + 0.95,
                            colors=raster_color, linewidth=0.4)
            row += 1
        band_end = row - 1
        cell_band_centers.append((cid, (band_start + band_end) / 2.0))
        # subtle band separator
        if cell_idx < len(ids) - 1:
            ax_r.axhline(row - 0.5, color='gray', lw=0.2, alpha=0.4)

    ax_r.set_ylim(row, 0)  # invert so cell 0 is at top
    ax_r.set_xlim(t_start_ms, t_end_ms)
    ax_r.set_ylabel('cell × epoch')
    # Tick at each cell-band center (subsample if many cells)
    if len(cell_band_centers) <= 30:
        ax_r.set_yticks([c for _, c in cell_band_centers])
        ax_r.set_yticklabels([str(cid) for cid, _ in cell_band_centers], fontsize=7)
    else:
        step = max(1, len(cell_band_centers) // 20)
        ax_r.set_yticks([c for _, c in cell_band_centers[::step]])
        ax_r.set_yticklabels([str(cid) for cid, _ in cell_band_centers[::step]], fontsize=7)

    if title is None:
        n_epochs_per_cell = max(len(s) for s in spikes_per_cell) if spikes_per_cell else 0
        title = (f'{cell_type} — {len(ids)} cells × {n_epochs_per_cell} epochs '
                 f'(rows = cell × epoch)')
    ax_r.set_title(title)

    # --- PSTH: mean per cell of per-epoch PSTHs, then mean across cells ---
    psth_per_cell = []
    for cell_epochs in spikes_per_cell:
        ep_psth = epoch_spikes_to_psth(
            cell_epochs, t_end_ms,
            psth_sigma_ms=psth_sigma_ms,
            sample_rate_hz=sample_rate_hz,
            t_start_ms=t_start_ms,
        )
        psth_per_cell.append(ep_psth.mean(axis=0))
    psth_per_cell = np.asarray(psth_per_cell)  # (n_cells, n_bins)
    psth_mean = psth_per_cell.mean(axis=0)
    psth_sem = psth_per_cell.std(axis=0) / np.sqrt(max(psth_per_cell.shape[0], 1))
    t = psth_time_axis(t_end_ms, sample_rate_hz, t_start_ms)
    color = psth_color or raster_color
    ax_p.plot(t, psth_mean, color=color, lw=1.5)
    ax_p.fill_between(t, psth_mean - psth_sem, psth_mean + psth_sem,
                      color=color, alpha=0.2, linewidth=0)
    ax_p.set_xlabel('time (ms)')
    ax_p.set_ylabel('rate (Hz)')

    # Stim onset/offset markers
    if pre_time_ms is not None and stim_time_ms is not None:
        for x, ls in [(pre_time_ms, '--'), (pre_time_ms + stim_time_ms, '--')]:
            ax_r.axvline(x, color='red', lw=0.6, ls=ls, alpha=0.7)
            ax_p.axvline(x, color='red', lw=0.6, ls=ls, alpha=0.7)

    return ax_r, ax_p
