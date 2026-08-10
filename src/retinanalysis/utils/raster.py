"""Raster + PSTH plots organized by cell type.

A raster shows one row per ``(cell, epoch)`` pair: spike times as vertical
ticks. The PSTH below is the mean instantaneous firing rate across all
rows (epochs averaged per cell, then averaged across cells).

Both the cell-stacked raster here and the per-cell raster in
:mod:`cell_plot_archive` accept a ``groupby_conditions`` argument — a dict
``{key: per_epoch_values}`` (analogous to a MATLAB cell array of names
plus their per-epoch values). Rows get reordered hierarchically by the
keys in insertion order, colored by the last key, and visually separated
at each transition of the first key (so groups are easy to read).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from .psth import epoch_spikes_to_psth, psth_time_axis
from .style import color_for_celltype, colors_for_conditions, NEUTRAL_GRAY


def _stable_key_for_sort(v):
    """Sort key that works for mixed str / numeric values."""
    return (isinstance(v, str), v)


def grouped_row_order(
    groupby_conditions: Dict[str, Sequence],
    epoch_indices: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[Tuple[int, Tuple]], Dict, List[str]]:
    """Sort epochs by the condition keys in insertion order.

    Parameters
    ----------
    groupby_conditions : dict {key: per-epoch sequence}
        Each value is a sequence of length ``n_epochs``. Keys are sorted
        in the order they appear in the dict — the first key is the
        outer group (image), the last is the innermost color split (bg
        scale).
    epoch_indices : sequence[int], optional
        Restrict to a subset of epochs (default: every epoch).

    Returns
    -------
    order : list[int]
        Reordered epoch indices.
    groups : list[(orig_idx, value_tuple)]
        Same length as ``order``; each entry carries the original epoch
        index and the tuple of condition values in key order.
    color_map : dict
        ``{last_key_value: hex}`` for coloring rows by the innermost key.
    keys : list[str]
        Insertion-ordered keys (same as ``list(groupby_conditions)``).
    """
    keys = list(groupby_conditions.keys())
    n_full = len(next(iter(groupby_conditions.values())))
    if epoch_indices is None:
        epoch_indices = range(n_full)
    items = []
    for i in epoch_indices:
        tup = tuple(groupby_conditions[k][i] for k in keys)
        items.append((i, tup))
    items.sort(key=lambda x: tuple(_stable_key_for_sort(v) for v in x[1]))
    order = [i for i, _ in items]
    last_key = keys[-1] if keys else None
    if last_key is not None:
        last_vals = sorted(
            {groupby_conditions[last_key][i] for i in epoch_indices},
            key=_stable_key_for_sort,
        )
        color_map = colors_for_conditions(last_vals)
    else:
        color_map = {}
    return order, items, color_map, keys


def plot_single_cell_raster(
    spike_times_by_epoch: Sequence[np.ndarray],
    t_start_ms: float = 0.0,
    t_end_ms: Optional[float] = None,
    groupby_conditions: Optional[Dict[str, Sequence]] = None,
    ax: Optional[Axes] = None,
    default_color: str = 'k',
    pre_time_ms: Optional[float] = None,
    stim_time_ms: Optional[float] = None,
    group_separator_color: str = '#bbbbbb',
    title: Optional[str] = None,
) -> Axes:
    """Single-cell raster (one row per epoch) with optional hierarchical grouping.

    ``groupby_conditions`` is a dict (insertion-ordered) of per-epoch
    condition values. Rows are sorted by the keys in order, colored by
    the last key's value, and a thin separator line is drawn whenever the
    *first* key's value changes (so image groups read as distinct blocks).
    Without ``groupby_conditions``, rows stay in epoch order and use
    ``default_color`` uniformly.

    Returns the axis it drew into.
    """
    n_epochs = len(spike_times_by_epoch)

    # Auto-pick t_end if not given
    if t_end_ms is None:
        max_t = 0.0
        for arr in spike_times_by_epoch:
            a = np.asarray(arr)
            if a.size:
                max_t = max(max_t, float(a.max()))
        t_end_ms = max(max_t, t_start_ms + 1.0)

    if ax is None:
        _, ax = plt.subplots(figsize=(11, max(3.5, n_epochs * 0.08)))

    # Row ordering + coloring
    if groupby_conditions:
        order, items, color_map, keys = grouped_row_order(groupby_conditions)
        row_colors = [color_map.get(tup[-1], NEUTRAL_GRAY) for _, tup in items]
        first_vals_in_row_order = [tup[0] for _, tup in items]
    else:
        order = list(range(n_epochs))
        items = [(i, (i,)) for i in order]
        keys = []
        color_map = {}
        row_colors = [default_color] * n_epochs
        first_vals_in_row_order = list(range(n_epochs))

    # Build LineCollection for all spikes
    segs = []
    cols = []
    for row, orig_idx in enumerate(order):
        a = np.asarray(spike_times_by_epoch[orig_idx], dtype=float)
        if a.size == 0:
            continue
        m = (a >= t_start_ms) & (a <= t_end_ms)
        if not m.any():
            continue
        sel = a[m]
        ys_lo = np.full(sel.size, row + 0.05)
        ys_hi = np.full(sel.size, row + 0.95)
        seg = np.stack([
            np.column_stack([sel, ys_lo]),
            np.column_stack([sel, ys_hi]),
        ], axis=1)
        segs.append(seg)
        cols.extend([row_colors[row]] * sel.size)
    if segs:
        lc = LineCollection(np.concatenate(segs), colors=cols, linewidths=0.6)
        ax.add_collection(lc)

    # Image-group separators (lines + side labels) at first-key transitions
    if keys and len(keys) >= 1:
        prev = object()
        group_rows = []  # (row_start, value)
        for row, v in enumerate(first_vals_in_row_order):
            if v != prev:
                group_rows.append((row, v))
                prev = v
        # Draw separator above each new group (except row 0)
        for row, _v in group_rows[1:]:
            ax.axhline(row, color=group_separator_color, lw=0.6, alpha=0.7)
        # Group labels on the right margin
        for j, (row_start, v) in enumerate(group_rows):
            row_end = group_rows[j + 1][0] if j + 1 < len(group_rows) else len(order)
            ax.text(
                t_end_ms * 1.005, (row_start + row_end) / 2.0,
                str(v), ha='left', va='center', fontsize=7,
                color='#444444',
            )

    # Stim onset/offset
    if pre_time_ms is not None and stim_time_ms is not None and stim_time_ms > 0:
        for x in (pre_time_ms, pre_time_ms + stim_time_ms):
            ax.axvline(x, color='red', lw=0.5, ls='--', alpha=0.6)

    ax.set_xlim(t_start_ms, t_end_ms)
    ax.set_ylim(len(order), 0)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('epoch (grouped)' if keys else 'epoch')
    if title:
        ax.set_title(title)

    # Legend for innermost color split
    if keys and color_map:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color=c, lw=2, label=f'{keys[-1]}={v}')
            for v, c in color_map.items()
        ]
        ax.legend(handles=handles, loc='upper right',
                  fontsize=7, framealpha=0.8)
    return ax


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
    raster_color: Optional[str] = None,
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

    if raster_color is None:
        raster_color = color_for_celltype(cell_type)

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


# ---------------------------------------------------------------------------
# Population raster across epochs: has the block drifted?
# ---------------------------------------------------------------------------

def epoch_raster_data(response_block, cell_type: str,
                      cell_ids: Optional[Iterable[int]] = None,
                      epoch_range: Optional[Tuple[int, int]] = None,
                      epoch_indices: Optional[Sequence[int]] = None):
    """``(cell_ids, spikes_by_cell, n_epochs)`` for one cell type.

    ``spikes_by_cell[i][e]`` is the ms spike-time array for cell ``i`` in epoch
    ``e``. Cells come back sorted by id so a row means the same cell in every
    epoch panel — which is the whole point when comparing early against late.

    ``epoch_range`` is a ``(start, stop)`` half-open slice. Pass the range an
    analysis actually uses and the panels show those epochs, so what you look
    at is what the numbers were computed from. ``epoch_indices`` does the same
    for a set of epochs no slice can express — one condition of an alternating
    protocol, say — and takes precedence when both are given.
    """
    ids, spikes = _gather_cell_spike_times(response_block, cell_type,
                                           cell_ids=cell_ids)
    if not ids:
        return [], [], 0
    order = np.argsort(np.asarray(ids))
    ids = [int(ids[i]) for i in order]
    spikes = [spikes[i] for i in order]
    if epoch_indices is not None:
        idx = [int(i) for i in epoch_indices]
        spikes = [[s[i] for i in idx if i < len(s)] for s in spikes]
    elif epoch_range is not None:
        spikes = [list(s)[epoch_range[0]:epoch_range[1]] for s in spikes]
    n_epochs = max((len(s) for s in spikes), default=0)
    return ids, spikes, n_epochs


def plot_epoch_rasters(response_block, cell_type: str,
                       n_first: int = 3, n_last: int = 3,
                       cell_ids: Optional[Iterable[int]] = None,
                       epoch_range: Optional[Tuple[int, int]] = None,
                       epoch_indices: Optional[Sequence[int]] = None,
                       t_start_ms: float = 0.0,
                       t_end_ms: Optional[float] = None,
                       pre_time_ms: Optional[float] = None,
                       stim_time_ms: Optional[float] = None,
                       color: Optional[str] = None,
                       max_yticks: int = 12,
                       title: Optional[str] = None):
    """Population raster for the first and last epochs of a block.

    One panel per epoch, y is cell (labelled by cell id), x is time within the
    epoch. The top row is the first ``n_first`` epochs and the bottom row the
    last ``n_last``, so a response that faded, or a unit that dropped out
    partway through the block, shows up as a row that is dense on top and
    empty underneath. When the block is short enough that the two ends would
    overlap, every epoch is drawn once in a single row instead.

    Cell order is by id and identical in every panel, so a row can be read
    straight across.

    Returns the Figure, or None when no cell of this type has spikes.
    """
    ids, spikes, n_epochs = epoch_raster_data(response_block, cell_type,
                                              cell_ids=cell_ids,
                                              epoch_range=epoch_range,
                                              epoch_indices=epoch_indices)
    if not ids or n_epochs == 0:
        print(f'No cells of type {cell_type} with spike times.')
        return None

    # Panels are titled by the epoch's index in the block, not its position in
    # the selection, so a panel can be found again in the epoch table.
    if epoch_indices is not None:
        epoch_labels = [int(i) for i in epoch_indices][:n_epochs]
    elif epoch_range is not None:
        epoch_labels = list(range(epoch_range[0], epoch_range[0] + n_epochs))
    else:
        epoch_labels = list(range(n_epochs))

    # Overlapping ends would draw the same epoch twice and imply a comparison
    # that isn't there, so short blocks collapse to one row of everything.
    # Equality is not overlap: e.g. 6 first + 6 last epochs from a 12-epoch
    # condition should form two complete, non-overlapping rows.
    if n_first <= 0 or n_last <= 0 or n_first + n_last > n_epochs:
        rows = [(list(range(n_epochs)), f'all {n_epochs} epochs')]
    else:
        rows = [(list(range(n_first)), f'first {n_first} epochs'),
                (list(range(n_epochs - n_last, n_epochs)), f'last {n_last} epochs')]

    if t_end_ms is None:
        finite = [float(np.max(arr)) for cell in spikes for arr in cell
                  if arr is not None and len(arr)]
        t_end_ms = max(finite) if finite else 1.0

    color = color or color_for_celltype(cell_type)
    n_cols = max(len(idx) for idx, _ in rows)

    fig, axs = plt.subplots(
        len(rows), n_cols, squeeze=False, sharex=True, sharey=True,
        figsize=(2.5 * n_cols + 1.0, 0.045 * len(ids) * len(rows) + 1.6 * len(rows)))

    step = max(1, int(np.ceil(len(ids) / max_yticks)))
    for r, (epoch_idx, row_label) in enumerate(rows):
        for c in range(n_cols):
            ax = axs[r][c]
            if c >= len(epoch_idx):
                ax.set_axis_off()
                continue
            epoch = epoch_idx[c]

            # One LineCollection for the whole panel: a plot call per spike is
            # unusably slow at a hundred cells times a few thousand spikes.
            segments = []
            for row, cell in enumerate(spikes):
                if epoch >= len(cell) or cell[epoch] is None:
                    continue
                for t in np.asarray(cell[epoch], dtype=float):
                    if t_start_ms <= t <= t_end_ms:
                        segments.append([(t, row + 0.1), (t, row + 0.9)])
            ax.add_collection(LineCollection(segments, colors=color,
                                             linewidths=0.5))

            if pre_time_ms is not None and stim_time_ms is not None:
                for x in (pre_time_ms, pre_time_ms + stim_time_ms):
                    ax.axvline(x, color=NEUTRAL_GRAY, linestyle='--',
                               linewidth=0.7, alpha=0.7)

            ax.set_xlim(t_start_ms, t_end_ms)
            ax.set_ylim(0, len(ids))
            ax.set_title(f'epoch {epoch_labels[epoch]}', fontsize=8)
            if c == 0:
                ax.set_yticks(np.arange(0, len(ids), step) + 0.5)
                ax.set_yticklabels([ids[i] for i in range(0, len(ids), step)],
                                   fontsize=7)
                ax.set_ylabel(f'{row_label}\ncell ID', fontsize=8)
            if r == len(rows) - 1:
                ax.set_xlabel('Time in epoch (ms)', fontsize=8)

    fig.suptitle(title or f'{cell_type} — {len(ids)} cells, {n_epochs} epochs')
    fig.tight_layout()
    return fig


# Bookkeeping labels, not cell types: cells the EI match dropped, and cells the
# typing file never classified. Both are worth being able to look at, but
# neither should be what the selector opens on — 'Unmatched' is usually the
# largest group in the block.
_NON_TYPE_LABELS = {'Unmatched', 'Unknown'}


def _cell_type_options(response_block, cell_types: Optional[Sequence[str]],
                       minimum_n: int,
                       cell_ids: Optional[Iterable[int]] = None
                       ) -> List[Tuple[str, str]]:
    """``(label, cell_type)`` pairs, densest real type first, basics on show.

    ``cell_ids`` restricts the counts as well as the selection, so a label
    describes the cells that will actually be drawn rather than the whole
    block — the difference matters once a QC filter has been applied.
    """
    df = response_block.df_spike_times
    if 'cell_type' not in df.columns:
        response_block.add_cell_types()
        df = response_block.df_spike_times

    if cell_ids is not None:
        df = df[df['cell_id'].isin({int(c) for c in cell_ids})]

    options = []
    for cell_type, rows in df.groupby('cell_type'):
        if cell_types is not None and cell_type not in cell_types:
            continue
        n_cells = len(rows)
        if n_cells < minimum_n:
            continue
        # Mean spikes per cell per epoch — the one number that says whether
        # this type is worth opening.
        per_epoch = [len(arr) for cell in rows['spike_times']
                     for arr in cell if arr is not None]
        mean_spikes = float(np.mean(per_epoch)) if per_epoch else 0.0
        options.append((str(cell_type) in _NON_TYPE_LABELS, n_cells,
                        f'{cell_type} · {n_cells} cells · '
                        f'{mean_spikes:.0f} spikes/cell/epoch',
                        str(cell_type)))

    options.sort(key=lambda t: (t[0], -t[1]))
    return [(label, cell_type) for _, _, label, cell_type in options]


def browse_epoch_rasters(response_block, cell_types: Optional[Sequence[str]] = None,
                         minimum_n: int = 3, n_first: int = 3, n_last: int = 3,
                         pre_time_ms: Optional[float] = None,
                         stim_time_ms: Optional[float] = None,
                         t_end_ms: Optional[float] = None,
                         cell_ids: Optional[Iterable[int]] = None,
                         epoch_range: Optional[Tuple[int, int]] = None,
                         dpi: int = 110, **kwargs):
    """Dropdown over cell types, showing one type's epoch rasters at a time.

    The label carries what you need to choose without opening anything: how
    many cells the type has and how hard they fire per epoch. Each type is
    rendered on first selection and cached after, so a type you never open
    costs nothing — which matters here because a dense type is a few hundred
    thousand ticks to draw.

    ``cell_ids`` restricts to a subset — the cells that survived QC, normally.
    The dropdown counts shrink with it, so a label says how many cells the
    panel actually draws.

    Falls back to rendering every type inline when ipywidgets is missing.
    Returns the widget, or None on the fallback path.
    """
    from retinanalysis.utils.browse import figure_to_png, png_browser

    options = _cell_type_options(response_block, cell_types, minimum_n,
                                 cell_ids=cell_ids)
    if not options:
        print(f'No cell type has {minimum_n} or more cells with spike times.')
        return None

    def _render(cell_type):
        fig = plot_epoch_rasters(
            response_block, cell_type, n_first=n_first, n_last=n_last,
            pre_time_ms=pre_time_ms, stim_time_ms=stim_time_ms,
            t_end_ms=t_end_ms, cell_ids=cell_ids,
            epoch_range=epoch_range, **kwargs)
        return None, figure_to_png(fig, dpi=dpi)

    box = png_browser(options, _render, description='Cell type:',
                      empty_message='No cell types to browse.')
    if box is not None:
        return box

    for _, cell_type in options:
        plot_epoch_rasters(response_block, cell_type, n_first=n_first,
                           n_last=n_last, pre_time_ms=pre_time_ms,
                           stim_time_ms=stim_time_ms, t_end_ms=t_end_ms,
                           cell_ids=cell_ids, epoch_range=epoch_range, **kwargs)
        plt.show()
    return None


# ---------------------------------------------------------------------------
# Spike count per epoch: did the block hold up?
# ---------------------------------------------------------------------------

def epoch_count_matrix(response_block, cell_type: str,
                       cell_ids: Optional[Iterable[int]] = None,
                       epoch_range: Optional[Tuple[int, int]] = None,
                       epoch_indices: Optional[Sequence[int]] = None,
                       t_start_ms: float = 0.0,
                       t_end_ms: Optional[float] = None):
    """``(cell_ids, counts)`` where ``counts`` is ``(n_cells, n_epochs)``.

    Spike count per cell per epoch, optionally restricted to a time window
    within the epoch. Rows are ordered by cell id, matching
    :func:`plot_epoch_rasters` so the two figures can be read together.
    ``epoch_range`` / ``epoch_indices`` restrict the columns the same way
    they do there.
    """
    from .protocol_qc import per_epoch_spike_counts

    ids, spikes, n_epochs = epoch_raster_data(response_block, cell_type,
                                              cell_ids=cell_ids,
                                              epoch_range=epoch_range,
                                              epoch_indices=epoch_indices)
    if not ids or n_epochs == 0:
        return [], np.zeros((0, 0))

    counts = np.zeros((len(ids), n_epochs), dtype=float)
    for row, cell in enumerate(spikes):
        per_epoch = per_epoch_spike_counts(cell, t_start_ms=t_start_ms,
                                           t_end_ms=t_end_ms)
        counts[row, :len(per_epoch)] = per_epoch[:n_epochs]
    return ids, counts


# Fixed marker order, paired with the fixed color order. Shape is a second,
# fully colorblind-safe identity channel: the cell-type palette has one
# adjacent pair (OnS/OffS) that only just clears the deuteranopia threshold,
# and three hues below 3:1 contrast on white, so color alone cannot carry
# identity here.
_SERIES_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

# Past 8 series adjacent colors blur whatever the palette, so the tail is
# dropped rather than given a generated hue.
_MAX_SERIES = 8


def plot_epoch_spike_counts(response_block, cell_types: Optional[Sequence[str]] = None,
                            minimum_n: int = 3,
                            include_non_types: bool = False,
                            t_start_ms: float = 0.0,
                            t_end_ms: Optional[float] = None,
                            title: Optional[str] = None):
    """Mean spikes per cell against epoch index, one line per cell type.

    Two stacked panels over the same x axis, because the two questions need
    different units and putting them on one axis would mean a second y scale:

    - **Top, absolute** — mean spikes per cell per epoch. This is the raw
      answer to "how much did each type fire in each epoch", but types differ
      by an order of magnitude in rate, so a sparse type sits flat near zero
      and its shape is unreadable.
    - **Bottom, relative** — the same series divided by each type's own mean
      across epochs, so every type starts from 1.0 and the panel shows *change*
      rather than rate. This is where drift and adaptation are legible, and
      it is the only way to compare a 400-spike type against a 40-spike one
      without a second axis.

    Shaded bands are the standard error across cells within a type. Every
    series carries a distinct marker shape as well as its color, and the
    legend names each: three of the cell-type hues fall below 3:1 contrast
    against white and the OnS/OffS pair only just clears the deuteranopia
    threshold, so identity must not rest on hue alone.

    ``Unmatched`` and ``Unknown`` are left out unless ``include_non_types``
    is set. They are bookkeeping labels rather than populations — averaging
    over "every cell the EI match dropped" is not a quantity about anything —
    and on a typical block ``Unmatched`` is the largest group, so it would
    dominate a panel it has no business being in.

    Returns the Figure, or None when no type clears ``minimum_n``.
    """
    from .style import apply_publication_style, colors_for_celltypes

    apply_publication_style()

    options = _cell_type_options(response_block, cell_types, minimum_n)
    if not include_non_types:
        options = [(lbl, ct) for lbl, ct in options if ct not in _NON_TYPE_LABELS]
    if len(options) > _MAX_SERIES:
        dropped = [ct for _, ct in options[_MAX_SERIES:]]
        print(f'Showing the {_MAX_SERIES} densest types; dropped {dropped}.')
        options = options[:_MAX_SERIES]
    if not options:
        print(f'No cell type has {minimum_n} or more cells with spike times.')
        return None

    series = []
    for _, cell_type in options:
        ids, counts = epoch_count_matrix(response_block, cell_type,
                                         t_start_ms=t_start_ms, t_end_ms=t_end_ms)
        if not ids:
            continue
        mean = counts.mean(axis=0)
        sem = (counts.std(axis=0, ddof=1) / np.sqrt(counts.shape[0])
               if counts.shape[0] > 1 else np.zeros_like(mean))
        series.append((cell_type, len(ids), mean, sem))

    if not series:
        print('No cell type had spike times to count.')
        return None

    n_epochs = max(len(m) for _, _, m, _ in series)
    epochs = np.arange(n_epochs)

    # One map for the whole set, so an uncanonical type (Amacrine, OnMystery)
    # gets its own unused Okabe-Ito slot. Resolving per series instead hands
    # every unmapped type the same fallback, and they render identically.
    colors = colors_for_celltypes([ct for ct, _, _, _ in series])

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7.6, 6.0))
    for ax, relative in zip(axs, (False, True)):
        for s_idx, (cell_type, n_cells, mean, sem) in enumerate(series):
            color = colors[cell_type]
            # Each type's own mean over epochs is the denominator, so 1.0 means
            # "this type's typical epoch" rather than any cross-type baseline.
            scale = mean.mean() if relative and mean.mean() > 0 else 1.0
            y, err = mean / scale, sem / scale
            x = np.arange(len(y))
            ax.plot(x, y, color=color, linewidth=2.0,
                    marker=_SERIES_MARKERS[s_idx % len(_SERIES_MARKERS)],
                    markersize=5, markeredgewidth=0,
                    label=f'{cell_type} (n={n_cells})')
            ax.fill_between(x, y - err, y + err, color=color, alpha=0.18,
                            linewidth=0)

        if relative:
            ax.axhline(1.0, color=NEUTRAL_GRAY, linewidth=0.8)
            ax.set_ylabel("Relative to the type's own mean")
            ax.set_xlabel('Epoch index')
        else:
            ax.set_ylabel('Mean spikes per cell')

        ax.set_xlim(-0.5, n_epochs - 0.5 + 0.06 * n_epochs)
        # ``set_xticks(None)`` is not a request for automatic ticks: recent
        # Matplotlib versions reject it because tick locations must be a 1-D
        # sequence.  Leave the default locator untouched for long protocols.
        if n_epochs <= 25:
            ax.set_xticks(epochs)
        ax.grid(True, linewidth=0.5, alpha=0.35)
        ax.set_axisbelow(True)

    axs[0].legend(bbox_to_anchor=[1.02, 1], loc='upper left')
    fig.suptitle(title or f'Spikes per epoch — '
                          f'{getattr(response_block, "datafile_name", "block")}')
    fig.tight_layout()
    return fig


def plot_epoch_count_heatmap(response_block, cell_type: str,
                             normalize: str = 'cell',
                             cell_ids: Optional[Iterable[int]] = None,
                             t_start_ms: float = 0.0,
                             t_end_ms: Optional[float] = None,
                             max_yticks: int = 20,
                             vmax_ratio: float = 4.0,
                             title: Optional[str] = None):
    """Every cell's spike count in every epoch, as a cell × epoch image.

    The whole distribution over cells that the mean in
    :func:`plot_epoch_spike_counts` averages away — whether a dip is the
    population easing off together, or three cells dropping out while the rest
    hold steady. Rows are cells (ordered by id, as in the rasters), columns are
    epochs.

    An image rather than a 3-D surface on purpose. The data is a value on a
    grid of two discrete axes, and a surface would hide short rows behind tall
    ones, make values unreadable off the height axis, and imply that neighbouring
    cell ids are near each other in some meaningful sense. A heatmap shows every
    cell at once with no occlusion.

    ``normalize='cell'`` (default) divides each row by that cell's own mean
    across epochs and colors the log of that ratio on a diverging scale
    centered at 1× — blue below, vermillion above, neutral where nothing
    changed. This is what makes epoch structure visible; raw counts are
    dominated by cells simply having different firing rates. The scale is
    logarithmic so that halving and doubling sit the same distance from the
    midpoint; ``vmax_ratio`` sets the ends (4.0 → a quarter to four times).
    ``normalize=None`` shows raw counts on a single-hue sequential ramp
    instead.

    Returns the Figure, or None when the type has no cells with spikes.
    """
    from .style import apply_publication_style, diverging_cmap

    apply_publication_style()

    ids, counts = epoch_count_matrix(response_block, cell_type, cell_ids=cell_ids,
                                     t_start_ms=t_start_ms, t_end_ms=t_end_ms)
    if not len(ids):
        print(f'No cells of type {cell_type} with spike times.')
        return None

    if normalize == 'cell':
        # Silent cells would divide by zero; they stay at the midpoint, which
        # is honest — a cell with no spikes did not change across epochs.
        denom = counts.mean(axis=1, keepdims=True)
        ratio = np.divide(counts, denom, out=np.ones_like(counts),
                          where=denom > 0)
        # Log ratio, so halving and doubling are the same distance from the
        # midpoint. On a linear ratio scale everything below 1 is squeezed
        # into a third of the range and a silenced cell looks much like a
        # merely quiet one.
        floor = 2.0 ** -(np.log2(vmax_ratio) + 2)
        image = np.log2(np.maximum(ratio, floor))
        cmap = diverging_cmap()
        vmax = float(np.log2(vmax_ratio))
        vmin = -vmax
        cbar_label = "Spikes relative to the cell's own mean"
    else:
        image, cmap = counts, 'Blues'
        vmin, vmax = 0.0, float(counts.max()) if counts.size else 1.0
        cbar_label = 'Spikes in epoch'

    fig, ax = plt.subplots(figsize=(0.35 * counts.shape[1] + 4.0,
                                    0.06 * len(ids) + 2.2))
    im = ax.imshow(image, aspect='auto', interpolation='nearest',
                   cmap=cmap, vmin=vmin, vmax=vmax)

    step = max(1, int(np.ceil(len(ids) / max_yticks)))
    ax.set_yticks(np.arange(0, len(ids), step))
    ax.set_yticklabels([ids[i] for i in range(0, len(ids), step)], fontsize=7)
    ax.set_ylabel('Cell ID')
    ax.set_xlabel('Epoch index')
    if counts.shape[1] <= 25:
        ax.set_xticks(np.arange(counts.shape[1]))
    ax.set_title(title or f'{cell_type} — {len(ids)} cells, '
                          f'{counts.shape[1]} epochs')

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    if normalize == 'cell':
        # Ticks in ratio, not log2 — nobody wants to read "-1" for "half".
        ticks = np.arange(np.floor(vmin), np.ceil(vmax) + 1)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{2.0 ** t:g}×' for t in ticks])
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def browse_epoch_count_heatmaps(response_block,
                                cell_types: Optional[Sequence[str]] = None,
                                minimum_n: int = 3, dpi: int = 110, **kwargs):
    """Dropdown over cell types, one cell × epoch heatmap at a time.

    Same selector as :func:`browse_epoch_rasters` — labels carry the cell count
    and mean spikes per cell per epoch, real types sort ahead of ``Unmatched``
    and ``Unknown``, and each type renders once then caches.
    """
    from retinanalysis.utils.browse import figure_to_png, png_browser

    options = _cell_type_options(response_block, cell_types, minimum_n)
    if not options:
        print(f'No cell type has {minimum_n} or more cells with spike times.')
        return None

    def _render(cell_type):
        return None, figure_to_png(
            plot_epoch_count_heatmap(response_block, cell_type, **kwargs),
            dpi=dpi)

    box = png_browser(options, _render, description='Cell type:',
                      empty_message='No cell types to browse.')
    if box is not None:
        return box

    for _, cell_type in options:
        plot_epoch_count_heatmap(response_block, cell_type, **kwargs)
        plt.show()
    return None
