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
    cell_types: Sequence[str] = ('OnP', 'OnM'),
    n_cells_per_type: int = 3,
    rate_col: str = 'mean_rate_hz',
    require_visual_qc_good: bool = True,
    sample_strategy: str = 'random',
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Pick ``n_cells_per_type`` cells per type for sorting QC.

    Reads ``<OUTPUT>/<exp>/<protocol_subdir>/qc.csv`` and (when present)
    ``visual_qc.csv`` written by ``analyze_experiment``. Builds the
    candidate pool as **QC-pass ∩ visual-QC ``'good'``** (or just
    QC-pass when visual_qc.csv is absent), then samples per cell type.

    Parameters
    ----------
    cell_types : sequence[str]
        Which cell types to sample. Default ``('OnP', 'OnM')``.
    n_cells_per_type : int
        Number of cells to keep per type.
    sample_strategy : ``'random'`` (default) or ``'top_rate'``
        - ``'random'``: uniform sample from the candidate pool (recommended
          for sorting QC — top-firing cells are systematically biased
          toward easy-to-sort high-amp units and tell you less about
          the marginal cases).
        - ``'top_rate'``: top ``n_cells_per_type`` by ``rate_col``
          descending (reproducible without a seed; useful for digging
          into the most-active cells specifically).
    random_seed : int, optional
        Seed for ``random`` strategy. ``None`` (default) draws a fresh
        sample every call. Set an integer for reproducible runs.

    Returns
    -------
    pandas.DataFrame
        At least ``cell_id, cell_type, mean_rate_hz``. Empty when
        nothing passes the filters.
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

    if 'cell_type' in pool.columns and cell_types:
        pool = pool.loc[pool['cell_type'].isin(list(cell_types))]

    if sample_strategy not in ('random', 'top_rate'):
        raise ValueError(
            f'sample_strategy must be "random" or "top_rate", got {sample_strategy!r}')

    # Sample per cell type (in the cell_types order so output is grouped).
    rng = np.random.default_rng(random_seed)
    if cell_types and 'cell_type' in pool.columns:
        keep_rows = []
        for ct in cell_types:
            sub = pool.loc[pool['cell_type'] == ct]
            if sub.empty:
                continue
            n = min(n_cells_per_type, len(sub))
            if sample_strategy == 'top_rate' and rate_col in sub.columns:
                picked = sub.sort_values(rate_col, ascending=False).head(n)
            else:
                # Random: numpy choice over the row indices for determinism
                # when random_seed is provided.
                idx = rng.choice(len(sub), size=n, replace=False)
                picked = sub.iloc[np.sort(idx)]
            keep_rows.append(picked)
        sample = pd.concat(keep_rows, ignore_index=True) if keep_rows else pool.head(0)
    else:
        n = min(n_cells_per_type, len(pool))
        if sample_strategy == 'top_rate' and rate_col in pool.columns:
            sample = pool.sort_values(rate_col, ascending=False).head(n)
        else:
            idx = rng.choice(len(pool), size=n, replace=False) if n > 0 else []
            sample = pool.iloc[np.sort(idx)] if n > 0 else pool.head(0)

    keep = [c for c in ('cell_id', 'cell_type', rate_col, 'n_epochs')
            if c in sample.columns]
    return sample[keep].reset_index(drop=True)


def _hp_filter(x: np.ndarray, fs: float, cutoff_hz: float = 300.0,
               order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth high-pass — what the sorter sees during sorting.

    Without this, the raw trace's low-frequency drift moves the apparent
    spike "height" around and the spike-time markers can look like
    they're stuck at the drift level instead of at the spike trough.
    """
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, cutoff_hz, btype='highpass', fs=fs, output='sos')
    return sosfiltfilt(sos, x).astype(np.float32)


def _plot_epoch_trace_with_raster(
    rt,
    rb,
    protocol_cell_id: int,
    epoch_idx: int,
    ax_trace: plt.Axes,
    ax_raster: plt.Axes,
    *,
    target_color: str = 'tab:red',
    hp_cutoff_hz: float = 300.0,
    spike_marker_size: int = 4,
) -> None:
    """Render one (raster + trace) row for one cell × one epoch.

    Uses the **protocol** cell_id (not the noise/chunk one) to pull both
    the spike times and the EI peak electrode — they live in
    ``rb.vcd`` (protocol VCD) and ``rb.df_spike_times`` respectively,
    keyed by protocol_cell_id. The noise_cell_id is shown in the
    raster label for clarity.

    Top axis: thin raster strip with vertical ticks at the cell's spike
    times for this epoch.  Bottom axis: 300 Hz high-pass-filtered
    trace on the cell's primary electrode with small red dots at the
    spike samples — filter removes drift so the dot heights line up
    with the spike troughs the way the sorter saw them.
    """
    from ..classes.raw import primary_electrode_of_cell

    if rt.epoch_idx != epoch_idx:
        rt.load_epoch_index(epoch_idx, verbose=False)

    electrode_idx = primary_electrode_of_cell(rb, protocol_cell_id)
    raw_ts = rt.data[electrode_idx, :].astype(np.float32)
    # High-pass filter for display so the spike marks line up with the
    # actual spike waveform (not the low-frequency baseline drift).
    try:
        trace = _hp_filter(raw_ts, fs=float(rt.sample_rate),
                            cutoff_hz=hp_cutoff_hz)
        filt_label = f'{int(hp_cutoff_hz)} Hz HP'
    except Exception:
        trace = raw_ts
        filt_label = 'raw'
    time_s = np.arange(len(trace)) / rt.sample_rate

    # Pull this cell's spike times for this epoch (ms → s). The
    # protocol_cell_id is the key in df_spike_times['cell_id'].
    df_st = rb.df_spike_times
    cell_row = df_st.index[df_st['cell_id'] == int(protocol_cell_id)]
    if len(cell_row) == 0:
        sts_ms = np.array([])
        noise_cell_id = None
    else:
        sts_ms = np.asarray(df_st.at[cell_row[0], 'spike_times'][epoch_idx])
        noise_cell_id = (int(df_st.at[cell_row[0], 'noise_id'])
                          if 'noise_id' in df_st.columns
                             and not pd.isna(df_st.at[cell_row[0], 'noise_id'])
                          else None)
    sts_s = sts_ms / 1000.0

    # --- raster strip (top): thin row of vertical lines per spike
    ax_raster.vlines(sts_s, 0.05, 0.95, color=target_color, lw=0.6)
    ax_raster.set_xlim(time_s[0], time_s[-1])
    ax_raster.set_ylim(0, 1)
    ax_raster.set_yticks([])
    ax_raster.set_xticks([])
    for spine in ('top', 'right', 'left', 'bottom'):
        ax_raster.spines[spine].set_visible(False)
    n_spikes = int(sts_s.size)
    noise_label = f', noise#{noise_cell_id}' if noise_cell_id is not None else ''
    ax_raster.set_ylabel(
        f'proto#{protocol_cell_id}{noise_label}\n({n_spikes} sp)',
        fontsize=7, rotation=0, ha='right', va='center',
    )

    # --- raw trace (bottom): high-pass filtered for display
    ax_trace.plot(time_s, trace, color='0.25', lw=0.4, zorder=1)
    # Mark spikes on the trace. The "spike time" reported by Vision/Kilosort
    # may be the threshold-crossing or the template-peak sample, which can
    # be a couple of samples off from the local trough of the raw waveform.
    # Snap each marker to the local minimum in ±2 ms so the dots sit on
    # the visible spike, not slightly above it.
    if sts_s.size:
        sample_idx = np.round(sts_s * rt.sample_rate).astype(int)
        sample_idx = sample_idx[(sample_idx >= 0) & (sample_idx < trace.size)]
        snap_w = int(0.002 * rt.sample_rate)  # ±2 ms
        snapped = []
        for s in sample_idx:
            lo = max(0, s - snap_w)
            hi = min(trace.size, s + snap_w + 1)
            snapped.append(lo + int(np.argmin(trace[lo:hi])))
        snapped = np.asarray(snapped, dtype=int)
        ax_trace.scatter(snapped / rt.sample_rate, trace[snapped],
                         color=target_color, s=spike_marker_size, zorder=3,
                         linewidths=0)
    ax_trace.set_xlim(time_s[0], time_s[-1])
    ax_trace.set_ylabel(f'{filt_label}', fontsize=8)
    ax_trace.set_title(
        f'epoch {epoch_idx} · electrode {electrode_idx}',
        fontsize=8, loc='left', pad=2,
    )
    ax_trace.tick_params(axis='both', labelsize=7)


def sample_and_plot_sorting_qc(
    response_block,
    *,
    exp_name: Optional[str] = None,
    protocol_subdir: Optional[str] = None,
    append_datafile_to_subdir: bool = False,
    output_root: Optional[str] = None,
    cell_types: Sequence[str] = ('OnP', 'OnM'),
    n_cells_per_type: int = 3,
    n_epochs: int = 4,
    require_visual_qc_good: bool = True,
    sample_strategy: str = 'random',
    random_seed: Optional[int] = None,
    save_dir: Optional[str] = None,
    dpi: int = 250,
    overwrite: bool = True,
    figsize_per_epoch: Tuple[float, float] = (12.0, 1.7),
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[Path]]:
    """Sample top-firing cells per type and write full-epoch sorting-QC PNGs.

    One **figure per cell**; inside the figure, each requested epoch
    is a full-width row containing two stacked panels (sharing the
    epoch's time axis):

    - **Raster strip** (thin): vertical ticks at every spike assigned
      to the cell for that epoch.
    - **Raw trace**: voltage on the cell's primary electrode for the
      whole epoch, with red dots at the spike samples — so you can
      eyeball whether each assigned spike has a real waveform.

    Saved as high-DPI PNGs in
    ``<OUTPUT>/<exp>/<protocol_subdir>/sorting_qc/`` (one per cell).

    Parameters
    ----------
    cell_types : sequence[str]
        Cell types to sample (default ``('OnP', 'OnM')``).
    n_cells_per_type : int
        Number of cells to sample per type.
    n_epochs : int
        Number of epochs to plot per cell (always the first
        ``n_epochs`` epochs). Each epoch is one row in the figure.
    sample_strategy : ``'random'`` (default) or ``'top_rate'``
        How to pick cells from the candidate pool. Random avoids the
        bias toward easy-to-sort high-amplitude units; ``top_rate``
        gives a reproducible "best firers" subset.
    random_seed : int, optional
        Seed for random sampling. ``None`` = fresh sample each call.
    save_dir : str, optional
        Override the default save directory.
    dpi : int
        PNG resolution. 200-300 is appropriate for visual inspection.
    overwrite : bool
        Re-render existing PNGs (default True).
    figsize_per_epoch : (w, h)
        Per-row figure size in inches.

    Returns
    -------
    (sample_df, png_paths)
        DataFrame of cells chosen + list of PNG paths written.
    """
    from ..classes.raw import RawTraces

    # 1. Pick cells
    sample_df = sample_sorting_qc_cells(
        response_block,
        exp_name=exp_name, protocol_subdir=protocol_subdir,
        append_datafile_to_subdir=append_datafile_to_subdir,
        output_root=output_root,
        cell_types=cell_types,
        n_cells_per_type=n_cells_per_type,
        require_visual_qc_good=require_visual_qc_good,
        sample_strategy=sample_strategy,
        random_seed=random_seed,
    )
    if verbose:
        print(f'sampled {len(sample_df)} cells:')
        if not sample_df.empty:
            print(sample_df.to_string(index=False))
    if sample_df.empty:
        return sample_df, []

    # 2. Resolve save directory. The folder name *always* stamps the
    # protocol short name + datafile so PNGs from different datafiles on
    # the same date can't collide — independent of whatever
    # protocol_subdir / append_datafile_to_subdir §17 used for qc.csv.
    exp = exp_name or response_block.exp_name
    short = _resolve_protocol_subdir(
        response_block, protocol_subdir,
        getattr(response_block, 'datafile_name', None),
        append_datafile_to_subdir,
    )
    from .cell_plot_archive import protocol_short_name as _ps
    proto_short = _ps(response_block.protocol_name)
    datafile = getattr(response_block, 'datafile_name', None) or 'unknown'
    qc_folder_name = f'sorting_qc_{proto_short}_{datafile}'

    root = Path(output_root) if output_root else Path(OUTPUT_DIR)
    save_root = (Path(save_dir) if save_dir is not None
                 else root / exp / short / qc_folder_name)
    save_root.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f'save dir: {save_root}')

    # 3. Build the raw-trace loader (caches one epoch's worth at a time).
    rt = RawTraces(response_block)

    n_ep = min(n_epochs, response_block.n_epochs)
    epoch_idxs = list(range(n_ep))

    # Protocol cell IDs (what qc.csv stores and what we use throughout).
    cell_ids = sample_df['cell_id'].astype(int).tolist()
    cell_type_list = (sample_df['cell_type'].tolist()
                      if 'cell_type' in sample_df.columns
                      else [''] * len(cell_ids))

    # Look up the matched noise/chunk cell_id for each protocol cell so the
    # title can show both — the user is allowed to confuse them otherwise.
    df_st = response_block.df_spike_times
    noise_id_lookup = {}
    if 'noise_id' in df_st.columns:
        for _, r in df_st.iterrows():
            try:
                noise_id_lookup[int(r['cell_id'])] = int(r['noise_id'])
            except (ValueError, TypeError):
                pass

    png_paths: List[Path] = []

    # Plot. Outer loop on cells; per cell we re-load each epoch's raw
    # data once and reuse it across the (raster + trace) pair.
    for proto_cid, ctype in zip(cell_ids, cell_type_list):
        noise_cid = noise_id_lookup.get(int(proto_cid))
        noise_tag = f'_noise{noise_cid}' if noise_cid is not None else ''
        png_path = save_root / f'cell_proto{proto_cid:04d}{noise_tag}_{ctype}_sorting_qc.png'
        if png_path.exists() and not overwrite:
            if verbose:
                print(f'  skip (exists): {png_path}')
            png_paths.append(png_path)
            continue

        # Per cell: n_epochs rows × 2 sub-rows (raster strip + trace).
        fig = plt.figure(
            figsize=(figsize_per_epoch[0], figsize_per_epoch[1] * n_ep),
        )
        gs = fig.add_gridspec(
            n_ep * 2, 1,
            height_ratios=[0.18, 1.0] * n_ep,
            hspace=0.32,
        )
        for j, ep in enumerate(epoch_idxs):
            ax_raster = fig.add_subplot(gs[j * 2])
            ax_trace = fig.add_subplot(gs[j * 2 + 1], sharex=ax_raster)
            try:
                _plot_epoch_trace_with_raster(
                    rt, response_block, proto_cid, ep,
                    ax_trace=ax_trace, ax_raster=ax_raster,
                )
            except Exception as exc:
                ax_trace.set_title(
                    f'proto#{proto_cid}, epoch {ep}: {exc!r}', fontsize=7)
                ax_trace.axis('off')
                ax_raster.axis('off')
            if j == n_ep - 1:
                ax_trace.set_xlabel('time (s)', fontsize=9)

        noise_text = f'  ·  noise#{noise_cid}' if noise_cid is not None else ''
        fig.suptitle(
            f'{exp} / {getattr(response_block, "datafile_name", "?")}  —  '
            f'protocol#{proto_cid}{noise_text}  ({ctype}) :  sorting QC',
            fontsize=10, y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        png_paths.append(png_path)
        if verbose:
            print(f'  wrote: {png_path}')

    return sample_df, png_paths
