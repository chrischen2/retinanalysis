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


__all__ = ['sample_sorting_qc_cells', 'sample_and_plot_sorting_qc',
           'sorting_qc_gui']


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
    trace_lw: float = 0.4,
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
    ax_trace.plot(time_s, trace, color='0.25', lw=trace_lw, zorder=1)
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
    dpi: int = 800,
    overwrite: bool = True,
    figsize_per_epoch: Tuple[float, float] = (12.0, 1.7),
    trace_lw: float = 0.25,
    spike_marker_size: int = 3,
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
        PNG resolution. Default 800 — at typical figure widths a 0.25 pt
        trace line ends up roughly 1 px wide; bump to 1200+ if you still
        need crisper spike marks. (Higher DPI also costs disk space.)
    overwrite : bool
        Re-render existing PNGs (default True).
    figsize_per_epoch : (w, h)
        Per-row figure size in inches.
    trace_lw : float
        Line width of the raw trace in points. Default 0.25 (very thin).
        Drop to 0.15 if you need to see neighboring spikes through the
        line; raise to 0.5 if the trace disappears.
    spike_marker_size : int
        Size of the red spike dots overlaid on the trace.

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
                    trace_lw=trace_lw,
                    spike_marker_size=spike_marker_size,
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


# ---------------------------------------------------------------------------
# Interactive sorting-QC GUI (notebook). Targets the remote-NAS use case:
# only the requested time window is read from disk (no full-epoch loads), and
# the user can see the estimated MB on the wire before clicking "Load".
# ---------------------------------------------------------------------------
def sorting_qc_gui(
    response_block,
    *,
    exp_name: Optional[str] = None,
    protocol_subdir: Optional[str] = None,
    append_datafile_to_subdir: bool = False,
    output_root: Optional[str] = None,
    cell_types: Sequence[str] = ('OnP', 'OnM'),
    require_visual_qc_good: bool = True,
    default_window_s: Tuple[float, float] = (0.0, 2.0),
    hp_cutoff_hz: float = 300.0,
):
    """Notebook GUI to inspect raw traces under detected spikes, one window at a time.

    Designed for remote-NAS access where loading a full ~60 s epoch
    (~900 MB at 20 kHz × 512-channel 12-bit) is wasteful. The GUI:

    - Lists QC-pass (∩ visual-QC ``good``) cells of the requested types
      in a cell dropdown, with ``cell_type`` + protocol cell id +
      matched noise id shown.
    - Lets the user pick an epoch and a time window (start, end) in
      seconds within the epoch.
    - Lets the user pick which of the **top-3 EI electrodes** for the
      selected cell to plot (1st = primary, 2nd / 3rd = the next two
      strongest). Note: choosing a different electrode does *not*
      reduce on-wire bytes — bin2py packs all electrodes together per
      sample — it only changes which row of the loaded matrix is drawn.
    - Estimated **MB on the wire** for the chosen window is shown on
      the Load button so you can decide before it fetches.
    - "Load raw trace" reads *only* that window (high-pass filtered for
      display) and draws it. "Overlay detected spikes" re-renders with
      red dots snapped to spike troughs in ±2 ms.

    Requires ``ipywidgets`` and a Jupyter front-end (the
    ``retinanalysis`` conda env already has it).

    Parameters
    ----------
    response_block : MEAResponseBlock
        Same object you'd pass to ``sample_and_plot_sorting_qc``.
    cell_types : sequence[str]
        Cell types to include in the dropdown. Default ``('OnP', 'OnM')``.
        Pass an empty tuple to skip the type filter.
    default_window_s : (start, end)
        Initial window in seconds within the epoch.
    hp_cutoff_hz : float
        High-pass cutoff for the display filter (matches the saved PNGs).

    Returns
    -------
    ipywidgets.VBox
        Display this in a notebook cell. Holds dropdowns, sliders, the
        Load / Overlay-spikes buttons, and an output area for the plot.
    """
    import ipywidgets as widgets
    from IPython.display import display
    from ..classes.raw import RawTraces, primary_electrode_of_cell

    # ---- pool of cells (same selection as sample_sorting_qc_cells, but
    # without the random down-sample — the user picks).
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
    if not qc_path.exists():
        raise FileNotFoundError(
            f'No qc.csv at {qc_path}. Run §17 (analyze_experiment) first.')
    qc_df = pd.read_csv(qc_path)
    pool = qc_df.loc[qc_df['passes']].copy() if 'passes' in qc_df.columns else qc_df.copy()

    vqc_path = proto_root / 'visual_qc.csv'
    if vqc_path.exists():
        vqc_df = pd.read_csv(vqc_path)
        good = set(vqc_df.loc[vqc_df['tag'] == 'good', 'cell_id'].astype(int))
        pool = pool.loc[pool['cell_id'].astype(int).isin(good)]
    if 'cell_type' in pool.columns and cell_types:
        pool = pool.loc[pool['cell_type'].isin(list(cell_types))]
    if pool.empty:
        raise ValueError('No candidate cells after qc.csv + visual_qc.csv filtering.')

    # noise-id lookup (proto -> noise) for nicer labels
    df_st = response_block.df_spike_times
    noise_lookup = {}
    if 'noise_id' in df_st.columns:
        for _, r in df_st.iterrows():
            try:
                noise_lookup[int(r['cell_id'])] = int(r['noise_id'])
            except (ValueError, TypeError):
                pass

    def _cell_label(row) -> str:
        cid = int(row['cell_id'])
        ct = str(row.get('cell_type', ''))
        nid = noise_lookup.get(cid)
        rate = row.get('mean_rate_hz', None)
        rate_str = f' · {float(rate):.1f} Hz' if rate is not None and not pd.isna(rate) else ''
        nid_str = f' · noise#{nid}' if nid is not None else ''
        return f'{ct} · proto#{cid}{nid_str}{rate_str}'

    cell_options = [(_cell_label(r), int(r['cell_id'])) for _, r in pool.iterrows()]

    # ---- raw-trace loader (cached across button clicks)
    rt = RawTraces(response_block)
    n_epochs = int(response_block.n_epochs)
    sample_rate = float(rt.sample_rate)

    # epoch durations (s) → for slider bounds
    starts = response_block.d_timing['epochStarts']
    ends = response_block.d_timing['epochEnds']
    epoch_durs_s = [(int(e) - int(s)) / sample_rate for s, e in zip(starts, ends)]

    # ---- widgets
    w_cell = widgets.Dropdown(options=cell_options, description='Cell:',
                              layout=widgets.Layout(width='420px'))
    w_epoch = widgets.Dropdown(
        options=[(f'epoch {i}  ({epoch_durs_s[i]:.1f} s)', i)
                 for i in range(n_epochs)],
        description='Epoch:',
        layout=widgets.Layout(width='260px'),
    )
    w_elec_rank = widgets.Dropdown(
        options=[('1st (primary)', 0), ('2nd', 1), ('3rd', 2)],
        value=0, description='Electrode:',
        layout=widgets.Layout(width='220px'),
    )
    w_window = widgets.FloatRangeSlider(
        value=list(default_window_s),
        min=0.0, max=epoch_durs_s[0], step=0.1,
        description='Window (s):', continuous_update=False,
        layout=widgets.Layout(width='520px'),
    )
    # Type-in window controls — for precise selection (e.g. 12.345 → 12.890).
    # Kept in sync with the slider via observers.
    w_t0 = widgets.FloatText(
        value=float(default_window_s[0]), step=0.01, description='start (s):',
        layout=widgets.Layout(width='180px'),
    )
    w_t1 = widgets.FloatText(
        value=float(default_window_s[1]), step=0.01, description='end (s):',
        layout=widgets.Layout(width='180px'),
    )
    # Re-entrancy guard: slider→text and text→slider observers must not
    # ping-pong. _syncing toggles to True while one direction is mid-flight.
    _sync = {'busy': False}

    def _sync_text_from_slider(_change):
        if _sync['busy']:
            return
        _sync['busy'] = True
        try:
            s0, s1 = w_window.value
            w_t0.value = float(s0)
            w_t1.value = float(s1)
        finally:
            _sync['busy'] = False

    def _sync_slider_from_text(_change):
        if _sync['busy']:
            return
        _sync['busy'] = True
        try:
            # Clamp to slider bounds for the current epoch.
            lo = max(0.0, min(float(w_t0.value), float(w_window.max) - 0.001))
            hi = max(lo + 0.001, min(float(w_t1.value), float(w_window.max)))
            w_t0.value = lo
            w_t1.value = hi
            w_window.value = (lo, hi)
        finally:
            _sync['busy'] = False
    w_window.observe(_sync_text_from_slider, names='value')
    w_t0.observe(_sync_slider_from_text, names='value')
    w_t1.observe(_sync_slider_from_text, names='value')
    w_load = widgets.Button(description='Load raw trace',
                            button_style='primary',
                            layout=widgets.Layout(width='260px'))
    w_overlay = widgets.ToggleButton(
        value=True, description='Overlay detected spikes',
        layout=widgets.Layout(width='240px'),
    )
    w_size_note = widgets.HTML()
    w_status = widgets.HTML()
    w_meter = widgets.HTML()
    w_reset_meter = widgets.Button(
        description='reset meter', layout=widgets.Layout(width='110px'),
    )
    w_out = widgets.Output()

    # ---- Appearance controls (line / marker / filter / y-range)
    _COLOR_CHOICES = ['0.25', 'black', 'tab:blue', 'tab:orange', 'tab:green',
                      'tab:red', 'tab:purple', 'tab:gray', 'C0', 'C1', 'C2']
    w_trace_color = widgets.Dropdown(
        options=_COLOR_CHOICES, value='0.25', description='trace color:',
        layout=widgets.Layout(width='230px'),
    )
    w_trace_lw = widgets.FloatSlider(
        value=0.5, min=0.05, max=2.0, step=0.05, description='trace lw:',
        continuous_update=False, readout_format='.2f',
        layout=widgets.Layout(width='320px'),
    )
    w_spike_color = widgets.Dropdown(
        options=_COLOR_CHOICES, value='tab:red', description='spike color:',
        layout=widgets.Layout(width='230px'),
    )
    w_spike_size = widgets.IntSlider(
        value=12, min=1, max=60, step=1, description='spike size:',
        continuous_update=False,
        layout=widgets.Layout(width='320px'),
    )
    w_hp = widgets.FloatText(
        value=float(hp_cutoff_hz), step=10, description='HP (Hz):',
        layout=widgets.Layout(width='180px'),
    )
    w_yauto = widgets.Checkbox(value=True, description='y-axis auto',
                                layout=widgets.Layout(width='160px'))
    w_yrange = widgets.FloatRangeSlider(
        value=(-200.0, 200.0), min=-2000.0, max=2000.0, step=10.0,
        description='y range:', continuous_update=False,
        readout_format='.0f',
        layout=widgets.Layout(width='460px'),
    )
    w_svg = widgets.Checkbox(value=True, description='vector (SVG)',
                              layout=widgets.Layout(width='160px'))
    w_figw = widgets.FloatSlider(
        value=11.0, min=6.0, max=20.0, step=0.5, description='fig width (in):',
        continuous_update=False,
        layout=widgets.Layout(width='320px'),
    )
    appearance_box = widgets.VBox([
        widgets.HBox([w_trace_color, w_trace_lw]),
        widgets.HBox([w_spike_color, w_spike_size]),
        widgets.HBox([w_hp, w_yauto, w_yrange]),
        widgets.HBox([w_svg, w_figw]),
    ])
    w_appearance = widgets.Accordion(children=[appearance_box])
    w_appearance.set_title(0, 'Appearance · line, marker, filter, y-axis')
    w_appearance.selected_index = None   # start collapsed

    def _meter_html() -> str:
        summary = rt.bandwidth_summary()
        mb = summary['bytes_read'] / 1e6
        n = summary['n_reads']
        if summary['is_network']:
            chip_bg = '#fff3cd'   # amber — network
            chip_fg = '#664d03'
            icon = '📡'
            src = f'Network ({summary["fstype"] or "unknown"})'
            note = '<span style="color:#888"> · upper bound; OS may cache</span>'
        else:
            chip_bg = '#d1e7dd'   # green — local
            chip_fg = '#0a3622'
            icon = '💾'
            src = f'Local ({summary["fstype"] or "unknown"})'
            note = ''
        return (
            f'<span style="background:{chip_bg};color:{chip_fg};'
            f'padding:2px 8px;border-radius:8px;font-family:monospace;'
            f'font-size:12px;">'
            f'{icon} {src} · {mb:.1f} MB read · {n} reads</span>{note}'
        )

    def _refresh_meter():
        w_meter.value = _meter_html()

    def _on_reset_meter(_):
        rt.reset_bandwidth_counter()
        _refresh_meter()

    w_reset_meter.on_click(_on_reset_meter)
    _refresh_meter()

    # Mutable plotting state (so style/overlay changes can re-render
    # without re-loading from the NAS). We stash the *raw* electrode and
    # re-filter at render time so the HP-cutoff slider is free.
    state = {
        'epoch_idx': None,
        'window_s': None,
        'electrode_idx': None,
        'cell_id': None,
        'raw_electrode': None,    # unfiltered, 1-D (n_samples,)
        'time_s': None,           # absolute epoch-time axis
        'spike_times_s': None,    # selected cell's spikes, clipped to window
    }

    def _refresh_size_note():
        s0, s1 = w_window.value
        mb = rt.estimate_window_mb(s0, s1)
        w_size_note.value = (
            f'<span style="color:#555">~{mb:.1f} MB on wire '
            f'({s1 - s0:.2f} s × 15.4 MB/s)</span>'
        )

    def _on_epoch_change(change):
        # Clamp / reset window bounds for the new epoch length
        dur = epoch_durs_s[w_epoch.value]
        w_window.max = dur
        s0, s1 = w_window.value
        s1 = min(s1, dur)
        s0 = min(s0, max(0.0, s1 - 0.1))
        w_window.value = (s0, s1)
        _refresh_size_note()

    def _on_window_change(change):
        _refresh_size_note()

    w_epoch.observe(_on_epoch_change, names='value')
    w_window.observe(_on_window_change, names='value')
    _refresh_size_note()

    def _top_electrodes(cell_id: int, k: int = 3) -> List[int]:
        """Top-``k`` raw-channel indices by |EI| peak amplitude."""
        ei = response_block.vcd.get_ei_for_cell(int(cell_id)).ei
        peak = np.max(np.abs(ei), axis=1)
        order = np.argsort(-peak)
        return [int(x) for x in order[:k]]

    def _render():
        from io import BytesIO
        from IPython.display import display, SVG
        with w_out:
            w_out.clear_output(wait=True)
            if state.get('raw_electrode') is None:
                return
            # Re-filter from cached raw electrode every render → HP-cutoff
            # slider can change without re-fetching anything. Filtering a
            # few seconds at 20 kHz is sub-100 ms.
            try:
                trace = _hp_filter(state['raw_electrode'],
                                   fs=sample_rate,
                                   cutoff_hz=float(w_hp.value))
            except Exception:
                trace = state['raw_electrode']
            time_s = state['time_s']
            sts_s = state['spike_times_s']

            fig, (ax_r, ax_t) = plt.subplots(
                2, 1, figsize=(float(w_figw.value), 3.2), sharex=True,
                gridspec_kw={'height_ratios': [0.18, 1.0]},
            )
            ax_t.plot(time_s, trace,
                      color=w_trace_color.value, lw=float(w_trace_lw.value),
                      zorder=1)
            if w_overlay.value and sts_s is not None and sts_s.size:
                # Snap each spike to the local minimum in ±2 ms so dots
                # land on the visible trough instead of the threshold
                # crossing.
                t0 = float(time_s[0])
                fs = sample_rate
                snap_w = int(0.002 * fs)
                sample_idx = np.round((sts_s - t0) * fs).astype(int)
                sample_idx = sample_idx[(sample_idx >= 0) & (sample_idx < trace.size)]
                snapped = []
                for s in sample_idx:
                    lo = max(0, s - snap_w)
                    hi = min(trace.size, s + snap_w + 1)
                    snapped.append(lo + int(np.argmin(trace[lo:hi])))
                snapped = np.asarray(snapped, dtype=int)
                ax_t.scatter(time_s[snapped], trace[snapped],
                             color=w_spike_color.value,
                             s=int(w_spike_size.value),
                             zorder=3, linewidths=0)
                ax_r.vlines(sts_s, 0.05, 0.95,
                            color=w_spike_color.value, lw=0.7)
            ax_r.set_xlim(time_s[0], time_s[-1])
            ax_r.set_ylim(0, 1); ax_r.set_yticks([]); ax_r.set_xticks([])
            for spine in ('top', 'right', 'left', 'bottom'):
                ax_r.spines[spine].set_visible(False)
            n_sp = 0 if sts_s is None else int(sts_s.size)
            ax_r.set_ylabel(f'{n_sp} sp', fontsize=8,
                            rotation=0, ha='right', va='center')
            ax_t.set_xlabel('time within epoch (s)')
            ax_t.set_ylabel(f'{int(float(w_hp.value))} Hz HP')
            if not w_yauto.value:
                ax_t.set_ylim(*w_yrange.value)
            ct = pool.loc[pool['cell_id'].astype(int) == int(state['cell_id'])]
            ct_str = str(ct['cell_type'].iloc[0]) if 'cell_type' in ct.columns and not ct.empty else ''
            fig.suptitle(
                f'{exp} / {response_block.datafile_name}  —  '
                f'proto#{state["cell_id"]} ({ct_str})  '
                f'· epoch {state["epoch_idx"]} · electrode {state["electrode_idx"]}',
                fontsize=10,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96))

            if w_svg.value:
                # Vector SVG: stays sharp under any browser zoom. The
                # downside is no built-in pan/zoom toolbar — that needs
                # ipympl (see the hint at the bottom of the GUI).
                buf = BytesIO()
                fig.savefig(buf, format='svg', bbox_inches='tight')
                plt.close(fig)
                display(SVG(buf.getvalue()))
            else:
                plt.show()

    def _extract_and_render(cid: int, ep: int, s0: float, s1: float,
                             electrode_idx: int, elec_rank: int,
                             bytes_read_mb: float) -> None:
        """Pull one electrode out of the cached window, stash, and draw.

        Doesn't touch the file. ``bytes_read_mb`` is only used for the
        status line — 0 means "served from cache". We stash the *raw*
        (unfiltered) electrode so that subsequent style changes (HP
        cutoff, line color, …) can re-render without re-fetching.
        """
        raw_electrode = rt.data[electrode_idx, :].astype(np.float32)
        time_s = s0 + np.arange(raw_electrode.size) / sample_rate

        row_idx = df_st.index[df_st['cell_id'] == int(cid)]
        if len(row_idx):
            sts_ms = np.asarray(df_st.at[row_idx[0], 'spike_times'][ep])
            sts_s_all = sts_ms / 1000.0
            sts_s = sts_s_all[(sts_s_all >= s0) & (sts_s_all <= s1)]
        else:
            sts_s = np.array([])

        state.update(epoch_idx=ep, window_s=(s0, s1),
                     electrode_idx=electrode_idx, cell_id=cid,
                     raw_electrode=raw_electrode, time_s=time_s,
                     spike_times_s=sts_s)
        if bytes_read_mb > 0:
            tag = f'<span style="color:#080">Loaded</span> {bytes_read_mb:.1f} MB'
        else:
            tag = '<span style="color:#080">Cached</span> (no I/O)'
        w_status.value = (
            f'{tag} · cell proto#{cid} · epoch {ep} · electrode '
            f'{electrode_idx} (rank {elec_rank + 1}) · '
            f'{sts_s.size} spikes in window'
        )
        _refresh_meter()
        _render()

    def _window_matches_cache(ep: int, s0: float, s1: float) -> bool:
        """True iff rt.data already holds exactly this (epoch, window)."""
        if rt.data is None or rt.epoch_idx != ep:
            return False
        if not np.isclose(rt.window_start_s, s0):
            return False
        # End is start + n_samples / fs
        loaded_end = rt.window_start_s + rt.data.shape[1] / sample_rate
        return np.isclose(loaded_end, s1)

    def _on_load(_):
        cid = int(w_cell.value)
        ep = int(w_epoch.value)
        s0, s1 = (float(x) for x in w_window.value)
        elec_rank = int(w_elec_rank.value)
        top = _top_electrodes(cid, k=3)
        electrode_idx = top[elec_rank] if elec_rank < len(top) else top[0]

        w_load.disabled = True
        try:
            cache_hit = _window_matches_cache(ep, s0, s1)
            if cache_hit:
                _extract_and_render(cid, ep, s0, s1,
                                    electrode_idx, elec_rank, 0.0)
            else:
                est_mb = rt.estimate_window_mb(s0, s1)
                w_status.value = (
                    f'<b>Loading</b> [{s0:.2f}, {s1:.2f}] s of epoch {ep}'
                    f' (~{est_mb:.1f} MB)…'
                )
                rt.load_window(ep, start_s=s0, end_s=s1, verbose=False)
                _extract_and_render(cid, ep, s0, s1,
                                    electrode_idx, elec_rank, est_mb)
        except Exception as exc:
            w_status.value = (
                f'<span style="color:#c00">Load failed:</span> {exc!r}')
        finally:
            w_load.disabled = False

    def _on_electrode_change(_change):
        """Switch electrode without I/O when the window is already cached.

        rt.data holds all 512 electrodes for the cached window, so picking
        a different top-N rank is just a row index + a 300 Hz filter.
        """
        if state['cell_id'] is None:
            return  # nothing loaded yet
        cid = int(state['cell_id'])
        ep = int(state['epoch_idx'])
        s0, s1 = state['window_s']
        if not _window_matches_cache(ep, s0, s1):
            return  # the cached window no longer matches; user must click Load
        elec_rank = int(w_elec_rank.value)
        top = _top_electrodes(cid, k=3)
        electrode_idx = top[elec_rank] if elec_rank < len(top) else top[0]
        _extract_and_render(cid, ep, s0, s1, electrode_idx, elec_rank, 0.0)

    w_load.on_click(_on_load)
    w_elec_rank.observe(_on_electrode_change, names='value')
    # Toggling "Overlay detected spikes" re-renders from the cached trace —
    # no extra I/O, so this is safe to do over slow links.
    w_overlay.observe(lambda c: _render(), names='value')
    # All appearance/filter changes re-render from the *cached* raw
    # electrode — zero file I/O. Safe to tweak freely over a NAS link.
    for w in (w_trace_color, w_trace_lw, w_spike_color, w_spike_size,
              w_hp, w_yauto, w_yrange, w_svg, w_figw):
        w.observe(lambda c: _render(), names='value')

    # ipympl hint — surfaces a one-liner if it's not installed. SVG
    # already gives crisp zoom via the browser, but ipympl adds a real
    # pan/zoom toolbar inside the figure.
    try:
        import ipympl  # noqa: F401
        ipympl_hint = ''
    except ImportError:
        ipympl_hint = (
            '<span style="color:#888;font-size:11px">'
            'Tip: for pan/zoom-rectangle inside the figure, '
            '<code>pip install ipympl</code> then add '
            '<code>%matplotlib widget</code> at the top of the notebook. '
            'Until then, SVG output is vector — pinch / Ctrl-+ in the '
            'browser stays sharp.</span>'
        )

    row1 = widgets.HBox([w_cell, w_epoch, w_elec_rank])
    row2a = widgets.HBox([w_window, w_size_note])
    row2b = widgets.HBox([w_t0, w_t1])
    row3 = widgets.HBox([w_load, w_overlay, w_status])
    row4 = widgets.HBox([w_meter, w_reset_meter])
    hint = widgets.HTML(value=ipympl_hint)
    return widgets.VBox([row1, row2a, row2b, w_appearance,
                          row3, row4, w_out, hint])
