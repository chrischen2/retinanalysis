"""Per-cell, per-protocol firing-rate quality control.

A given cell can pass classification on a noise chunk yet behave badly
inside a downstream protocol — drifting away, dropping out for runs of
trials, or just barely firing. These metrics quantify *trial-to-trial
reliability* of a cell's response inside a single protocol block, so we
can filter out the unreliable ones before any group-level analysis.

Input shape mirrors what's already in the codebase:
``response_block.df_spike_times`` has one row per (cell_id, cell_type) with
a ``spike_times`` column whose value is a list of length ``n_epochs``,
each entry a 1-D array of spike times in **milliseconds** relative to
epoch onset (see ``utils/raster.py``).

Metrics emitted per cell (all in spikes/trial units unless noted):

- ``mean_count``: average spike count per epoch within the analysis window.
- ``cv_count``:   std / mean of per-epoch counts. CV << 1 ≈ Poisson-or-tighter.
- ``fano``:       var / mean of per-epoch counts (1 ≈ Poisson, >> 1 ≈ bursty).
  **Caveat**: scales with mean count, so threshold-based filtering on Fano
  doesn't transfer across protocols. A reliable cell firing 300 spikes in
  a 30 s trial can have Fano ≈ 40-100. Use CV for cross-protocol filtering;
  reported here mainly as a diagnostic.
- ``silent_trial_frac``: fraction of epochs with zero spikes.
- ``silent_run_max``: longest *consecutive* run of zero-spike epochs (catches
  cells that drop out halfway through and never come back — Fano alone can
  miss this when the silent block is short relative to total epochs).
- ``drift_score``: |slope| of (spike count) vs (trial index) normalized by
  mean count; large = systematic disappearance / increase across trials.
- ``reliability_r``: split-half Pearson r between mean PSTH on
  even-numbered vs odd-numbered trials. 1 = perfectly reproducible.

``filter_cells_by_qc`` returns a boolean ``passes`` column for the default
thresholds; callers can override any threshold per-protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .psth import epoch_spikes_to_psth
from ..config.settings import OUTPUT_DIR


__all__ = [
    'QCThresholds',
    'per_epoch_spike_counts',
    'cell_qc_metrics',
    'block_qc_metrics',
    'filter_cells_by_qc',
    'save_protocol_qc',
    'load_protocol_qc',
    'protocol_qc_csv_path',
    'resolve_protocol_subdir',
    'load_or_compute_protocol_qc',
    'epoch_population_counts',
    'suggest_epoch_range',
    'plot_epoch_range',
    'epoch_condition_table',
]


def protocol_qc_csv_path(exp_name: str, protocol: str,
                         output_root: Optional[str] = None) -> Path:
    """Return ``<OUTPUT_DIR>/<exp>/<protocol>/qc.csv``."""
    root = Path(output_root) if output_root else Path(OUTPUT_DIR)
    return root / exp_name / protocol / 'qc.csv'


def save_protocol_qc(
    qc_df: pd.DataFrame,
    exp_name: str,
    protocol: str = 'eye_movement_alt_bg',
    output_root: Optional[str] = None,
) -> Path:
    """Persist per-cell protocol QC results to disk.

    Writes the full ``filter_cells_by_qc(...)`` DataFrame (metrics +
    ``passes`` column) so downstream tools can filter by the same
    automated criteria the archive used, without re-running the QC.

    Parameters
    ----------
    qc_df : pandas.DataFrame
        Output of :func:`filter_cells_by_qc`.
    exp_name : str
    protocol : str
        Short protocol name. Default ``'eye_movement_alt_bg'``.
    output_root : str, optional
        Override ``OUTPUT_DIR``.

    Returns
    -------
    pathlib.Path
        The CSV path written.
    """
    path = protocol_qc_csv_path(exp_name, protocol, output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = qc_df.copy()
    if 'exp_name' not in df.columns:
        df.insert(0, 'exp_name', exp_name)
    df.to_csv(path, index=False)
    return path


def load_protocol_qc(
    exp_names: Optional[List[str]] = None,
    output_root: Optional[str] = None,
    protocol: str = 'eye_movement_alt_bg',
) -> pd.DataFrame:
    """Concat per-experiment ``qc.csv`` into one DataFrame.

    Empty DataFrame when nothing is found.
    """
    root = Path(output_root) if output_root else Path(OUTPUT_DIR)
    if exp_names is None:
        if not root.is_dir():
            return pd.DataFrame()
        exp_names = [p.name for p in sorted(root.iterdir()) if p.is_dir()]
    dfs = []
    for exp in exp_names:
        path = root / exp / protocol / 'qc.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if 'exp_name' not in df.columns:
            df['exp_name'] = exp
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


@dataclass
class QCThresholds:
    """Default thresholds. Override per-call from caller / config.

    Set any threshold to ``None`` to skip that check entirely. The defaults
    here are tuned for "did this cell respond at all and stay responsive
    across trials" — they are intentionally lenient on burstiness/noise
    because some real RGC types are highly variable. Tighten per protocol
    when you have specific reliability requirements.

    ``min_reliability_r`` defaults to ``None`` because split-half PSTH
    correlation is meaningful only when every epoch presents the *same*
    stimulus condition. Protocols that alternate conditions across
    epochs (e.g. ``EyeMovementTrajectoryAlternatingBackground``) will
    give artificially low reliability and should keep this off — group
    epochs by condition first if you need a reliability check.

    **Firing-rate gate is adaptive to epoch length.** ``min_rate_hz``
    sets the per-epoch threshold via
    ``threshold = min_rate_hz × epoch_duration_s``; a cell passes the
    rate check when at least ``min_frac_epochs_above_rate`` of its
    epochs hit that threshold. This works across protocols with
    different stimulus durations without any tuning. Set
    ``min_rate_hz=None`` to disable.
    """

    # Adaptive firing-rate check — scales with epoch length.
    min_rate_hz: Optional[float] = 1.0
    min_frac_epochs_above_rate: Optional[float] = 0.8

    # "Drop silent epochs, keep if ≥ 2/3 remain." We don't actually drop
    # epochs from spike_times (that'd break per-condition PSTH plumbing
    # downstream); instead require the cell to have at least this fraction
    # of epochs with ≥1 spike.
    min_frac_non_silent_epochs: Optional[float] = 2.0 / 3.0

    # Legacy absolute-count threshold (kept for back-compat; superseded by
    # the rate-based check above). Set to ``None`` to disable.
    min_mean_count: Optional[float] = None

    max_cv: Optional[float] = 3.0                  # noise / dropout
    max_fano: Optional[float] = None               # scales with mean — off by default
    max_silent_trial_frac: Optional[float] = 0.5   # >50% trials silent → unreliable
    max_silent_run: Optional[int] = 10             # >=10 consecutive silent trials
    max_drift_score: Optional[float] = 0.1         # |slope|/mean per-trial > 10%
    min_reliability_r: Optional[float] = None      # split-half PSTH r; off by default

    def asdict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Single-cell metrics
# ---------------------------------------------------------------------------

def per_epoch_spike_counts(
    spike_times_by_epoch: Sequence[np.ndarray],
    t_start_ms: float = 0.0,
    t_end_ms: Optional[float] = None,
) -> np.ndarray:
    """Count spikes per epoch inside ``[t_start_ms, t_end_ms]``.

    ``t_end_ms=None`` keeps every spike at or after ``t_start_ms``.
    """
    out = np.empty(len(spike_times_by_epoch), dtype=float)
    for i, arr in enumerate(spike_times_by_epoch):
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            out[i] = 0.0
            continue
        mask = a >= t_start_ms
        if t_end_ms is not None:
            mask &= a <= t_end_ms
        out[i] = float(mask.sum())
    return out


def _longest_run_of_zeros(x: np.ndarray) -> int:
    """Length of the longest run of consecutive zeros in ``x``."""
    if x.size == 0:
        return 0
    best = run = 0
    for v in x:
        if v == 0:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _drift_score(counts: np.ndarray) -> float:
    """|slope| of counts vs trial-index, normalized by mean count.

    Returns 0 when there are <2 trials or mean count is zero.
    """
    n = counts.size
    if n < 2:
        return 0.0
    mean = counts.mean()
    if mean <= 0:
        return 0.0
    x = np.arange(n, dtype=float)
    # least squares slope
    x_c = x - x.mean()
    slope = float((x_c * (counts - mean)).sum() / (x_c ** 2).sum())
    return float(abs(slope) / mean)


def _split_half_reliability(
    spike_times_by_epoch: Sequence[np.ndarray],
    t_end_ms: float,
    psth_sigma_ms: float,
    sample_rate_hz: float,
    t_start_ms: float,
) -> float:
    """Pearson r between mean PSTHs on even- vs odd-indexed trials.

    Returns 0 if either half has zero variance (e.g. all-silent trials).
    """
    n = len(spike_times_by_epoch)
    if n < 2:
        return float('nan')
    psth = epoch_spikes_to_psth(
        spike_times_by_epoch, t_end_ms,
        psth_sigma_ms=psth_sigma_ms,
        sample_rate_hz=sample_rate_hz,
        t_start_ms=t_start_ms,
    )
    even = psth[::2].mean(axis=0)
    odd = psth[1::2].mean(axis=0)
    if even.std() == 0 or odd.std() == 0:
        return 0.0
    r = float(np.corrcoef(even, odd)[0, 1])
    return r if np.isfinite(r) else 0.0


def cell_qc_metrics(
    spike_times_by_epoch: Sequence[np.ndarray],
    t_start_ms: float = 0.0,
    t_end_ms: Optional[float] = None,
    psth_sigma_ms: float = 10.0,
    sample_rate_hz: float = 1000.0,
    min_rate_hz: Optional[float] = 1.0,
) -> dict:
    """Compute QC metrics for one cell's per-epoch spike-time lists.

    ``min_rate_hz`` controls the adaptive rate check. The per-epoch
    count threshold is ``min_rate_hz × epoch_duration_s``; the metric
    ``frac_epochs_above_rate`` is the fraction of epochs at or above
    that threshold. Pass ``None`` to skip the computation.

    See module docstring for metric definitions.
    """
    counts = per_epoch_spike_counts(spike_times_by_epoch, t_start_ms, t_end_ms)
    n = counts.size
    mean = float(counts.mean()) if n else 0.0
    std = float(counts.std()) if n else 0.0
    var = float(counts.var()) if n else 0.0

    silent_trial_frac = float((counts == 0).mean()) if n else float('nan')
    frac_non_silent_epochs = (1.0 - silent_trial_frac) if n else float('nan')
    n_non_silent_epochs = int((counts > 0).sum()) if n else 0
    silent_run_max = int(_longest_run_of_zeros(counts))
    drift = _drift_score(counts)

    # Adaptive rate check — work in seconds so it generalizes across
    # protocols. With no time bound we can't derive a duration, so the
    # rate-based fields fall back to NaN and the filter step skips them.
    if t_end_ms is not None:
        epoch_duration_s = max((t_end_ms - t_start_ms) / 1000.0, 0.0)
    else:
        epoch_duration_s = float('nan')
    mean_rate_hz = (mean / epoch_duration_s) if epoch_duration_s > 0 else float('nan')
    min_count_per_epoch = int(counts.min()) if n else 0
    if min_rate_hz is not None and epoch_duration_s > 0 and n:
        count_threshold = min_rate_hz * epoch_duration_s
        frac_epochs_above_rate = float((counts >= count_threshold).mean())
    else:
        count_threshold = float('nan')
        frac_epochs_above_rate = float('nan')

    # For reliability we need an upper time bound — fall back to the max
    # spike across all epochs if t_end_ms is unset.
    if t_end_ms is None:
        max_t = 0.0
        for arr in spike_times_by_epoch:
            if len(arr):
                max_t = max(max_t, float(np.asarray(arr).max()))
        t_end_for_psth = max(max_t, t_start_ms + 1.0)
    else:
        t_end_for_psth = float(t_end_ms)
    reliability = _split_half_reliability(
        spike_times_by_epoch, t_end_for_psth,
        psth_sigma_ms, sample_rate_hz, t_start_ms,
    )

    return {
        'n_epochs': n,
        'epoch_duration_s': epoch_duration_s,
        'mean_count': mean,
        'std_count': std,
        'min_count_per_epoch': min_count_per_epoch,
        'mean_rate_hz': mean_rate_hz,
        'count_threshold': count_threshold,
        'frac_epochs_above_rate': frac_epochs_above_rate,
        'cv_count': float(std / mean) if mean > 0 else float('inf'),
        'fano': float(var / mean) if mean > 0 else float('inf'),
        'silent_trial_frac': silent_trial_frac,
        'n_non_silent_epochs': n_non_silent_epochs,
        'frac_non_silent_epochs': frac_non_silent_epochs,
        'silent_run_max': silent_run_max,
        'drift_score': drift,
        'reliability_r': reliability,
    }


# ---------------------------------------------------------------------------
# Block-level convenience: run metrics over every cell in a ResponseBlock
# ---------------------------------------------------------------------------

def block_qc_metrics(
    response_block,
    t_start_ms: float = 0.0,
    t_end_ms: Optional[float] = None,
    cell_types: Optional[Iterable[str]] = None,
    psth_sigma_ms: float = 10.0,
    sample_rate_hz: float = 1000.0,
    min_rate_hz: Optional[float] = 1.0,
    epoch_range: Optional[Tuple[int, int]] = None,
    epoch_indices: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Compute QC metrics for every cell in ``response_block.df_spike_times``.

    Returns a DataFrame with one row per cell. Columns:
    ``cell_id, cell_type, n_epochs, epoch_duration_s, mean_count,
    std_count, min_count_per_epoch, mean_rate_hz, count_threshold,
    frac_epochs_above_rate, cv_count, fano, silent_trial_frac,
    silent_run_max, drift_score, reliability_r``.

    ``min_rate_hz`` controls the adaptive rate threshold (see
    :func:`cell_qc_metrics`). Optional ``cell_types`` filter intersects
    with whatever the response block has labeled. Unlabeled cells are
    kept (cell_type may be NaN).

    ``epoch_range`` is a ``(start, stop)`` half-open slice, normally the
    range :func:`suggest_epoch_range` picked. Judge cells on the epochs you
    intend to analyze: a block whose first trials are dead fails every cell
    on ``silent_run_max`` and ``drift_score`` when scored whole, which
    reports a property of the block as a property of each cell.

    ``epoch_indices`` selects an arbitrary set of epochs by index instead,
    for the case a contiguous slice cannot express: on a protocol that
    alternates conditions, the epochs of one condition are every other one.
    Scoring "does this cell fire" needs the condition the cell is *meant* to
    fire in, otherwise the gate charges a cell for going quiet when the
    stimulus told it to. Indices are into the block's full epoch list and
    take precedence over ``epoch_range``.
    """
    df = response_block.df_spike_times
    if cell_types is not None:
        want = set(cell_types)
        df = df[df['cell_type'].isin(want)]

    if t_end_ms is None and min_rate_hz is not None:
        # Without an epoch length the rate gate has nothing to divide by:
        # epoch_duration_s comes back NaN, mean_rate_hz with it, and the
        # threshold comparison is False for every cell — so the whole block
        # fails QC silently and looks like data so bad nothing survived.
        # Epoch duration is a block property, not a cell one, so infer it once
        # from the latest spike anywhere in the block. Pass t_end_ms from the
        # protocol's own preTime + stimTime + tailTime when you have it; this
        # is a floor, and it is short by however long the last epoch stayed
        # quiet after its final spike.
        latest = 0.0
        for spikes in df['spike_times']:
            for arr in spikes:
                a = np.asarray(arr, dtype=float)
                if a.size:
                    latest = max(latest, float(a.max()))
        if latest > 0:
            t_end_ms = latest
            print(f'block_qc_metrics: no t_end_ms given; inferring an epoch '
                  f'length of {latest / 1000:.1f} s from the latest spike in '
                  f'the block. Pass t_end_ms for an exact rate gate.')

    if epoch_indices is not None:
        epoch_indices = [int(i) for i in epoch_indices]

    rows = []
    for _, r in df.iterrows():
        spikes = r['spike_times']
        if epoch_indices is not None:
            spikes = [spikes[i] for i in epoch_indices if i < len(spikes)]
        elif epoch_range is not None:
            spikes = list(spikes)[epoch_range[0]:epoch_range[1]]
        m = cell_qc_metrics(
            spikes,
            t_start_ms=t_start_ms,
            t_end_ms=t_end_ms,
            psth_sigma_ms=psth_sigma_ms,
            sample_rate_hz=sample_rate_hz,
            min_rate_hz=min_rate_hz,
        )
        m['cell_id'] = int(r['cell_id'])
        m['cell_type'] = r.get('cell_type', None)
        rows.append(m)
    cols = ['cell_id', 'cell_type', 'n_epochs', 'epoch_duration_s',
            'mean_count', 'std_count', 'min_count_per_epoch',
            'mean_rate_hz', 'count_threshold', 'frac_epochs_above_rate',
            'cv_count', 'fano', 'silent_trial_frac',
            'n_non_silent_epochs', 'frac_non_silent_epochs',
            'silent_run_max', 'drift_score', 'reliability_r']
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=cols)
    return out[cols]


def resolve_protocol_subdir(
    response_block,
    *,
    protocol_subdir: Optional[str] = None,
    append_datafile_to_subdir: bool = False,
    datafile_name: Optional[str] = None,
) -> str:
    """Return the per-protocol subdir name that §4/§6/§9 all share.

    Default = protocol short name. Override via ``protocol_subdir`` or
    ``append_datafile_to_subdir=True`` (which appends ``_<datafile>`` to
    the short name; the datafile is pulled from ``response_block`` when
    not passed explicitly). Centralized here so the notebook QC cell and
    ``analyze_experiment`` resolve identical paths.
    """
    from .cell_plot_archive import protocol_short_name
    short = protocol_short_name(response_block.protocol_name)
    if protocol_subdir is not None:
        return protocol_subdir
    if append_datafile_to_subdir:
        df = datafile_name or getattr(response_block, 'datafile_name', None)
        if df:
            return f'{short}_{df}'
    return short


def load_or_compute_protocol_qc(
    response_block,
    exp_name: str,
    *,
    protocol_subdir: Optional[str] = None,
    append_datafile_to_subdir: bool = False,
    datafile_name: Optional[str] = None,
    overwrite: bool = False,
    min_rate_hz: float = 1.0,
    min_frac_epochs: float = 0.8,
    min_frac_non_silent: float = 2.0 / 3.0,
    output_root: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load ``qc.csv`` if present, otherwise compute fresh and save.

    Wraps the §4 notebook flow into a single call:

    1. Resolve the per-protocol subdir under ``<OUTPUT_DIR>/<exp>/`` so
       this matches §6/§9 exactly.
    2. If ``qc.csv`` exists at that path and ``overwrite=False``, load it.
    3. Otherwise compute :func:`block_qc_metrics` →
       :func:`filter_cells_by_qc` over the full epoch window and persist
       via :func:`save_protocol_qc`.
    4. When ``verbose=True``, print the same summary the notebook used to:
       per-cell-type pass rate, gate definitions, first failing cells.

    Returns the (loaded or computed) QC DataFrame with the ``passes``
    column.
    """
    subdir = resolve_protocol_subdir(
        response_block,
        protocol_subdir=protocol_subdir,
        append_datafile_to_subdir=append_datafile_to_subdir,
        datafile_name=datafile_name,
    )
    qc_path = protocol_qc_csv_path(exp_name, subdir, output_root=output_root)

    if qc_path.exists() and not overwrite:
        qc = pd.read_csv(qc_path)
        if verbose:
            print(f'Loaded qc.csv from {qc_path}  ({len(qc)} cells)')
            print('Pass overwrite=True to recompute with current thresholds.')
    else:
        if verbose:
            if qc_path.exists():
                print(f'overwrite=True: recomputing and overwriting {qc_path}')
            else:
                print('No qc.csv on disk yet — computing fresh.')
        t_total_ms = (
            response_block.d_timing['pre_time_ms']
            + response_block.d_timing['stim_time_ms']
            + response_block.d_timing['tail_time_ms']
        )
        qc_metrics = block_qc_metrics(
            response_block, t_start_ms=0, t_end_ms=t_total_ms,
            min_rate_hz=min_rate_hz,
        )
        qc = filter_cells_by_qc(qc_metrics, thresholds=QCThresholds(
            min_rate_hz=min_rate_hz,
            min_frac_epochs_above_rate=min_frac_epochs,
            min_frac_non_silent_epochs=min_frac_non_silent,
        ))
        saved = save_protocol_qc(qc, exp_name, protocol=subdir,
                                  output_root=output_root)
        if verbose:
            print(f'qc.csv → {saved}')

    if verbose and not qc.empty:
        epoch_s = qc['epoch_duration_s'].iloc[0]
        print(f'\nepoch window: {epoch_s:.1f} s')
        print(f'  rate gate:        ≥ {min_rate_hz:.1f} Hz × {epoch_s:.1f} s = '
              f'{min_rate_hz*epoch_s:.0f} spikes/epoch in '
              f'≥{100*min_frac_epochs:.0f}% of epochs')
        print(f'  silent-epoch gate: ≥ {100*min_frac_non_silent:.0f}% '
              f'of epochs have ≥1 spike')
        n_pass = int(qc['passes'].sum())
        print(f'\nTotal cells: {len(qc)},  Passing QC: {n_pass} '
              f'({100*n_pass/len(qc):.1f}%)')
        print('\nPass rate by cell type:')
        for ct, sub in qc.groupby('cell_type'):
            rate = sub['mean_rate_hz'].median()
            print(f'  {ct:<12}  {int(sub.passes.sum()):>4} / {len(sub):>4}  '
                  f'({100*sub.passes.mean():3.0f}%)   '
                  f'median rate: {rate:5.1f} Hz')

    return qc


def filter_cells_by_qc(
    metrics_df: pd.DataFrame,
    thresholds: Optional[QCThresholds] = None,
) -> pd.DataFrame:
    """Add a boolean ``passes`` column to a metrics DataFrame.

    A cell passes when **every** metric is within its threshold. The
    threshold object itself is returned alongside as ``metrics_df.attrs['thresholds']``
    so the choice is recoverable from a saved DataFrame.
    """
    th = thresholds or QCThresholds()
    if metrics_df.empty:
        out = metrics_df.copy()
        out['passes'] = pd.Series(dtype=bool)
        out.attrs['thresholds'] = th.asdict()
        return out

    passes = pd.Series(True, index=metrics_df.index)
    checks = [
        ('mean_count', th.min_mean_count, '>='),
        ('frac_epochs_above_rate', th.min_frac_epochs_above_rate, '>='),
        ('frac_non_silent_epochs', th.min_frac_non_silent_epochs, '>='),
        ('cv_count', th.max_cv, '<='),
        ('fano', th.max_fano, '<='),
        ('silent_trial_frac', th.max_silent_trial_frac, '<='),
        ('silent_run_max', th.max_silent_run, '<='),
        ('drift_score', th.max_drift_score, '<='),
        ('reliability_r', th.min_reliability_r, '>='),
    ]
    for col, thr, op in checks:
        if thr is None:
            continue
        if col not in metrics_df.columns:
            # Older metrics frames (pre-rate-check) may not carry the new
            # columns. Skip the check rather than crash so loading legacy
            # data still works.
            continue
        col_v = metrics_df[col]
        passes &= (col_v >= thr) if op == '>=' else (col_v <= thr)
    out = metrics_df.copy()
    out['passes'] = passes.values
    out.attrs['thresholds'] = th.asdict()
    return out


# ---------------------------------------------------------------------------
# Epoch range: where in the block is the recording usable?
# ---------------------------------------------------------------------------

def epoch_population_counts(response_block, cell_types=None, minimum_n: int = 3,
                            t_start_ms: float = 0.0,
                            t_end_ms: Optional[float] = None) -> np.ndarray:
    """Total spikes per epoch, summed over every cell of the wanted types.

    A population number rather than a per-cell one on purpose: one cell going
    quiet is a cell-QC question, while the whole population going quiet is an
    epoch question, and these are the epochs.
    """
    df = response_block.df_spike_times
    if 'cell_type' not in df.columns:
        response_block.add_cell_types()
        df = response_block.df_spike_times

    if cell_types is not None:
        df = df[df['cell_type'].isin(list(cell_types))]
        keep = [ct for ct, rows in df.groupby('cell_type') if len(rows) >= minimum_n]
        df = df[df['cell_type'].isin(keep)]

    if df.empty:
        return np.zeros(0)

    n_epochs = max(len(s) for s in df['spike_times'])
    totals = np.zeros(n_epochs)
    for spikes in df['spike_times']:
        counts = per_epoch_spike_counts(spikes, t_start_ms, t_end_ms)
        totals[:len(counts)] += counts[:n_epochs]
    return totals


def suggest_epoch_range(response_block, cell_types=None, minimum_n: int = 3,
                        condition_values=None, min_fraction: float = 0.5,
                        t_start_ms: float = 0.0,
                        t_end_ms: Optional[float] = None) -> dict:
    """Longest run of consecutive epochs where the population fires normally.

    A block often starts before the retina has settled or ends after it has
    given up, and those epochs are not data. This finds the usable middle: it
    scores each epoch by total population spikes, normalizes, and returns the
    longest contiguous run at or above ``min_fraction`` of normal.

    **Normalize within condition, or the experiment looks like dropout.**
    Pass ``condition_values`` — one value per epoch, from the parameter that
    alternates — and each epoch is divided by the median of the epochs sharing
    its condition. Without that, a protocol alternating a bright and a dim
    epoch shows every other epoch at a fraction of the median, and a
    threshold on raw rate would throw away one entire condition while
    reporting it as quality control. With it, a dim epoch is compared against
    other dim epochs and only a genuinely dead stretch falls out.

    A **contiguous** run, not a mask of every epoch that passes: a block goes
    bad by drifting, not by scattering, and letting the selection be
    non-contiguous silently unbalances the conditions.

    Returns a dict with ``start``, ``stop`` (exclusive), ``n_kept``,
    ``n_epochs``, ``normalized`` (per-epoch score), ``passes`` (bool array)
    and ``min_fraction``.
    """
    totals = epoch_population_counts(response_block, cell_types=cell_types,
                                     minimum_n=minimum_n, t_start_ms=t_start_ms,
                                     t_end_ms=t_end_ms)
    n = totals.size
    if n == 0:
        return {'start': 0, 'stop': 0, 'n_kept': 0, 'n_epochs': 0,
                'normalized': totals, 'passes': np.zeros(0, dtype=bool),
                'min_fraction': min_fraction}

    normalized = np.ones(n)
    if condition_values is not None and len(condition_values) >= n:
        values = np.asarray(condition_values[:n], dtype=object)
        for value in set(values.tolist()):
            mask = values == value
            median = float(np.median(totals[mask])) if mask.any() else 0.0
            normalized[mask] = totals[mask] / median if median > 0 else 0.0
    else:
        median = float(np.median(totals))
        normalized = totals / median if median > 0 else np.zeros(n)

    passes = normalized >= min_fraction

    # Longest contiguous run of True.
    best_len = best_start = 0
    run_start = None
    for i, ok in enumerate(np.append(passes, False)):
        if ok and run_start is None:
            run_start = i
        elif not ok and run_start is not None:
            if i - run_start > best_len:
                best_len, best_start = i - run_start, run_start
            run_start = None

    return {'start': int(best_start), 'stop': int(best_start + best_len),
            'n_kept': int(best_len), 'n_epochs': int(n),
            'normalized': normalized, 'passes': passes,
            'totals': totals, 'min_fraction': min_fraction}


def plot_epoch_range(result: dict, condition_values=None, ax=None,
                     title: Optional[str] = None):
    """Per-epoch population score with the chosen range shaded.

    Points are colored by condition when ``condition_values`` is given, which
    is what shows that the alternation is the experiment rather than a
    problem — the normalization has put both conditions on the same scale, so
    a point far below the line is a real dropout in either.
    """
    import matplotlib.pyplot as plt

    from .style import NEUTRAL_GRAY, apply_publication_style, colors_for_conditions

    apply_publication_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 3.2))

    normalized = result['normalized']
    x = np.arange(len(normalized))

    if condition_values is not None and len(condition_values) >= len(x):
        values = list(condition_values[:len(x)])
        levels = sorted(set(values), key=lambda v: (isinstance(v, str), v))
        colors = colors_for_conditions(levels)
        for level in levels:
            mask = np.array([v == level for v in values])
            ax.plot(x[mask], normalized[mask], 'o', markersize=5,
                    color=colors[level], label=f'{level}')
        ax.legend(title='condition', bbox_to_anchor=[1.02, 1], loc='upper left')
    else:
        ax.plot(x, normalized, 'o', markersize=5, color=NEUTRAL_GRAY)

    ax.axhline(result['min_fraction'], color=NEUTRAL_GRAY, linewidth=1.0)
    ax.axhline(1.0, color=NEUTRAL_GRAY, linewidth=0.6, alpha=0.5)
    if result['n_kept']:
        ax.axvspan(result['start'] - 0.5, result['stop'] - 0.5,
                   color='#0072B2', alpha=0.10, linewidth=0)

    ax.set_xlabel('Epoch index')
    ax.set_ylabel('Population spikes\n(relative to its condition)')
    ax.set_ylim(bottom=0)
    ax.set_title(title or f"keeping epochs {result['start']}–{result['stop'] - 1} "
                          f"({result['n_kept']} of {result['n_epochs']})")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    return ax


def epoch_condition_table(stim_block, response_block=None, cell_types=None,
                          minimum_n: int = 3, t_start_ms: float = 0.0,
                          t_end_ms: Optional[float] = None,
                          source=None) -> pd.DataFrame:
    """One row per epoch: the conditions it ran, and how much the retina fired.

    The two halves of an epoch belong together — what was shown and what came
    back — and putting them in one table is what makes an epoch range
    choosable by eye. Reading a condition table and a spike-count plot side by
    side leaves you doing the join in your head.

    Columns: ``epoch``, one per condition axis (whichever parameters vary
    across epochs), then ``n_spikes`` and ``spikes_per_cell`` when a
    ``response_block`` is given. Counts are summed over the cells of
    ``cell_types`` that clear ``minimum_n``, so they describe the population
    rather than any one cell.
    """
    from .protocol_source import block_parameters

    table = block_parameters(stim_block, source=source)
    keys = table.query('epoch_specific')['parameter'].tolist() if len(table) else []

    df_epochs = getattr(stim_block, 'df_epochs', None)
    n_epochs = len(df_epochs) if df_epochs is not None else 0

    out = pd.DataFrame({'epoch': np.arange(n_epochs)})
    for key in keys:
        if key in df_epochs.columns:
            out[key] = df_epochs[key].to_numpy()
        else:
            out[key] = [d.get(key) for d in df_epochs['epoch_parameters']]

    if response_block is not None:
        counts = epoch_population_counts(response_block, cell_types=cell_types,
                                         minimum_n=minimum_n,
                                         t_start_ms=t_start_ms, t_end_ms=t_end_ms)
        df = response_block.df_spike_times
        if cell_types is not None and 'cell_type' in df.columns:
            df = df[df['cell_type'].isin(list(cell_types))]
            df = df[df.groupby('cell_type')['cell_id'].transform('size') >= minimum_n]
        n_cells = max(len(df), 1)

        n = min(len(out), counts.size)
        out = out.iloc[:n].copy()
        out['n_spikes'] = counts[:n].astype(int)
        out['spikes_per_cell'] = (counts[:n] / n_cells).round(1)

    return out
