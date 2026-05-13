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
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .psth import epoch_spikes_to_psth


__all__ = [
    'QCThresholds',
    'per_epoch_spike_counts',
    'cell_qc_metrics',
    'block_qc_metrics',
    'filter_cells_by_qc',
]


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
    """

    min_mean_count: Optional[float] = 1.0          # too quiet to analyze
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
) -> dict:
    """Compute QC metrics for one cell's per-epoch spike-time lists.

    See module docstring for metric definitions.
    """
    counts = per_epoch_spike_counts(spike_times_by_epoch, t_start_ms, t_end_ms)
    n = counts.size
    mean = float(counts.mean()) if n else 0.0
    std = float(counts.std()) if n else 0.0
    var = float(counts.var()) if n else 0.0

    silent_trial_frac = float((counts == 0).mean()) if n else float('nan')
    silent_run_max = int(_longest_run_of_zeros(counts))
    drift = _drift_score(counts)

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
        'mean_count': mean,
        'std_count': std,
        'cv_count': float(std / mean) if mean > 0 else float('inf'),
        'fano': float(var / mean) if mean > 0 else float('inf'),
        'silent_trial_frac': silent_trial_frac,
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
) -> pd.DataFrame:
    """Compute QC metrics for every cell in ``response_block.df_spike_times``.

    Returns a DataFrame with one row per cell:
    ``[cell_id, cell_type, n_epochs, mean_count, std_count, cv_count, fano,
       silent_trial_frac, silent_run_max, drift_score, reliability_r]``.

    Optional ``cell_types`` filter intersects with whatever the response
    block has labeled. Unlabeled cells are kept (cell_type may be NaN).
    """
    df = response_block.df_spike_times
    if cell_types is not None:
        want = set(cell_types)
        df = df[df['cell_type'].isin(want)]
    rows = []
    for _, r in df.iterrows():
        m = cell_qc_metrics(
            r['spike_times'],
            t_start_ms=t_start_ms,
            t_end_ms=t_end_ms,
            psth_sigma_ms=psth_sigma_ms,
            sample_rate_hz=sample_rate_hz,
        )
        m['cell_id'] = int(r['cell_id'])
        m['cell_type'] = r.get('cell_type', None)
        rows.append(m)
    cols = ['cell_id', 'cell_type', 'n_epochs', 'mean_count', 'std_count',
            'cv_count', 'fano', 'silent_trial_frac', 'silent_run_max',
            'drift_score', 'reliability_r']
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=cols)
    return out[cols]


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
        col_v = metrics_df[col]
        passes &= (col_v >= thr) if op == '>=' else (col_v <= thr)
    out = metrics_df.copy()
    out['passes'] = passes.values
    out.attrs['thresholds'] = th.asdict()
    return out
