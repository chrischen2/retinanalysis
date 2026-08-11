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
    'plot_qc_mosaic',
    'epoch_condition_table',
    'drop_epoch_conditions',
    'population_template_qc',
]


def drop_epoch_conditions(epoch_table: pd.DataFrame, column: str,
                          values, *, strict: bool = True):
    """Return ``(kept, dropped)`` after excluding whole condition levels.

    The input order is preserved, which matters when the caller uses the
    returned epoch indices for drift or repeat analyses. ``values`` may be one
    scalar or an iterable. With ``strict=True`` (default), misspelled levels
    raise instead of silently leaving the table unchanged.
    """
    if column not in epoch_table.columns:
        raise KeyError(f'epoch table has no {column!r} column')
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        values = [values]
    requested = set(values)
    available = set(epoch_table[column].dropna().unique())
    unknown = requested - available
    if strict and unknown:
        raise ValueError(
            f'Unknown {column} value(s): {sorted(unknown)}; '
            f'available: {sorted(available)}')

    mask = epoch_table[column].isin(requested)
    dropped = epoch_table.loc[mask].copy().reset_index(drop=True)
    kept = epoch_table.loc[~mask].copy().reset_index(drop=True)
    if kept.empty:
        raise ValueError(f'excluding {column}={sorted(requested)} removes '
                         'every selected epoch')
    return kept, dropped


def _shape_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Shape correlation with an explicit verdict for flat traces."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float('nan')
    a, b = a[ok], b[ok]
    sa, sb = float(a.std()), float(b.std())
    eps = np.finfo(float).eps
    if sa <= eps and sb <= eps:
        return 1.0
    if sa <= eps or sb <= eps:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else float('nan')


def population_template_qc(
    response_block,
    epoch_table: pd.DataFrame,
    condition_keys: Sequence[str],
    *,
    epoch_column: str = 'epoch',
    cell_types: Optional[Iterable[str]] = None,
    candidate_cell_ids: Optional[Iterable[int]] = None,
    t_start_ms: float = 0.0,
    t_end_ms: Optional[float] = None,
    psth_sigma_ms: float = 50.0,
    sample_rate_hz: float = 100.0,
    min_cells_per_type: int = 5,
    min_template_r: float = 0.25,
    min_conditions: int = 2,
    min_pass_fraction: float = 0.5,
    verbose: bool = True,
):
    """Find cells whose condition PSTHs disagree with their population.

    For each ``cell_type × condition`` group, repeated-epoch PSTHs are
    averaged per cell. Every cell is then correlated with the mean of all
    *other* cells of its type (a leave-one-cell-out template, avoiding
    self-correlation). Correlation measures temporal shape, not firing-rate
    amplitude. A flat trace against a structured template receives zero.

    The downstream verdict is deliberately conservative: a cell is rejected
    only when it is evaluable in at least ``min_conditions`` and passes less
    than ``min_pass_fraction`` of them. Small types without enough cells to
    define a population template are retained as ``kept_unscored``.

    Returns ``(condition_detail, cell_summary)``. The detail table has one row
    per cell and condition; the summary has one row per candidate cell and a
    boolean ``passes_template`` suitable for merging with firing-rate QC.
    """
    keys = list(condition_keys)
    required = set(keys) | {epoch_column}
    missing = required.difference(epoch_table.columns)
    if missing:
        raise KeyError(f'epoch table missing columns: {sorted(missing)}')
    if t_end_ms is None or float(t_end_ms) <= float(t_start_ms):
        raise ValueError('t_end_ms must be greater than t_start_ms')
    if int(min_cells_per_type) < 2:
        raise ValueError('min_cells_per_type must be at least 2')
    if int(min_conditions) < 1:
        raise ValueError('min_conditions must be at least 1')
    if not -1 <= float(min_template_r) <= 1:
        raise ValueError('min_template_r must be between -1 and 1')
    if not 0 <= float(min_pass_fraction) <= 1:
        raise ValueError('min_pass_fraction must be between 0 and 1')

    cells = response_block.df_spike_times.copy()
    if cell_types is not None:
        cells = cells[cells['cell_type'].isin(set(cell_types))]
    if candidate_cell_ids is not None:
        candidate_ids = {int(v) for v in candidate_cell_ids}
        cells = cells[cells['cell_id'].astype(int).isin(candidate_ids)]
    if cells.empty:
        empty_detail = pd.DataFrame(columns=[
            'cell_id', 'cell_type', 'condition', 'n_repeats', 'self_r',
            'template_r', 'type_n', 'template_evaluable',
            'passes_condition'])
        empty_summary = pd.DataFrame(columns=[
            'cell_id', 'cell_type', 'n_conditions_seen',
            'n_conditions_evaluable', 'n_conditions_passed',
            'template_pass_fraction', 'median_template_r', 'min_template_r',
            'passes_template', 'template_status', 'failed_conditions'])
        return empty_detail, empty_summary

    # Preserve condition order from the retained epoch table. Epoch indices
    # remain indices into each cell's full spike_times list.
    condition_rows = []
    group_arg = keys[0] if len(keys) == 1 else keys
    for condition, rows in epoch_table.groupby(group_arg, sort=False,
                                                dropna=False):
        condition = condition if isinstance(condition, tuple) else (condition,)
        epoch_indices = rows[epoch_column].astype(int).tolist()
        condition_rows.append((condition, epoch_indices))

    detail_rows = []
    for condition, epoch_indices in condition_rows:
        condition_label = ', '.join(
            f'{key}={value}' for key, value in zip(keys, condition))
        for cell_type, type_cells in cells.groupby('cell_type', sort=False):
            records, means, self_values = [], [], []
            for row in type_cells.itertuples():
                selected = [row.spike_times[i] for i in epoch_indices
                            if 0 <= i < len(row.spike_times)]
                if not selected:
                    continue
                per_epoch = epoch_spikes_to_psth(
                    selected, float(t_end_ms),
                    psth_sigma_ms=float(psth_sigma_ms),
                    sample_rate_hz=float(sample_rate_hz),
                    t_start_ms=float(t_start_ms))
                mean = per_epoch.mean(axis=0)
                pairwise = [_shape_corr(per_epoch[i], per_epoch[j])
                            for i in range(len(per_epoch))
                            for j in range(i + 1, len(per_epoch))]
                self_r = (float(np.nanmean(pairwise)) if pairwise
                          and np.isfinite(pairwise).any() else np.nan)
                records.append(row)
                means.append(mean)
                self_values.append(self_r)

            if not means:
                continue
            matrix = np.stack(means)
            type_n = len(matrix)
            usable = type_n >= int(min_cells_per_type)
            total = matrix.sum(axis=0)
            for idx, row in enumerate(records):
                template = ((total - matrix[idx]) / (type_n - 1)
                            if type_n > 1 else np.full(matrix.shape[1], np.nan))
                template_r = (_shape_corr(matrix[idx], template)
                              if usable else np.nan)
                evaluable = usable and np.isfinite(template_r)
                passes = bool(evaluable and template_r >= min_template_r)
                detail_rows.append({
                    'cell_id': int(row.cell_id),
                    'cell_type': cell_type,
                    'condition': condition_label,
                    'condition_values': condition,
                    'n_repeats': len(epoch_indices),
                    'self_r': self_values[idx],
                    'template_r': template_r,
                    'type_n': type_n,
                    'template_evaluable': bool(evaluable),
                    'passes_condition': passes,
                })

    detail = pd.DataFrame(detail_rows)
    summary_rows = []
    for row in cells.itertuples():
        sub = detail[detail['cell_id'] == int(row.cell_id)]
        scored = sub[sub['template_evaluable']]
        n_eval = len(scored)
        n_pass = int(scored['passes_condition'].sum())
        pass_fraction = n_pass / n_eval if n_eval else np.nan
        passes = (True if n_eval < int(min_conditions)
                  else pass_fraction >= float(min_pass_fraction))
        failed = scored.loc[~scored['passes_condition'], 'condition'].tolist()
        summary_rows.append({
            'cell_id': int(row.cell_id),
            'cell_type': row.cell_type,
            'n_conditions_seen': int(len(sub)),
            'n_conditions_evaluable': int(n_eval),
            'n_conditions_passed': int(n_pass),
            'template_pass_fraction': float(pass_fraction),
            'median_template_r': (float(scored['template_r'].median())
                                  if n_eval else np.nan),
            'min_template_r': (float(scored['template_r'].min())
                               if n_eval else np.nan),
            'passes_template': bool(passes),
            'template_status': ('kept_unscored' if n_eval < int(min_conditions)
                                else ('kept' if passes else 'outlier')),
            'failed_conditions': failed,
        })
    summary = pd.DataFrame(summary_rows)
    detail.attrs.update({
        'condition_keys': keys,
        'min_template_r': float(min_template_r),
    })
    summary.attrs.update({
        'condition_keys': keys,
        'min_template_r': float(min_template_r),
        'min_conditions': int(min_conditions),
        'min_pass_fraction': float(min_pass_fraction),
    })

    if verbose:
        rejected = summary[~summary['passes_template']]
        unscored = int((summary['template_status'] == 'kept_unscored').sum())
        print(f'population-template QC: {len(summary) - len(rejected)}/'
              f'{len(summary)} cells retained; {len(rejected)} outliers; '
              f'{unscored} retained without enough template evidence')
        if len(rejected):
            print('Rejected template-outlier cells:')
            print(rejected[[
                'cell_id', 'cell_type', 'n_conditions_evaluable',
                'n_conditions_passed', 'template_pass_fraction',
                'median_template_r', 'failed_conditions']].to_string(index=False))
    return detail, summary


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


def plot_qc_mosaic(pipeline, qc, cell_types=None, std_scaling: float = 1.6,
                   ncols: int = 4, typing_file: Optional[str] = None,
                   title: Optional[str] = None, zoom: bool = True,
                   pad_frac: float = 0.04):
    """The mosaic with the QC decision drawn on it: kept filled, dropped open.

    A pass/fail table says how many cells a gate removed; it does not say
    *where* they were. Those are different failures. A gate that drops weak
    cells scattered through the mosaic has thinned the population evenly and
    the survivors still tile the array; a gate that empties one corner has
    removed a region of retina, and any population average afterwards is an
    average over the part that stayed. Only the mosaic separates the two.

    One panel per cell type, every RF at ``std_scaling`` σ. All panels get the
    same window, which is what lets you compare where a type's survivors sit
    against another's; per-panel autoscaling would put each type's own
    footprint in the same box and hide exactly that. Cells in ``qc`` that
    never matched a noise cluster have no receptive field to draw and are
    counted in the figure title rather than silently dropped from the picture.

    Parameters
    ----------
    pipeline : MEAPipeline
        Supplies ``match_dict`` (noise id → protocol id) and the analysis
        chunk the RFs live in.
    qc : pandas.DataFrame
        Output of :func:`filter_cells_by_qc` — needs ``cell_id`` (protocol
        ids), ``cell_type`` and the boolean ``passes``.
    cell_types : sequence[str], optional
        Panels to draw, in this order. Default: every type in ``qc``.
    ncols : int
        Panels per row.
    zoom : bool
        Frame the shared window on the cells rather than on the whole canvas.
        How much this crops depends on the chunk — a population filling the
        display barely moves, one sitting in a corner goes from speck to
        mosaic. The window can extend past the canvas edge, since a cell whose
        σ ellipse runs off the display is still drawn whole. ``False`` restores
        the full canvas, the frame to use when where the cells sat on the
        display is itself the point.
    pad_frac : float
        Margin around the zoomed window, as a fraction of its larger side.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    from .style import NEUTRAL_GRAY, apply_publication_style, colors_for_celltypes
    from .vision_utils import get_ells

    apply_publication_style()

    analysis_chunk = pipeline.analysis_chunk
    rf_params = getattr(analysis_chunk, 'rf_params', {}) or {}
    # match_dict is keyed noise id → protocol id; the QC table speaks protocol.
    to_noise = {int(v): int(k) for k, v in pipeline.match_dict.items()}

    if cell_types is None:
        cell_types = [ct for ct in qc['cell_type'].dropna().unique()]
    cell_types = [ct for ct in cell_types if (qc['cell_type'] == ct).any()]
    if not cell_types:
        raise ValueError('None of the requested cell types are in the QC table.')

    type_colors = colors_for_celltypes(list(cell_types))

    # {cell_type: {'kept': {noise_id: protocol_id}, 'dropped': {...}}}, plus a
    # count of the cells that have no RF to draw.
    drawn, n_no_rf = {}, 0
    for ct in cell_types:
        rows = qc[qc['cell_type'] == ct]
        groups = {}
        for label in ('kept', 'dropped'):
            wanted = rows[rows['passes'] == (label == 'kept')]['cell_id'].astype(int)
            ids = {}
            for pid in wanted:
                nid = to_noise.get(int(pid))
                if nid is None or nid not in rf_params:
                    n_no_rf += 1
                    continue
                ids[nid] = int(pid)
            groups[label] = ids
        drawn[ct] = groups

    # Geometry once, in canvas pixels, through the same path every other
    # mosaic in the package uses.
    by_type = {f'{ct}|{label}': list(ids)
               for ct, groups in drawn.items()
               for label, ids in groups.items() if ids}
    d_ells, _ = (get_ells(analysis_chunk, by_type, std_scaling=std_scaling,
                          units='pixels') if by_type else ({}, None))

    canvas_w, canvas_h = analysis_chunk.canvas_size

    # The window every panel shares. Zoomed, it is the box holding all the
    # drawn RFs of every type — both groups, since a gate that emptied a
    # region is only visible if the region is still in frame.
    x_lo, x_hi, y_lo, y_hi = 0.0, float(canvas_w), 0.0, float(canvas_h)
    all_ells = [e for ells in d_ells.values() for e in ells.values()]
    if zoom and all_ells:
        radii = np.array([max(e.width, e.height) / 2 for e in all_ells])
        cx = np.array([e.center[0] for e in all_ells])
        cy = np.array([e.center[1] for e in all_ells])
        x_lo, x_hi = float((cx - radii).min()), float((cx + radii).max())
        y_lo, y_hi = float((cy - radii).min()), float((cy + radii).max())
        pad = pad_frac * max(x_hi - x_lo, y_hi - y_lo)
        x_lo, x_hi, y_lo, y_hi = x_lo - pad, x_hi + pad, y_lo - pad, y_hi + pad

    ncols = max(1, min(int(ncols), len(cell_types)))
    nrows = int(np.ceil(len(cell_types) / ncols))
    panel_w = 3.6
    panel_aspect = (y_hi - y_lo) / (x_hi - x_lo) if x_hi > x_lo else 1.0
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(panel_w * ncols,
                                      panel_w * panel_aspect * nrows + 0.6))

    for ax, ct in zip(axes.ravel(), cell_types):
        color = type_colors.get(ct, NEUTRAL_GRAY)
        for label in ('dropped', 'kept'):      # kept on top
            for ell in d_ells.get(f'{ct}|{label}', {}).values():
                keep = label == 'kept'
                ax.add_patch(Ellipse(
                    xy=ell.center, width=ell.width, height=ell.height,
                    angle=ell.angle,
                    facecolor=color if keep else 'none',
                    edgecolor=color,
                    alpha=0.55 if keep else 0.9,
                    linewidth=0.9 if keep else 0.7,
                    linestyle='-' if keep else ':',
                    zorder=3 if keep else 2))
        n_kept, n_drop = (len(drawn[ct]['kept']), len(drawn[ct]['dropped']))
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_hi, y_lo)                # canvas y runs downwards
        ax.set_aspect('equal')
        ax.set_title(f'{ct} — {n_kept} kept, {n_drop} dropped',
                     fontsize=9, color=color)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes.ravel()[len(cell_types):]:
        ax.set_axis_off()

    fig.legend(handles=[
        Line2D([0], [0], marker='o', linestyle='none', markersize=8,
               markerfacecolor=NEUTRAL_GRAY, markeredgecolor=NEUTRAL_GRAY,
               alpha=0.55, label='kept'),
        Line2D([0], [0], marker='o', linestyle='none', markersize=8,
               markerfacecolor='none', markeredgecolor=NEUTRAL_GRAY,
               label='dropped')],
        loc='lower center', ncol=2, frameon=False, fontsize=9)

    suptitle = title or f'{std_scaling:g} σ receptive fields, by QC outcome'
    if n_no_rf:
        suptitle += f'  ({n_no_rf} cells with no matched RF not drawn)'
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


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
