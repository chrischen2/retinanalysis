"""Analysis for ``EyeMovementTrajectoryAlternatingBackground``.

The protocol cycles through ``imageNames × backgroundScale`` conditions
(see ``turner-package/+edu/+washington/+riekelab/+turner/+protocols/
EyeMovementTrajectoryAlternatingBackground.m``). For each cell, the
canonical analysis is:

1. Group epochs by ``currentBackgroundScale`` (and optionally
   ``currentImageName``).
2. Compute the Gaussian-smoothed PSTH per epoch.
3. Average per-condition PSTHs across epochs / cells of the same type.
4. Plot one panel per cell type with traces overlaid by condition so the
   reader can see how each type's response is modulated by background.

This module exposes :func:`analyze` (numeric results) and
:func:`plot_psth_by_condition` (visualization). Lower-level smoothing
lives in :mod:`retinanalysis.utils.psth`.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pathlib import Path

from joblib import Parallel, delayed

from retinanalysis.utils.psth import epoch_spikes_to_psth, psth_time_axis
from retinanalysis.utils.style import colors_for_conditions
from retinanalysis.utils.victor_purpura import (
    victor_purpura_distance,
    victor_purpura_distance_matrix,
    victor_purpura_cross_matrix,
    victor_purpura_batch_pairs,
)
from retinanalysis.config.settings import OUTPUT_DIR

PROTOCOL_NAME = 'edu.washington.riekelab.turner.protocols.EyeMovementTrajectoryAlternatingBackground'

# Default keys to split conditions by. The protocol runs each natural
# image at two background scales, so the canonical analysis groups by
# **(image, backgroundScale)** before averaging — otherwise per-image
# response structure (very different across images) is washed out.
DEFAULT_CONDITION_KEYS = ['currentImageName', 'currentBackgroundScale']

# Back-compat alias for callers that still pass a single key.
DEFAULT_CONDITION_KEY = DEFAULT_CONDITION_KEYS[-1]


def _epoch_total_ms(stim_block) -> float:
    bp = stim_block.d_epoch_block_params or {}
    return float(bp.get('preTime', 0) + bp.get('stimTime', 0) + bp.get('tailTime', 0))


def _resolve_keys(condition_keys, condition_key) -> List[str]:
    """Honor either the new ``condition_keys=`` list or the legacy single key."""
    if condition_keys is not None:
        return list(condition_keys)
    if condition_key is not None:
        return [condition_key]
    return list(DEFAULT_CONDITION_KEYS)


def analyze(
    pipeline,
    cell_types: Iterable[str],
    condition_keys: Optional[Sequence[str]] = None,
    condition_key: Optional[str] = None,  # legacy single-key alias
    psth_sigma_ms: float = 10.0,
    sample_rate_hz: float = 1000.0,
    minimum_n: int = 3,
) -> Dict:
    """Compute per-(cell-type × condition) mean PSTHs.

    Conditions are grouped by **all** keys in ``condition_keys`` (default:
    ``['currentImageName', 'currentBackgroundScale']``). Each unique
    combination becomes one bin — natural images differ enough in their
    response structure that averaging across them would wash out the
    per-image dynamics.

    Returns
    -------
    dict with keys:
        ``time_ms``: ``(n_bins,)`` bin centers in ms from epoch onset.
        ``condition_keys``: list of keys used (preserves order).
        ``conditions``: sorted list of condition tuples (one entry per
            unique combination); each tuple has ``len(condition_keys)``
            elements in the same order.
        ``cell_types``: list of types that passed ``minimum_n``.
        ``psth``: ``{cell_type: {condition_tuple: (n_cells, n_bins)}}``.
        ``preTime_ms, stimTime_ms, tailTime_ms``: protocol params.
        ``condition_key``: back-compat — equals ``condition_keys[-1]`` so
            the legacy single-key plot helper still works.
    """
    sb = pipeline.stim
    rb = pipeline.resp
    bp = sb.d_epoch_block_params or {}
    pre_ms = float(bp.get('preTime', 0))
    stim_ms = float(bp.get('stimTime', 0))
    tail_ms = float(bp.get('tailTime', 0))
    t_end_ms = pre_ms + stim_ms + tail_ms

    keys = _resolve_keys(condition_keys, condition_key)

    # Ensure every key has a column on df_epochs (pull from epoch_parameters
    # dict if missing) so we can build per-epoch tuples uniformly.
    for k in keys:
        if k not in sb.df_epochs.columns:
            sb.df_epochs[k] = [p.get(k) for p in sb.df_epochs['epoch_parameters']]

    epoch_idx = sb.df_epochs['epoch_index'].astype(int).tolist()
    per_epoch_tuples = list(zip(*[sb.df_epochs[k].tolist() for k in keys]))
    idx2tuple = dict(zip(epoch_idx, per_epoch_tuples))
    conditions = sorted(set(t for t in per_epoch_tuples if all(v is not None for v in t)))

    type_counts = rb.df_spike_times['cell_type'].value_counts()
    cell_types = [t for t in cell_types if type_counts.get(t, 0) >= minimum_n]

    psth_by_type: Dict[str, Dict] = {}
    for ct in cell_types:
        df_ct = rb.df_spike_times.query('cell_type == @ct')
        by_cond: Dict[Tuple, np.ndarray] = {}
        for cond in conditions:
            cond_epoch_idxs = [i for i, t in idx2tuple.items() if t == cond]
            per_cell_psth = []
            for _, row in df_ct.iterrows():
                # row['spike_times'] is a list indexed by epoch_index (0..n_epochs-1).
                epochs_in_cond = [row['spike_times'][i] for i in cond_epoch_idxs
                                  if i < len(row['spike_times'])]
                if not epochs_in_cond:
                    continue
                ep_psth = epoch_spikes_to_psth(
                    epochs_in_cond, t_end_ms,
                    psth_sigma_ms=psth_sigma_ms,
                    sample_rate_hz=sample_rate_hz,
                )
                per_cell_psth.append(ep_psth.mean(axis=0))
            if per_cell_psth:
                by_cond[cond] = np.stack(per_cell_psth)  # (n_cells, n_bins)
        psth_by_type[ct] = by_cond

    return {
        'time_ms': psth_time_axis(t_end_ms, sample_rate_hz, 0.0),
        'condition_keys': keys,
        'conditions': conditions,
        'cell_types': cell_types,
        'psth': psth_by_type,
        'preTime_ms': pre_ms,
        'stimTime_ms': stim_ms,
        'tailTime_ms': tail_ms,
        # Back-compat: last key is what plot_psth_by_condition uses for legend text.
        'condition_key': keys[-1],
        'psth_sigma_ms': psth_sigma_ms,
    }


def _format_cond_tuple(cond, keys):
    """Pretty-print a condition tuple: ``key1=v1, key2=v2``."""
    if isinstance(cond, tuple):
        return ', '.join(f'{k}={v}' for k, v in zip(keys, cond))
    return f'{keys[0]}={cond}'


def plot_psth_by_condition(
    results: Dict,
    axes: Optional[np.ndarray] = None,
    ncols: int = 2,
    show_individual_cells: bool = False,
    individual_alpha: float = 0.2,
) -> np.ndarray:
    """Plot the analyze() result.

    - **One key**: original layout — one panel per cell type, all conditions
      overlaid in that panel.
    - **Two keys** (primary, secondary): for each cell type, one panel per
      primary value (e.g. image) with traces colored by the secondary value
      (e.g. background scale). Multi-figure: each cell type gets its own.

    Returns the (final) axes array from the last figure created.
    """
    types = results['cell_types']
    conditions = results['conditions']
    time_ms = results['time_ms']
    keys = results.get('condition_keys', [results.get('condition_key')])

    pre = results['preTime_ms']
    stim = results['stimTime_ms']

    # ----- 1-key path -----
    if len(keys) <= 1:
        nrows = int(np.ceil(len(types) / ncols))
        if axes is None:
            fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 3.0 * nrows),
                                     sharex=True, squeeze=False)
        axes = np.atleast_2d(axes)
        flat_axes = axes.flatten()
        # Unique values (may be tuples of len 1 or plain scalars)
        plain_conds = [c[0] if isinstance(c, tuple) else c for c in conditions]
        cond_colors_flat = colors_for_conditions(plain_conds)

        for i, ct in enumerate(types):
            ax = flat_axes[i]
            for cond, plain in zip(conditions, plain_conds):
                mat = results['psth'].get(ct, {}).get(cond)
                if mat is None or mat.size == 0:
                    continue
                color = cond_colors_flat[plain]
                if show_individual_cells:
                    for row in mat:
                        ax.plot(time_ms, row, color=color,
                                alpha=individual_alpha, linewidth=0.6)
                mean = mat.mean(axis=0)
                sem = mat.std(axis=0) / np.sqrt(max(mat.shape[0], 1))
                ax.plot(time_ms, mean, color=color, linewidth=1.6,
                        label=f'{_format_cond_tuple(cond, keys)}  (n={mat.shape[0]})')
                ax.fill_between(time_ms, mean - sem, mean + sem,
                                color=color, alpha=0.2, linewidth=0)
            if stim > 0:
                ax.axvline(pre, color='red', lw=0.6, ls='--', alpha=0.7)
                ax.axvline(pre + stim, color='red', lw=0.6, ls='--', alpha=0.7)
            ax.set_title(ct)
            ax.set_xlabel('time (ms)')
            ax.set_ylabel('rate (Hz)')
            ax.legend(loc='upper right', fontsize=8, framealpha=0.7)

        for j in range(len(types), len(flat_axes)):
            flat_axes[j].axis('off')
        if hasattr(axes[0, 0], 'figure'):
            axes[0, 0].figure.tight_layout()
        return axes

    # ----- 2-key path: per-image grid, one figure per cell type -----
    primary_values = sorted({c[0] for c in conditions},
                            key=lambda x: (isinstance(x, str), x))
    secondary_values = sorted({c[1] for c in conditions},
                              key=lambda x: (isinstance(x, str), x))
    sec_colors = colors_for_conditions(secondary_values)
    primary_key, secondary_key = keys[0], keys[1]

    n_p = len(primary_values)
    g_ncols = min(5, n_p)
    g_nrows = int(np.ceil(n_p / g_ncols))

    last_axes = None
    for ct in types:
        fig, ax_grid = plt.subplots(
            g_nrows, g_ncols,
            figsize=(2.4 * g_ncols, 1.8 * g_nrows + 0.5),
            sharex=True, sharey=True, squeeze=False,
        )
        fig.suptitle(f'{ct}', fontsize=11)
        # Precompute y-max across panels for shared scaling
        y_max = 0.0
        per_panel = {}
        for p in primary_values:
            for s in secondary_values:
                mat = results['psth'].get(ct, {}).get((p, s))
                if mat is None or mat.size == 0:
                    continue
                mu = mat.mean(axis=0)
                sem = mat.std(axis=0) / np.sqrt(max(mat.shape[0], 1))
                per_panel[(p, s)] = (mu, sem, mat.shape[0])
                y_max = max(y_max, float((mu + sem).max()))

        handles = []
        for i, p in enumerate(primary_values):
            r, c = divmod(i, g_ncols)
            ax = ax_grid[r, c]
            for s in secondary_values:
                d = per_panel.get((p, s))
                if d is None:
                    continue
                mu, sem, n = d
                line, = ax.plot(time_ms, mu, color=sec_colors[s], lw=1.0,
                                label=f'{secondary_key}={s}')
                ax.fill_between(time_ms, mu - sem, mu + sem,
                                color=sec_colors[s], alpha=0.2, linewidth=0)
                if i == 0:
                    handles.append(line)
            if stim > 0:
                ax.axvline(pre, color='red', lw=0.4, ls='--', alpha=0.5)
                ax.axvline(pre + stim, color='red', lw=0.4, ls='--', alpha=0.5)
            ax.set_title(f'{primary_key}={p}', fontsize=8)
            ax.set_ylim(0, y_max * 1.05 if y_max > 0 else 1.0)
            if r != g_nrows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('time (ms)', fontsize=8)
            if c != 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel('rate (Hz)', fontsize=8)
        # Hide unused trailing panels
        for j in range(n_p, g_nrows * g_ncols):
            r, c = divmod(j, g_ncols)
            ax_grid[r, c].axis('off')
        if handles:
            fig.legend(handles=handles, loc='upper right',
                       bbox_to_anchor=(0.99, 0.97), fontsize=8,
                       framealpha=0.7, title=secondary_key)
        fig.tight_layout(rect=(0, 0, 0.95, 0.96))
        last_axes = ax_grid
    return last_axes


# ===========================================================================
# Offline-data analyses (operate on OfflineDataset, no DataJoint required)
# ===========================================================================

def _resolve_condition_keys_from_offline(offline, condition_keys):
    """Use caller-provided keys, or fall back to ``offline.meta['condition_keys']``."""
    if condition_keys is not None:
        return list(condition_keys)
    keys = offline.meta.get('condition_keys')
    if keys is None:
        return list(DEFAULT_CONDITION_KEYS)
    if isinstance(keys, np.ndarray):
        keys = keys.tolist()
    return [str(k) for k in keys if k]


def _epoch_indices_by_condition(offline, condition_keys):
    """Return ``{condition_tuple: np.array(epoch_indices)}``."""
    df = offline.epochs
    out = {}
    for cond, sub in df.groupby(list(condition_keys), sort=True):
        if not isinstance(cond, tuple):
            cond = (cond,)
        cond = tuple(_clean_value(v) for v in cond)
        out[cond] = sub['epoch_index'].astype(int).to_numpy()
    return out


def _clean_value(v):
    """Strip the placeholder empty-string we use for missing condition values."""
    if isinstance(v, bytes):
        v = v.decode('utf-8')
    if isinstance(v, str) and v == '':
        return None
    return v


def analyze_offline(
    offline,
    cell_types: Optional[Iterable[str]] = None,
    condition_keys: Optional[Sequence[str]] = None,
    minimum_n: int = 3,
) -> Dict:
    """Offline equivalent of :func:`analyze` — works without DataJoint.

    Pulls the pre-computed PSTHs that ``save_offline_data`` wrote, groups
    by condition, and averages per (cell type, condition).

    Parameters
    ----------
    offline : OfflineDataset
    cell_types : iterable[str], optional
        Restrict to these types. Default: every type with ≥ ``minimum_n``.
    condition_keys : sequence[str], optional
        Per-epoch columns to split conditions by. Default:
        ``offline.meta['condition_keys']``.
    minimum_n : int
        Skip cell types with fewer than this many cells.

    Returns
    -------
    dict
        Same structure as :func:`analyze`: ``time_ms, condition_keys,
        conditions, cell_types, psth, preTime_ms, stimTime_ms, tailTime_ms``.
    """
    keys = _resolve_condition_keys_from_offline(offline, condition_keys)
    cond_to_epochs = _epoch_indices_by_condition(offline, keys)
    conditions = sorted(cond_to_epochs.keys(),
                        key=lambda c: tuple(str(x) for x in c))

    df_cells = offline.cells
    if cell_types is None:
        type_counts = df_cells['cell_type'].value_counts()
    else:
        type_counts = (df_cells.loc[df_cells['cell_type'].isin(list(cell_types)),
                                    'cell_type'].value_counts())
    cell_types_kept = [t for t, n in type_counts.items() if n >= minimum_n]

    time_ms = offline.psth_time_ms()
    pre_ms = float(offline.timing.get('preTime_ms', 0))
    stim_ms = float(offline.timing.get('stimTime_ms', 0))
    tail_ms = float(offline.timing.get('tailTime_ms', 0))

    psth_by_type: Dict[str, Dict[Tuple, np.ndarray]] = {}
    for ct in cell_types_kept:
        cids = df_cells.loc[df_cells['cell_type'] == ct, 'cell_id'].astype(int).tolist()
        by_cond: Dict[Tuple, List[np.ndarray]] = {}
        for cid in cids:
            psth = offline.psth_matrix(cid)  # (n_epochs, n_bins)
            for cond, ep_idx in cond_to_epochs.items():
                in_range = ep_idx[ep_idx < psth.shape[0]]
                if in_range.size == 0:
                    continue
                by_cond.setdefault(cond, []).append(psth[in_range].mean(axis=0))
        psth_by_type[ct] = {c: np.stack(v) for c, v in by_cond.items() if v}

    return {
        'time_ms': time_ms,
        'condition_keys': keys,
        'conditions': conditions,
        'cell_types': cell_types_kept,
        'psth': psth_by_type,
        'preTime_ms': pre_ms,
        'stimTime_ms': stim_ms,
        'tailTime_ms': tail_ms,
        'condition_key': keys[-1],
        'psth_sigma_ms': float(offline.timing.get('psth_sigma_ms', 10.0)),
    }


def _window_spikes_seconds(
    spike_times_ms: np.ndarray,
    t_start_ms: float,
    t_end_ms: float,
) -> np.ndarray:
    """Slice + convert to seconds, rebased to the window start."""
    arr = np.asarray(spike_times_ms, dtype=float)
    sel = arr[(arr >= t_start_ms) & (arr < t_end_ms)]
    return (sel - t_start_ms) / 1000.0


def _vp_one_cell(
    cell_payload: Dict,
    groups: Dict[Tuple, List[Tuple]],
    cost_per_sec: float,
    window_sec: float,
    window_offset_ms: float,
    exp_name: str,
) -> List[Dict]:
    """Per-cell VP work. Runs in a joblib worker — top-level so it pickles.

    All file I/O (HDF5 spike-time reads) has already happened in the
    main process; this only takes the pre-windowed train arrays and
    runs the bulk-C VP calls. Each worker re-imports the C kernel
    (the .dylib is cached on disk) so there's no NAS or DJ traffic
    from the worker side.
    """
    cid = cell_payload['cell_id']
    ct = cell_payload['cell_type']
    trains_by_cond: Dict[Tuple, List[np.ndarray]] = cell_payload['trains_by_cond']

    # Within-condition baseline (mean of upper-triangle entries).
    d_within: Dict[Tuple, float] = {}
    n_within: Dict[Tuple, int] = {}
    for cond, trains in trains_by_cond.items():
        if len(trains) < 2:
            d_within[cond], n_within[cond] = float('nan'), 0
            continue
        D = victor_purpura_distance_matrix(trains, cost_per_sec)
        iu = np.triu_indices(len(trains), k=1)
        d_within[cond] = float(D[iu].mean())
        n_within[cond] = int(iu[0].size)

    out_rows: List[Dict] = []
    for group_key, conds in groups.items():
        for i in range(len(conds)):
            ci = conds[i]
            if ci not in trains_by_cond:
                continue
            for j in range(i + 1, len(conds)):
                cj = conds[j]
                if cj not in trains_by_cond:
                    continue
                Xc = victor_purpura_cross_matrix(
                    trains_by_cond[ci], trains_by_cond[cj], cost_per_sec)
                d_cross = float(Xc.mean()) if Xc.size else float('nan')
                n_cross = int(Xc.size)
                within_avg = 0.5 * (d_within[ci] + d_within[cj])
                out_rows.append({
                    'exp_name': exp_name,
                    'cell_id': cid,
                    'cell_type': ct,
                    'group_key': group_key,
                    'cond_i': ci,
                    'cond_j': cj,
                    'd_within_i': d_within[ci],
                    'd_within_j': d_within[cj],
                    'd_within_avg': within_avg,
                    'd_across': d_cross,
                    'd_diff': d_cross - within_avg,
                    'n_pairs_within_i': n_within[ci],
                    'n_pairs_within_j': n_within[cj],
                    'n_pairs_across': n_cross,
                    'window_sec': window_sec,
                    'window_offset_ms': window_offset_ms,
                    'cost_per_sec': cost_per_sec,
                })
    return out_rows


def spike_distance_analysis(
    offline,
    *,
    condition_keys: Optional[Sequence[str]] = None,
    pair_within: Optional[Sequence[str]] = ('currentImageName',),
    pair_across: Optional[Sequence[str]] = None,
    window_sec: float = 5.0,
    window_offset_ms: Optional[float] = None,
    cost_per_sec: float = 4.0,
    n_trials_cap: Optional[int] = None,
    cell_types: Optional[Iterable[str]] = None,
    minimum_n: int = 3,
    rng_seed: int = 0,
    n_jobs: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Within-condition vs across-condition Victor–Purpura distance per cell.

    For each cell, take a single time window from each trial's spike
    train (default: the first ``window_sec`` seconds after preTime),
    then compute pairwise VP distances. By default we pair *across*
    ``currentBackgroundScale`` while *holding* ``currentImageName``
    fixed — i.e. the asked-for "compare low vs high backgroundScale
    for each image" pattern.

    A *positive* ``d_diff`` means trials inside a condition are more
    similar to each other than to trials of the other condition — i.e.
    the across-axis (backgroundScale) modulates the response.

    Parameters
    ----------
    offline : OfflineDataset
    pair_within : sequence[str], optional
        Condition keys held *constant* when picking pairs (default
        ``('currentImageName',)``). Within-condition baselines and
        across pairings are computed inside each unique combination of
        these keys. Set ``None`` or ``[]`` to compare every condition
        pair (the old behavior).
    pair_across : sequence[str], optional
        Condition keys allowed to differ when picking across pairs.
        Default = remaining keys (``condition_keys`` minus
        ``pair_within``).
    window_sec : float
        Window duration in seconds. Default 5 s.
    window_offset_ms : float, optional
        Window start in ms from epoch onset. Default: ``preTime_ms``.
    cost_per_sec : float
        VP cost (reciprocal is the metric's timescale). ``cost=4`` ≈
        250 ms — appropriate for RGC response structure.
    n_trials_cap : int, optional
        Subsample to at most this many trials per condition before
        computing pairs.
    cell_types : iterable, optional
        Restrict to these types.
    minimum_n : int
        Skip cell types with fewer than this many cells.
    rng_seed : int
        Deterministic subsampling.
    n_jobs : int
        Worker processes for the per-cell VP computation. **Default 1**
        because the C kernel already parallelizes *within* each bulk
        VP call using POSIX threads (saturating all CPU cores via
        ``vp_pairwise`` / ``vp_self_pairwise``). Setting ``n_jobs > 1``
        forks Python-level workers on top of that and usually
        over-subscribes the CPU, slowing things down. Set the env var
        ``SPKD_NUM_THREADS=N`` to cap the C-layer thread pool. Use
        ``n_jobs > 1`` only when you've explicitly set
        ``SPKD_NUM_THREADS=1`` to disable C threading (rare).
    verbose : bool
        Print a one-line summary before dispatch.

    Returns
    -------
    pandas.DataFrame
        One row per (cell × condition-pair).
    """
    keys = _resolve_condition_keys_from_offline(offline, condition_keys)
    cond_to_epochs = _epoch_indices_by_condition(offline, keys)

    # Resolve pair_within / pair_across
    pair_within = [k for k in (pair_within or []) if k in keys]
    if pair_across is None:
        pair_across = [k for k in keys if k not in pair_within]
    else:
        pair_across = [k for k in pair_across if k in keys]

    # Pick which axis indices of the condition tuple are "within" vs "across"
    within_idx = [keys.index(k) for k in pair_within]
    across_idx = [keys.index(k) for k in pair_across]

    def _key_within(cond):  # the "image" group key
        return tuple(cond[i] for i in within_idx)

    def _key_across(cond):  # the "backgroundScale" group key
        return tuple(cond[i] for i in across_idx)

    # Group conditions by the "within" key, then compute pairings inside each group.
    groups: Dict[Tuple, List[Tuple]] = {}
    for cond in cond_to_epochs:
        groups.setdefault(_key_within(cond), []).append(cond)
    # Stable ordering of conditions within each group (e.g. low → high scale).
    for g in groups.values():
        g.sort(key=_key_across)

    pre_ms = float(offline.timing.get('preTime_ms', 0))
    if window_offset_ms is None:
        window_offset_ms = pre_ms
    t_start = float(window_offset_ms)
    t_end = t_start + window_sec * 1000.0

    df_cells = offline.cells
    if cell_types is not None:
        df_cells = df_cells.loc[df_cells['cell_type'].isin(list(cell_types))]
    counts = df_cells['cell_type'].value_counts()
    keep_types = set(counts.index[counts >= minimum_n])
    df_cells = df_cells.loc[df_cells['cell_type'].isin(keep_types)]

    rng = np.random.RandomState(rng_seed)

    # ------------------------------------------------------------------
    # Pre-extract every cell's windowed trains in the MAIN process. This
    # is fast (HDF5 reads + numpy slicing) and gives us a clean
    # serializable payload to ship to workers — no HDF5 file handles or
    # offline-dataset references cross process boundaries.
    # ------------------------------------------------------------------
    payloads: List[Dict] = []
    for _, cell_row in df_cells.iterrows():
        cid = int(cell_row['cell_id'])
        ct = cell_row.get('cell_type', '')
        sts_list = offline.spike_times(cid)

        trains_by_cond: Dict[Tuple, List[np.ndarray]] = {}
        for cond, ep_idx in cond_to_epochs.items():
            sel = ep_idx[ep_idx < len(sts_list)]
            if sel.size == 0:
                continue
            trains = [_window_spikes_seconds(sts_list[i], t_start, t_end)
                      for i in sel]
            if n_trials_cap is not None and len(trains) > n_trials_cap:
                idx = rng.choice(len(trains), n_trials_cap, replace=False)
                trains = [trains[i] for i in sorted(idx)]
            trains_by_cond[cond] = trains

        payloads.append({
            'cell_id': cid,
            'cell_type': ct,
            'trains_by_cond': trains_by_cond,
        })

    # ------------------------------------------------------------------
    # Build ONE flat batch of every VP pair across every cell × condition.
    # vp_batch_pairs internally fans the pair list across CPU cores via
    # pthreads, so the thread-pool setup is amortized over the whole
    # workload — a huge improvement over the previous shape where each
    # call had only 3–9 pairs and pthread overhead always lost.
    # ------------------------------------------------------------------
    all_trains: List[np.ndarray] = []
    train_index_by_cond: List[Dict[Tuple, np.ndarray]] = []  # per cell
    for p in payloads:
        cond_to_idx: Dict[Tuple, np.ndarray] = {}
        for cond, trains in p['trains_by_cond'].items():
            start = len(all_trains)
            all_trains.extend(trains)
            cond_to_idx[cond] = np.arange(start, start + len(trains), dtype=int)
        train_index_by_cond.append(cond_to_idx)

    # Pair lists with provenance so we can demultiplex distances later.
    # Each entry: (cell_idx, kind, ci, cj, group_key, n_within_per_cond, …)
    pair_a: List[int] = []
    pair_b: List[int] = []
    # Provenance per "block" — a contiguous slice of pairs that share
    # the same (cell_idx, cond_i, cond_j) tuple. Avoids one provenance
    # entry per pair (n_pairs is ~10k+).
    blocks: List[Dict] = []
    for cell_idx, payload in enumerate(payloads):
        ci_to_idx = train_index_by_cond[cell_idx]
        trains_by_cond = payload['trains_by_cond']

        # Within-condition pairs (upper triangle).
        for cond, trains in trains_by_cond.items():
            ntr = len(trains)
            if ntr < 2:
                continue
            idx_arr = ci_to_idx[cond]
            iu_i, iu_j = np.triu_indices(ntr, k=1)
            block_start = len(pair_a)
            pair_a.extend(idx_arr[iu_i].tolist())
            pair_b.extend(idx_arr[iu_j].tolist())
            blocks.append({
                'cell_idx': cell_idx, 'kind': 'within',
                'cond_i': cond, 'cond_j': cond,
                'group_key': None,
                'slice': (block_start, len(pair_a)),
                'n_pairs': iu_i.size,
            })

        # Across-condition pairs (rectangular blocks within an image group).
        for group_key, conds in groups.items():
            for i in range(len(conds)):
                ci = conds[i]
                if ci not in trains_by_cond:
                    continue
                for j in range(i + 1, len(conds)):
                    cj = conds[j]
                    if cj not in trains_by_cond:
                        continue
                    idx_i = ci_to_idx[ci]
                    idx_j = ci_to_idx[cj]
                    grid_a, grid_b = np.meshgrid(idx_i, idx_j, indexing='ij')
                    block_start = len(pair_a)
                    pair_a.extend(grid_a.ravel().tolist())
                    pair_b.extend(grid_b.ravel().tolist())
                    blocks.append({
                        'cell_idx': cell_idx, 'kind': 'cross',
                        'cond_i': ci, 'cond_j': cj,
                        'group_key': group_key,
                        'slice': (block_start, len(pair_a)),
                        'n_pairs': idx_i.size * idx_j.size,
                    })

    n_pairs = len(pair_a)
    if verbose:
        print(f'spike_distance_analysis: {len(payloads)} cells, '
              f'{n_pairs} VP pairs in one batch '
              f'(set SPKD_NUM_THREADS=N to cap C thread pool)')

    if n_pairs > 0:
        pair_array = np.column_stack([np.asarray(pair_a, dtype=np.int32),
                                       np.asarray(pair_b, dtype=np.int32)])
        all_distances = victor_purpura_batch_pairs(
            all_trains, pair_array, cost_per_sec)
    else:
        all_distances = np.zeros(0, dtype=np.float64)

    # Optional joblib layer is intentionally bypassed when there's
    # exactly one batch — the C thread pool already saturates cores.
    # n_jobs is retained as a parameter for the rare "C threading
    # disabled" case (SPKD_NUM_THREADS=1) where the user wants to
    # parallelize at Python level instead.

    # ------------------------------------------------------------------
    # Demultiplex: collect per-cell d_within (per condition), then
    # walk blocks to assemble result rows in the original shape.
    # ------------------------------------------------------------------
    per_cell_d_within: List[Dict[Tuple, float]] = [
        {} for _ in payloads]
    per_cell_n_within: List[Dict[Tuple, int]] = [
        {} for _ in payloads]
    per_cell_d_cross: List[Dict[Tuple, float]] = [
        {} for _ in payloads]
    per_cell_n_cross: List[Dict[Tuple, int]] = [
        {} for _ in payloads]
    per_cell_group: List[Dict[Tuple, Tuple]] = [
        {} for _ in payloads]
    for b in blocks:
        s0, s1 = b['slice']
        dists = all_distances[s0:s1]
        if b['kind'] == 'within':
            per_cell_d_within[b['cell_idx']][b['cond_i']] = (
                float(dists.mean()) if dists.size else float('nan'))
            per_cell_n_within[b['cell_idx']][b['cond_i']] = int(dists.size)
        else:
            key = (b['cond_i'], b['cond_j'])
            per_cell_d_cross[b['cell_idx']][key] = (
                float(dists.mean()) if dists.size else float('nan'))
            per_cell_n_cross[b['cell_idx']][key] = int(dists.size)
            per_cell_group[b['cell_idx']][key] = b['group_key']

    rows = []
    for cell_idx, payload in enumerate(payloads):
        cid = payload['cell_id']
        ct = payload['cell_type']
        trains_by_cond = payload['trains_by_cond']
        # Fill in NaN for conds that had < 2 trials.
        for cond in trains_by_cond:
            per_cell_d_within[cell_idx].setdefault(cond, float('nan'))
            per_cell_n_within[cell_idx].setdefault(cond, 0)
        for (ci, cj), d_cross in per_cell_d_cross[cell_idx].items():
            within_avg = 0.5 * (per_cell_d_within[cell_idx][ci]
                                 + per_cell_d_within[cell_idx][cj])
            rows.append({
                'exp_name': offline.exp_name,
                'cell_id': cid,
                'cell_type': ct,
                'group_key': per_cell_group[cell_idx][(ci, cj)],
                'cond_i': ci,
                'cond_j': cj,
                'd_within_i': per_cell_d_within[cell_idx][ci],
                'd_within_j': per_cell_d_within[cell_idx][cj],
                'd_within_avg': within_avg,
                'd_across': d_cross,
                'd_diff': d_cross - within_avg,
                'n_pairs_within_i': per_cell_n_within[cell_idx][ci],
                'n_pairs_within_j': per_cell_n_within[cell_idx][cj],
                'n_pairs_across': per_cell_n_cross[cell_idx][(ci, cj)],
                'window_sec': window_sec,
                'window_offset_ms': window_offset_ms,
                'cost_per_sec': cost_per_sec,
            })

    cols = ['exp_name', 'cell_id', 'cell_type', 'group_key', 'cond_i', 'cond_j',
            'd_within_i', 'd_within_j', 'd_within_avg', 'd_across', 'd_diff',
            'n_pairs_within_i', 'n_pairs_within_j', 'n_pairs_across',
            'window_sec', 'window_offset_ms', 'cost_per_sec']
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Movie-repeat (cycle 1 vs cycle 2)
# ---------------------------------------------------------------------------

def movie_repeat_analysis(
    offline,
    *,
    cycle_sec: float = 15.0,
    drop_first_sec: float = 1.0,
    condition_keys: Optional[Sequence[str]] = None,
    cell_types: Optional[Iterable[str]] = None,
    minimum_n: int = 3,
    cost_per_sec: float = 4.0,
    compute_vp: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compare cycle-1 vs cycle-2 response (movie shown twice).

    Protocol: ``preTime + stimTime`` where ``stimTime = 2 × cycle_sec``.
    For each cell × condition, build the per-trial PSTH over cycle 1
    and cycle 2 (each ``cycle_sec - drop_first_sec`` long), then report:

    - ``corr``       — Pearson correlation between mean cycle-1 and cycle-2 PSTHs.
    - ``rmse``       — RMS deviation between mean cycle-1 and cycle-2 PSTHs (Hz).
    - ``rate_ratio`` — mean rate cycle-2 / mean rate cycle-1 (adaptation index).
    - ``vp_distance``— per-trial average VP distance(cycle-1, cycle-2). Picks
                       up trial-by-trial timing differences invisible to
                       trial-averaged correlation.
    - ``n_trials``   — number of trials averaged.

    Returns
    -------
    pandas.DataFrame
        One row per (cell, condition).
    """
    keys = _resolve_condition_keys_from_offline(offline, condition_keys)
    cond_to_epochs = _epoch_indices_by_condition(offline, keys)

    pre_ms = float(offline.timing.get('preTime_ms', 0))
    stim_ms = float(offline.timing.get('stimTime_ms', 0))
    cycle_ms = cycle_sec * 1000.0
    drop_ms = drop_first_sec * 1000.0

    if stim_ms + 1e-6 < 2 * cycle_ms:
        raise ValueError(
            f'movie_repeat_analysis: stim_ms={stim_ms} but cycle_sec={cycle_sec}'
            f' implies a 2× cycle of {2*cycle_ms} ms — protocol mismatch.'
        )

    df_cells = offline.cells
    if cell_types is not None:
        df_cells = df_cells.loc[df_cells['cell_type'].isin(list(cell_types))]
    counts = df_cells['cell_type'].value_counts()
    keep_types = set(counts.index[counts >= minimum_n])
    df_cells = df_cells.loc[df_cells['cell_type'].isin(keep_types)]

    # Slice indices into the saved PSTH
    time_ms = offline.psth_time_ms()
    bin_ms = float(time_ms[1] - time_ms[0]) if len(time_ms) > 1 else 1.0

    c1_start = int(round((pre_ms + drop_ms) / bin_ms))
    c1_end = int(round((pre_ms + cycle_ms) / bin_ms))
    c2_start = int(round((pre_ms + cycle_ms + drop_ms) / bin_ms))
    c2_end = int(round((pre_ms + 2 * cycle_ms) / bin_ms))
    nb = min(c1_end - c1_start, c2_end - c2_start)
    if nb <= 1:
        raise ValueError('movie_repeat_analysis: window too short for current PSTH bin width.')
    c1_end = c1_start + nb
    c2_end = c2_start + nb

    # Same windows in ms for VP
    c1_ms = (pre_ms + drop_ms, pre_ms + cycle_ms)
    c2_ms = (pre_ms + cycle_ms + drop_ms, pre_ms + 2 * cycle_ms)

    # ------------------------------------------------------------------
    # First pass: build per-(cell, condition) PSTH stats and (if
    # compute_vp) collect every (cycle-1, cycle-2) train pair into one
    # flat batch. Dispatch the VP batch ONCE with C-level pthread fan-out
    # — same shape fix as spike_distance_analysis.
    # ------------------------------------------------------------------
    per_cc_stats: List[Dict] = []   # one entry per (cell, condition) row
    vp_trains: List[np.ndarray] = []
    vp_pair_blocks: List[Tuple[int, int]] = []  # (start, end) into vp_trains pairs

    for _, cell_row in df_cells.iterrows():
        cid = int(cell_row['cell_id'])
        ct = cell_row.get('cell_type', '')
        psth = offline.psth_matrix(cid)  # (n_epochs, n_bins)
        sts_list = offline.spike_times(cid) if compute_vp else None

        for cond, ep_idx in cond_to_epochs.items():
            sel = ep_idx[ep_idx < psth.shape[0]]
            if sel.size == 0:
                continue

            c1 = psth[sel, c1_start:c1_end]   # (n_trials, nb)
            c2 = psth[sel, c2_start:c2_end]
            mean_c1 = c1.mean(axis=0)
            mean_c2 = c2.mean(axis=0)
            if mean_c1.std() < 1e-9 or mean_c2.std() < 1e-9:
                corr = np.nan
            else:
                corr = float(np.corrcoef(mean_c1, mean_c2)[0, 1])
            rmse = float(np.sqrt(np.mean((mean_c1 - mean_c2) ** 2)))
            r1 = float(mean_c1.mean())
            r2 = float(mean_c2.mean())
            rate_ratio = (r2 / r1) if r1 > 1e-9 else float('nan')

            # Append c1/c2 trains for this (cell, condition) into the
            # global batch; record the slice for later mean aggregation.
            block_start = len(vp_trains) // 2
            if compute_vp:
                for i in sel:
                    vp_trains.append(
                        _window_spikes_seconds(sts_list[i], *c1_ms))
                    vp_trains.append(
                        _window_spikes_seconds(sts_list[i], *c2_ms))
            block_end = len(vp_trains) // 2

            per_cc_stats.append({
                'cid': cid, 'ct': ct, 'cond': cond,
                'n_trials': int(sel.size),
                'r1': r1, 'r2': r2, 'rate_ratio': rate_ratio,
                'corr': corr, 'rmse': rmse,
                'vp_slice': (block_start, block_end),
            })

    # One C call processes every consecutive (c1, c2) pair in vp_trains.
    if compute_vp and len(vp_trains) >= 2:
        n_pairs = len(vp_trains) // 2
        idx = np.arange(n_pairs, dtype=np.int32) * 2
        pair_array = np.column_stack([idx, idx + 1])
        if verbose:
            print(f'movie_repeat_analysis: {n_pairs} cycle-1/cycle-2 VP '
                  f'pairs in one batch')
        vp_per_pair = victor_purpura_batch_pairs(
            vp_trains, pair_array, cost_per_sec)
    else:
        vp_per_pair = np.zeros(0, dtype=np.float64)

    # Second pass: assemble the rows DataFrame, looking up each
    # (cell, condition)'s mean VP from its slice of the batch result.
    rows = []
    for s in per_cc_stats:
        if compute_vp:
            a, b = s['vp_slice']
            vp_mean = (float(vp_per_pair[a:b].mean())
                        if b > a else float('nan'))
        else:
            vp_mean = float('nan')

        rows.append({
            'exp_name': offline.exp_name,
            'cell_id': s['cid'],
            'cell_type': s['ct'],
            'condition': s['cond'],
            'n_trials': s['n_trials'],
            'rate_cycle1_hz': s['r1'],
            'rate_cycle2_hz': s['r2'],
            'rate_ratio': s['rate_ratio'],
            'corr_cycle12': s['corr'],
            'rmse_cycle12_hz': s['rmse'],
            'vp_cycle12': vp_mean,
            'cycle_sec': cycle_sec,
            'drop_first_sec': drop_first_sec,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Population-level time-scale metrics
# ---------------------------------------------------------------------------

def population_time_scale_metrics(
    offline,
    *,
    primary_key: str = 'currentBackgroundScale',
    cell_types: Optional[Iterable[str]] = None,
    minimum_n: int = 3,
    smooth_ms: float = 50.0,
) -> Dict:
    """Time-resolved population metrics comparing the two-level primary key.

    For each cell type, the response of cell ``c`` at time ``t`` under
    primary condition ``p`` is a single rate value (Hz); the population
    response is the vector across cells at that time. We then compare
    the two conditions at each time bin:

    - **Cohen's d** (per cell, then averaged across cells): ``(μ_p1 -
      μ_p2) / pooled_sd``. Time-resolved discriminability.
    - **Population Euclidean distance**: ``‖μ_p1(t) - μ_p2(t)‖`` over
      cells.
    - **Population cosine distance**: ``1 - cos(μ_p1(t), μ_p2(t))``.
    - **Cumulative response divergence**: cumulative integral of the
      mean absolute rate difference over time — turns small persistent
      effects into a monotonically growing signal.
    - **Trial-by-trial decoder score**: per time bin, the AUC of a
      one-feature classifier using a single cell's rate to distinguish
      conditions (best across cells, averaged across cells).

    Parameters
    ----------
    primary_key : str
        Condition column to compare (default ``currentBackgroundScale``).
        Currently expects exactly two unique values.
    smooth_ms : float
        Width of a uniform-kernel smoothing applied to the time-series
        metrics for plotting. Set to 0 to skip.

    Returns
    -------
    dict
        ``time_ms``, ``cell_types``, ``primary_values``, plus per
        cell-type sub-dicts with the metrics above as ``(n_time_bins,)``
        arrays.
    """
    df = offline.epochs
    if primary_key not in df.columns:
        raise KeyError(f'{primary_key!r} not in offline.epochs columns: {list(df.columns)}')
    primary_vals = sorted([v for v in df[primary_key].unique()
                           if not (isinstance(v, str) and v == '')])
    if len(primary_vals) < 2:
        raise ValueError(f'Need 2 levels of {primary_key!r}; got {primary_vals}')
    if len(primary_vals) > 2:
        primary_vals = primary_vals[:2]
    p1, p2 = primary_vals[0], primary_vals[1]
    e1 = df.loc[df[primary_key] == p1, 'epoch_index'].astype(int).to_numpy()
    e2 = df.loc[df[primary_key] == p2, 'epoch_index'].astype(int).to_numpy()

    df_cells = offline.cells
    if cell_types is not None:
        df_cells = df_cells.loc[df_cells['cell_type'].isin(list(cell_types))]
    counts = df_cells['cell_type'].value_counts()
    keep_types = [t for t, n in counts.items() if n >= minimum_n]

    time_ms = offline.psth_time_ms()

    out: Dict = {
        'time_ms': time_ms,
        'cell_types': keep_types,
        'primary_key': primary_key,
        'primary_values': [p1, p2],
        'smooth_ms': smooth_ms,
    }

    for ct in keep_types:
        cids = df_cells.loc[df_cells['cell_type'] == ct, 'cell_id'].astype(int).tolist()
        if not cids:
            continue
        # Stack: (n_cells, n_bins) means per primary value
        mu1 = []
        mu2 = []
        cohens_d_per_cell = []
        auc_per_cell = []
        for cid in cids:
            psth = offline.psth_matrix(cid)  # (n_epochs, n_bins)
            e1_in = e1[e1 < psth.shape[0]]
            e2_in = e2[e2 < psth.shape[0]]
            if e1_in.size == 0 or e2_in.size == 0:
                continue
            m1 = psth[e1_in].mean(axis=0)
            m2 = psth[e2_in].mean(axis=0)
            s1 = psth[e1_in].std(axis=0)
            s2 = psth[e2_in].std(axis=0)
            pooled = np.sqrt(0.5 * (s1 ** 2 + s2 ** 2)) + 1e-6
            mu1.append(m1)
            mu2.append(m2)
            cohens_d_per_cell.append((m1 - m2) / pooled)
            auc_per_cell.append(_per_bin_auc(psth[e1_in], psth[e2_in]))

        if not mu1:
            continue
        mu1 = np.stack(mu1)
        mu2 = np.stack(mu2)
        cohens_d = np.stack(cohens_d_per_cell)
        auc = np.stack(auc_per_cell)

        # Population vector distance / cosine
        diff = mu1 - mu2                          # (n_cells, n_bins)
        euclid = np.sqrt((diff ** 2).sum(axis=0))  # (n_bins,)
        num = (mu1 * mu2).sum(axis=0)
        denom = (np.linalg.norm(mu1, axis=0)
                 * np.linalg.norm(mu2, axis=0) + 1e-12)
        cosine_sim = num / denom
        cosine_dist = 1.0 - cosine_sim

        # Cumulative absolute difference (sum over cells, cumulative over time)
        bin_sec = (time_ms[1] - time_ms[0]) / 1000.0 if len(time_ms) > 1 else 1.0
        cum_div = np.cumsum(np.abs(diff).sum(axis=0)) * bin_sec

        # Smoothing for plot-ready curves
        s = _box_smooth(smooth_ms, time_ms)
        smooth = lambda y: (np.convolve(y, s, mode='same') if s.size > 1 else y)

        out[ct] = {
            'n_cells': int(mu1.shape[0]),
            'mu_p1': mu1, 'mu_p2': mu2,
            'cohens_d_mean': smooth(cohens_d.mean(axis=0)),
            'cohens_d_abs_mean': smooth(np.abs(cohens_d).mean(axis=0)),
            'pop_euclid_dist': smooth(euclid),
            'pop_cosine_dist': smooth(cosine_dist),
            'cum_abs_divergence': cum_div,
            'auc_mean': smooth(auc.mean(axis=0)),
            'auc_max': smooth(auc.max(axis=0)),
        }
    return out


def _box_smooth(width_ms: float, time_ms: np.ndarray) -> np.ndarray:
    """Unit-area boxcar of width ``width_ms``."""
    if width_ms <= 0 or len(time_ms) < 2:
        return np.array([1.0])
    bin_ms = float(time_ms[1] - time_ms[0])
    n = max(1, int(round(width_ms / bin_ms)))
    return np.ones(n) / n


def _per_bin_auc(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Per-time-bin Mann-Whitney AUC distinguishing two trial sets.

    Vectorized via rank statistics. ``x1`` is ``(n1, n_bins)``, ``x2``
    is ``(n2, n_bins)``. Returns ``(n_bins,)`` with AUC in [0, 1].
    """
    n1, nb = x1.shape
    n2 = x2.shape[0]
    if n1 == 0 or n2 == 0:
        return np.full(nb, np.nan)
    combined = np.concatenate([x1, x2], axis=0)  # (n1+n2, nb)
    # Rank along axis 0 per bin
    ranks = np.argsort(np.argsort(combined, axis=0), axis=0) + 1
    rank_sum_1 = ranks[:n1].sum(axis=0)
    u1 = rank_sum_1 - n1 * (n1 + 1) / 2.0
    auc = u1 / (n1 * n2)
    return auc


# ---------------------------------------------------------------------------
# Cross-date aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_psth_across_dates(
    offline_by_date: Dict[str, 'OfflineDataset'],
    *,
    cell_types: Optional[Iterable[str]] = None,
    condition_keys: Optional[Sequence[str]] = None,
    minimum_n: int = 3,
) -> Dict:
    """Pool per-cell mean PSTHs across dates for a population analysis.

    Returns dict shaped like :func:`analyze_offline` but pooling each
    cell type's (n_cells, n_bins) matrix across every date that has
    matching PSTH bin widths.

    Notes
    -----
    Requires every date's offline file to share the same ``time_ms``
    grid (same preTime/stimTime/tailTime and sample rate). Mismatched
    dates are dropped with a warning.
    """
    if not offline_by_date:
        return {'cell_types': [], 'psth': {}, 'conditions': []}

    # Pick the first date's time grid as canonical
    ref_exp, ref = next(iter(offline_by_date.items()))
    ref_time = ref.psth_time_ms()
    nb_ref = ref_time.size

    # Pool PSTHs per (type, condition)
    per_date_results = {}
    for exp, ds in offline_by_date.items():
        if ds.psth_time_ms().size != nb_ref:
            print(f'[aggregate_psth_across_dates] {exp}: PSTH grid mismatch — skipping')
            continue
        per_date_results[exp] = analyze_offline(
            ds, cell_types=cell_types, condition_keys=condition_keys,
            minimum_n=1,  # individual-date filter is too strict for pooling
        )

    # Discover the union of types / conditions
    all_types: List[str] = []
    all_conditions: List[Tuple] = []
    for r in per_date_results.values():
        for t in r['cell_types']:
            if t not in all_types:
                all_types.append(t)
        for c in r['conditions']:
            if c not in all_conditions:
                all_conditions.append(c)

    pooled: Dict[str, Dict[Tuple, np.ndarray]] = {ct: {} for ct in all_types}
    for ct in all_types:
        for cond in all_conditions:
            chunks = [r['psth'][ct][cond] for r in per_date_results.values()
                      if ct in r['psth'] and cond in r['psth'][ct]]
            if not chunks:
                continue
            pooled[ct][cond] = np.concatenate(chunks, axis=0)
        # Drop types failing the global cell-count threshold
        n_total = max((m.shape[0] for m in pooled[ct].values()), default=0)
        if n_total < minimum_n:
            pooled.pop(ct)

    kept_types = list(pooled.keys())

    return {
        'time_ms': ref_time,
        'condition_keys': per_date_results[ref_exp]['condition_keys'],
        'conditions': all_conditions,
        'cell_types': kept_types,
        'psth': pooled,
        'preTime_ms': per_date_results[ref_exp]['preTime_ms'],
        'stimTime_ms': per_date_results[ref_exp]['stimTime_ms'],
        'tailTime_ms': per_date_results[ref_exp]['tailTime_ms'],
        'condition_key': per_date_results[ref_exp]['condition_keys'][-1],
        'n_dates': len(per_date_results),
    }


# ---------------------------------------------------------------------------
# Per-date analysis-results persistence (so we don't redo the heavy lifting)
# ---------------------------------------------------------------------------

# CSV filenames live next to offline.h5 at <OUTPUT>/<exp>/<protocol>/
_SPIKE_DIST_CSV = 'spike_distance.csv'
_MOVIE_REPEAT_CSV = 'movie_repeat.csv'


def _analysis_dir(exp_name: str, protocol: str = 'eye_movement_alt_bg',
                  output_root: Optional[str] = None) -> Path:
    root = Path(output_root) if output_root else Path(OUTPUT_DIR)
    return root / exp_name / protocol


def save_spike_distance(df: pd.DataFrame, exp_name: str,
                        protocol: str = 'eye_movement_alt_bg',
                        output_root: Optional[str] = None) -> Path:
    """Persist :func:`spike_distance_analysis` output as CSV."""
    path = _analysis_dir(exp_name, protocol, output_root) / _SPIKE_DIST_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df['cond_i'] = df['cond_i'].apply(_tuple_to_str)
    df['cond_j'] = df['cond_j'].apply(_tuple_to_str)
    if 'group_key' in df.columns:
        df['group_key'] = df['group_key'].apply(_tuple_to_str)
    df.to_csv(path, index=False)
    return path


def save_movie_repeat(df: pd.DataFrame, exp_name: str,
                      protocol: str = 'eye_movement_alt_bg',
                      output_root: Optional[str] = None) -> Path:
    """Persist :func:`movie_repeat_analysis` output as CSV."""
    path = _analysis_dir(exp_name, protocol, output_root) / _MOVIE_REPEAT_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df['condition'] = df['condition'].apply(_tuple_to_str)
    df.to_csv(path, index=False)
    return path


def _tuple_to_str(t) -> str:
    if isinstance(t, tuple):
        return '|'.join('' if v is None else str(v) for v in t)
    return str(t)


def load_spike_distance_many(
    exp_names: Optional[Iterable[str]] = None,
    *,
    protocol: str = 'eye_movement_alt_bg',
    output_root: Optional[str] = None,
) -> pd.DataFrame:
    """Concat every available ``spike_distance.csv`` into a long DataFrame."""
    return _load_csv_many(exp_names, _SPIKE_DIST_CSV, protocol, output_root)


def load_movie_repeat_many(
    exp_names: Optional[Iterable[str]] = None,
    *,
    protocol: str = 'eye_movement_alt_bg',
    output_root: Optional[str] = None,
) -> pd.DataFrame:
    """Concat every available ``movie_repeat.csv`` into a long DataFrame."""
    return _load_csv_many(exp_names, _MOVIE_REPEAT_CSV, protocol, output_root)


def _load_csv_many(exp_names, basename, protocol, output_root):
    root = Path(output_root) if output_root else Path(OUTPUT_DIR)
    if exp_names is None:
        if not root.is_dir():
            return pd.DataFrame()
        exp_names = [p.name for p in sorted(root.iterdir()) if p.is_dir()]
    dfs = []
    for exp in exp_names:
        p = root / exp / protocol / basename
        if p.exists():
            df = pd.read_csv(p)
            if 'exp_name' not in df.columns:
                df['exp_name'] = exp
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def run_protocol_analyses(
    offline,
    *,
    protocol: str = 'eye_movement_alt_bg',
    output_root: Optional[str] = None,
    save: bool = True,
    spike_distance_kwargs: Optional[Dict] = None,
    movie_repeat_kwargs: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict:
    """Run VP + movie-repeat per date and (optionally) save CSVs.

    Returns
    -------
    dict
        ``{'spike_distance': DataFrame, 'movie_repeat': DataFrame}``.
        When ``save=True``, also writes the CSVs next to ``offline.h5``.
    """
    sd_kw = dict(spike_distance_kwargs or {})
    mr_kw = dict(movie_repeat_kwargs or {})

    if verbose:
        print(f'[{offline.exp_name}] spike_distance_analysis…')
    sd = spike_distance_analysis(offline, **sd_kw)
    if verbose:
        print(f'  → {len(sd)} cell × condition-pair rows')

    if verbose:
        print(f'[{offline.exp_name}] movie_repeat_analysis…')
    mr = movie_repeat_analysis(offline, **mr_kw)
    if verbose:
        print(f'  → {len(mr)} cell × condition rows')

    if save:
        sd_path = save_spike_distance(sd, offline.exp_name, protocol, output_root)
        mr_path = save_movie_repeat(mr, offline.exp_name, protocol, output_root)
        if verbose:
            print(f'  saved: {sd_path}\n         {mr_path}')

    return {'spike_distance': sd, 'movie_repeat': mr}
