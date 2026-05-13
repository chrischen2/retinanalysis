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

from retinanalysis.utils.psth import epoch_spikes_to_psth, psth_time_axis
from retinanalysis.utils.style import colors_for_conditions

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
