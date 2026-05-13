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

PROTOCOL_NAME = 'edu.washington.riekelab.turner.protocols.EyeMovementTrajectoryAlternatingBackground'

# Default key to split conditions by. Override with `condition_key=` to
# split by image, NDF, etc. — anything in df_epochs columns.
DEFAULT_CONDITION_KEY = 'currentBackgroundScale'


def _epoch_total_ms(stim_block) -> float:
    bp = stim_block.d_epoch_block_params or {}
    return float(bp.get('preTime', 0) + bp.get('stimTime', 0) + bp.get('tailTime', 0))


def analyze(
    pipeline,
    cell_types: Iterable[str],
    condition_key: str = DEFAULT_CONDITION_KEY,
    psth_sigma_ms: float = 10.0,
    sample_rate_hz: float = 1000.0,
    minimum_n: int = 3,
) -> Dict:
    """Compute per-(cell-type × condition) mean PSTHs.

    Returns
    -------
    dict with keys:
        ``time_ms``: ``(n_bins,)`` bin centers in ms from epoch onset.
        ``conditions``: sorted list of condition values.
        ``cell_types``: list of types that passed ``minimum_n``.
        ``psth``: dict ``{cell_type: dict{condition: (n_cells, n_bins) }}``
            — per-cell mean PSTH (mean across that cell's epochs in this
            condition).
        ``preTime_ms, stimTime_ms, tailTime_ms``: protocol params.
        ``condition_key``: which df_epochs column was used to split.
    """
    sb = pipeline.stim
    rb = pipeline.resp
    bp = sb.d_epoch_block_params or {}
    pre_ms = float(bp.get('preTime', 0))
    stim_ms = float(bp.get('stimTime', 0))
    tail_ms = float(bp.get('tailTime', 0))
    t_end_ms = pre_ms + stim_ms + tail_ms

    if condition_key not in sb.df_epochs.columns:
        # Try to pull from epoch_parameters per-epoch dict.
        sb.df_epochs[condition_key] = [
            p.get(condition_key) for p in sb.df_epochs['epoch_parameters']
        ]

    # Map epoch_index → condition value
    epoch_cond = (sb.df_epochs.set_index('epoch_index')[condition_key]
                  .to_dict())
    conditions = sorted(set(v for v in epoch_cond.values() if v is not None))

    # Filter cells to requested types with enough representation
    type_counts = rb.df_spike_times['cell_type'].value_counts()
    cell_types = [t for t in cell_types if type_counts.get(t, 0) >= minimum_n]

    psth_by_type: Dict[str, Dict] = {}
    for ct in cell_types:
        df_ct = rb.df_spike_times.query('cell_type == @ct')
        by_cond: Dict[float, np.ndarray] = {}
        for cond in conditions:
            cond_epoch_idxs = [
                idx for idx, c in epoch_cond.items() if c == cond
            ]
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
        'conditions': conditions,
        'cell_types': cell_types,
        'psth': psth_by_type,
        'preTime_ms': pre_ms,
        'stimTime_ms': stim_ms,
        'tailTime_ms': tail_ms,
        'condition_key': condition_key,
        'psth_sigma_ms': psth_sigma_ms,
    }


def plot_psth_by_condition(
    results: Dict,
    axes: Optional[np.ndarray] = None,
    ncols: int = 2,
    show_individual_cells: bool = False,
    individual_alpha: float = 0.2,
) -> np.ndarray:
    """Plot the analyze() result — one panel per cell type, conditions overlaid.

    Returns the axes array.
    """
    types = results['cell_types']
    conditions = results['conditions']
    time_ms = results['time_ms']

    nrows = int(np.ceil(len(types) / ncols))
    if axes is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 3.0 * nrows),
                                 sharex=True, squeeze=False)
    axes = np.atleast_2d(axes)
    flat_axes = axes.flatten()

    for i, ct in enumerate(types):
        ax = flat_axes[i]
        for ci, cond in enumerate(conditions):
            mat = results['psth'].get(ct, {}).get(cond)
            if mat is None or mat.size == 0:
                continue
            color = f'C{ci}'
            if show_individual_cells:
                for row in mat:
                    ax.plot(time_ms, row, color=color,
                            alpha=individual_alpha, linewidth=0.6)
            mean = mat.mean(axis=0)
            sem = mat.std(axis=0) / np.sqrt(max(mat.shape[0], 1))
            ax.plot(time_ms, mean, color=color, linewidth=1.6,
                    label=f'{results["condition_key"]}={cond}  (n={mat.shape[0]})')
            ax.fill_between(time_ms, mean - sem, mean + sem,
                            color=color, alpha=0.2, linewidth=0)
        # Stim onset/offset markers
        pre = results['preTime_ms']
        stim = results['stimTime_ms']
        if stim > 0:
            ax.axvline(pre, color='red', lw=0.6, ls='--', alpha=0.7)
            ax.axvline(pre + stim, color='red', lw=0.6, ls='--', alpha=0.7)
        ax.set_title(ct)
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('rate (Hz)')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.7)

    # Hide unused axes
    for j in range(len(types), len(flat_axes)):
        flat_axes[j].axis('off')

    if axes is not None and hasattr(axes[0, 0], 'figure'):
        axes[0, 0].figure.tight_layout()
    return axes
