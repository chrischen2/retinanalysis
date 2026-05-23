"""Analysis for ``monitorVariableMeanNoiseEpochs`` (1-D temporal noise).

The protocol delivers full-field Gaussian-contrast noise around a mean
intensity that alternates (or randomly cycles, when ``numel(meanIntensity)
> 2``) across epochs. Each epoch logs ``currentMean``, ``noiseSeed``,
and ``intensityOverFrame`` (a per-frame intensity trace, length
``ceil(stimTime * frameRate / frameDwell)``).

Canonical analysis (port of ``analyzeVariableMeanNoiseMonitor.m``):

1. Group epochs by ``currentMean``.
2. Reconstruct each epoch's per-sample stimulus from
   ``intensityOverFrame`` (linearly interpolated to the PSTH sample
   rate).
3. For each (cell, mean) pair fit a linear filter via FFT-based
   reverse-correlation (cascadegraph ``compute_filter``) and a sigmoid
   nonlinearity to the binned generator-vs-response cloud
   (cascadegraph ``sample_nl`` + ``SigmoidNlNode.fit_to_sample``).
4. Optional "switching" mode: restrict to epochs preceded by a *different*
   mean, so the fit reflects transient adaptation immediately after a
   mean step (``lowToHigh`` / ``highToLow`` …).
5. Optional "phase-split" mode: cut each epoch into N equal-time slices
   and refit per slice to recover the within-epoch adaptation timecourse.

Module exposes the same surface as :mod:`retinanalysis.protocols.eye_movement_alt_bg`:

* :func:`analyze`, :func:`analyze_offline`, :func:`plot_psth_by_condition`
  — per-(cell-type × mean) PSTH summary.
* :func:`ln_fit_per_condition`, :func:`ln_fit_switching`,
  :func:`ln_fit_split_phases` — LN models.
* :func:`plot_ln_models` — filter + NL side-by-side.
* :func:`aggregate_ln_across_dates` — cross-date pooling.
* :func:`save_ln_fits` / :func:`load_ln_fits_many` — CSV persistence of
  summary stats (filter peak, peak-time, sigmoid params).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cascadegraph as cg

from retinanalysis.utils.psth import epoch_spikes_to_psth, psth_time_axis
from retinanalysis.utils.style import colors_for_conditions
from retinanalysis.config.settings import OUTPUT_DIR


# Both lab variants of the same protocol class produce identical params.
PROTOCOL_NAMES = (
    'edu.washington.riekelab.chris.protocols.monitorVariableMeanNoiseEpochs',
    'edu.washington.riekelab.vyom.protocols.monitorVariableMeanNoiseEpochs',
)
PROTOCOL_NAME = PROTOCOL_NAMES[0]  # convention used by sibling modules

DEFAULT_CONDITION_KEYS = ['currentMean']
DEFAULT_CONDITION_KEY = DEFAULT_CONDITION_KEYS[-1]
PROTOCOL_SHORT = 'one_d_noise'

# Stage refresh rate. ``intensityOverFrame`` is sampled at
# ``frameRate / frameDwell``. The protocol comment hard-codes 60 Hz.
DEFAULT_FRAME_RATE_HZ = 60.0


# =========================================================================
# PSTH-by-condition (mirrors eye_movement_alt_bg.analyze / analyze_offline)
# =========================================================================

def _resolve_keys(condition_keys, condition_key) -> List[str]:
    if condition_keys is not None:
        return list(condition_keys)
    if condition_key is not None:
        return [condition_key]
    return list(DEFAULT_CONDITION_KEYS)


def _clean_value(v):
    if isinstance(v, bytes):
        v = v.decode('utf-8')
    if isinstance(v, str) and v == '':
        return None
    return v


def analyze(
    pipeline,
    cell_types: Iterable[str],
    condition_keys: Optional[Sequence[str]] = None,
    condition_key: Optional[str] = None,
    psth_sigma_ms: float = 10.0,
    sample_rate_hz: float = 1000.0,
    minimum_n: int = 3,
) -> Dict:
    """Per-(cell-type × ``currentMean``) mean PSTHs from a live pipeline."""
    sb = pipeline.stim
    rb = pipeline.resp
    bp = sb.d_epoch_block_params or {}
    pre_ms = float(bp.get('preTime', 0))
    stim_ms = float(bp.get('stimTime', 0))
    tail_ms = float(bp.get('tailTime', 0))
    t_end_ms = pre_ms + stim_ms + tail_ms

    keys = _resolve_keys(condition_keys, condition_key)
    for k in keys:
        if k not in sb.df_epochs.columns:
            sb.df_epochs[k] = [p.get(k) for p in sb.df_epochs['epoch_parameters']]

    epoch_idx = sb.df_epochs['epoch_index'].astype(int).tolist()
    per_epoch_tuples = list(zip(*[sb.df_epochs[k].tolist() for k in keys]))
    idx2tuple = dict(zip(epoch_idx, per_epoch_tuples))
    conditions = sorted(
        set(t for t in per_epoch_tuples if all(v is not None for v in t)),
        key=lambda c: tuple(0 if v is None else v for v in c),
    )

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
                by_cond[cond] = np.stack(per_cell_psth)
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
        'condition_key': keys[-1],
        'psth_sigma_ms': psth_sigma_ms,
    }


def _resolve_condition_keys_from_offline(offline, condition_keys):
    if condition_keys is not None:
        return list(condition_keys)
    keys = offline.meta.get('condition_keys')
    if keys is None:
        return list(DEFAULT_CONDITION_KEYS)
    if isinstance(keys, np.ndarray):
        keys = keys.tolist()
    return [str(k) for k in keys if k]


def _epoch_indices_by_condition(offline, condition_keys):
    df = offline.epochs
    out = {}
    for cond, sub in df.groupby(list(condition_keys), sort=True):
        if not isinstance(cond, tuple):
            cond = (cond,)
        cond = tuple(_clean_value(v) for v in cond)
        out[cond] = sub['epoch_index'].astype(int).to_numpy()
    return out


def analyze_offline(
    offline,
    cell_types: Optional[Iterable[str]] = None,
    condition_keys: Optional[Sequence[str]] = None,
    minimum_n: int = 3,
) -> Dict:
    """Offline equivalent of :func:`analyze` — operates on an HDF5 store."""
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
            psth = offline.psth_matrix(cid)
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


def _format_cond_tuple(cond, keys):
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
    """One panel per cell type; ``currentMean`` overlaid by color."""
    types = results['cell_types']
    conditions = results['conditions']
    time_ms = results['time_ms']
    keys = results.get('condition_keys', [results.get('condition_key')])
    pre = results['preTime_ms']
    stim = results['stimTime_ms']

    nrows = int(np.ceil(max(len(types), 1) / ncols))
    if axes is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 3.0 * nrows),
                                 sharex=True, squeeze=False)
    axes = np.atleast_2d(axes)
    flat_axes = axes.flatten()
    plain_conds = [c[0] if isinstance(c, tuple) else c for c in conditions]
    cond_colors = colors_for_conditions(plain_conds)

    for i, ct in enumerate(types):
        ax = flat_axes[i]
        for cond, plain in zip(conditions, plain_conds):
            mat = results['psth'].get(ct, {}).get(cond)
            if mat is None or mat.size == 0:
                continue
            color = cond_colors[plain]
            if show_individual_cells:
                for row in mat:
                    ax.plot(time_ms, row, color=color,
                            alpha=individual_alpha, linewidth=0.6)
            mean = mat.mean(axis=0)
            sem = mat.std(axis=0) / np.sqrt(max(mat.shape[0], 1))
            ax.plot(time_ms, mean, color=color, linewidth=1.4,
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


# =========================================================================
# Stimulus reconstruction
# =========================================================================

def _stim_window_bin_range(offline) -> Tuple[int, int, float]:
    """Slice indices into the PSTH covering preTime → preTime+stimTime.

    Returns ``(start, end, bin_ms)``.
    """
    time_ms = offline.psth_time_ms()
    bin_ms = float(time_ms[1] - time_ms[0]) if len(time_ms) > 1 else 1.0
    pre_ms = float(offline.timing.get('preTime_ms', 0))
    stim_ms = float(offline.timing.get('stimTime_ms', 0))
    start = int(round(pre_ms / bin_ms))
    end = int(round((pre_ms + stim_ms) / bin_ms))
    return start, end, bin_ms


def _frame_dwell(offline) -> int:
    fd = offline.epoch_block_params.get('frameDwell', 1)
    try:
        return max(1, int(fd))
    except (TypeError, ValueError):
        return 1


def reconstruct_epoch_stim(
    offline,
    epoch_index: int,
    n_samples: int,
    *,
    sample_rate_hz: Optional[float] = None,
    frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
) -> np.ndarray:
    """Reconstruct one epoch's per-sample stimulus from ``intensityOverFrame``.

    The protocol stores one intensity value per *update* (a group of
    ``frameDwell`` monitor frames). We linearly interpolate from those
    update times onto the PSTH sample grid so stim and response share an
    index.
    """
    if sample_rate_hz is None:
        sample_rate_hz = float(offline.timing.get('sample_rate_hz', 1000.0))
    if 'intensityOverFrame' not in offline.epoch_array_keys:
        raise KeyError(
            "Offline file is missing per-epoch 'intensityOverFrame' arrays — "
            "rebuild the offline store (load_or_build_offline(..., overwrite=True)) "
            "so the array stash is populated."
        )
    iof = offline.epoch_array('intensityOverFrame', epoch_idx=epoch_index)
    fd = _frame_dwell(offline)
    update_rate = float(frame_rate_hz) / float(fd)
    stim_dt = 1.0 / sample_rate_hz
    t_resp = np.arange(n_samples) * stim_dt
    t_stim = np.arange(iof.size) / update_rate
    if iof.size < 2:
        return np.full(n_samples, iof[0] if iof.size else 0.0)
    return np.interp(t_resp, t_stim, iof, left=iof[0], right=iof[-1])


def _build_stim_response_matrices(
    offline,
    cell_id: int,
    epoch_indices: Sequence[int],
    sample_rate_hz: Optional[float] = None,
    frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
    frequency_cutoff_hz: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Per-epoch (stim, response) matrices restricted to the stim window.

    Optionally low-pass the response (mirrors MATLAB
    ``applyFrequencyCutoff``). Returns (stim, response, sampling_interval).
    """
    start, end, bin_ms = _stim_window_bin_range(offline)
    n_samples = end - start
    if n_samples <= 1:
        raise ValueError(
            f'PSTH window too short: start={start} end={end} (bin={bin_ms} ms).'
        )

    psth = offline.psth_matrix(cell_id)
    sel = np.asarray([i for i in epoch_indices if i < psth.shape[0]],
                     dtype=int)
    if sel.size == 0:
        raise ValueError(f'Cell {cell_id}: no epochs selected.')

    resp = psth[sel, start:end].astype(np.float64)
    sampling_interval = bin_ms / 1000.0

    if sample_rate_hz is None:
        sample_rate_hz = 1.0 / sampling_interval

    stim_rows = []
    for ei in sel:
        s = reconstruct_epoch_stim(
            offline, int(ei), n_samples,
            sample_rate_hz=sample_rate_hz,
            frame_rate_hz=frame_rate_hz,
        )
        # Mean-subtract within epoch so the filter captures contrast
        # variations and not the (epoch-specific) DC level — matches the
        # MATLAB pipeline's reverse-correlation convention.
        stim_rows.append(s - s.mean())
    stim = np.stack(stim_rows)

    if frequency_cutoff_hz is not None:
        resp = cg.apply_frequency_cutoff(resp, frequency_cutoff_hz,
                                         sampling_interval)

    return stim, resp, sampling_interval


# =========================================================================
# LN model fits
# =========================================================================

# Default knobs (mirror MATLAB SETTINGS in analyzeVariableMeanNoiseMonitor.m)
LN_FILTER_MS_DEFAULT = 1000.0
LN_FREQ_CUTOFF_HZ_DEFAULT = 15.0 / 2.0
LN_NUM_BINS_DEFAULT = 50
LN_BIN_TYPE_DEFAULT = 'equalN'


def _fit_ln_for_window(
    stim: np.ndarray,
    resp: np.ndarray,
    sampling_interval: float,
    *,
    filter_ms: float = LN_FILTER_MS_DEFAULT,
    frequency_cutoff_hz: float = LN_FREQ_CUTOFF_HZ_DEFAULT,
    num_bins: int = LN_NUM_BINS_DEFAULT,
    bin_type: str = LN_BIN_TYPE_DEFAULT,
    correct_stim_power: bool = True,
    use_anticausal: bool = False,
) -> Dict:
    """Fit one LN model (filter + sigmoid NL) to one (stim, resp) batch."""
    filter_pts = max(1, int(round((filter_ms / 1000.0) / sampling_interval)))
    filter_pts = min(filter_pts, stim.shape[1] // 2)

    filt_causal, filt_anti = cg.compute_filter(
        stim, resp, filter_pts,
        correct_stim_power=correct_stim_power,
        frequency_cutoff=frequency_cutoff_hz,
        sampling_interval=sampling_interval,
    )
    filt = (np.concatenate([filt_causal, filt_anti])
            if use_anticausal else filt_causal)

    generator = cg.convolve_filter_with_stim(filt, stim, use_anticausal)

    # sample_nl/equalN requires divisibility — trim the last samples.
    if bin_type == 'equalN':
        n_total = generator.size
        usable = (n_total // num_bins) * num_bins
        if usable < num_bins:
            raise ValueError(
                f'Not enough samples ({n_total}) for {num_bins} equalN bins.'
            )
        g = generator.ravel()[:usable]
        r = resp.ravel()[:usable]
        nl_x, nl_y = cg.sample_nl(g, r, num_bins, bin_type)
    else:
        nl_x, nl_y = cg.sample_nl(generator, resp, num_bins, bin_type)

    sig = cg.SigmoidNlNode()
    sig_params = sig.fit_to_sample(nl_x, nl_y)

    filt_time_s = np.arange(1, filt.size + 1) * sampling_interval
    # Filter peak (signed amplitude that drives the strongest response).
    peak_idx = int(np.argmax(np.abs(filt)))
    return {
        'filter': filt,
        'filter_time_s': filt_time_s,
        'filter_peak': float(filt[peak_idx]),
        'filter_peak_time_s': float(filt_time_s[peak_idx]),
        'nl_x': nl_x,
        'nl_y': nl_y,
        'sigmoid_params': {
            'alpha': float(sig_params[0]),
            'beta': float(sig_params[1]),
            'gamma': float(sig_params[2]),
            'epsilon': float(sig_params[3]),
        },
        'sigmoid_fn': sig.process,
        'sampling_interval': sampling_interval,
        'n_epochs': int(stim.shape[0]),
    }


def _switching_epoch_indices(
    offline,
    *,
    condition_keys: Optional[Sequence[str]] = None,
) -> Dict[Tuple, np.ndarray]:
    """Group epoch indices by ``(prev_mean → current_mean)`` transitions.

    Mirrors the MATLAB ``[intLegends{i} 'To' intLegends{j}]`` logic.
    Returns ``{(prev, current): np.array(epoch_indices)}``; keys with
    ``prev == current`` correspond to "no switch" (steady-state inside a
    run of the same mean).
    """
    keys = _resolve_condition_keys_from_offline(offline, condition_keys)
    df = offline.epochs.sort_values('epoch_index').reset_index(drop=True)
    if not keys:
        return {}
    primary = keys[0]
    if primary not in df.columns:
        df[primary] = np.nan
    means = df[primary].to_numpy()
    epoch_idx = df['epoch_index'].to_numpy().astype(int)

    out: Dict[Tuple, List[int]] = {}
    for i in range(1, len(means)):
        prev, curr = means[i - 1], means[i]
        out.setdefault((prev, curr), []).append(int(epoch_idx[i]))
    return {k: np.array(v, dtype=int) for k, v in out.items()}


def ln_fit_per_condition(
    offline,
    *,
    cell_types: Optional[Iterable[str]] = None,
    minimum_n: int = 1,
    condition_keys: Optional[Sequence[str]] = None,
    filter_ms: float = LN_FILTER_MS_DEFAULT,
    frequency_cutoff_hz: float = LN_FREQ_CUTOFF_HZ_DEFAULT,
    num_bins: int = LN_NUM_BINS_DEFAULT,
    bin_type: str = LN_BIN_TYPE_DEFAULT,
    correct_stim_power: bool = True,
    use_anticausal: bool = False,
    frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
    verbose: bool = False,
) -> Dict:
    """Fit one LN model per (cell × ``currentMean``) condition.

    Returns
    -------
    dict
        ``{
            'mode': 'per_condition',
            'condition_keys': [...],
            'cell_types': [...],
            'conditions': [(mean1,), (mean2,), ...],
            'fits': {cell_type: {cell_id: {cond: fit_dict, ...}}},
            'sampling_interval': float,
          }``
        Each ``fit_dict`` is the output of :func:`_fit_ln_for_window`.
    """
    keys = _resolve_condition_keys_from_offline(offline, condition_keys)
    cond_to_epochs = _epoch_indices_by_condition(offline, keys)
    conditions = sorted(cond_to_epochs.keys(),
                        key=lambda c: tuple(0 if v is None else v for v in c))

    df_cells = offline.cells
    if cell_types is not None:
        df_cells = df_cells.loc[df_cells['cell_type'].isin(list(cell_types))]
    counts = df_cells['cell_type'].value_counts()
    kept_types = [t for t, n in counts.items() if n >= minimum_n]

    sampling_interval_ref = None
    fits: Dict[str, Dict[int, Dict[Tuple, Dict]]] = {}
    for ct in kept_types:
        cids = df_cells.loc[df_cells['cell_type'] == ct, 'cell_id'].astype(int).tolist()
        per_cell: Dict[int, Dict[Tuple, Dict]] = {}
        for cid in cids:
            per_cond: Dict[Tuple, Dict] = {}
            for cond in conditions:
                ep = cond_to_epochs[cond]
                if ep.size == 0:
                    continue
                try:
                    stim, resp, dt = _build_stim_response_matrices(
                        offline, cid, ep,
                        sample_rate_hz=float(offline.timing.get('sample_rate_hz', 1000.0)),
                        frame_rate_hz=frame_rate_hz,
                        frequency_cutoff_hz=frequency_cutoff_hz,
                    )
                    fit = _fit_ln_for_window(
                        stim, resp, dt,
                        filter_ms=filter_ms,
                        frequency_cutoff_hz=frequency_cutoff_hz,
                        num_bins=num_bins, bin_type=bin_type,
                        correct_stim_power=correct_stim_power,
                        use_anticausal=use_anticausal,
                    )
                except Exception as exc:
                    if verbose:
                        print(f'  [{ct} cell {cid} cond={cond}] LN fit failed: {exc!r}')
                    continue
                per_cond[cond] = fit
                sampling_interval_ref = fit['sampling_interval']
            if per_cond:
                per_cell[cid] = per_cond
        fits[ct] = per_cell

    return {
        'mode': 'per_condition',
        'condition_keys': keys,
        'cell_types': kept_types,
        'conditions': conditions,
        'fits': fits,
        'sampling_interval': sampling_interval_ref,
    }


def ln_fit_switching(
    offline,
    *,
    target_condition,
    cell_types: Optional[Iterable[str]] = None,
    minimum_n: int = 1,
    condition_keys: Optional[Sequence[str]] = None,
    filter_ms: float = LN_FILTER_MS_DEFAULT,
    frequency_cutoff_hz: float = LN_FREQ_CUTOFF_HZ_DEFAULT,
    num_bins: int = LN_NUM_BINS_DEFAULT,
    bin_type: str = LN_BIN_TYPE_DEFAULT,
    correct_stim_power: bool = True,
    use_anticausal: bool = False,
    frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
    verbose: bool = False,
) -> Dict:
    """LN fit restricted to epochs that landed on ``target_condition``
    *after* a different mean (the "switching" view in the MATLAB script).

    Bucket key is ``(prev_mean → target_condition)``; each bucket gets
    one LN fit per cell. ``target_condition`` can be a scalar (e.g. the
    "high" mean value) — every distinct ``prev_mean`` produces a bucket.
    """
    keys = _resolve_condition_keys_from_offline(offline, condition_keys)
    transitions = _switching_epoch_indices(offline, condition_keys=keys)
    # Keep only transitions ending at target_condition (≠ prev).
    relevant = {(prev, curr): idx for (prev, curr), idx in transitions.items()
                if curr == target_condition and prev != curr}

    df_cells = offline.cells
    if cell_types is not None:
        df_cells = df_cells.loc[df_cells['cell_type'].isin(list(cell_types))]
    counts = df_cells['cell_type'].value_counts()
    kept_types = [t for t, n in counts.items() if n >= minimum_n]

    sampling_interval_ref = None
    fits: Dict[str, Dict[int, Dict[Tuple, Dict]]] = {}
    for ct in kept_types:
        cids = df_cells.loc[df_cells['cell_type'] == ct, 'cell_id'].astype(int).tolist()
        per_cell: Dict[int, Dict[Tuple, Dict]] = {}
        for cid in cids:
            per_trans: Dict[Tuple, Dict] = {}
            for trans, ep in relevant.items():
                try:
                    stim, resp, dt = _build_stim_response_matrices(
                        offline, cid, ep,
                        sample_rate_hz=float(offline.timing.get('sample_rate_hz', 1000.0)),
                        frame_rate_hz=frame_rate_hz,
                        frequency_cutoff_hz=frequency_cutoff_hz,
                    )
                    fit = _fit_ln_for_window(
                        stim, resp, dt,
                        filter_ms=filter_ms,
                        frequency_cutoff_hz=frequency_cutoff_hz,
                        num_bins=num_bins, bin_type=bin_type,
                        correct_stim_power=correct_stim_power,
                        use_anticausal=use_anticausal,
                    )
                except Exception as exc:
                    if verbose:
                        print(f'  [{ct} cell {cid} trans={trans}] LN fit failed: {exc!r}')
                    continue
                per_trans[trans] = fit
                sampling_interval_ref = fit['sampling_interval']
            if per_trans:
                per_cell[cid] = per_trans
        fits[ct] = per_cell

    return {
        'mode': 'switching',
        'target_condition': target_condition,
        'condition_keys': keys,
        'transitions': sorted(relevant.keys()),
        'cell_types': kept_types,
        'fits': fits,
        'sampling_interval': sampling_interval_ref,
    }


def ln_fit_split_phases(
    offline,
    *,
    condition,
    n_phases: int = 5,
    cell_types: Optional[Iterable[str]] = None,
    minimum_n: int = 1,
    condition_keys: Optional[Sequence[str]] = None,
    filter_ms: float = LN_FILTER_MS_DEFAULT,
    frequency_cutoff_hz: float = LN_FREQ_CUTOFF_HZ_DEFAULT,
    num_bins: int = LN_NUM_BINS_DEFAULT,
    bin_type: str = LN_BIN_TYPE_DEFAULT,
    correct_stim_power: bool = True,
    use_anticausal: bool = False,
    frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
    verbose: bool = False,
) -> Dict:
    """Slice each epoch's stim window into ``n_phases`` equal time chunks
    and refit one LN model per slice — the within-epoch adaptation
    timecourse from the MATLAB script.

    ``condition`` is one of the tuples returned by :func:`analyze_offline`
    (e.g. ``(0.65,)`` for the "high" mean) or a scalar for the
    common single-key case.
    """
    keys = _resolve_condition_keys_from_offline(offline, condition_keys)
    cond_to_epochs = _epoch_indices_by_condition(offline, keys)
    cond_tup = condition if isinstance(condition, tuple) else (condition,)
    if cond_tup not in cond_to_epochs:
        raise KeyError(
            f'Condition {cond_tup} not found in offline file. '
            f'Available: {list(cond_to_epochs)}')
    ep_idx = cond_to_epochs[cond_tup]
    if ep_idx.size == 0:
        raise ValueError(f'No epochs for condition {cond_tup}.')

    df_cells = offline.cells
    if cell_types is not None:
        df_cells = df_cells.loc[df_cells['cell_type'].isin(list(cell_types))]
    counts = df_cells['cell_type'].value_counts()
    kept_types = [t for t, n in counts.items() if n >= minimum_n]

    start, end, bin_ms = _stim_window_bin_range(offline)
    n_samples = end - start
    phase_len = n_samples // n_phases
    if phase_len < 2:
        raise ValueError(
            f'Phase length {phase_len} samples too short for {n_phases} phases.')
    # Inclusive slice boundaries [phase_start, phase_end) over the
    # stim-window-relative index.
    slices = [(i * phase_len,
               (i + 1) * phase_len if i < n_phases - 1 else n_samples)
              for i in range(n_phases)]

    sampling_interval_ref = None
    fits: Dict[str, Dict[int, List[Optional[Dict]]]] = {}
    for ct in kept_types:
        cids = df_cells.loc[df_cells['cell_type'] == ct, 'cell_id'].astype(int).tolist()
        per_cell: Dict[int, List[Optional[Dict]]] = {}
        for cid in cids:
            try:
                stim, resp, dt = _build_stim_response_matrices(
                    offline, cid, ep_idx,
                    sample_rate_hz=float(offline.timing.get('sample_rate_hz', 1000.0)),
                    frame_rate_hz=frame_rate_hz,
                    frequency_cutoff_hz=frequency_cutoff_hz,
                )
            except Exception as exc:
                if verbose:
                    print(f'  [{ct} cell {cid}] window build failed: {exc!r}')
                continue
            per_phase: List[Optional[Dict]] = []
            for s0, s1 in slices:
                try:
                    fit = _fit_ln_for_window(
                        stim[:, s0:s1], resp[:, s0:s1], dt,
                        filter_ms=filter_ms,
                        frequency_cutoff_hz=frequency_cutoff_hz,
                        num_bins=num_bins, bin_type=bin_type,
                        correct_stim_power=correct_stim_power,
                        use_anticausal=use_anticausal,
                    )
                except Exception as exc:
                    if verbose:
                        print(f'  [{ct} cell {cid} phase=({s0},{s1})] LN fit failed: {exc!r}')
                    per_phase.append(None)
                    continue
                per_phase.append(fit)
                sampling_interval_ref = fit['sampling_interval']
            per_cell[cid] = per_phase
        fits[ct] = per_cell

    return {
        'mode': 'split_phases',
        'condition': cond_tup,
        'condition_keys': keys,
        'n_phases': n_phases,
        'phase_slices': slices,
        'phase_window_ms': [(s0 * bin_ms, s1 * bin_ms) for s0, s1 in slices],
        'cell_types': kept_types,
        'fits': fits,
        'sampling_interval': sampling_interval_ref,
    }


# =========================================================================
# LN plotting
# =========================================================================

def _cond_label(cond) -> str:
    if isinstance(cond, tuple):
        return ', '.join(f'{v:g}' if isinstance(v, (int, float)) else str(v)
                         for v in cond)
    return f'{cond:g}' if isinstance(cond, (int, float)) else str(cond)


def plot_ln_models(
    result: Dict,
    cell_type: Optional[str] = None,
    cell_id: Optional[int] = None,
    *,
    average_across_cells: bool = True,
    normalize_nl_x: bool = True,
    ax_filter=None,
    ax_nl=None,
) -> Tuple:
    """Plot filter (left) + sigmoid NL (right) for an LN-fit result.

    - ``mode='per_condition'``: one trace per ``currentMean`` value.
    - ``mode='switching'``: one trace per ``(prev → curr)`` transition.
    - ``mode='split_phases'``: one trace per within-epoch phase (p1 … pN).

    ``average_across_cells=True`` averages filters/NLs across every cell
    of the chosen type; ``False`` (with ``cell_id``) plots a single cell.
    """
    if cell_type is None:
        cell_type = result['cell_types'][0]
    if cell_type not in result['fits'] or not result['fits'][cell_type]:
        raise ValueError(f'No fits for cell_type={cell_type!r}.')

    if ax_filter is None or ax_nl is None:
        fig, (ax_filter, ax_nl) = plt.subplots(1, 2, figsize=(11, 4))

    mode = result['mode']
    if mode == 'per_condition':
        bucket_keys = result['conditions']
        bucket_labels = [_cond_label(c) for c in bucket_keys]
    elif mode == 'switching':
        bucket_keys = list(result['transitions'])
        bucket_labels = [f'{_cond_label(prev)}→{_cond_label(curr)}'
                         for (prev, curr) in bucket_keys]
    elif mode == 'split_phases':
        bucket_keys = list(range(result['n_phases']))
        bucket_labels = [f'p{i+1}' for i in bucket_keys]
    else:
        raise ValueError(f'Unknown result mode: {mode}')

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(bucket_keys), 1)))

    for k, label, color in zip(bucket_keys, bucket_labels, colors):
        # Collect (filter, nl_x, nl_y) for the chosen cells.
        per_cell_filters, per_cell_nls, t_filt = [], [], None
        cell_iter = ([cell_id] if cell_id is not None
                     else list(result['fits'][cell_type].keys()))
        for cid in cell_iter:
            cd = result['fits'][cell_type].get(cid)
            if cd is None:
                continue
            if mode == 'split_phases':
                fit = cd[k] if isinstance(cd, list) and k < len(cd) else None
            else:
                fit = cd.get(k)
            if fit is None:
                continue
            per_cell_filters.append(fit['filter'])
            per_cell_nls.append((fit['nl_x'], fit['nl_y'], fit['sigmoid_fn']))
            t_filt = fit['filter_time_s']
        if not per_cell_filters:
            continue

        if average_across_cells and len(per_cell_filters) > 1:
            # Normalize each cell's filter then average for shape comparison
            normed = [f / (np.max(np.abs(f)) + 1e-12) for f in per_cell_filters]
            filt_mean = np.mean(np.stack(normed), axis=0)
            ax_filter.plot(t_filt, filt_mean, color=color, lw=1.6,
                           label=f'{label}  (n={len(per_cell_filters)})')
        else:
            for f in per_cell_filters:
                ax_filter.plot(t_filt, f / (np.max(np.abs(f)) + 1e-12),
                               color=color, lw=1.2, alpha=0.7,
                               label=label if f is per_cell_filters[0] else None)

        # NL: scatter sample, line the sigmoid fit. Average across cells
        # by pooling the (x, sigmoid(x)) curves on a common x grid.
        nl_xs = [nlx for nlx, _, _ in per_cell_nls]
        if normalize_nl_x:
            nl_xs_n = [x / (np.max(np.abs(x)) + 1e-12) for x in nl_xs]
        else:
            nl_xs_n = nl_xs
        for (x_raw, x_n, (_, nly, _)) in zip(nl_xs, nl_xs_n, per_cell_nls):
            ax_nl.plot(x_n, nly, 'o', color=color, markersize=2.5,
                       alpha=0.4)
        if average_across_cells and len(per_cell_nls) > 1:
            grid = np.linspace(np.min([x.min() for x in nl_xs]),
                               np.max([x.max() for x in nl_xs]), 200)
            ys = np.stack([fit_fn(grid) for (_, _, fit_fn) in per_cell_nls])
            if normalize_nl_x:
                grid_plot = grid / (np.max(np.abs(grid)) + 1e-12)
            else:
                grid_plot = grid
            ax_nl.plot(grid_plot, ys.mean(axis=0), color=color, lw=1.6,
                       label=f'{label}')
        else:
            for (x_raw, x_n, (_, _, fit_fn)) in zip(nl_xs, nl_xs_n, per_cell_nls):
                xs_fit = np.linspace(x_raw.min(), x_raw.max(), 200)
                ax_nl.plot(xs_fit / (np.max(np.abs(xs_fit)) + 1e-12)
                           if normalize_nl_x else xs_fit,
                           fit_fn(xs_fit), color=color, lw=1.5,
                           label=label)

    ax_filter.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax_filter.set_xlabel('time (s)')
    ax_filter.set_ylabel('filter (normalized)')
    title_bits = [f'{cell_type}']
    if mode == 'switching':
        title_bits.append(f'→ target={_cond_label(result["target_condition"])}')
    elif mode == 'split_phases':
        title_bits.append(f'cond={_cond_label(result["condition"])}, '
                          f'{result["n_phases"]} phases')
    ax_filter.set_title('filter | ' + ' '.join(title_bits))
    ax_filter.legend(fontsize=8, loc='best')

    ax_nl.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax_nl.axvline(0, color='gray', lw=0.5, alpha=0.5)
    ax_nl.set_xlabel('filtered stimulus' +
                     (' (normalized)' if normalize_nl_x else ''))
    ax_nl.set_ylabel('rate (Hz)')
    ax_nl.set_title('nonlinearity')
    ax_nl.legend(fontsize=8, loc='best')

    if hasattr(ax_filter, 'figure'):
        ax_filter.figure.tight_layout()
    return ax_filter, ax_nl


# =========================================================================
# Cross-date aggregation
# =========================================================================

def aggregate_psth_across_dates(
    offline_by_date: Dict[str, 'OfflineDataset'],
    *,
    cell_types: Optional[Iterable[str]] = None,
    condition_keys: Optional[Sequence[str]] = None,
    minimum_n: int = 3,
) -> Dict:
    """Pool per-cell mean PSTHs across dates (same shape as ``analyze_offline``)."""
    if not offline_by_date:
        return {'cell_types': [], 'psth': {}, 'conditions': []}

    ref_exp, ref = next(iter(offline_by_date.items()))
    ref_time = ref.psth_time_ms()

    per_date = {}
    for exp, ds in offline_by_date.items():
        if ds.psth_time_ms().size != ref_time.size:
            print(f'[aggregate_psth_across_dates] {exp}: PSTH grid mismatch — skipping')
            continue
        per_date[exp] = analyze_offline(
            ds, cell_types=cell_types, condition_keys=condition_keys,
            minimum_n=1,
        )

    all_types: List[str] = []
    all_conditions: List[Tuple] = []
    for r in per_date.values():
        for t in r['cell_types']:
            if t not in all_types:
                all_types.append(t)
        for c in r['conditions']:
            if c not in all_conditions:
                all_conditions.append(c)

    pooled: Dict[str, Dict[Tuple, np.ndarray]] = {ct: {} for ct in all_types}
    for ct in all_types:
        for cond in all_conditions:
            chunks = [r['psth'][ct][cond] for r in per_date.values()
                      if ct in r['psth'] and cond in r['psth'][ct]]
            if not chunks:
                continue
            pooled[ct][cond] = np.concatenate(chunks, axis=0)
        n_total = max((m.shape[0] for m in pooled[ct].values()), default=0)
        if n_total < minimum_n:
            pooled.pop(ct)

    return {
        'time_ms': ref_time,
        'condition_keys': per_date[ref_exp]['condition_keys'],
        'conditions': all_conditions,
        'cell_types': list(pooled.keys()),
        'psth': pooled,
        'preTime_ms': per_date[ref_exp]['preTime_ms'],
        'stimTime_ms': per_date[ref_exp]['stimTime_ms'],
        'tailTime_ms': per_date[ref_exp]['tailTime_ms'],
        'condition_key': per_date[ref_exp]['condition_keys'][-1],
        'n_dates': len(per_date),
    }


def aggregate_ln_across_dates(
    offline_by_date: Dict[str, 'OfflineDataset'],
    *,
    cell_types: Optional[Iterable[str]] = None,
    **fit_kwargs,
) -> Dict:
    """Run :func:`ln_fit_per_condition` per date, pool by (cell_type × cond).

    Each cell becomes one row in ``filters[cell_type][cond]`` (n_cells ×
    n_filter_pts) and one row in ``sigmoid_params[cell_type][cond]``
    (n_cells × 4) — pooled across every date in ``offline_by_date``.
    """
    pooled_filters: Dict[str, Dict[Tuple, List[np.ndarray]]] = {}
    pooled_params: Dict[str, Dict[Tuple, List[np.ndarray]]] = {}
    pooled_meta: Dict[str, Dict[Tuple, List[Tuple]]] = {}
    sampling_interval = None
    filter_time_s = None

    for exp, ds in offline_by_date.items():
        try:
            res = ln_fit_per_condition(
                ds, cell_types=cell_types, **fit_kwargs)
        except Exception as exc:
            print(f'[aggregate_ln_across_dates] {exp}: FAILED — {exc!r}')
            continue
        if sampling_interval is None:
            sampling_interval = res['sampling_interval']
        for ct, per_cell in res['fits'].items():
            pooled_filters.setdefault(ct, {})
            pooled_params.setdefault(ct, {})
            pooled_meta.setdefault(ct, {})
            for cid, per_cond in per_cell.items():
                for cond, fit in per_cond.items():
                    if filter_time_s is None:
                        filter_time_s = fit['filter_time_s']
                    pooled_filters[ct].setdefault(cond, []).append(fit['filter'])
                    sp = fit['sigmoid_params']
                    pooled_params[ct].setdefault(cond, []).append(
                        np.array([sp['alpha'], sp['beta'], sp['gamma'], sp['epsilon']]))
                    pooled_meta[ct].setdefault(cond, []).append((exp, cid))

    return {
        'mode': 'aggregate_ln',
        'cell_types': list(pooled_filters.keys()),
        'filter_time_s': filter_time_s,
        'sampling_interval': sampling_interval,
        # Per (ct, cond): stacked arrays. Filters kept on their original
        # scale — caller can normalize for shape vs. amplitude analyses.
        'filters': {ct: {c: np.stack(v) for c, v in d.items()}
                    for ct, d in pooled_filters.items()},
        'sigmoid_params': {ct: {c: np.stack(v) for c, v in d.items()}
                           for ct, d in pooled_params.items()},
        'meta': {ct: {c: list(v) for c, v in d.items()}
                 for ct, d in pooled_meta.items()},
    }


# =========================================================================
# CSV persistence (per-date analysis results)
# =========================================================================

_LN_FITS_CSV = 'ln_fits.csv'


def _ln_summary_rows(result: Dict, exp_name: str) -> List[Dict]:
    """Flatten a :func:`ln_fit_per_condition` result into per-row summaries."""
    rows = []
    mode = result['mode']
    for ct, per_cell in result.get('fits', {}).items():
        for cid, per_cond in per_cell.items():
            iterable = (per_cond.items() if isinstance(per_cond, dict)
                        else enumerate(per_cond))
            for cond, fit in iterable:
                if fit is None:
                    continue
                sp = fit['sigmoid_params']
                rows.append({
                    'exp_name': exp_name,
                    'cell_type': ct,
                    'cell_id': int(cid),
                    'mode': mode,
                    'condition': _cond_label(cond),
                    'n_epochs': fit['n_epochs'],
                    'filter_peak': fit['filter_peak'],
                    'filter_peak_time_s': fit['filter_peak_time_s'],
                    'sigmoid_alpha': sp['alpha'],
                    'sigmoid_beta': sp['beta'],
                    'sigmoid_gamma': sp['gamma'],
                    'sigmoid_epsilon': sp['epsilon'],
                })
    return rows


def save_ln_fits(
    result: Dict,
    exp_name: str,
    *,
    protocol: str = PROTOCOL_SHORT,
    output_root: Optional[str] = None,
) -> Path:
    """Persist a :func:`ln_fit_per_condition` summary as CSV next to ``offline.h5``."""
    root = Path(output_root) if output_root else Path(OUTPUT_DIR)
    out_path = root / exp_name / protocol / _LN_FITS_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(_ln_summary_rows(result, exp_name))
    df.to_csv(out_path, index=False)
    return out_path


def load_ln_fits_many(
    exp_names: Optional[Iterable[str]] = None,
    *,
    protocol: str = PROTOCOL_SHORT,
    output_root: Optional[str] = None,
) -> pd.DataFrame:
    """Concat every available ``ln_fits.csv`` across dates."""
    root = Path(output_root) if output_root else Path(OUTPUT_DIR)
    if exp_names is None:
        if not root.is_dir():
            return pd.DataFrame()
        exp_names = [p.name for p in sorted(root.iterdir()) if p.is_dir()]
    dfs = []
    for exp in exp_names:
        p = root / exp / protocol / _LN_FITS_CSV
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
    protocol: str = PROTOCOL_SHORT,
    output_root: Optional[str] = None,
    save: bool = True,
    ln_fit_kwargs: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict:
    """Run the per-condition LN fit and (optionally) persist the summary CSV.

    Returns ``{'ln_fits': <result dict>}``.
    """
    kw = dict(ln_fit_kwargs or {})
    if verbose:
        print(f'[{offline.exp_name}] ln_fit_per_condition…')
    res = ln_fit_per_condition(offline, **kw)
    n_rows = sum(len(c) for ct in res['fits'].values() for c in ct.values())
    if verbose:
        print(f'  → {n_rows} cell × condition LN fits')
    if save:
        path = save_ln_fits(res, offline.exp_name, protocol=protocol,
                            output_root=output_root)
        if verbose:
            print(f'  saved: {path}')
    return {'ln_fits': res}
