"""PSTH and Gaussian-kernel spike-rate utilities.

Matches the MATLAB convention in
``spatialIntegration/analysis/utils/spikeTimeToPSTH.m``: a Gaussian kernel
with sigma in milliseconds, convolved with a binary spike train sampled
at ``sample_rate_hz``, then scaled by ``sample_rate_hz`` so the output is
in spikes/s. Defaults (``psth_sigma_ms=10``, ``sample_rate_hz=1000``)
mirror the single-cell analyses in that package.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Sequence


def gaussian_filter_1d(sigma_samples: float) -> np.ndarray:
    """Mirror ``gaussFilter1D.m``: x = -5*sigma..5*sigma, area = 1."""
    n = int(round(5 * sigma_samples))
    x = np.arange(-n, n + 1, dtype=float)
    amp = np.exp(-x ** 2 / (2 * sigma_samples ** 2))
    amp /= amp.sum()
    return amp


def spike_times_to_psth(
    spike_times_ms: np.ndarray,
    t_end_ms: float,
    psth_sigma_ms: float = 10.0,
    sample_rate_hz: float = 1000.0,
    t_start_ms: float = 0.0,
) -> np.ndarray:
    """Convolve a single epoch's spike-time list with a Gaussian → rate (Hz).

    Returns a ``(n_bins,)`` array where ``n_bins = round((t_end_ms -
    t_start_ms) / 1000 * sample_rate_hz)``. Bin width is
    ``1000 / sample_rate_hz`` ms; bin ``k`` is centered at
    ``t_start_ms + (k + 0.5) * bin_width_ms``.

    Spikes outside ``[t_start_ms, t_end_ms]`` are silently dropped.
    """
    dur_ms = float(t_end_ms - t_start_ms)
    n_bins = int(round(dur_ms / 1000.0 * sample_rate_hz))
    if n_bins <= 0:
        return np.zeros(0)

    arr = np.asarray(spike_times_ms, dtype=float)
    arr = arr[(arr >= t_start_ms) & (arr < t_end_ms)]
    if arr.size:
        idx = np.floor((arr - t_start_ms) / 1000.0 * sample_rate_hz).astype(int)
        idx = np.clip(idx, 0, n_bins - 1)
    else:
        idx = np.array([], dtype=int)

    spike_binary = np.zeros(n_bins, dtype=float)
    if idx.size:
        # Increment in case of multiple spikes in one bin
        np.add.at(spike_binary, idx, 1.0)

    sigma_samples = float(psth_sigma_ms) / 1000.0 * sample_rate_hz
    kernel = gaussian_filter_1d(sigma_samples)
    return sample_rate_hz * np.convolve(spike_binary, kernel, mode='same')


def epoch_spikes_to_psth(
    spike_times_by_epoch: Sequence[np.ndarray],
    t_end_ms: float,
    psth_sigma_ms: float = 10.0,
    sample_rate_hz: float = 1000.0,
    t_start_ms: float = 0.0,
) -> np.ndarray:
    """Stack per-epoch PSTHs → ``(n_epochs, n_bins)`` in Hz.

    ``spike_times_by_epoch`` is what's stored in
    ``response_block.df_spike_times.spike_times`` (a list of 1-D ms arrays,
    one per epoch).
    """
    return np.stack([
        spike_times_to_psth(s, t_end_ms, psth_sigma_ms, sample_rate_hz, t_start_ms)
        for s in spike_times_by_epoch
    ])


def psth_time_axis(
    t_end_ms: float,
    sample_rate_hz: float = 1000.0,
    t_start_ms: float = 0.0,
) -> np.ndarray:
    """Return bin-center times (ms) matching :func:`spike_times_to_psth`."""
    dur_ms = float(t_end_ms - t_start_ms)
    n_bins = int(round(dur_ms / 1000.0 * sample_rate_hz))
    bin_ms = 1000.0 / sample_rate_hz
    return t_start_ms + (np.arange(n_bins) + 0.5) * bin_ms


def check_psth_timing(rb_or_pipeline, verbose: bool = True) -> dict:
    """Diagnose PSTH alignment: TTL onset vs nominal preTime vs first stim frame.

    Spike times in ``df_spike_times`` are stored relative to the TTL
    trigger (sample 0 of each epoch). The PSTH plotted by
    ``plot_psth_by_condition`` puts a dashed vertical line at
    ``preTime_ms`` to mark stimulus onset. But the projector / stage
    typically doesn't display the first stim frame *exactly* at
    ``preTime`` — there's a 1-2 frame offset captured in
    ``d_timing['actual_onset_times_ms']`` (the timestamp of frame
    index ``floor(preTime * frameRate)`` in the rig's frame log).

    This function reports the median per-epoch offset
    ``actual_onset_ms - preTime_ms``. A small *positive* number
    (~17-33 ms at 60 Hz) means the response peaks should appear
    SLIGHTLY RIGHT of the dashed line — which is normal. A *negative*
    or large positive offset is unusual and warrants checking the
    rig clock / frame-monitor sample rate.

    Returns ``{'pre_time_ms', 'actual_onset_ms_median',
    'offset_ms_median', 'offset_frames', 'n_epochs',
    'stage_frame_rate', 'note'}``.
    """
    if hasattr(rb_or_pipeline, 'resp'):
        rb = rb_or_pipeline.resp
    else:
        rb = rb_or_pipeline
    t = rb.d_timing
    pre = float(t.get('pre_time_ms', 0.0))
    actual = t.get('actual_onset_times_ms', None)
    fr = t.get('stage_frame_rate', None)
    if actual is None or len(actual) == 0:
        if verbose:
            print(f'check_psth_timing: no actual_onset_times_ms recorded '
                  f'(frame-monitor data missing). preTime_ms={pre}.')
        return {'pre_time_ms': pre, 'actual_onset_ms_median': None,
                'offset_ms_median': None, 'offset_frames': None,
                'n_epochs': 0, 'stage_frame_rate': fr,
                'note': 'actual onsets missing'}
    actual_arr = np.asarray(actual, dtype=float)
    med = float(np.median(actual_arr))
    offset = med - pre
    offset_frames = (offset * fr / 1000.0) if (fr is not None and fr > 0) else None
    if verbose:
        print(f'PSTH alignment diagnostic:')
        print(f'  Symphony preTime_ms          : {pre:.2f}')
        print(f'  actual stim onset (median)   : {med:.2f} ms')
        print(f'  offset (actual − preTime)    : {offset:+.2f} ms'
              + (f'  (~{offset_frames:+.2f} frames @ {fr:.1f} Hz)'
                  if offset_frames is not None else ''))
        if abs(offset) < 50:
            print(f'  → small projector pipeline delay, normal.')
        elif offset > 0:
            print(f'  → actual onset is > 50 ms LATER than preTime. '
                  f'Check frame_times_ms and preTime decoding.')
        else:
            print(f'  → actual onset is BEFORE preTime — unusual. '
                  f'Either preTime was incorrectly stored, or the '
                  f'frame-monitor / TTL alignment is off.')
    return {'pre_time_ms': pre, 'actual_onset_ms_median': med,
            'offset_ms_median': offset, 'offset_frames': offset_frames,
            'n_epochs': int(actual_arr.size), 'stage_frame_rate': fr,
            'note': 'ok' if abs(offset) < 50 else 'large offset'}
