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
