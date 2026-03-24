"""Peri-Stimulus Time Histogram (ported from Clarinet +builtinExtractors/measurePSTH)."""

import numpy as np


def measure_psth(spike_times_list, duration, bin_width=0.01,
                 smoothing_window=0.0):
    """Compute a peri-stimulus time histogram from spike times.

    Parameters
    ----------
    spike_times_list : list[np.ndarray]
        List of spike-time arrays (one per epoch), in **seconds** relative
        to stimulus onset.
    duration : float
        Total epoch duration in seconds.
    bin_width : float
        Histogram bin width in seconds (default 0.01).
    smoothing_window : float
        Width of Gaussian smoothing window in seconds (default 0 = none).

    Returns
    -------
    bin_centers : np.ndarray
        Time axis in seconds (centre of each bin).
    firing_rate : np.ndarray
        Mean firing rate in Hz for each bin.
    """
    bins = np.arange(0, duration + bin_width, bin_width)

    # Pool all spike times across epochs
    all_spikes = np.concatenate(spike_times_list) if spike_times_list else np.array([])
    count = np.histogram(all_spikes, bins=bins)[0].astype(float)

    # Optional Gaussian smoothing
    if smoothing_window > 0:
        win_bins = max(1, int(round(smoothing_window / bin_width)))
        if win_bins > 1:
            w = _gausswin(win_bins)
            w = w / np.sum(w)
            count = np.convolve(count, w, mode="same")

    n_epochs = max(1, len(spike_times_list))
    firing_rate = count / (n_epochs * bin_width)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return bin_centers, firing_rate


def _gausswin(n):
    """Gaussian window matching MATLAB's gausswin (alpha=2.5)."""
    alpha = 2.5
    half = (n - 1) / 2
    t = np.arange(n) - half
    return np.exp(-0.5 * (alpha * t / half) ** 2) if half > 0 else np.ones(n)
