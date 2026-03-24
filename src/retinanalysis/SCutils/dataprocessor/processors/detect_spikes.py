"""Spike detection using k-means clustering (ported from Clarinet +builtinProcessors/detectSpikes)."""

import numpy as np


def detect_spikes(data, sample_rate, refractory_period=1.5e-3,
                  search_window=1.0e-3):
    """Detect spikes using multiple hypothesis testing with k-means clustering.

    The algorithm high-pass filters the trace, finds negative peaks, measures
    rebound amplitudes on each side, then clusters peaks into spike vs.
    non-spike using 2-means on [amplitude, left_rebound, right_rebound].
    A signal-to-noise check (> 5 std) must pass for spikes to be returned.

    Parameters
    ----------
    data : np.ndarray
        1-D amplitude array.
    sample_rate : float
        Sampling rate in Hz.
    refractory_period : float
        Refractory period in seconds (default 1.5 ms).
    search_window : float
        Window in seconds to look for rebounds around each peak (default 1.0 ms).

    Returns
    -------
    np.ndarray
        Array of spike-time sample indices (0-indexed). Empty if no spikes
        pass the SNR criterion.
    """
    search_win = int(search_window * sample_rate)

    # High-pass filter at 500 Hz
    filtered = _highpass_filter(data, 500, sample_rate)

    # Flip if positive peaks are larger than negative
    if abs(np.max(filtered)) > abs(np.min(filtered)):
        filtered = -filtered

    # Get negative peaks
    peak_times, peak_amps = _get_peaks(filtered, direction=-1)
    if len(peak_times) < 3:
        return np.array([], dtype=int)

    # Keep only negative deflections
    neg_mask = peak_amps < 0
    peak_times = peak_times[neg_mask]
    peak_amps = np.abs(peak_amps[neg_mask])

    if len(peak_times) < 3:
        return np.array([], dtype=int)

    # Measure rebounds
    left_reb, right_reb = _get_rebounds(peak_times, filtered, search_win)

    # K-means clustering (2 clusters)
    features = np.column_stack([peak_amps, left_reb, right_reb])
    try:
        from sklearn.cluster import KMeans

        init = np.array([
            [np.median(peak_amps), np.median(left_reb), np.median(right_reb)],
            [np.max(peak_amps), np.max(left_reb), np.max(right_reb)],
        ])
        km = KMeans(n_clusters=2, init=init, n_init=1, max_iter=10000)
        labels = km.fit_predict(features)
        centroids = km.cluster_centers_
    except Exception:
        return np.array([], dtype=int)

    # Spike cluster has the larger mean peak amplitude
    spike_cluster = np.argmax(centroids[:, 0])
    spike_mask = labels == spike_cluster

    spike_t = peak_times[spike_mask]
    spike_a = peak_amps[spike_mask]
    nonspike_a = peak_amps[~spike_mask]

    # Signal-to-noise check
    if len(nonspike_a) == 0 or np.std(nonspike_a) == 0:
        return spike_t
    snr = (np.mean(spike_a) - np.mean(nonspike_a)) / np.std(nonspike_a)
    if snr < 5:
        return np.array([], dtype=int)

    return spike_t


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _highpass_filter(data, cutoff_hz, sample_rate):
    """Zero-out low-frequency FFT bins below *cutoff_hz*."""
    n = len(data)
    freq_step = sample_rate / n
    keep_pts = int(round(cutoff_hz / freq_step))
    fft_data = np.fft.fft(data)
    fft_data[:keep_pts] = 0
    fft_data[-keep_pts:] = 0
    return np.real(np.fft.ifft(fft_data))


def _get_peaks(data, direction=-1):
    """Find local peaks using the second derivative of the sign of diff."""
    d = np.diff(np.sign(np.diff(data)))
    if direction > 0:
        idx = np.where(d < 0)[0] + 1
    else:
        idx = np.where(d > 0)[0] + 1
    return idx, data[idx]


def _get_rebounds(peak_times, trace, search_window):
    """Measure positive rebound amplitudes on each side of each peak."""
    left = np.zeros(len(peak_times))
    right = np.zeros(len(peak_times))
    hw = max(1, search_window // 2)

    for i, pt in enumerate(peak_times):
        start = max(0, pt - hw)
        end_ = min(len(trace), pt + hw)

        seg_left = trace[start:pt + 1]
        seg_right = trace[pt:end_]

        d_left = np.diff(np.sign(np.diff(seg_left)))
        peaks_left = np.where(d_left < 0)[0] + 1
        d_right = np.diff(np.sign(np.diff(seg_right)))
        peaks_right = np.where(d_right < 0)[0] + 1

        left[i] = seg_left[peaks_left[0]] if len(peaks_left) > 0 else 0
        right[i] = seg_right[peaks_right[0]] if len(peaks_right) > 0 else 0

    return left, right
