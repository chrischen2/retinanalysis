import numpy as np

import retinanalysis.utils.spike_detector as spike_detector
from retinanalysis.utils.spike_detector import get_peaks, get_rebounds


def _loop_rebounds(peaks_ind, trace, search_interval):
    """Reference implementation used before rebound lookup was vectorized."""
    peaks = trace[peaks_ind]
    rebounds = {'Left': np.zeros_like(peaks), 'Right': np.zeros_like(peaks)}
    for index, peak in enumerate(peaks):
        start = max(0, peaks_ind[index] - round(search_interval / 2))
        end = min(peaks_ind[index] + round(search_interval / 2), len(trace) - 1)
        direction = 1 if peak < 0 else -1
        left, _ = get_peaks(trace[start:peaks_ind[index]], direction)
        right, _ = get_peaks(trace[peaks_ind[index]:end], direction)
        rebounds['Left'][index] = left[0] if left.size else 0
        rebounds['Right'][index] = right[0] if right.size else 0
    return rebounds


def test_get_rebounds_matches_per_peak_reference():
    rng = np.random.default_rng(42)
    traces = [
        rng.normal(size=1000),
        np.sin(np.linspace(0, 30 * np.pi, 1000)) + rng.normal(scale=0.1, size=1000),
        np.repeat(rng.normal(size=250), 4),
    ]
    for trace in traces:
        _, minima = get_peaks(trace, -1)
        _, maxima = get_peaks(trace, 1)
        peaks = np.sort(np.concatenate((minima[::3], maxima[::4])))
        peaks = peaks[trace[peaks] != 0]
        for search_interval in (0, 1, 2, 5, 12, 13, 100):
            expected = _loop_rebounds(peaks, trace, search_interval)
            actual = get_rebounds(peaks, trace, search_interval)
            np.testing.assert_array_equal(actual['Left'], expected['Left'])
            np.testing.assert_array_equal(actual['Right'], expected['Right'])


def test_get_rebounds_handles_empty_candidates():
    result = get_rebounds(np.array([], dtype=int), np.arange(10.0), 12)
    assert result['Left'].shape == (0,)
    assert result['Right'].shape == (0,)


def test_detector_removes_moving_median_before_high_pass(monkeypatch):
    """The default 100-sample moving median removes slow baseline offsets."""
    trace = np.r_[np.full(150, 4.0), np.full(150, 40.0)]
    captured = {}

    def capture_high_pass(values, cutoff, interval):
        captured['values'] = np.asarray(values).copy()
        return np.zeros_like(values, dtype=float)

    monkeypatch.setattr(spike_detector, 'high_pass_filter', capture_high_pass)
    spike_detector.detector(trace, sample_rate=1000.0)

    detrended = captured['values'][0]
    np.testing.assert_allclose(detrended[:100], 0.0)
    np.testing.assert_allclose(detrended[-100:], 0.0)
    assert np.max(np.abs(detrended)) <= 36.0


def test_detector_default_median_window_is_100_samples(monkeypatch):
    captured = {}
    def capture_median(values, window):
        captured['window'] = window
        return np.zeros_like(values)
    monkeypatch.setattr(spike_detector, 'moving_median', capture_median)
    monkeypatch.setattr(spike_detector, 'high_pass_filter',
                        lambda values, cutoff, interval: np.zeros_like(values))
    spike_detector.detector(np.zeros(200), sample_rate=10_000.0)
    assert captured == {'window': 100}


def test_moving_median_matches_matlab_even_window_and_shrinking_endpoints():
    traces = np.vstack([np.arange(10.), np.arange(10.)[::-1]])
    expected = np.array([[.5, 1., 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.],
                         [8.5, 8., 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 1.]])
    np.testing.assert_array_equal(spike_detector.moving_median(traces, 4), expected)
    for window in (3, 100):
        half = window // 2
        expected = np.array([[np.median(row[max(0, i-half):min(len(row), i+window-half)])
                              for i in range(len(row))] for row in traces])
        np.testing.assert_array_equal(spike_detector.moving_median(traces, window), expected)


def test_detector_consumes_shared_preprocessed_trace(monkeypatch):
    captured = {}

    def preprocess(values, sample_rate, median_window_samples, cutoff_frequency):
        captured.update({
            'values': np.asarray(values).copy(),
            'sample_rate': sample_rate,
            'median_window_samples': median_window_samples,
            'cutoff_frequency': cutoff_frequency,
        })
        return np.zeros((1, np.asarray(values).size), dtype=float)

    monkeypatch.setattr(spike_detector, 'preprocess_spike_traces', preprocess)
    source = np.arange(20.0)
    spike_detector.detector(
        source, sample_rate=2000.0, median_window_samples=10,
        cutoff_frequency=250.0)

    np.testing.assert_array_equal(captured.pop('values'), source)
    assert captured == {
        'sample_rate': 2000.0,
        'median_window_samples': 10,
        'cutoff_frequency': 250.0,
    }


def test_detector_can_disable_moving_median(monkeypatch):
    trace = np.arange(20.0)
    captured = {}

    def capture_high_pass(values, cutoff, interval):
        captured['values'] = np.asarray(values).copy()
        return np.zeros_like(values, dtype=float)

    monkeypatch.setattr(spike_detector, 'high_pass_filter', capture_high_pass)
    spike_detector.detector(trace, sample_rate=1000.0,
                            median_window_samples=None)

    np.testing.assert_array_equal(captured['values'][0], trace)


def test_moving_median_prevents_baseline_step_false_spikes():
    trace = np.random.default_rng(1).normal(0.0, 0.3, 20_000)
    trace[10_000:] += 100.0

    uncorrected, _, _ = spike_detector.detector(
        trace, sample_rate=10_000.0, median_window_samples=None)
    corrected, _, _ = spike_detector.detector(trace, sample_rate=10_000.0)

    assert len(uncorrected[0]) > 0
    assert len(corrected[0]) == 0


def test_pooled_clustering_rejects_small_events_promoted_by_empty_epochs(monkeypatch):
    """A shared boundary rejects contamination that wins a local two-cluster fit."""
    traces = np.zeros((3, 40), dtype=float)
    # Epochs 0/1 contain real spikes (30), contamination (12), and noise (2).
    # Epoch 2 lacks real spikes; a local fit promotes its contamination to the
    # high-amplitude cluster, whereas the pooled fit retains the 30-unit scale.
    traces[0, [5, 13, 21, 29]] = [-30.0, -12.0, -4.0, -2.0]
    traces[1, [7, 15, 23, 31]] = [-30.0, -12.0, -4.0, -2.0]
    traces[2, [9, 19, 29]] = [-12.0, -4.0, -2.0]

    monkeypatch.setattr(
        spike_detector, 'high_pass_filter',
        lambda values, cutoff, interval: np.asarray(values, dtype=float))
    fit_sizes = []

    def relative_amplitude_fit(features, n_clusters, verbose=False):
        fit_sizes.append(len(features))
        is_spike = features[:, 0] > 0.75 * np.max(features[:, 0])
        labels = np.where(is_spike, 0, 1)
        return labels, is_spike

    monkeypatch.setattr(spike_detector, 'fit_kmeans', relative_amplitude_fit)

    local_times, _, _ = spike_detector.detector(
        traces, sample_rate=1000.0, min_peak_amplitude=10.0,
        max_trial_length_s=5.0, cluster_across_trials=False)
    local_fit_sizes = fit_sizes.copy()
    fit_sizes.clear()
    pooled_times, _, _ = spike_detector.detector(
        traces, sample_rate=1000.0, min_peak_amplitude=10.0,
        max_trial_length_s=5.0, cluster_across_trials=True)

    assert len(local_fit_sizes) == 3
    assert len(local_times[2]) == 1       # contamination promoted locally
    assert fit_sizes == [11]              # one fit using every candidate
    assert [len(times) for times in pooled_times] == [1, 1, 0]


def test_whole_epoch_clustering_preserves_sparse_spikes_and_rejects_quiet_noise():
    """Quiet seconds must not redefine the spike class or veto true spikes."""
    rng = np.random.default_rng(42)
    trace = rng.normal(0, 1, 40000)
    expected = np.arange(1000, 9000, 300)
    x = np.arange(-20, 21)
    waveform = -20 * np.exp(-(x / 2) ** 2) + 5 * np.exp(-((x - 6) / 3) ** 2)
    for sample in expected:
        trace[sample - 20:sample + 21] += waveform
    detected, _, _ = spike_detector.detector(trace, max_trial_length_s=None)
    assert len(detected[0]) == len(expected)
    np.testing.assert_allclose(detected[0], expected, atol=2, rtol=0)
    assert not np.any(detected[0] >= 10000)
