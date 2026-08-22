import numpy as np

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
