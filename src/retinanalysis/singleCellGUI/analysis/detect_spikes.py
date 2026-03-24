"""Spike detection using k-means clustering (ported from Clarinet detectSpikes/MHT)."""

import param
import numpy as np
from retinanalysis.singleCellGUI.analysis._base import AnalysisPlugin


class DetectSpikes(AnalysisPlugin):
    name = param.String(default="Detect Spikes")
    description = param.String(default="Detect spikes using multiple hypothesis testing with k-means clustering")
    refractory_period = param.Number(default=1.5e-3, bounds=(0, 0.1), doc="Refractory period in seconds")
    search_window = param.Number(default=1.2e-3, bounds=(0, 0.1), doc="Search window for rebounds in seconds")

    def process(self, trace, sample_rate, **kwargs):
        spike_times = self._detect_spikes_mht(trace, sample_rate)
        result = np.zeros_like(trace)
        if len(spike_times) > 0:
            valid = spike_times[spike_times < len(trace)]
            result[valid] = 1.0
        return result

    def _detect_spikes_mht(self, trace, sample_rate):
        """Port of detectSpikesMHT from Clarinet."""
        refractory = int(self.refractory_period * sample_rate)
        search_win = int(self.search_window * sample_rate)

        # High-pass filter at 500 Hz
        data = self._highpass_filter(trace, 500, sample_rate)

        # Flip if positive peaks larger
        if abs(np.max(data)) > abs(np.min(data)):
            data = -data

        # Get negative peaks
        peak_times, peak_amps = self._get_peaks(data, direction=-1)
        if len(peak_times) < 3:
            return np.array([], dtype=int)

        # Only negative deflections
        neg_mask = peak_amps < 0
        peak_times = peak_times[neg_mask]
        peak_amps = np.abs(peak_amps[neg_mask])

        if len(peak_times) < 3:
            return np.array([], dtype=int)

        # Get rebounds
        left_reb, right_reb = self._get_rebounds(peak_times, data, search_win)

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

        # Spike cluster has larger peak amplitude
        spike_cluster = np.argmax(centroids[:, 0])
        spike_mask = labels == spike_cluster

        spike_t = peak_times[spike_mask]
        spike_a = peak_amps[spike_mask]
        nonspike_a = peak_amps[~spike_mask]

        # Check signal-to-noise
        if len(nonspike_a) == 0 or np.std(nonspike_a) == 0:
            return spike_t
        sig_f = (np.mean(spike_a) - np.mean(nonspike_a)) / np.std(nonspike_a)
        if sig_f < 5:
            return np.array([], dtype=int)

        return spike_t

    @staticmethod
    def _highpass_filter(data, cutoff_hz, sample_rate):
        n = len(data)
        freq_step = sample_rate / n
        keep_pts = int(round(cutoff_hz / freq_step))
        fft_data = np.fft.fft(data)
        fft_data[:keep_pts] = 0
        fft_data[-keep_pts:] = 0
        return np.real(np.fft.ifft(fft_data))

    @staticmethod
    def _get_peaks(data, direction=-1):
        d = np.diff(np.sign(np.diff(data)))
        if direction > 0:
            idx = np.where(d < 0)[0] + 1
        else:
            idx = np.where(d > 0)[0] + 1
        return idx, data[idx]

    @staticmethod
    def _get_rebounds(peak_times, trace, search_window):
        left = np.zeros(len(peak_times))
        right = np.zeros(len(peak_times))
        hw = max(1, search_window // 2)

        for i, pt in enumerate(peak_times):
            start = max(0, pt - hw)
            end_ = min(len(trace), pt + hw)

            # Look for positive rebounds around negative peaks
            seg_left = trace[start:pt + 1]
            seg_right = trace[pt:end_]

            d_left = np.diff(np.sign(np.diff(seg_left)))
            peaks_left = np.where(d_left < 0)[0] + 1
            d_right = np.diff(np.sign(np.diff(seg_right)))
            peaks_right = np.where(d_right < 0)[0] + 1

            left[i] = seg_left[peaks_left[0]] if len(peaks_left) > 0 else 0
            right[i] = seg_right[peaks_right[0]] if len(peaks_right) > 0 else 0

        return left, right

    def get_label(self):
        return "Spike detection"
