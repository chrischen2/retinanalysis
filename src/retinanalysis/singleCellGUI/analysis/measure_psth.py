"""Peri-Stimulus Time Histogram (ported from Clarinet measurePSTH).

This is an extractor that operates on groups of epochs, computing
firing rate histograms from spike times.
"""

import param
import numpy as np
from retinanalysis.singleCellGUI.analysis._base import AnalysisPlugin


class MeasurePSTH(AnalysisPlugin):
    name = param.String(default="PSTH")
    description = param.String(default="Peri-stimulus time histogram from spike times")
    bin_width = param.Number(default=0.01, bounds=(0.001, 1.0), doc="Bin width in seconds")
    smoothing_window = param.Number(default=0, bounds=(0, 1.0), doc="Gaussian smoothing window (s). 0=no smoothing")

    def process(self, trace, sample_rate, **kwargs):
        # Single-trace: return spike-based histogram
        spike_times = kwargs.get('spike_times', None)
        pre_time_ms = kwargs.get('pre_time_ms', 0) or 0

        duration = len(trace) / sample_rate
        n_bin_samples = int(self.bin_width * sample_rate)
        bins = np.arange(0, len(trace), n_bin_samples)

        if spike_times is not None and len(spike_times) > 0:
            count = np.histogram(spike_times, bins=np.append(bins, len(trace)))[0]
        else:
            count = np.zeros(len(bins))

        count = count.astype(float)

        if self.smoothing_window > 0:
            win_samples = int(round(sample_rate * self.smoothing_window / n_bin_samples))
            if win_samples > 1:
                w = self._gausswin(win_samples)
                w = w / np.sum(w)
                count = np.convolve(count, w, mode='same')

        freq = count / self.bin_width

        # Upsample back to trace length for overlay display
        result = np.zeros_like(trace, dtype=float)
        for i, b in enumerate(bins):
            end = bins[i + 1] if i + 1 < len(bins) else len(trace)
            result[b:end] = freq[i] if i < len(freq) else 0
        return result

    @staticmethod
    def _gausswin(n):
        """Gaussian window similar to MATLAB gausswin."""
        alpha = 2.5
        half = (n - 1) / 2
        t = np.arange(n) - half
        return np.exp(-0.5 * (alpha * t / half) ** 2)

    def get_label(self):
        return f"PSTH (bin={self.bin_width}s)"
