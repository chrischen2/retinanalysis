"""FFT-based frequency filtering (ported from Clarinet filterFrequencies)."""

import param
import numpy as np
from retinanalysis.singleCellGUI.analysis._base import AnalysisPlugin


class FilterFrequencies(AnalysisPlugin):
    name = param.String(default="Filter Frequencies")
    description = param.String(default="FFT-based highpass, lowpass, and notch filtering")
    highpass_freq = param.Number(default=0, bounds=(0, None), doc="High-pass cutoff (Hz). 0=disabled")
    lowpass_freq = param.Number(default=0, bounds=(0, None), doc="Low-pass cutoff (Hz). 0=disabled")
    notch_freq = param.Number(default=0, bounds=(0, None), doc="Notch frequency (Hz). 0=disabled")

    def process(self, trace, sample_rate, **kwargs):
        if self.highpass_freq == 0 and self.lowpass_freq == 0 and self.notch_freq == 0:
            return trace

        n = len(trace)
        nfft = int(2 ** np.ceil(np.log2(n)))
        freqs = np.linspace(0, sample_rate / 2, nfft // 2 + 1)

        y = np.fft.fft(trace, n=nfft)

        # Apply filters in frequency domain
        if self.lowpass_freq > 0:
            y[freqs >= self.lowpass_freq] = 0
        if self.highpass_freq > 0:
            y[freqs <= self.highpass_freq] = 0
        if self.notch_freq > 0:
            notch_idx = np.argmin(np.abs(freqs - self.notch_freq))
            y[notch_idx] = 0

        # Make conjugate symmetric
        nf = len(freqs)
        for k in range(nf, len(y)):
            y[k] = np.conj(y[(len(y) - k) % len(y)])

        result = np.real(np.fft.ifft(y))
        return result[:n]

    def get_label(self):
        parts = []
        if self.highpass_freq > 0:
            parts.append(f"HP>{self.highpass_freq}Hz")
        if self.lowpass_freq > 0:
            parts.append(f"LP<{self.lowpass_freq}Hz")
        if self.notch_freq > 0:
            parts.append(f"Notch {self.notch_freq}Hz")
        return " ".join(parts) if parts else "Freq filter"
