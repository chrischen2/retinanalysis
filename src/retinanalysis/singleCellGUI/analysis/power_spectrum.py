"""Power spectral density (ported from Clarinet powerSpectrum)."""

import param
import numpy as np
from retinanalysis.singleCellGUI.analysis._base import AnalysisPlugin


class PowerSpectrum(AnalysisPlugin):
    name = param.String(default="Power Spectrum")
    description = param.String(default="Compute power spectral density using FFT")
    include_pre_points = param.Boolean(default=True, doc="Include pre-stimulus points in spectrum")

    def process(self, trace, sample_rate, pre_time_ms=None, **kwargs):
        if not self.include_pre_points and pre_time_ms is not None:
            pre_pts = int(pre_time_ms * sample_rate / 1000)
            data = trace[pre_pts:]
        else:
            data = trace

        if len(data) == 0:
            return np.zeros_like(trace)

        dt = 1.0 / sample_rate
        n = len(data)
        fft_data = np.fft.fft(data)
        power = np.real(fft_data * np.conj(fft_data))
        power = 2 * power * dt / n

        # Pad or truncate to match original trace length
        result = np.zeros_like(trace)
        result[:min(len(power), len(result))] = power[:min(len(power), len(result))]
        return result

    def get_label(self):
        return "Power spectrum"
