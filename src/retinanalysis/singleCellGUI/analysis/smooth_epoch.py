"""Smooth epoch data (ported from Clarinet smoothEpoch)."""

import param
import numpy as np
from retinanalysis.singleCellGUI.analysis._base import AnalysisPlugin


class SmoothEpoch(AnalysisPlugin):
    name = param.String(default="Smooth Epoch")
    description = param.String(default="Smooth data along time dimension")
    mode = param.Selector(
        objects=['moving', 'savgol'],
        default='moving',
        doc="Smoothing method"
    )
    span = param.Integer(default=5, bounds=(3, 1001), doc="Smoothing window size in points (must be odd)")

    def process(self, trace, sample_rate, **kwargs):
        span = self.span if self.span % 2 == 1 else self.span + 1

        if self.mode == 'moving':
            kernel = np.ones(span) / span
            return np.convolve(trace, kernel, mode='same')
        elif self.mode == 'savgol':
            from scipy.signal import savgol_filter
            polyorder = min(3, span - 1)
            return savgol_filter(trace, span, polyorder)
        return trace

    def get_label(self):
        return f"Smooth ({self.mode}, {self.span})"
