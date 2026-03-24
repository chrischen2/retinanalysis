"""Subtract linear drift from epoch (ported from Clarinet subtractDrift)."""

import param
import numpy as np
from retinanalysis.singleCellGUI.analysis._base import AnalysisPlugin


class SubtractDrift(AnalysisPlugin):
    name = param.String(default="Subtract Drift")
    description = param.String(default="Remove linear drift by fitting and subtracting a 1st-order polynomial")
    start_point = param.Integer(default=1, bounds=(1, None), doc="Sample index to start fitting from")

    def process(self, trace, sample_rate, **kwargs):
        n = len(trace)
        start = max(0, self.start_point - 1)  # 0-indexed
        x_fit = np.arange(start, n)
        coeffs = np.polyfit(x_fit, trace[start:], 1)
        x_all = np.arange(n)
        fit = coeffs[0] * x_all + coeffs[1]
        return trace - fit

    def get_label(self):
        return "Drift subtracted"
