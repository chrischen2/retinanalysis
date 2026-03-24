"""Baseline subtraction plugin."""
 
import param
import numpy as np
from retinanalysis.singleCellGUI.analysis._base import AnalysisPlugin
 
 
class BaselineAdjustment(AnalysisPlugin):
    name = param.String(default="Baseline Adjustment")
    description = param.String(default="Subtract baseline from trace using pre-stimulus, mean, or median")
    method = param.Selector(
        objects=['pre_stim', 'mean_subtract', 'median_subtract'],
        default='pre_stim',
        doc="Baseline estimation method"
    )
 
    def process(self, trace, sample_rate, pre_time_ms=None, **kwargs):
        if self.method == 'pre_stim' and pre_time_ms is not None:
            pre_samples = int(pre_time_ms * sample_rate / 1000)
            pre_samples = max(1, min(pre_samples, len(trace)))
            baseline = np.mean(trace[:pre_samples])
        elif self.method == 'median_subtract':
            baseline = np.median(trace)
        else:
            baseline = np.mean(trace)
        return trace - baseline
 
    def get_label(self):
        return f"Baseline ({self.method})"