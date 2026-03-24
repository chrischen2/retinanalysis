"""Truncate epoch data (ported from Clarinet truncateEpoch + truncatePrePoints)."""

import param
import numpy as np
from retinanalysis.singleCellGUI.analysis._base import AnalysisPlugin


class TruncateEpoch(AnalysisPlugin):
    name = param.String(default="Truncate Epoch")
    description = param.String(default="Extract a subset of epoch data by sample range")
    start_sample = param.Integer(default=1, bounds=(1, None), doc="Start sample (1-indexed)")
    length_samples = param.Integer(default=5000, bounds=(1, None), doc="Number of samples to keep")

    def process(self, trace, sample_rate, **kwargs):
        start = max(0, self.start_sample - 1)
        end = start + self.length_samples
        segment = trace[start:end]
        # Pad with zeros to maintain original length
        result = np.zeros_like(trace)
        result[:len(segment)] = segment
        return result

    def get_label(self):
        return f"Truncated [{self.start_sample}:{self.start_sample + self.length_samples}]"


class TruncatePrePoints(AnalysisPlugin):
    name = param.String(default="Truncate Pre-Points")
    description = param.String(default="Remove pre-stimulus points from epoch")

    def process(self, trace, sample_rate, pre_time_ms=None, **kwargs):
        if pre_time_ms is None:
            return trace
        pre_pts = int(pre_time_ms * sample_rate / 1000)
        result = np.zeros_like(trace)
        stim_data = trace[pre_pts:]
        result[:len(stim_data)] = stim_data
        return result

    def get_label(self):
        return "Pre-points removed"
