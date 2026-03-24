"""Average epochs and calculate variance (ported from Clarinet averageAndVariance).

This is an extractor that operates on groups of epochs rather than single traces.
It computes the mean trace and variance (SD or SEM) across selected epochs.
"""

import param
import numpy as np
from retinanalysis.singleCellGUI.analysis._base import AnalysisPlugin


class AverageAndVariance(AnalysisPlugin):
    name = param.String(default="Average & Variance")
    description = param.String(default="Compute mean trace and variance (SD or SEM) across selected epochs")
    variance_type = param.Selector(
        objects=['SD', 'SEM'], default='SD',
        doc="Variance type: standard deviation or standard error of the mean"
    )

    def process(self, trace, sample_rate, **kwargs):
        # For single-trace calls, just return the trace unchanged.
        # The real work happens in process_group().
        return trace

    def process_group(self, traces):
        """Compute mean and variance across a group of traces.

        Parameters
        ----------
        traces : list[np.ndarray]
            List of 1-D amplitude arrays (one per epoch).

        Returns
        -------
        mean_trace : np.ndarray
        variance_trace : np.ndarray
        """
        if not traces:
            return np.array([]), np.array([])

        # Stack traces (truncate to shortest)
        min_len = min(len(t) for t in traces)
        data = np.column_stack([t[:min_len] for t in traces])

        mean_trace = np.mean(data, axis=1)

        if self.variance_type == 'SEM':
            var_trace = np.std(data, axis=1) / np.sqrt(data.shape[1])
        else:
            var_trace = np.std(data, axis=1)

        return mean_trace, var_trace

    def get_label(self):
        return f"Mean ± {self.variance_type}"
