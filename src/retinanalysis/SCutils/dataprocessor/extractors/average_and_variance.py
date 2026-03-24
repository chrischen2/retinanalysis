"""Average epochs and compute variance (ported from Clarinet +builtinExtractors/averageAndVariance)."""

import numpy as np


def average_and_variance(traces, variance_type="SD"):
    """Compute the mean trace and variance across a group of epochs.

    Parameters
    ----------
    traces : list[np.ndarray] or np.ndarray
        Either a list of 1-D arrays (one per epoch) or a 2-D array of
        shape ``(n_epochs, n_samples)``.  If traces differ in length the
        shortest is used.
    variance_type : str
        ``"SD"`` for standard deviation (default) or ``"SEM"`` for
        standard error of the mean.

    Returns
    -------
    mean_trace : np.ndarray
    variance_trace : np.ndarray
    """
    if isinstance(traces, np.ndarray) and traces.ndim == 2:
        data = traces
    else:
        if len(traces) == 0:
            return np.array([]), np.array([])
        min_len = min(len(t) for t in traces)
        data = np.column_stack([t[:min_len] for t in traces])

    mean_trace = np.mean(data, axis=1)

    if variance_type == "SEM":
        var_trace = np.std(data, axis=1) / np.sqrt(data.shape[1])
    else:
        var_trace = np.std(data, axis=1)

    return mean_trace, var_trace
