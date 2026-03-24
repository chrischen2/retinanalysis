"""Subtract linear drift from epoch (ported from Clarinet +builtinProcessors/subtractDrift)."""

import numpy as np


def subtract_drift(data, start_point=0):
    """Remove linear drift by fitting and subtracting a 1st-order polynomial.

    A linear fit is computed from *start_point* to the end of the trace,
    then evaluated over the entire trace and subtracted.

    Parameters
    ----------
    data : np.ndarray
        1-D amplitude array.
    start_point : int
        0-indexed sample from which to begin fitting (default 0).

    Returns
    -------
    np.ndarray
        Drift-corrected trace.
    """
    n = len(data)
    x_fit = np.arange(start_point, n)
    coeffs = np.polyfit(x_fit, data[start_point:], 1)
    x_all = np.arange(n)
    fit = coeffs[0] * x_all + coeffs[1]
    return data - fit
