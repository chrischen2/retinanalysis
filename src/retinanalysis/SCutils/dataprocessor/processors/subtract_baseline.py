"""Subtract baseline from epoch (ported from Clarinet +builtinProcessors/subtractBaseline)."""

import numpy as np


def subtract_baseline(data, pre_points=0, sample_rate=None, pre_time=None):
    """Subtract the mean of pre-stimulus points from the trace.

    If *pre_points* is 0 and both *sample_rate* and *pre_time* are given,
    the number of pre-stimulus points is computed automatically as
    ``int(pre_time * sample_rate / 1000)`` (pre_time in ms).

    If *pre_points* is still 0 after auto-calculation, the global mean is
    subtracted instead.

    Parameters
    ----------
    data : np.ndarray
        1-D amplitude array.
    pre_points : int
        Number of pre-stimulus samples to use as baseline (default 0 = auto).
    sample_rate : float or None
        Sampling rate in Hz (needed for auto-calculation).
    pre_time : float or None
        Pre-stimulus duration in **milliseconds** (needed for auto-calculation).

    Returns
    -------
    np.ndarray
        Baseline-subtracted trace.
    """
    if pre_points == 0 and sample_rate is not None and pre_time is not None:
        pre_points = int(pre_time * sample_rate / 1000)
    if pre_points > 0:
        baseline = np.mean(data[:pre_points])
    else:
        baseline = np.mean(data)
    return data - baseline
