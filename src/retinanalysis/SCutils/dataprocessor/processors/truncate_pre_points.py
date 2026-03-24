"""Remove pre-stimulus points (ported from Clarinet +builtinProcessors/truncatePrePoints)."""

import numpy as np


def truncate_pre_points(data, pre_points=0, sample_rate=None, pre_time=None):
    """Remove pre-stimulus samples from the beginning of a trace.

    If *pre_points* is 0 and both *sample_rate* and *pre_time* are given,
    the number of pre-stimulus points is computed automatically as
    ``int(pre_time * sample_rate / 1000)`` (pre_time in ms).

    Parameters
    ----------
    data : np.ndarray
        1-D amplitude array.
    pre_points : int
        Number of pre-stimulus samples to remove (default 0 = auto).
    sample_rate : float or None
        Sampling rate in Hz (needed for auto-calculation).
    pre_time : float or None
        Pre-stimulus duration in **milliseconds** (needed for auto-calculation).

    Returns
    -------
    np.ndarray
        Stimulus-only portion of the trace.
    """
    if pre_points == 0 and sample_rate is not None and pre_time is not None:
        pre_points = int(pre_time * sample_rate / 1000)
    return data[pre_points:].copy()
