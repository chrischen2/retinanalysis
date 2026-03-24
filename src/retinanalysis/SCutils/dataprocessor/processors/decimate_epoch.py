"""Decimate epoch by a given factor (ported from Clarinet +builtinProcessors/decimateEpoch)."""

import numpy as np
from scipy.signal import decimate as _decimate


def decimate_epoch(data, factor=2):
    """Decimate (downsample after anti-alias filtering) by *factor*.

    Parameters
    ----------
    data : np.ndarray
        1-D amplitude array.
    factor : int
        Decimation factor (default 2). Must be >= 1.

    Returns
    -------
    np.ndarray
        Decimated trace with ``len(data) // factor`` samples.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return data.copy()
    return _decimate(data, factor)
