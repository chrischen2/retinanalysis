"""Interpolate epoch by a given factor (ported from Clarinet +builtinProcessors/interpolateEpoch)."""

import numpy as np
from scipy.signal import resample_poly


def interpolate_epoch(data, factor=2):
    """Upsample a trace by *factor* using polyphase FIR interpolation.

    This mirrors MATLAB's ``interp(data, factor)`` which performs
    low-pass FIR interpolation.

    Parameters
    ----------
    data : np.ndarray
        1-D amplitude array.
    factor : int
        Interpolation factor (default 2). Must be >= 1.

    Returns
    -------
    np.ndarray
        Interpolated trace with ``len(data) * factor`` samples.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return data.copy()
    return resample_poly(data, up=factor, down=1)
