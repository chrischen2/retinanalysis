"""Smooth epoch data (ported from Clarinet +builtinProcessors/smoothEpoch)."""

import numpy as np


def smooth_epoch(data, mode="moving", span=5):
    """Smooth a trace along the time dimension.

    Parameters
    ----------
    data : np.ndarray
        1-D amplitude array.
    mode : str
        Smoothing method. Supported values:

        * ``"moving"`` -- moving-average (uniform kernel convolution)
        * ``"savgol"`` -- Savitzky-Golay filter (3rd-order polynomial)
        * ``"lowess"`` -- locally weighted scatterplot smoothing (requires *statsmodels*)
        * ``"loess"``  -- alias for lowess with 2nd-degree polynomial
        * ``"rlowess"`` -- robust lowess
        * ``"rloess"``  -- robust loess

    span : int
        Smoothing window size in points (default 5). Will be forced to the
        nearest odd number if even.

    Returns
    -------
    np.ndarray
        Smoothed trace (same length as *data*).
    """
    span = span if span % 2 == 1 else span + 1

    if mode == "moving":
        kernel = np.ones(span) / span
        return np.convolve(data, kernel, mode="same")

    if mode in ("savgol", "sgolay"):
        from scipy.signal import savgol_filter
        polyorder = min(3, span - 1)
        return savgol_filter(data, span, polyorder)

    if mode in ("lowess", "rlowess", "loess", "rloess"):
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
        except ImportError:
            raise ImportError(
                f"Mode '{mode}' requires the statsmodels package. "
                "Install it with: pip install statsmodels"
            )
        frac = min(1.0, span / len(data))
        robust = mode.startswith("r")
        it = 3 if robust else 0
        result = lowess(
            data,
            np.arange(len(data)),
            frac=frac,
            it=it,
            return_sorted=True,
        )
        return result[:, 1]

    raise ValueError(f"Unknown smoothing mode: {mode!r}")
