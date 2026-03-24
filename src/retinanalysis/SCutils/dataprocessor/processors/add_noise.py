"""Add zero-mean Gaussian white noise (ported from Clarinet +builtinProcessors/addNoise)."""

import numpy as np


def add_noise(data, amplitude=10.0, rng=None):
    """Add zero-mean Gaussian white noise to a trace.

    Parameters
    ----------
    data : np.ndarray
        1-D amplitude array.
    amplitude : float
        Standard deviation of the Gaussian noise (default 10.0).
    rng : np.random.Generator or None
        Random number generator for reproducibility. If *None*, a new
        default generator is created.

    Returns
    -------
    np.ndarray
        Noisy trace (same shape as *data*).
    """
    if rng is None:
        rng = np.random.default_rng()
    return data + rng.normal(0, amplitude, size=data.shape)
