"""Apply multiplicative cumulative-Gaussian nonlinearity (ported from Clarinet +builtinProcessors/applyNonLinearity)."""

import numpy as np
from scipy.stats import norm


def apply_nonlinearity(data, mean=10.0, sd=5.0):
    """Apply a multiplicative nonlinearity using the cumulative Gaussian.

    Each sample is multiplied by the corresponding weight of the cumulative
    normal distribution: ``norm.cdf(data, mean, sd) * data``.

    Parameters
    ----------
    data : np.ndarray
        1-D amplitude array.
    mean : float
        Mean of the cumulative Gaussian (default 10.0).
    sd : float
        Standard deviation of the cumulative Gaussian (default 5.0).

    Returns
    -------
    np.ndarray
        Transformed trace (same shape as *data*).
    """
    return norm.cdf(data, loc=mean, scale=sd) * data
