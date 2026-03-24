"""Truncate epoch to a sample range (ported from Clarinet +builtinProcessors/truncateEpoch)."""

import numpy as np


def truncate_epoch(data, start_sample=0, length_samples=5000):
    """Extract a contiguous segment from a trace.

    Parameters
    ----------
    data : np.ndarray
        1-D amplitude array.
    start_sample : int
        Start position (0-indexed, default 0).
    length_samples : int
        Number of samples to keep (default 5000).

    Returns
    -------
    np.ndarray
        Truncated segment of length up to *length_samples*.
    """
    return data[start_sample : start_sample + length_samples].copy()
