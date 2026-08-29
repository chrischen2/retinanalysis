"""Nonlinearity sampling utilities."""

from __future__ import annotations

import numpy as np


def sample_nl(
    input_signal: np.ndarray,
    response: np.ndarray,
    num_bins: int,
    bin_type: str = "equalWidth",
) -> tuple[np.ndarray, np.ndarray]:
    """Return binned sampling of the relationship between two signals.

    Parameters
    ----------
    input_signal : np.ndarray
        Matrix of row vectors (epochs x time), or 1D vector.
    response : np.ndarray
        Matrix of row vectors (epochs x time), or 1D vector.
    num_bins : int
        Number of bins.
    bin_type : str
        Either "equalWidth" or "equalN".

    Returns
    -------
    nl_x : np.ndarray
        Bin center x-values.
    nl_y : np.ndarray
        Mean y-values per bin.
    """
    if input_signal.shape != response.shape:
        raise ValueError("Input matrices must have same size")
    if num_bins <= 1:
        raise ValueError("Number of bins must be greater than 1")

    # Flatten row-major (matches MATLAB reshape(x', 1, []))
    x_flat = np.atleast_2d(input_signal).ravel()
    y_flat = np.atleast_2d(response).ravel()
    num_points = len(x_flat)

    nl_x = np.zeros(num_bins)
    nl_y = np.zeros(num_bins)

    if bin_type == "equalWidth":
        counts, bin_edges = np.histogram(x_flat, bins=num_bins)
        # Bin centers are midpoints
        nl_x = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # Assign bin indices (digitize returns 1-based; clip to valid range)
        bin_idx = np.digitize(x_flat, bin_edges)
        bin_idx = np.clip(bin_idx, 1, num_bins)
        for i in range(num_bins):
            mask = bin_idx == (i + 1)
            if mask.any():
                nl_y[i] = np.mean(y_flat[mask])

    elif bin_type == "equalN":
        if num_points % num_bins != 0:
            raise ValueError(
                "For equalN bin type, number of points must be evenly divisible by number of bins"
            )
        count_in_bin = num_points // num_bins
        sorted_indices = np.argsort(x_flat)
        sorted_x = x_flat[sorted_indices]
        sorted_y = y_flat[sorted_indices]
        for i in range(num_bins):
            start = i * count_in_bin
            end = start + count_in_bin
            nl_x[i] = np.mean(sorted_x[start:end])
            nl_y[i] = np.mean(sorted_y[start:end])
    else:
        raise ValueError(f"bin_type '{bin_type}' not recognized. Use 'equalWidth' or 'equalN'")

    return nl_x, nl_y
