"""Metrics for evaluating model predictions."""

from __future__ import annotations

import numpy as np


def compute_variance_explained(
    predicted: np.ndarray, measured: np.ndarray
) -> np.ndarray:
    """Compute row-wise R-squared between two signals.

    Parameters
    ----------
    predicted : np.ndarray
        Predicted signal (1D or 2D with rows as epochs).
    measured : np.ndarray
        Measured signal (same shape as predicted).

    Returns
    -------
    r_squared : np.ndarray
        R-squared value(s). Scalar if inputs are 1D, vector if 2D.
    """
    predicted = np.atleast_2d(predicted)
    measured = np.atleast_2d(measured)

    if predicted.shape != measured.shape:
        raise ValueError("Input matrices must have same size")

    response_mean = measured.mean(axis=1, keepdims=True)
    ss_err = np.sum((measured - predicted) ** 2, axis=1)
    ss_tot = np.sum((measured - response_mean) ** 2, axis=1)

    r_squared = 1 - ss_err / ss_tot

    if r_squared.size == 1:
        return float(r_squared[0])
    return r_squared
