"""Signal processing utilities."""

from __future__ import annotations

import numpy as np


def apply_frequency_cutoff_to_fft(
    original_fft: np.ndarray,
    freq_cutoff: float,
    sampling_interval: float,
) -> np.ndarray:
    """Eliminate frequencies above cutoff from FFT-domain data.

    Parameters
    ----------
    original_fft : np.ndarray
        FFT data (1D or 2D with rows as epochs).
    freq_cutoff : float
        Cutoff frequency in Hz.
    sampling_interval : float
        Sampling interval in seconds.

    Returns
    -------
    out : np.ndarray
        Filtered FFT data.
    """
    original_fft = np.atleast_2d(original_fft).copy()
    time_points = original_fft.shape[-1]
    freq_step = 1.0 / (sampling_interval * time_points)
    cutoff_pts = round(freq_cutoff / freq_step)

    # Zero out frequencies beyond cutoff (symmetric in FFT)
    if cutoff_pts + 1 < time_points - cutoff_pts:
        original_fft[..., cutoff_pts + 1 : time_points - cutoff_pts] = 0

    if original_fft.shape[0] == 1:
        return original_fft.squeeze(axis=0)
    return original_fft


def apply_frequency_cutoff(
    original: np.ndarray,
    freq_cutoff: float,
    sampling_interval: float,
) -> np.ndarray:
    """Eliminate frequencies above cutoff from time-domain data.

    Parameters
    ----------
    original : np.ndarray
        Time-domain data (1D or 2D with rows as epochs).
    freq_cutoff : float
        Cutoff frequency in Hz.
    sampling_interval : float
        Sampling interval in seconds.

    Returns
    -------
    out : np.ndarray
        Filtered time-domain data.
    """
    was_1d = original.ndim == 1
    original = np.atleast_2d(original)

    fft_data = np.fft.fft(original, axis=1)
    fft_cutoff = apply_frequency_cutoff_to_fft(fft_data, freq_cutoff, sampling_interval)
    fft_cutoff = np.atleast_2d(fft_cutoff)
    result = np.real(np.fft.ifft(fft_cutoff, axis=1))

    if was_1d:
        return result.squeeze(axis=0)
    return result


def baseline_subtract(
    original: np.ndarray, points_to_avg: int
) -> np.ndarray:
    """Subtract the mean of the first n points from each row.

    Parameters
    ----------
    original : np.ndarray
        Input matrix with rows as epochs.
    points_to_avg : int
        Number of points at the start to average for baseline.

    Returns
    -------
    out : np.ndarray
        Baseline-subtracted data.
    """
    original = np.atleast_2d(original)
    baseline_means = original[:, :points_to_avg].mean(axis=1, keepdims=True)
    return original - baseline_means
