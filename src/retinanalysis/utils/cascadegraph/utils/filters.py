"""Filter computation and convolution utilities."""

from __future__ import annotations

import numpy as np

from ..utils.signal import apply_frequency_cutoff_to_fft


def compute_filter(
    stim: np.ndarray,
    response: np.ndarray,
    filter_pts: int,
    correct_stim_power: bool = False,
    frequency_cutoff: float | None = None,
    sampling_interval: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the filter that best predicts response given white noise stimulus.

    This is the cross-correlation between stimulus and response, computed in
    the frequency domain.

    Parameters
    ----------
    stim : np.ndarray
        Matrix of row vectors (epochs x time).
    response : np.ndarray
        Matrix of row vectors (epochs x time).
    filter_pts : int
        Length of one side of the filter (causal or anticausal half).
    correct_stim_power : bool
        If True, normalize by stimulus power spectrum.
    frequency_cutoff : float, optional
        Cutoff frequency in Hz. Must be provided with sampling_interval.
    sampling_interval : float, optional
        Sampling interval in seconds.

    Returns
    -------
    filter_causal : np.ndarray
        Causal half of the filter.
    filter_anticausal : np.ndarray
        Anticausal half of the filter.
    """
    if (frequency_cutoff is None) != (sampling_interval is None):
        raise ValueError("frequency_cutoff and sampling_interval must be provided together")

    stim = np.atleast_2d(stim)
    response = np.atleast_2d(response)

    stim_fft = np.fft.fft(stim, axis=1)
    resp_fft = np.fft.fft(response, axis=1)

    filter_fft = np.mean(resp_fft * np.conj(stim_fft), axis=0)

    if correct_stim_power:
        filter_fft = filter_fft / np.mean(stim_fft * np.conj(stim_fft), axis=0)

    if frequency_cutoff is not None:
        filter_fft = apply_frequency_cutoff_to_fft(filter_fft, frequency_cutoff, sampling_interval)

    filter_fft[0] = 0  # remove DC component
    filter_full = np.real(np.fft.ifft(filter_fft))

    filter_causal = filter_full[:filter_pts]
    filter_anticausal = filter_full[-filter_pts:]

    return filter_causal, filter_anticausal


def convolve_filter_with_stim(
    filter_vec: np.ndarray,
    stim: np.ndarray,
    filter_has_anticausal_half: bool = False,
) -> np.ndarray:
    """Convolve a filter with each row of a stimulus matrix via FFT.

    Parameters
    ----------
    filter_vec : np.ndarray
        Filter vector (no zero padding).
    stim : np.ndarray
        Matrix of row vectors (epochs x time), or 1D vector.
    filter_has_anticausal_half : bool
        If True, the filter has structure on both sides of t=0.

    Returns
    -------
    convolution : np.ndarray
        Convolved signal, same shape as stim.
    """
    filter_vec = np.asarray(filter_vec).ravel()
    was_1d = stim.ndim == 1
    stim = np.atleast_2d(stim).copy()

    if len(filter_vec) % 2 != 0:
        raise ValueError("Filter must have an even number of points")

    filter_length = len(filter_vec)
    stim_length = stim.shape[1]

    # Mean-subtract each row
    stim = stim - stim.mean(axis=1, keepdims=True)

    # Zero pad to match lengths
    length_diff = abs(filter_length - stim_length)
    if filter_length < stim_length:
        midpoint = filter_length // 2
        if filter_has_anticausal_half:
            filter_vec = np.concatenate([
                filter_vec[:midpoint],
                np.zeros(length_diff),
                filter_vec[midpoint:],
            ])
        else:
            filter_vec = np.concatenate([filter_vec, np.zeros(length_diff)])
    else:
        stim = np.concatenate([stim, np.zeros((stim.shape[0], length_diff))], axis=1)

    # FFT-based convolution
    filter_fft = np.fft.fft(filter_vec)
    stim_fft = np.fft.fft(stim, axis=1)
    convolution = np.real(np.fft.ifft(stim_fft * filter_fft, axis=1))

    if was_1d:
        return convolution.squeeze(axis=0)
    return convolution
