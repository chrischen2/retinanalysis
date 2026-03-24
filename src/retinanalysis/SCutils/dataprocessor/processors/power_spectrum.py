"""Power spectral density (ported from Clarinet +builtinProcessors/powerSpectrum)."""

import numpy as np


def power_spectrum(data, sample_rate, pre_points=None):
    """Compute the power spectral density of a trace using FFT.

    Parameters
    ----------
    data : np.ndarray
        1-D amplitude array.
    sample_rate : float
        Sampling rate in Hz.
    pre_points : int or None
        If given and > 0, exclude the first *pre_points* samples before
        computing the spectrum (i.e. analyse only the stimulus portion).

    Returns
    -------
    freqs : np.ndarray
        Frequency axis in Hz (positive frequencies only).
    power : np.ndarray
        Power spectral density values.
    """
    if pre_points is not None and pre_points > 0:
        data = data[pre_points:]

    if len(data) == 0:
        return np.array([]), np.array([])

    dt = 1.0 / sample_rate
    n = len(data)
    fft_data = np.fft.rfft(data)
    power = np.abs(fft_data) ** 2
    power = 2 * power * dt / n

    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, power
