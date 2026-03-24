"""FFT-based frequency filtering (ported from Clarinet +builtinProcessors/filterFrequencies)."""

import numpy as np


def filter_frequencies(data, sample_rate, highpass_freq=0.0, lowpass_freq=0.0,
                       notch_freq=0.0):
    """Filter a trace in the frequency domain (lowpass / highpass / notch).

    Parameters
    ----------
    data : np.ndarray
        1-D amplitude array.
    sample_rate : float
        Sampling rate in Hz.
    highpass_freq : float
        High-pass cutoff frequency in Hz (0 = disabled).
    lowpass_freq : float
        Low-pass cutoff frequency in Hz (0 = disabled).
    notch_freq : float
        Notch frequency to remove in Hz (0 = disabled).

    Returns
    -------
    np.ndarray
        Filtered trace (same length as *data*).
    """
    if highpass_freq == 0 and lowpass_freq == 0 and notch_freq == 0:
        return data.copy()

    n = len(data)
    nfft = int(2 ** np.ceil(np.log2(n)))
    freqs = np.linspace(0, sample_rate / 2, nfft // 2 + 1)

    y = np.fft.fft(data, n=nfft)

    # Apply filters to positive-frequency half only
    nf = len(freqs)
    if lowpass_freq > 0:
        mask = np.where(freqs >= lowpass_freq)[0]
        y[mask] = 0
    if highpass_freq > 0:
        mask = np.where(freqs <= highpass_freq)[0]
        y[mask] = 0
    if notch_freq > 0:
        notch_idx = np.argmin(np.abs(freqs - notch_freq))
        y[notch_idx] = 0

    # Reconstruct conjugate-symmetric spectrum
    for k in range(nf, len(y)):
        y[k] = np.conj(y[(len(y) - k) % len(y)])

    result = np.real(np.fft.ifft(y))
    return result[:n]
