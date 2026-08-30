"""Utility functions for signal processing and analysis."""

from ..utils.filters import compute_filter, convolve_filter_with_stim
from ..utils.nonlinearity import sample_nl
from ..utils.metrics import compute_variance_explained
from ..utils.signal import (
    apply_frequency_cutoff,
    apply_frequency_cutoff_to_fft,
    baseline_subtract,
)
