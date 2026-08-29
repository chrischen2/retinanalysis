"""Utility functions for signal processing and analysis."""

from cascadegraph.utils.filters import compute_filter, convolve_filter_with_stim
from cascadegraph.utils.nonlinearity import sample_nl
from cascadegraph.utils.metrics import compute_variance_explained
from cascadegraph.utils.signal import (
    apply_frequency_cutoff,
    apply_frequency_cutoff_to_fft,
    baseline_subtract,
)
