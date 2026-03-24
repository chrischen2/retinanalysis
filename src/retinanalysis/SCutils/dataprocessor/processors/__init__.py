"""Built-in signal processors (ported from Clarinet +builtinProcessors)."""

from .add_noise import add_noise
from .apply_nonlinearity import apply_nonlinearity
from .decimate_epoch import decimate_epoch
from .detect_spikes import detect_spikes
from .filter_frequencies import filter_frequencies
from .interpolate_epoch import interpolate_epoch
from .power_spectrum import power_spectrum
from .smooth_epoch import smooth_epoch
from .subtract_baseline import subtract_baseline
from .subtract_drift import subtract_drift
from .truncate_epoch import truncate_epoch
from .truncate_pre_points import truncate_pre_points

__all__ = [
    "add_noise",
    "apply_nonlinearity",
    "decimate_epoch",
    "detect_spikes",
    "filter_frequencies",
    "interpolate_epoch",
    "power_spectrum",
    "smooth_epoch",
    "subtract_baseline",
    "subtract_drift",
    "truncate_epoch",
    "truncate_pre_points",
]
