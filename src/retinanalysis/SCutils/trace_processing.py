"""Protocol-independent preprocessing for single-cell amplifier traces.

These functions deliberately operate on arrays rather than DataJoint rows or
protocol parameters. Protocol modules remain responsible for forming exact
condition groups before calling :func:`align_epoch_group_baselines`; this keeps
baseline offsets from leaking across recording modes, stimulus conditions, or
mean-light levels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


EPOCH_BASELINE_TARGETS = ('first_epoch', 'first_two_mean', 'median')


def milliseconds_to_samples(value_ms: Optional[float], sample_rate: float,
                            parameter_name: str = 'window_ms', *,
                            allow_zero: bool = False) -> Optional[int]:
    """Convert a user-facing millisecond interval to amplifier samples."""
    if value_ms is None:
        return None
    value = float(value_ms)
    rate = float(sample_rate)
    invalid = (not np.isfinite(value) or value < 0
               or (not allow_zero and value == 0))
    if invalid:
        qualifier = 'non-negative' if allow_zero else 'positive'
        raise ValueError(f'{parameter_name} must be finite and {qualifier}')
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError('sample_rate must be finite and positive')
    if value == 0:
        return 0
    return max(int(round(value / 1e3 * rate)), 1)


def block_average(trace: np.ndarray, factor: int) -> np.ndarray:
    """Average consecutive non-overlapping samples, dropping a short tail."""
    factor = max(int(factor), 1)
    values = np.asarray(trace, dtype=float)
    if values.ndim != 1:
        raise ValueError('trace must be one-dimensional')
    if factor == 1:
        return values
    width = (values.size // factor) * factor
    return values[:width].reshape(-1, factor).mean(axis=1)


def preprocess_spike_trace(trace: np.ndarray, sample_rate: float,
                           median_window_ms: Optional[float] = 5.0,
                           high_pass_hz: float = 300.0) -> np.ndarray:
    """Median-detrend and high-pass one trace exactly as the detector does."""
    from retinanalysis.utils.spike_detector import preprocess_spike_traces

    median_samples = milliseconds_to_samples(
        median_window_ms, sample_rate, 'spike_median_window_ms',
        allow_zero=True)
    return preprocess_spike_traces(
        trace, sample_rate=sample_rate,
        median_window_samples=median_samples,
        cutoff_frequency=float(high_pass_hz))[0]


def preprocess_whole_cell_trace(
        trace: np.ndarray, sample_rate: float,
        bin_ms: float = 5.0) -> Tuple[np.ndarray, float]:
    """Smooth and reduce a current trace by non-overlapping bin averages."""
    factor = milliseconds_to_samples(
        bin_ms, sample_rate, 'whole_cell_bin_ms', allow_zero=False)
    return block_average(trace, factor), float(sample_rate) / factor


def normalize_epoch_baseline_target(value) -> str:
    """Validate and normalize an epoch-group baseline target policy."""
    method = str(value).strip().lower()
    if method not in EPOCH_BASELINE_TARGETS:
        choices = ', '.join(repr(choice) for choice in EPOCH_BASELINE_TARGETS)
        raise ValueError(
            f'baseline target must be one of {choices}; got {value!r}')
    return method


@dataclass(frozen=True)
class EpochBaselineAlignment:
    """Result and audit values from one already-separated condition group."""

    traces: np.ndarray
    epoch_means: np.ndarray
    offsets: np.ndarray
    target: float
    method: str
    reference_n: int
    reference_mask: np.ndarray


def align_epoch_group_baselines(
        traces: np.ndarray,
        target: str = 'first_epoch') -> EpochBaselineAlignment:
    """Align whole-cell epochs within one caller-defined condition group.

    Parameters
    ----------
    traces
        Two-dimensional ``epoch x time`` current array in acquisition order.
    target
        ``'first_epoch'`` (default) anchors to the first epoch's mean;
        ``'first_two_mean'`` averages the first two epoch means; ``'median'``
        uses the median across every epoch.

    Notes
    -----
    This function intentionally does not infer or combine condition labels.
    Call it separately for every recording type, duration, contrast, light
    mean, and any other stimulus dimension whose natural baseline must remain
    distinct.
    """
    values = np.asarray(traces, dtype=float)
    if values.ndim != 2:
        raise ValueError('traces must be a two-dimensional epoch x time array')
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError('traces must contain at least one epoch and time sample')

    method = normalize_epoch_baseline_target(target)
    epoch_means = values.mean(axis=1)
    if method == 'first_epoch':
        reference_n = 1
        baseline = float(epoch_means[0])
    elif method == 'first_two_mean':
        reference_n = min(2, len(epoch_means))
        baseline = float(np.mean(epoch_means[:reference_n]))
    else:
        reference_n = len(epoch_means)
        baseline = float(np.median(epoch_means))
    offsets = baseline - epoch_means
    reference_mask = np.arange(len(epoch_means)) < reference_n
    return EpochBaselineAlignment(
        traces=values + offsets[:, None],
        epoch_means=epoch_means,
        offsets=offsets,
        target=baseline,
        method=method,
        reference_n=reference_n,
        reference_mask=reference_mask)


__all__ = [
    'EPOCH_BASELINE_TARGETS', 'EpochBaselineAlignment',
    'align_epoch_group_baselines', 'block_average',
    'milliseconds_to_samples', 'normalize_epoch_baseline_target',
    'preprocess_spike_trace', 'preprocess_whole_cell_trace',
]
