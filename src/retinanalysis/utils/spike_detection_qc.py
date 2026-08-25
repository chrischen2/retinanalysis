"""Protocol-independent visual checks for detected spikes in raw epochs."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def plot_spike_detection_qc(
        traces,
        spike_times: Sequence,
        sample_rate: float,
        fraction: float = 0.30,
        random_state: Optional[int] = None,
        spike_time_unit: str = 'samples',
        stimulus_window_ms: Optional[Tuple[float, float]] = None,
        max_epochs_per_figure: int = 10,
        close_figures: bool = False,
        title: Optional[str] = None):
    """Plot raw traces and detected spikes for a random subset of epochs.

    This function is intentionally protocol-independent: pass any 2-D array of
    ``(epoch, sample)`` amplifier traces and the matching per-epoch spike times.
    ``spike_time_unit`` may be ``'samples'``, ``'ms'``, or ``'s'``. The sampled
    epoch indices are sorted for easy comparison with the source data.

    Parameters
    ----------
    fraction
        Fraction of epochs to inspect, in ``(0, 1]``. At least one epoch is
        selected, using ``ceil(n_epochs * fraction)``.
    random_state
        Seed for reproducible sampling. Use ``None`` for a new sample each run.
    stimulus_window_ms
        Optional ``(onset, offset)`` pair drawn as a shaded region.
    max_epochs_per_figure
        Paginate larger samples so individual spike waveforms remain visible.
    close_figures
        Close each figure after creating it while retaining the returned figure
        object. This is useful for pull-down widgets that explicitly display
        one figure at a time and prevents Matplotlib's open-figure warning.

    Returns
    -------
    figures, selected_indices
        A list of Matplotlib figures and the selected zero-based epoch indices.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    data = np.asarray(traces, dtype=float)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError('traces must be a non-empty 2-D (epoch, sample) array')
    if len(spike_times) != data.shape[0]:
        raise ValueError('spike_times must contain one entry per trace')
    if not np.isfinite(sample_rate) or float(sample_rate) <= 0:
        raise ValueError('sample_rate must be positive')
    if not np.isfinite(fraction) or not 0 < float(fraction) <= 1:
        raise ValueError('fraction must satisfy 0 < fraction <= 1')
    if int(max_epochs_per_figure) != max_epochs_per_figure or max_epochs_per_figure < 1:
        raise ValueError('max_epochs_per_figure must be a positive integer')

    unit = str(spike_time_unit).lower()
    scale_to_ms = {'samples': 1e3 / float(sample_rate), 'ms': 1.0, 's': 1e3}
    if unit not in scale_to_ms:
        raise ValueError("spike_time_unit must be 'samples', 'ms', or 's'")
    if stimulus_window_ms is not None:
        if len(stimulus_window_ms) != 2:
            raise ValueError('stimulus_window_ms must be an (onset, offset) pair')
        stim_on, stim_off = map(float, stimulus_window_ms)
        if not np.isfinite([stim_on, stim_off]).all() or stim_off <= stim_on:
            raise ValueError('stimulus_window_ms must have finite onset < offset')

    n_epochs = data.shape[0]
    n_select = min(n_epochs, max(1, int(np.ceil(n_epochs * float(fraction)))))
    rng = np.random.default_rng(random_state)
    selected = np.sort(rng.choice(n_epochs, size=n_select, replace=False))
    time_ms = np.arange(data.shape[1], dtype=float) / float(sample_rate) * 1e3

    style.apply_publication_style()
    figures = []
    page_size = int(max_epochs_per_figure)
    for page_start in range(0, n_select, page_size):
        page = selected[page_start:page_start + page_size]
        fig, axes = plt.subplots(
            len(page), 1, sharex=True, squeeze=False,
            figsize=(10.0, max(2.2 * len(page), 2.8)), constrained_layout=True)
        axes = axes[:, 0]
        for ax, epoch_index in zip(axes, page):
            trace = data[epoch_index]
            times_ms = np.asarray(
                [] if spike_times[epoch_index] is None else spike_times[epoch_index],
                dtype=float) * scale_to_ms[unit]
            valid = np.isfinite(times_ms) & (times_ms >= 0) & (times_ms < time_ms[-1] + 1e3 / sample_rate)
            times_ms = times_ms[valid]
            sample_indices = np.clip(
                np.rint(times_ms / 1e3 * float(sample_rate)).astype(int),
                0, trace.size - 1)

            ax.plot(time_ms, trace, color='#333333', lw=0.65, label='raw trace')
            if times_ms.size:
                ax.scatter(times_ms, trace[sample_indices], s=18, color='#D55E00',
                           zorder=3, label='detected spike')
            if stimulus_window_ms is not None:
                ax.axvspan(stim_on, stim_off, color='#F0C000', alpha=0.16, lw=0,
                           label='stimulus')
            ax.set_ylabel(f'Epoch {epoch_index}')
            ax.set_title(f'{times_ms.size} detected spike(s)', fontsize=9, loc='left')
            ax.margins(x=0)

        axes[-1].set_xlabel('Time (ms)')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, fontsize=8, loc='upper right')
        page_number = page_start // page_size + 1
        n_pages = int(np.ceil(n_select / page_size))
        heading = title or 'Spike-detection QC'
        if n_pages > 1:
            heading += f' — page {page_number}/{n_pages}'
        fig.suptitle(
            f'{heading}\nrandom {n_select}/{n_epochs} epochs ({100 * n_select / n_epochs:.1f}%)',
            fontsize=11)
        figures.append(fig)
        if close_figures:
            plt.close(fig)

    return figures, selected
