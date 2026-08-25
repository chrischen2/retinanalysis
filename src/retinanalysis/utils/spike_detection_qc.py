"""Protocol-independent visual checks for detected spikes in raw epochs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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


@dataclass
class SpikeDetectionQCBrowser:
    """Handles returned by :func:`spike_detection_qc_browser`."""

    widget: Any
    selector: Any
    image: Any
    figures: List[Any]
    selected_epochs: List[np.ndarray]
    option_labels: List[str]
    png_cache: Dict[int, bytes] = field(default_factory=dict, repr=False)


def spike_detection_qc_browser(
        datasets: Sequence[Mapping],
        fraction: float = 0.30,
        random_state: Optional[int] = None,
        epochs_per_view: int = 1,
        image_dpi: int = 130,
        display_widget: bool = True,
        verbose: bool = True) -> Optional[SpikeDetectionQCBrowser]:
    """Build a compact pull-down browser for spike-detection QC.

    This is protocol-independent. Each dataset mapping must contain ``traces``,
    ``spike_times``, and ``sample_rate``. Optional keys are:

    - ``label``: short pull-down label;
    - ``title``: figure heading;
    - ``spike_time_unit``: ``'samples'``, ``'ms'``, or ``'s'``;
    - ``stimulus_window_ms``: onset/offset shading;
    - ``source_group``, ``block_ids``, and ``epoch_indices``: stable source
      identity and readable epoch labels;
    - ``epoch_keys`` and ``epoch_labels``: protocol-specific identity/labels
      that override the block-based defaults.

    The random subset is selected independently in each dataset. Duplicate
    source epochs are removed across datasets. Figures are rendered into one
    :class:`ipywidgets.Image`; changing the pull-down replaces its PNG bytes,
    so notebook frontends cannot accumulate repeated trace outputs.
    """
    import io
    import ipywidgets as widgets

    datasets = list(datasets)
    if not datasets:
        if verbose:
            print('No datasets are available for spike-detection QC.')
        return None
    if int(epochs_per_view) != epochs_per_view or epochs_per_view < 1:
        raise ValueError('epochs_per_view must be a positive integer')
    if int(image_dpi) != image_dpi or image_dpi < 1:
        raise ValueError('image_dpi must be a positive integer')

    figures_out: List[Any] = []
    selected_out: List[np.ndarray] = []
    option_labels: List[str] = []
    seen_views = set()
    page_size = int(epochs_per_view)

    for dataset_index, dataset in enumerate(datasets):
        missing = [key for key in ('traces', 'spike_times', 'sample_rate')
                   if key not in dataset]
        if missing:
            raise ValueError(f'dataset {dataset_index} is missing {missing}')
        name = str(dataset.get('label', f'Dataset {dataset_index + 1}'))
        seed = (None if random_state is None
                else int(random_state) + dataset_index)
        figures, selected = plot_spike_detection_qc(
            dataset['traces'], dataset['spike_times'],
            sample_rate=dataset['sample_rate'], fraction=fraction,
            random_state=seed,
            spike_time_unit=dataset.get('spike_time_unit', 'samples'),
            stimulus_window_ms=dataset.get('stimulus_window_ms'),
            max_epochs_per_figure=page_size, close_figures=True,
            title=dataset.get('title', name))
        selected_out.append(selected)
        if verbose:
            print(f'{name}: checking epochs {selected.tolist()}')

        n_epochs = len(dataset['traces'])
        def optional_list(key):
            value = dataset.get(key)
            return [] if value is None else list(value)

        block_ids = optional_list('block_ids')
        epoch_indices = optional_list('epoch_indices')
        epoch_keys = optional_list('epoch_keys')
        epoch_labels = optional_list('epoch_labels')
        for values, field_name in ((block_ids, 'block_ids'),
                                   (epoch_indices, 'epoch_indices'),
                                   (epoch_keys, 'epoch_keys'),
                                   (epoch_labels, 'epoch_labels')):
            if values and len(values) != n_epochs:
                raise ValueError(
                    f'dataset {dataset_index} {field_name} must match traces')

        for view_index, figure in enumerate(figures):
            page = selected[view_index * page_size:(view_index + 1) * page_size]
            page_keys, page_labels = [], []
            for axis, local_epoch_value in zip(figure.axes, page):
                local_epoch = int(local_epoch_value)
                if epoch_labels:
                    epoch_label = str(epoch_labels[local_epoch])
                elif block_ids:
                    block_id = int(block_ids[local_epoch])
                    epoch_in_block = (int(epoch_indices[local_epoch])
                                      if epoch_indices else
                                      sum(int(value) == block_id
                                          for value in block_ids[:local_epoch]))
                    epoch_label = f'block {block_id} | epoch {epoch_in_block}'
                else:
                    epoch_label = f'epoch {local_epoch}'

                if epoch_keys:
                    epoch_key = ('explicit', epoch_keys[local_epoch])
                elif block_ids:
                    epoch_key = (dataset.get('source_group', name),
                                 int(block_ids[local_epoch]), epoch_in_block)
                else:
                    epoch_key = (dataset_index, local_epoch)
                try:
                    hash(epoch_key)
                except TypeError as exc:
                    raise ValueError('epoch_keys entries must be hashable') from exc
                page_keys.append(epoch_key)
                page_labels.append(epoch_label)
                axis.set_ylabel(epoch_label.replace(' | ', '\n'))

            view_key = tuple(page_keys)
            if view_key in seen_views:
                continue
            seen_views.add(view_key)
            figures_out.append(figure)
            option_labels.append(f'{name} | {", ".join(page_labels)}')

    if not figures_out:
        if verbose:
            print('No unique epochs are available for spike-detection QC.')
        return None

    selector = widgets.Dropdown(
        options=[(label, index) for index, label in enumerate(option_labels)],
        description='QC view:', layout=widgets.Layout(width='520px'),
        style={'description_width': '70px'})
    image = widgets.Image(format='png', layout=widgets.Layout(max_width='100%'))
    browser = SpikeDetectionQCBrowser(
        widget=widgets.VBox([selector, image]), selector=selector, image=image,
        figures=figures_out, selected_epochs=selected_out,
        option_labels=option_labels)

    def _show_selection(change=None):
        index = int(selector.value)
        if index not in browser.png_cache:
            buffer = io.BytesIO()
            browser.figures[index].savefig(
                buffer, format='png', dpi=int(image_dpi), bbox_inches='tight')
            browser.png_cache[index] = buffer.getvalue()
        image.value = browser.png_cache[index]

    selector.observe(_show_selection, names='value')
    _show_selection()
    if display_widget:
        from IPython.display import display
        display(browser.widget)
    return browser
