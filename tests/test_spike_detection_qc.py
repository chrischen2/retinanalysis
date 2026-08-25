import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

import retinanalysis as ra


def test_plot_spike_detection_qc_samples_reproducible_fraction():
    traces = np.zeros((10, 100), dtype=float)
    traces[:, 20] = -2.0
    spike_times = [np.array([20]) for _ in range(10)]

    figures, selected = ra.plot_spike_detection_qc(
        traces, spike_times, sample_rate=1000.0, fraction=0.30,
        random_state=7, stimulus_window_ms=(10.0, 60.0),
        max_epochs_per_figure=2)
    _, selected_again = ra.plot_spike_detection_qc(
        traces, spike_times, sample_rate=1000.0, fraction=0.30,
        random_state=7)

    assert len(selected) == 3
    assert np.array_equal(selected, selected_again)
    assert np.all(np.diff(selected) > 0)
    assert len(figures) == 2
    assert sum(len(fig.axes) for fig in figures) == 3
    assert figures[0].axes[0].collections  # detected-spike marker
    assert figures[0].axes[0].patches      # stimulus window


def test_plot_spike_detection_qc_accepts_ms_and_none_entries():
    traces = np.zeros((2, 50), dtype=float)
    figures, selected = ra.plot_spike_detection_qc(
        traces, [np.array([10.0]), None], sample_rate=1000.0,
        fraction=1.0, spike_time_unit='ms', random_state=0)

    assert selected.tolist() == [0, 1]
    assert len(figures) == 1
    assert len(figures[0].axes) == 2


def test_plot_spike_detection_qc_can_close_figures_for_widget_display():
    import matplotlib.pyplot as plt

    plt.close('all')
    figures, _ = ra.plot_spike_detection_qc(
        np.zeros((3, 50)), [[], [], []], sample_rate=1000.0,
        fraction=1.0, max_epochs_per_figure=1, close_figures=True)

    assert len(figures) == 3
    assert not plt.get_fignums()
    assert all(len(figure.axes) == 1 for figure in figures)


def test_spike_detection_qc_browser_replaces_one_image_and_deduplicates_sources():
    t = np.linspace(0.0, 4.0 * np.pi, 100)

    def dataset(label, block_id, epoch_index, frequency):
        return {
            'label': label,
            'traces': np.asarray([np.sin(frequency * t)]),
            'spike_times': [np.array([20.0])],
            'sample_rate': 1000.0,
            'spike_time_unit': 'ms',
            'source_group': '2026-08-25_E',
            'block_ids': np.array([block_id]),
            'epoch_indices': np.array([epoch_index]),
        }

    browser = ra.spike_detection_qc_browser(
        [dataset('Condition 1', 100, 7, 1.0),
         dataset('Condition 2 duplicate', 100, 7, 2.0),
         dataset('Condition 3', 101, 7, 3.0)],
        fraction=1.0, random_state=0, display_widget=False, verbose=False)

    assert browser.option_labels == [
        'Condition 1 | block 100 | epoch 7',
        'Condition 3 | block 101 | epoch 7',
    ]
    assert [child.__class__.__name__ for child in browser.widget.children] == [
        'Dropdown', 'Image']
    first_png = browser.image.value
    browser.selector.value = 1
    assert browser.image.value and browser.image.value != first_png
    assert set(browser.png_cache) == {0, 1}


def test_spike_detection_qc_browser_accepts_protocol_specific_epoch_identity():
    browser = ra.spike_detection_qc_browser([{
        'label': 'Protocol A',
        'traces': np.zeros((2, 50)),
        'spike_times': [[], []],
        'sample_rate': 1000.0,
        'epoch_keys': [('file-a', 4), ('file-a', 9)],
        'epoch_labels': ['sweep 4', 'sweep 9'],
    }], fraction=1.0, display_widget=False, verbose=False)

    assert browser.option_labels == [
        'Protocol A | sweep 4', 'Protocol A | sweep 9']


@pytest.mark.parametrize('fraction', [0, -0.1, 1.1])
def test_plot_spike_detection_qc_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match='fraction'):
        ra.plot_spike_detection_qc(
            np.zeros((2, 10)), [[], []], sample_rate=1000.0,
            fraction=fraction)
