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


@pytest.mark.parametrize('fraction', [0, -0.1, 1.1])
def test_plot_spike_detection_qc_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match='fraction'):
        ra.plot_spike_detection_qc(
            np.zeros((2, 10)), [[], []], sample_rate=1000.0,
            fraction=fraction)
