import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from retinanalysis.utils.spatial_recovery import (
    plot_population_modulation_over_time,
    population_modulation_summary,
)


def _modulation():
    rows = []
    for start in (0.5, 5.0):
        end = 5.0 if start == 0.5 else 10.0
        for cell_id, cell_type, rate, m1 in (
                (1, 'OnM', 10.0, 0.10), (2, 'OnM', 20.0, 0.20),
                (3, 'OnP', 30.0, 0.30), (4, 'OnP', 40.0, 0.40)):
            rows.append({
                'cell_id': cell_id, 'cell_type': cell_type,
                't_start': start, 't_end': end,
                't_mid': (start + end) / 2, 'f0_hz': rate, 'm1': m1,
            })
    return pd.DataFrame(rows)


def test_population_modulation_summary_converts_depth_and_hz():
    summary = population_modulation_summary(_modulation())
    onm = summary.query("cell_type == 'OnM'").iloc[0]
    assert onm.n_cells == 2
    assert np.isclose(onm.modulation_depth, 0.30)
    # Mean across cells of 2 * rate * m1: mean(2, 8) = 5 Hz.
    assert np.isclose(onm.modulation_amplitude_hz, 5.0)


def test_population_modulation_plot_has_absolute_and_relative_panels():
    summary = population_modulation_summary(_modulation())
    fig, axes = plot_population_modulation_over_time(
        summary, cell_types=('OnM', 'OnP'))
    assert len(axes) == 2
    assert 'Hz' in axes[0].get_ylabel()
    assert 'mean rate' in axes[1].get_ylabel()
    plt.close(fig)
