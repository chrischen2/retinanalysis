import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from retinanalysis.utils.response_phase import (
    phase_period_scan,
    plot_phase_alignment,
)


def _geometry():
    return {
        'center_x': 50.0, 'center_y': 40.0,
        'canvas_w': 100, 'canvas_h': 80,
        'aperture_diameter_px': 80.0,
        'aperture_diameter_um': 160.0,
        'microns_per_pixel': 2.0,
        'spatial_freq_cyc_per_px': 1.0 / 40.0,
        'orientation_deg': 0.0,
    }


def _phases():
    geometry = _geometry()
    axis = np.array([-15.0, -5.0, 5.0, 15.0])
    stim = np.pi / 2 - 2 * np.pi * axis / 40.0
    return pd.DataFrame({
        'cell_type': ['OnM'] * 4,
        'center_x': geometry['center_x'] + axis,
        'center_y': [geometry['center_y']] * 4,
        'axis_px': axis,
        'inside_aperture': [True] * 4,
        'f1_strength': [0.5] * 4,
        'resp_phase_rad': stim,
        'residual_rad': np.zeros(4),
    })


def test_phase_scan_reports_micrometers_and_plot_uses_them():
    geometry = _geometry()
    phases = _phases()
    scan = phase_period_scan(
        phases, geometry, orientations_deg=[0.0], n_periods=120,
        min_cells_per_type=3)
    assert np.allclose(scan['periods_um'], scan['periods_px'] * 2.0)
    assert scan['true_period_um'] == 80.0
    assert scan['best_period_um'] == scan['best_period_px'] * 2.0

    fig = plot_phase_alignment(phases, geometry, scan)
    assert 'µm' in fig.axes[0].get_xlabel()
    assert 'µm' in fig.axes[1].get_xlabel()
    assert 'µm' in fig.axes[2].get_xlabel()
    plt.close(fig)
