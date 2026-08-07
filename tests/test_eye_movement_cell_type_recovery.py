import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from retinanalysis.utils.population_code import (
    compare_eye_movement_cell_type_timescales,
    fit_eye_movement_cell_type_recovery,
    plot_eye_movement_cell_type_across_dates,
    plot_eye_movement_cell_type_comparison,
    summarize_eye_movement_cell_type_recovery,
)


def _trajectory(exp_names=('20230101C', '20230202C')):
    rows = []
    times = np.array([2.5, 7.5, 12.5, 17.5, 22.5, 27.5])
    for date_i, exp_name in enumerate(exp_names):
        for cell_type, tau, n_cells in [('OnM', 5.0, 18), ('OnP', 10.0, 31)]:
            for image in range(4):
                curve = 0.15 + 0.7 * (1 - np.exp(-(times - times.min()) / tau))
                curve = curve + 0.01 * image + 0.005 * date_i
                for time, value in zip(times, curve):
                    rows.append({
                        'exp_name': exp_name, 'image': image,
                        'normalize': 'centered', 'cell_type': cell_type,
                        't_since_movie_s': time, 'rho_corrected': value,
                        'n_cells': n_cells,
                    })
    return pd.DataFrame(rows)


def test_eye_movement_cell_type_recovery_fit_and_pairing():
    summary = summarize_eye_movement_cell_type_recovery(_trajectory())
    assert set(summary.cell_type) == {'OnM', 'OnP'}
    assert summary.groupby(['exp_name', 'cell_type']).size().eq(6).all()
    assert np.allclose(
        summary.groupby(['exp_name', 'cell_type']).first().recovery_fraction, 0)

    fits = fit_eye_movement_cell_type_recovery(summary, n_boot=20)
    assert len(fits) == 4
    assert fits.fit_error.eq('').all()
    comparison = compare_eye_movement_cell_type_timescales(fits)
    assert comparison.exp_name.nunique() == 2
    assert (comparison['tau_s_diff_OnM_minus_OnP'] < 0).all()


def test_eye_movement_cell_type_recovery_plots():
    summary = summarize_eye_movement_cell_type_recovery(_trajectory())
    fits = fit_eye_movement_cell_type_recovery(summary, n_boot=10)
    one_date = summary[summary.exp_name == '20230101C']
    one_fit = fits[fits.exp_name == '20230101C']
    fig, axes = plot_eye_movement_cell_type_comparison(one_date, one_fit)
    assert len(axes) == 2
    plt.close(fig)
    fig, axes = plot_eye_movement_cell_type_across_dates(summary, fits)
    assert len(axes) == 3
    plt.close(fig)
