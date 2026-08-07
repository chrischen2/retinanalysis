from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')

from retinanalysis.utils.spatial_recovery import (
    cell_type_recovery_summary,
    compare_cell_type_timescales,
    fit_cell_type_recovery,
    plot_cell_type_recovery_across_dates,
    plot_cell_type_recovery_comparison,
)


def _recovery():
    times = [1.0, 5.0, 15.0, 30.0]
    cells = [(1, 'OnM'), (2, 'OnM'), (3, 'OnP'), (4, 'OnP')]
    profiles = {
        1: [1.0, 2.0, 3.0, 4.0],
        2: [2.0, 4.0, 6.0, 8.0],
        3: [1.0, 3.0, 5.0, 6.0],
        4: [2.0, 6.0, 10.0, 12.0],
    }
    rows = []
    for wi, t in enumerate(times):
        for cell_id, cell_type in cells:
            value = profiles[cell_id][wi]
            rows.append({
                'window': f'w{wi}', 't_start': t - 0.5, 't_end': t + 0.5,
                't_mid': t, 'cell_id': cell_id, 'cell_type': cell_type,
                'f0_hz': value * 10, 'm1': value, 'm2': value / 2,
                'f1_resolved': True,
            })
    pbr = SimpleNamespace(
        geometry={'bar_width_um': 150.0, 'mean_intensity': 0.3},
        drift_freq_hz=2.01, epochs=[0, 1, 2],
    )
    return {'coarse': {'pbr': pbr, 'modulation': pd.DataFrame(rows)}}


def test_cell_type_summary_normalizes_each_cell_before_averaging():
    summary = cell_type_recovery_summary(
        _recovery(), exp_name='20230101C', cell_types=('OnM', 'OnP'))
    onm = summary.query("cell_type == 'OnM'").sort_values('t_mid')
    onp = summary.query("cell_type == 'OnP'").sort_values('t_mid')

    np.testing.assert_allclose(onm['f1_late_fraction'], [.25, .5, .75, 1])
    np.testing.assert_allclose(onp['f1_late_fraction'], [1 / 6, .5, 5 / 6, 1])
    assert onm['n_cells'].tolist() == [2, 2, 2, 2]
    assert set(summary['cell_type']) == {'OnM', 'OnP'}


def test_cell_type_fits_and_comparison_keep_pathway_identity():
    summary = cell_type_recovery_summary(
        _recovery(), exp_name='20230101C', cell_types=('OnM', 'OnP'))
    fits = fit_cell_type_recovery(summary, n_boot=0)
    comparison = compare_cell_type_timescales(fits)

    assert set(fits['cell_type']) == {'OnM', 'OnP'}
    assert fits['tau_s'].notna().all()
    assert comparison.loc[0, 'cell_type_a'] == 'OnM'
    assert comparison.loc[0, 'cell_type_b'] == 'OnP'
    np.testing.assert_allclose(
        comparison.loc[0, 'tau_s_diff_OnM_minus_OnP'],
        comparison.loc[0, 'tau_s_OnM'] - comparison.loc[0, 'tau_s_OnP'])


def test_cell_type_comparison_plots_single_and_multiple_dates():
    one = cell_type_recovery_summary(
        _recovery(), exp_name='20230101C', cell_types=('OnM', 'OnP'))
    two = one.copy()
    two['exp_name'] = '20230202C'
    combined = pd.concat([one, two], ignore_index=True)
    fits = fit_cell_type_recovery(combined, n_boot=0)

    fig1, axes1 = plot_cell_type_recovery_comparison(
        one, fits.query("exp_name == '20230101C'"), condition='coarse')
    fig2, axes2 = plot_cell_type_recovery_across_dates(
        combined, fits, condition='coarse')
    assert len(axes1) == 2
    assert len(axes2) == 3
    fig1.clf()
    fig2.clf()
