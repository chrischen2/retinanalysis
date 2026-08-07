import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from retinanalysis.utils.analysis_results import load_analysis_bundle

from retinanalysis.utils.population_code import (
    load_eye_movement_many,
    plot_eye_movement_across_dates,
    save_eye_movement_cross_date_summary,
    save_eye_movement_results,
    summarize_eye_movement_dates,
)


def _tables(offset=0.0):
    types = ['OnP', 'OnP', 'OnM', 'OnM']
    cycles = [1, 2, 1, 2]
    similarity = pd.DataFrame({
        'normalize': ['centered'] * 4,
        'cell_type': types,
        'cycle': cycles,
        'rho_corrected': [0.3, 0.7, 0.4, 0.8],
        'rho_cell': [0.2, 0.6, 0.3, 0.7],
        'rate_a_hz': [10, 9, 12, 10],
        'rate_b_hz': [6, 8, 7, 9],
    })
    similarity[['rho_corrected', 'rho_cell']] += offset
    cycle = pd.DataFrame({
        'normalize': ['centered', 'centered'],
        'cell_type': ['OnP', 'OnM'],
        'rho_cycle_corrected': [0.5, 0.6],
        'rho_delta': [0.0, 0.1],
        'delta_ratio': [1.4, 1.5],
    })
    distance = pd.DataFrame({
        'cell_type': types,
        'cycle': cycles,
        'd_excess': [0.2, 0.1, 0.3, 0.15],
        'd_excess_count_only': [0.4, 0.2, 0.5, 0.3],
    })
    trajectory = pd.DataFrame({
        'normalize': ['centered'] * 4,
        'cell_type': types,
        't_since_movie_s': [2.5, 7.5, 2.5, 7.5],
        'rho_corrected': [0.2, 0.6, 0.3, 0.7],
    })
    return similarity, cycle, distance, trajectory


def test_eye_movement_date_and_cross_date_bundles(tmp_path):
    for exp, offset in [('20230101C', 0.0), ('20230202C', 0.05)]:
        sim, cycle, dist, traj = _tables(offset)
        recovery = pd.DataFrame({
            'cell_type': ['OnP', 'OnM'], 't_since_movie_s': [2.5, 2.5],
            'rho_corrected': [0.2, 0.3], 'recovery_fraction': [0.0, 0.0],
        })
        fits = pd.DataFrame({
            'cell_type': ['OnP', 'OnM'], 'tau_s': [9.0, 5.0],
            't50_s': [6.2, 3.5], 'n_cells': [30, 18],
        })
        comparison = pd.DataFrame({
            'cell_type_a': ['OnM'], 'cell_type_b': ['OnP'],
            'tau_s_diff_OnM_minus_OnP': [-4.0],
        })
        save_eye_movement_results(
            exp, similarity=sim, cycle_interaction=cycle,
            spike_distance=dist, trajectory=traj,
            cell_type_recovery=recovery, cell_type_fits=fits,
            cell_type_comparison=comparison,
            output_root=tmp_path, verbose=False)

    combined = load_eye_movement_many(output_root=tmp_path)
    assert combined['similarity']['exp_name'].nunique() == 2
    assert combined['cell_type_recovery_fits']['exp_name'].nunique() == 2
    reduced = summarize_eye_movement_dates(combined)
    assert reduced['similarity_by_date']['exp_name'].nunique() == 2

    fig, axes = plot_eye_movement_across_dates(
        combined, cell_types=['OnP', 'OnM'])
    assert len(axes) == 3
    paths = save_eye_movement_cross_date_summary(
        combined, figures={'cross date': fig}, output_root=tmp_path,
        verbose=False)
    assert paths['folder'] == (tmp_path / 'protocol_analysis' / 'emtraj'
                               / 'summary')
    assert (paths['folder'] / 'plots' / 'cross_date.png').is_file()
    pooled = load_analysis_bundle('emtraj', summary=True, output_root=tmp_path)
    assert pooled['meta']['paired_cell_type_dates'] == 2
    assert pooled['meta']['cell_types'] == ['OnM', 'OnP']
    plt.close(fig)
