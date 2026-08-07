import matplotlib
import numpy as np
import pandas as pd
from types import SimpleNamespace

matplotlib.use('Agg')

from retinanalysis.utils.spatial_recovery import (
    load_recovery_many,
    normalize_recovery_summary,
    plot_recovery_across_dates,
    save_recovery_summary,
    saved_recovery_stats,
    recovery_summary_table,
)


def _summary(exp_name, rate, f1, decode):
    return pd.DataFrame({
        'exp_name': [exp_name, exp_name],
        'condition': ['bar=150, mean=0.3'] * 2,
        'window': ['0–5s', '5–10s'],
        't_start': [0.0, 5.0],
        't_end': [5.0, 10.0],
        't_mid': [2.5, 7.5],
        'rate_hz': rate,
        'f1': f1,
        'decode_matched': decode,
        'decode_matched_chance': [0.1, 0.1],
        'n_cells': [40, 40],
        'n_epochs': [4, 4],
    })


def test_normalize_recovery_summary_within_each_date():
    raw = pd.concat([
        _summary('20230101C', [2.0, 4.0], [0.2, 0.4], [0.25, 0.5]),
        _summary('20230202C', [6.0, 3.0], [0.6, 0.3], [0.6, 0.4]),
    ], ignore_index=True)

    normalized = normalize_recovery_summary(raw)
    a = normalized.query("exp_name == '20230101C'")
    b = normalized.query("exp_name == '20230202C'")
    np.testing.assert_allclose(a['rate_late_fraction'], [0.5, 1.0])
    np.testing.assert_allclose(a['f1_late_fraction'], [0.5, 1.0])
    np.testing.assert_allclose(a['decode_matched_index'], [0.375, 1.0])
    np.testing.assert_allclose(b['rate_late_fraction'], [2.0, 1.0])
    np.testing.assert_allclose(b['f1_late_fraction'], [2.0, 1.0])
    np.testing.assert_allclose(b['decode_matched_index'], [5.0 / 3.0, 1.0])


def test_save_lists_existing_dates_then_loads_combined_dataset(tmp_path, capsys):
    save_recovery_summary(
        _summary('20230101C', [2.0, 4.0], [0.2, 0.4], [0.25, 0.5]),
        '20230101C', output_root=tmp_path,
    )
    capsys.readouterr()

    save_recovery_summary(
        _summary('20230202C', [6.0, 3.0], [0.6, 0.3], [0.6, 0.4]),
        '20230202C', output_root=tmp_path,
    )
    printed = capsys.readouterr().out
    assert 'vmdg dates saved before this update' in printed
    assert '20230101C' in printed

    combined = load_recovery_many(output_root=tmp_path)
    assert sorted(combined['exp_name'].unique()) == ['20230101C', '20230202C']
    stats = saved_recovery_stats(output_root=tmp_path)
    assert stats[['exp_name', 'n_conditions', 'n_windows']].to_dict('records') == [
        {'exp_name': '20230101C', 'n_conditions': 1, 'n_windows': 2},
        {'exp_name': '20230202C', 'n_conditions': 1, 'n_windows': 2},
    ]

    fig, axes = plot_recovery_across_dates(
        combined, condition='bar=150, mean=0.3',
    )
    assert len(axes) == 3
    fig.clf()


def test_recovery_summary_table_builds_stable_condition_window_rows():
    windows = ['0–5s', '5–10s']
    timing = {
        'window': windows,
        't_start': [0.0, 5.0],
        't_end': [5.0, 10.0],
        't_mid': [2.5, 7.5],
    }
    modulation = pd.DataFrame({
        **{key: np.repeat(value, 2) for key, value in timing.items()},
        'cell_id': [1, 2, 1, 2],
        'f0_hz': [2.0, 4.0, 4.0, 8.0],
        'm1': [0.2, 0.4, 0.4, 0.8],
        'm2': [0.1, 0.2, 0.2, 0.4],
        'f1_resolved': [True, True, True, True],
    })
    naive = modulation.copy()
    naive['m1'] += 0.05

    def decoding(values, chance, with_shuffle):
        frame = pd.DataFrame({
            'window': windows,
            'accuracy': values,
            'chance_accuracy': [chance, chance],
        })
        if with_shuffle:
            frame['shuffle_accuracy'] = [chance, chance]
            frame['shuffle_accuracy_sd'] = [0.01, 0.01]
        return frame

    pbr = SimpleNamespace(
        counts=np.zeros((4, 12, 2)), epochs=[0, 1, 2, 3],
        geometry={'bar_width_um': 150.0, 'mean_intensity': 0.3},
        drift_freq_hz=2.01,
    )
    recovery = {'bar=150, mean=0.3': {
        'pbr': pbr,
        'modulation': modulation,
        'modulation_naive': naive,
        'full': decoding([0.3, 0.7], 1 / 12, True),
        'matched': decoding([0.25, 0.5], 1 / 12, False),
        'polarity_blind': decoding([0.2, 0.6], 1 / 6, True),
        'coherence': pd.DataFrame(),
        'reliability': pd.DataFrame({
            'window': windows,
            'reliability': [0.5, 0.7],
            'reliability_sd': [0.1, 0.1],
        }),
    }}

    summary = recovery_summary_table(recovery, exp_name='20230101C')
    assert summary['window'].tolist() == windows
    np.testing.assert_allclose(summary['rate_hz'], [3.0, 6.0])
    np.testing.assert_allclose(summary['f1'], [0.3, 0.6])
    np.testing.assert_allclose(summary['f1_late_fraction'], [0.5, 1.0])
    np.testing.assert_allclose(summary['decode_matched_index'], [0.4, 1.0])
    assert summary['n_cells'].tolist() == [2, 2]
    assert summary['n_epochs'].tolist() == [4, 4]
