import json

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from retinanalysis.utils.analysis_results import (
    analysis_output_dir,
    list_analysis_dates,
    load_analysis_bundle,
    save_analysis_bundle,
    save_analysis_summary,
    saved_analysis_stats,
)


def test_date_and_summary_bundles_use_pickle_json_and_plot_folders(
        tmp_path, capsys):
    table = pd.DataFrame({'cell_id': [1, 2], 'value': [0.2, 0.4]})
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0.2, 0.4])

    paths = save_analysis_bundle(
        'shortproc', '20230101C', {'result': table},
        metadata={'datafile': 'data001'}, figures={'main result': fig},
        output_root=tmp_path,
    )
    printed = capsys.readouterr().out
    assert 'dates saved before this update' in printed
    expected = (tmp_path / 'protocol_analysis' / 'shortproc'
                / '20230101C')
    assert paths['folder'] == expected
    assert (expected / 'analysis.pkl').is_file()
    assert (expected / 'meta.json').is_file()
    assert (expected / 'plots' / 'main_result.png').is_file()
    assert not list(expected.glob('*.csv'))

    with (expected / 'meta.json').open() as handle:
        meta = json.load(handle)
    assert meta['datafile'] == 'data001'
    assert meta['objects']['result']['rows'] == 2
    loaded = load_analysis_bundle(
        'shortproc', '20230101C', output_root=tmp_path)
    pd.testing.assert_frame_equal(loaded['analysis']['result'], table)
    assert list_analysis_dates('shortproc', output_root=tmp_path) == ['20230101C']
    assert saved_analysis_stats('shortproc', output_root=tmp_path).loc[
        0, 'n_plots'] == 1

    summary = save_analysis_summary(
        'shortproc', {'combined': table}, figures={'pooled': fig},
        output_root=tmp_path)
    expected_summary = (tmp_path / 'protocol_analysis' / 'shortproc'
                        / 'summary')
    assert summary['folder'] == expected_summary
    assert (expected_summary / 'analysis.pkl').is_file()
    assert (expected_summary / 'meta.json').is_file()
    assert (expected_summary / 'plots' / 'pooled.png').is_file()
    assert analysis_output_dir(
        'shortproc', summary=True, output_root=tmp_path) == expected_summary
    plt.close(fig)


def test_resaving_date_exactly_replaces_data_metadata_and_plot_set(
        tmp_path, capsys):
    old_table = pd.DataFrame({'cell_id': [1], 'value': [0.2]})
    new_table = pd.DataFrame({'cell_id': [2, 3], 'value': [0.7, 0.9]})
    old_fig, old_ax = plt.subplots()
    old_ax.plot([0, 1])
    stale_fig, stale_ax = plt.subplots()
    stale_ax.plot([1, 0])
    new_fig, new_ax = plt.subplots()
    new_ax.plot([0.7, 0.9])

    save_analysis_bundle(
        'shortproc', '20230101C', {'result': old_table},
        metadata={'run': 'old'},
        figures={'main': old_fig, 'stale': stale_fig},
        output_root=tmp_path,
    )
    paths = save_analysis_bundle(
        'shortproc', '20230101C', {'result': new_table},
        metadata={'run': 'new'}, figures={'main': new_fig},
        output_root=tmp_path,
    )

    printed = capsys.readouterr().out
    assert 'Replacing existing shortproc/20230101C' in printed
    loaded = load_analysis_bundle(
        'shortproc', '20230101C', output_root=tmp_path)
    pd.testing.assert_frame_equal(loaded['analysis']['result'], new_table)
    assert loaded['meta']['run'] == 'new'
    assert loaded['meta']['write_mode'] == 'overwrite_existing'
    assert loaded['meta']['plots'] == ['plots/main.png']
    assert sorted(path.name for path in paths['plots'].glob('*.png')) == [
        'main.png']
    assert list_analysis_dates('shortproc', output_root=tmp_path) == [
        '20230101C']

    plt.close(old_fig)
    plt.close(stale_fig)
    plt.close(new_fig)
