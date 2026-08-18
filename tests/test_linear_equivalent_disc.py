"""Tests for SCutils.protocols.linear_equivalent_disc — pure helpers only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retinanalysis.SCutils.protocols import linear_equivalent_disc as led


def test_analysis_notebook_selects_retinanalysis_kernel():
    notebook_path = Path(__file__).parents[1] / 'SingCell_Notebooks' / 'analyzeConeDisc.ipynb'
    notebook = json.loads(notebook_path.read_text())

    assert notebook['metadata']['kernelspec']['name'] == 'retinanalysis'


# --- stimulus tag mapping --------------------------------------------------

@pytest.mark.parametrize('tag, expected', [
    ('image', 'image'),
    ('intensity', 'disc'),
    ('linConeIntensity', 'cone_disc'),      # LinearEquivalentDiscConeLin spelling
    ('lin cone intensity', 'cone_disc'),    # Annulus / LinearEquivalentDisc spelling
    ('  image  ', 'image'),
    ('something else', ''),
])
def test_category_of_handles_both_spellings(tag, expected):
    assert led.category_of(tag) == expected


# --- NLI -------------------------------------------------------------------

def test_compute_nli_matches_the_matlab_formula():
    img, disc = np.array([10.0, 4.0]), np.array([5.0, 8.0])
    expected = (img - disc) / (np.abs(img) + np.abs(disc))
    assert np.allclose(led.compute_nli(img, disc, threshold=0.0), expected)


def test_compute_nli_is_zero_below_threshold():
    """Neither response clears the threshold, so the index is noise -> 0."""
    assert led.compute_nli([1.0], [0.5], threshold=3.0).tolist() == [0.0]


def test_compute_nli_keeps_patches_where_one_response_clears_threshold():
    out = led.compute_nli([10.0], [0.0], threshold=3.0)
    assert out.tolist() == [1.0]


def test_compute_nli_drops_non_finite():
    out = led.compute_nli([0.0, 10.0], [0.0, 5.0], threshold=0.0)
    assert out.size == 1 and out[0] == pytest.approx(1 / 3)


def test_compute_nli_bounds():
    rng = np.random.RandomState(0)
    img, disc = rng.uniform(0, 100, 200), rng.uniform(0, 100, 200)
    out = led.compute_nli(img, disc, threshold=0.0)
    assert np.all(out >= -1) and np.all(out <= 1)


def test_nli_sign_means_image_preferred():
    assert led.compute_nli([10.0], [2.0], 0.0)[0] > 0     # image drove the cell more
    assert led.compute_nli([2.0], [10.0], 0.0)[0] < 0


def test_threshold_table_matches_matlab():
    """These stay in the MATLAB's per-window units: spikes, and pA*s."""
    assert led.NLI_THRESHOLD == {'extracellular': 3.0, 'exc': 10.0, 'inh': 5.0}


def test_spike_thresholds_convert_to_rates_per_window():
    """3 spikes in 0.2 s is 15 Hz; the offset window is longer, so its rate is lower."""
    on, off = led.working_thresholds('extracellular', stim_s=0.2, offset_s=0.4, spiking=True)
    assert on == pytest.approx(15.0)
    assert off == pytest.approx(7.5)


def test_whole_cell_thresholds_use_the_stimulus_duration_for_both_windows():
    """The MATLAB integrated both windows with stimPts, so both scale the same."""
    on, off = led.working_thresholds('exc', stim_s=0.2, offset_s=0.4, spiking=False)
    assert on == off == pytest.approx(50.0)          # 10 pA*s / 0.2 s
    assert led.working_thresholds('inh', 0.2, 0.4, spiking=False)[0] == pytest.approx(25.0)


def test_threshold_conversion_preserves_which_patches_are_kept():
    """Rescaling responses and threshold together must not change the NLI."""
    stim_s = 0.2
    counts_img, counts_disc = np.array([2.0, 10.0]), np.array([1.0, 4.0])
    as_counts = led.compute_nli(counts_img, counts_disc, led.NLI_THRESHOLD['extracellular'])
    on, _ = led.working_thresholds('extracellular', stim_s, stim_s, spiking=True)
    as_rates = led.compute_nli(counts_img / stim_s, counts_disc / stim_s, on)
    assert np.allclose(as_counts, as_rates)


def test_zero_duration_falls_back_to_the_raw_threshold():
    on, off = led.working_thresholds('extracellular', stim_s=0.0, offset_s=0.0, spiking=True)
    assert on == 3.0 and off == 3.0


# --- protocol handling -----------------------------------------------------

def test_stimulus_site_from_protocol_name():
    assert led.stimulus_site('LinearEquivalentAnnulus') == 'surround'
    assert led.stimulus_site('LinearEquivalentDiscConeLin') == 'center'
    assert led.stimulus_site('LinearEquivalentDisc') == 'center'


def test_only_linear_equivalent_disc_needs_the_filter():
    """The other two protocols always carry linearizeCones, so they are never filtered."""
    assert led.NEEDS_LINEARIZE_FILTER == ('LinearEquivalentDisc',)
    assert set(led.PROTOCOLS) == {'LinearEquivalentDiscConeLin', 'LinearEquivalentAnnulus',
                                  'LinearEquivalentDisc'}


def test_find_blocks_display_is_compact():
    from retinanalysis.SCutils import explore as sc

    assert 'protocol' not in led.FIND_BLOCKS_DISPLAY_COLUMNS
    assert not ({'onlineAnalysis', 'backgroundIntensity', 'light_setting',
                 'WeberConstant', 'block_id'} & set(led.FIND_BLOCKS_DISPLAY_COLUMNS))
    assert {'exp_name', 'cell_label', 'cell_type_short', 'n_epochs'}.issubset(
        led.FIND_BLOCKS_DISPLAY_COLUMNS)
    frame = pd.DataFrame([{column: 'x' for column in led.FIND_BLOCKS_DISPLAY_COLUMNS}])
    html = sc.scroll_table(frame, show=False)
    assert all(f'<th>{column}</th>' in html for column in led.FIND_BLOCKS_DISPLAY_COLUMNS)
    assert all(f'<th>{column}</th>' not in html for column in
               ('onlineAnalysis', 'backgroundIntensity', 'light_setting',
                'WeberConstant', 'block_id'))


def test_linearized_only_filters_only_old_same_named_disc_blocks():
    frame = pd.DataFrame({
        'protocol': ['LinearEquivalentDisc', 'LinearEquivalentDisc',
                     'LinearEquivalentDiscConeLin', 'LinearEquivalentAnnulus'],
        'parameters': [{}, {'linearizeCones': True}, {}, {}],
    })
    kept, dropped = led._linearized_only(frame)
    assert dropped == 1
    assert kept['protocol'].tolist() == [
        'LinearEquivalentDisc', 'LinearEquivalentDiscConeLin',
        'LinearEquivalentAnnulus']


def test_find_protocol_cells_returns_only_unique_date_and_cell(monkeypatch):
    blocks = pd.DataFrame({
        'exp_name': ['2026-01-01_E', '2026-01-01_E', '2026-01-02_E'],
        'cell_label': ['Cell1', 'Cell1', 'Cell2'],
        'protocol': ['LinearEquivalentAnnulus'] * 3,
        'parameters': [{}] * 3,
    })
    monkeypatch.setattr(led, '_protocol_block_rows', lambda protocols: blocks)
    found = led.find_protocol_cells('LinearEquivalentAnnulus', show=False)
    assert list(found.columns) == ['exp_name', 'cell_label', 'protocol']
    assert found.to_dict('records') == [
        {'exp_name': '2026-01-01_E', 'cell_label': 'Cell1',
         'protocol': 'LinearEquivalentAnnulus'},
        {'exp_name': '2026-01-02_E', 'cell_label': 'Cell2',
         'protocol': 'LinearEquivalentAnnulus'},
    ]


def test_protocol_cells_from_blocks_reuses_detailed_discovery():
    blocks = pd.DataFrame({
        'exp_name': ['2026-01-02_E', '2026-01-01_E', '2026-01-01_E'],
        'cell_label': ['Cell2', 'Cell1', 'Cell1'],
        'cell_type_short': ['OFF-parasol', 'ON-parasol', 'ON-parasol'],
        'protocol': ['LinearEquivalentAnnulus'] * 3,
        'group_properties': [
            {'recordingTechnique': 'whole-cell'},
            {'recordingTechnique': 'cell-attached'},
            {'recordingTechnique': 'cell-attached'},
        ],
        'onlineAnalysis': ['exc', 'extracellular', 'extracellular'],
        'filter_wheel_ndf': [1.0, 0.0, 0.5],
        'block_id': [3, 1, 2],
    })
    found = led.protocol_cells_from_blocks(blocks, show=False)
    assert found.to_dict('records') == [
        {'date_index': 1, 'exp_name': '2026-01-01_E', 'cell_label': 'Cell1',
         'cell_type_short': 'ON-parasol', 'recording_technique': 'cell-attached',
         'onlineAnalysis': 'extracellular', 'filter_wheel_values': [0.0, 0.5],
         'protocol': 'LinearEquivalentAnnulus'},
        {'date_index': 2, 'exp_name': '2026-01-02_E', 'cell_label': 'Cell2',
         'cell_type_short': 'OFF-parasol', 'recording_technique': 'whole-cell',
         'onlineAnalysis': 'exc', 'filter_wheel_values': [1.0],
         'protocol': 'LinearEquivalentAnnulus'},
    ]


def test_describe_experiment_protocol_builds_group_and_block_columns(monkeypatch):
    blocks = pd.DataFrame({
        'exp_name': ['2026-01-01_E'],
        'cell_label': ['Cell1'],
        'epoch_group': ['Control'],
        'group_properties': [{'recordingTechnique': 'cell-attached'}],
        'parameters': [{'onlineAnalysis': 'none', 'imageName': 'block-image'}],
        'block_id': [123],
        'protocol': ['LinearEquivalentAnnulus'],
        'start_time': pd.to_datetime(['2026-01-01 12:00']),
    })
    monkeypatch.setattr(led, '_protocol_block_rows',
                        lambda protocols, exp_names=None: blocks)
    monkeypatch.setattr(led, '_first_epoch_metadata',
                        lambda block_ids: ({123: {'onlineAnalysis': 'extracellular',
                                                  'NDF': 0.5,
                                                  'imageName': 'epoch-image'}},
                                           pd.Series({123: 10})))
    found = led.describe_experiment_protocol(
        '2026-01-01_E', 'LinearEquivalentAnnulus', show=False)
    assert list(found.columns) == [
        'exp_name', 'cell_label', 'epoch_group', 'recording_technique',
        'onlineAnalysis', 'block_id', 'protocol', 'filter_wheel_ndf', 'imageName']
    assert found.iloc[0].to_dict() == {
        'exp_name': '2026-01-01_E', 'cell_label': 'Cell1', 'epoch_group': 'Control',
        'recording_technique': 'cell-attached', 'onlineAnalysis': 'extracellular',
        'block_id': 123, 'protocol': 'LinearEquivalentAnnulus',
        'filter_wheel_ndf': 0.5, 'imageName': 'epoch-image'}


def test_record_key_is_hdf5_safe_and_separates_sites():
    a = led.record_key('2026-05-08_E', 'Cell4', 'extracellular', 'center', 0.0, 0.5)
    b = led.record_key('2026-05-08_E', 'Cell4', 'extracellular', 'surround', 0.0, 0.5)
    assert a != b
    for key in (a, b):
        assert '/' not in key and '.' not in key
    assert 'FWNaN' in led.record_key('x', 'C1', 'exc', 'center', float('nan'), 0.15)


def test_shared_light_helpers_are_reused():
    from retinanalysis.SCutils.protocols import spot_annular_grating as sag
    for name in ('light_level_rstar', 'light_setting', 'apply_rstar_mapping'):
        assert getattr(led, name) is getattr(sag, name)


# --- recording mode from the amplifier -------------------------------------

def test_recording_mode_helpers_are_the_shared_ones():
    """One implementation, so every protocol resolves the mode identically."""
    from retinanalysis.SCutils import recording_mode as rm
    from retinanalysis.SCutils.protocols import spot_annular_grating as sag
    for name in ('resolve_recording_mode', 'check_series_resistance',
                 'read_series_resistance', 'trace_is_spiking', 'read_stage_ndfs'):
        assert getattr(led, name) is getattr(rm, name)
        assert getattr(sag, name) is getattr(rm, name)


def test_series_resistance_table_uses_one_batched_path_lookup(tmp_path, monkeypatch):
    import h5py
    from retinanalysis.SCutils import recording_mode as rm
    from retinanalysis.utils import datajoint_utils

    h5_path = tmp_path / 'recordings.h5'
    with h5py.File(h5_path, 'w') as h5:
        for block_id, values in {1: [0.0, 0.0], 2: [8e6, 25e6]}.items():
            for epoch_index, value in enumerate(values):
                epoch = h5.create_group(f'block-{block_id}/epoch-{epoch_index}')
                amp_node = epoch.create_group(
                    'stimuli/Amp1-device/dataConfigurationSpans/span_0/Amp1')
                amp_node.attrs['seriesResistance'] = value

    calls = []

    def batched(block_ids, amp='Amp1'):
        calls.append((list(block_ids), amp))
        return {
            1: ['block-1/epoch-0', 'block-1/epoch-1'],
            2: ['block-2/epoch-0', 'block-2/epoch-1'],
        }

    monkeypatch.setattr(rm, '_amp_epoch_groups_by_block', batched)
    monkeypatch.setattr(datajoint_utils, 'get_h5_file', lambda exp_name: str(h5_path))
    monkeypatch.setattr(
        rm, 'read_series_resistance',
        lambda *args, **kwargs: pytest.fail('per-block response query should not run'))

    blocks = pd.DataFrame({'exp_name': ['X_E', 'X_E'], 'block_id': [1, 2]})
    result = rm.series_resistance_table(blocks, verbose=False)
    sampled = rm.series_resistance_table(
        blocks, verbose=False, sample_one_per_block=True)

    assert calls == [([1, 2], 'Amp1'), ([1, 2], 'Amp1')]
    assert result.loc[0, 'series_resistance'] == 0.0
    assert result.loc[1, 'series_resistance'] == pytest.approx(16.5e6)
    assert result.loc[1, 'n_epochs_high_rs'] == 1
    assert sampled.loc[1, 'series_resistance'] == pytest.approx(8e6)
    assert sampled.loc[1, 'n_epochs_rs'] == 2
    assert sampled.loc[1, 'n_epochs_high_rs'] == 0


def _spiking_trace(n=12, length=3000, seed=0):
    rng = np.random.RandomState(seed)
    data = rng.randn(n, length)
    for trial in range(n):
        for centre in rng.choice(np.arange(50, length - 50), 20, replace=False):
            data[trial, centre - 2:centre + 3] += np.array([6., -12., -40., -12., 6.])
    return data


def test_unset_label_is_the_common_case_and_resolves_from_the_amplifier():
    """'none' covers 142 of 331 blocks here, so it must resolve, not pass through."""
    mode, note = led.resolve_recording_mode('none', 0.0, _spiking_trace(), 1e4)
    assert mode == 'extracellular' and 'contains spikes' in note

    assert led.resolve_recording_mode('none', 7e6, np.full((4, 100), -5.0))[0] == 'exc'
    assert led.resolve_recording_mode('none', 7e6, np.full((4, 100), 5.0))[0] == 'inh'


def _one_block(**overrides):
    row = {'exp_name': 'X_E', 'block_id': 1, 'cell_label': 'Cell1',
           'cell_type_short': 'ON-parasol', 'onlineAnalysis': 'exc', 'site': 'center',
           'filter_wheel_ndf': 0.0, 'backgroundIntensity': 0.5, 'n_epochs': 10,
           'protocol': 'P', 'light_setting': 'FW0/bg0.5', 'rstar': np.nan,
           'light_level': '?', 'WeberConstant': 0.1, 'maxIntensity': 7500.0}
    row['imageName'] = '00152'
    row.update(overrides)
    return row


def test_group_blocks_warns_when_labels_were_never_resolved(capsys):
    df = pd.DataFrame([_one_block(block_id=1, onlineAnalysis='none'),
                       _one_block(block_id=2, onlineAnalysis='exc')])
    led.group_blocks(df, show=True)
    out = capsys.readouterr().out
    assert 'unresolved onlineAnalysis' in out and 'check_series_resistance' in out


def test_group_blocks_carries_max_intensity_and_the_recorded_label():
    df = pd.DataFrame([_one_block()])
    g = led.group_blocks(df, show=False)
    assert g.loc[0, 'max_intensity'] == 7500.0
    assert g.loc[0, 'maxIntensity'] == 7500.0
    assert g.loc[0, 'image_names'] == ['00152']
    assert g.loc[0, 'recorded_labels'] == 'exc'


def test_group_table_display_keeps_only_requested_columns(monkeypatch):
    from retinanalysis.SCutils import explore as sc

    captured = {}
    monkeypatch.setattr(
        sc, 'tree_table',
        lambda frame, **kwargs: captured.update(frame=frame.copy(), kwargs=kwargs))
    led.group_blocks(pd.DataFrame([_one_block()]), show=True)

    assert tuple(captured['frame'].columns) == led.GROUP_DISPLAY_COLUMNS
    assert captured['kwargs']['levels'] == ['exp_name', 'cell_label', 'cell_type_short']


def test_group_blocks_prefers_the_pre_relabel_column_for_recorded_labels():
    df = pd.DataFrame([_one_block(onlineAnalysis='extracellular')])
    df['onlineAnalysis_recorded'] = 'none'
    g = led.group_blocks(df, show=False)
    # Grouped by what it is analyzed as, but reporting what was recorded.
    assert g.loc[0, 'onlineAnalysis'] == 'extracellular'
    assert g.loc[0, 'recorded_labels'] == 'none'


def test_select_condition_blocks_uses_cell_mode_and_filter_wheel(monkeypatch):
    blocks = pd.DataFrame([
        _one_block(block_id=1, imageName='00152', n_epochs=120,
                   backgroundIntensity=0.2, maxIntensity=1000.0),
        _one_block(block_id=2, imageName='01769', n_epochs=90,
                   backgroundIntensity=0.1, maxIntensity=1000.0),
        _one_block(block_id=3, imageName='00152', filter_wheel_ndf=1.0),
        _one_block(block_id=4, imageName='00152', onlineAnalysis='inh'),
        _one_block(block_id=5, imageName='00152', cell_label='Cell2'),
    ])
    blocks['start_time'] = pd.date_range('2026-01-01', periods=len(blocks))
    selected = led.select_condition_blocks(
        blocks, 'Cell1', 'exc', 0.0, show=False)

    assert selected['block_id'].tolist() == [1, 2]
    summary = selected.attrs['image_summary']
    assert summary[['imageName', 'epochs']].to_dict('records') == [
        {'imageName': '00152', 'epochs': 120},
        {'imageName': '01769', 'epochs': 90},
    ]
    assert summary['meanIntensity'].tolist() == [200.0, 100.0]


def test_condition_summary_does_not_merge_reused_patch_indices():
    epochs = pd.DataFrame([
        {'imageName': image, 'patchIndex': 1.0, 'category': category, 'response': value}
        for image, values in [('A', (2, 1, 2)), ('B', (10, 5, 8))]
        for category, value in zip(('image', 'disc', 'cone_disc'), values)
    ])
    patches = led.summarize_patch_responses(epochs, threshold=0.0)

    assert patches['patch_key'].tolist() == ['A:1', 'B:1']
    assert patches['image_mean'].tolist() == [2.0, 10.0]
    assert patches['nli_disc'].tolist() == pytest.approx([1 / 3, 1 / 3])


def test_condition_patch_error_bars_are_sem_across_repeats():
    epochs = pd.DataFrame([
        {'imageName': 'A', 'patchIndex': 1.0, 'category': category, 'response': response}
        for category, responses in {
            'image': [2.0, 4.0], 'disc': [1.0, 3.0],
            'cone_disc': [2.0, 2.0],
        }.items()
        for response in responses
    ])
    patch = led.summarize_patch_responses(epochs, threshold=3.0).iloc[0]

    assert patch['image_mean'] == 3.0
    assert patch['image_sem'] == pytest.approx(1.0)
    assert patch['disc_sem'] == pytest.approx(1.0)
    assert patch['cone_disc_sem'] == 0.0
    assert patch['image_n'] == patch['disc_n'] == patch['cone_disc_n'] == 2
    assert patch['nli_disc'] == pytest.approx(0.2)


def _condition_analysis_for_output():
    patches = pd.DataFrame({
        'imageName': ['00152', '00152', '01769'],
        'patchIndex': [1.0, 2.0, 1.0],
        'patch_key': ['00152:1', '00152:2', '01769:1'],
        'image_mean': [4.0, 8.0, 6.0], 'image_sem': [1.0, .5, .25],
        'image_n': [2, 2, 2],
        'disc_mean': [2.0, 4.0, 3.0], 'disc_sem': [.5, .25, .1],
        'disc_n': [2, 2, 2],
        'cone_disc_mean': [3.0, 7.0, 5.0], 'cone_disc_sem': [.4, .3, .2],
        'cone_disc_n': [2, 2, 2],
        'nli_disc': [1 / 3, 1 / 3, 1 / 3],
        'nli_cone_disc': [1 / 7, 1 / 15, 1 / 11],
    })
    image_summary = pd.DataFrame({
        'imageName': ['00152', '01769'], 'block_ids': [[11], [12]],
        'epochs': [12, 6], 'maxIntensity': [1000.0, 1000.0],
        'backgroundIntensity': [.2, .1], 'meanIntensity': [200.0, 100.0],
    })
    return led.ConditionAnalysis(
        exp_name='2026-01-01_E', cell_label='Cell1', cell_type='ON-parasol',
        online_analysis='extracellular', filter_wheel_ndf=0.0,
        block_ids=[11, 12], protocols=['LinearEquivalentAnnulus'], site='surround',
        image_summary=image_summary, epoch_responses=pd.DataFrame(),
        patch_responses=patches, units='spikes', threshold=3.0)


def test_condition_population_table_labels_every_patch_and_image_intensity():
    table = led.condition_population_table(_condition_analysis_for_output())

    assert len(table) == 3
    assert table['date'].unique().tolist() == ['2026-01-01_E']
    assert table['onlineAnalysis'].unique().tolist() == ['extracellular']
    assert table['patch_key'].tolist() == ['00152:1', '00152:2', '01769:1']
    assert table['meanIntensity'].tolist() == [200.0, 200.0, 100.0]
    assert {'image_response', 'disc_response', 'cone_disc_response',
            'nli_image_vs_disc', 'nli_image_vs_cone_disc'}.issubset(table.columns)


def test_condition_output_save_is_idempotent_and_loads_population_rows(tmp_path):
    import h5py

    analysis = _condition_analysis_for_output()
    first = led.save_condition_output(analysis, output_dir=tmp_path, verbose=False)
    second = led.save_condition_output(analysis, output_dir=tmp_path, verbose=False)
    loaded = led.load_condition_outputs(tmp_path)

    assert first == second
    assert len(list(tmp_path.glob('*.h5'))) == 1
    assert not list(tmp_path.glob('*.csv'))
    with h5py.File(first, 'r') as stored:
        assert set(stored) == {'block_ids', 'image_summary', 'patch_responses'}
        assert stored['patch_responses/image_mean'].compression == 'gzip'
        assert len(stored['patch_responses/patch_key']) == 3
    assert len(loaded) == 3
    assert loaded['imageName'].tolist() == ['00152', '00152', '01769']


def test_condition_index_lists_metadata_without_expanding_patch_rows(tmp_path):
    from dataclasses import replace

    analysis = _condition_analysis_for_output()
    led.save_condition_output(analysis, output_dir=tmp_path, verbose=False)
    led.save_condition_output(analysis, output_dir=tmp_path, verbose=False)
    led.save_condition_output(replace(analysis, cell_label='Cell2',
                                      filter_wheel_ndf=1.0),
                              output_dir=tmp_path, verbose=False)

    index = led.load_condition_index(tmp_path)

    assert len(list(tmp_path.glob('*.h5'))) == 2
    assert index.columns.tolist() == [
        'date', 'cell_label', 'cell_type', 'onlineAnalysis', 'filter_wheel_ndf']
    assert index.to_dict('records') == [
        {'date': '2026-01-01_E', 'cell_label': 'Cell1',
         'cell_type': 'ON-parasol', 'onlineAnalysis': 'extracellular',
         'filter_wheel_ndf': 0.0},
        {'date': '2026-01-01_E', 'cell_label': 'Cell2',
         'cell_type': 'ON-parasol', 'onlineAnalysis': 'extracellular',
         'filter_wheel_ndf': 1.0},
    ]


def test_image_nli_summary_keeps_one_row_per_cell_fw_and_image(tmp_path):
    from dataclasses import replace

    analysis = _condition_analysis_for_output()
    second_images = analysis.image_summary.copy()
    second_images['meanIntensity'] /= 10
    led.save_condition_output(analysis, output_dir=tmp_path, verbose=False)
    led.save_condition_output(
        replace(analysis, cell_label='Cell2', cell_type='OFF-parasol',
                filter_wheel_ndf=1.0, image_summary=second_images),
        output_dir=tmp_path, verbose=False)
    led.save_condition_output(
        replace(analysis, cell_label='Cell3', protocols=['LinearEquivalentDiscConeLin']),
        output_dir=tmp_path, verbose=False)

    summary = led.load_condition_image_nli_summary(tmp_path)

    assert len(summary) == 4
    assert summary.columns.tolist() == led.IMAGE_NLI_SUMMARY_COLUMNS
    assert summary[['cell_id', 'filter_wheel_ndf', 'imageName']].duplicated().sum() == 0
    first = summary.loc[(summary['cell_label'].eq('Cell1'))
                        & (summary['imageName'].eq('00152'))].iloc[0]
    assert first['meanIntensity'] == 200.0
    assert first['n_patches'] == 2
    assert first['mean_nli_disc'] == pytest.approx(1 / 3)
    assert first['mean_nli_cone_disc'] == pytest.approx((1 / 7 + 1 / 15) / 2)
    assert set(summary['cell_type']) == {'ON-parasol', 'OFF-parasol'}


def test_image_nli_population_plot_has_one_panel_per_cell_type(tmp_path):
    from dataclasses import replace
    import matplotlib.pyplot as plt

    analysis = _condition_analysis_for_output()
    led.save_condition_output(analysis, output_dir=tmp_path, verbose=False)
    led.save_condition_output(replace(analysis, cell_label='Cell2',
                                      cell_type='OFF-parasol'),
                              output_dir=tmp_path, verbose=False)
    summary = led.load_condition_image_nli_summary(tmp_path)

    fig = led.plot_image_nli_by_cell_type(summary, log_x=False)
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]

    assert len(visible_axes) == 2
    assert {ax.get_title().split(' | ')[0] for ax in visible_axes} == {
        'ON-parasol', 'OFF-parasol'}
    assert all(ax.get_ylabel() == 'mean NLI across image patches' for ax in visible_axes)
    assert all(len(ax.collections) == 2 for ax in visible_axes)
    plt.close('all')


def test_matching_saved_condition_rehydrates_without_raw_data(tmp_path):
    analysis = _condition_analysis_for_output()
    led.save_condition_output(analysis, output_dir=tmp_path, verbose=False)
    blocks = pd.DataFrame({
        'exp_name': ['2026-01-01_E', '2026-01-01_E'],
        'cell_label': ['Cell1', 'Cell1'],
        'cell_type_short': ['ON-parasol', 'ON-parasol'],
        'onlineAnalysis': ['extracellular', 'extracellular'],
        'filter_wheel_ndf': [0.0, 0.0],
        'protocol': ['LinearEquivalentAnnulus', 'LinearEquivalentAnnulus'],
        'site': ['surround', 'surround'], 'block_id': [11, 12],
    })

    restored = led.load_condition_output(blocks, output_dir=tmp_path, verbose=False)
    assert restored is not None and restored.loaded_from_saved
    assert restored.block_ids == [11, 12]
    assert restored.image_summary['epochs'].tolist() == [12, 6]
    assert restored.patch_responses['patch_key'].tolist() == [
        '00152:1', '00152:2', '01769:1']

    automatic = led.analyze_condition(
        blocks, saved_output_dir=tmp_path, verbose=False)
    assert automatic.loaded_from_saved
    assert automatic.patch_responses.equals(restored.patch_responses)

    stale = blocks.copy()
    stale.loc[1, 'block_id'] = 13
    assert led.load_condition_output(stale, output_dir=tmp_path, verbose=False) is None


def test_condition_nli_plot_uses_ecdf_and_mean_sem_subplot():
    import matplotlib.pyplot as plt

    figures = led.plot_condition(_condition_analysis_for_output(), columns=2)
    nli_figure = figures[2]
    distribution_axis, mean_axis = nli_figure.axes

    assert len(nli_figure.axes) == 2
    assert len(distribution_axis.patches) == 0
    assert len(distribution_axis.lines) >= 5  # two distributions, two means, zero
    first_cdf = distribution_axis.lines[0]
    assert first_cdf.get_drawstyle() == 'steps-post'
    assert first_cdf.get_ydata().tolist() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert distribution_axis.get_ylabel() == 'cumulative fraction'
    assert distribution_axis.get_title() == 'onset NLI empirical CDF'
    assert len(mean_axis.collections) == 2   # two error-bar point collections
    plt.close('all')


def test_example_patch_params_from_blocks_skips_pre_cone_rows(monkeypatch):
    blocks = pd.DataFrame([
        {'exp_name': 'old_E', 'block_id': 1, 'linearizeCones': np.nan},
        {'exp_name': 'new_E', 'block_id': 2, 'linearizeCones': 1.0},
    ])
    calls = []

    def params(exp_name, block_id, patch_index=None, image_name=None):
        calls.append((exp_name, block_id, patch_index, image_name))
        return {'equivalentIntensityConeLin': 0.25}

    monkeypatch.setattr(led, 'example_patch_params', params)
    result = led.example_patch_params_from_blocks(blocks, patch_index=3)

    assert result['equivalentIntensityConeLin'] == 0.25
    assert calls == [('new_E', 2, 3, None)]


def test_example_patch_params_from_blocks_rejects_missing_cone_value(monkeypatch):
    blocks = pd.DataFrame([
        {'exp_name': 'old_E', 'block_id': 1, 'linearizeCones': np.nan},
    ])
    monkeypatch.setattr(
        led, 'example_patch_params',
        lambda *args, **kwargs: {'equivalentIntensity': 0.2})

    with pytest.raises(ValueError, match='none of the selected blocks records'):
        led.example_patch_params_from_blocks(blocks)


def test_plot_stimulus_example_never_labels_a_missing_cone_value_as_nan():
    with pytest.raises(ValueError, match='no cone-linearized equivalent intensity'):
        led.plot_stimulus_example({'equivalentIntensity': 0.2})


def test_example_patch_params_from_blocks_filters_image_name(monkeypatch):
    blocks = pd.DataFrame([
        {'exp_name': 'X_E', 'block_id': 1, 'linearizeCones': 1,
         'imageName': '00152'},
        {'exp_name': 'X_E', 'block_id': 2, 'linearizeCones': 1,
         'imageName': '01769'},
    ])
    calls = []

    def params(exp_name, block_id, patch_index=None, image_name=None):
        calls.append((block_id, image_name))
        return {'imageName': image_name, 'equivalentIntensityConeLin': 0.3}

    monkeypatch.setattr(led, 'example_patch_params', params)
    result = led.example_patch_params_from_blocks(blocks, image_name='01769')

    assert result['imageName'] == '01769'
    assert calls == [(2, '01769')]


def test_stimulus_example_widget_lists_and_redraws_image_names(monkeypatch):
    import matplotlib.pyplot as plt

    blocks = pd.DataFrame({'imageName': ['01769', '00152', '01769']})
    rendered = []

    def params(_blocks, patch_index=None, image_name=None):
        rendered.append(image_name)
        return {'imageName': image_name}

    monkeypatch.setattr(led, 'example_patch_params_from_blocks', params)
    monkeypatch.setattr(led, 'plot_stimulus_example', lambda _params: plt.figure())

    widget = led.stimulus_example_widget(blocks)
    dropdown, output = widget.children
    dropdown.value = '01769'

    assert list(dropdown.options) == ['00152', '01769']
    assert rendered == ['00152', '01769']
    assert output is not None
