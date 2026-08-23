"""Tests for SCutils.protocols.linear_equivalent_disc — pure helpers only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retinanalysis.SCutils.protocols import linear_equivalent_disc as led


@pytest.mark.parametrize('name', [
    'analyzeConeAnnulusDisc.ipynb',
    'analyzeConeCenterDisc.ipynb',
])
def test_analysis_notebooks_select_retinanalysis_kernel(name):
    notebook_path = Path(__file__).parents[1] / 'SingCell_Notebooks' / name
    notebook = json.loads(notebook_path.read_text())

    assert notebook['metadata']['kernelspec']['name'] == 'retinanalysis'
    assert notebook['metadata']['kernelspec']['display_name'] == (
        'retinanalysis (Python 3.11)')
    import_source = ''.join(notebook['cells'][1]['source'])
    assert 'sys.version_info[:2] != (3, 11)' in import_source
    assert 'sys.executable' in import_source


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
        'cell_type': ['RGC\\ON-parasol', 'RGC\\ON-parasol', 'horizontal'],
        'protocol': ['LinearEquivalentAnnulus'] * 3,
        'parameters': [{}] * 3,
    })
    monkeypatch.setattr(led, '_protocol_block_rows', lambda protocols: blocks)
    found = led.find_protocol_cells('LinearEquivalentAnnulus', show=False)
    assert list(found.columns) == [
        'exp_name', 'cell_label', 'cell_type_short', 'protocol']
    assert found.to_dict('records') == [
        {'exp_name': '2026-01-01_E', 'cell_label': 'Cell1',
         'cell_type_short': 'ON-parasol',
         'protocol': 'LinearEquivalentAnnulus'},
        {'exp_name': '2026-01-02_E', 'cell_label': 'Cell2',
         'cell_type_short': 'horizontal',
         'protocol': 'LinearEquivalentAnnulus'},
    ]


def test_find_protocol_cells_combines_center_names_and_drops_old_disc(monkeypatch):
    blocks = pd.DataFrame({
        'exp_name': ['old_E', 'new_E', 'cone_E'],
        'cell_label': ['Cell1', 'Cell2', 'Cell3'],
        'cell_type': ['RGC\\ON-parasol', 'RGC\\OFF-parasol', 'horizontal'],
        'protocol': ['LinearEquivalentDisc', 'LinearEquivalentDisc',
                     'LinearEquivalentDiscConeLin'],
        'parameters': [{}, {'linearizeCones': True}, {'linearizeCones': True}],
    })
    requested = []

    def block_rows(protocols):
        requested.extend(protocols)
        return blocks

    monkeypatch.setattr(led, '_protocol_block_rows', block_rows)
    found = led.find_protocol_cells(
        ('LinearEquivalentDiscConeLin', 'LinearEquivalentDisc'), show=False)

    assert requested == ['LinearEquivalentDiscConeLin', 'LinearEquivalentDisc']
    assert found['exp_name'].tolist() == ['cone_E', 'new_E']
    assert found['cell_type_short'].tolist() == ['horizontal', 'OFF-parasol']
    assert set(found['protocol']) == {
        'LinearEquivalentDiscConeLin', 'LinearEquivalentDisc'}


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


def test_mode_check_reuses_response_metadata_and_one_trace_per_group(monkeypatch):
    from retinanalysis.SCutils import recording_mode as rm

    blocks = pd.DataFrame([
        _one_block(block_id=1, onlineAnalysis='none', group_id=10),
        _one_block(block_id=2, onlineAnalysis='none', group_id=10),
        _one_block(block_id=3, onlineAnalysis='extracellular', group_id=11),
    ])
    blocks['group_properties'] = [{'recordingTechnique': 'whole-cell'}] * 3
    response_table = pd.DataFrame({
        'response_id': [11, 12, 13], 'block_id': [1, 2, 3],
        'h5path': ['epoch-1/responses/Amp1', 'epoch-2/responses/Amp1',
                   'epoch-3/responses/Amp1'],
        'sample_rate': [1e4, 1e4, 1e4],
    })
    response_calls = []
    monkeypatch.setattr(
        rm, '_amp_response_table',
        lambda block_ids, amp='Amp1': response_calls.append(list(block_ids)) or response_table)
    monkeypatch.setattr(
        rm, 'series_resistance_table',
        lambda *args, **kwargs: pd.DataFrame({
            'block_id': [1, 2, 3], 'series_resistance': [0.0, 0.0, 8e6],
            'series_resistance_min': [0.0, 0.0, 8e6],
            'series_resistance_max': [0.0, 0.0, 8e6],
            'n_epochs_rs': [1, 1, 1], 'n_epochs_high_rs': [0, 0, 0],
        }))

    def trace_samples(df, **kwargs):
        assert df['block_id'].tolist() == [1, 3]
        assert kwargs['response_table'] is response_table
        assert kwargs['n_trials_by_block'] == {3: 1}
        return {
            1: (_spiking_trace(), 1e4),
            3: (np.full((1, 100), -5.0), 1e4),
        }

    monkeypatch.setattr(rm, '_amp_trace_samples', trace_samples)
    resolutions = []

    def resolve(label, series_resistance, **kwargs):
        resolutions.append((label, series_resistance))
        return ('extracellular' if series_resistance == 0 else 'exc', 'resolved')

    monkeypatch.setattr(rm, 'resolve_recording_mode', resolve)

    result = rm.check_series_resistance(blocks, show=False)

    assert response_calls == [[1, 2, 3]]
    assert resolutions == [('none', 0.0)]
    assert result['onlineAnalysis'].tolist() == ['extracellular', 'extracellular', 'exc']


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


def test_select_condition_blocks_uses_cell_mode_and_filter_wheel(monkeypatch, capsys):
    from retinanalysis.SCutils import explore as sc

    monkeypatch.setattr(sc, 'scroll_table', lambda *args, **kwargs: None)
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
        blocks, 'Cell1', 'exc', 0.0, show=True)

    assert selected['block_id'].tolist() == [1, 2]
    assert 'X_E/Cell1 (ON-parasol) | exc | FilterWheel 0' in capsys.readouterr().out
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


def test_condition_save_alerts_and_removes_matching_h5_and_legacy_csv(
        tmp_path, capsys):
    import shutil

    analysis = _condition_analysis_for_output()
    canonical = led.save_condition_output(analysis, output_dir=tmp_path, verbose=False)
    duplicate_h5 = tmp_path / 'duplicate-name.h5'
    shutil.copyfile(canonical, duplicate_h5)
    duplicate_csv = tmp_path / 'legacy-copy.csv'
    led.condition_population_table(analysis).to_csv(duplicate_csv, index=False)

    saved = led.save_condition_output(analysis, output_dir=tmp_path, verbose=True)
    output = capsys.readouterr().out

    assert saved == canonical and canonical.exists()
    assert not duplicate_h5.exists()
    assert not duplicate_csv.exists()
    assert 'ALERT: found 3 saved copy/copies' in output
    assert 'ALERT: replaced the existing canonical saved condition' in output
    assert 'ALERT: removed 2 duplicate saved copy/copies' in output


def test_condition_save_does_not_remove_a_different_filter_wheel(tmp_path):
    from dataclasses import replace

    analysis = _condition_analysis_for_output()
    other = replace(analysis, filter_wheel_ndf=1.0)
    other_path = led.save_condition_output(other, output_dir=tmp_path, verbose=False)

    led.save_condition_output(analysis, output_dir=tmp_path, verbose=False)

    assert other_path.exists()
    assert len(list(tmp_path.glob('*.h5'))) == 2


def test_default_condition_outputs_use_separate_center_and_annulus_folders(
        tmp_path, monkeypatch):
    from dataclasses import replace

    monkeypatch.setattr(led, 'store_dir', lambda: tmp_path)
    annulus = _condition_analysis_for_output()
    center = replace(
        annulus, cell_label='Cell2', site='center',
        protocols=['LinearEquivalentDisc'])

    annulus_path = led.save_condition_output(annulus, verbose=False)
    center_path = led.save_condition_output(center, verbose=False)

    assert annulus_path.parent == tmp_path / 'condition_outputs' / 'annulus_disc'
    assert center_path.parent == tmp_path / 'condition_outputs' / 'center_disc'
    assert led.condition_output_dir('LinearEquivalentDiscConeLin') == center_path.parent
    assert len(led.load_condition_outputs()) == 6
    assert set(led.load_condition_index(protocol='LinearEquivalentAnnulus')['cell_label']) == {
        'Cell1'}
    assert set(led.load_condition_index(
        protocol=('LinearEquivalentDisc', 'LinearEquivalentDiscConeLin'))['cell_label']) == {
            'Cell2'}


def test_default_condition_loader_falls_back_to_legacy_shared_folder(
        tmp_path, monkeypatch):
    monkeypatch.setattr(led, 'store_dir', lambda: tmp_path)
    analysis = _condition_analysis_for_output()
    led.save_condition_output(
        analysis, output_dir=led.condition_output_dir(), verbose=False)
    blocks = pd.DataFrame({
        'exp_name': ['2026-01-01_E', '2026-01-01_E'],
        'cell_label': ['Cell1', 'Cell1'],
        'cell_type_short': ['ON-parasol', 'ON-parasol'],
        'onlineAnalysis': ['extracellular', 'extracellular'],
        'filter_wheel_ndf': [0.0, 0.0],
        'protocol': ['LinearEquivalentAnnulus', 'LinearEquivalentAnnulus'],
        'site': ['surround', 'surround'], 'block_id': [11, 12],
    })

    restored = led.load_condition_output(blocks, verbose=False)

    assert restored is not None and restored.loaded_from_saved
    assert restored.block_ids == [11, 12]


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


def test_saved_condition_uses_larger_conflicting_intensity_with_warning(tmp_path):
    from dataclasses import replace

    analysis = _condition_analysis_for_output()
    image_summary = analysis.image_summary.copy()
    image_summary[['maxIntensity', 'meanIntensity']] = image_summary[[
        'maxIntensity', 'meanIntensity']].astype(object)
    image_summary.loc[0, 'maxIntensity'] = '7700, 77000'
    image_summary.loc[0, 'meanIntensity'] = '1199.85, 11998.5'
    path = led.save_condition_output(
        replace(analysis, image_summary=image_summary),
        output_dir=tmp_path, verbose=False)

    with pytest.warns(RuntimeWarning, match='using larger value') as warnings:
        loaded = led._read_condition_h5(path)

    corrected = loaded.image_summary.loc[
        loaded.image_summary['imageName'].eq('00152')].iloc[0]
    assert corrected['maxIntensity'] == 77000.0
    assert corrected['meanIntensity'] == 11998.5
    assert len(warnings) == 1
    assert "meanIntensity='1199.85, 11998.5' -> 11998.5" in str(warnings[0].message)


def test_patch_nli_loader_reads_h5_without_averaging_or_other_protocols(tmp_path):
    from dataclasses import replace

    analysis = _condition_analysis_for_output()
    led.save_condition_output(analysis, output_dir=tmp_path, verbose=False)
    led.save_condition_output(
        replace(analysis, cell_label='Cell2', protocols=['LinearEquivalentDiscConeLin']),
        output_dir=tmp_path, verbose=False)

    patches = led.load_condition_patch_nli(tmp_path)

    assert len(patches) == 3
    assert patches.columns.tolist() == led.PATCH_NLI_COLUMNS
    assert patches['patch_key'].tolist() == ['00152:1', '00152:2', '01769:1']
    assert patches['nli_disc'].tolist() == pytest.approx([1 / 3] * 3)
    assert patches['meanIntensity'].tolist() == [200.0, 200.0, 100.0]


def test_population_loaders_accept_both_center_protocol_names(tmp_path):
    from dataclasses import replace

    analysis = _condition_analysis_for_output()
    for cell_label, protocol in (
            ('Cell1', 'LinearEquivalentDiscConeLin'),
            ('Cell2', 'LinearEquivalentDisc')):
        led.save_condition_output(
            replace(analysis, cell_label=cell_label, protocols=[protocol], site='center'),
            output_dir=tmp_path, verbose=False)
    led.save_condition_output(
        replace(analysis, cell_label='Cell3'), output_dir=tmp_path, verbose=False)

    protocols = ('LinearEquivalentDiscConeLin', 'LinearEquivalentDisc')
    images = led.load_condition_image_nli_summary(tmp_path, protocol=protocols)
    patches = led.load_condition_patch_nli(tmp_path, protocol=protocols)
    index = led.load_condition_index(tmp_path, protocol=protocols)

    assert set(images['cell_label']) == {'Cell1', 'Cell2'}
    assert set(patches['cell_label']) == {'Cell1', 'Cell2'}
    assert set(index['cell_label']) == {'Cell1', 'Cell2'}
    assert set(images['protocol']) == set(protocols)


def test_pooled_patch_nli_plot_has_density_and_empirical_cdf(tmp_path):
    import matplotlib.pyplot as plt

    led.save_condition_output(
        _condition_analysis_for_output(), output_dir=tmp_path, verbose=False)
    patches = led.load_condition_patch_nli(tmp_path)
    fig = led.plot_pooled_patch_nli_distributions(patches, bins=20)
    density_axis, cdf_axis = fig.axes

    assert density_axis.get_title() == '20-bin pooled patch density'
    assert density_axis.get_ylabel() == 'density'
    assert cdf_axis.get_title() == 'pooled patch empirical CDF'
    assert cdf_axis.get_ylabel() == 'cumulative fraction'
    cdf_lines = [line for line in cdf_axis.lines if line.get_drawstyle() == 'steps-post']
    assert len(cdf_lines) == 2
    assert all(line.get_ydata()[-1] == 1 for line in cdf_lines)
    plt.close('all')


def test_pooled_patch_nli_plot_handles_an_empty_center_dataset():
    import matplotlib.pyplot as plt

    empty = pd.DataFrame(columns=led.PATCH_NLI_COLUMNS)
    fig = led.plot_pooled_patch_nli_distributions(
        empty, title_prefix='Cone-linearized center disc')

    assert all(ax.texts[0].get_text() == 'no saved patch NLI data'
               for ax in fig.axes)
    assert fig._suptitle.get_text().startswith('Cone-linearized center disc')
    plt.close('all')


def test_patch_nli_distribution_plot_has_one_row_per_cell_type():
    import matplotlib.pyplot as plt

    rows = pd.DataFrame({
        'cell_type': ['ON-parasol', 'ON-parasol', 'OFF-parasol'],
        'nli_disc': [-.2, .1, .3],
        'nli_cone_disc': [-.1, .2, .4],
    })
    fig = led.plot_patch_nli_distributions_by_cell_type(rows, bins=25)

    assert len(fig.axes) == 4
    assert [fig.axes[index].get_ylabel().split('\n')[0] for index in (0, 2)] == [
        'ON-parasol', 'OFF-parasol']
    assert all(ax.get_title() == '25-bin patch density' for ax in fig.axes[::2])
    plt.close('all')


def test_light_level_summary_uses_requested_bins_and_cell_image_sem():
    rows = pd.DataFrame({
        'cell_type': ['ON-parasol'] * 8,
        'cell_id': ['d/C1', 'd/C1', 'd/C2', 'd/C2', 'd/C3', 'd/C3', 'd/C4', 'd/C4'],
        'imageName': [str(i) for i in range(8)],
        'meanIntensity': [499, 500, 1000, 1499, 1500, 3500, 6000, 20000],
        'mean_nli_disc': [9, 0, 1, 2, 3, 4, 5, 6],
        'mean_nli_cone_disc': [9, 0, -1, -2, -3, -4, -5, -6],
    })

    summary = led.summarize_image_nli_light_levels(rows)

    assert summary['light_level'].tolist() == [
        '500-1500', '1500-3500', '3500-6000', '6000-20000']
    assert summary['n_cell_images'].tolist() == [3, 1, 1, 2]
    assert summary['meanIntensity'].tolist() == pytest.approx(
        [(500 + 1000 + 1499) / 3, 1500, 3500, 13000])
    assert summary['mean_nli_disc'].tolist() == pytest.approx([1, 3, 4, 5.5])
    assert summary.loc[0, 'sem_nli_disc'] == pytest.approx(1 / np.sqrt(3))
    assert np.isnan(summary.loc[1, 'sem_nli_disc'])


def test_cell_patch_summary_averages_all_images_within_each_cell_first():
    rows = pd.DataFrame({
        'cell_type': ['ON-parasol'] * 8,
        'cell_id': ['d/C1'] * 5 + ['d/C2'] * 3,
        'exp_name': ['d'] * 8,
        'cell_label': ['C1'] * 5 + ['C2'] * 3,
        'imageName': ['A', 'A', 'B', 'C', 'edge', 'A', 'D', 'D'],
        'patch_key': [f'p{i}' for i in range(8)],
        'meanIntensity': [1000, 1000, 1200, 10000, 1500, 900, 6000, 20000],
        'nli_disc': [0, 1, 2, 4, 99, 5, 6, 8],
        'nli_cone_disc': [0, -1, -2, -4, -99, -5, -6, -8],
    })

    summary = led.summarize_cell_patch_nli_light_levels(rows)

    c1_low = summary.loc[(summary['cell_id'].eq('d/C1'))
                         & (summary['light_level'].eq('~1k'))].iloc[0]
    c2_high = summary.loc[(summary['cell_id'].eq('d/C2'))
                          & (summary['light_level'].eq('~10k'))].iloc[0]
    assert c1_low['n_images'] == 2
    assert c1_low['n_patches'] == 3
    assert c1_low['mean_nli_disc'] == pytest.approx(1.0)
    assert c1_low['mean_nli_cone_disc'] == pytest.approx(-1.0)
    assert c2_high['n_patches'] == 2
    assert c2_high['mean_nli_disc'] == pytest.approx(7.0)
    assert c2_high['meanIntensity'] == 13000.0
    assert not (summary['mean_nli_disc'] == 99).any()  # 1500 is outside the two bins


def test_high_light_cell_summary_uses_7000_as_inclusive_lower_cutoff():
    rows = pd.DataFrame({
        'cell_type': ['ON-parasol'] * 5,
        'cell_id': ['d/C1'] * 3 + ['d/C2'] * 2,
        'exp_name': ['d'] * 5,
        'cell_label': ['C1'] * 3 + ['C2'] * 2,
        'imageName': ['low', 'A', 'B', 'A', 'B'],
        'patch_key': [f'p{i}' for i in range(5)],
        'meanIntensity': [6999, 7000, 30000, 8000, 9000],
        'nli_disc': [99, .2, .4, -.2, 0],
        'nli_cone_disc': [99, .1, .3, -.1, .1],
    })

    summary = led.summarize_cell_patch_nli_above(rows)

    assert summary['cell_id'].tolist() == ['d/C1', 'd/C2']
    assert summary['min_intensity'].eq(7000).all()
    assert summary.loc[0, 'mean_nli_disc'] == pytest.approx(.3)
    assert summary.loc[0, 'meanIntensity'] == 18500
    assert not (summary['mean_nli_disc'] == 99).any()


def test_high_light_cell_plot_draws_one_pair_per_cell_type():
    import matplotlib.pyplot as plt

    summary = pd.DataFrame({
        'cell_type': ['ON-parasol', 'ON-parasol', 'OFF-parasol'],
        'cell_id': ['d/C1', 'd/C2', 'd/C3'],
        'exp_name': ['d'] * 3,
        'cell_label': ['C1', 'C2', 'C3'],
        'min_intensity': [7000.] * 3,
        'meanIntensity': [9000., 10000., 11000.],
        'n_images': [2] * 3,
        'n_patches': [20] * 3,
        'mean_nli_disc': [.1, .3, -.2],
        'mean_nli_cone_disc': [0, .2, -.1],
    }, columns=led.HIGH_LIGHT_CELL_NLI_COLUMNS)

    fig = led.plot_cell_patch_nli_paired_above(summary)
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]

    assert len(visible_axes) == 2
    assert len(visible_axes[0].lines) == 3  # two paired cells and the zero line
    assert fig._suptitle.get_text().endswith('≥7,000 R*')
    plt.close('all')


def test_high_light_cell_plot_shows_empty_cutoff_message():
    import matplotlib.pyplot as plt

    empty = pd.DataFrame(columns=led.HIGH_LIGHT_CELL_NLI_COLUMNS)
    fig = led.plot_cell_patch_nli_paired_above(empty)

    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible_axes) == 1
    assert visible_axes[0].texts[0].get_text() == (
        'no cells at or above the light cutoff')
    plt.close('all')


def test_cell_patch_population_plot_uses_cell_means_and_sem():
    import matplotlib.pyplot as plt

    summary = pd.DataFrame({
        'cell_type': ['ON-parasol'] * 4 + ['OFF-parasol'] * 2,
        'cell_id': ['d/C1', 'd/C2'] * 2 + ['d/C3'] * 2,
        'exp_name': ['d'] * 6,
        'cell_label': ['C1', 'C2'] * 2 + ['C3'] * 2,
        'light_level': ['~1k', '~1k', '~10k', '~10k', '~1k', '~10k'],
        'light_min': [500, 500, 6000, 6000, 500, 6000],
        'light_max': [1500, 1500, 20000, 20000, 1500, 20000],
        'meanIntensity': [1000, 1100, 10000, 11000, 900, 9000],
        'n_images': [2] * 6, 'n_patches': [20] * 6,
        'mean_nli_disc': [.1, .3, .2, .4, -.2, -.1],
        'mean_nli_cone_disc': [0, .2, .1, .3, -.1, 0],
    }, columns=led.CELL_PATCH_NLI_COLUMNS)

    fig = led.plot_cell_patch_nli_by_light(summary)
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]

    assert len(visible_axes) == 2
    assert all(ax.get_ylabel() == 'cell mean NLI; population mean ± SEM'
               for ax in visible_axes)
    assert all([tick.get_text().split('\n')[0] for tick in ax.get_xticklabels()]
               == ['~1k', '~10k'] for ax in visible_axes)
    plt.close('all')


def test_image_nli_population_plot_has_one_panel_per_cell_type(tmp_path):
    from dataclasses import replace
    import matplotlib.pyplot as plt

    analysis = _condition_analysis_for_output()
    plot_images = analysis.image_summary.copy()
    plot_images['meanIntensity'] *= 10
    analysis = replace(analysis, image_summary=plot_images)
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
    assert all(ax.get_ylabel() == 'population mean NLI ± SEM' for ax in visible_axes)
    assert all({line.get_label() for line in ax.lines}.issuperset({
        'image vs standard disc', 'image vs cone-lin disc'}) for ax in visible_axes)
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


def test_condition_sample_pairs_use_full_keys_and_distinct_images_first():
    analysis = _condition_analysis_for_output()

    assert led.condition_sample_pairs(analysis, n_pairs=2) == [
        ('00152', 2.0), ('01769', 1.0)]


def test_condition_sample_pairs_cover_both_disc_types_when_available():
    analysis = _condition_analysis_for_output()
    analysis.patch_responses.loc[:, 'cone_disc_mean'] = np.nan
    analysis.patch_responses.loc[
        analysis.patch_responses['patch_key'].eq('00152:1'), 'cone_disc_mean'] = 3.0

    assert led.condition_sample_pairs(analysis, n_pairs=2) == [
        ('00152', 2.0), ('00152', 1.0)]


def test_condition_sample_psth_is_trial_normalized(monkeypatch):
    import matplotlib.pyplot as plt

    analysis = _condition_analysis_for_output()
    analysis.patch_responses = analysis.patch_responses.loc[
        analysis.patch_responses['patch_key'].eq('00152:2')]
    blocks = pd.DataFrame({
        'exp_name': ['2026-01-01_E'], 'block_id': [11],
        'onlineAnalysis': ['extracellular'],
    })
    trials = pd.DataFrame([
        {'imageName': '00152', 'patchIndex': 2.0, 'category': category,
         'block_id': 11, 'epoch_index': repeat, 'pre_ms': 10.0,
         'stim_ms': 20.0, 'tail_ms': 10.0, 'time_ms': None,
         'values': np.array([5.0, 15.0])}
        for category in ('image', 'disc', 'cone_disc')
        for repeat in range(2)
    ])
    monkeypatch.setattr(led, '_load_condition_sample_trials',
                        lambda *args, **kwargs: trials)

    figure = led.plot_condition_sample_psths(
        blocks, analysis, n_pairs=1, bin_ms=10.0, smooth_ms=0.0)
    axis = figure.axes[0]

    assert len(axis.lines) == 5  # three PSTHs plus onset and offset
    assert axis.lines[0].get_ydata().max() == pytest.approx(100.0)
    assert axis.get_ylabel() == 'firing rate (Hz)'
    assert '00152 : patch 2' in axis.get_title()
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


def test_stimulus_triplet_frames_match_annulus_protocol_layers(monkeypatch):
    grid = np.linspace(-100, 100, 9)
    radius = np.hypot(*np.meshgrid(grid, grid))
    patch = np.full((9, 9), .6)
    annulus = (radius <= 80) & (radius >= 50)
    monkeypatch.setattr(
        led, 'image_patch',
        lambda *args, **kwargs: (patch, annulus, 100.0, .25))
    params = {
        'imageName': '00152', 'currentPatchLocation': [10, 20],
        'annulusInnerDiameter': 100, 'annulusOuterDiameter': 160,
        'backgroundIntensity': .2, 'centerSpotDiameter': 40,
        'centerSpotContrast': .5, 'equivalentIntensity': .4,
        'equivalentIntensityConeLin': .35,
    }

    frames, extent, background = led.stimulus_triplet_frames(params)

    assert extent == 100
    assert background == .2
    assert [frame[0, 0] for frame in frames] == pytest.approx([.2, .2, .2])
    assert [frame[4, 4] for frame in frames] == pytest.approx([.3, .3, .3])
    assert [frame[4, 6] for frame in frames] == pytest.approx([.6, .4, .35])


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

    def params(_blocks, patch_index=None, image_name=None, count=4):
        rendered.append(image_name)
        return [{'imageName': image_name}]

    monkeypatch.setattr(led, 'example_patch_sequence_from_blocks', params)
    monkeypatch.setattr(led, 'plot_stimulus_sequence', lambda _params: plt.figure())

    widget = led.stimulus_example_widget(blocks)
    dropdown, output = widget.children
    dropdown.value = '01769'

    assert list(dropdown.options) == ['00152', '01769']
    assert rendered == ['00152', '01769']
    assert output is not None
    assert len(output.outputs) == 1


def test_plot_stimulus_sequence_draws_tilted_triplets_and_arrows(monkeypatch):
    import matplotlib.pyplot as plt

    patch = np.arange(25, dtype=float).reshape(5, 5) / 25
    mask = np.ones_like(patch, dtype=bool)
    monkeypatch.setattr(
        led, 'image_patch',
        lambda *args, **kwargs: (patch, mask, 100.0, .5))
    params = [{
        'imageName': '00152', 'imagePatchIndex': index,
        'currentPatchLocation': [10, 20], 'annulusInnerDiameter': 50,
        'annulusOuterDiameter': 200, 'equivalentIntensity': .4,
        'equivalentIntensityConeLin': .6, 'backgroundIntensity': .2,
        'centerSpotDiameter': 30, 'centerSpotContrast': .5,
    } for index in (1, 2)]

    fig = led.plot_stimulus_sequence(params)
    ax = fig.axes[0]

    assert len(ax.images) == 6
    assert sum(text.get_text().startswith('patch ') for text in ax.texts) == 2
    assert any(text.get_text() == 'time' for text in ax.texts)
    assert not ax.axison
    plt.close('all')
