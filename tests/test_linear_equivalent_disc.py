"""Tests for SCutils.protocols.linear_equivalent_disc — pure helpers only."""
import numpy as np
import pandas as pd
import pytest

from retinanalysis.SCutils.protocols import linear_equivalent_disc as led


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
    assert g.loc[0, 'recorded_labels'] == 'exc'


def test_group_blocks_prefers_the_pre_relabel_column_for_recorded_labels():
    df = pd.DataFrame([_one_block(onlineAnalysis='extracellular')])
    df['onlineAnalysis_recorded'] = 'none'
    g = led.group_blocks(df, show=False)
    # Grouped by what it is analyzed as, but reporting what was recorded.
    assert g.loc[0, 'onlineAnalysis'] == 'extracellular'
    assert g.loc[0, 'recorded_labels'] == 'none'
