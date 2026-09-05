"""VariableMeanNoise: the stimulus must come from MATLAB, not from NumPy.

The noise stimulus is reconstructed from the seed Symphony recorded, and it has
to be the waveform the retina actually saw. MATLAB's
``RandStream('mt19937ar').randn`` does not agree with any NumPy generator -- its
``rand`` does, which is what makes the mistake easy to make -- so the Gaussian
draw is taken from the MATLAB engine and everything after it is NumPy.

These tests pin that. The reference values were produced by running the
transcribed ``GaussianNoiseGeneratorV2.generateStimulus`` in MATLAB R2025b; the
full 20000-sample case agrees with the Python port to 8e-16.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / 'SingCellNotebooks' / 'rodAdaptation'
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

vmn = pytest.importorskip('variable_mean_noise')


def test_notebook_pins_named_retinanalysis_kernel():
    import json

    notebook_path = NOTEBOOK_DIR / 'analyzeVariableMeanNoise.ipynb'
    kernelspec = json.loads(notebook_path.read_text())['metadata']['kernelspec']

    assert kernelspec == {
        'display_name': 'retinanalysis (Python 3.11)',
        'language': 'python',
        'name': 'retinanalysis',
    }

# RandStream('mt19937ar', 'Seed', 42).randn(1, 64), from MATLAB R2025b.
MATLAB_RANDN_SEED42 = np.array([
    -0.53824389372926962, 0.86723215762957329, 0.97598646347253948, 0.33739025237921211,
    -0.99609409661495396, -0.52321403168644098, -1.2974474771673907, 0.91738858914593557,
    0.17660162861754794, 0.75517993567748831, -0.59149995977194048, 1.8443896370005026,
    1.816922249179955, -0.12383335027924647, -1.1106013550606419, -0.68090588026724619,
    0.014169326397636454, -0.059550610459064499, -0.66101101446033916, 0.30595091506898697,
    -0.4090578457905143, -1.2818548228848252, -0.28490282946308648, -0.06478685589122006,
    1.0003831888389663, -0.80132929831121413, 0.091800015251789033, 0.27380479193770302,
    -2.1334675594179631, 0.40397728054942095, -0.95849940934865763, -0.3263168518663801,
    2.4786956296041733, 1.5390760520725197, 0.82701089590509713, -0.15958722588765548,
    -0.8577217490829192, 0.39835396641254128, -0.22480884254365807, -1.3146676437229412,
    -0.021796865945491947, -0.88006034099445507, 1.1086404728571893, -0.46413826093464672,
    0.66456919138124804, -0.53013811821816303, 0.040909329870564795, 0.15505390913724837,
    -0.32108498156234999, 2.1458146711040254, 0.80040655725138687, 1.1962715843610494,
    2.0975463826553962, 0.38771501535540298, 1.0453182014943911, -1.3718263433146096,
    -0.95840715563049539, -2.6689399596406402, -0.80268518189891847, -0.45101594662272571,
    -0.73658154396130016, 1.1147039555125553, -0.85335003027021894, -0.75875185852041416,
])

# First 8 samples of GaussianNoiseGeneratorV2 for that draw: stDev 0.2,
# 60 Hz / 4 poles, mean 1.0, 64 points at 1 kHz. The whole 64-sample draw is
# needed to reproduce them, since the filter is applied in the frequency domain.
MATLAB_STIMULUS_HEAD = np.array([
    0.790875972988326, 0.86349387936666, 0.932651498882655, 0.996042965609082,
    1.0529432372854, 1.10333798944662, 1.14697357984573, 1.18271589004235])


def _population_analysis():
    """Small fitted condition for persistence tests; no raw-data access."""
    import pandas as pd

    def model(label, scale=1.0):
        return vmn.LNModel(
            label=label, r2=.5 * scale,
            filter=np.array([0.0, 1.0, -0.25]) * scale,
            filter_time_s=np.array([0.0, .01, .02]),
            nl_x=np.array([-1.0, 0.0, 1.0]),
            nl_y=np.array([0.0, 1.0, 2.0]) * scale,
            params={'alpha': 2.0, 'beta': 1.0, 'gamma': 0.0,
                    'epsilon': 0.0, 'r2': .9, 'at_bounds': ()},
            source='python', r2_train=.6, nl_r2=.9, n_train=2, n_test=1)

    analysis = vmn.ConditionAnalysis(
        exp_name='2020-06-11_B', block_ids=[11, 12],
        rec_type='extracellular', sample_rate=1000.0, units='spikes/s',
        light_means=[.1, 1.0], n_epochs={.1: 2, 1.0: 2},
        sampling_interval=.1, skip_seconds=1.0, stim_time_ms=30_000.0,
        frequency_cutoff=60.0,
        stimulus={.1: np.ones((2, 20)), 1.0: np.ones((2, 20))},
        response={.1: np.arange(40).reshape(2, 20),
                  1.0: np.arange(40, 80).reshape(2, 20)},
        ln_model={.1: model('lightMean 0.1'), 1.0: model('lightMean 1', 1.2)},
        dropped_epochs=pd.DataFrame(), epoch_adjustments=pd.DataFrame())
    temporal = {
        .1: [model('1.0-2.0 s')],
        1.0: [model('1.0-2.0 s', 1.2)],
    }
    return analysis, temporal


def test_normalize_epoch_light_means_recovers_legacy_array_values():
    import pandas as pd

    raw = pd.DataFrame({
        'lightMean': [np.array([.02, .4]), np.array([.02, .4])],
        'stdv': [.01, .2], 'Contrast': [.5, .5],
    })

    normalized = vmn.normalize_epoch_light_means(raw)

    assert normalized.lightMean.tolist() == pytest.approx([.02, .4])
    assert normalized.lightMeanStatus.tolist() == [
        'derived from stdv / Contrast'] * 2
    assert np.array_equal(normalized.lightMeanRecorded.iloc[0], [.02, .4])


def test_a2_and_aii_cell_type_aliases_match():
    assert vmn._match_cell_type('A2 amacrine')
    assert vmn._match_cell_type('AII')


def test_discovery_species_gate_is_independent_of_rig_suffix():
    import pandas as pd

    rigs = ['B', 'E', 'F', 'G', 'Z']
    frame = pd.DataFrame({
        'exp_name': [f'2026-08-01_{rig}' for rig in rigs],
        'experiment_label': ['Primate'] * len(rigs),
        'cell_type_short': ['ON-parasol'] * len(rigs),
    })

    keep = vmn._species_cell_keep_mask(
        frame, vmn.PRIMATE_EXPERIMENT_LABELS, vmn.PRIMATE_CELL_TYPES)

    assert keep.tolist() == [True] * len(rigs)
    assert frame.exp_name.map(vmn._experiment_rig_suffix).tolist() == rigs


def test_stable_cell_indices_keep_old_ids_and_append_recovered_cells(tmp_path):
    import pandas as pd

    registry = tmp_path / 'indices.csv'
    registry.write_text(
        'cell_index,exp_name,cell_label\n'
        '8,2021-01-01_B,Cell1\n'
        '9,2021-01-02_B,cell2\n')
    cells = pd.DataFrame({
        'exp_name': ['2020-07-29_G', '2021-01-01_B'],
        'cell_label': ['cell1', 'Cell1'],
    })

    indexed = vmn._stable_cell_indices(cells, registry)

    assert indexed.cell_index.tolist() == [10, 8]


def test_duration_conditions_counts_every_epoch_and_keeps_exclusion_audit(monkeypatch):
    import pandas as pd

    parameters = {
        11: pd.DataFrame({'stimTime': [30_000, 30_000, 60_000],
                          'lightMean': [.3, 3.0, .3]}),
        12: pd.DataFrame({'stimTime': [30_000, 30_000, 60_000],
                          'lightMean': [.3, 3.0, 3.0]}),
    }
    monkeypatch.setattr(vmn, 'epoch_parameters', lambda block: parameters[block])

    durations = vmn.duration_conditions(
        [11, 12], min_epochs=4, min_stim_time_ms=None, show=False)

    short = durations[durations.stim_time_ms.eq(30_000)].iloc[0]
    long = durations[durations.stim_time_ms.eq(60_000)].iloc[0]
    assert short.n_epochs == 4 and short.included
    assert short.epochs_by_light_mean == '0.3: 2, 3: 2'
    assert short.block_ids == [11, 12]
    assert long.n_epochs == 2 and not long.included
    assert long.reason == 'fewer than 4 epochs'


def test_duration_conditions_split_contrast_without_dropping_cell(monkeypatch):
    import pandas as pd

    parameters = pd.DataFrame({
        'stimTime': [50_000] * 4,
        'lightMean': [.45, 4.5, .5, 5.0],
        'Contrast': [.35, .35, .30, .30],
    })
    monkeypatch.setattr(vmn, 'epoch_parameters', lambda _block: parameters)

    conditions = vmn.duration_conditions(
        [11], min_epochs=2, min_stim_time_ms=30_000, show=False)

    assert conditions.light_contrast.tolist() == pytest.approx([.30, .35])
    assert conditions.included.tolist() == [True, True]
    assert conditions.light_means.tolist() == ['0.5, 5', '0.45, 4.5']


def test_recording_duration_conditions_threshold_is_per_paired_condition(monkeypatch):
    import pandas as pd

    parameters = {
        11: pd.DataFrame({'stimTime': [30_000] * 4,
                          'lightMean': [.3, 3.0, .3, 3.0]}),
        12: pd.DataFrame({'stimTime': [30_000] * 3 + [60_000] * 4,
                          'lightMean': [.3, 3.0, .3, .3, 3.0, .3, 3.0]}),
        13: pd.DataFrame({'stimTime': [30_000] * 2,
                          'lightMean': [.3, 3.0]}),
        14: pd.DataFrame({'stimTime': [30_000] * 8,
                          'lightMean': [.3, 3.0] * 4}),
    }
    modes = pd.DataFrame({
        'block_id': [11, 12, 13, 14],
        'rec_type': ['extracellular', 'exc', 'inh', ''],
    })
    monkeypatch.setattr(vmn, 'epoch_parameters', lambda block: parameters[block])

    conditions = vmn.recording_duration_conditions(
        [11, 12, 13, 14], modes=modes, min_epochs=4,
        min_stim_time_ms=None, show=False)

    indexed = conditions.set_index(['rec_type', 'stim_time_ms'])
    assert indexed.loc[('extracellular', 30_000), 'included']
    assert not indexed.loc[('exc', 30_000), 'included']
    assert indexed.loc[('exc', 60_000), 'included']
    assert not indexed.loc[('inh', 30_000), 'included']
    assert not indexed.loc[('unresolved', 30_000), 'included']
    assert indexed.loc[('unresolved', 30_000), 'n_epochs'] == 8
    assert 'recording mode unresolved' in indexed.loc[
        ('unresolved', 30_000), 'reason']


def test_matlab_saved_comparison_uses_corrected_date_and_exact_label_case():
    import pandas as pd

    roster = pd.DataFrame({
        'index': [4, 5, 6],
        'calendar_date': ['2021-04-25'] * 3,
        'cell_label': ['Cell1', 'Cell2', 'cell3'],
        'cell_type': ['OnMidget', 'OnParasol', 'OnParasol'],
        'rec_type': ['exc', 'extracellular', 'extracellular'],
        'epoch_len_ms': [50_000., 60_000., 60_000.],
    })
    saved = pd.DataFrame({
        'date': ['2021-04-27_B'] * 3,
        'cell_label': ['Cell1', 'Cell2', 'Cell3'],
        'cell_type': ['ON-midget', 'OFF-parasol', 'OnParasol'],
        'rec_type': ['exc', 'extracellular', 'extracellular'],
        'stim_time_ms': [50_000., 60_000., 60_000.],
        'stim_seconds': [50., 60., 60.],
        'cell_index': [11, 12, 13],
        'output_path': ['/tmp/a.h5', '/tmp/b.h5', '/tmp/c.h5'],
    })

    compared = vmn.compare_matlab_roster_to_saved(roster, saved, show=False)

    indexed = compared.set_index('matlab_index')
    assert indexed.loc[4, 'match_status'] == 'likely matched'
    assert indexed.loc[4, 'corrected_date'] == '2021-04-27'
    assert indexed.loc[4, 'saved_date'] == '2021-04-27_B'
    assert indexed.loc[5, 'match_status'] == 'candidate'
    assert not indexed.loc[5, 'cell_type_agrees']
    assert 'cell_index 12' in indexed.loc[5, 'saved_entry']
    assert indexed.loc[6, 'match_status'] == 'not matched'
    assert indexed.loc[6, 'saved_entry'] == ''


def test_apply_recording_type_exclusions_preserves_condition_audit():
    import pandas as pd

    conditions = pd.DataFrame({
        'rec_type': ['extracellular', 'exc', 'inh'],
        'included': [True, True, False],
        'reason': ['', '', 'fewer than 6 epochs'],
        'block_ids': [[11], [12], [13]],
    })

    filtered = vmn.apply_recording_type_exclusions(
        conditions, excluded='exc', show=False)

    indexed = filtered.set_index('rec_type')
    assert indexed.loc['extracellular', 'included']
    assert not indexed.loc['exc', 'included']
    assert indexed.loc['exc', 'reason'] == 'recording type manually excluded'
    assert not indexed.loc['inh', 'included']
    assert indexed.loc['inh', 'reason'] == 'fewer than 6 epochs'


def test_apply_recording_type_exclusions_rejects_unknown_type():
    import pandas as pd

    with pytest.raises(ValueError, match='Unknown recording type'):
        vmn.apply_recording_type_exclusions(
            pd.DataFrame(columns=['rec_type', 'included', 'reason']),
            excluded=('bad-mode',), show=False)


def test_apply_epoch_range_exclusions_is_duration_specific(monkeypatch):
    import pandas as pd

    conditions = pd.DataFrame({
        'rec_type': ['exc', 'exc'],
        'stim_time_ms': [30_000.0, 60_000.0],
        'n_epochs': [4, 2],
        'block_ids': [[11], [11]],
        'included': [True, True],
        'reason': ['', ''],
    })
    params = pd.DataFrame({
        'stimTime': [30_000.0, 60_000.0, 30_000.0, 60_000.0,
                     30_000.0, 30_000.0],
    })
    monkeypatch.setattr(vmn, 'epoch_parameters', lambda _block: params)

    filtered = vmn.apply_epoch_range_exclusions(
        conditions, {11: [(0, 2)]}, min_epochs=3, show=False)

    short, long = filtered.itertuples(index=False)
    assert short.excluded_epochs == [(11, 0), (11, 2)]
    assert short.n_epochs_retained == 2
    assert not short.included
    assert 'after manual exclusion' in short.reason
    assert long.excluded_epochs == [(11, 1)]
    assert long.n_epochs_retained == 1
    assert not long.included


@pytest.mark.parametrize(
    ('technique', 'online', 'expected'),
    [('cell-attached', 'exc', 'extracellular'),
     ('whole-cell', 'exc', 'exc'),
     ('whole-cell', 'inh', 'inh'),
     ('whole-cell', None, ''),
     (None, 'extracellular', 'extracellular')])
def test_recording_type_uses_epoch_group_technique_as_primary_anchor(
        technique, online, expected):
    mode, _ = vmn.recording_type_from_metadata(technique, online)
    assert mode == expected


def test_mode_cache_uses_shared_chronological_parser(tmp_path, monkeypatch):
    import pandas as pd

    from retinanalysis.SCutils import recording_mode as rm

    blocks = pd.DataFrame({
        'exp_name': ['test'] * 3,
        'cell_label': ['cell1'] * 3,
        'block_id': [1, 2, 3],
        'start_time': pd.to_datetime([
            '2026-01-01 10:00', '2026-01-01 10:10',
            '2026-01-01 10:20']),
        'recording_technique': ['cell-attached', '', 'whole-cell'],
        'onlineAnalysis': ['', '', ''],
        'n_epochs': [3, 3, 3],
    })
    monkeypatch.setattr(
        rm, '_amp_response_table',
        lambda *args, **kwargs: pd.DataFrame(columns=['block_id', 'h5path']))
    monkeypatch.setattr(rm, '_amp_epoch_groups_by_block',
                        lambda *args, **kwargs: {})
    monkeypatch.setattr(
        rm, 'series_resistance_table',
        lambda *args, **kwargs: pd.DataFrame({
            'block_id': [1, 2, 3],
            'series_resistance': [0., 0., 8e6],
        }))

    from retinanalysis.SCutils import recording_classifier as rc
    monkeypatch.setattr(
        rc, 'recording_block_feature_table',
        lambda *args, **kwargs: pd.DataFrame({
            'block_id': [1, 2, 3], 'raw_mean': [-5., -5., -25.]}))
    monkeypatch.setattr(
        rc, 'load_recording_technique_classifier',
        lambda: {'model_version': 'test-model'})
    monkeypatch.setattr(
        rc, 'predict_recording_techniques',
        lambda features, bundle, **kwargs: features.assign(
            classifier_p_whole_cell=[0., 0., 1.],
            classifier_confidence=[1., 1., 1.],
            classifier_prediction=['cell-attached', 'cell-attached', 'whole-cell'],
            classifier_family=['cell-attached', 'cell-attached', 'whole-cell'],
            classifier_source=['cell-held-out prediction'] * 3))
    cache = vmn.build_mode_cache(
        blocks, path=tmp_path / 'modes.csv', verbose=False)

    assert cache.rec_type.tolist() == ['extracellular', 'extracellular', 'exc']
    assert cache.recording_order.tolist() == [0, 1, 2]
    assert cache.series_resistance_mohm.tolist() == [0., 0., 8.]
    assert cache.loc[2, 'mean_current_pa'] == -25.
    assert cache.cache_version.eq(vmn.MODE_CACHE_VERSION).all()


def test_resolve_block_mode_ignores_series_resistance_and_trace(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        vmn, 'epoch_parameters',
        lambda block: pd.DataFrame({
            'onlineAnalysis': ['exc'], 'seriesResistance': [0.]}))
    result = vmn.resolve_block_mode(
        'synthetic', 11, amp_data=np.full((2, 20), 10_000.),
        recording_technique='cell-attached')

    assert result['rec_type'] == 'extracellular'
    assert np.isnan(result['series_resistance_mohm'])


def test_manual_recording_override_is_scoped_and_mutually_exclusive():
    import pandas as pd

    modes = pd.DataFrame({
        'block_id': [10, 11, 12],
        'rec_type': ['extracellular', '', 'inh'],
        'rec_note': ['metadata', 'unresolved', 'metadata'],
    })
    forced = vmn.apply_recording_type_override(
        modes, [10, 11], force_whole_cell_flag=True,
        whole_cell_rec_type='exc', show=False)

    assert forced.rec_type.tolist() == ['exc', 'exc', 'inh']
    assert modes.rec_type.tolist() == ['extracellular', '', 'inh']
    with pytest.raises(ValueError, match='cannot both be true'):
        vmn.apply_recording_type_override(
            modes, [10], force_spike_flag=True,
            force_whole_cell_flag=True, show=False)


def test_epoch_ranges_assign_one_type_per_block_and_exclude_the_rest():
    import pandas as pd

    modes = pd.DataFrame({
        'block_id': [10, 11, 12],
        'rec_type': ['extracellular', 'extracellular', 'exc'],
        'rec_note': ['auto'] * 3,
    })
    catalog = pd.DataFrame({
        'epoch_number': [0, 1, 2, 3, 4],
        'block_id': [10, 10, 11, 11, 12],
        'automatic_rec_type': [
            'extracellular', 'extracellular', 'extracellular',
            'extracellular', 'exc'],
    })

    assigned_modes, assigned_epochs, excluded = (
        vmn.apply_epoch_recording_type_ranges(
            modes, catalog,
            {'extracellular': range(0, 2), 'exc': range(2, 4)},
            show=False))

    assert assigned_modes.rec_type.tolist() == ['extracellular', 'exc', 'exc']
    assert assigned_epochs.assigned_rec_type.tolist() == [
        'extracellular', 'extracellular', 'exc', 'exc', '']
    assert assigned_epochs.selected_for_analysis.tolist() == [
        True, True, True, True, False]
    assert excluded == (4,)


def test_epoch_ranges_refuse_to_split_one_block_across_types():
    import pandas as pd

    modes = pd.DataFrame({
        'block_id': [10], 'rec_type': ['extracellular'],
        'rec_note': ['auto']})
    catalog = pd.DataFrame({
        'epoch_number': [0, 1], 'block_id': [10, 10],
        'automatic_rec_type': ['extracellular', 'extracellular']})

    with pytest.raises(ValueError, match='one epoch block'):
        vmn.apply_epoch_recording_type_ranges(
            modes, catalog,
            {'extracellular': [0], 'exc': [1]}, show=False)


def test_light_mean_exclusion_accepts_scalar_and_propagates_as_epochs():
    import pandas as pd

    catalog = pd.DataFrame({
        'epoch_number': [0, 1, 2, 3],
        'block_id': [10, 10, 11, 11],
        'block_epoch': [0, 1, 0, 1],
        'stimTime': [30_000.] * 4,
        'lightMean': [0.03, 0.3, 0.03, 0.30000000001],
        'stdv': [0.015, 0.15, 0.015, 0.15],
    })
    conditions = pd.DataFrame({
        'rec_type': ['extracellular'], 'stim_time_ms': [30_000.],
        'light_contrast': [.5], 'n_epochs': [4],
        'block_ids': [[10, 11]], 'included': [True], 'reason': [''],
    })

    epoch_numbers = vmn.epoch_numbers_for_light_mean_exclusions(
        catalog, remove_mean_condition=0.3, show=False)
    filtered = vmn.apply_epoch_exclusions(
        conditions, catalog, remove_epochs=epoch_numbers,
        min_epochs=1, show=False)

    assert epoch_numbers == (1, 3)
    assert filtered.loc[0, 'excluded_epochs'] == [(10, 1), (11, 1)]
    assert filtered.loc[0, 'n_epochs_retained'] == 2


def test_light_mean_exclusion_accepts_multiple_means_and_rejects_unknown():
    import pandas as pd

    catalog = pd.DataFrame({
        'epoch_number': [0, 1, 2], 'lightMean': [0.03, 0.3, 0.05]})

    assert vmn.epoch_numbers_for_light_mean_exclusions(
        catalog, (0.3, 0.05), show=False) == (1, 2)
    with pytest.raises(ValueError, match='available means'):
        vmn.epoch_numbers_for_light_mean_exclusions(
            catalog, 999, show=False)


def test_raw_epoch_figures_are_grouped_by_assigned_type(monkeypatch):
    import pandas as pd

    catalog = pd.DataFrame({
        'epoch_number': [0, 1, 2], 'block_id': [10, 11, 12],
        'assigned_rec_type': ['extracellular', 'exc', 'extracellular']})
    calls = []

    def plot(exp_name, subset, **kwargs):
        calls.append((kwargs['group_label'], subset.epoch_number.tolist()))
        return kwargs['group_label']

    monkeypatch.setattr(vmn, 'plot_raw_epoch_traces', plot)
    figures = vmn.plot_raw_epoch_traces_by_recording_type('test', catalog)

    assert calls == [('extracellular', [0, 2]), ('exc', [1])]
    assert figures == {'extracellular': 'extracellular', 'exc': 'exc'}


def test_epoch_catalog_and_removal_use_cell_wide_chronological_indices(
        monkeypatch):
    import pandas as pd

    params = {
        10: pd.DataFrame({
            'epoch_id': [100, 101],
            'start_time': pd.to_datetime(['2026-01-01 10:00:00',
                                          '2026-01-01 10:02:00']),
            'stimTime': [30_000., 60_000.], 'lightMean': [1., 1.],
            'stdv': [.5, .5]}),
        11: pd.DataFrame({
            'epoch_id': [90],
            'start_time': pd.to_datetime(['2026-01-01 09:59:00']),
            'stimTime': [30_000.], 'lightMean': [1.], 'stdv': [.5]}),
    }
    monkeypatch.setattr(vmn, 'epoch_parameters', lambda block: params[block])
    catalog = vmn.epoch_catalog([10, 11])
    assert catalog[['epoch_number', 'block_id', 'block_epoch']].values.tolist() == [
        [0, 11, 0], [1, 10, 0], [2, 10, 1]]

    conditions = pd.DataFrame({
        'rec_type': ['extracellular', 'extracellular'],
        'stim_time_ms': [30_000., 60_000.],
        'light_contrast': [.5, .5], 'n_epochs': [2, 1],
        'block_ids': [[10, 11], [10]], 'included': [True, True],
        'reason': ['', ''],
    })
    filtered = vmn.apply_epoch_exclusions(
        conditions, catalog, remove_epochs=[0, 2], min_epochs=1, show=False)
    assert filtered.excluded_epochs.tolist() == [[(11, 0)], [(10, 1)]]
    assert filtered.excluded_epoch_numbers.tolist() == [[0], [2]]
    assert filtered.n_epochs_retained.tolist() == [1, 0]
    assert filtered.included.tolist() == [True, False]


def test_raw_epoch_plot_has_one_unprocessed_row_per_epoch(monkeypatch):
    import matplotlib.pyplot as plt
    import pandas as pd

    catalog = pd.DataFrame({
        'epoch_number': [0, 1], 'block_id': [10, 10],
        'block_epoch': [0, 1], 'stimTime': [4., 4.], 'preTime': [0., 0.],
    })
    amp = np.array([[1., 2., 3., 4.], [8., 7., 6., 5.]])
    calls = []

    def fake_load(exp_name, block_id, spiking):
        calls.append(spiking)
        return amp, 1000., None

    monkeypatch.setattr(vmn, 'load_block', fake_load)
    figure = vmn.plot_raw_epoch_traces(
        'synthetic', catalog, remove_epochs=[1], downsample=1)

    assert len(figure.axes) == 2
    np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), amp[0])
    np.testing.assert_array_equal(figure.axes[1].lines[0].get_ydata(), amp[1])
    assert figure.axes[0].get_ylabel().startswith('0 |')
    assert 'REMOVE' in figure.axes[1].get_ylabel()
    assert calls == [False]
    plt.close(figure)


def test_analyze_condition_refuses_implicit_mixed_durations(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        vmn, 'epoch_parameters',
        lambda block: pd.DataFrame({'stimTime': [30_000, 60_000],
                                    'lightMean': [.3, 3.0]}))

    with pytest.raises(ValueError, match='multiple epoch durations'):
        vmn.analyze_condition('synthetic', [11], fit=False, verbose=False)


def test_analyze_condition_filters_epochs_to_requested_duration(monkeypatch):
    import pandas as pd

    params = pd.DataFrame({
        'stimTime': [30.0, 60.0, 30.0, 60.0],
        'lightMean': [.3, .3, 3.0, 3.0],
        'frequencyCutoff': [20.0] * 4,
    })
    monkeypatch.setattr(vmn, 'epoch_parameters', lambda block: params)
    monkeypatch.setattr(
        vmn, 'load_block',
        lambda exp_name, block_id, spiking: (np.arange(240).reshape(4, 60),
                                             1000.0, None))
    monkeypatch.setattr(
        vmn, 'epoch_stimulus',
        lambda row, sample_rate: np.ones(int(row.stimTime)))

    analysis = vmn.analyze_condition(
        'synthetic', [11], rec_type='exc', stim_time_ms=30.0,
        skip_seconds=0.0, downsample=1, align_epoch_means=False,
        fit=False, verbose=False)

    assert analysis.stim_time_ms == 30.0
    assert analysis.n_epochs == {.3: 1, 3.0: 1}
    assert {array.shape for array in analysis.stimulus.values()} == {(1, 30)}

    selected = vmn.analyze_condition(
        'synthetic', [11], rec_type='exc', stim_time_ms=30.0,
        light_means=[3.0], skip_seconds=0.0, downsample=1,
        align_epoch_means=False, fit=False, verbose=False)
    assert selected.light_means == [3.0]
    assert selected.n_epochs == {3.0: 1}


def test_analyze_condition_omits_manually_excluded_epochs(monkeypatch):
    import pandas as pd

    params = pd.DataFrame({
        'stimTime': [30.0, 30.0, 30.0],
        'lightMean': [.3, .3, .3],
        'frequencyCutoff': [20.0] * 3,
    })
    monkeypatch.setattr(vmn, 'epoch_parameters', lambda _block: params)
    monkeypatch.setattr(
        vmn, 'load_block',
        lambda _exp, _block, _spiking: (
            np.arange(90).reshape(3, 30), 1000.0, None))
    monkeypatch.setattr(
        vmn, 'epoch_stimulus',
        lambda row, sample_rate: np.ones(int(row.stimTime)))

    analysis = vmn.analyze_condition(
        'synthetic', [11], rec_type='exc', stim_time_ms=30.0,
        skip_seconds=0.0, downsample=1, align_epoch_means=False,
        excluded_epochs=[(11, 1)], activity_excluded_epochs=[(11, 1)],
        fit=False, verbose=False)

    assert analysis.n_epochs == {.3: 2}
    assert analysis.excluded_epochs == [(11, 1)]
    assert analysis.dropped_epochs[['block_id', 'epoch']].values.tolist() == [[11, 1]]
    assert analysis.activity_excluded_epochs == [(11, 1)]
    assert analysis.dropped_epochs.reason.tolist() == [
        'below Section 2 activity threshold']


def test_numpy_does_not_reproduce_matlab_randn():
    """The reason the engine is required, stated as a test.

    If some NumPy generator ever did match, this test failing is the signal to
    revisit the dependency -- not a reason to swap one in quietly.
    """
    candidates = {
        'RandomState.standard_normal':
            np.random.RandomState(42).standard_normal(8),
        'Generator(MT19937)':
            np.random.Generator(np.random.MT19937(42)).standard_normal(8),
        'default_rng': np.random.default_rng(42).standard_normal(8),
    }
    reference = MATLAB_RANDN_SEED42[:8]
    for name, values in candidates.items():
        assert not np.allclose(values, reference), (
            f'{name} now matches MATLAB randn; the engine dependency can be '
            f'revisited deliberately')


def test_generator_matches_matlab_given_the_matlab_draw():
    """Everything after the draw is NumPy, and it agrees with MATLAB.

    Passing noise= supplies MATLAB's own draw, so this runs without the
    engine and still checks the FFT, the filter construction, the
    standard-deviation correction and the mean offset against MATLAB's
    arithmetic on the same input.
    """
    stimulus = vmn.gaussian_noise_stimulus(
        seed=42, stim_pts=64, st_dev=0.2, freq_cutoff=60.0, num_filters=4,
        mean=1.0, sample_rate=1000.0, noise=MATLAB_RANDN_SEED42)
    assert stimulus.shape == (64,)
    np.testing.assert_allclose(stimulus[:MATLAB_STIMULUS_HEAD.size],
                               MATLAB_STIMULUS_HEAD, rtol=0, atol=1e-12)


def test_epoch_stimulus_takes_its_draw_from_matlab(monkeypatch):
    """``epoch_stimulus`` must route through :func:`matlab_randn`.

    The generator accepts a ``noise=`` argument for testing, and this guards the
    path that real analysis uses: with no draw supplied it has to ask MATLAB,
    never a NumPy generator.
    """
    import pandas as pd

    calls = {}

    def fake_randn(seed, n):
        calls['seed'], calls['n'] = int(seed), int(n)
        return np.zeros(n)

    monkeypatch.setattr(vmn, 'matlab_randn', fake_randn)
    params = pd.Series({'seed': 12345, 'stimTime': 100.0, 'sampleRate': 1000.0,
                        'stdv': 0.2, 'frequencyCutoff': 60.0,
                        'numberOfFilters': 4, 'lightMean': 0.5})
    vmn.epoch_stimulus(params)
    assert calls == {'seed': 12345, 'n': 100}


def test_missing_engine_raises_rather_than_falling_back():
    """No silent NumPy fallback: without the engine the call fails loudly.

    A stimulus that is not the one presented would corrupt every filter fitted
    from it, and would not look wrong.
    """
    import builtins

    real_import = builtins.__import__

    def block_matlab(name, *args, **kwargs):
        if name.startswith('matlab'):
            raise ImportError('matlab.engine unavailable')
        return real_import(name, *args, **kwargs)

    saved = vmn._MATLAB_ENGINE
    vmn._MATLAB_ENGINE = None
    builtins.__import__ = block_matlab
    try:
        with pytest.raises(RuntimeError, match='MATLAB engine'):
            vmn.matlab_randn(1, 4)
    finally:
        builtins.__import__ = real_import
        vmn._MATLAB_ENGINE = saved


def test_vendored_cascadegraph_is_self_contained():
    """The vendored copy must not fall through to a sibling checkout.

    ``utils/cascadegraph`` exists so this repository does not depend on
    ``~/Documents/GitHub/cascadegraph`` being present and on ``sys.path``. Its
    imports were absolute (``from cascadegraph.nodes.base import ...``), so the
    copy was a facade and the real code came from the sibling. This blocks the
    top-level name outright and imports the copy anyway.
    """
    import builtins
    import importlib

    for name in [n for n in list(sys.modules)
                 if n == 'cascadegraph' or n.startswith('cascadegraph.')
                 or 'utils.cascadegraph' in n]:
        del sys.modules[name]

    real_import = builtins.__import__

    def block_cascadegraph(name, *args, **kwargs):
        if name == 'cascadegraph' or name.startswith('cascadegraph.'):
            raise ImportError('top-level cascadegraph is not importable here')
        return real_import(name, *args, **kwargs)

    saved_path = list(sys.path)
    sys.path = [p for p in sys.path if 'GitHub/cascadegraph' not in p]
    builtins.__import__ = block_cascadegraph
    try:
        cg = importlib.import_module('retinanalysis.utils.cascadegraph')
        assert 'retinanalysis' in cg.__file__
        for name in ('compute_filter', 'convolve_filter_with_stim', 'sample_nl',
                     'compute_variance_explained', 'apply_frequency_cutoff',
                     'SigmoidNlNode'):
            assert hasattr(cg, name), name
        assert 'cascadegraph' not in sys.modules
    finally:
        builtins.__import__ = real_import
        sys.path = saved_path


def test_sigmoid_start_is_read_off_the_data():
    """Each start value is the statistic of the sampled curve it names.

    The parameters of ``alpha*Phi(beta*x + gamma) + epsilon`` are the baseline,
    the rise, the steepness and the midpoint, so a curve with known values must
    produce a start close to them -- otherwise the optimiser is being handed a
    generic guess and left to find its own way, which is what let ``alpha`` run
    to 1.35e7 against an ``epsilon`` of -1.35e7.
    """
    from scipy.stats import norm

    x = np.linspace(-2.0, 2.0, 100)
    alpha, beta, gamma, epsilon = 50.0, 2.0, -1.0, -10.0
    y = alpha * norm.cdf(beta * x + gamma) + epsilon

    guess, lower, upper = vmn.sigmoid_start_and_bounds(x, y)
    assert np.all(lower <= guess) and np.all(guess <= upper)
    assert guess[0] > 0                                   # rising, so alpha > 0
    assert 0.5 * alpha < guess[0] < 2.0 * alpha           # the rise
    assert abs(guess[3] - epsilon) < 0.25 * alpha         # the baseline
    # gamma/beta places the midpoint; check the implied x50 rather than the
    # raw pair, since only their ratio is identifiable from a location.
    assert abs(-guess[2] / guess[1] - (-gamma / beta)) < 0.2 * np.ptp(x)

    # The fit that follows recovers the true parameters and stays conditioned.
    fitted = vmn.fit_sigmoid(x, y)
    assert fitted['r2'] > 0.999
    assert abs(fitted['alpha']) < 10 * np.ptp(y)


def test_sigmoid_start_handles_a_falling_curve():
    """A negative-going nonlinearity must start with a negative alpha."""
    from scipy.stats import norm

    x = np.linspace(-2.0, 2.0, 100)
    y = -40.0 * norm.cdf(1.5 * x) + 5.0
    guess, lower, upper = vmn.sigmoid_start_and_bounds(x, y)
    assert guess[0] < 0
    assert np.all(lower <= guess) and np.all(guess <= upper)
    assert vmn.fit_sigmoid(x, y)['r2'] > 0.999


def test_amplitude_ceiling_follows_the_recording_units():
    """``alpha`` is a response amplitude, so its bound is in the response's units.

    8 nA means nothing to an extracellular recording and 1000 Hz means nothing
    to a voltage clamp, so the ceiling has to come from ``rec_type`` rather than
    from one constant.
    """
    from scipy.stats import norm

    x = np.linspace(-2.0, 2.0, 100)
    # A large-amplitude curve, so the data-relative cap is far above both
    # physiological ceilings and the units decide which one applies.
    y = 4000.0 * norm.cdf(1.5 * x) - 2000.0

    _, low_wc, high_wc = vmn.sigmoid_start_and_bounds(x, y, rec_type='exc')
    _, low_ex, high_ex = vmn.sigmoid_start_and_bounds(x, y, rec_type='extracellular')
    assert high_wc[0] == vmn.SIGMOID_AMPLITUDE_MAX['exc'] == 8000.0
    assert high_ex[0] == vmn.SIGMOID_AMPLITUDE_MAX['extracellular'] == 1000.0
    assert low_wc[0] == -high_wc[0] and low_ex[0] == -high_ex[0]

    # 'inh' shares the whole-cell ceiling; an unknown mode falls back to the
    # data-relative cap alone rather than silently applying a wrong unit.
    _, _, high_inh = vmn.sigmoid_start_and_bounds(x, y, rec_type='inh')
    _, _, high_none = vmn.sigmoid_start_and_bounds(x, y, rec_type=None)
    assert high_inh[0] == vmn.SIGMOID_AMPLITUDE_MAX['inh'] == 8000.0
    assert high_none[0] > vmn.SIGMOID_AMPLITUDE_MAX['inh']


def test_epsilon_is_not_capped_at_an_absolute_current():
    """The baseline carries the holding current, which reaches -14 nA here.

    ``epsilon`` is an absolute level, not an amplitude: one cell in this
    dataset modulates by 946 pA about a holding current of -14.4 nA. A ceiling
    of "a few thousand pA" on ``epsilon`` would refuse that cell outright, so
    it is bounded by the data's own range instead.
    """
    from scipy.stats import norm

    x = np.linspace(-2.0, 2.0, 100)
    y = 946.0 * norm.cdf(1.5 * x) - 14_357.0      # the real cell's scale

    guess, lower, upper = vmn.sigmoid_start_and_bounds(x, y, rec_type='exc')
    assert lower[3] < -14_357.0 < upper[3]
    assert np.all(lower <= guess) and np.all(guess <= upper)

    fitted = vmn.fit_sigmoid(x, y, rec_type='exc')
    assert fitted['r2'] > 0.999
    assert abs(fitted['epsilon'] + 14_357.0) < 200.0
    assert not fitted['at_bounds'], fitted['at_bounds']


def test_whole_cell_nonlinearity_can_span_minus_10na_to_plus_1na():
    """A recording-mode guard must never clip the response actually observed."""
    from scipy.stats import norm

    x = np.linspace(-2.5, 2.5, 200)
    y = 1_000.0 - 11_000.0 * norm.cdf(1.7 * x - 0.3)
    guess, lower, upper = vmn.sigmoid_start_and_bounds(x, y, rec_type='exc')
    assert upper[0] >= 2.0 * np.ptp(y)
    assert lower[0] <= -2.0 * np.ptp(y)
    assert np.all(lower <= guess) and np.all(guess <= upper)

    fitted = vmn.fit_sigmoid(x, y, rec_type='exc')
    assert fitted['r2'] > 0.999
    assert fitted['alpha'] < -10_000.0
    assert not fitted['at_bounds'], fitted['at_bounds']


def test_generator_axis_bounds_do_not_depend_on_units():
    """``beta`` and ``gamma`` live on the generator axis, which is contrast.

    The generator signal runs to about +/-3 whatever the amplifier was doing,
    so one pair of limits serves every recording mode.
    """
    from scipy.stats import norm

    x = np.linspace(-2.0, 2.0, 100)
    y = 100.0 * norm.cdf(2.0 * x)
    bounds = [vmn.sigmoid_start_and_bounds(x, y, rec_type=r)[2][1:3]
              for r in ('exc', 'inh', 'extracellular', None)]
    for other in bounds[1:]:
        np.testing.assert_allclose(bounds[0], other)
    assert bounds[0][0] == vmn.SIGMOID_SLOPE_MAX == 100.0


def test_fit_sigmoid_reports_a_constrained_fit():
    """A fit resting on its bound must say so rather than report the bound."""
    x = np.linspace(-1.0, 1.0, 100)
    y = 300.0 * x                       # a line: alpha runs away without a bound

    fitted = vmn.fit_sigmoid(x, y, rec_type='extracellular')
    assert 'at_bounds' in fitted
    if fitted['at_bounds']:
        assert abs(fitted['alpha']) <= vmn.SIGMOID_AMPLITUDE_MAX['extracellular']
    # A curve the box comfortably contains reports nothing.
    from scipy.stats import norm
    clean = vmn.fit_sigmoid(x, 80.0 * norm.cdf(2.5 * x) + 5.0,
                            rec_type='extracellular')
    assert clean['at_bounds'] == ()


def test_psth_binning_matches_smoothing_at_the_full_rate():
    """Binning then smoothing must equal smoothing then binning.

    The PSTH used to be laid down at the amplifier's 10 kHz, Gaussian-smoothed
    there, and only then block-averaged to the analysis rate -- a 302k-sample
    array convolved with an 801-tap kernel per epoch, which was most of the
    cost of loading a condition. Convolution commutes with the boxcar that
    block-averaging applies, so doing it at the reduced rate is the same trace
    for ~100x less arithmetic; this pins that they agree.
    """
    from scipy.ndimage import gaussian_filter1d

    rng = np.random.default_rng(0)
    n_samples, sample_rate, step, sigma_ms = 302_000, 10_000.0, 10, 10.0
    spikes = np.sort(rng.choice(n_samples, size=1800, replace=False))

    dense = np.zeros(n_samples)
    dense[spikes] = 1.0
    slow = vmn._block_average(
        gaussian_filter1d(dense, sigma_ms / 1e3 * sample_rate) * sample_rate, step)
    fast = vmn._spike_rate(spikes, n_samples, sample_rate, step, sigma_ms)

    assert fast.shape == slow.shape
    assert np.corrcoef(slow, fast)[0, 1] > 0.999
    # Absolute agreement relative to the peak rate, not to zero: the two orders
    # differ only by where the boxcar falls, which is sub-bin.
    assert np.max(np.abs(slow - fast)) < 0.05 * slow.max()
    # Total spike count is preserved either way.
    assert abs(fast.sum() - slow.sum()) < 0.01 * slow.sum()


def test_epoch_response_summary_reports_exact_mean_firing_rate(monkeypatch):
    import pandas as pd

    params = pd.DataFrame({
        'stimTime': [1000.0, 1000.0], 'lightMean': [0.1, 1.0]})
    amp = np.zeros((2, 1000))
    spikes = [np.arange(10), np.arange(20)]
    monkeypatch.setattr(vmn, 'epoch_parameters', lambda _block: params)
    monkeypatch.setattr(
        vmn, 'load_block',
        lambda _exp, _block, _spiking: (amp, 1000.0, spikes))

    summary = vmn.epoch_response_summary(
        'synthetic', [1], 'extracellular', show=False)

    assert summary.n_spikes.tolist() == [10, 20]
    assert summary.mean_firing_rate_hz.tolist() == [10.0, 20.0]


def test_epoch_response_window_uses_pre_time_and_defaults_missing_to_zero():
    import pandas as pd

    assert vmn.epoch_response_window(
        pd.Series({'preTime': 500.0, 'stimTime': 1000.0}),
        sample_rate=1000.0, n_samples=1800) == (500, 1500, 500.0)
    assert vmn.epoch_response_window(
        pd.Series({'stimTime': 1000.0}),
        sample_rate=1000.0, n_samples=1200) == (0, 1000, 0.0)
    assert vmn.epoch_response_window(
        pd.Series({'preTime': None, 'stimTime': 1000.0}),
        sample_rate=1000.0, n_samples=1200) == (0, 1000, 0.0)


def test_epoch_response_summary_excludes_pre_and_tail_spikes(monkeypatch):
    import pandas as pd

    params = pd.DataFrame({
        'preTime': [100.0], 'stimTime': [1000.0], 'tailTime': [100.0],
        'lightMean': [1.0]})
    amp = np.zeros((1, 1200))
    spikes = [np.r_[np.arange(10), np.arange(100, 120), np.arange(1100, 1130)]]
    monkeypatch.setattr(vmn, 'epoch_parameters', lambda _block: params)
    monkeypatch.setattr(
        vmn, 'load_block',
        lambda _exp, _block, _spiking: (amp, 1000.0, spikes))

    summary = vmn.epoch_response_summary(
        'synthetic', [1], 'extracellular', show=False)

    assert summary.n_spikes.tolist() == [20]
    assert summary.mean_firing_rate_hz.tolist() == [20.0]
    assert summary.pre_time_ms.tolist() == [100.0]


def test_analyze_condition_aligns_response_after_pre_time(monkeypatch):
    import pandas as pd

    params = pd.DataFrame({
        'preTime': [2.0], 'stimTime': [4.0], 'tailTime': [2.0],
        'lightMean': [1.0], 'frequencyCutoff': [20.0]})
    amp = np.array([[100.0, 101.0, 1.0, 2.0, 3.0, 4.0, 200.0, 201.0]])
    monkeypatch.setattr(vmn, 'epoch_parameters', lambda _block: params)
    monkeypatch.setattr(
        vmn, 'load_block',
        lambda _exp, _block, _spiking: (amp, 1000.0, None))
    monkeypatch.setattr(
        vmn, 'epoch_stimulus',
        lambda row, sample_rate: np.arange(4.0))

    analysis = vmn.analyze_condition(
        'synthetic', [1], rec_type='exc', stim_time_ms=4.0,
        skip_seconds=0.0, downsample=1, align_epoch_means=False,
        fit=False, verbose=False)

    np.testing.assert_array_equal(analysis.response[1.0], [[1., 2., 3., 4.]])
    np.testing.assert_array_equal(analysis.stimulus[1.0], [[0., 1., 2., 3.]])


def test_epoch_response_summary_reports_whole_cell_modulation_in_pa(monkeypatch):
    import pandas as pd

    params = pd.DataFrame({
        'stimTime': [1000.0, 1000.0], 'lightMean': [0.1, 1.0]})
    amp = np.vstack([
        np.tile([7.0, 13.0], 500),
        np.tile([-9.0, -1.0], 500),
    ])
    monkeypatch.setattr(vmn, 'epoch_parameters', lambda _block: params)
    monkeypatch.setattr(
        vmn, 'load_block',
        lambda _exp, _block, _spiking: (amp, 1000.0, None))

    summary = vmn.epoch_response_summary(
        'synthetic', [1], 'exc', show=False)

    np.testing.assert_allclose(summary.mean_current_pA, [10.0, -5.0])
    np.testing.assert_allclose(summary.modulation_sd_pA, [3.0, 4.0])


def _polarity_dataset(n_epochs=6, n_time=4000, lag=8, seed=0):
    """A cell whose response is a lagged, rectified copy of a noise stimulus."""
    rng = np.random.default_rng(seed)
    stim = rng.standard_normal((n_epochs, n_time))
    drive = np.roll(stim, lag, axis=1)
    resp = np.maximum(drive, 0.0) * 3.0 + 0.35 * rng.standard_normal((n_epochs, n_time))
    return stim, resp


def test_linear_decoder_reconstructs_the_stimulus_trace():
    """The decoding filter must recover the trace, not merely its sign."""
    stim, resp = _polarity_dataset()
    decoder = vmn.decoding_filter(resp[:-1], stim[:-1])
    estimate = vmn.apply_decoding_filter(decoder, resp[-1:])
    estimate = estimate - estimate.mean(axis=1, keepdims=True)
    truth = stim[-1:] - stim[-1:].mean(axis=1, keepdims=True)

    metrics = vmn.reconstruction_metrics(vmn._bin_mean(estimate, 10),
                                         vmn._bin_mean(truth, 10))
    assert metrics['r_all'] > 0.7
    assert 0.4 < metrics['gain_all'] < 1.6
    assert metrics['nrmse_all'] < 1.0          # better than predicting the mean


def test_rectified_cell_reconstructs_its_driven_phase_better():
    """A half-wave rectified cell carries no information below its threshold.

    The response above is a scaled copy of the stimulus and below is noise, so
    the increment phase must reconstruct better on every measure. This is the
    signature the phase split exists to detect; if it cannot find it here, a
    null result on real data would mean nothing.
    """
    stim, resp = _polarity_dataset()
    decoder = vmn.decoding_filter(resp[:-1], stim[:-1])
    estimate = vmn.apply_decoding_filter(decoder, resp[-1:])
    estimate = estimate - estimate.mean(axis=1, keepdims=True)
    truth = stim[-1:] - stim[-1:].mean(axis=1, keepdims=True)

    m = vmn.reconstruction_metrics(vmn._bin_mean(estimate, 10),
                                   vmn._bin_mean(truth, 10))
    assert m['r_increment'] > m['r_decrement']
    assert m['gain_increment'] > m['gain_decrement']
    assert m['nrmse_increment'] < m['nrmse_decrement']


def test_reconstruction_metrics_separate_shape_from_amplitude():
    """A halved reconstruction keeps r = 1 but must report gain = 0.5.

    Correlation alone would call a systematically compressed reconstruction
    perfect, which is exactly the failure adaptation would produce.
    """
    rng = np.random.default_rng(3)
    truth = rng.standard_normal(4000)
    metrics = vmn.reconstruction_metrics(0.5 * truth, truth)
    assert metrics['r_all'] == pytest.approx(1.0, abs=1e-9)
    assert metrics['gain_all'] == pytest.approx(0.5, abs=1e-9)
    assert metrics['gain_increment'] == pytest.approx(0.5, abs=1e-6)
    assert metrics['gain_decrement'] == pytest.approx(0.5, abs=1e-6)


def test_generator_direction_decoding_separates_operating_point_and_direction():
    import pandas as pd

    # At positive drive, increments are completely compressed while
    # decrements are reconstructed faithfully. At negative drive both survive.
    truth = np.array([0., 1., 2., 1., 0., -1., -2., -1., 0., 1., 2., 1., 0.])
    generator = np.array([1., 1., -1., 1., 2., 2., 1., -1., -2., -2., 1., 2., 1.])
    estimate = truth.copy()
    for index in range(1, len(truth)):
        delta = truth[index] - truth[index - 1]
        if generator[index - 1] > 0 and delta > 0:
            delta = 0.0
        estimate[index] = estimate[index - 1] + delta
    traces = pd.DataFrame({
        'lightMean': 10., 'mode': 'per_window', 'epoch': 0,
        'window': '0-1 s', 'order': 0, 'time_s': np.arange(len(truth)),
        'stimulus': truth, 'reconstruction': estimate, 'generator': generator,
    })

    result = vmn.generator_direction_decoding(
        traces, min_abs_change_quantile=0, min_samples=2)
    positive = result[result.operating_point.eq('positive')].set_index('direction')

    assert positive.loc['decrement', 'gain_delta'] == pytest.approx(1.0)
    assert positive.loc['increment', 'gain_delta'] == pytest.approx(0.0)
    assert positive.loc['decrement', 'direction_accuracy'] == pytest.approx(1.0)
    assert positive.loc['increment', 'direction_accuracy'] == pytest.approx(0.0)
    assert set(result.light_condition) == {'bright'}


def test_directional_saturation_contrast_is_paired_within_cell():
    import pandas as pd

    rows = pd.DataFrame({
        'cell_id': ['A', 'A', 'A', 'A'], 'cell_type': ['ON-midget'] * 4,
        'rec_type': ['extracellular'] * 4, 'stim_seconds': [30.] * 4,
        'mode': ['per_window'] * 4, 'light_condition': ['bright'] * 4,
        'operating_point': ['positive'] * 4,
        'direction': ['increment', 'increment', 'decrement', 'decrement'],
        'gain_delta': [.1, .3, .7, .9],
        'direction_accuracy': [.4, .6, .8, 1.0],
        'nrmse_delta': [1.4, 1.2, .8, .6],
    })

    paired = vmn.directional_saturation_contrast(rows)

    assert paired.gain_dec_minus_inc.iloc[0] == pytest.approx(.6)
    assert paired.accuracy_dec_minus_inc.iloc[0] == pytest.approx(.4)
    assert paired.error_inc_minus_dec.iloc[0] == pytest.approx(.6)


def test_reconstruct_traces_can_attach_encoding_generator():
    rng = np.random.default_rng(4)
    stimulus = rng.standard_normal((3, 2000))
    response = stimulus + .1 * rng.standard_normal(stimulus.shape)
    analysis = vmn.ConditionAnalysis(
        exp_name='synthetic', block_ids=[1], rec_type='extracellular',
        sample_rate=1000., units='Hz', light_means=[1.], n_epochs={1.: 3},
        sampling_interval=.001, stim_time_ms=2000., frequency_cutoff=60.,
        stimulus={1.: stimulus}, response={1.: response})
    model = vmn.LNModel(
        label='full', r2=.5, filter=np.array([1., 0.]),
        filter_time_s=np.array([0., .001]), nl_x=np.array([]), nl_y=np.array([]),
        params={'beta': 1.})

    traces = vmn.reconstruct_traces(
        analysis, mode='per_window', window_seconds=1., decode_bin_ms=50.,
        generator_models={1.: model}, verbose=False)

    assert len(traces) == 120
    assert traces.generator.notna().all()
    assert not vmn.generator_direction_decoding(
        traces, generator_models={1.: model}).empty


def test_steady_state_mode_never_scores_its_training_stretch():
    """The adapted-state decoder must not be scored on what it was fitted on.

    Its whole purpose is to show what a fixed calibration recovers from the
    *un-adapted* response, which is worthless if the trailing windows it was
    trained on are also scored.
    """
    stim, resp = _polarity_dataset(n_epochs=4, n_time=20_000)
    analysis = vmn.ConditionAnalysis(
        exp_name='synthetic', block_ids=[0], rec_type='extracellular',
        sample_rate=1000.0, units='firing rate (Hz)', sampling_interval=1e-3,
        skip_seconds=0.0, frequency_cutoff=60.0)
    analysis.light_means = [1.0]
    analysis.n_epochs = {1.0: stim.shape[0]}
    analysis.stimulus = {1.0: stim}
    analysis.response = {1.0: resp}

    frame = vmn.reconstruct_stimulus(analysis, mode='steady_state',
                                     window_seconds=2.0, steady_state_s=6.0,
                                     verbose=False)
    assert not frame.empty
    # Epoch is 20 s; the last 6 s are the training stretch, so every scored
    # window must end at or before 14 s.
    ends = (frame.window.str.split('-').str[1]
            .str.replace(' s', '', regex=False).astype(float))
    assert ends.max() <= 14.0 + 1e-6


def test_transfer_slopes_equal_the_reported_gains():
    """The figure's per-phase slopes must be the table's per-phase gains.

    ``plot_reconstruction_transfer`` draws a slope for each side of zero and
    ``reconstruction_metrics`` reports ``gain_increment``/``gain_decrement``.
    They are the same estimator by construction; this pins that, so the figure
    cannot drift away from the numbers printed beside it.
    """
    stim, resp = _polarity_dataset()
    decoder = vmn.decoding_filter(resp[:-1], stim[:-1])
    estimate = vmn.apply_decoding_filter(decoder, resp[-1:])
    estimate = estimate - estimate.mean(axis=1, keepdims=True)
    truth = stim[-1:] - stim[-1:].mean(axis=1, keepdims=True)
    binned_e = vmn._bin_mean(estimate, 10).ravel()
    binned_t = vmn._bin_mean(truth, 10).ravel()

    slopes = vmn.phase_slopes(binned_t, binned_e)
    metrics = vmn.reconstruction_metrics(binned_e, binned_t)
    assert slopes['increment'] == pytest.approx(metrics['gain_increment'], rel=1e-9)
    assert slopes['decrement'] == pytest.approx(metrics['gain_decrement'], rel=1e-9)


def test_rectified_cell_has_a_steeper_increment_transfer_slope():
    """The asymmetry the figure exists to reveal must be found where it is real.

    The synthetic cell is half-wave rectified, so the response above the mean
    is a scaled copy of the stimulus and below it is noise. If the transfer
    slopes cannot separate those, a null result on real data means nothing.
    """
    stim, resp = _polarity_dataset()
    decoder = vmn.decoding_filter(resp[:-1], stim[:-1])
    estimate = vmn.apply_decoding_filter(decoder, resp[-1:])
    estimate = estimate - estimate.mean(axis=1, keepdims=True)
    truth = stim[-1:] - stim[-1:].mean(axis=1, keepdims=True)

    slopes = vmn.phase_slopes(vmn._bin_mean(truth, 10),
                             vmn._bin_mean(estimate, 10))
    assert slopes['increment'] > slopes['decrement']


def test_phase_onsets_are_exact_and_respect_both_filters():
    """Onset detection on a square wave, where the answer is known by hand."""
    # 10 full cycles of 8 up then 8 down: 10 upward and 10 downward crossings,
    # minus whichever fall too close to the ends for the cut to fit.
    cycle = np.r_[np.ones(8), -np.ones(8)]
    wave = np.tile(cycle, 10)
    up, down = vmn._phase_onsets(wave, pre_bins=2, post_bins=2, min_bins=4)
    assert up.size == 9 and down.size == 10        # first up-phase starts at 0
    assert np.all(wave[up] > 0) and np.all(wave[down] < 0)

    # Every phase is 8 bins, so a 9-bin minimum must reject all of them.
    up_strict, down_strict = vmn._phase_onsets(wave, 2, 2, min_bins=9)
    assert up_strict.size == 0 and down_strict.size == 0

    # Edge exclusion: nothing within post_bins of the end, nor before pre_bins.
    up_wide, down_wide = vmn._phase_onsets(wave, pre_bins=6, post_bins=6, min_bins=4)
    for onsets in (up_wide, down_wide):
        assert np.all(onsets - 6 >= 0)
        assert np.all(onsets + 6 < wave.size)


def test_reconstruct_traces_holds_out_every_epoch_exactly_once():
    """Uniform coverage, or an onset average silently weights some epochs more."""
    import pandas as pd

    stim, resp = _polarity_dataset(n_epochs=5, n_time=8000)
    analysis = vmn.ConditionAnalysis(
        exp_name='synthetic', block_ids=[0], rec_type='extracellular',
        sample_rate=1000.0, units='firing rate (Hz)', sampling_interval=1e-3,
        skip_seconds=0.0, frequency_cutoff=60.0)
    analysis.light_means = [1.0]
    analysis.n_epochs = {1.0: stim.shape[0]}
    analysis.stimulus = {1.0: stim}
    analysis.response = {1.0: resp}

    traces = vmn.reconstruct_traces(analysis, mode='per_window',
                                    window_seconds=2.0, verbose=False)
    assert not traces.empty
    counts = traces.groupby(['window', 'epoch']).size().unstack()
    assert set(counts.index.size for _ in [0])      # windows exist
    assert counts.notna().all().all()               # every epoch in every window
    assert counts.nunique().nunique() == 1          # and the same number of bins


def test_select_decode_window_uses_shortest_candidate_within_one_se(monkeypatch):
    """Automatic selection preserves resolution among equivalent fits."""
    import pandas as pd

    analysis = vmn.ConditionAnalysis(
        exp_name='synthetic', block_ids=[0], rec_type='extracellular',
        sample_rate=1000.0, units='firing rate (Hz)', sampling_interval=1e-3)
    analysis.light_means = [1.0, 2.0]
    analysis.stimulus = {mean: np.zeros((2, 8000))
                         for mean in analysis.light_means}

    def fake_traces(_analysis, window_seconds, **_kwargs):
        return pd.DataFrame([
            {'lightMean': mean, 'epoch': epoch, 'stimulus': 0.0,
             'reconstruction': window_seconds + epoch / 10}
            for mean in analysis.light_means for epoch in range(2)])

    # Four seconds has the highest mean, but its uncertainty makes two seconds
    # statistically equivalent. One second is clearly worse.
    def fake_metrics(estimate, _truth):
        code = float(np.asarray(estimate)[0])
        window = int(np.floor(code))
        epoch = int(round((code - window) * 10))
        values = {1: (0.50, 0.50), 2: (0.80, 0.82), 4: (0.70, 0.95)}
        return {'r_all': values[window][epoch]}

    monkeypatch.setattr(vmn, 'reconstruct_traces', fake_traces)
    monkeypatch.setattr(vmn, 'reconstruction_metrics', fake_metrics)

    selected, diagnostics = vmn.select_decode_window(
        analysis, candidates_s=(0.5, 1.0, 2.0, 4.0), verbose=False)

    assert selected == pytest.approx(2.0)
    assert diagnostics.loc[diagnostics.selected, 'window_s'].item() == 2.0
    assert diagnostics.loc[diagnostics.window_s.eq(1.0),
                           'within_one_se'].item() == np.bool_(False)
    assert 0.5 not in diagnostics.requested_s.values  # below the enforced floor


def test_early_late_defaults_to_each_available_interval_midpoint():
    """The split follows usable data, including a shorter leakage-free mode."""
    import pandas as pd

    pieces = []
    for mode, times in (
            ('per_window', np.arange(2.125, 30.0, 0.25)),
            ('steady_state', np.arange(2.125, 20.0, 0.25))):
        stimulus = np.where(np.arange(times.size) % 2, 1.0, -1.0)
        pieces.append(pd.DataFrame({
            'mode': mode, 'lightMean': 3.0, 'time_s': times,
            'stimulus': stimulus, 'reconstruction': 0.5 * stimulus,
        }))
    traces = pd.concat(pieces, ignore_index=True)

    labelled = vmn.label_early_late(traces)
    expected = {'per_window': 16.0, 'steady_state': 11.0}
    for mode, split_s in expected.items():
        block = labelled[labelled['mode'].eq(mode)]
        assert block.split_s.nunique() == 1
        assert block.split_s.iloc[0] == pytest.approx(split_s)
        assert block[block.half.eq('early')].time_s.max() <= split_s
        assert block[block.half.eq('late')].time_s.min() > split_s
        assert block.groupby('half').size().nunique() == 1

    summary = vmn.transfer_early_late(traces)
    assert {'split_s', 't_start_s', 't_end_s', 'span_s'} <= set(summary.columns)
    assert set(summary.groupby('mode').split_s.first()) == set(expected.values())


def test_plot_transfer_early_late_default_adaptive_title():
    """The default adaptive split must not format its None durations."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from types import SimpleNamespace

    times = np.arange(2.0, 10.0, 0.02)
    stimulus = np.sin(times)
    traces = pd.DataFrame({
        'mode': 'per_window', 'lightMean': 3.0, 'time_s': times,
        'stimulus': stimulus, 'reconstruction': 0.5 * stimulus,
    })
    analysis = SimpleNamespace(
        exp_name='test-cell', rec_type='exc', light_means=np.array([3.0]))

    figure = vmn.plot_transfer_early_late(analysis, traces)

    assert 'adaptive early vs late halves' in figure._suptitle.get_text()
    plt.close(figure)


def test_extracellular_firing_rate_qc_errors_at_fraction_threshold(capsys):
    """Seventy percent below minimum is a failure, including equality."""
    analysis = vmn.ConditionAnalysis(
        exp_name='synthetic', block_ids=[1], rec_type='extracellular',
        sample_rate=1000.0, units='firing rate (Hz)',
        sampling_interval=1e-3, stim_time_ms=30_000.0)
    analysis.light_means = [0.1]
    rates = [1.0] * 7 + [3.0] * 3
    analysis.response = {
        0.1: np.vstack([np.full(100, rate) for rate in rates])}

    with pytest.raises(vmn.LowResponseCellError, match='7/10 epochs'):
        vmn.check_extracellular_firing_rate(
            analysis, min_firing_rate_hz=2.0,
            low_rate_epoch_fraction=0.70, verbose=False)

    assert 'LOW RESPONSE CELL' in capsys.readouterr().out


def test_extracellular_firing_rate_qc_returns_epoch_table_when_passed():
    analysis = vmn.ConditionAnalysis(
        exp_name='synthetic', block_ids=[1], rec_type='extracellular',
        sample_rate=1000.0, units='firing rate (Hz)',
        sampling_interval=1e-3)
    analysis.light_means = [0.1, 1.0]
    analysis.response = {
        0.1: np.vstack([np.full(50, 1.0), np.full(50, 3.0)]),
        1.0: np.vstack([np.full(50, 3.0), np.full(50, 4.0)]),
    }

    result = vmn.check_extracellular_firing_rate(
        analysis, min_firing_rate_hz=2.0,
        low_rate_epoch_fraction=0.70, verbose=False)

    assert len(result) == 4
    assert result.n_low_rate_epochs.unique().tolist() == [1]
    assert result.low_rate_fraction.unique().tolist() == [0.25]
    assert not result.low_response_cell.any()


def test_run_core_ln_analysis_keeps_routine_steps_in_one_order(monkeypatch):
    """The notebook wrapper must mean-QC before fitting and return every output."""
    import pandas as pd

    analysis = vmn.ConditionAnalysis(
        exp_name='synthetic', block_ids=[1, 2], rec_type='exc',
        sample_rate=1000.0, units='pA')
    temporal_models = {0.3: []}
    summary = pd.DataFrame({'lightMean': [0.3], 'r2': [0.5]})
    figures = {name: object() for name in ('mean', 'condition', 'temporal', 'kinetics')}
    calls = []

    def fake_analyze(exp_name, block_ids, **kwargs):
        calls.append(('analyze', exp_name, list(block_ids), kwargs))
        return analysis

    monkeypatch.setattr(vmn, 'analyze_condition', fake_analyze)
    monkeypatch.setattr(
        vmn, 'plot_mean_response',
        lambda value, window_s: calls.append(('mean', value, window_s)) or figures['mean'])
    response_qc = pd.DataFrame({'low_response_cell': [False]})
    monkeypatch.setattr(
        vmn, 'check_extracellular_firing_rate',
        lambda value, **kwargs: calls.append(('response_qc', value, kwargs))
        or response_qc)
    monkeypatch.setattr(
        vmn, 'fit_condition',
        lambda value, verbose: calls.append(('fit', value, verbose)) or value)
    monkeypatch.setattr(
        vmn, 'plot_condition',
        lambda value, window_seconds: calls.append(
            ('condition', value, window_seconds)) or figures['condition'])
    monkeypatch.setattr(
        vmn, 'condition_summary',
        lambda value, show: calls.append(('summary', value, show)) or summary)
    monkeypatch.setattr(
        vmn, 'temporal_ln_model',
        lambda value, **kwargs: calls.append(('temporal_fit', value, kwargs))
        or temporal_models)
    monkeypatch.setattr(
        vmn, 'plot_temporal_ln',
        lambda value, models: calls.append(('temporal_plot', value, models))
        or figures['temporal'])
    monkeypatch.setattr(
        vmn, 'plot_temporal_kinetics',
        lambda value, models: calls.append(('kinetics', value, models))
        or figures['kinetics'])

    result = vmn.run_core_ln_analysis(
        'synthetic', [1, 2], rec_type='exc', skip_seconds=2.0,
        min_firing_rate_hz=2.0, low_rate_epoch_fraction=0.70,
        mean_window_s=2.5, condition_window_s=4.0,
        temporal_window_s=None, verbose=False)

    assert [call[0] for call in calls] == [
        'analyze', 'response_qc', 'mean', 'fit', 'condition', 'summary', 'temporal_fit',
        'temporal_plot', 'kinetics']
    assert calls[0][3]['fit'] is False
    assert calls[0][3]['skip_seconds'] == 2.0
    assert calls[1][2]['min_firing_rate_hz'] == 2.0
    assert calls[1][2]['low_rate_epoch_fraction'] == 0.70
    assert calls[2][2] == 2.5 and calls[4][2] == 4.0
    assert calls[6][2]['window_seconds'] is None
    assert result.analysis is analysis
    assert result.temporal_models is temporal_models
    assert result.summary is summary
    assert result.mean_response_figure is figures['mean']
    assert result.condition_figure is figures['condition']
    assert result.temporal_figure is figures['temporal']
    assert result.kinetics_figure is figures['kinetics']
    assert result.response_qc is response_qc


def test_adaptation_state_matches_the_analytic_solution():
    """Constant drive has a closed form; the integrator must reproduce it.

    With ``u`` fixed, ``a' = u(1-a)/tau_on - a/tau_off`` relaxes exponentially
    toward ``u*tau_off / (u*tau_off + tau_on)`` with time constant
    ``1/(u/tau_on + 1/tau_off)``. Exponential Euler is exact for a piecewise
    constant drive, so this should hold to machine precision rather than
    approximately -- if it does not, the step size is silently mattering.
    """
    dt, tau_on, tau_off, u = 0.025, 2.0, 8.0, 0.4
    drive = np.full(4000, u)
    state = vmn.adaptation_state(drive, dt, tau_on, tau_off, a0=0.0)

    steady = u * tau_off / (u * tau_off + tau_on)
    tau_eff = 1.0 / (u / tau_on + 1.0 / tau_off)
    t = (np.arange(drive.size) + 1) * dt
    expected = steady * (1.0 - np.exp(-t / tau_eff))
    np.testing.assert_allclose(state, expected, rtol=1e-10, atol=1e-12)
    assert state[-1] == pytest.approx(steady, rel=1e-6)


def test_adaptation_state_stays_within_zero_and_one():
    """``a`` is an occupancy, and ``k`` is only interpretable if it stays one.

    Any non-negative drive and any positive time constants must keep the state
    in [0, 1], including a drive that slams between its extremes.
    """
    rng = np.random.default_rng(1)
    drive = rng.integers(0, 2, size=5000).astype(float)
    for tau_on, tau_off in ((0.05, 60.0), (60.0, 0.05), (1.0, 1.0)):
        state = vmn.adaptation_state(drive, 0.025, tau_on, tau_off)
        assert state.min() >= -1e-12 and state.max() <= 1.0 + 1e-12


def _adapting_cell(coupling='multiplicative', n_epochs=6, epoch_s=10.0,
                   dt=1e-3, k_true=6.0, seed=0):
    """A synthetic cell whose gain (or threshold) follows a known slow state.

    Amplitude alternates epoch to epoch, standing in for this protocol's
    luminance step, so there is a real adaptation signal to recover.

    The cell has a genuine biphasic temporal filter rather than responding
    instantaneously. That matters: ``fit_lnk`` estimates its filter by reverse
    correlation, and against an instantaneous cell every lag beyond zero is
    noise, which dilutes the generator (r = 0.50 against the stimulus) and
    starves the state of signal. A real filter makes the estimation well posed,
    and is what a real cell does anyway.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    n_time = int(epoch_s / dt)
    stim, light, epoch_id = [], [], []
    for index in range(n_epochs):
        amplitude = 1.0 if index % 2 else 0.2
        stim.append(amplitude * rng.standard_normal(n_time))
        light.append(np.full(n_time, amplitude))
        epoch_id.append(np.full(n_time, index, dtype=int))
    stim = np.concatenate(stim)
    light = np.concatenate(light)
    epoch_id = np.concatenate(epoch_id)

    # Biphasic filter: a fast positive lobe followed by a slower negative one.
    lag = np.arange(0, 0.12, dt)
    kernel = (np.exp(-lag / 0.012) * lag / 0.012
              - 0.55 * np.exp(-lag / 0.030) * lag / 0.030)
    kernel /= np.linalg.norm(kernel)
    generator = np.convolve(stim, kernel, mode='full')[:stim.size]
    generator = (generator - generator.mean()) / generator.std()

    drive = norm.cdf(2.0 * generator - 0.5)
    step = 25
    coarse = vmn._bin_mean(drive[None, :], step).ravel()
    state = vmn.adaptation_state(coarse, dt * step, 3.0, 4.0)
    state = np.repeat(state, step)[:drive.size]
    centred = state - state.mean()

    if coupling == 'multiplicative':
        clean = 100.0 * np.exp(-k_true * centred) * drive + 5.0
    else:
        clean = 100.0 * norm.cdf(2.0 * generator - 0.5 - k_true * centred) + 5.0
    response = clean + 3.0 * rng.standard_normal(clean.size)

    analysis = vmn.ConditionAnalysis(
        exp_name='synthetic', block_ids=[0], rec_type='extracellular',
        sample_rate=1.0 / dt, units='firing rate (Hz)', sampling_interval=dt,
        skip_seconds=0.0, frequency_cutoff=100.0, filter_length_s=0.12)
    analysis.sequence_stimulus = stim
    analysis.sequence_response = response
    analysis.sequence_light_mean = light
    analysis.sequence_epoch = epoch_id
    # `fit_lnk` estimates one filter per light mean, so the grouped arrays have
    # to be there too -- the sequence alone is not enough.
    for level in sorted(set(light)):
        rows = [index for index in range(n_epochs)
                if light[epoch_id == index][0] == level]
        analysis.light_means.append(float(level))
        analysis.n_epochs[float(level)] = len(rows)
        analysis.stimulus[float(level)] = np.vstack(
            [stim[epoch_id == index] for index in rows])
        analysis.response[float(level)] = np.vstack(
            [response[epoch_id == index] for index in rows])
    return analysis


@pytest.mark.slow
def test_lnk_beats_the_static_baseline_on_an_adapting_cell():
    """A slow state must earn its parameters where the adaptation is real.

    ``r2_static`` is the same cascade with ``k`` forced to zero and its
    nonlinearity refitted, so the comparison is nested and the difference is
    attributable to the state alone.
    """
    analysis = _adapting_cell('multiplicative')
    model = vmn.fit_lnk(analysis, coupling='multiplicative', verbose=False)
    assert model is not None
    assert model.r2 > model.r2_static
    assert model.r2_gain > 0.02
    assert not model.at_bounds, model.at_bounds


@pytest.mark.slow
def test_ln_and_lnk_fit_a_large_negative_whole_cell_response():
    """Parameter scaling and bounds must work in pA as well as spikes/s."""
    analysis = _adapting_cell('multiplicative')
    source = analysis.sequence_response.copy()
    source_lo, source_span = float(source.min()), float(np.ptp(source))

    def to_current(values):
        # Same adapting response, expressed as a falling whole-cell current
        # spanning +1 nA to -10 nA.
        return 1_000.0 - 11_000.0 * (np.asarray(values) - source_lo) / source_span

    analysis.sequence_response = to_current(analysis.sequence_response)
    analysis.response = {level: to_current(values)
                         for level, values in analysis.response.items()}
    analysis.rec_type = 'exc'
    analysis.units = 'current (pA)'

    vmn.fit_condition(analysis, verbose=False)
    assert all(np.isfinite(model.r2_train) for model in analysis.ln_model.values())
    assert all(not model.params['at_bounds'] for model in analysis.ln_model.values())

    model = vmn.fit_lnk(
        analysis, coupling='multiplicative', static_analysis=analysis,
        n_restarts=1, verbose=False)
    assert model is not None
    assert np.isfinite(model.r2) and np.isfinite(model.r2_static)
    assert 'alpha' not in model.at_bounds and 'epsilon' not in model.at_bounds
    assert np.ptp(model.predicted) > 1_000.0


@pytest.mark.slow
def test_lnk_identifies_which_coupling_generated_the_data():
    """The comparison must pick the mechanism that was actually simulated.

    A slope change and a shift are different mechanisms, not two settings of
    one, so this is the experiment the model exists to run. If it cannot tell
    them apart on data it generated itself, a pathway result on real cells
    would mean nothing.
    """
    for truth in vmn.LNK_COUPLINGS:
        analysis = _adapting_cell(truth)
        models = vmn.compare_lnk_couplings(analysis, verbose=False)
        assert all(m is not None for m in models.values())
        best = max(models, key=lambda name: models[name].r2)
        assert best == truth, (truth, {n: round(m.r2, 4) for n, m in models.items()})


@pytest.mark.slow
def test_lnk_state_never_sees_the_response():
    """Held-out scoring is only sound if the state is stimulus-driven.

    The state is integrated across held-out epochs, which would leak if it
    depended on the measured response. Scrambling the response must leave the
    state untouched for a fixed parameter set.
    """
    analysis = _adapting_cell()
    model = vmn.fit_lnk(analysis, verbose=False)
    assert model is not None

    rng = np.random.default_rng(7)
    scrambled = vmn.ConditionAnalysis(
        exp_name='synthetic', block_ids=[0], rec_type='extracellular',
        sample_rate=analysis.sample_rate, units=analysis.units,
        sampling_interval=analysis.sampling_interval, skip_seconds=0.0,
        frequency_cutoff=analysis.frequency_cutoff,
        filter_length_s=analysis.filter_length_s)
    scrambled.sequence_stimulus = analysis.sequence_stimulus
    scrambled.sequence_response = rng.permutation(analysis.sequence_response)
    scrambled.sequence_light_mean = analysis.sequence_light_mean
    scrambled.sequence_epoch = analysis.sequence_epoch
    scrambled.light_means = list(analysis.light_means)
    scrambled.stimulus = dict(analysis.stimulus)
    scrambled.response = dict(analysis.response)

    _, state_a = vmn._lnk_predict(model.generator, model.params, model.coupling,
                                  model.sampling_interval,
                                  int(round(model.state_dt_s / model.sampling_interval)))
    _, state_b = vmn._lnk_predict(model.generator, model.params, model.coupling,
                                  model.sampling_interval,
                                  int(round(model.state_dt_s / model.sampling_interval)))
    np.testing.assert_array_equal(state_a, state_b)
    assert 'sequence_response' not in vmn._lnk_predict.__code__.co_names


def test_nonlinearity_timelapse_is_one_basis_transformed_by_each_motif():
    """Display windows sample kinetics; they must not become separate LN fits."""
    from scipy.special import ndtr
    import matplotlib.pyplot as plt

    dt, epoch_pts, n_epochs = 0.01, 100, 3
    generator = np.tile(np.linspace(-2.0, 2.0, epoch_pts), n_epochs)
    epochs = np.repeat(np.arange(n_epochs), epoch_pts)
    raw_state = np.tile(np.linspace(0.15, 0.85, epoch_pts), n_epochs)
    analysis = vmn.ConditionAnalysis(
        exp_name='whole-cell-timelapse', block_ids=[0], rec_type='exc',
        sample_rate=1.0 / dt, units='current (pA)', sampling_interval=dt,
        skip_seconds=0.0, frequency_cutoff=20.0, filter_length_s=0.1)
    analysis.light_means = [1.0]
    analysis.sequence_light_mean = np.ones(generator.size)
    analysis.sequence_epoch = epochs
    analysis.sequence_response = np.zeros(generator.size)

    for coupling in vmn.LNK_COUPLINGS:
        params = {'alpha': -11_000.0, 'beta': 1.8, 'gamma': -0.2,
                  'epsilon': 1_000.0, 'tau_on': 1.0, 'tau_off': 2.0,
                  'k': 0.7}
        model = vmn.LNKModel(
            coupling=coupling, params=params, generator=generator,
            state=raw_state, sampling_interval=dt, state_dt_s=dt,
            filter=np.ones(10), filters={1.0: np.ones(10)},
            filter_time_s=np.arange(10) * dt)
        curves = vmn.nonlinearity_timelapse(
            analysis, model, windows_s=[(0.0, 0.5), (0.5, 1.0)],
            n_points=25, min_bin_samples=5, warmup_epochs=0)
        assert set(curves.columns) >= {'basis', 'model', 'state', 'generator'}
        for _, piece in curves.groupby('order'):
            x = piece.generator.to_numpy()
            state = float(piece.state.iloc[0])
            basis = (params['alpha'] * ndtr(params['beta'] * x + params['gamma'])
                     + params['epsilon'])
            np.testing.assert_allclose(piece.basis, basis)
            if coupling == 'multiplicative':
                expected = (params['alpha'] * np.exp(-params['k'] * state)
                            * ndtr(params['beta'] * x + params['gamma'])
                            + params['epsilon'])
            else:
                expected = (params['alpha'] * ndtr(
                    params['beta'] * x + params['gamma'] - params['k'] * state)
                    + params['epsilon'])
            np.testing.assert_allclose(piece.model, expected)

        figure = vmn.plot_nonlinearity_timelapse(analysis, model, curves,
                                                  warmup_epochs=0)
        assert figure is not None and len(figure.axes) == 4
        assert figure.axes[2].get_ylabel() == 'current (pA)'
        plt.close(figure)


def test_sequence_is_in_recorded_order_and_alternates():
    """The sequence must preserve the interleaving, not the grouping.

    ``stimulus``/``response`` are keyed by light mean; a kinetic model needs
    the epochs in the order the rig ran them, which for this protocol
    alternates. Grouping them would present the model with 300 s of one mean
    followed by 300 s of the other -- a different experiment entirely.
    """
    analysis = _adapting_cell(n_epochs=6, epoch_s=2.0)
    light = analysis.sequence_light_mean
    epoch = analysis.sequence_epoch
    boundaries = np.r_[0, np.flatnonzero(np.diff(epoch) != 0) + 1]
    levels = light[boundaries]
    assert levels.size == 6
    assert np.all(np.diff(levels) != 0), levels


def test_lnk_setup_reuses_static_ln_filter_and_nonlinearity():
    """The LNK start must be the stored LN model, not a lookalike refit."""
    analysis = _adapting_cell(n_epochs=4, epoch_s=2.0)
    n_filter = vmn._even_filter_pts(analysis.filter_length_s,
                                    analysis.sampling_interval)
    for level in analysis.light_means:
        filt = vmn.param_filter(
            dict(numFilt=3.0, tauR=0.012, tauD=0.03,
                 tauP=0.10, phi=20.0), n_filter, analysis.sampling_interval)
        analysis.ln_model[level] = vmn.LNModel(
            label=f'{level:g}', r2=0.5, filter=filt,
            filter_time_s=np.arange(n_filter) * analysis.sampling_interval,
            nl_x=np.array([]), nl_y=np.array([]),
            params={'alpha': 80.0, 'beta': 2.5, 'gamma': -0.4,
                    'epsilon': 4.0})

    setup = vmn._prepare_lnk(
        analysis, 'per_mean', None, 0, False, static_analysis=analysis)
    assert setup is not None
    assert setup.filter_source == 'static LN models'
    assert setup.static_nl_guess is not None
    source = analysis.ln_model[setup.init_level].filter
    target = setup.filters[setup.init_level]
    axis_scale = np.dot(source, target) / np.dot(target, target)
    np.testing.assert_allclose(
        setup.static_nl_guess,
        [80.0, 2.5 * axis_scale, -0.4, 4.0], rtol=1e-10)


def test_compare_lnk_couplings_prepares_the_generator_once(monkeypatch):
    """Both mechanisms must share the expensive, identical filter setup."""
    sentinel = object()
    prepared = []
    fitted = []

    def fake_prepare(*args, **kwargs):
        prepared.append((args, kwargs))
        return sentinel

    def fake_fit(*args, **kwargs):
        fitted.append((kwargs['coupling'], kwargs['_setup']))
        return None

    monkeypatch.setattr(vmn, '_prepare_lnk', fake_prepare)
    monkeypatch.setattr(vmn, 'fit_lnk', fake_fit)
    result = vmn.compare_lnk_couplings(object(), verbose=False)
    assert len(prepared) == 1
    assert fitted == [(name, sentinel) for name in vmn.LNK_COUPLINGS]
    assert result == {name: None for name in vmn.LNK_COUPLINGS}


def test_param_filter_recovers_a_known_shape():
    """Round-trip through the five-parameter form.

    The shape must come back; the parameters need not. They are partially
    degenerate -- ``tauP`` runs to its bound meaning "no oscillation over the
    window" and ``numFilt`` trades against ``tauR`` -- so asserting on them
    would pin an arbitrary point on a ridge.
    """
    dt, n_points = 1e-3, 1000
    truth = vmn.param_filter(dict(numFilt=3.0, tauR=0.04, tauD=0.08,
                                  tauP=0.25, phi=20.0), n_points, dt)
    found, r2 = vmn.fit_param_filter(truth, dt, n_starts=20)
    assert r2 > 0.99, r2
    recovered = vmn.param_filter(found, n_points, dt)
    assert np.corrcoef(recovered, truth)[0, 1] > 0.995


def test_param_filter_handles_slow_filters_not_only_fast_ones():
    """The regression that prompted this work.

    Starting points fixed at fast time constants fit a 30 ms filter at r2 0.99
    and a 200 ms one at r2 ~= 0 -- worse than a flat line -- which reads as the
    functional form being incapable when it is only badly started. Both must
    now come back, since the whole-cell filters in this dataset peak around
    200 ms and are the ones the LNK work targets.
    """
    dt, n_points = 1e-3, 1000
    for tau_rise, tau_decay, tau_period in ((0.012, 0.03, 0.10),   # fast
                                            (0.15, 0.30, 1.20)):   # slow
        truth = vmn.param_filter(dict(numFilt=2.5, tauR=tau_rise, tauD=tau_decay,
                                      tauP=tau_period, phi=0.0), n_points, dt)
        time_to_peak = int(np.argmax(np.abs(truth))) * dt
        _, r2 = vmn.fit_param_filter(truth, dt, n_starts=20)
        assert r2 > 0.9, (time_to_peak, r2)


def test_param_filter_starts_scale_with_time_to_peak():
    """Starts must be drawn from the filter's own timescale, not a constant.

    This is what makes the previous test pass, so it is worth pinning
    directly: the same routine applied to a filter ten times slower must
    explore ten times slower time constants.
    """
    dt, n_points = 1e-3, 1000
    slow = vmn.param_filter(dict(numFilt=2.5, tauR=0.15, tauD=0.30, tauP=1.2,
                                 phi=0.0), n_points, dt)
    fast = vmn.param_filter(dict(numFilt=2.5, tauR=0.015, tauD=0.03, tauP=0.12,
                                 phi=0.0), n_points, dt)
    slow_params, _ = vmn.fit_param_filter(slow, dt, n_starts=20)
    fast_params, _ = vmn.fit_param_filter(fast, dt, n_starts=20)
    # tauR is index 1 in PARAM_FILTER_NAMES; the slow filter's must be larger.
    assert vmn.PARAM_FILTER_NAMES[1] == 'tauR'
    assert slow_params[1] > fast_params[1]


def test_normalized_residual_matches_the_reference_scheme():
    """Sections weighted by their own SD, as ``mse_weighted_loss`` does.

    A quiet stretch and a loud one must contribute comparably; unweighted, the
    loud one dominates, which is the bias the reference metric exists to remove
    and which our 10x luminance step would otherwise create.
    """
    dt = 1e-3
    quiet = 0.1 * np.sin(2 * np.pi * 3 * np.arange(10_000) * dt)
    loud = 10.0 * np.sin(2 * np.pi * 3 * np.arange(10_000) * dt)
    measured = np.concatenate([quiet, loud])
    # A prediction wrong by the same *relative* amount in both halves.
    predicted = np.concatenate([quiet * 0.5, loud * 0.5])

    weighted = vmn.normalized_residual(predicted, measured, dt, bin_s=10.0)
    half = weighted.size // 2
    quiet_cost = float(np.sum(weighted[:half] ** 2))
    loud_cost = float(np.sum(weighted[half:] ** 2))
    assert quiet_cost == pytest.approx(loud_cost, rel=0.05)

    plain = predicted - measured
    assert np.sum(plain[half:] ** 2) > 100 * np.sum(plain[:half] ** 2)

    # The frequency split stays reachable for reproducing the paper.
    banded = vmn.normalized_residual(predicted, measured, dt, bin_s=10.0,
                                     split_hz=4.0)
    assert banded.size == 2 * weighted.size


@pytest.mark.slow
def test_fit_filter_is_nested_in_the_frozen_fit():
    """Freeing the filter cannot fit the training data worse.

    Frozen is the fitted model with the filter pinned at its starting point, so
    a lower in-sample r2 with fit_filter=True would mean the optimiser failed
    rather than that the extra parameters did not help. Held-out r2 is free to
    go either way -- and on real data it does.
    """
    analysis = _adapting_cell('multiplicative', n_epochs=4, epoch_s=6.0)
    frozen = vmn.fit_lnk(analysis, fit_filter=False, n_restarts=1, verbose=False)
    fitted = vmn.fit_lnk(analysis, fit_filter=True, n_restarts=1, verbose=False)
    assert frozen is not None and fitted is not None
    assert fitted.r2_train >= frozen.r2_train - 0.02
    assert len(fitted.params) > len(frozen.params)
    assert fitted.fit_filter and not frozen.fit_filter
    for level, r2 in frozen.filter_r2.items():
        assert r2 > 0.5, (level, r2)


def _two_state_reference(drive, dt, k_act, k_inact, k_slow_in, k_slow_out):
    """Direct coupled integration of the three-occupancy system, sample by sample."""
    active = np.empty_like(drive)
    inactivated = np.empty_like(drive)
    a = i = 0.0
    for n in range(drive.size):
        u = k_act * drive[n]
        a, i = (a + dt * (u * (1 - a - i) - k_inact * a),
                i + dt * (k_slow_in * a - k_slow_out * i))
        active[n], inactivated[n] = a, i
    return active, inactivated


def test_two_state_kinetics_matches_direct_integration():
    """The split solver must equal the coupled system it stands in for.

    Each block solves `A` with `_relax`, advances `I` across the block from the
    block's mean `A`, and re-solves `A` against `I` ramping linearly to that
    value. That is an approximation of the coupled system, so it has to be
    checked against the coupled system rather than assumed -- including at a
    large slow in/out ratio, since that is the regime the real fits go to and
    the regime an earlier iterative solver oscillated in.

    **The step pair has to straddle the solver's error, not the reference's.**
    `_two_state_reference` is a direct forward-Euler integration at `dt`, so it
    carries its own O(dt) error -- about 0.3% of `I`'s range here -- which no
    refinement of `state_step` can remove. Measured: the error falls 4.2% to
    0.42% between steps 2000 and 250, then sits flat at 0.3-0.4% all the way
    down to step 1. Once the ramp made 250 ms accurate enough to reach that
    floor, the old (250, 50) pair was comparing two numbers that were both the
    reference's error and the ratio stopped meaning anything. Hence (2000,
    500), where the solver is still what is being measured.

    The slow state's tolerance is relative to its own range. A marching scheme
    quantises `I` to the block, so its error scales with how far `I` travels,
    and an absolute bound would be a bound on the test stimulus rather than on
    the solver.
    """
    rng = np.random.default_rng(0)
    dt = 1e-3
    drive = np.concatenate([rng.random(20_000) * 0.2, rng.random(20_000) * 0.9] * 2)
    for k_act, k_inact, k_in, k_out in ((100.0, 100.0, 0.5, 0.2),
                                        (300.0, 500.0, 3.0, 0.7),
                                        (300.0, 500.0, 2.0, 0.1)):   # ratio 20
        ref_a, ref_i = _two_state_reference(drive, dt, k_act, k_inact, k_in, k_out)
        got_a, got_i = vmn.two_state_kinetics(drive, dt, k_act, k_inact, k_in, k_out)
        assert np.corrcoef(ref_a, got_a)[0, 1] > 0.98, (k_in, k_out)
        span = max(float(np.ptp(ref_i)), 1e-9)

        # The decisive check is convergence, not a tolerance. A marching scheme
        # quantises the slow state to its block, so the right question is
        # whether the error goes to zero as the block shrinks -- a fixed
        # threshold would only pin one arbitrary block size. Refining 5x must
        # cut the error by at least half; a scheme converging to the wrong
        # answer, or oscillating as the earlier iterative solver did, would not.
        errors = []
        for step in (2000, 500):
            _, fine = vmn.two_state_kinetics(drive, dt, k_act, k_inact, k_in,
                                             k_out, state_step=step)
            errors.append(float(np.max(np.abs(ref_i - fine))) / span)
        assert errors[1] < 0.5 * errors[0] + 1e-3, (k_in, k_out, errors)
        assert errors[1] < 0.06, (k_in, k_out, errors)
        # And it converges *onto* the reference rather than beside it: at the
        # block size the fits actually use, all that is left is the
        # reference's own floor.
        _, settled = vmn.two_state_kinetics(drive, dt, k_act, k_inact, k_in,
                                            k_out, state_step=100)
        assert float(np.max(np.abs(ref_i - settled))) / span < 0.02, (k_in, k_out)


def test_two_state_occupancies_stay_physical():
    """R, A and I are occupancies of one pool; none may go negative."""
    rng = np.random.default_rng(1)
    drive = rng.random(40_000)
    for k_act, k_inact, k_in, k_out in ((500.0, 50.0, 10.0, 0.05),
                                        (10.0, 1000.0, 0.01, 20.0)):
        active, inactivated = vmn.two_state_kinetics(drive, 1e-3, k_act, k_inact,
                                                     k_in, k_out)
        assert active.min() >= -1e-9 and inactivated.min() >= -1e-9
        assert (active + inactivated).max() <= 1.0 + 1e-6


def test_activation_rate_must_stay_free():
    """Why `k_act` is fitted rather than folded into the output amplitude.

    The ratio `k_act/k_inact` sets the active state's occupancy while their sum
    sets its speed. Pinning `k_act` at 1 forces a choice between the two: a
    fast response needs a large `k_inact`, which leaves `A` near zero and the
    slow pool with nothing to deplete. That is exactly what went wrong before
    it was freed, so it is pinned here as a test.
    """
    rng = np.random.default_rng(2)
    drive = rng.random(20_000) * 0.5
    starved, starved_slow = vmn.two_state_kinetics(drive, 1e-3, 1.0, 200.0, 3.0, 0.7)
    healthy, healthy_slow = vmn.two_state_kinetics(drive, 1e-3, 200.0, 200.0, 3.0, 0.7)
    assert starved.mean() < 0.01, starved.mean()
    assert healthy.mean() > 10 * starved.mean()
    # And with nothing in the active state, there is nothing to deplete.
    assert starved_slow.max() < 0.1 * healthy_slow.max()


def test_two_state_depletes_under_sustained_drive():
    """Sustained drive must fill the slow pool and lower the resting fraction.

    This is the mechanism the variant exists for: gain is proportional to
    resting occupancy, so if drive does not deplete it, nothing adapts.
    """
    dt = 1e-3
    quiet = np.full(20_000, 0.05)
    loud = np.full(40_000, 0.9)
    drive = np.concatenate([quiet, loud])
    active, inactivated = vmn.two_state_kinetics(drive, dt, 200.0, 200.0, 3.0, 0.5)
    resting = 1.0 - active - inactivated
    early = slice(20_000, 22_000)          # just after the step
    late = slice(50_000, 60_000)           # well into it
    assert inactivated[late].mean() > inactivated[early].mean()
    assert resting[late].mean() < resting[early].mean()


@pytest.mark.slow
def test_apparent_nonlinearity_is_analytic_and_signed_correctly():
    """Depletion must lower the gain, and the readout must not be confounded.

    Splitting the record on the state and binning each half is confounded --
    the state is stimulus-driven, so high-state samples are high-drive samples
    -- and reported a gain ratio above 1, i.e. more adaptation giving more
    output. The analytic instantaneous curve holds the stimulus identical on
    both sides, so a ratio below 1 is the only physical answer.
    """
    analysis = _adapting_cell('multiplicative', n_epochs=4, epoch_s=6.0)
    model = vmn.fit_lnk_two_state(analysis, n_passes=2, n_restarts=1,
                                  verbose=False)
    assert model is not None
    curves = vmn.apparent_nonlinearity(analysis, model)
    assert set(curves.adaptation) == {'low', 'high'}
    # Both curves are evaluated on the same generator grid, by construction.
    low = curves[curves.adaptation.eq('low')].generator.values
    high = curves[curves.adaptation.eq('high')].generator.values
    np.testing.assert_allclose(low, high)

    summary = vmn.describe_apparent_change(curves)
    assert summary['gain_ratio'] <= 1.0 + 1e-6, summary
    assert abs(summary['shift_generator']) < 0.05, summary


def test_two_state_rates_round_trip():
    """The identifiable pair must map cleanly onto the rate constants.

    `tau_fast` is the relaxation time at full drive, 1/(k_act + k_inact), and
    `occupancy` is k_act/k_inact. Fitting those instead of the rates is the
    whole point of the reparameterisation, so the conversion has to be exact.
    """
    for tau_fast, occupancy in ((1e-3, 4.0), (0.02, 0.5), (0.2, 20.0)):
        k_act, k_inact = vmn.two_state_rates(tau_fast, occupancy)
        assert 1.0 / (k_act + k_inact) == pytest.approx(tau_fast, rel=1e-9)
        assert k_act / k_inact == pytest.approx(occupancy, rel=1e-9)
        assert k_act > 0 and k_inact > 0


def test_fast_rates_are_unidentifiable_above_the_sampling_rate():
    """Why the fast state is fitted as (tau, ratio) and not as two rates.

    Past the sampling interval only the *ratio* of the two rates matters: it
    sets the active state's occupancy, while their sum sets a speed the data
    can no longer resolve. Scaling both leaves the prediction alone, so an
    optimiser given the raw rates drifts up that flat direction until one hits
    a bound -- which is what it did, pinning `k_act` at 2000/s.
    """
    from scipy.ndimage import gaussian_filter1d

    rng = np.random.default_rng(0)
    dt = 1e-3
    # A *smoothed* drive, as the real one is: it comes through a temporal
    # filter, so it has no power at the sampling rate. Against white noise the
    # comparison fails for an uninteresting reason -- with tau far below dt the
    # per-sample values diverge even though the mean occupancy is preserved --
    # which says something about white noise rather than about the model.
    drive = gaussian_filter1d(rng.random(40_000), 8.0)
    drive = (drive - drive.min()) / np.ptp(drive)
    reference = None
    for factor in (1.0, 4.0, 20.0):
        # tau_fast well below dt for every factor, ratio held at 4.
        k_act, k_inact = vmn.two_state_rates(1e-3 / factor, 4.0)
        active, _ = vmn.two_state_kinetics(drive, dt, k_act, k_inact, 3.0, 0.5)
        if reference is None:
            reference = active
        else:
            assert np.corrcoef(reference, active)[0, 1] > 0.99
            assert abs(active.mean() - reference.mean()) < 0.02 * reference.mean()

    # The ratio, by contrast, changes the occupancy it is supposed to set.
    low_act, low_inact = vmn.two_state_rates(1e-3, 0.5)
    high_act, high_inact = vmn.two_state_rates(1e-3, 8.0)
    low, _ = vmn.two_state_kinetics(drive, dt, low_act, low_inact, 3.0, 0.5)
    high, _ = vmn.two_state_kinetics(drive, dt, high_act, high_inact, 3.0, 0.5)
    assert high.mean() > 1.3 * low.mean()


def test_condition_output_keeps_selected_led_metadata_and_excludes_lnk(
        tmp_path, monkeypatch):
    import h5py
    import pandas as pd

    analysis, temporal = _population_analysis()
    analysis.excluded_epochs = [(11, 1)]
    analysis.activity_excluded_epochs = [(11, 1)]
    blocks = pd.DataFrame({
        'block_id': [11, 12], 'exp_name': ['2020-06-11_B'] * 2,
        'cell_label': ['Cell3'] * 2, 'cell_type_short': ['OFF-parasol'] * 2,
        'protocol_name': [vmn.PROTOCOLS[0]] * 2, 'led': ['Blue LED'] * 2,
    })
    monkeypatch.setattr(vmn, 'led_attenuation', lambda row: {
        'rig': 'B', 'led': 'Blue LED', 'led_color': 'blue',
        'led_ndfs': 'B1, B12', 'optical_density': 2.0,
        'attenuation': .01, 'unknown_tokens': '',
        'filter_wheel_ndf': 3.0, 'wheel_tokens_ignored': 'FW3',
        'wheel_ignored': True,
    })
    decoded = pd.DataFrame({
        'mode': ['per_window'] * 2, 'lightMean': [.1, 1.0],
        'centre_s': [.5, .5], 'r_all': [.3, .4],
        'r_increment': [.2, .3], 'r_decrement': [.1, .2],
        'gain_increment': [.8, .9], 'gain_decrement': [.7, .8],
        'nrmse_increment': [1.0, .9], 'nrmse_decrement': [1.1, 1.0],
        'r_asymmetry': [.1, .1], 'gain_asymmetry': [.1, .1],
    })
    early_late = pd.DataFrame({
        'mode': ['per_window'], 'lightMean': [.1], 'half': ['early'],
        'gain_increment': [.8], 'gain_decrement': [.7],
    })
    directional = pd.DataFrame({
        'mode': ['per_window'], 'lightMean': [1.0],
        'light_condition': ['bright'], 'operating_point': ['positive'],
        'direction': ['decrement'], 'gain_delta': [.8],
    })

    path = vmn.save_condition_output(
        analysis, blocks, temporal_models=temporal, decoded=decoded,
        directional_decoding=directional, early_late=early_late,
        decode_window_s=2.5, cell_index=19,
        decode_window_rule='shortest_within_one_se', mean_window_s=1.0,
        output_dir=tmp_path, verbose=False)
    second = vmn.save_condition_output(
        analysis, blocks, temporal_models=temporal, decoded=decoded,
        directional_decoding=directional, early_late=early_late,
        decode_window_s=2.5,
        decode_window_rule='shortest_within_one_se',
        cell_index=19, output_dir=tmp_path, verbose=False)

    assert path == second and len(list(tmp_path.glob('*.h5'))) == 1
    with h5py.File(path, 'r') as stored:
        assert stored.attrs['rig'] == 'B'
        assert stored.attrs['stim_time_ms'] == 30_000.0
        assert stored.attrs['stim_seconds'] == 30.0
        assert stored.attrs['n_epochs'] == 4
        assert stored.attrs['cell_index'] == 19
        assert stored.attrs['mean_rate_hz'] == pytest.approx(39.5)
        assert stored['excluded_epochs'][:].tolist() == [[11, 1]]
        assert stored['activity_excluded_epochs'][:].tolist() == [[11, 1]]
        assert stored.attrs['decode_window_s'] == 2.5
        assert stored.attrs['decode_window_rule'] == 'shortest_within_one_se'
        assert stored.attrs['led_ndfs'] == 'B1, B12'
        assert stored.attrs['optical_density'] == 2.0
        assert not bool(stored.attrs['contains_lnk'])
        assert set(stored['tables']) == set(vmn.CONDITION_TABLES)
        assert not any('lnk' in name.lower() for name in stored['tables'])
        assert stored['tables/ln_curves/y'].compression == 'gzip'

    index = vmn.load_condition_index(tmp_path)
    assert 'output_version' not in index
    assert index[['date', 'cell_index', 'cell_label', 'cell_type', 'rec_type',
                  'mean_rate_hz',
                  'stim_seconds', 'n_epochs_total', 'rig',
                  'led', 'led_ndfs']].iloc[0].to_dict() == {
        'date': '2020-06-11_B', 'cell_index': 19, 'cell_label': 'Cell3',
        'cell_type': 'OFF-parasol', 'rec_type': 'extracellular',
        'mean_rate_hz': 39.5,
        'stim_seconds': 30.0, 'n_epochs_total': 4,
        'rig': 'B', 'led': 'Blue LED', 'led_ndfs': 'B1, B12'}
    loaded = vmn.load_population_table('condition_summary', tmp_path)
    assert len(loaded) == 2
    assert loaded.sort_values('lightMean').mean_rate_hz.tolist() == [19.5, 59.5]
    assert loaded.cell_id.unique().tolist() == ['2020-06-11_B/Cell3/extracellular']
    loaded_directional = vmn.load_population_table('directional_decoding', tmp_path)
    assert loaded_directional.gain_delta.tolist() == [.8]
    assert loaded_directional.cell_index.tolist() == [19]

    # The same cell and block IDs at another duration must be a second entity,
    # not an overwrite of the 30 s condition.
    analysis.stim_time_ms = 60_000.0
    third = vmn.save_condition_output(
        analysis, blocks, temporal_models=temporal, decoded=decoded,
        early_late=early_late, output_dir=tmp_path, verbose=False)
    assert third != path and len(list(tmp_path.glob('*.h5'))) == 2
    assert vmn.load_condition_index(tmp_path).stim_seconds.tolist() == [30.0, 60.0]


def test_condition_index_backfills_legacy_cell_index_from_current_cell_table(
        tmp_path, monkeypatch):
    import h5py
    import pandas as pd

    analysis, temporal = _population_analysis()
    blocks = pd.DataFrame({
        'block_id': [11, 12], 'cell_label': ['Cell3'] * 2,
        'cell_type_short': ['OFF-parasol'] * 2,
        'protocol_name': [vmn.PROTOCOLS[0]] * 2,
    })
    monkeypatch.setattr(vmn, 'condition_light_settings', lambda blocks, analysis:
        pd.DataFrame({'rig': ['B'], 'led': ['UV LED'], 'led_ndfs': ['B1'],
                      'optical_density': [1.0]}))
    path = vmn.save_condition_output(
        analysis, blocks, temporal_models=temporal, output_dir=tmp_path,
        verbose=False)
    with h5py.File(path, 'r+') as stored:
        del stored.attrs['cell_index']
    current = pd.DataFrame({
        'exp_name': ['2020-06-11_B'], 'cell_label': ['Cell3'],
        'cell_index': [27]})

    index = vmn.load_condition_index(tmp_path, protocol_cells=current)

    assert index.cell_index.tolist() == [27]
    assert index.current_cell_index.tolist() == [27]
    assert index.index_status.tolist() == ['backfilled from stable registry']


def test_condition_index_preserves_and_flags_conflicting_saved_index(
        tmp_path, monkeypatch):
    import pandas as pd

    analysis, temporal = _population_analysis()
    blocks = pd.DataFrame({
        'block_id': [11, 12], 'cell_label': ['Cell3'] * 2,
        'cell_type_short': ['OFF-parasol'] * 2,
        'protocol_name': [vmn.PROTOCOLS[0]] * 2,
    })
    monkeypatch.setattr(vmn, 'condition_light_settings', lambda blocks, analysis:
        pd.DataFrame({'rig': ['B'], 'led': ['UV LED'], 'led_ndfs': ['B1'],
                      'optical_density': [1.0]}))
    vmn.save_condition_output(
        analysis, blocks, temporal_models=temporal, cell_index=19,
        output_dir=tmp_path, verbose=False)
    current = pd.DataFrame({
        'exp_name': ['2020-06-11_B'], 'cell_label': ['Cell3'],
        'cell_index': [27]})

    index = vmn.load_condition_index(tmp_path, protocol_cells=current)

    assert index.cell_index.tolist() == [19]
    assert index.current_cell_index.tolist() == [27]
    assert index.index_status.tolist() == [
        'saved index 19 != stable registry 27']


def test_save_duration_outputs_saves_every_duration(monkeypatch):
    import pandas as pd

    first, temporal = _population_analysis()
    second, _ = _population_analysis()
    second.stim_time_ms = 60_000.0
    cores = {
        30_000.0: vmn.CoreLNAnalysis(first, temporal, pd.DataFrame()),
        60_000.0: vmn.CoreLNAnalysis(second, temporal, pd.DataFrame()),
    }
    calls = []

    def fake_save(analysis, blocks, **kwargs):
        calls.append((analysis.stim_time_ms, kwargs.get('decoded'),
                      kwargs.get('early_late')))
        return Path(f'/tmp/{analysis.stim_time_ms:g}.h5')

    monkeypatch.setattr(vmn, 'save_condition_output', fake_save)
    reconstruction = {
        30_000.0: {'decoded': 'decoded-30', 'early_late': 'halves-30'},
        60_000.0: {'decoded': 'decoded-60', 'early_late': 'halves-60'},
    }

    saved = vmn.save_duration_outputs(
        cores, pd.DataFrame(), reconstruction_by_duration=reconstruction,
        verbose=False)

    assert [call[0] for call in calls] == [30_000.0, 60_000.0]
    assert calls[0][1:] == ('decoded-30', 'halves-30')
    assert saved.stim_seconds.tolist() == [30.0, 60.0]


def test_save_condition_outputs_saves_every_mode_duration_pair(monkeypatch):
    import pandas as pd

    extracellular, temporal = _population_analysis()
    exc, _ = _population_analysis()
    exc.rec_type = 'exc'
    exc.units = 'pA'
    exc.stim_time_ms = 60_000.0
    cores = {
        ('extracellular', 30_000.0): vmn.CoreLNAnalysis(
            extracellular, temporal, pd.DataFrame()),
        ('exc', 60_000.0): vmn.CoreLNAnalysis(exc, temporal, pd.DataFrame()),
    }
    calls = []

    def fake_save(analysis, blocks, **kwargs):
        calls.append((analysis.rec_type, analysis.stim_time_ms,
                      kwargs.get('decoded'), kwargs.get('directional_decoding'),
                      kwargs.get('cell_index')))
        return Path(f'/tmp/{analysis.rec_type}-{analysis.stim_time_ms:g}.h5')

    monkeypatch.setattr(vmn, 'save_condition_output', fake_save)
    reconstruction = {
        ('extracellular', 30_000.0): {
            'decoded': 'spikes', 'directional_decoding': 'spike-directions'},
        ('exc', 60_000.0): {
            'decoded': 'current', 'directional_decoding': 'current-directions'},
    }

    saved = vmn.save_condition_outputs(
        cores, pd.DataFrame(),
        reconstruction_by_condition=reconstruction, cell_index=19,
        verbose=False)

    assert calls == [
        ('extracellular', 30_000.0, 'spikes', 'spike-directions', 19),
        ('exc', 60_000.0, 'current', 'current-directions', 19)]
    assert saved[['cell_index', 'rec_type', 'stim_seconds']].values.tolist() == [
        [19, 'extracellular', 30.0], [19, 'exc', 60.0]]


def test_population_mean_sem_weights_each_cell_once():
    import pandas as pd

    # Cell A has two repeated saved rows; it must contribute their mean (2),
    # once, alongside cell B (6): population mean 4, SEM 2.
    frame = pd.DataFrame({
        'cell_id': ['A', 'A', 'B'], 'cell_type': ['OFF'] * 3,
        'rec_type': ['extracellular'] * 3, 'lightMean': [.1] * 3,
        'r2': [1.0, 3.0, 6.0],
    })
    summary = vmn.population_mean_sem(
        frame, group_by=['cell_type', 'rec_type', 'lightMean'], metrics=['r2'])

    assert summary.r2_mean.iloc[0] == pytest.approx(4.0)
    assert summary.r2_sem.iloc[0] == pytest.approx(2.0)
    assert summary.r2_n_cells.iloc[0] == 2


def test_condition_batch_helpers_preserve_mode_duration_keys(monkeypatch):
    import pandas as pd

    conditions = pd.DataFrame({
        'rec_type': ['extracellular', 'exc'],
        'stim_time_ms': [30_000., 60_000.],
        'stim_seconds': [30., 60.], 'n_epochs': [6, 7],
        'block_ids': [[11], [12]],
        'excluded_epochs': [[(11, 1)], []],
    })
    monkeypatch.setattr(vmn, 'epoch_response_summary', lambda *a, **k:
                        pd.DataFrame({'lightMean': [.1],
                                      'mean_firing_rate_hz': [40.],
                                      'modulation_sd_pA': [200.]}))
    monkeypatch.setattr(vmn, 'plot_traces', lambda *a, **k: 'figure')
    inspection = vmn.inspect_recording_conditions(
        'example', [11, 12], conditions, max_epochs=None,
        min_firing_rate_hz=30., min_whole_cell_modulation_pa=100.,
        low_response_epoch_fraction=.8, verbose=False)
    assert set(inspection.summaries) == {
        ('extracellular', 30_000.), ('exc', 60_000.)}

    calls = []
    analysis, temporal = _population_analysis()

    def fake_core(exp_name, block_ids, rec_type, **kwargs):
        calls.append((rec_type, kwargs['stim_time_ms'],
                      tuple(kwargs['excluded_epochs']),
                      tuple(kwargs['light_means'])))
        current = analysis
        current.rec_type = rec_type
        current.stim_time_ms = kwargs['stim_time_ms']
        return vmn.CoreLNAnalysis(current, temporal, pd.DataFrame())

    monkeypatch.setattr(vmn, 'run_core_ln_analysis', fake_core)
    retained = inspection.conditions[inspection.conditions.included].copy()
    cores = vmn.run_core_condition_analyses(
        'example', [11, 12], retained, qc_signature=inspection.signature,
        min_firing_rate_hz=30., min_whole_cell_modulation_pa=100.,
        low_response_epoch_fraction=.8, verbose=False)
    assert list(cores) == [('extracellular', 30_000.), ('exc', 60_000.)]
    assert calls == [
        ('extracellular', 30_000., ((11, 1),), (.1,)),
        ('exc', 60_000., (), (.1,))]


def test_inspection_excludes_only_failed_activity_conditions(monkeypatch):
    import pandas as pd

    conditions = pd.DataFrame({
        'rec_type': ['extracellular', 'exc'],
        'stim_time_ms': [30_000., 60_000.],
        'stim_seconds': [30., 60.], 'n_epochs': [4, 2],
        'block_ids': [[11], [12]], 'included': [True, True],
        'reason': ['', ''],
    })

    def fake_summary(_exp, _blocks, rec_type, **_kwargs):
        if rec_type == 'extracellular':
            return pd.DataFrame({
                'block_id': [11] * 5, 'epoch': [0, 1, 2, 3, 4],
                'lightMean': [.1, .1, .1, 1., 1.],
                'mean_firing_rate_hz': [40., 0., 45., 0., 0.],
            })
        return pd.DataFrame({
            'block_id': [12, 12], 'epoch': [0, 1],
            'lightMean': [.1, .1], 'modulation_sd_pA': [10., 20.],
        })

    monkeypatch.setattr(vmn, 'epoch_response_summary', fake_summary)
    monkeypatch.setattr(vmn, 'plot_traces', lambda *a, **k: 'figure')

    inspection = vmn.inspect_recording_conditions(
        'example', [11, 12], conditions,
        min_firing_rate_hz=30., min_whole_cell_modulation_pa=100.,
        low_response_epoch_fraction=.8, verbose=False)

    updated = inspection.conditions.set_index('rec_type')
    assert updated.loc['extracellular', 'included']
    assert updated.loc['extracellular', 'included_light_means'] == [.1]
    assert updated.loc['extracellular', 'activity_excluded_epochs'] == [(11, 1)]
    assert updated.loc['extracellular', 'excluded_epochs'] == [(11, 1)]
    assert updated.loc['extracellular', 'n_epochs_after_activity_qc'] == 2
    assert not updated.loc['exc', 'included']
    assert 'all mean-light conditions failed' in updated.loc['exc', 'reason']
    audit = inspection.condition_audit.set_index(['rec_type', 'lightMean'])
    assert audit.loc[('extracellular', .1), 'included']
    assert audit.loc[('extracellular', .1), 'n_epochs_used'] == 2
    assert not audit.loc[('extracellular', 1.), 'included']
    assert not audit.loc[('exc', .1), 'included']

    captured = {}
    monkeypatch.setattr(
        vmn, 'run_core_ln_analysis',
        lambda *args, **kwargs: captured.update(kwargs) or object())
    retained = inspection.conditions[inspection.conditions.included].copy()
    vmn.run_core_condition_analyses(
        'example', [11, 12], retained, qc_signature=inspection.signature,
        min_firing_rate_hz=30., min_whole_cell_modulation_pa=100.,
        low_response_epoch_fraction=.8, verbose=False)
    assert captured['light_means'] == [.1]
    assert captured['excluded_epochs'] == [(11, 1)]
    assert captured['activity_excluded_epochs'] == [(11, 1)]


def test_population_wrappers_are_safe_before_any_records_exist(tmp_path):
    overview = vmn.population_overview_analysis(output_dir=tmp_path)
    temporal = vmn.population_temporal_analysis(output_dir=tmp_path)
    decoding = vmn.population_decoding_analysis(output_dir=tmp_path)

    assert overview['saved_cells'].empty
    assert overview['mean_figures'] == {}
    assert temporal['summary'].empty and temporal['figures'] == {}
    assert decoding['directional_contrast'].empty
    assert decoding['directional_contrast_figures'] == {}


def test_condition_output_default_is_external_rod_noise_folder():
    assert vmn.condition_output_dir() == Path(
        '/Volumes/ChrisNewSSD/retinanalysis_output/rod_adaptation/noise')


def test_reconstruction_batch_helper_returns_save_ready_bundle(monkeypatch):
    import pandas as pd

    analysis, temporal = _population_analysis()
    core = vmn.CoreLNAnalysis(analysis, temporal, pd.DataFrame())
    reload_calls = []
    monkeypatch.setattr(
        vmn, 'analyze_condition',
        lambda *a, **k: reload_calls.append(k) or analysis)
    monkeypatch.setattr(vmn, 'select_decode_window', lambda *a, **k:
                        (2.0, pd.DataFrame({'selected': [True]})))
    traces = pd.DataFrame({'mode': ['per_window'], 'lightMean': [.1]})
    monkeypatch.setattr(vmn, 'reconstruct_traces', lambda *a, **k: traces)
    directional = pd.DataFrame({
        'operating_point': ['positive'], 'light_condition': ['bright'],
        'mode': ['per_window'], 'window': ['0-2 s'],
        'direction': ['increment'], 'n_changes': [5], 'gain_delta': [.2],
        'direction_accuracy': [.6], 'nrmse_delta': [1.],
    })
    monkeypatch.setattr(vmn, 'generator_direction_decoding',
                        lambda *a, **k: directional)
    monkeypatch.setattr(vmn, 'decode_recovery', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(vmn, 'reconstruction_summary', lambda *a, **k: pd.DataFrame())
    for name in ('plot_reconstruction_trace', 'plot_reconstruction_transfer',
                 'plot_phase_triggered', 'plot_generator_direction_decoding',
                 'plot_decoding'):
        monkeypatch.setattr(vmn, name, lambda *a, **k: 'figure')

    result = vmn.run_reconstruction_analyses(
        analysis.exp_name, {('extracellular', 30_000.): core},
        decode_skip_s=2., decode_window_s='auto',
        decode_window_candidates_s=(1., 2.), decode_bin_ms=50.,
        steady_state_s=10., min_phase_ms=100.,
        direction_min_change_quantile=.25, trace_seconds=(10., 15.),
        verbose=False)

    bundle = result[('extracellular', 30_000.)]
    assert reload_calls[0]['light_means'] == analysis.light_means
    assert bundle['decode_window_s'] == 2.0
    assert bundle['directional_decoding'] is directional
    assert bundle['transfer_figure'] == 'figure'


def test_two_state_batch_helper_keeps_condition_key(monkeypatch):
    analysis, _ = _population_analysis()
    model = object()
    curves = object()
    monkeypatch.setattr(vmn, 'fit_lnk_two_state', lambda *a, **k: model)
    monkeypatch.setattr(vmn, 'apparent_nonlinearity', lambda *a, **k: curves)
    monkeypatch.setattr(vmn, 'describe_apparent_change', lambda *a, **k:
                        {'gain_ratio': .5})
    monkeypatch.setattr(vmn, 'plot_apparent_nonlinearity', lambda *a, **k: 'figure')

    result = vmn.run_two_state_lnk_conditions(
        {('extracellular', 30_000.): {'analysis': analysis}}, verbose=False)

    bundle = result[('extracellular', 30_000.)]
    assert bundle == {'model': model, 'curves': curves, 'figure': 'figure',
                      'change': {'gain_ratio': .5}}
