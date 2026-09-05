"""Mixed recording modes within a single epoch group."""
import numpy as np
import pandas as pd

from retinanalysis.SCutils import recording_mode as rm


def test_single_cell_parser_uses_block_invariant_and_chronological_transition():
    # Deliberately scramble both blocks and epochs. The parser must sort by
    # acquisition time, decide once per block, and broadcast that decision.
    records = pd.DataFrame({
        'exp_name': ['test'] * 6,
        'cell_label': ['cell1'] * 6,
        'block_id': [13, 10, 12, 11, 10, 14],
        'start_time': pd.to_datetime([
            '2026-01-01 10:30', '2026-01-01 10:00',
            '2026-01-01 10:20', '2026-01-01 10:10',
            '2026-01-01 10:01', '2026-01-01 10:40']),
        'recording_technique': [
            '', 'cell-attached', 'whole-cell', '', 'cell-attached',
            # This stale label is impossible after the whole-cell transition.
            'cell-attached'],
        'onlineAnalysis': ['', '', '', '', '', ''],
        'series_resistance': [0., 0., 8e6, 0., 0., 0.],
        'mean_current': [20., np.nan, -30., np.nan, np.nan, 15.],
    })

    parsed = rm.parse_single_cell_recording_modes(records)

    by_block = parsed.drop_duplicates('block_id').set_index('block_id')
    assert by_block.recording_order.to_dict() == {
        10: 0, 11: 1, 12: 2, 13: 3, 14: 4}
    assert by_block.recording_family.to_dict() == {
        10: 'cell-attached', 11: 'cell-attached', 12: 'whole-cell',
        13: 'whole-cell', 14: 'whole-cell'}
    assert by_block.rec_type.to_dict() == {
        10: 'extracellular', 11: 'extracellular', 12: 'exc',
        13: 'inh', 14: 'inh'}
    assert parsed.loc[parsed.block_id.eq(10), 'rec_type'].nunique() == 1
    assert 'later block cannot be cell-attached' in by_block.loc[14, 'rec_note']


def test_single_cell_parser_does_not_treat_zero_rs_or_no_spikes_as_a_mode():
    records = pd.DataFrame({
        'exp_name': ['test'], 'cell_label': ['cell1'], 'block_id': [1],
        'start_time': pd.to_datetime(['2026-01-01']),
        'recording_technique': [''], 'onlineAnalysis': [''],
        'series_resistance': [0.], 'mean_current': [-100.],
        'contains_spikes': [False],
    })

    parsed = rm.parse_single_cell_recording_modes(
        records, spike_evidence_column='contains_spikes')

    assert parsed.loc[0, 'recording_family'] == ''
    assert parsed.loc[0, 'rec_type'] == ''
    assert 'unresolved' in parsed.loc[0, 'rec_note']


def test_single_cell_parser_accepts_only_positive_spike_evidence():
    records = pd.DataFrame({
        'exp_name': ['test'], 'cell_label': ['cell1'], 'block_id': [1],
        'contains_spikes': [True],
    })

    parsed = rm.parse_single_cell_recording_modes(
        records, spike_evidence_column='contains_spikes')

    assert parsed.loc[0, 'recording_family'] == 'cell-attached'
    assert parsed.loc[0, 'rec_type'] == 'extracellular'


def test_high_confidence_classifier_can_correct_stale_group_metadata():
    records = pd.DataFrame({
        'exp_name': ['test'] * 4, 'cell_label': ['cell1'] * 4,
        'block_id': [1, 2, 3, 4],
        'start_time': pd.date_range('2026-01-01', periods=4, freq='min'),
        'recording_technique': ['cell-attached'] * 4,
        'classifier_family': ['cell-attached', 'whole-cell', 'whole-cell',
                              'whole-cell'],
        'mean_current': [-50., -800., -1200., -1500.],
    })

    parsed = rm.parse_single_cell_recording_modes(
        records, classifier_family_column='classifier_family')

    assert parsed.rec_type.tolist() == [
        'extracellular', 'exc', 'exc', 'exc']
    assert parsed.recording_family_source.tolist()[1:] == [
        'high-confidence block classifier'] * 3
    assert 'classifier corrects' in parsed.loc[1, 'rec_note']


def test_positive_series_resistance_overrides_cell_attached_metadata():
    records = pd.DataFrame({
        'exp_name': ['test'], 'cell_label': ['cell1'], 'block_id': [1],
        'recording_technique': ['cell-attached'],
        'onlineAnalysis': ['extracellular'],
        'series_resistance': [7e6], 'mean_current': [-25.],
    })

    parsed = rm.parse_single_cell_recording_modes(records)

    assert parsed.loc[0, 'recording_family'] == 'whole-cell'
    assert parsed.loc[0, 'rec_type'] == 'exc'
    assert parsed.loc[0, 'recording_family_source'] == 'positive series resistance'


def test_unlabelled_blocks_in_one_group_use_their_own_trace(monkeypatch):
    blocks = pd.DataFrame({
        'exp_name': ['synthetic_G'] * 3, 'block_id': [1, 2, 3],
        'group_id': [10] * 3, 'onlineAnalysis': [None] * 3,
        'recording_technique': ['cell-attached'] * 3,
        'epoch_series_resistance': [0., 0., 0.],
    })
    monkeypatch.setattr(rm, '_amp_response_table', lambda *a, **k:
                        pd.DataFrame(columns=['block_id', 'h5path']))
    monkeypatch.setattr(rm, 'series_resistance_table', lambda *a, **k:
                        pd.DataFrame({
                            'block_id': [1, 2, 3],
                            'series_resistance': [np.nan] * 3,
                            'n_epochs_rs': [0] * 3,
                            'n_epochs_high_rs': [0] * 3}))

    def samples(requested, **kwargs):
        assert requested.block_id.tolist() == [1, 2, 3]
        return {i: (np.full((2, 100), level), 10000.)
                for i, level in [(1, -10.), (2, -1000.), (3, 1000.)]}

    monkeypatch.setattr(rm, '_amp_trace_samples', samples)
    monkeypatch.setattr(rm, 'trace_is_spiking',
                        lambda data, *a, **k: True)
    # Broad currents can fool the high-pass detector, as in block 37940.
    monkeypatch.setattr(rm, 'prominent_event_width_ms',
                        lambda data, rate: .4 if np.mean(data) == -10. else 6.)
    result = rm.check_series_resistance(
        blocks, block_level_evidence=True, drop=False, show=False)
    assert result.onlineAnalysis.tolist() == ['extracellular', 'exc', 'inh']
    assert result.series_resistance_source.tolist() == ['epoch parameters'] * 3
    assert result.onlineAnalysis_recorded.isna().all()

    def no_raw_reads(*args, **kwargs):
        raise AssertionError('metadata-only mode must not sample raw responses')

    monkeypatch.setattr(rm, '_amp_trace_samples', no_raw_reads)
    metadata = rm.check_series_resistance(
        blocks, block_level_evidence=True, infer_from_raw_trace=False,
        drop=False, show=False)
    assert metadata.onlineAnalysis.tolist() == ['extracellular'] * 3


def test_missing_label_does_not_default_to_extracellular(monkeypatch):
    monkeypatch.setattr(rm, 'trace_is_spiking', lambda *a, **k: False)
    mode, _ = rm.resolve_recording_mode(None, 0., np.full((2, 100), -1000.))
    assert mode == 'exc'


def test_raw_width_distinguishes_narrow_spikes_from_broad_currents():
    time = np.arange(10000) / 10000.
    narrow = np.zeros_like(time)
    broad = np.zeros_like(time)
    for centre in np.arange(.05, 1., .05):
        narrow -= 100 * np.exp(-.5 * ((time - centre) / .0002) ** 2)
        broad -= 100 * np.exp(-.5 * ((time - centre) / .003) ** 2)
    assert rm.prominent_event_width_ms(narrow[None], 10000.) < 1.5
    assert rm.prominent_event_width_ms(broad[None], 10000.) > 1.5


def test_resistance_can_be_stored_in_amplifier_background(tmp_path):
    import h5py

    with h5py.File(tmp_path / 'epoch.h5', 'w') as f:
        node = f.create_group('backgrounds/Amp1-id/dataConfigurationSpans/span_0/Amp1')
        node.attrs['seriesResistance'] = 8e6
        assert rm._epoch_series_resistance(f) == 8e6


def test_resistance_reader_skips_legacy_response_dataset(tmp_path):
    import h5py

    with h5py.File(tmp_path / 'epoch.h5', 'w') as f:
        f.create_dataset('responses/Amp1-id', data=np.zeros(2))
        node = f.create_group(
            'backgrounds/Amp1-id/dataConfigurationSpans/span_0/Amp1')
        node.attrs['seriesResistance'] = 9e6
        assert rm._epoch_series_resistance(f) == 9e6


def test_resistance_reader_accepts_auisql_epoch_dataset(tmp_path):
    import h5py

    with h5py.File(tmp_path / 'epoch.h5', 'w') as f:
        dataset = f.create_dataset('response-uuid', data=np.zeros(2))
        assert np.isnan(rm._epoch_series_resistance(dataset))


def test_resistance_reader_skips_legacy_span_dataset(tmp_path):
    import h5py

    with h5py.File(tmp_path / 'epoch.h5', 'w') as f:
        f.create_dataset(
            'responses/Amp1-id/dataConfigurationSpans/span_0', data=np.zeros(2))
        node = f.create_group(
            'backgrounds/Amp1-id/dataConfigurationSpans/span_0/Amp1')
        node.attrs['seriesResistance'] = 10e6
        assert rm._epoch_series_resistance(f) == 10e6


def test_trace_sampler_reads_only_requested_seconds(tmp_path, monkeypatch):
    import h5py
    from retinanalysis.utils import datajoint_utils as djutils

    path = tmp_path / 'traces.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('uuid', data=np.arange(10000))
        f.create_dataset('response/data/quantity', data=np.arange(10000))
    monkeypatch.setattr(djutils, 'get_h5_file', lambda exp: str(path))
    paths = pd.DataFrame({'block_id': [1, 1], 'sample_rate': [1000., 1000.],
                          'h5path': ['uuid', 'response']})
    result = rm._amp_trace_samples(
        pd.DataFrame({'exp_name': ['test'], 'block_id': [1]}),
        response_table=paths, trace_seconds=3.)
    traces, rate = result[1]
    assert traces.shape == (2, 3000)
    assert traces[0, -1] == 2999
    assert rate == 1000.
