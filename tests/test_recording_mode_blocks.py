"""Mixed recording modes within a single epoch group."""
import numpy as np
import pandas as pd

from retinanalysis.SCutils import recording_mode as rm


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
