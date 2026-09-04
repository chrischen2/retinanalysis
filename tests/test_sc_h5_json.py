import h5py
import json

from retinanalysis.SCutils.h5_json import (
    repair_h5_paths,
    update_single_cell_json,
)
from retinanalysis.utils.parse_data import find_new_h5_files


def test_repair_h5_paths_restores_response_and_stimulus_paths(tmp_path):
    response_uuid = '11111111-1111-1111-1111-111111111111'
    stimulus_uuid = '22222222-2222-2222-2222-222222222222'
    h5_path = tmp_path / 'experiment.h5'
    with h5py.File(h5_path, 'w') as h5_file:
        epoch = h5_file.create_group('experiment/epochs/epoch-1')
        epoch.create_group(f'responses/Amp1-{response_uuid}')
        epoch.create_group(f'stimuli/UV LED-{stimulus_uuid}')

    metadata = {
        'animals': [{'epochs': [{
            'responses': {'Amp1': {'uuid': response_uuid}},
            'stimuli': {'UV LED': {'uuid': stimulus_uuid}},
        }]}]
    }

    assert repair_h5_paths(metadata, h5_path) == 2
    epoch = metadata['animals'][0]['epochs'][0]
    assert epoch['responses']['Amp1']['h5path'].endswith(response_uuid)
    assert epoch['stimuli']['UV LED']['h5path'].endswith(stimulus_uuid)
    assert repair_h5_paths(metadata, h5_path) == 0


def test_find_new_h5_files_prefers_primary_over_auisql(tmp_path):
    h5_dir = tmp_path / 'h5'
    json_dir = tmp_path / 'json'
    h5_dir.mkdir()
    json_dir.mkdir()
    primary = h5_dir / '2026-09-04_G.h5'
    auisql = h5_dir / '2026-09-04_G.auisql.h5'
    primary.touch()
    auisql.touch()

    assert find_new_h5_files(h5_dir, json_dir) == [primary]


def test_find_new_h5_files_uses_auisql_as_fallback(tmp_path):
    h5_dir = tmp_path / 'h5'
    json_dir = tmp_path / 'json'
    h5_dir.mkdir()
    json_dir.mkdir()
    auisql = h5_dir / '2026-09-04_G.auisql.h5'
    auisql.touch()

    assert find_new_h5_files(h5_dir, json_dir) == [auisql]

    (json_dir / '2026-09-04_G.json').write_text('{}')
    assert find_new_h5_files(h5_dir, json_dir) == []


def test_update_single_cell_json_publishes_canonical_auisql_name(
        tmp_path, monkeypatch):
    volume = tmp_path / 'SingleCellSSD'
    h5_dir = volume / 'single_cell' / 'chris_data' / 'h5'
    json_dir = volume / 'single_cell' / 'chris_data' / 'json'
    h5_dir.mkdir(parents=True)
    json_dir.mkdir()
    auisql = h5_dir / '2026-09-04_G.auisql.h5'
    auisql.touch()
    auisql.with_suffix('').touch()

    def fake_convert(database_path, h5_path, out_path):
        with open(out_path, 'w') as stream:
            json.dump({'source': str(h5_path)}, stream)

    monkeypatch.setattr(
        'retinanalysis.SCutils.auisql_json.convert_auisql_to_json',
        fake_convert)

    report = update_single_cell_json(volume=volume, verbose=False)

    canonical_json = json_dir / '2026-09-04_G.json'
    assert report.pending == (auisql,)
    assert report.created == (canonical_json,)
    assert canonical_json.is_file()
    assert not (json_dir / '2026-09-04_G.auisql.json').exists()


def test_repair_h5_paths_skips_unstored_auisql_stimulus(tmp_path):
    metadata = {'stimuli': {'UV LED': {
        'uuid': 'not-in-h5', 'h5path': '', 'dataStored': False,
    }}}
    h5_path = tmp_path / 'response.h5'
    with h5py.File(h5_path, 'w'):
        pass

    assert repair_h5_paths(metadata, h5_path) == 0
