import json

import h5py
import numpy as np
import pandas as pd

from retinanalysis.SCutils import recording_mode
from retinanalysis.utils import datajoint_utils
from retinanalysis.utils.datajoint_utils import read_h5_response_trace


def test_read_h5_response_trace_supports_symphony_group(tmp_path):
    path = tmp_path / 'experiment.h5'
    expected = np.array([1.5, 2.5, 3.5])
    with h5py.File(path, 'w') as h5_file:
        response = h5_file.create_group('epoch/responses/Amp1-response')
        response.create_group('data').create_dataset('quantity', data=expected)

    with h5py.File(path, 'r') as h5_file:
        actual = read_h5_response_trace(
            h5_file, '/epoch/responses/Amp1-response')

    np.testing.assert_array_equal(actual, expected)


def test_database_round_trip_loads_direct_auisql_dataset(tmp_path):
    """Persisted data_file + h5path can reload AUISQL numeric samples."""
    data_file = tmp_path / '2022-09-09_B.auisql.h5'
    response_uuid = 'ABCDEF01-2345-6789-ABCD-EF0123456789'
    expected = np.array([-2823.125, -2812.1875, -2805.0])
    with h5py.File(data_file, 'w') as h5_file:
        h5_file.create_dataset(response_uuid, data=expected)

    database_record = json.loads(json.dumps({
        'data_file': str(data_file),
        'h5path': '/' + response_uuid,
    }))
    with h5py.File(database_record['data_file'], 'r') as h5_file:
        actual = read_h5_response_trace(
            h5_file, database_record['h5path'])

    np.testing.assert_array_equal(actual, expected)


def test_epochblock_amp_loader_reads_auisql_database_paths(
        tmp_path, monkeypatch):
    path = tmp_path / 'experiment.auisql.h5'
    with h5py.File(path, 'w') as h5_file:
        h5_file.create_dataset('first', data=[1.0, 2.0])
        h5_file.create_dataset('second', data=[3.0, 4.0])

    class Query:
        def fetch(self, **kwargs):
            return pd.DataFrame({
                'device_name': ['Amp1', 'Amp1'],
                'h5path': ['/first', '/second'],
                'sample_rate': [10000.0, 10000.0],
            })

    monkeypatch.setattr(
        datajoint_utils, 'get_epochblock_response_query',
        lambda exp_name, block_id: Query())

    traces, sample_rate = datajoint_utils.get_epochblock_amp_data(
        '2022-09-09_B', 1, str_h5=str(path), verbose=False)

    np.testing.assert_array_equal(traces, [[1.0, 2.0], [3.0, 4.0]])
    assert sample_rate == 10000.0


def test_activity_sampler_reads_auisql_database_paths(tmp_path, monkeypatch):
    path = tmp_path / 'experiment.auisql.h5'
    with h5py.File(path, 'w') as h5_file:
        h5_file.create_dataset('response', data=[-2.0, -1.0])
    monkeypatch.setattr(datajoint_utils, 'get_h5_file', lambda exp: str(path))
    blocks = pd.DataFrame({
        'exp_name': ['2022-09-09_B'],
        'block_id': [7],
    })
    response_table = pd.DataFrame({
        'block_id': [7],
        'h5path': ['/response'],
        'sample_rate': [10000.0],
    })

    result = recording_mode._amp_trace_samples(
        blocks, response_table=response_table, verbose=False)

    traces, sample_rate = result[7]
    np.testing.assert_array_equal(traces, [[-2.0, -1.0]])
    assert sample_rate == 10000.0
