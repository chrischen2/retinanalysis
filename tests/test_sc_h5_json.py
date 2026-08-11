import h5py

from retinanalysis.SCutils.h5_json import repair_h5_paths


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
