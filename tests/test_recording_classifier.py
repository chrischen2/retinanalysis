"""Block-level recording-technique classifier."""
import numpy as np
import pandas as pd

from retinanalysis.SCutils import recording_classifier as rc


def test_feature_extractor_returns_one_finite_block_vector():
    time = np.arange(20000) / 10000.
    traces = np.vstack([
        np.sin(2 * np.pi * 20 * time),
        np.sin(2 * np.pi * 20 * time + .2),
    ])

    features = rc.extract_recording_block_features(traces, 10000.)

    assert tuple(features) == rc.FEATURE_NAMES
    assert np.isfinite(list(features.values())).all()


def test_feature_table_concatenates_epochs_into_one_row_per_block(monkeypatch):
    from retinanalysis.SCutils import recording_mode as rm

    blocks = pd.DataFrame({
        'exp_name': ['test', 'test'], 'block_id': [1, 1],
        'cell_label': ['cell1', 'cell1'],
    })
    monkeypatch.setattr(
        rm, '_amp_response_table',
        lambda ids: pd.DataFrame(columns=['block_id', 'h5path']))
    monkeypatch.setattr(
        rm, '_amp_trace_samples',
        lambda *args, **kwargs: {
            1: (np.vstack([np.arange(100.), np.arange(100.) + 1]), 1000.)})

    table = rc.recording_block_feature_table(blocks, verbose=False)

    assert len(table) == 1
    assert table.loc[0, 'block_id'] == 1
    assert table.loc[0, 'n_epochs_sampled'] == 2


def test_predictor_uses_cell_held_out_probability_for_training_blocks():
    from sklearn.dummy import DummyClassifier

    training = np.zeros((2, len(rc.FEATURE_NAMES)))
    model = DummyClassifier(strategy='prior').fit(training, [0, 1])
    bundle = {
        'model': model,
        'oof_p_whole_cell_by_block': {10: .97},
    }
    features = pd.DataFrame(
        np.zeros((2, len(rc.FEATURE_NAMES))), columns=rc.FEATURE_NAMES)
    features['block_id'] = [10, 11]

    predicted = rc.predict_recording_techniques(
        features, bundle, min_confidence=.9)

    assert predicted.loc[0, 'classifier_family'] == 'whole-cell'
    assert predicted.loc[0, 'classifier_source'] == 'cell-held-out prediction'
    assert predicted.loc[1, 'classifier_family'] == ''


def test_checked_in_model_retroactively_corrects_index_182():
    training = pd.read_csv(
        rc.DEFAULT_MODEL_PATH.with_name(
            'recording_technique_classifier_training.csv'))
    rows = training[training.block_id.isin(
        [37936, 37937, 37938, 37939, 37940, 37941])]

    predicted = rc.predict_recording_techniques(
        rows, rc.load_recording_technique_classifier())
    by_block = predicted.set_index('block_id')

    assert by_block.loc[37936, 'classifier_family'] == 'cell-attached'
    assert by_block.loc[37938, 'classifier_family'] == 'whole-cell'
    assert by_block.loc[37938, 'classifier_p_whole_cell'] >= .99
    assert by_block.classifier_source.eq('cell-held-out prediction').all()
