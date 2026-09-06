import numpy as np
import pytest

from retinanalysis.SCutils import trace_processing as tp


def test_trace_processing_helpers_are_available_from_scutils():
    import retinanalysis as ra

    assert ra.SCutils.align_epoch_group_baselines is tp.align_epoch_group_baselines
    assert ra.SCutils.EPOCH_BASELINE_TARGETS == tp.EPOCH_BASELINE_TARGETS
    assert ra.SCutils.preprocess_spike_trace is tp.preprocess_spike_trace
    assert ra.SCutils.preprocess_whole_cell_trace is tp.preprocess_whole_cell_trace


def test_block_average_and_millisecond_conversion():
    assert tp.milliseconds_to_samples(5.0, 10_000.0) == 50
    np.testing.assert_array_equal(
        tp.block_average(np.arange(7.0), 2), [0.5, 2.5, 4.5])


def test_first_epoch_baseline_alignment_preserves_separate_group_levels():
    modulation = np.array([-1.0, 1.0, -1.0, 1.0])
    low = np.vstack([
        10.0 + modulation, 30.0 + modulation, 100.0 + modulation])
    high = np.vstack([
        100.0 + modulation, 140.0 + modulation, 300.0 + modulation])

    low_result = tp.align_epoch_group_baselines(low)
    high_result = tp.align_epoch_group_baselines(high)

    np.testing.assert_allclose(low_result.traces.mean(axis=1), 10.0)
    np.testing.assert_allclose(high_result.traces.mean(axis=1), 100.0)
    assert high_result.target - low_result.target == 90.0
    assert low_result.reference_mask.tolist() == [True, False, False]
    assert low_result.reference_n == 1


@pytest.mark.parametrize(
    ('method', 'target', 'reference_n'),
    [('first_two_mean', 20.0, 2), ('median', 30.0, 3)])
def test_baseline_alignment_retains_older_target_policies(
        method, target, reference_n):
    traces = np.array([[10.0, 10.0], [30.0, 30.0], [100.0, 100.0]])

    result = tp.align_epoch_group_baselines(traces, target=method)

    np.testing.assert_allclose(result.traces.mean(axis=1), target)
    assert result.target == target
    assert result.reference_n == reference_n


def test_baseline_alignment_requires_one_two_dimensional_group():
    with pytest.raises(ValueError, match='epoch x time'):
        tp.align_epoch_group_baselines(np.arange(4.0))
    with pytest.raises(ValueError, match='baseline target'):
        tp.align_epoch_group_baselines(np.ones((2, 4)), target='unknown')
