from types import SimpleNamespace

import numpy as np
import pandas as pd

from retinanalysis.classes.dedup import generate_extended_pairings
from retinanalysis.utils.dedup import apply_dedup
from retinanalysis.utils.vision_utils import get_spike_xarr


def _block():
    block = SimpleNamespace(
        cell_ids=np.array([11, 12, 13]),
        d_EIs={
            11: np.array([[1.0, -8.0]]),
            12: np.array([[1.0, -4.0]]),
            13: np.array([[1.0, -2.0]]),
        },
        df_spike_times=pd.DataFrame({
            'cell_id': [11, 12, 13],
            'cell_type': ['OnP', 'OnP', 'OffP'],
            'noise_id': [101, 102, 103],
            'spike_times': [
                [np.array([1.0, 5.0]), np.array([2.0])],
                [np.array([1.2, 8.0]), np.array([4.0])],
                [np.array([9.0]), np.array([6.0])],
            ],
        }),
    )
    block.d_EI_error = {cid: np.ones((1, 2)) for cid in block.cell_ids}
    block.df_spike_times['binned_spikes'] = [
        np.ones((2, 3)) for _ in block.cell_ids
    ]
    block.binned_spikes = np.ones((3, 2, 3))
    block.bin_rate = 100.0
    block.time_bins_ms = np.arange(3)
    block.n_epochs = 2
    return block


def test_generate_extended_pairings_uses_full_transitive_closure():
    pairs = {(1, 2), (2, 3), (3, 4), (10, 11)}
    assert generate_extended_pairings(pairs) == {(1, 2, 3, 4), (10, 11)}


def test_apply_dedup_unions_spikes_and_removes_duplicate_row():
    block = _block()

    log = apply_dedup(
        block, [(11, 12)], refractory_ms=0.5, verbose=False,
    )

    assert block.cell_ids.tolist() == [11, 13]
    assert block.df_spike_times['cell_id'].tolist() == [11, 13]
    merged = block.df_spike_times.loc[0, 'spike_times']
    np.testing.assert_array_equal(merged[0], np.array([1.0, 5.0, 8.0]))
    np.testing.assert_array_equal(merged[1], np.array([2.0, 4.0]))
    assert log.loc[0, 'representative'] == 11
    assert log.loc[0, 'dropped'] == (12,)
    assert log.loc[0, 'n_spikes_added_to_rep'] == 2


def test_apply_dedup_preview_does_not_mutate_block():
    block = _block()

    apply_dedup(block, [(11, 12)], inplace=False, verbose=False)

    assert block.cell_ids.tolist() == [11, 12, 13]
    assert block.df_spike_times['cell_id'].tolist() == [11, 12, 13]


def test_apply_dedup_synchronizes_pipeline_state_for_downstream_analysis():
    block = _block()
    pipeline = SimpleNamespace(
        resp=block,
        analysis_chunk=SimpleNamespace(),
        match_dict={101: 11, 102: 12, 103: 13},
        corr_dict={101: 0.95, 102: 0.93, 103: 0.91},
    )

    apply_dedup(pipeline, [(11, 12)], verbose=False)

    assert set(block.d_EIs) == {11, 13}
    assert set(block.d_EI_error) == {11, 13}
    assert pipeline.match_dict == {101: 11, 103: 13}
    assert pipeline.corr_dict == {101: 0.95, 103: 0.91}
    assert 'binned_spikes' not in block.df_spike_times
    assert not hasattr(block, 'binned_spikes')
    assert not hasattr(block, 'bin_rate')
    assert not hasattr(block, 'time_bins_ms')
    assert block.dedup_log.loc[0, 'representative'] == 11
    assert block.dedup_log.loc[0, 'dropped'] == (12,)

    # This is the spike-array entry point used throughout both notebooks.
    # It must see the merged train and must not expose the deleted cluster.
    spikes = get_spike_xarr(block)
    assert spikes.coords['cell_id'].values.tolist() == [11, 13]
    np.testing.assert_array_equal(
        spikes.sel(cell_id=11, epoch=0).item(),
        np.array([1.0, 5.0, 8.0]),
    )
