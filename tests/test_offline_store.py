"""Smoke + roundtrip tests for the offline HDF5 data store.

Does NOT spin up DataJoint or the pipeline — it forges a minimal
``pipeline``-shaped object with the attributes ``save_offline_data``
actually reads, writes the HDF5, and verifies every saved field round-
trips on reload.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from retinanalysis.utils.offline_store import (
    save_offline_data, load_offline_data, OfflineDataset, offline_h5_path,
)


def _fake_pipeline(tmp_root: Path):
    """Build a minimal pipeline-shaped object with exactly the fields
    ``save_offline_data`` touches.
    """
    pre_ms, stim_ms, tail_ms = 100.0, 200.0, 50.0
    n_epochs = 4
    cell_ids = [101, 202]
    # 2 conditions, alternating
    cond_vals = [0.1, 1.0, 0.1, 1.0]
    epoch_params = [{'currentBackgroundScale': v, 'currentImageName': 'img1'}
                    for v in cond_vals]

    df_epochs = pd.DataFrame({
        'epoch_index': list(range(n_epochs)),
        'epoch_parameters': epoch_params,
        'currentBackgroundScale': cond_vals,
        'currentImageName': ['img1'] * n_epochs,
    })

    stim = SimpleNamespace(
        df_epochs=df_epochs,
        d_epoch_block_params={'preTime': pre_ms, 'stimTime': stim_ms, 'tailTime': tail_ms},
    )

    rng = np.random.RandomState(0)
    spike_rows = []
    for cid in cell_ids:
        sts_per_epoch = [np.sort(rng.uniform(0, pre_ms + stim_ms + tail_ms, size=10))
                         for _ in range(n_epochs)]
        spike_rows.append({'cell_id': cid, 'cell_type': 'OnP',
                           'noise_id': cid + 1000, 'spike_times': sts_per_epoch})
    df_spike = pd.DataFrame(spike_rows)
    resp = SimpleNamespace(
        df_spike_times=df_spike,
        exp_name='99999999X',
        datafile_name='data999',
        protocol_name='edu.washington.riekelab.turner.protocols.EyeMovementTrajectoryAlternatingBackground',
        ss_version='kilosort2.5',
    )
    ac = SimpleNamespace(
        exp_name='99999999X',
        chunk_name='chunkX',
        pixels_per_stixel=4.0,
        microns_per_stixel=20.0,
        rf_params={101 + 1000: {'center_x': 1.0, 'center_y': 2.0,
                                'std_x': 3.0, 'std_y': 4.0, 'rot': 0.1},
                   202 + 1000: {'center_x': 5.0, 'center_y': 6.0,
                                'std_x': 7.0, 'std_y': 8.0, 'rot': -0.2}},
    )
    pipeline = SimpleNamespace(
        resp=resp, stim=stim, analysis_chunk=ac,
        match_dict={101 + 1000: 101, 202 + 1000: 202},
        corr_dict={101 + 1000: 0.95, 202 + 1000: 0.88},
    )

    qc = pd.DataFrame({'cell_id': cell_ids, 'passes': [True, True]})
    return pipeline, qc


def test_offline_roundtrip(tmp_path):
    pipeline, qc = _fake_pipeline(tmp_path)

    out = save_offline_data(
        pipeline, qc=qc, output_root=tmp_path,
        psth_sigma_ms=10.0, sample_rate_hz=1000.0,
        cell_match_df=None,  # skip auto-build (no EI)
        verbose=False,
    )
    assert out.exists()

    ds = load_offline_data('99999999X', protocol='eye_movement_alt_bg',
                          output_root=tmp_path)
    assert isinstance(ds, OfflineDataset)
    assert ds.exp_name == '99999999X'
    assert sorted(ds.cell_ids) == [101, 202]
    assert ds.cell_types() == ['OnP']
    assert ds.timing['preTime_ms'] == 100.0
    assert ds.timing['stimTime_ms'] == 200.0
    assert ds.timing['tailTime_ms'] == 50.0
    assert ds.timing['sample_rate_hz'] == 1000.0

    # 4 epochs, all two condition columns present
    assert len(ds.epochs) == 4
    assert 'currentBackgroundScale' in ds.epochs.columns
    assert 'currentImageName' in ds.epochs.columns

    # Spike times come back as lists of float arrays
    sts = ds.spike_times(101)
    assert len(sts) == 4
    assert all(isinstance(s, np.ndarray) for s in sts)
    assert all(s.dtype == np.float64 for s in sts)

    # PSTH shape: (n_epochs, n_bins) where n_bins = (pre+stim+tail) ms * sample_rate / 1000
    psth = ds.psth_matrix(101)
    n_bins = int(round((100 + 200 + 50) / 1000 * 1000))
    assert psth.shape == (4, n_bins)

    # STA / pixel-scale attrs are on each cell row
    row = ds.cells.loc[101]
    assert float(row['sta_center_x']) == 1.0
    assert float(row['sta_std_y']) == 4.0
    assert float(row['pixels_per_stixel']) == 4.0


def test_offline_filters_to_qc_pass(tmp_path):
    pipeline, _ = _fake_pipeline(tmp_path)
    # Fail one cell — only the other should be saved.
    qc = pd.DataFrame({'cell_id': [101, 202], 'passes': [True, False]})
    save_offline_data(pipeline, qc=qc, output_root=tmp_path,
                      cell_match_df=None, verbose=False)
    ds = load_offline_data('99999999X', output_root=tmp_path)
    assert ds.cell_ids == [101]


def test_offline_visual_qc_intersects(tmp_path):
    pipeline, qc = _fake_pipeline(tmp_path)
    # Even though both pass automated QC, only 101 is visually 'good'.
    vqc = pd.DataFrame({'cell_id': [101, 202], 'tag': ['good', 'bad']})
    save_offline_data(pipeline, qc=qc, visual_qc_df=vqc, output_root=tmp_path,
                      cell_match_df=None, verbose=False)
    ds = load_offline_data('99999999X', output_root=tmp_path)
    assert ds.cell_ids == [101]


# ---------------------------------------------------------------------------
# Per-cell PNG pruner — used when re-archiving after visual QC tagging
# ---------------------------------------------------------------------------

import os
from retinanalysis.utils.cell_plot_archive import _prune_stale_cell_pngs


def test_prune_stale_cell_pngs(tmp_path):
    cells_root = tmp_path / 'cells'
    # Seed: a few cell-type dirs with both raster and psth PNGs.
    for ct, ids in [('OnP', [10, 20, 30]), ('OffP', [40, 50])]:
        (cells_root / ct).mkdir(parents=True)
        for cid in ids:
            for kind in ('raster', 'psth'):
                (cells_root / ct / f'cell_{cid}_{kind}.png').touch()
        # A non-canonical file: pruner must leave it alone.
        (cells_root / ct / 'README.txt').touch()
    # Cell-type dir whose only file is a stale PNG — should be removed entirely.
    (cells_root / 'OnM').mkdir()
    (cells_root / 'OnM' / 'cell_99_raster.png').touch()

    kept = {10, 30, 50}  # bad: 20, 40, 99
    n = _prune_stale_cell_pngs(str(cells_root), kept)

    assert n == 5  # cell_20×2 + cell_40×2 + cell_99×1
    # Surviving canonical PNGs match exactly the kept set
    survivors = {p.name for p in cells_root.rglob('cell_*.png')}
    assert survivors == {
        'cell_10_raster.png', 'cell_10_psth.png',
        'cell_30_raster.png', 'cell_30_psth.png',
        'cell_50_raster.png', 'cell_50_psth.png',
    }
    # README files survive
    assert (cells_root / 'OnP' / 'README.txt').exists()
    assert (cells_root / 'OffP' / 'README.txt').exists()
    # Empty cell-type dir gets removed
    assert not (cells_root / 'OnM').exists()


def test_prune_stale_cell_pngs_noop_when_kept_is_superset(tmp_path):
    """No files should be removed when every existing cell is in ``kept_ids``."""
    cells_root = tmp_path / 'cells'
    (cells_root / 'OnP').mkdir(parents=True)
    for cid in [10, 20]:
        (cells_root / 'OnP' / f'cell_{cid}_raster.png').touch()

    n = _prune_stale_cell_pngs(str(cells_root), kept_ids={10, 20, 99})
    assert n == 0
    assert len(list(cells_root.rglob('cell_*.png'))) == 2
