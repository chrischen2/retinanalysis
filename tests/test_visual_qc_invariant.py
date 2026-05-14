"""Safety check: archiving must never overwrite manual visual_qc tags.

The chrisMain workflow is iterative — the user runs §17/§18 to render
PNGs, tags them via §16, then re-runs §17/§18 to collapse the archive
to ``tag == 'good'``. If any code path inside ``analyze_experiment``
were to silently rewrite ``visual_qc.csv`` (e.g. via a stray ``to_csv``
on the wrong dataframe, or a future refactor that recomputes tags),
the user's manual review would be silently destroyed and they'd have
to redo it from scratch.

This test pins the invariant: build a fake ``visual_qc.csv`` with a
known checksum, run ``analyze_experiment`` against the live archive
for ``20220823C``, and assert the file's bytes are unchanged. The test
uses a temp ``output_root`` so it touches nothing in the real archive.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import retinanalysis as ra


@pytest.mark.slow
def test_visual_qc_csv_byte_identical_after_archive():
    """``visual_qc.csv`` must be byte-identical before and after a
    re-run of :func:`analyze_experiment`.

    Skipped silently when the test data isn't on disk (CI without SSD).
    """
    import os
    if not os.path.isdir(os.path.join(ra.ANALYSIS_DIR, '20220823C')):
        pytest.skip('20220823C not available on this machine')

    with tempfile.TemporaryDirectory() as tmp:
        # 1) Build the initial archive so PNGs + index.csv exist.
        ra.analyze_experiment(
            '20220823C', datafile_name='data035',
            cell_types=['OnP', 'OffM'],
            output_root=tmp, overwrite=True, fit_calibration=False,
            n_jobs=1, verbose=False,
        )

        # 2) Plant a hand-built visual_qc.csv with a stable checksum.
        proto_dir = Path(tmp) / '20220823C' / 'eye_movement_alt_bg'
        vqc_path = proto_dir / 'visual_qc.csv'
        idx = pd.read_csv(proto_dir / 'index.csv')
        sample = idx.head(5).copy()
        vqc = pd.DataFrame({
            'exp_name': ['20220823C'] * 5,
            'cell_id': sample['cell_id'].astype(int).tolist(),
            'cell_type': sample['cell_type'].tolist(),
            'tag': ['good', 'good', 'bad', 'good', 'bad'],
            'timestamp': ['2026-05-13T00:00:00'] * 5,
            'inspector': ['safety-test'] * 5,
        })
        vqc.to_csv(vqc_path, index=False)
        before_hash = hashlib.sha256(vqc_path.read_bytes()).hexdigest()
        before_mtime = vqc_path.stat().st_mtime_ns

        # 3) Re-run the archive (this is the operation that must NOT
        # rewrite visual_qc.csv). respect_visual_qc=True (default) so
        # the function reads the file — it must read-only.
        ra.analyze_experiment(
            '20220823C', datafile_name='data035',
            cell_types=['OnP', 'OffM'],
            output_root=tmp, overwrite=True, fit_calibration=False,
            n_jobs=1, verbose=False,
        )

        after_hash = hashlib.sha256(vqc_path.read_bytes()).hexdigest()
        after_mtime = vqc_path.stat().st_mtime_ns

        assert after_hash == before_hash, (
            'visual_qc.csv bytes changed after analyze_experiment re-ran. '
            f'before={before_hash} after={after_hash}'
        )
        assert after_mtime == before_mtime, (
            'visual_qc.csv mtime changed after analyze_experiment re-ran '
            '— some code path rewrote the file even though content was '
            'restored.'
        )
