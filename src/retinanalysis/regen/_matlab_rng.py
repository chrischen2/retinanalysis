"""MATLAB-backed RNG helpers (byte-exact parity).

These wrap the MATLAB engine to produce uniform / normal random values
that match exactly what the Symphony protocol produced on the rig:

    rs = RandStream('mt19937ar', 'Seed', seed);
    rs.rand(n_rows, n_cols)
    rs.randn(n_rows, n_cols)

Why use these instead of pure NumPy
-----------------------------------
- ``.rand()``: NumPy's ``np.random.RandomState(seed).rand(...)`` is already
  byte-exact with MATLAB (same MT19937, same legacy seeding). You don't
  strictly need MATLAB for uniform parity — but using these helpers means
  one code path regardless of distribution.
- ``.randn()``: MATLAB uses Marsaglia-Tsang ziggurat, NumPy uses polar
  Marsaglia. Same distribution, different per-sample values. To get
  byte-exact replay of the actual stimulus shown on the rig, you must
  call MATLAB.

Each call here creates a fresh ``RandStream`` in MATLAB seeded with the
given integer. That mirrors the protocols' per-epoch seeding pattern
(``noiseStream = RandStream(..., 'Seed', obj.seed)``), so a single
batched call generates all draws for that epoch in one MATLAB round-trip.

If MATLAB engine isn't available, callers should catch the
``MatlabUnavailableError`` and fall back to a numpy-only path.
"""

from __future__ import annotations

import numpy as np

from retinanalysis.utils.matlab_engine import get_matlab_engine


class MatlabUnavailableError(RuntimeError):
    """Raised when a MATLAB-backed path is requested but the engine can't start."""


def _require_engine():
    eng = get_matlab_engine()
    if eng is None:
        raise MatlabUnavailableError(
            'MATLAB engine is not available. Install with: '
            'pip install /Applications/MATLAB_R<release>.app/extern/engines/python'
        )
    return eng


def matlab_rand(seed: int, n_rows: int, n_cols: int = 1) -> np.ndarray:
    """Byte-exact MATLAB ``RandStream('mt19937ar','Seed',seed).rand(n_rows, n_cols)``.

    Returns a NumPy float64 array of shape ``(n_rows, n_cols)``. MATLAB
    fills column-major; the result is converted to NumPy's row-major
    convention while preserving the value at each (i, j) — i.e. element
    ``[i, j]`` matches what MATLAB shows in row i, column j.
    """
    eng = _require_engine()
    # Build the values in MATLAB and pull them back as a numeric array.
    arr = eng.eval(
        f"reshape(RandStream('mt19937ar','Seed',{int(seed)}).rand({int(n_rows)},{int(n_cols)}),"
        f"{int(n_rows)},{int(n_cols)})",
        nargout=1,
    )
    return np.asarray(arr, dtype=np.float64).reshape(int(n_rows), int(n_cols))


def matlab_randn(seed: int, n_rows: int, n_cols: int = 1) -> np.ndarray:
    """Byte-exact MATLAB ``RandStream('mt19937ar','Seed',seed).randn(n_rows, n_cols)``.

    Uses MATLAB's default ``NormalTransform='Ziggurat'``.
    """
    eng = _require_engine()
    arr = eng.eval(
        f"reshape(RandStream('mt19937ar','Seed',{int(seed)}).randn({int(n_rows)},{int(n_cols)}),"
        f"{int(n_rows)},{int(n_cols)})",
        nargout=1,
    )
    return np.asarray(arr, dtype=np.float64).reshape(int(n_rows), int(n_cols))
