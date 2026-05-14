"""Victor–Purpura spike-train distance.

Python port of ``spkd_with_scr.m`` (Daniel Reich / Jonathan Victor):

    d = spkd(tli, tlj, cost)

The cost parameter is the cost per unit time to move a spike — its
reciprocal is the timescale of the metric. A larger ``cost`` means
moving spikes is expensive, so the metric "cares" about small
differences and falls back to insert/delete (cost = 1) once spikes are
more than ``1/cost`` apart in time. Spike times must be in seconds when
``cost`` is in units of 1/s (the convention in the MATLAB original).
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from typing import Optional, Tuple


__all__ = ['victor_purpura_distance', 'victor_purpura_distance_matrix',
           'victor_purpura_cross_matrix']


# ---------------------------------------------------------------------------
# C extension: build on first import, ctypes-loaded
# ---------------------------------------------------------------------------

_EXT_DIR = Path(__file__).parent / '_vpext'
_LIB_NAME = 'libspkd.dylib' if sys.platform == 'darwin' else 'libspkd.so'
_LIB_PATH = _EXT_DIR / _LIB_NAME
_LIB = None  # populated lazily


def _build_ext(verbose: bool = False) -> Optional[Path]:
    """Compile the C extension. Returns the .so/.dylib path or ``None``."""
    src = _EXT_DIR / 'spkd.c'
    if not src.exists():
        return None
    cmd = ['cc', '-O3', '-ffast-math', '-shared', '-fPIC',
           '-o', str(_LIB_PATH), str(src)]
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        if verbose:
            print(f'[victor_purpura] built {_LIB_PATH}')
        return _LIB_PATH
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        if verbose:
            print(f'[victor_purpura] C build failed ({exc!r}); using pure-Python fallback')
        return None


def _load_lib():
    """Try to load the compiled lib; return None to signal pure-Python fallback."""
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        _build_ext()
    if not _LIB_PATH.exists():
        return None
    lib = ctypes.CDLL(str(_LIB_PATH))

    lib.vp_distance.restype = ctypes.c_double
    lib.vp_distance.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double,
    ]
    lib.vp_pairwise.restype = None
    lib.vp_pairwise.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_double),
    ]
    lib.vp_self_pairwise.restype = None
    lib.vp_self_pairwise.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def _flatten(trains):
    """Concat trains into one float64 buffer + int32 lengths buffer."""
    lens = np.array([len(t) for t in trains], dtype=np.int32)
    if lens.sum() == 0:
        times = np.empty(0, dtype=np.float64)
    else:
        times = np.concatenate([np.ascontiguousarray(t, dtype=np.float64)
                                 for t in trains])
    return times, lens


def victor_purpura_distance(
    tli: np.ndarray,
    tlj: np.ndarray,
    cost: float,
) -> float:
    """Single-pair VP distance. Faithful port of ``spkd_with_scr.m``.

    Routes through a compiled C kernel when available (``_vpext/libspkd``);
    falls back to a pure-Python DP otherwise.

    Parameters
    ----------
    tli, tlj : 1-D array
        Spike times (seconds — or any unit, as long as it matches
        ``1/cost``).
    cost : float
        Cost per unit time to shift a spike. Setting ``cost = 0``
        recovers the count-difference distance; ``cost → ∞`` recovers
        the no-coincidence distance.

    Returns
    -------
    float
        VP distance (lower-bounded by ``abs(len(tli) - len(tlj))``,
        upper-bounded by ``len(tli) + len(tlj)``).
    """
    tli = np.ascontiguousarray(tli, dtype=np.float64).ravel()
    tlj = np.ascontiguousarray(tlj, dtype=np.float64).ravel()
    nspi = tli.size
    nspj = tlj.size

    # Fast paths (cheaper than the DP for either branch)
    if nspi == 0:
        return float(nspj)
    if nspj == 0:
        return float(nspi)
    if cost == 0:
        return float(abs(nspi - nspj))

    lib = _load_lib()
    if lib is not None:
        return float(lib.vp_distance(
            tli.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nspi,
            tlj.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nspj,
            ctypes.c_double(cost),
        ))

    # Pure-Python fallback
    scr = np.zeros((nspi + 1, nspj + 1), dtype=np.float64)
    scr[:, 0] = np.arange(nspi + 1)
    scr[0, :] = np.arange(nspj + 1)
    for i in range(1, nspi + 1):
        ti = tli[i - 1]
        prev_row = scr[i - 1]
        cur_row = scr[i]
        for j in range(1, nspj + 1):
            cur_row[j] = min(
                prev_row[j] + 1.0,
                cur_row[j - 1] + 1.0,
                prev_row[j - 1] + cost * abs(ti - tlj[j - 1]),
            )
    return float(scr[nspi, nspj])


def victor_purpura_cross_matrix(
    trains_a: list,
    trains_b: list,
    cost: float,
) -> np.ndarray:
    """Pairwise VP distances ``D[i, j] = d(trains_a[i], trains_b[j])``.

    Bulk path: one C call computes all pairs at once, amortizing
    Python/ctypes overhead. Falls back to a Python loop when the C
    extension is unavailable.
    """
    if not trains_a or not trains_b:
        return np.zeros((len(trains_a), len(trains_b)), dtype=np.float64)
    lib = _load_lib()
    if lib is None:
        return np.array(
            [[victor_purpura_distance(a, b, cost) for b in trains_b]
             for a in trains_a],
            dtype=np.float64,
        )
    a_times, a_lens = _flatten(trains_a)
    b_times, b_lens = _flatten(trains_b)
    out = np.zeros((len(trains_a), len(trains_b)), dtype=np.float64)
    lib.vp_pairwise(
        a_times.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        a_lens.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(len(trains_a)),
        b_times.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b_lens.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(len(trains_b)),
        ctypes.c_double(cost),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return out


def victor_purpura_distance_matrix(
    trains: list,
    cost: float,
    symmetric: bool = True,
) -> np.ndarray:
    """Symmetric pairwise VP distance matrix.

    Parameters
    ----------
    trains : list of 1-D arrays
        Each entry is a spike-time array (seconds).
    cost : float
        Forwarded to :func:`victor_purpura_distance`.
    symmetric : bool
        Kept for back-compat. The C path always returns a symmetric
        matrix with zero diagonal.

    Returns
    -------
    ndarray of shape (n, n)
    """
    n = len(trains)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    lib = _load_lib()
    if lib is None:
        # Pure-Python fallback
        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = victor_purpura_distance(trains[i], trains[j], cost)
                D[i, j] = d
                D[j, i] = d
        return D
    times, lens = _flatten(trains)
    out = np.zeros((n, n), dtype=np.float64)
    lib.vp_self_pairwise(
        times.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        lens.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(n),
        ctypes.c_double(cost),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return out
