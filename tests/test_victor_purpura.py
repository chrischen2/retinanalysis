"""Unit tests for the Victor-Purpura distance — port of ``spkd_with_scr.m``.

Pins the algorithm against hand-computable cases so future refactors
(e.g. swapping in a numba/Cython implementation) can't silently
regress.
"""
from __future__ import annotations

import numpy as np

try:  # pytest is a dev dep but not in the conda env
    import pytest
    _approx = pytest.approx
except ImportError:
    class _Approx:
        def __init__(self, x, tol=1e-9):
            self.x = x; self.tol = tol
        def __eq__(self, other):
            return abs(float(other) - float(self.x)) < self.tol
        def __repr__(self):
            return f'approx({self.x})'
    def _approx(x, abs=1e-9):
        return _Approx(x, abs)

from retinanalysis.utils.victor_purpura import (
    victor_purpura_distance,
    victor_purpura_distance_matrix,
)


def test_identical_trains_are_zero():
    assert victor_purpura_distance([0.1, 0.5, 0.9], [0.1, 0.5, 0.9], 1.0) == 0.0


def test_empty_vs_n_costs_n():
    assert victor_purpura_distance([], [], 1.0) == 0.0
    assert victor_purpura_distance([], [0.1, 0.2, 0.3], 1.0) == 3.0
    assert victor_purpura_distance([0.1, 0.2], [], 1.0) == 2.0


def test_small_shift_cheaper_than_delete_insert():
    # Single spike shifted by 0.5s at cost=1/s: shift = 0.5, delete+insert = 2 → 0.5.
    assert victor_purpura_distance([1.0], [1.5], 1.0) == _approx(0.5)


def test_large_shift_falls_back_to_delete_insert():
    # Shift cost (5.0) > delete+insert (2.0) → metric returns 2.
    assert victor_purpura_distance([1.0], [6.0], 1.0) == _approx(2.0)


def test_zero_cost_recovers_count_difference():
    # cost=0: any shift is free → distance == |n_a - n_b|.
    assert victor_purpura_distance([1, 2, 3], [4, 5], 0.0) == 1.0
    assert victor_purpura_distance([1, 2, 3, 4, 5], [], 0.0) == 5.0


def test_high_cost_recovers_no_coincidence():
    # cost=∞: only exact-time matches register. Two spikes at the same
    # time costs 0; everything else costs 2 (delete+insert).
    big = 1e9
    assert victor_purpura_distance([1.0, 2.0], [1.0, 2.0], big) == 0.0
    # All four spikes at distinct times → 4 (delete two, insert two).
    assert victor_purpura_distance([1.0, 2.0], [3.0, 4.0], big) == _approx(4.0)


def test_matrix_is_symmetric_and_zero_diagonal():
    trains = [
        np.array([0.1, 0.5, 0.9]),
        np.array([0.1, 0.6]),
        np.array([0.2, 0.4, 0.6, 0.8]),
        np.array([]),
    ]
    D = victor_purpura_distance_matrix(trains, cost=2.0)
    assert D.shape == (4, 4)
    assert np.allclose(np.diag(D), 0.0)
    assert np.allclose(D, D.T)


def test_matrix_matches_pairwise_calls():
    rng = np.random.RandomState(0)
    trains = [np.sort(rng.uniform(0, 5, size=10)) for _ in range(4)]
    D = victor_purpura_distance_matrix(trains, cost=3.0)
    # Spot-check 3 off-diagonal pairs against the scalar fn.
    for (i, j) in [(0, 1), (1, 3), (2, 3)]:
        expected = victor_purpura_distance(trains[i], trains[j], 3.0)
        assert D[i, j] == _approx(expected)


def test_distance_bounds():
    """Triangle: VP distance is bounded by ``|n_a - n_b|`` (low) and ``n_a + n_b`` (high)."""
    rng = np.random.RandomState(1)
    for _ in range(20):
        a = np.sort(rng.uniform(0, 5, size=rng.randint(0, 15)))
        b = np.sort(rng.uniform(0, 5, size=rng.randint(0, 15)))
        d = victor_purpura_distance(a, b, cost=1.0)
        assert d >= abs(len(a) - len(b)) - 1e-9
        assert d <= len(a) + len(b) + 1e-9
