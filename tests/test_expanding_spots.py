"""Tests for SCutils.protocols.expanding_spots.

Covers the model and the pure helpers — no database or SSD needed. The full
analyze_expanding_spots() path is exercised by demos/4_patchdata_demo.ipynb.
"""
import numpy as np
import pandas as pd
import pytest

from retinanalysis.SCutils.protocols import expanding_spots as es


def test_dog_matches_matlab_formula():
    """dog_area_summation must equal DoG.m evaluated on the same params."""
    Kc, sigma_c, Ks, sigma_s, base = 1.3, 40.0, 0.8, 300.0, 0.1
    for d in (0.0, 40.0, 120.0, 400.0, 1200.0):
        r = d / 2  # DoG.m: r = spotSizes ./ 2
        expected = (base
                    + Kc * (1 - np.exp(-(r ** 2) / (2 * sigma_c ** 2)))
                    - Ks * (1 - np.exp(-(r ** 2) / (2 * sigma_s ** 2))))
        got = es.dog_area_summation(d, Kc, sigma_c, Ks, sigma_s, base)
        assert np.isclose(got, expected), (d, got, expected)


def test_dog_is_baseline_at_zero_and_saturates():
    p = dict(Kc=1.0, sigma_c=30.0, Ks=0.4, sigma_s=200.0, base=0.2)
    assert np.isclose(es.dog_area_summation(0.0, **p), p['base'])
    # As diameter -> infinity both terms saturate: base + Kc - Ks.
    assert np.isclose(es.dog_area_summation(1e6, **p), p['base'] + p['Kc'] - p['Ks'])


def test_bounds_match_matlab_fitdog():
    assert es.DOG_LOWER == (0.0, 2.0, 0.0, 10.0, 0.0)
    assert es.DOG_UPPER == (3.0, 200.0, 3.0, 1000.0, 1.0)
    assert es.DOG_P0 == (1.0, 5.0, 3.0, 500.0, 0.0)


def test_fit_recovers_known_sigmas():
    truth = dict(Kc=1.0, sigma_c=35.0, Ks=0.55, sigma_s=250.0, base=0.05)
    sizes = np.repeat([20, 40, 60, 80, 120, 160, 200, 300, 400, 600], 3)
    y = es.dog_area_summation(sizes, **truth)
    y += np.random.RandomState(0).normal(0, 0.01, y.shape)
    fit = es.fit_dog_area_summation(sizes, y)
    assert fit['sigma_c'] == pytest.approx(truth['sigma_c'], abs=5)
    assert fit['r2'] > 0.99
    # surround is the weaker, broader term -> looser tolerance
    assert fit['sigma_s'] == pytest.approx(truth['sigma_s'], rel=0.4)


def test_fit_needs_enough_points():
    with pytest.raises(ValueError, match='at least 5'):
        es.fit_dog_area_summation([40, 80, 120], [1.0, 2.0, 3.0])


def test_fit_ignores_non_finite_points():
    sizes = np.repeat([20, 40, 80, 160, 320, 640], 2).astype(float)
    y = es.dog_area_summation(sizes, 1.0, 40.0, 0.5, 300.0, 0.0)
    y[0] = np.nan
    fit = es.fit_dog_area_summation(sizes, y)
    assert np.isfinite(fit['sigma_c']) and fit['r2'] > 0.99


def test_count_spikes_in_window_uses_open_interval():
    # MATLAB: x > prePts & x < prePts + stimPts
    spikes = [np.array([0.1, 0.25, 0.3, 0.49, 0.5, 0.6])]
    got = es.count_spikes_in_window(spikes, pre_s=0.25, stim_s=0.25)
    assert got.tolist() == [2]  # 0.3 and 0.49; 0.25 and 0.5 are excluded
    assert es.count_spikes_in_window([np.array([])], 0.25, 0.25).tolist() == [0]


@pytest.fixture
def summary_frame():
    return pd.DataFrame({
        'cell_label': ['Cell1', 'Cell2', 'Cell2', 'Cell3'],
        'protocol': ['SingleSpot', 'ExpandingSpots', 'ExpandingSpots', 'ExpandingSpots'],
        'recording_technique': ['cell-attached', 'cell-attached', 'cell-attached', 'whole-cell'],
        'duration_minutes': [1.0, 0.5, 2.0, 3.0],
        'block_id': [10, 11, 12, 13],
    })


def test_find_block_picks_longest_for_requested_cell(summary_frame):
    assert es.find_expanding_spots_block(summary_frame, 'Cell2') == (12, 'Cell2')


def test_find_block_falls_back_when_cell_lacks_protocol(summary_frame, capsys):
    block_id, cell = es.find_expanding_spots_block(summary_frame, 'Cell1')
    assert (block_id, cell) == (12, 'Cell2')
    assert 'no cell-attached ExpandingSpots' in capsys.readouterr().out


def test_find_block_skips_whole_cell(summary_frame):
    """Cell3's block is whole-cell, so it must never be chosen for spike counting."""
    assert es.find_expanding_spots_block(summary_frame, None)[1] == 'Cell2'


def test_find_block_raises_when_nothing_matches(summary_frame):
    only_whole_cell = summary_frame[summary_frame['recording_technique'].eq('whole-cell')]
    with pytest.raises(ValueError, match='no cell-attached ExpandingSpots'):
        es.find_expanding_spots_block(only_whole_cell, None)


def test_psth_integrates_to_spike_count():
    """A rate in Hz integrated over the window must equal the spike count.

    This is the tightest available check on the spikeTimeToPSTH port: the
    Gaussian kernel integrates to 1, so area under the PSTH is spikes.
    """
    spikes = [np.array([0.30, 0.35, 0.40]), np.array([0.30, 0.50])]
    sizes = np.array([100.0])
    spot_size = np.array([100.0, 100.0])
    psth, t = es.mean_psth_by_spot_size(spikes, spot_size, sizes,
                                        epoch_duration_s=1.0, psth_sigma_ms=10.0)
    assert psth.shape == (1, 1000)
    area = psth[0].sum() / 1000.0          # Hz * s = spikes
    assert area == pytest.approx(2.5, abs=0.02)   # mean of 3 and 2 spikes


def test_psth_kernel_width_affects_peak_not_area():
    spikes = [np.array([0.5])]
    kw = dict(spot_size=np.array([50.0]), sizes=np.array([50.0]), epoch_duration_s=1.0)
    narrow, _ = es.mean_psth_by_spot_size(spikes, psth_sigma_ms=2.0, **kw)
    wide, _ = es.mean_psth_by_spot_size(spikes, psth_sigma_ms=20.0, **kw)
    assert narrow.max() > wide.max()
    assert narrow.sum() == pytest.approx(wide.sum(), rel=0.01)


def test_psth_empty_spikes_is_all_zero():
    psth, _ = es.mean_psth_by_spot_size([np.array([])], np.array([10.0]),
                                        np.array([10.0]), epoch_duration_s=0.5)
    assert psth.shape == (1, 500) and not psth.any()


def test_result_summary_string():
    res = es.ExpandingSpotsResult(
        exp_name='X', block_id=1, cell_label='Cell1', cell_type='RGC\\OFF-parasol',
        spot_sizes=np.array([40.0, 80.0]), epoch_spot_size=np.array([40.0, 80.0]),
        epoch_counts=np.array([1, 5]), mean_counts=np.array([1.0, 5.0]),
        sem_counts=np.array([0.0, 0.0]), scale=5.0,
        fit={'Kc': 1.0, 'sigma_c': 42.0, 'Ks': 0.5, 'sigma_s': 150.0, 'base': 0.0, 'r2': 0.9},
        spike_times_s=[np.array([0.3]), np.array([0.3])], pre_s=0.25, stim_s=0.25,
        epoch_duration_s=0.75, sample_rate=1e4)
    assert res.n_epochs == 2
    s = res.summary()
    assert 'sigma_c = 42.0 um' in s and 'sigma_s = 150.0 um' in s
