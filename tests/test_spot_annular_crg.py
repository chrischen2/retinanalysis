"""Tests for SCutils.protocols.spot_annular_crg.

Pure helpers only — no database or SSD. The shared light-level / cone-model
helpers are tested in test_spot_annular_grating.py; this file covers what is
specific to the reversing protocol.
"""
import numpy as np
import pytest

from retinanalysis.SCutils.protocols import spot_annular_crg as crg


# --- harmonics -------------------------------------------------------------

def test_harmonics_recover_a_pure_tone():
    sr, f = 1000.0, 4.0
    t = np.arange(0, 2.0, 1 / sr)
    f1, f2 = crg.harmonic_amplitudes(3.0 * np.sin(2 * np.pi * f * t), sr, f)
    assert f1 == pytest.approx(3.0, abs=1e-6)
    assert f2 == pytest.approx(0.0, abs=1e-6)


def test_harmonics_recover_the_second_harmonic():
    sr, f = 1000.0, 4.0
    t = np.arange(0, 2.0, 1 / sr)
    f1, f2 = crg.harmonic_amplitudes(3.0 * np.sin(2 * np.pi * 2 * f * t), sr, f)
    assert f1 == pytest.approx(0.0, abs=1e-6)
    assert f2 == pytest.approx(3.0, abs=1e-6)


def test_harmonics_ignore_a_dc_offset():
    sr, f = 1000.0, 4.0
    t = np.arange(0, 2.0, 1 / sr)
    base = 2.0 * np.sin(2 * np.pi * f * t)
    assert (crg.harmonic_amplitudes(base + 50.0, sr, f)
            == pytest.approx(crg.harmonic_amplitudes(base, sr, f)))


def test_harmonics_of_empty_trace_are_nan():
    f1, f2 = crg.harmonic_amplitudes(np.array([]), 1000.0, 4.0)
    assert np.isnan(f1) and np.isnan(f2)


# --- cycle folding ---------------------------------------------------------

def test_fold_cycles_averages_identical_cycles():
    sr, f = 1000.0, 4.0            # 250 samples per cycle
    one = np.arange(250.0)
    folded = crg.fold_cycles(np.tile(one, 8), sr, f)
    assert folded.size == 250
    assert np.allclose(folded, one)


def test_fold_cycles_drops_the_first_cycle():
    sr, f = 1000.0, 4.0
    odd = np.concatenate([np.full(250, 99.0), np.tile(np.zeros(250), 3)])
    assert crg.fold_cycles(odd, sr, f, drop_cycles=1).max() == 0.0
    assert crg.fold_cycles(odd, sr, f, drop_cycles=0).max() > 0.0


def test_fold_cycles_keeps_all_cycles_when_drop_would_empty_it():
    sr, f = 1000.0, 4.0
    single = np.ones(250)
    assert np.allclose(crg.fold_cycles(single, sr, f, drop_cycles=1), 1.0)


def test_fold_cycles_handles_a_partial_trailing_cycle():
    sr, f = 1000.0, 4.0
    trace = np.concatenate([np.ones(250), np.ones(250), np.ones(120)])
    folded = crg.fold_cycles(trace, sr, f)
    assert folded.size == 250 and np.allclose(folded, 1.0)


# --- stimulus frames -------------------------------------------------------

def test_two_phases_swap_which_bars_are_bright():
    kw = dict(aperture_diameter=0, annulus_inner_diameter=0, annulus_outer_diameter=800,
              bar_width=100, background_intensity=0.5, spot_intensity=0.0,
              bright_bar_contrast=0.9, dark_bar_contrast=-0.5)
    a, b, extent = crg.stimulus_frames(**kw)
    assert not np.allclose(a, b)
    # Same set of intensities on screen in both phases -- the reason this
    # protocol has no bright/dark cancellation null.
    assert sorted(np.unique(np.round(a, 6))) == sorted(np.unique(np.round(b, 6)))


def test_phase_a_puts_the_bright_bar_set_at_the_dark_peak():
    bg, ap, an = 0.5, 0.9, 0.5
    a, b, extent = crg.stimulus_frames(0, 0, 800, 100, bg, 0.0, ap, -an)
    # both frames contain both peaks, just in swapped positions
    for frame in (a, b):
        assert frame.min() == pytest.approx(bg * (1 - an))
        assert frame.max() == pytest.approx(bg * (1 + ap))


def test_frames_respect_the_annulus_mask():
    bg = 0.3
    a, b, extent = crg.stimulus_frames(0, 400, 800, 100, bg, 0.0, 0.9, -1.0)
    n = a.shape[0]
    g = np.linspace(-extent, extent, n)
    r = np.hypot(*np.meshgrid(g, g))
    assert np.allclose(a[r < 199], bg) and np.allclose(b[r < 199], bg)
    assert np.allclose(a[r > 401], bg) and np.allclose(b[r > 401], bg)


# --- record key ------------------------------------------------------------

def test_record_key_includes_temporal_frequency():
    two = crg.record_key('2026-04-23_E', 'Cell1', 'extracellular', 'center', 2.0, 0.0, 0.5)
    four = crg.record_key('2026-04-23_E', 'Cell1', 'extracellular', 'center', 4.0, 0.0, 0.5)
    assert two != four
    assert '2Hz' in two and '4Hz' in four
    for key in (two, four):
        assert '/' not in key and '.' not in key


def test_record_key_handles_nan_filter_wheel():
    key = crg.record_key('x', 'Cell1', 'exc', 'center', 4.0, float('nan'), 0.15)
    assert 'FWNaN' in key and '.' not in key


# --- shared helpers are the same objects, not copies -----------------------

def test_shared_helpers_are_reused_from_the_flashed_module():
    from retinanalysis.SCutils.protocols import spot_annular_grating as sag
    for name in ('light_level_rstar', 'cone_predict_dark_contrast', 'grating_site',
                 'apply_rstar_mapping', 'select_canonical', 'weber_curve'):
        assert getattr(crg, name) is getattr(sag, name)
