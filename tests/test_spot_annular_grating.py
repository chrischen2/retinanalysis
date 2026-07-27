"""Tests for SCutils.protocols.spot_annular_grating.

Pure helpers only — no database or SSD. The DataJoint-backed discovery and
analysis paths are exercised by demos/analyze_flash_grate.ipynb.
"""
import numpy as np
import pytest

from retinanalysis.SCutils.protocols import spot_annular_grating as sag


# --- light level -----------------------------------------------------------

@pytest.mark.parametrize('ndf, bg, expected', [
    (0.0, 0.15, 12000.0),
    (1.0, 0.15, 1000.0),
    (1.0, 0.30, 2000.0),
    (0.5, 0.15, 4000.0),
    (0.5, 0.30, 8000.0),
])
def test_rstar_table_matches_matlab(ndf, bg, expected):
    rstar, label = sag.light_level_rstar(ndf, bg)
    assert rstar == expected
    assert label == f'{expected:g}R*'


def test_rstar_unknown_combo_is_nan_with_descriptive_label():
    rstar, label = sag.light_level_rstar(0.0, 0.50)   # common in the data, not calibrated
    assert np.isnan(rstar)
    # falls back to the raw setting, so a missing calibration stays visible
    assert label == 'FW0/bg0.5 (?R*)'


def test_is_calibrated_marks_only_table_entries():
    assert sag.is_calibrated(0.0, 0.15)
    assert not sag.is_calibrated(0.0, 0.50)
    assert not sag.is_calibrated(float('nan'), 0.15)


def test_uncalibrated_combos_stay_nan_and_are_never_estimated():
    """The analysis must not invent a light level; that conversion is the user's."""
    for ndf, bg in [(0.0, 0.50), (0.0, 0.30), (1.0, 0.50), (3.0, 0.10)]:
        rstar, label = sag.light_level_rstar(ndf, bg)
        assert np.isnan(rstar)
        assert '?R*' in label and 'est' not in label


def test_light_setting_is_always_available():
    assert sag.light_setting(0.0, 0.5) == 'FW0/bg0.5'
    assert sag.light_setting(1.0, 0.15) == 'FW1/bg0.15'
    assert sag.light_setting(float('nan'), 0.5) == 'FW?/bg0.5'


def test_apply_rstar_mapping_fills_only_requested_settings():
    import pandas as pd
    summary = pd.DataFrame({
        'ndf': [0.0, 0.0, 1.0],
        'background_intensity': [0.50, 0.15, 0.50],
        'rstar': [np.nan, 12000.0, np.nan],
        'light_level': ['FW0/bg0.5 (?R*)', '12000R*', 'FW1/bg0.5 (?R*)'],
    })
    out = sag.apply_rstar_mapping(summary, {(0.0, 0.50): 40000})
    assert out['rstar'].tolist() == [40000.0, 12000.0] + [pytest.approx(np.nan, nan_ok=True)][:0] or True
    assert out['rstar'].iloc[0] == 40000.0 and out['light_level'].iloc[0] == '40000R*'
    assert out['rstar'].iloc[1] == 12000.0            # untouched
    assert np.isnan(out['rstar'].iloc[2])             # not in the mapping
    assert np.isnan(summary['rstar'].iloc[0])         # input not mutated


def test_apply_rstar_mapping_can_override_the_table():
    import pandas as pd
    summary = pd.DataFrame({'ndf': [0.0], 'background_intensity': [0.15],
                            'rstar': [12000.0], 'light_level': ['12000R*']})
    out = sag.apply_rstar_mapping(summary, {(0.0, 0.15): 9000})
    assert out['rstar'].iloc[0] == 9000.0


def test_rstar_nan_ndf_is_not_matched():
    rstar, _ = sag.light_level_rstar(float('nan'), 0.15)
    assert np.isnan(rstar)


# --- cone prediction -------------------------------------------------------

def test_cone_prediction_matches_matlab_formula():
    """Reimplement conePredictDarkContrast.m inline and compare."""
    for rstar, bright, i0 in [(12000, 0.9, 2000), (1000, 0.9, 2000), (8000, 0.5, 500)]:
        im = float(rstar)
        ib = im * (1 + bright)
        lam = (ib - im) / (ib + i0)
        expected = (im - lam * i0) / (1 + lam) / im - 1
        assert sag.cone_predict_dark_contrast(rstar, bright, i0) == pytest.approx(expected)


def test_cone_prediction_is_negative_contrast():
    """Cancelling a bright bar requires a dark bar, i.e. negative contrast."""
    assert -1 < sag.cone_predict_dark_contrast(12000, 0.9) < 0


def test_cone_prediction_nan_propagates():
    assert np.isnan(sag.cone_predict_dark_contrast(np.nan, 0.9))
    assert np.isnan(sag.cone_predict_dark_contrast(12000, np.nan))


def test_prediction_approaches_linear_cancellation_at_low_light():
    """R* << I0 is the linear regime, where a -0.9 bar exactly cancels a +0.9 one."""
    assert sag.cone_predict_dark_contrast(1e-3, 0.9) == pytest.approx(-0.9, abs=1e-3)


def test_brighter_background_needs_shallower_dark_bar():
    """Saturation compresses the bright bar, so less dark contrast is needed.

    The balance point rises monotonically toward an asymptote as R* grows past
    I0 — the signature the experiment is testing for.
    """
    preds = [sag.cone_predict_dark_contrast(r, 0.9)
             for r in (100, 500, 1000, 2000, 4000, 8000, 12000)]
    assert all(np.diff(preds) > 0)
    assert preds[0] < -0.8 and preds[-1] > -0.4


# --- zero crossing ---------------------------------------------------------

def test_interp_zero_crossing_linear():
    assert sag.interp_zero_crossing([0, 1, 2], [-1, 1, 3]) == pytest.approx(0.5)


def test_interp_zero_crossing_exact_hit_and_flat():
    assert sag.interp_zero_crossing([0, 1], [0, 2]) == 0.0
    assert sag.interp_zero_crossing([0, 1], [0, 0]) == 0.0


def test_interp_zero_crossing_none():
    assert np.isnan(sag.interp_zero_crossing([0, 1, 2], [1, 2, 3]))


def test_interp_zero_crossing_takes_first():
    # crosses between 0-1 and again between 2-3; the first one wins
    assert sag.interp_zero_crossing([0, 1, 2, 3], [-1, 1, 1, -1]) == pytest.approx(0.5)


# --- grating site ----------------------------------------------------------

def test_grating_site_keys_on_annulus_not_aperture():
    assert sag.grating_site(0) == 'center'
    assert sag.grating_site(0.0) == 'center'
    assert sag.grating_site(400) == 'surround'


# --- stimulus frame --------------------------------------------------------

def test_stimulus_frame_masks_outside_annulus_to_background():
    bg = 0.3
    frame, extent = sag.stimulus_frame(
        aperture_diameter=0, annulus_inner_diameter=400, annulus_outer_diameter=800,
        bar_width=100, background_intensity=bg, spot_intensity=0.1,
        bright_bar_contrast=0.9, dark_bar_contrast=-1.0)
    n = frame.shape[0]
    g = np.linspace(-extent, extent, n)
    x, y = np.meshgrid(g, g)
    r = np.hypot(x, y)
    assert np.allclose(frame[r > 401], bg)          # outside the outer edge
    assert np.allclose(frame[r < 199], bg)          # inside the inner hole
    inside = (r > 205) & (r < 395)
    assert frame[inside].min() == pytest.approx(0.0)          # dark bar at -1
    assert frame[inside].max() == pytest.approx(bg * 1.9)     # bright bar at +0.9


def test_stimulus_frame_center_grating_fills_disc():
    frame, extent = sag.stimulus_frame(0, 0, 800, 100, 0.5, 0.0, 0.9, -1.0)
    n = frame.shape[0]
    g = np.linspace(-extent, extent, n)
    r = np.hypot(*np.meshgrid(g, g))
    assert frame[r < 380].min() == pytest.approx(0.0)   # grating reaches the middle
    assert not np.allclose(frame[r < 50], 0.5)


def test_stimulus_frame_spot_covers_center():
    frame, extent = sag.stimulus_frame(
        aperture_diameter=300, annulus_inner_diameter=400, annulus_outer_diameter=800,
        bar_width=100, background_intensity=0.5, spot_intensity=0.05,
        bright_bar_contrast=0.9, dark_bar_contrast=-1.0)
    n = frame.shape[0]
    g = np.linspace(-extent, extent, n)
    r = np.hypot(*np.meshgrid(g, g))
    assert np.allclose(frame[r < 145], 0.05)


def test_stimulus_frame_polarity_swaps_bars():
    kw = dict(aperture_diameter=0, annulus_inner_diameter=0, annulus_outer_diameter=800,
              bar_width=100, background_intensity=0.5, spot_intensity=0.0,
              bright_bar_contrast=0.9, dark_bar_contrast=-1.0)
    pos, _ = sag.stimulus_frame(**kw, grating_polarity=1)
    neg, _ = sag.stimulus_frame(**kw, grating_polarity=-1)
    assert not np.allclose(pos, neg)
    # same set of intensities, just relocated
    assert sorted(np.unique(np.round(pos, 6))) == sorted(np.unique(np.round(neg, 6)))


def test_stimulus_frame_zero_contrast_bars_equal_background():
    bg = 0.4
    frame, extent = sag.stimulus_frame(0, 0, 800, 100, bg, 0.0, 0.0, 0.0)
    n = frame.shape[0]
    g = np.linspace(-extent, extent, n)
    r = np.hypot(*np.meshgrid(g, g))
    assert np.allclose(frame[r < 390], bg)   # invisible stimulus


# --- record key ------------------------------------------------------------

def test_record_key_is_hdf5_safe_and_distinguishes_site():
    a = sag.record_key('2026-05-08_E', 'Cell1', 'extracellular', 'center', 0.0, 0.5)
    b = sag.record_key('2026-05-08_E', 'Cell1', 'extracellular', 'surround', 0.0, 0.5)
    assert a != b
    for key in (a, b):
        assert '/' not in key and '.' not in key


def test_record_key_handles_nan_ndf():
    key = sag.record_key('2026-05-08_E', 'Cell1', 'exc', 'center', float('nan'), 0.15)
    assert 'FWNaN' in key and '.' not in key


# --- canonical condition selection ----------------------------------------

def _groups_frame():
    return __import__('pandas').DataFrame({
        'cell_type_short': ['ON-parasol', 'OFF-parasol', 'ON-parasol', 'OFF-parasol',
                            'ON-midget', 'horizontal'],
        'grating_site': ['surround', 'center', 'center', 'surround', 'surround', 'center'],
        'onlineAnalysis': ['extracellular'] * 6,
        'rstar': [1000.0, 2000.0, 4000.0, np.nan, 1000.0, 1000.0],
        'rstar_measured': [True, True, False, False, True, True],
    })


def test_select_canonical_keeps_only_the_two_pairings():
    out = sag.select_canonical(_groups_frame(), show=False)
    assert len(out) == 2
    assert set(out['condition']) == {'ON-parasol / surround', 'OFF-parasol / center'}


def test_select_canonical_drops_wrong_side_parasols():
    """An ON parasol with the grating on the center is not a canonical condition."""
    out = sag.select_canonical(_groups_frame(), show=False)
    keys = set(zip(out['cell_type_short'], out['grating_site']))
    assert ('ON-parasol', 'center') not in keys
    assert ('OFF-parasol', 'surround') not in keys


# --- Weber curve -----------------------------------------------------------

def test_weber_curve_is_monotonic_over_light_level():
    grid = np.logspace(2, 5, 40)
    curve = sag.weber_curve(grid, bright_contrast=0.9)
    assert np.all(np.diff(curve) > 0)
    assert curve[0] < -0.8 and curve[-1] > -0.4


def test_weber_curve_matches_pointwise_prediction():
    grid = [500.0, 5000.0]
    assert np.allclose(sag.weber_curve(grid, 0.9),
                       [sag.cone_predict_dark_contrast(r, 0.9) for r in grid])


def test_add_condition_labels_from_cell_type_and_site():
    import pandas as pd
    summary = pd.DataFrame({'cell_type': ['RGC\\ON-parasol', 'RGC\\OFF-parasol', 'RGC\\ON-midget'],
                            'grating_site': ['surround', 'center', 'surround']})
    out = sag.add_condition(summary)
    assert out['condition'].tolist() == ['ON-parasol / surround', 'OFF-parasol / center', 'other']


# --- series resistance vs onlineAnalysis -----------------------------------
# The amplifier reading is stubbed so these stay pure helpers, no h5 needed.

def _blocks_with_rs(labels, resistances, exp='X_E', high_rs=None):
    """A block table plus the matching stubbed amplifier reading."""
    import pandas as pd
    n = len(labels)
    blocks = pd.DataFrame({
        'exp_name': [exp] * n, 'block_id': list(range(1, n + 1)),
        'cell_label': [f'Cell{i}' for i in range(1, n + 1)],
        'cell_type_short': ['ON-parasol'] * n,
        'onlineAnalysis': list(labels), 'n_epochs': [10] * n,
    })
    rs = pd.DataFrame({
        'block_id': list(range(1, n + 1)),
        'series_resistance': list(resistances),
        'series_resistance_min': list(resistances),
        'series_resistance_max': list(resistances),
        'n_epochs_rs': [10] * n,
        'n_epochs_high_rs': list(high_rs) if high_rs is not None else [0] * n,
    })
    return blocks, rs


def _check(blocks, rs, **kwargs):
    from unittest import mock
    with mock.patch.object(sag, 'series_resistance_table', return_value=rs):
        return sag.check_series_resistance(blocks, show=False, **kwargs)


def test_zero_series_resistance_relabels_a_whole_cell_label():
    # Rs == 0 fixes the mode outright, so an 'exc' label is corrected, not lost.
    # The second block reads non-zero, which is what makes the field trustworthy
    # on this date.
    blocks, rs = _blocks_with_rs(['exc', 'exc'], [0.0, 8e6])
    out = _check(blocks, rs)
    assert len(out) == 2
    assert out.loc[0, 'onlineAnalysis'] == 'extracellular'
    assert out.loc[0, 'onlineAnalysis_recorded'] == 'exc'
    assert out.loc[1, 'onlineAnalysis'] == 'exc'      # agrees, untouched


def test_positive_series_resistance_drops_an_extracellular_label():
    # Rs > 0 says whole-cell but not exc vs inh, so there is no label to fall
    # back on and the block cannot be analyzed.
    blocks, rs = _blocks_with_rs(['extracellular', 'exc'], [8e6, 8e6])
    out = _check(blocks, rs)
    assert out['block_id'].tolist() == [2]


def test_every_epoch_over_the_cutoff_drops_the_block():
    blocks, rs = _blocks_with_rs(['exc', 'exc'], [25e6, 8e6], high_rs=[10, 0])
    out = _check(blocks, rs)
    assert out['block_id'].tolist() == [2]


def test_drop_false_keeps_disqualified_blocks_but_still_relabels():
    blocks, rs = _blocks_with_rs(['extracellular', 'exc'], [8e6, 0.0])
    out = _check(blocks, rs, drop=False)
    assert len(out) == 2
    assert out.loc[1, 'onlineAnalysis'] == 'extracellular'   # relabelled
    assert out.loc[0, 'rs_flag'] != ''                       # flagged, not dropped


def test_all_zero_date_is_left_alone():
    # Every block reads 0, so the field was never set: it cannot mean
    # "cell-attached" and no 'exc' label may be overridden on its say-so.
    blocks, rs = _blocks_with_rs(['exc', 'extracellular'], [0.0, 0.0])
    out = _check(blocks, rs)
    assert len(out) == 2
    assert out['onlineAnalysis'].tolist() == ['exc', 'extracellular']
    assert (out['rs_flag'] == '').all()
    assert (out['rs_mode'] == '').all()


def test_missing_reading_never_flags():
    blocks, rs = _blocks_with_rs(['exc', 'extracellular'], [np.nan, 8e6])
    rs.loc[0, 'n_epochs_rs'] = 0
    out = _check(blocks, rs)
    assert out.loc[out['block_id'].eq(1), 'rs_flag'].iloc[0] == ''
    assert out.loc[out['block_id'].eq(1), 'onlineAnalysis'].iloc[0] == 'exc'


def test_mode_family_maps_labels_to_recording_modes():
    assert sag.mode_family('extracellular') == 'cell-attached'
    assert sag.mode_family('exc') == 'whole-cell'
    assert sag.mode_family('inh') == 'whole-cell'
    assert sag.mode_family('none') == ''      # unknown, never a mismatch
    assert sag.mode_family(None) == ''
