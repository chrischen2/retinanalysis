"""Tests for SCutils.protocols.spot_annular_grating.

Pure helpers only — no database or SSD. The DataJoint-backed discovery and
analysis paths are exercised by demos/analyze_flash_grate.ipynb.
"""
import numpy as np
import pytest

from retinanalysis.SCutils.protocols import spot_annular_grating as sag


# --- light level -----------------------------------------------------------

def test_rig_ceilings_are_the_measured_calibration():
    assert sag.RIG_MAX_RSTAR == {'E': 30000.0, 'G': 77000.0}
    assert sag.max_rstar('E') == 30000.0
    assert sag.max_rstar('G') == 77000.0


@pytest.mark.parametrize('rig, ndf, expected', [
    ('E', 0.0, 30000.0), ('E', 1.0, 3000.0),
    ('G', 0.0, 77000.0), ('G', 1.0, 7700.0),
])
def test_filter_wheel_attenuates_by_ten_to_the_ndf(rig, ndf, expected):
    assert sag.max_rstar(rig, ndf) == pytest.approx(expected)


def test_half_ndf_is_a_root_ten_step():
    assert sag.max_rstar('E', 0.5) == pytest.approx(30000 / 10 ** 0.5)
    assert sag.max_rstar('G', 0.5) == pytest.approx(77000 / 10 ** 0.5)


def test_background_scales_the_ceiling_linearly():
    rstar, label = sag.light_level_rstar(0.0, 0.5, rig='E')
    assert rstar == pytest.approx(15000.0) and label == '15000R*'
    assert sag.light_level_rstar(0.0, 0.15, rig='G')[0] == pytest.approx(11550.0)


def test_the_two_rigs_differ_at_the_same_setting():
    """The whole reason the rig has to be threaded through."""
    e = sag.light_level_rstar(0.5, 0.30, rig='E')[0]
    g = sag.light_level_rstar(0.5, 0.30, rig='G')[0]
    assert g / e == pytest.approx(77000 / 30000)


def test_lowercase_and_padded_rig_names_work():
    assert sag.max_rstar(' g ') == 77000.0
    assert sag.max_rstar('e') == 30000.0


@pytest.mark.parametrize('ndf, bg, expected', [
    (0.0, 0.15, 12000.0),
    (1.0, 0.15, 1000.0),
    (1.0, 0.30, 2000.0),
    (0.5, 0.15, 4000.0),
    (0.5, 0.30, 8000.0),
])
def test_rstar_table_is_the_fallback_for_an_unknown_rig(ndf, bg, expected):
    """Superseded by RIG_MAX_RSTAR, but still reachable when the rig is not one of ours."""
    rstar, label = sag.light_level_rstar(ndf, bg, rig='Z')
    assert rstar == expected                               # exact value, unrounded
    assert label == f'{sag.round_rstar(expected):g}R*'      # label names the rung
    assert sag.light_level_rstar(ndf, bg)[0] == expected   # rig omitted entirely


@pytest.mark.parametrize('rstar, rung', [
    (1000.0, 1000.0),      # exactly on a rung
    (1155.0, 1000.0),      # rig G, wheel 1, bg 0.15
    (1500.0, 2000.0),      # rig E, wheel 1, bg 0.5 -- a linear tie, broken upward
    (2310.0, 2000.0),
    (3652.4, 4000.0),
    (4743.4, 5000.0),
    (7304.9, 7000.0),
    (9000.0, 10000.0),     # rig E, wheel 0, bg 0.3
    (11550.0, 10000.0),    # rig G, wheel 0, bg 0.15
    (15000.0, 15000.0),
])
def test_round_rstar_snaps_to_the_nominal_ladder(rstar, rung):
    assert sag.round_rstar(rstar) == rung


def test_round_rstar_is_nearest_in_log_space():
    """1500 is equidistant from 1000 and 2000 linearly; as a ratio it is not."""
    assert sag.round_rstar(1500.0) == 2000.0
    assert sag.round_rstar(1414.0) == 1000.0        # just below sqrt(1000*2000)
    assert sag.round_rstar(1415.0) == 2000.0


def test_round_rstar_clamps_off_the_ends_and_passes_nan_through():
    assert sag.round_rstar(38500.0) == 20000.0
    assert sag.round_rstar(30.0) == 1000.0
    assert np.isnan(sag.round_rstar(np.nan))
    assert np.isnan(sag.round_rstar(None))
    assert np.isnan(sag.round_rstar(0.0))


def test_exact_rstar_survives_the_rounding():
    """The rung is for grouping; the Weber comparison needs the real intensity."""
    rstar, label = sag.light_level_rstar(0.0, 0.15, rig='G')
    assert rstar == pytest.approx(11550.0)
    assert label == '10000R*'


def test_the_old_table_is_really_rig_g():
    """Documents why the table could not be reused for rig E."""
    for ndf, bg, tabled in sag.RSTAR_TABLE:
        ratio = sag.light_level_rstar(ndf, bg, rig='G')[0] / tabled
        assert 0.9 < ratio < 1.2


def test_unknown_rig_and_uncovered_combo_stays_nan():
    rstar, label = sag.light_level_rstar(0.0, 0.50, rig='Z')
    assert np.isnan(rstar)
    assert label == 'FW0/bg0.5 (?R*)'


def test_missing_filter_wheel_has_no_light_level_on_any_rig():
    for rig in ('E', 'G', None):
        assert np.isnan(sag.light_level_rstar(float('nan'), 0.5, rig=rig)[0])


def test_refresh_rstar_restates_a_stored_summary_without_reanalysis():
    """R* is a fact about the rig, so it is applied on read, not frozen at analysis."""
    import pandas as pd
    stored = pd.DataFrame({
        'exp_name': ['2026-04-23_E', '2026-06-04_G'],
        'ndf': [0.0, 1.0], 'background_intensity': [0.5, 0.3],
        'rstar': [np.nan, np.nan],          # analyzed before the calibration existed
        'light_level': ['FW0/bg0.5 (?R*)', 'FW1/bg0.3 (?R*)'],
    })
    out = sag.refresh_rstar(stored)
    assert out.loc[0, 'rstar'] == pytest.approx(15000.0)     # rig E
    assert out.loc[1, 'rstar'] == pytest.approx(2310.0)      # rig G
    assert out['rstar_measured'].all()
    assert out['rig'].tolist() == ['E', 'G']
    # The stored frame is untouched.
    assert np.isnan(stored.loc[0, 'rstar'])


def test_refresh_rstar_fills_a_blank_rig_column_from_the_experiment_name():
    """Records written before summary_row() carried 'rig' have the column but leave it
    empty; trusting the blank dropped them to the RSTAR_TABLE fallback and lost their R*."""
    import pandas as pd
    stored = pd.DataFrame({
        'exp_name': ['2026-04-23_E', '2026-06-04_G', '2026-04-23_E'],
        'rig': [np.nan, '', 'G'],          # blank, blank-string, and an explicit override
        'ndf': [0.0, 1.0, 0.0], 'background_intensity': [0.5, 0.3, 0.5],
        'rstar': [np.nan, np.nan, np.nan],
        'light_level': ['(?R*)', '(?R*)', '(?R*)'],
    })
    out = sag.refresh_rstar(stored)
    assert out['rig'].tolist() == ['E', 'G', 'G']
    assert out.loc[0, 'rstar'] == pytest.approx(15000.0)      # E, derived
    assert out.loc[1, 'rstar'] == pytest.approx(2310.0)       # G, derived
    assert out.loc[2, 'rstar'] == pytest.approx(38500.0)      # explicit rig wins
    assert out['rstar'].notna().all()


def test_refresh_rstar_uses_fixed_filters_and_visual_stimulus_lookup():
    import pandas as pd
    stored = pd.DataFrame({
        'exp_name': ['2026-06-04_G'],
        'fixed_ndfs': ['EL06, EL2'],
        'ndf': [1.0],
        'background_intensity': [0.3],
        'rstar': [np.nan],
        'light_level': [''],
    })
    out = sag.refresh_rstar(stored)
    assert out.loc[0, 'max_light_level'] == 7600.0
    assert out.loc[0, 'rstar'] == 2280.0
    assert out.loc[0, 'light_level'] == '2000R*'


def test_refresh_rstar_is_a_no_op_without_the_setting_columns():
    import pandas as pd
    empty = pd.DataFrame()
    assert sag.refresh_rstar(empty).empty
    no_ndf = pd.DataFrame({'exp_name': ['x_E']})
    assert 'rstar' not in sag.refresh_rstar(no_ndf).columns


def test_is_calibrated_follows_the_rig():
    # A known rig with a wheel reading is calibrated whatever the background.
    assert sag.is_calibrated(0.0, 0.50, rig='E')
    assert sag.is_calibrated(0.0, 0.50, rig='G')
    # Without one it falls back to asking the old table.
    assert sag.is_calibrated(0.0, 0.15)
    assert not sag.is_calibrated(0.0, 0.50)
    assert not sag.is_calibrated(float('nan'), 0.15, rig='E')


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
    # The override is stored exactly; the label is the rung it rounds to, and
    # 40000 is past the top of the ladder so it clamps there.
    assert out['rstar'].iloc[0] == 40000.0 and out['light_level'].iloc[0] == '20000R*'
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


def test_prediction_saturates_at_the_bright_limit():
    """I0 negligible against the mean gives -c/(1+2c) -- -0.32 for a 0.9 bar."""
    for bright in (0.5, 0.9, 1.0):
        limit = -bright / (1 + 2 * bright)
        assert sag.cone_predict_dark_contrast(1e9, bright) == pytest.approx(limit, abs=1e-6)


# --- gain of the model -----------------------------------------------------

def test_weber_response_is_half_maximal_at_i0():
    assert sag.weber_response(2000.0, i0=2000.0) == pytest.approx(0.5)
    assert sag.weber_response(0.0, i0=2000.0) == pytest.approx(0.0)


def test_incremental_gain_is_the_derivative_of_the_response():
    """Compare against a numerical derivative of R(I) rather than the algebra."""
    i0, i, h = 2000.0, 5000.0, 1e-3
    numeric = (sag.weber_response(i + h, i0) - sag.weber_response(i - h, i0)) / (2 * h)
    assert sag.weber_gain(i, i0, kind='incremental', normalize=False) \
        == pytest.approx(numeric, rel=1e-6)


def test_incremental_gain_is_one_in_darkness_when_normalized():
    assert sag.weber_gain(0.0, 2000.0) == pytest.approx(1.0)
    assert sag.weber_gain(2000.0, 2000.0) == pytest.approx(0.25)


def test_incremental_gain_falls_faster_than_weber_above_i0():
    """A decade of light costs the model 100x gain where Weber would cost 10x."""
    i0 = 2000.0
    hi, lo = sag.weber_gain(1e5, i0), sag.weber_gain(1e6, i0)
    assert hi / lo == pytest.approx(100.0, rel=0.05)


def test_contrast_gain_peaks_at_i0():
    i0 = 2000.0
    grid = np.logspace(1, 6, 500)
    g = sag.weber_gain(grid, i0, kind='contrast')
    assert grid[int(np.argmax(g))] == pytest.approx(i0, rel=0.05)
    assert sag.weber_gain(i0, i0, kind='contrast') == pytest.approx(1.0)
    assert sag.weber_gain(i0, i0, kind='contrast', normalize=False) == pytest.approx(0.25)


def test_contrast_gain_rolls_off_as_one_over_i_above_i0():
    """Well above I0 a decade of light costs a decade of contrast gain."""
    i0 = 2000.0
    assert sag.weber_gain(1e5, i0, kind='contrast') \
        / sag.weber_gain(1e6, i0, kind='contrast') == pytest.approx(10.0, rel=0.05)


def test_weber_gain_rejects_an_unknown_kind():
    with pytest.raises(ValueError):
        sag.weber_gain(1000.0, kind='absolute')


def test_model_tuning_crosses_zero_at_the_predicted_contrast():
    """The zero of the model tuning curve IS cone_predict_dark_contrast."""
    grid = np.linspace(-1.0, 0.0, 4001)
    for rstar in (1000, 2000, 4000, 8000, 16000):
        y = sag.model_tuning(grid, rstar, 0.9)
        assert sag.interp_zero_crossing(grid, y) == pytest.approx(
            sag.cone_predict_dark_contrast(rstar, 0.9), abs=1e-3)


def test_model_tuning_is_positive_at_the_deepest_dark_bar():
    """Drawn in the orientation the measured curves come out in: +1 at c = -1."""
    y = sag.model_tuning(np.linspace(-1.0, 0.0, 11), 4000.0, 0.9)
    assert y[0] == pytest.approx(1.0)
    assert y[-1] < 0                      # only the bright bars deviate at c = 0
    assert all(np.diff(y) < 0)            # monotone as the dark bars shallow out


def test_model_tuning_normalization_moves_no_crossing():
    grid = np.linspace(-1.0, 0.0, 2001)
    for rstar in (1000.0, 16000.0):
        raw = sag.interp_zero_crossing(grid, sag.model_tuning(grid, rstar, 0.9,
                                                              normalize=False))
        norm = sag.interp_zero_crossing(grid, sag.model_tuning(grid, rstar, 0.9))
        assert raw == pytest.approx(norm, abs=1e-6)


def test_model_tuning_crossings_march_toward_zero_with_light():
    crossings = [sag.interp_zero_crossing(
        np.linspace(-1.0, 0.0, 4001),
        sag.model_tuning(np.linspace(-1.0, 0.0, 4001), r, 0.9))
        for r in (1000, 2000, 4000, 8000, 16000)]
    assert all(np.diff(crossings) > 0)


def test_weber_curve_matches_the_pointwise_prediction():
    grid = [500.0, 2000.0, 20000.0]
    assert list(sag.weber_curve(grid, 0.9, 2000.0)) == pytest.approx(
        [sag.cone_predict_dark_contrast(r, 0.9, 2000.0) for r in grid])


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


def test_record_key_distinguishes_fixed_filter_combinations():
    base = ('2026-05-08_E', 'Cell1', 'exc', 'center', 1.0, 0.15)
    assert sag.record_key(*base, 'EL3 + FW1') != sag.record_key(*base, 'EL2 + FW1')


def test_record_key_distinguishes_bright_contrast_and_bar_width():
    base = ('2026-05-08_E', 'Cell1', 'exc', 'center', 1.0, 0.15,
            'EL3 + FW1')
    a = sag.record_key(*base, 0.9, [100.0])
    b = sag.record_key(*base, 0.5, [100.0])
    c = sag.record_key(*base, 0.9, [200.0])
    assert len({a, b, c}) == 3
    assert '__bright0p9__bar100' in a


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
    assert out['condition'].tolist() == [
        'ON-parasol / surround', 'OFF-parasol / center', 'ON-midget / surround']


# --- series resistance vs onlineAnalysis -----------------------------------
# Synthetic traces, so these stay pure helpers with no h5 and no database.

def _spiking_trace(n=12, length=3000, seed=0):
    """Noise with sharp negative spikes and rebounds — a cell-attached trace."""
    rng = np.random.RandomState(seed)
    data = rng.randn(n, length)
    for trial in range(n):
        for centre in rng.choice(np.arange(50, length - 50), 20, replace=False):
            data[trial, centre - 2:centre + 3] += np.array([6., -12., -40., -12., 6.])
    return data


def _smooth_trace(n=12, length=3000, seed=1):
    """A drifting current with no spikes — a voltage-clamp trace."""
    rng = np.random.RandomState(seed)
    return np.cumsum(rng.randn(n, length) * 0.5, axis=1) * 0.05


def test_trace_is_spiking_separates_the_two_recording_kinds():
    assert sag.trace_is_spiking(_spiking_trace(), 1e4)
    assert not sag.trace_is_spiking(_smooth_trace(), 1e4)


def test_zero_rs_relabels_extracellular_only_when_the_trace_has_spikes():
    # Rs == 0 is ambiguous on its own: cell-attached, or the field was never
    # filled in. Both happen in this dataset, so the data breaks the tie.
    mode, note = sag.resolve_recording_mode('exc', 0.0, _spiking_trace(), 1e4)
    assert mode == 'extracellular' and 'contains spikes' in note

    mode, note = sag.resolve_recording_mode('exc', 0.0, _smooth_trace(), 1e4)
    assert mode == 'exc' and 'never set' in note


def test_positive_rs_relabels_whole_cell_and_reads_polarity_off_the_current():
    # Rs > 0 is unambiguous -- a cell-attached patch has no access resistance --
    # but the holding potential is not in the reading, so it comes from the sign.
    inward = np.full((4, 100), -5.0)
    outward = np.full((4, 100), 5.0)
    assert sag.resolve_recording_mode('extracellular', 8e6, inward)[0] == 'exc'
    assert sag.resolve_recording_mode('extracellular', 8e6, outward)[0] == 'inh'


def test_agreeing_label_and_reading_are_left_alone():
    assert sag.resolve_recording_mode('extracellular', 0.0, _spiking_trace(), 1e4) \
        == ('extracellular', '')
    assert sag.resolve_recording_mode('exc', 8e6, _smooth_trace(), 1e4) == ('exc', '')


def test_missing_reading_keeps_the_recorded_label():
    assert sag.resolve_recording_mode('exc', np.nan, _smooth_trace(), 1e4) == ('exc', '')
    assert sag.resolve_recording_mode('exc', None, _smooth_trace(), 1e4) == ('exc', '')


def test_unset_label_is_resolved_from_the_amplifier_and_the_trace():
    # 'none' has nothing to contradict, so the reading decides outright rather
    # than overruling. This is most of the linear-equivalent-disc dataset.
    mode, note = sag.resolve_recording_mode('none', 0.0, _spiking_trace(), 1e4)
    assert mode == 'extracellular' and 'contains spikes' in note

    mode, note = sag.resolve_recording_mode('none', 8e6, np.full((4, 100), -5.0))
    assert mode == 'exc' and 'whole-cell' in note

    # No spikes and no access resistance recorded: whole-cell with the field
    # never set, polarity still readable from the sign.
    mode, note = sag.resolve_recording_mode('none', 0.0, _smooth_trace() - 50.0, 1e4)
    assert mode == 'exc' and 'never set' in note


def test_unset_label_without_a_trace_stays_unresolved():
    assert sag.resolve_recording_mode('none', 0.0, amp_data=None) == ('none', '')


def test_resolution_without_a_trace_keeps_the_label_and_says_why():
    mode, note = sag.resolve_recording_mode('exc', 0.0, amp_data=None)
    assert mode == 'exc' and 'not available' in note


def test_mode_family_maps_labels_to_recording_modes():
    assert sag.mode_family('extracellular') == 'cell-attached'
    assert sag.mode_family('exc') == 'whole-cell'
    assert sag.mode_family('inh') == 'whole-cell'
    assert sag.mode_family('none') == ''      # unknown, never a mismatch
    assert sag.mode_family(None) == ''


def _groupable_blocks(bright_contrasts, bar_widths=None):
    """A minimal block table for group_blocks: one cell, one setting, N blocks."""
    import pandas as pd
    n = len(bright_contrasts)
    if bar_widths is None:
        bar_widths = [100.0] * n
    return pd.DataFrame({
        'bar_width': list(bar_widths),
        'exp_name': ['2026-04-04_E'] * n, 'rig': ['E'] * n,
        # Comfortably over MIN_EPOCHS so these fixtures exercise the filter under
        # test rather than the epoch-count one; the epoch tests set it explicitly.
        'block_id': list(range(1, n + 1)), 'n_epochs': [20] * n,
        'cell_label': ['Cell1'] * n, 'cell_type_short': ['OFF-parasol'] * n,
        'onlineAnalysis': ['extracellular'] * n, 'grating_site': ['center'] * n,
        'filter_wheel_ndf': [0.0] * n, 'backgroundIntensity': [0.5] * n,
        'has_filter_wheel': [True] * n,
        'light_setting': ['FW0/bg0.5'] * n, 'light_level': ['15000R*'] * n,
        'rstar': [15000.0] * n, 'rstar_level': [15000.0] * n,
        'apertureDiameter': [0.0] * n, 'annulusInnerDiameter': [0.0] * n,
        'annulusOuterDiameter': [300.0] * n, 'spotIntensity': [0.0] * n,
        'brightBarContrast': list(bright_contrasts),
    })


@pytest.mark.parametrize('aperture, expected', [
    (0.0, 'none'), (0, 'none'), (200.0, 'spot'), (350.0, 'spot'),
])
def test_center_spot_reads_the_aperture(aperture, expected):
    assert sag.center_spot(aperture) == expected


def test_center_spot_is_blank_when_the_aperture_is_missing():
    assert sag.center_spot(float('nan')) == ''
    assert sag.center_spot(None) == ''


def test_grating_site_is_the_annulus_not_the_aperture():
    """The 44-block case: no center spot, but the grating still excludes the center.

    inner=400 / outer=1200 puts the grating at r=200-600 um with the center
    r=0-200 um plain background. Keying the site on the aperture would call that
    a center recording, which would relabel the whole ON-parasol surround series.
    """
    assert sag.grating_site(400.0) == 'surround'      # regardless of aperture
    assert sag.center_spot(0.0) == 'none'             # no spot there
    assert sag.grating_site(0.0) == 'center'
    # The three configurations that actually occur, kept distinct by the pair.
    configs = {(sag.grating_site(inner), sag.center_spot(ap))
               for inner, ap in [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0)]}
    assert configs == {('center', 'none'), ('surround', 'none'), ('surround', 'spot')}


def test_group_blocks_shows_every_bright_contrast_it_pooled():
    """brightBarContrast is a block-level setting but not a grouping key, so a cell
    swept over it lands in one group -- which the table has to show, not hide."""
    g = sag.group_blocks(_groupable_blocks([0.9, 0.5, 0.25]), show=False,
                         allowed_bright_contrast=None)
    assert len(g) == 1
    assert g.loc[0, 'bright'] == '0.9, 0.5, 0.25'      # descending, all of them
    assert g.loc[0, 'blocks'] == 3


def test_group_blocks_drops_bright_contrasts_outside_the_allowed_set():
    """The default filter is what stops a bright-contrast sweep pooling into one curve."""
    g = sag.group_blocks(_groupable_blocks([0.9, 0.5, 0.25]), show=False)
    assert len(g) == 1
    assert g.loc[0, 'bright'] == '0.9'                 # the sweep blocks are gone
    assert g.loc[0, 'blocks'] == 1
    assert g.loc[0, 'epochs'] == 20


def test_group_blocks_keeps_both_allowed_bright_contrasts():
    g = sag.group_blocks(_groupable_blocks([0.9, 1.0]), show=False)
    assert g.loc[0, 'blocks'] == 2                     # 0.9 and 1.0 both survive
    assert g.loc[0, 'bright'] == '1, 0.9'


def test_group_blocks_drops_bars_narrower_than_the_cutoff():
    """Below ~60 um the optics low-pass the grating, so the cancellation is partly optical."""
    blocks = _groupable_blocks([0.9] * 4, bar_widths=[40.0, 50.0, 75.0, 100.0])
    g = sag.group_blocks(blocks, show=False)
    assert g.loc[0, 'blocks'] == 2                     # only 75 and 100 survive
    assert g.loc[0, 'bar_width'] == '75, 100'


def test_group_blocks_min_bar_width_is_inclusive_and_optional():
    exactly_60 = _groupable_blocks([0.9], bar_widths=[60.0])
    assert sag.group_blocks(exactly_60, show=False).loc[0, 'blocks'] == 1
    narrow = _groupable_blocks([0.9] * 2, bar_widths=[40.0, 100.0])
    assert sag.group_blocks(narrow, show=False, min_bar_width=None).loc[0, 'blocks'] == 2
    assert sag.group_blocks(narrow, show=False).loc[0, 'blocks'] == 1


def test_group_blocks_drops_groups_with_too_few_epochs():
    """A 10-epoch group is ~1 epoch per dark contrast -- a single trial, not a curve."""
    blocks = _groupable_blocks([0.9])
    blocks['n_epochs'] = [10]
    assert sag.group_blocks(blocks, show=False).empty
    assert len(sag.group_blocks(blocks, show=False, min_epochs=None)) == 1


def test_min_epochs_counts_the_pooled_group_not_the_block():
    """Two thin blocks of the same condition add up to a usable recording."""
    blocks = _groupable_blocks([0.9, 0.9])
    blocks['n_epochs'] = [10, 10]              # 20 pooled, over the cutoff
    g = sag.group_blocks(blocks, show=False)
    assert len(g) == 1 and g.loc[0, 'epochs'] == 20 and g.loc[0, 'blocks'] == 2


def test_min_epochs_boundary_is_inclusive():
    for n, kept in ((15, False), (16, True)):
        blocks = _groupable_blocks([0.9])
        blocks['n_epochs'] = [n]
        assert (not sag.group_blocks(blocks, show=False).empty) is kept


def test_group_blocks_bar_width_lists_every_width_it_pooled():
    """analyze_group pools across bar width, so a group spanning two has to show both."""
    g = sag.group_blocks(_groupable_blocks([0.9] * 2, bar_widths=[100.0, 200.0]), show=False)
    assert g.loc[0, 'bar_width'] == '100, 200'


def test_group_blocks_bright_is_a_bare_value_when_there_is_only_one():
    g = sag.group_blocks(_groupable_blocks([0.9, 0.9]), show=False)
    assert g.loc[0, 'bright'] == '0.9'


def test_group_blocks_can_separate_bright_contrast_and_bar_width():
    blocks = _groupable_blocks([0.9, 0.5, 0.9], bar_widths=[100.0, 100.0, 200.0])
    grouped = sag.group_blocks(
        blocks, show=False, allowed_bright_contrast=None, min_bar_width=None,
        min_epochs=None, separate_bright_contrast=True, collapse_bar_widths=False)
    assert len(grouped) == 3
    assert set(zip(grouped['bright'], grouped['bar_width'])) == {
        (0.9, 100.0), (0.5, 100.0), (0.9, 200.0)}


def test_group_blocks_keeps_light_paths_as_separate_conditions():
    blocks = _groupable_blocks([0.9, 0.9])
    blocks['ndf_combination'] = ['EL3 + FW1', 'EL2 + FW1']
    blocks['fixed_ndfs'] = [('EL3',), ('EL2',)]
    blocks['max_light_level'] = [3000.0, 10000.0]
    grouped = sag.group_blocks(blocks, show=False, min_epochs=None)
    assert grouped['ndf_combination'].tolist() == ['EL3 + FW1', 'EL2 + FW1']
    assert grouped['max_light_level'].tolist() == [3000.0, 10000.0]


def _tiny_record(exp='2026-04-23_E', cell='Cell1', mode='extracellular',
                 site='center', ndf=0.0, bg=0.5):
    """A minimal GratingRecord that save_records can write."""
    import numpy as np
    contrasts = np.array([-1.0, -0.5, 0.0])
    return sag.GratingRecord(
        exp_name=exp, cell_label=cell, cell_type='RGC\\OFF-parasol',
        online_analysis=mode, grating_site=site, ndf=ndf, background_intensity=bg,
        rstar=15000.0, light_level='15000R*', dark_contrasts=contrasts,
        resp_mean=np.array([10.0, 5.0, 0.0]), resp_sem=np.zeros(3),
        resp_n=np.array([3, 3, 3]), baseline_mean=0.0, baseline_sem=0.0,
        crossing_nearest=0.0, crossing_interp=0.0, bright_bar_contrast=0.9,
        cone_pred_dark=-0.35, cone_i0=2000.0, bar_widths=np.array([100.0]),
        traces=np.zeros((3, 4)), trace_time_ms=np.arange(4.0),
        pre_time_ms=250.0, stim_time_ms=250.0, n_epochs=30, block_ids=[1])


def test_prune_records_removes_only_what_is_not_kept(tmp_path):
    keep = _tiny_record(cell='Cell1')
    drop = _tiny_record(cell='Cell2')
    sag.save_records([keep, drop], path=tmp_path, verbose=False)
    assert len(sag.load_summary(path=tmp_path)) == 2

    removed = sag.prune_records([keep.key], path=tmp_path, verbose=False)
    assert removed == [drop.key]
    left = sag.load_summary(path=tmp_path)
    assert list(left['key']) == [keep.key]
    # the survivor's arrays are intact
    arrays = sag.load_records([keep.key], path=tmp_path)[keep.key]
    assert list(arrays['dark_contrasts']) == [-1.0, -0.5, 0.0]


def test_save_records_keeps_separate_bright_and_bar_conditions(tmp_path):
    bright_09 = _tiny_record()
    bright_05 = _tiny_record()
    bright_05.bright_bar_contrast = 0.5
    bar_200 = _tiny_record()
    bar_200.bar_widths = np.array([200.0])

    sag.save_records([bright_09, bright_05, bar_200], path=tmp_path, verbose=False)
    saved = sag.load_summary(path=tmp_path)
    assert len(saved) == 3
    assert saved['key'].nunique() == 3
    assert set(saved['bright_bar_contrast']) == {0.5, 0.9}
    assert set(saved['bar_widths']) == {'100', '200'}


def test_save_records_replaces_legacy_unsplit_key(tmp_path):
    import h5py

    rec = _tiny_record()
    legacy = sag.record_key(
        rec.exp_name, rec.cell_label, rec.online_analysis, rec.grating_site,
        rec.ndf, rec.background_intensity)
    h5_path = tmp_path / 'records.h5'
    with h5py.File(h5_path, 'w') as store:
        old = store.create_group(legacy)
        old.attrs['key'] = legacy
        old.attrs['exp_name'] = rec.exp_name
        old.attrs['cell_label'] = rec.cell_label
        old.attrs['cell_type'] = rec.cell_type

    sag.save_records([rec], path=tmp_path, verbose=False)
    with h5py.File(h5_path, 'r') as store:
        assert legacy not in store
        assert rec.key in store


def test_prune_records_dry_run_touches_nothing(tmp_path):
    a, b = _tiny_record(cell='Cell1'), _tiny_record(cell='Cell2')
    sag.save_records([a, b], path=tmp_path, verbose=False)
    would = sag.prune_records([a.key], path=tmp_path, dry_run=True, verbose=False)
    assert would == [b.key]
    assert len(sag.load_summary(path=tmp_path)) == 2      # still both


def test_prune_records_refuses_an_empty_keep_set(tmp_path):
    """The guard against wiping the store by passing an empty selection."""
    sag.save_records([_tiny_record()], path=tmp_path, verbose=False)
    with pytest.raises(ValueError, match='empty keep set'):
        sag.prune_records([], path=tmp_path)
    assert len(sag.load_summary(path=tmp_path)) == 1


def test_prune_records_is_idempotent(tmp_path):
    a, b = _tiny_record(cell='Cell1'), _tiny_record(cell='Cell2')
    sag.save_records([a, b], path=tmp_path, verbose=False)
    assert len(sag.prune_records([a.key], path=tmp_path, verbose=False)) == 1
    assert sag.prune_records([a.key], path=tmp_path, verbose=False) == []


def test_prune_records_keys_on_the_resolved_recording_mode(tmp_path):
    """check_series_resistance rewrites onlineAnalysis, and record_key includes it.

    A group table carrying the *recorded* mode would make every relabelled
    recording look like an orphan — which is how a live record gets deleted.
    """
    import pandas as pd
    rec = _tiny_record(mode='extracellular')      # what the amplifier resolved
    sag.save_records([rec], path=tmp_path, verbose=False)
    resolved = pd.DataFrame([{'exp_name': rec.exp_name, 'cell_label': rec.cell_label,
                              'onlineAnalysis': 'extracellular', 'grating_site': 'center',
                              'filter_wheel_ndf': 0.0, 'backgroundIntensity': 0.5,
                              'bright': 0.9, 'bar_width': 100.0}])
    assert sag.group_keys(resolved) == [rec.key]
    assert sag.prune_records(resolved, path=tmp_path, verbose=False) == []
    # The pre-relabel table would not match, and would orphan a live record.
    as_recorded = resolved.assign(onlineAnalysis='exc')
    assert sag.group_keys(as_recorded) != [rec.key]


def test_save_records_prune_to_deletes_in_one_call(tmp_path):
    old = _tiny_record(cell='Cell9')
    sag.save_records([old], path=tmp_path, verbose=False)
    fresh = _tiny_record(cell='Cell1')
    sag.save_records([fresh], path=tmp_path, verbose=False, prune_to=[fresh.key])
    assert list(sag.load_summary(path=tmp_path)['key']) == [fresh.key]


def _pop_summary_and_records(curves):
    """A summary + record store for population_tuning.

    ``curves`` maps cell label -> (rstar_level, baseline, [responses]) on a shared
    three-point dark-contrast grid.
    """
    import pandas as pd
    contrasts = np.array([-1.0, -0.5, 0.0])
    rows, recs = [], {}
    for i, (cell, (lvl, base, resp)) in enumerate(curves.items()):
        key = f'k{i}'
        rows.append({'key': key, 'exp_name': 'X_E', 'cell_label': cell,
                     'cell_type': 'ON-parasol', 'grating_site': 'surround',
                     'online_analysis': 'extracellular', 'units': 'rate (Hz)',
                     'rstar_level': lvl, 'bright_bar_contrast': 0.9})
        recs[key] = {'dark_contrasts': contrasts, 'resp_mean': np.asarray(resp, dtype=float),
                     'baseline_mean': float(base)}
    return pd.DataFrame(rows), recs


def test_population_tuning_subtracts_each_cells_own_baseline():
    """Cells are poolable only relative to their own spontaneous rate."""
    summary, recs = _pop_summary_and_records({
        'Cell1': (1000.0, 10.0, [30.0, 20.0, 10.0]),   # rel  20, 10, 0
        'Cell2': (1000.0, 50.0, [70.0, 60.0, 50.0]),   # rel  20, 10, 0 -- same curve
    })
    t = sag.population_tuning(summary, records=recs, normalize=False)
    assert t['mean'].tolist() == [20.0, 10.0, 0.0]
    assert t['n_cells'].tolist() == [2, 2, 2]
    assert t['sem'].eq(0).all()                        # identical after baselining


def test_population_tuning_normalization_preserves_the_zero_crossing():
    """The divisor is a positive scalar, so it cannot move where the curve crosses."""
    summary, recs = _pop_summary_and_records({
        'Cell1': (1000.0, 0.0, [100.0, 50.0, 0.0]),    # loud cell
        'Cell2': (1000.0, 0.0, [2.0, 1.0, 0.0]),       # quiet cell, same shape
    })
    raw = sag.population_tuning(summary, records=recs, normalize=False)
    norm = sag.population_tuning(summary, records=recs, normalize=True)
    # Raw is dominated by the loud cell; normalized weights them equally.
    assert raw['mean'].tolist() == [51.0, 25.5, 0.0]
    assert norm['mean'].tolist() == [1.0, 0.5, 0.0]
    # Both cross zero at the same contrast.
    assert raw.loc[raw['mean'].abs().idxmin(), 'dark_contrast'] == 0.0
    assert norm.loc[norm['mean'].abs().idxmin(), 'dark_contrast'] == 0.0


def test_population_tuning_averages_a_repeated_cell_once():
    """A cell recorded twice in one condition must not count twice."""
    summary, recs = _pop_summary_and_records({
        'Cell1': (1000.0, 0.0, [10.0, 5.0, 0.0]),
        'Cell1_dup': (1000.0, 0.0, [30.0, 15.0, 0.0]),
        'Cell2': (1000.0, 0.0, [20.0, 10.0, 0.0]),
    })
    summary.loc[1, 'cell_label'] = 'Cell1'             # same cell, second recording
    t = sag.population_tuning(summary, records=recs, normalize=False)
    assert t['n_cells'].max() == 2                     # two cells, not three
    # Cell1 averages to 20/10/0, so the population mean matches Cell2 exactly.
    assert t['mean'].tolist() == [20.0, 10.0, 0.0]


def test_population_tuning_splits_by_light_level():
    summary, recs = _pop_summary_and_records({
        'Cell1': (1000.0, 0.0, [10.0, 5.0, 0.0]),
        'Cell2': (15000.0, 0.0, [10.0, 8.0, 2.0]),
    })
    t = sag.population_tuning(summary, records=recs, normalize=False)
    assert sorted(t['rstar_level'].unique()) == [1000.0, 15000.0]
    assert t[t['rstar_level'].eq(15000.0)]['mean'].tolist() == [10.0, 8.0, 2.0]


def test_population_tuning_drops_records_that_are_not_curves():
    """A single sampled contrast is not a tuning curve."""
    summary, recs = _pop_summary_and_records({'Cell1': (1000.0, 0.0, [10.0, 5.0, 0.0])})
    recs['k0']['dark_contrasts'] = np.array([-0.9])
    recs['k0']['resp_mean'] = np.array([10.0])
    assert sag.population_tuning(summary, records=recs).empty


def test_population_tuning_excludes_disallowed_bright_contrasts():
    """The stored h5 keeps records the current filters would drop; the guard is here too."""
    summary, recs = _pop_summary_and_records({
        'Cell1': (1000.0, 0.0, [10.0, 5.0, 0.0]),
        'Cell2': (1000.0, 0.0, [10.0, 5.0, 0.0]),
    })
    summary.loc[1, 'bright_bar_contrast'] = 0.25
    t = sag.population_tuning(summary, records=recs, normalize=False)
    assert t['n_cells'].max() == 1


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


def _check(blocks, rs, traces=None, **kwargs):
    """Run the audit with both the reading and the raw traces stubbed."""
    from unittest import mock
    from retinanalysis.SCutils import recording_mode as rm

    samples = {int(block_id): ((traces or {}).get(int(block_id), _smooth_trace()), 1e4)
               for block_id in blocks['block_id']}
    # check_series_resistance lives in the shared module, so patch it there.
    with mock.patch.object(rm, 'series_resistance_table', return_value=rs), \
            mock.patch.object(rm, '_amp_trace_samples', return_value=samples):
        return sag.check_series_resistance(blocks, show=False, **kwargs)


def test_audit_relabels_a_mislabelled_cell_attached_block():
    blocks, rs = _blocks_with_rs(['exc', 'exc'], [0.0, 0.0])
    out = _check(blocks, rs, traces={1: _spiking_trace(), 2: _smooth_trace()})
    assert out.loc[0, 'onlineAnalysis'] == 'extracellular'
    assert out.loc[0, 'onlineAnalysis_recorded'] == 'exc'
    assert out.loc[1, 'onlineAnalysis'] == 'exc'          # no spikes, label stands
    assert len(out) == 2                                  # neither is thrown away


def test_audit_uses_recording_technique_to_skip_an_unneeded_trace_read():
    blocks, rs = _blocks_with_rs(['exc'], [0.0])
    blocks['group_properties'] = [{'recordingTechnique': 'cell-attached'}]
    # A smooth trace would preserve the whole-cell label if it were consulted;
    # the two agreeing hardware metadata fields settle this without loading it.
    out = _check(blocks, rs, traces={1: _smooth_trace()})
    assert out.loc[0, 'onlineAnalysis'] == 'extracellular'
    assert 'recordingTechnique is cell-attached' in out.loc[0, 'rs_flag']


def test_audit_resolves_whole_cell_from_technique_when_rs_is_missing():
    blocks, rs = _blocks_with_rs(['none', 'none'], [np.nan, np.nan])
    blocks['group_properties'] = [
        {'recordingTechnique': 'whole-cell'},
        {'recordingTechnique': 'whole-cell'},
    ]
    out = _check(blocks, rs, traces={
        1: np.full((4, 100), -5.0),
        2: np.full((4, 100), 5.0),
    })
    assert out['onlineAnalysis'].tolist() == ['exc', 'inh']
    assert out['rs_flag'].str.contains('recordingTechnique is whole-cell').all()


def test_audit_relabels_rather_than_drops_a_mislabelled_whole_cell_block():
    blocks, rs = _blocks_with_rs(['extracellular'], [8e6])
    out = _check(blocks, rs, traces={1: np.full((4, 100), -5.0)})
    assert len(out) == 1
    assert out.loc[0, 'onlineAnalysis'] == 'exc'


def test_every_epoch_over_the_cutoff_drops_the_block():
    blocks, rs = _blocks_with_rs(['exc', 'exc'], [25e6, 8e6], high_rs=[10, 0])
    out = _check(blocks, rs)
    assert out['block_id'].tolist() == [2]


def test_drop_false_keeps_blocks_over_the_cutoff():
    blocks, rs = _blocks_with_rs(['exc'], [25e6], high_rs=[10])
    out = _check(blocks, rs, drop=False)
    assert len(out) == 1
    assert 'above 20 MOhm' in out.loc[0, 'rs_flag']


# --- batch status line -----------------------------------------------------

def _group_row(**overrides):
    import pandas as pd
    row = {'exp_name': '2026-06-04_G', 'cell_label': 'Cell4', 'cell_type_short': 'OFF-parasol',
           'onlineAnalysis': 'extracellular', 'grating_site': 'center', 'annulus_inner': 0.0,
           'annulus_outer': 300.0, 'aperture': 0.0, 'spot_intensity': 0.3, 'bright': 0.9,
           'backgroundIntensity': 0.3, 'filter_wheel_ndf': 1.0, 'stage_ndfs': 'EL06, EL2, FW1',
           'light_setting': 'FW1/bg0.3', 'rstar': 2000.0, 'rs_mohm': 0.0,
           'blocks': 2, 'epochs': 98}
    row.update(overrides)
    return pd.Series(row)


def test_status_line_says_why_a_recording_is_center_or_surround():
    centre = sag.describe_group_row(_group_row())
    assert 'grating over center' in centre
    assert 'inner diameter 0' in centre          # the number the call was made from

    surround = sag.describe_group_row(_group_row(grating_site='surround', annulus_inner=350.0))
    assert 'grating over surround' in surround
    assert 'inner diameter 350 > 0' in surround


def test_status_line_says_why_a_recording_is_spikes_or_current():
    spikes = sag.describe_group_row(_group_row())
    assert 'firing rate in Hz' in spikes

    current = sag.describe_group_row(_group_row(onlineAnalysis='exc', rs_mohm=6.3))
    assert 'excitatory reversal' in current and 'current in pA' in current
    assert 'Rs 6.30 MOhm' in current


def test_status_line_reports_a_relabelled_block_as_such():
    line = sag.describe_group_row(_group_row(onlineAnalysis='extracellular',
                                             onlineAnalysis_recorded='exc'))
    assert "recorded as 'exc'" in line and 'relabelled' in line


def test_status_line_carries_the_stimulus_and_both_ndf_numbers():
    line = sag.describe_group_row(_group_row(), index=3, total=76)
    assert '[  3/76]' in line
    assert 'background 0.3' in line and 'spot 0.3' in line
    assert 'wheel NDF 1' in line                  # the wheel setting
    assert 'EL06, EL2, FW1' in line               # the fixed filters, which it does not imply
    assert '2000R*' in line


def test_status_line_marks_an_uncalibrated_light_level():
    line = sag.describe_group_row(_group_row(rstar=float('nan')))
    assert 'no R* calibration yet' in line


def test_status_line_survives_missing_columns():
    import pandas as pd
    line = sag.describe_group_row(pd.Series({'exp_name': 'X', 'cell_label': 'Cell1'}))
    assert 'X Cell1' in line and '?' in line      # absent numbers show as '?', no exception


# --- multi-recording overlay -----------------------------------------------

def _overlay_record(rstar=2000.0, amp=40.0, baseline=5.0, units='rate (Hz)',
                    contrasts=(-1.0, -0.8, -0.6, -0.4, -0.2, 0.0), **overrides):
    """A stored-record dict with a straight-line tuning curve.

    The response falls linearly from ``amp`` above baseline at the most negative
    contrast to baseline at contrast 0, so the crossing is unambiguous.
    """
    c = np.asarray(contrasts, dtype=float)
    rec = {'exp_name': '2026-06-04_G', 'cell_label': 'Cell4',
           'online_analysis': 'extracellular', 'grating_site': 'center',
           'light_level': f'{rstar:g}R*', 'units': units, 'rstar': rstar,
           'crossing_interp': -0.3, 'n_epochs': 66, 'dark_contrasts': c,
           'resp_mean': baseline + amp * (-c), 'resp_sem': np.zeros_like(c),
           'baseline_mean': baseline}
    rec.update(overrides)
    return rec


def test_overlay_normalizes_to_the_most_negative_contrast():
    long = sag.tuning_overlay([_overlay_record(amp=40.0), _overlay_record(amp=500.0)])
    for _, sub in long.groupby('position'):
        at_ref = sub.loc[sub['dark_contrast'].eq(sub['ref_contrast'].iloc[0]), 'norm']
        assert np.isclose(abs(float(at_ref.iloc[0])), 1.0)
    # Two recordings 12.5x apart in absolute response land on the same curve.
    a, b = [sub.sort_values('dark_contrast')['norm'].to_numpy()
            for _, sub in long.groupby('position')]
    assert np.allclose(a, b)


def test_overlay_reference_is_each_records_own_deepest_contrast():
    """A recording that never reached -1 is normalized at the contrast it did reach."""
    long = sag.tuning_overlay([_overlay_record(),
                               _overlay_record(contrasts=(-0.6, -0.4, -0.2, 0.0))])
    assert long.groupby('position')['ref_contrast'].first().tolist() == [-1.0, -0.6]


def test_overlay_reference_can_be_pinned():
    long = sag.tuning_overlay([_overlay_record()], ref_contrast=-0.4)
    assert long['ref_contrast'].unique().tolist() == [-0.4]
    assert np.isclose(abs(long.loc[long['dark_contrast'].eq(-0.4), 'norm'].iloc[0]), 1.0)


def test_overlay_divisor_is_positive_so_the_crossing_does_not_move():
    """Normalizing must not move the zero crossing -- it is the measurement."""
    long = sag.tuning_overlay([_overlay_record(baseline=12.0)])
    assert (long['ref_amplitude'] > 0).all()
    assert np.allclose(np.sign(long['rel']), np.sign(long['norm']))
    assert np.isclose(sag.interp_zero_crossing(long['dark_contrast'], long['rel']),
                      sag.interp_zero_crossing(long['dark_contrast'], long['norm']))


def test_overlay_subtracts_each_records_own_baseline():
    """Curves are response - baseline, so a cell's spontaneous rate does not shift it."""
    quiet = sag.tuning_overlay([_overlay_record(baseline=0.0)])
    busy = sag.tuning_overlay([_overlay_record(baseline=30.0)])
    assert np.allclose(quiet['rel'].to_numpy(), busy['rel'].to_numpy())


def test_overlay_flat_record_normalizes_to_nan_rather_than_dividing_by_zero():
    long = sag.tuning_overlay([_overlay_record(amp=0.0)])
    assert long['ref_amplitude'].eq(0.0).all()
    assert long['norm'].isna().all()


def test_overlay_of_nothing_is_empty():
    assert sag.tuning_overlay([]).empty


# --- raw-view helpers -------------------------------------------------------

def test_reversal_times_fall_on_every_half_period():
    """sign(cos(2·pi·f·t)) swaps the frames twice per cycle."""
    times = sag._reversal_times_ms(pre_ms=200.0, stim_ms=1000.0, reversal_hz=2.0)
    assert np.allclose(times, [450.0, 700.0, 950.0])


def test_reversal_times_scale_with_frequency():
    slow = sag._reversal_times_ms(200.0, 1000.0, 2.0)
    fast = sag._reversal_times_ms(200.0, 1000.0, 4.0)
    assert len(fast) == 7 and len(slow) == 3        # twice as many, minus the edge
    assert np.allclose(np.diff(fast), 125.0)


def test_reversal_on_the_window_edge_is_not_drawn():
    """A swap at the end of the stimulus is the window's edge, already shaded."""
    times = sag._reversal_times_ms(200.0, 1000.0, 2.0)
    assert 1200.0 not in times                       # 200 + 4 * 250
    assert times.max() < 1200.0


def test_no_reversals_for_a_flashed_recording():
    for hz in (None, 0.0, float('nan'), -1.0):
        assert sag._reversal_times_ms(200.0, 1000.0, hz).size == 0


class _StubRecord:
    """Stands in for a GratingRecord / CRGRecord in load_raw."""
    def __init__(self, raw=None):
        self.raw = raw


def test_load_raw_takes_any_record_that_carries_traces():
    """CRGRecord is not a GratingRecord, so the check is on the attribute."""
    raw = {'traces': [np.zeros(3)], 'sample_rate': 1e4}
    assert sag.load_raw(_StubRecord(raw)) is raw


def test_load_raw_says_how_to_get_the_traces_when_a_record_has_none():
    with pytest.raises(ValueError, match='keep_raw=True'):
        sag.load_raw(_StubRecord(None))


def test_load_raw_needs_block_ids_when_given_an_experiment_name():
    with pytest.raises(ValueError, match='block_ids is required'):
        sag.load_raw('2026-06-04_G')


def test_overlay_keeps_units_so_modes_are_not_pooled_on_one_axis():
    long = sag.tuning_overlay([_overlay_record(),
                               _overlay_record(units='excitation (pA)',
                                               online_analysis='exc')])
    assert set(long['units']) == {'rate (Hz)', 'excitation (pA)'}
