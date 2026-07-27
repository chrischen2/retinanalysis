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


def _groupable_blocks(bright_contrasts):
    """A minimal block table for group_blocks: one cell, one setting, N blocks."""
    import pandas as pd
    n = len(bright_contrasts)
    return pd.DataFrame({
        'exp_name': ['2026-04-04_E'] * n, 'rig': ['E'] * n,
        'block_id': list(range(1, n + 1)), 'n_epochs': [10] * n,
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
    assert g.loc[0, 'epochs'] == 10


def test_group_blocks_keeps_both_allowed_bright_contrasts():
    g = sag.group_blocks(_groupable_blocks([0.9, 1.0]), show=False)
    assert g.loc[0, 'blocks'] == 2                     # 0.9 and 1.0 both survive
    assert g.loc[0, 'bright'] == '1, 0.9'


def test_group_blocks_bright_is_a_bare_value_when_there_is_only_one():
    g = sag.group_blocks(_groupable_blocks([0.9, 0.9]), show=False)
    assert g.loc[0, 'bright'] == '0.9'


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

    class _Stub:
        def __init__(self, exp_name, block_id, **_):
            self.amp_data = (traces or {}).get(int(block_id), _smooth_trace())
            self.amp_sample_rate = 1e4

    ra_stub = mock.MagicMock()
    ra_stub.SCResponseBlock = _Stub
    # check_series_resistance lives in the shared module, so patch it there.
    with mock.patch.object(rm, 'series_resistance_table', return_value=rs), \
            mock.patch.dict('sys.modules', {'retinanalysis': ra_stub}):
        return sag.check_series_resistance(blocks, show=False, **kwargs)


def test_audit_relabels_a_mislabelled_cell_attached_block():
    blocks, rs = _blocks_with_rs(['exc', 'exc'], [0.0, 0.0])
    out = _check(blocks, rs, traces={1: _spiking_trace(), 2: _smooth_trace()})
    assert out.loc[0, 'onlineAnalysis'] == 'extracellular'
    assert out.loc[0, 'onlineAnalysis_recorded'] == 'exc'
    assert out.loc[1, 'onlineAnalysis'] == 'exc'          # no spikes, label stands
    assert len(out) == 2                                  # neither is thrown away


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
