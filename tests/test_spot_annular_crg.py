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


# --- the flashed protocol's data hygiene, ported ---------------------------

def test_more_shared_helpers_are_reused_not_copied():
    """The two protocols must not drift apart on calibration or raw views."""
    from retinanalysis.SCutils.protocols import spot_annular_grating as sag
    for name in ('center_spot', 'round_rstar', 'max_rstar', 'rig_of', 'is_calibrated',
                 'load_raw', 'plot_raw_blocks', 'plot_raw_epochs'):
        assert getattr(crg, name) is getattr(sag, name)
    for name in ('RSTAR_LEVELS', 'ALLOWED_FILTER_WHEEL', 'ALLOWED_BRIGHT_CONTRAST',
                 'MIN_BAR_WIDTH', 'MIN_EPOCHS'):
        assert getattr(crg, name) == getattr(sag, name)


def test_config_keys_do_not_duplicate_bar_width():
    """barWidth moved into the flashed protocol's CONFIG_KEYS; adding it again would
    make the list misdescribe itself."""
    assert crg.CONFIG_KEYS.count('barWidth') == 1
    assert 'temporalFrequency' in crg.CONFIG_KEYS


def _crg_blocks(freqs, bar_widths=None, n_epochs=None, ndf=0.0):
    """A minimal CRG block table for group_blocks."""
    import pandas as pd
    n = len(freqs)
    bar_widths = [100.0] * n if bar_widths is None else list(bar_widths)
    n_epochs = [20] * n if n_epochs is None else list(n_epochs)
    return pd.DataFrame({
        'exp_name': ['2026-04-23_E'] * n, 'rig': ['E'] * n,
        'block_id': list(range(1, n + 1)), 'n_epochs': n_epochs,
        'cell_label': ['Cell1'] * n, 'cell_type_short': ['OFF-parasol'] * n,
        'onlineAnalysis': ['extracellular'] * n, 'grating_site': ['center'] * n,
        'center_spot': ['none'] * n, 'temporalFrequency': list(freqs),
        'bar_width': bar_widths, 'brightBarContrast': [0.9] * n,
        'filter_wheel_ndf': [ndf] * n, 'backgroundIntensity': [0.5] * n,
        'has_filter_wheel': [True] * n, 'light_setting': ['FW0/bg0.5'] * n,
        'light_level': ['15000R*'] * n, 'rstar': [15000.0] * n,
        'rstar_level': [15000.0] * n, 'rstar_measured': [True] * n,
        'apertureDiameter': [0.0] * n, 'annulusInnerDiameter': [0.0] * n,
        'annulusOuterDiameter': [300.0] * n,
    })


def test_temporal_frequency_splits_groups_but_bar_width_does_not():
    """F1/F2 are measured at the reversal frequency, so it can never be pooled;
    bar width is pooled by analyze_group, so a group may span several."""
    g = crg.group_blocks(_crg_blocks([2.0, 4.0]), show=False)
    assert len(g) == 2                                  # one per frequency
    g = crg.group_blocks(_crg_blocks([4.0, 4.0], bar_widths=[100.0, 150.0]), show=False)
    assert len(g) == 1 and g.loc[0, 'bar_width'] == '100, 150'


def test_crg_group_blocks_applies_the_shared_filters():
    # bar width
    g = crg.group_blocks(_crg_blocks([4.0, 4.0], bar_widths=[50.0, 100.0]), show=False)
    assert g.loc[0, 'bar_width'] == '100'
    # min epochs, on the pooled group
    thin = crg.group_blocks(_crg_blocks([4.0], n_epochs=[10]), show=False)
    assert thin.empty
    pooled = crg.group_blocks(_crg_blocks([4.0, 4.0], n_epochs=[10, 10]), show=False)
    assert len(pooled) == 1 and pooled.loc[0, 'epochs'] == 20
    # filter wheel
    assert crg.group_blocks(_crg_blocks([4.0], ndf=3.0), show=False).empty


def test_allowed_temporal_frequency_can_restrict():
    g = crg.group_blocks(_crg_blocks([2.0, 4.0]), show=False,
                         allowed_temporal_frequency=(4.0,))
    assert len(g) == 1 and g.loc[0, 'temporalFrequency'] == 4.0


def test_crg_group_keys_carry_the_temporal_frequency():
    blocks = _crg_blocks([2.0, 4.0])
    g = crg.group_blocks(blocks, show=False)
    keys = crg.group_keys(g)
    assert len(set(keys)) == 2
    assert any('2Hz' in k for k in keys) and any('4Hz' in k for k in keys)


def test_crg_prune_records_refuses_an_empty_keep_set():
    with pytest.raises(ValueError, match='empty keep set'):
        crg.prune_records([])


def test_crg_refresh_rstar_fills_a_blank_rig_from_the_experiment_name():
    import pandas as pd
    stored = pd.DataFrame({
        'exp_name': ['2026-04-23_E', '2026-06-04_G'],
        'rig': [np.nan, ''],
        'ndf': [0.0, 1.0], 'background_intensity': [0.5, 0.3],
        'rstar': [np.nan, np.nan], 'light_level': ['(?R*)', '(?R*)'],
    })
    out = crg.refresh_rstar(stored)
    assert out['rig'].tolist() == ['E', 'G']
    assert out.loc[0, 'rstar'] == pytest.approx(15000.0)
    assert out.loc[1, 'rstar'] == pytest.approx(2310.0)
    assert out['rstar_level'].tolist() == [15000.0, 2000.0]


# --- multi-recording overlay -----------------------------------------------

def _crg_record(rstar=2000.0, tf=2.0, amp=30.0, contrasts=(-1.0, -0.75, -0.5, -0.25, 0.0)):
    """A record dict with an F2 curve rising linearly with |dark contrast|."""
    c = np.asarray(contrasts, dtype=float)
    f2 = amp * np.abs(c)
    return {'exp_name': '2026-06-04_G', 'cell_label': 'Cell2',
            'online_analysis': 'extracellular', 'grating_site': 'center',
            'light_level': f'{rstar:g}R*', 'units': 'rate difference (Hz)',
            'rstar': rstar, 'temporal_frequency': tf, 'n_epochs': 21,
            'crossing_interp': -0.5, 'dark_contrasts': c, 'f2_mean': f2,
            'f1_mean': f2 * 0.3, 'resp_mean': f2 * 0.1,
            'resp_sem': np.zeros_like(c), 'baseline_mean': 0.0}


def test_crg_overlay_plots_the_harmonic_not_the_half_cycle_difference():
    long = crg.tuning_overlay([_crg_record(amp=30.0)])
    assert np.allclose(long.sort_values('dark_contrast')['rel'].to_numpy(),
                       [30.0, 22.5, 15.0, 7.5, 0.0])


def test_crg_overlay_does_not_subtract_a_baseline():
    """An amplitude is already a modulation depth, so there is nothing to subtract."""
    rec = _crg_record()
    rec['baseline_mean'] = 12.0          # would shift every point if it were used
    long = crg.tuning_overlay([rec])
    assert long['rel'].max() == pytest.approx(30.0)


def test_crg_overlay_normalizes_at_the_deepest_contrast():
    long = crg.tuning_overlay([_crg_record(amp=30.0), _crg_record(amp=90.0)])
    assert long.groupby('position')['ref_amplitude'].first().tolist() == [30.0, 90.0]
    at_ref = long[long['dark_contrast'].eq(-1.0)]['norm']
    assert np.allclose(at_ref.to_numpy(), 1.0)
    # Three-fold apart in absolute F2, identical once normalized.
    a, b = [sub.sort_values('dark_contrast')['norm'].to_numpy()
            for _, sub in long.groupby('position')]
    assert np.allclose(a, b)


def test_crg_overlay_can_still_show_the_half_cycle_difference():
    long = crg.tuning_overlay([_crg_record()], harmonic='resp_mean',
                              subtract_baseline=True)
    assert long['rel'].max() == pytest.approx(3.0)


def test_crg_overlay_carries_the_temporal_frequency_for_labelling():
    long = crg.tuning_overlay([_crg_record(tf=2.0), _crg_record(tf=4.0)])
    assert long.groupby('position')['temporal_frequency'].first().tolist() == [2.0, 4.0]


def test_crg_overlay_rejects_an_array_the_record_does_not_have():
    with pytest.raises(KeyError, match='f3_mean'):
        crg.tuning_overlay([_crg_record()], harmonic='f3_mean')
