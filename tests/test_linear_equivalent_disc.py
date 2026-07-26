"""Tests for SCutils.protocols.linear_equivalent_disc — pure helpers only."""
import numpy as np
import pandas as pd
import pytest

from retinanalysis.SCutils.protocols import linear_equivalent_disc as led


# --- stimulus tag mapping --------------------------------------------------

@pytest.mark.parametrize('tag, expected', [
    ('image', 'image'),
    ('intensity', 'disc'),
    ('linConeIntensity', 'cone_disc'),      # LinearEquivalentDiscConeLin spelling
    ('lin cone intensity', 'cone_disc'),    # Annulus / LinearEquivalentDisc spelling
    ('  image  ', 'image'),
    ('something else', ''),
])
def test_category_of_handles_both_spellings(tag, expected):
    assert led.category_of(tag) == expected


# --- NLI -------------------------------------------------------------------

def test_compute_nli_matches_the_matlab_formula():
    img, disc = np.array([10.0, 4.0]), np.array([5.0, 8.0])
    expected = (img - disc) / (np.abs(img) + np.abs(disc))
    assert np.allclose(led.compute_nli(img, disc, threshold=0.0), expected)


def test_compute_nli_is_zero_below_threshold():
    """Neither response clears the threshold, so the index is noise -> 0."""
    assert led.compute_nli([1.0], [0.5], threshold=3.0).tolist() == [0.0]


def test_compute_nli_keeps_patches_where_one_response_clears_threshold():
    out = led.compute_nli([10.0], [0.0], threshold=3.0)
    assert out.tolist() == [1.0]


def test_compute_nli_drops_non_finite():
    out = led.compute_nli([0.0, 10.0], [0.0, 5.0], threshold=0.0)
    assert out.size == 1 and out[0] == pytest.approx(1 / 3)


def test_compute_nli_bounds():
    rng = np.random.RandomState(0)
    img, disc = rng.uniform(0, 100, 200), rng.uniform(0, 100, 200)
    out = led.compute_nli(img, disc, threshold=0.0)
    assert np.all(out >= -1) and np.all(out <= 1)


def test_nli_sign_means_image_preferred():
    assert led.compute_nli([10.0], [2.0], 0.0)[0] > 0     # image drove the cell more
    assert led.compute_nli([2.0], [10.0], 0.0)[0] < 0


def test_threshold_table_matches_matlab():
    assert led.NLI_THRESHOLD == {'extracellular': 3.0, 'exc': 10.0, 'inh': 5.0}


# --- protocol handling -----------------------------------------------------

def test_stimulus_site_from_protocol_name():
    assert led.stimulus_site('LinearEquivalentAnnulus') == 'surround'
    assert led.stimulus_site('LinearEquivalentDiscConeLin') == 'center'
    assert led.stimulus_site('LinearEquivalentDisc') == 'center'


def test_only_linear_equivalent_disc_needs_the_filter():
    """The other two protocols always carry linearizeCones, so they are never filtered."""
    assert led.NEEDS_LINEARIZE_FILTER == ('LinearEquivalentDisc',)
    assert set(led.PROTOCOLS) == {'LinearEquivalentDiscConeLin', 'LinearEquivalentAnnulus',
                                  'LinearEquivalentDisc'}


def test_record_key_is_hdf5_safe_and_separates_sites():
    a = led.record_key('2026-05-08_E', 'Cell4', 'extracellular', 'center', 0.0, 0.5)
    b = led.record_key('2026-05-08_E', 'Cell4', 'extracellular', 'surround', 0.0, 0.5)
    assert a != b
    for key in (a, b):
        assert '/' not in key and '.' not in key
    assert 'FWNaN' in led.record_key('x', 'C1', 'exc', 'center', float('nan'), 0.15)


def test_shared_light_helpers_are_reused():
    from retinanalysis.SCutils.protocols import spot_annular_grating as sag
    for name in ('light_level_rstar', 'light_setting', 'apply_rstar_mapping'):
        assert getattr(led, name) is getattr(sag, name)
