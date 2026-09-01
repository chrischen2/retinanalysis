"""The LNK, checked against a real recording instead of a simulation.

``test_variable_mean_noise.py`` fits the LNK to a cell that ``_adapting_cell``
generated **with the LNK**, which is the right way to ask whether the fitter
recovers what it was given and the wrong way to ask whether the model survives
contact with a retina. Every failure this suite exists for was found on real
data and is invisible on synthetic: a nonlinearity that collapses to a step
function, a slow state that has nothing left to track, a fit that reports a time
constant at its bound. So these tests run on a cached recording.

**The recording is an ``.npz``, not a database query.** ``vmn.build_fixture``
loads one cell once -- DataJoint, the SSD, the MATLAB draw -- and writes the
sequence to ``SingCell_Notebooks/rodAdaptation/fixtures/``. Everything here
reads that file in about a tenth of a second, so the loop is: edit
``fit_lnk``, run ``pytest tests/test_lnk_real_data.py -m 'not slow'``, look.
Rebuild the fixture with::

    vmn.build_fixture('2020-06-11_B', [30174], rec_type='extracellular',
                      overwrite=True)

If the file is missing every test here skips rather than failing, since it is
data and not code.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / 'SingCell_Notebooks' / 'rodAdaptation'
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

vmn = pytest.importorskip('variable_mean_noise')

FIXTURE_NAME = '2020-06-11_B_extracellular'
FIXTURE = vmn.fixture_path(FIXTURE_NAME)

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(
        not FIXTURE.exists(),
        reason=(f'no cached recording at {FIXTURE}. Build one with '
                f'vmn.build_fixture(...) -- see this module docstring.')),
]


@pytest.fixture(scope='module')
def cell():
    """The cached recording, loaded once for the module."""
    return vmn.load_fixture(FIXTURE_NAME)


# --------------------------------------------------------------------------
# The cache itself: a fixture nobody trusts is worth nothing
# --------------------------------------------------------------------------

def test_the_cache_is_a_faithful_recording(cell):
    """The stored sequence must still be one continuous alternating record.

    Everything downstream reads `sequence_*` in recorded order and integrates a
    state along it, so the two properties that have to survive caching are the
    order and the alternation. A fixture that quietly grouped the epochs by
    light mean would still fit, and would fit a cell that was never recorded.
    """
    order = vmn._epoch_order(cell.sequence_epoch)
    assert len(order) > 10, 'too few epochs to constrain a slow state'
    levels = [float(cell.sequence_light_mean[cell.sequence_epoch == e][0])
              for e in order]
    assert set(levels) == set(cell.light_means)
    # The protocol alternates epoch to epoch, which is the luminance step the
    # state is fitted to. Consecutive repeats would mean the order was lost.
    assert all(a != b for a, b in zip(levels, levels[1:])), levels
    # One light mean per epoch, and each epoch the same length.
    for index in order:
        mask = cell.sequence_epoch == index
        assert np.unique(cell.sequence_light_mean[mask]).size == 1
    widths = {int(mask.sum()) for mask in
              (cell.sequence_epoch == index for index in order)}
    assert len(widths) == 1, widths


def test_the_grouped_arrays_are_the_sequence_regrouped(cell):
    """`stimulus`/`response` are rebuilt on load, so they must be recoverable.

    `save_analysis` stores only the sequence and `load_analysis` regroups it by
    light mean, on the grounds that the two are the same samples. If that were
    off by an epoch the LN filter would be estimated from one cell's stimulus
    against another's response and nothing downstream would say so.
    """
    order = vmn._epoch_order(cell.sequence_epoch)
    for level in cell.light_means:
        rows = [index for index in order
                if float(cell.sequence_light_mean[cell.sequence_epoch == index][0]) == level]
        assert cell.stimulus[level].shape[0] == len(rows)
        assert cell.n_epochs[level] == len(rows)
        for row, index in enumerate(rows):
            mask = cell.sequence_epoch == index
            np.testing.assert_array_equal(cell.stimulus[level][row],
                                          cell.sequence_stimulus[mask])
            np.testing.assert_array_equal(cell.response[level][row],
                                          cell.sequence_response[mask])


def test_the_cache_round_trips_through_disk(cell, tmp_path):
    """Write, read, and get the same recording back.

    Samples are stored as float32 and widened on load, so a second round trip
    is exact -- which is the property that matters: a fixture rewritten by a
    later version of the code must not drift from the one the numbers below
    were measured on.
    """
    path = vmn.save_analysis(cell, tmp_path / 'again.npz')
    again = vmn.load_analysis(path)
    for name in ('sequence_stimulus', 'sequence_response',
                 'sequence_light_mean', 'sequence_epoch'):
        np.testing.assert_array_equal(getattr(again, name), getattr(cell, name))
    assert again.light_means == cell.light_means
    assert again.sampling_interval == cell.sampling_interval
    assert again.rec_type == cell.rec_type
    assert again.frequency_cutoff == cell.frequency_cutoff


def test_decimating_keeps_every_epoch_whole(cell):
    """`decimate` must cut the sample rate and nothing else.

    Slicing the concatenated sequence would work only while the epoch length
    happened to divide by the factor; otherwise the decimation phase walks
    across epoch boundaries and the epochs come out unequal. `subset_analysis`
    cuts per epoch for that reason, so an odd factor is the test.
    """
    order = vmn._epoch_order(cell.sequence_epoch)
    width = int((cell.sequence_epoch == order[0]).sum())
    small = vmn.subset_analysis(cell, decimate=3)
    assert vmn._epoch_order(small.sequence_epoch) == order
    assert small.sampling_interval == pytest.approx(cell.sampling_interval * 3)
    widths = {int((small.sequence_epoch == index).sum()) for index in order}
    assert widths == {int(np.ceil(width / 3))}, widths
    # Same samples, just fewer of them.
    mask = cell.sequence_epoch == order[0]
    np.testing.assert_array_equal(
        small.sequence_stimulus[small.sequence_epoch == order[0]],
        cell.sequence_stimulus[mask][::3])


def test_dropping_epochs_preserves_the_alternation(cell):
    """`epochs_per_level` keeps the first n of each level, in recorded order.

    The state is driven by the luminance step, so a subset that put all the dim
    epochs before all the bright ones would be a different experiment. It also
    must not be reached for casually: see `subset_analysis`'s docstring for why
    the epoch axis is the one that carries the slow state's sample size.
    """
    small = vmn.subset_analysis(cell, epochs_per_level=2)
    order = vmn._epoch_order(small.sequence_epoch)
    assert order == sorted(order), order
    levels = [float(small.sequence_light_mean[small.sequence_epoch == e][0])
              for e in order]
    assert all(a != b for a, b in zip(levels, levels[1:])), levels
    assert all(count == 2 for count in small.n_epochs.values()), small.n_epochs


# --------------------------------------------------------------------------
# The slope bound, which only means anything on a real nonlinearity
# --------------------------------------------------------------------------

def test_the_slope_ceiling_scales_to_the_sampled_range(cell):
    """`max_slope_factor` is a multiple of the transition the data can show.

    `beta0` is `sqrt(2*pi)/x_range`, a transition spanning the sampled range,
    so `max_slope_factor` reads directly as "how many times narrower than the
    range the transition may be". The flat `SIGMOID_SLOPE_MAX` cannot say that,
    and on this cell's own numbers it sits 550x above `beta0` -- which is why a
    degenerate fit can reach a step function with `at_bounds` still empty.

    The axis here is the recorded stimulus, not the generator `fit_lnk` builds:
    getting the generator means estimating the filter, which is what the
    `slow` test below does. Both axes are in contrast units and the arithmetic
    under test is the same either way.
    """
    x = cell.sequence_stimulus
    y = cell.sequence_response
    guess, lower, upper = vmn.sigmoid_start_and_bounds(x, y, rec_type=cell.rec_type)
    beta0 = float(guess[1])
    assert upper[1] == vmn.SIGMOID_SLOPE_MAX
    assert vmn.SIGMOID_SLOPE_MAX > 50 * abs(beta0), (
        'the flat ceiling is meant to be far above what the data can show; if '
        'it is not, this cell needs a different number')

    _, lower_b, upper_b = vmn.sigmoid_start_and_bounds(
        x, y, rec_type=cell.rec_type, max_slope_factor=8.0)
    assert upper_b[1] == pytest.approx(8.0 * abs(beta0))
    assert lower_b[1] == pytest.approx(-8.0 * abs(beta0))
    # gamma's ceiling is derived from beta's, so it has to follow it down --
    # otherwise the midpoint can still be pushed off the sampled axis.
    assert upper_b[2] < upper[2]


def test_an_unbounded_slope_is_left_unbounded(cell):
    """The default must change nothing, on real data as on synthetic.

    `max_slope_factor=None` is the shipped default precisely so that adding the
    argument did not silently move every existing fit.
    """
    x, y = cell.sequence_stimulus, cell.sequence_response
    for kwargs in ({}, {'max_slope_factor': None}):
        _, lower, upper = vmn.sigmoid_start_and_bounds(x, y, rec_type=cell.rec_type,
                                                       **kwargs)
        assert upper[1] == vmn.SIGMOID_SLOPE_MAX
        assert lower[1] == -vmn.SIGMOID_SLOPE_MAX


# --------------------------------------------------------------------------
# The fits themselves. About 20 s each -- marked `slow`, run before committing
# --------------------------------------------------------------------------

@pytest.fixture(scope='module')
def free_fit(cell):
    """One unbounded fit, shared: the reference every slope test compares to.

    ``n_restarts=1`` because the second restart landed on the same optimum to
    four digits on this cell and doubles the wait.
    """
    model = vmn.fit_lnk(cell, coupling='multiplicative', n_restarts=1,
                        verbose=False)
    assert model is not None, 'the LNK would not fit the cached recording'
    return model


@pytest.mark.slow
def test_the_lnk_beats_its_static_baseline_on_a_real_cell(free_fit):
    """The slow state has to earn its parameters on a retina, not a simulation.

    ``r2_static`` is the same cascade with ``k`` forced to zero and its
    nonlinearity refitted, so the comparison is nested. Measured on the cached
    2020-06-11_B recording: held-out r2 0.763 against 0.622 static, `tau_on`
    0.81 s, nothing at a bound. The thresholds below are loose around those --
    this is a regression guard, not a claim that 0.763 is the right number.
    """
    assert free_fit.r2 > 0.65, free_fit.r2
    assert free_fit.r2 > free_fit.r2_static + 0.05, (free_fit.r2, free_fit.r2_static)
    assert free_fit.r2_gain > 0.05, free_fit.r2_gain
    # A time constant at its bound is a fit reporting its box, not the cell.
    assert not free_fit.at_bounds, free_fit.at_bounds
    assert 0.1 < free_fit.params['tau_on'] < 10.0, free_fit.params['tau_on']


@pytest.mark.slow
def test_the_slope_bound_moves_work_onto_the_state(cell, free_fit):
    """What `max_slope_factor` is for, on the only data that can show it.

    Left unbounded this cell's nonlinearity settles at `beta` 10.2 against a
    generator-axis `beta0` of 0.218 -- a transition 1/47 of the sampled range,
    steeper than the ~100-point binned nonlinearity can resolve, and reached
    with `at_bounds` empty because 10.2 is nowhere near the flat ceiling of
    100. That is the degeneracy a flat ceiling cannot catch.

    Bounding it moves explanatory work off the steepness and onto the slow
    state, which is the whole point: at factor 40 the fit gives up 0.004 of
    held-out r2 and `r2_gain` -- the fraction the state contributes -- rises
    from 0.141 to 0.190.

    **It is not free past that**, which is why the argument is opt-in and has
    no recommended value. Measured on this same recording: factor 24 costs
    0.038 of r2, and by 16 the bound has pushed `tau_on` onto its own floor,
    a state made fast enough to stand in for the steepness it is no longer
    allowed. `sigmoid_start_and_bounds` carries the table.
    """
    bounded = vmn.fit_lnk(cell, coupling='multiplicative', n_restarts=1,
                          max_slope_factor=40.0, verbose=False)
    assert bounded is not None
    # The bound binds -- otherwise this test is measuring nothing.
    assert 'beta' in bounded.at_bounds, bounded.at_bounds
    assert abs(bounded.params['beta']) < abs(free_fit.params['beta'])
    # And in this regime it buys state contribution for very little r2.
    assert bounded.r2 > free_fit.r2 - 0.02, (bounded.r2, free_fit.r2)
    assert bounded.r2_gain > free_fit.r2_gain, (bounded.r2_gain, free_fit.r2_gain)
    # The default has to leave the fit alone -- that is what makes it opt-in.
    assert not free_fit.at_bounds
