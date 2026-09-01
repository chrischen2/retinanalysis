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
import warnings
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


# --------------------------------------------------------------------------
# The state's integration grid, on the real drive it has to integrate
# --------------------------------------------------------------------------

@pytest.fixture(scope='module')
def drive(cell):
    """A realistic rectified drive: the recorded stimulus through the sigmoid.

    The grid error depends on what the state is integrating, so a synthetic
    drive would be measuring the wrong thing -- this protocol's drive steps
    between light levels and that step is what a coarse grid smears.

    **Two epochs, not the whole record.** The reference these tests compare
    against integrates at one sample per block, which is a Python loop over
    every block: 570 s of it costs 25 s and puts these tests outside the fast
    tier they belong in. Two epochs is 60 s spanning one luminance step -- the
    feature under test -- and gives 3.0% at 100 ms against the full record's
    3.6%, the same conclusion for a fortieth of the wait.
    """
    from scipy.stats import norm
    order = vmn._epoch_order(cell.sequence_epoch)
    width = int((cell.sequence_epoch == order[0]).sum())
    g = np.asarray(cell.sequence_stimulus[:2 * width], dtype=float)
    return norm.cdf(2.0 * (g - g.mean()) / g.std() - 0.5)


def test_the_one_state_grid_resolves_the_taus_it_allows(cell, drive):
    """`state_dt_ms` has to resolve the fastest `tau_on` the bounds permit.

    `adaptation_state` is exponential Euler, so the step is a speed choice and
    not a stability one -- which is exactly why it can be wrong quietly. At the
    old 25 ms default the state was 5% of its own range off at this cell's
    fitted taus and 57% off at the 0.05 s bound, and fits do reach that bound.
    At 5 ms it is under 1% and 14%.
    """
    dt = cell.sampling_interval

    def state_at(step, tau_on, tau_off):
        coarse = vmn._bin_mean(drive[None, :], step).ravel()
        fine = vmn.adaptation_state(coarse, dt * step, tau_on, tau_off)
        return np.repeat(fine, step)[:drive.size]

    default_step = max(int(round(5.0 / 1e3 / dt)), 1)
    for tau_on, tau_off, tolerance in ((0.81, 1.09, 0.02), (0.15, 0.80, 0.05)):
        reference = state_at(1, tau_on, tau_off)
        error = (np.max(np.abs(state_at(default_step, tau_on, tau_off) - reference))
                 / np.ptp(reference))
        assert error < tolerance, (tau_on, tau_off, error)

    # And the error has to fall with the step, or the grid is not the thing
    # being measured.
    reference = state_at(1, 0.15, 0.80)
    errors = [np.max(np.abs(state_at(max(int(round(ms / 1e3 / dt)), 1), 0.15, 0.80)
                            - reference)) / np.ptp(reference)
              for ms in (40.0, 20.0, 10.0)]
    assert errors[0] > errors[1] > errors[2], errors


def test_the_slow_pool_ramps_across_the_block(cell, drive):
    """Holding `I` fixed puts a staircase straight onto the model's output.

    The gain is proportional to `R = 1 - A - I`, so quantising `I` at the block
    rate quantises `A`. Measured on this recording, holding was 30% of `A`'s
    range off at 250 ms and about the same at 10 s as at 1.4 s -- the signature
    of a quantisation artefact, not of unresolved dynamics. The ramp costs a
    second `_relax` per block and takes 100 ms to 3.6%.
    """
    dt = cell.sampling_interval
    k_act, k_inact = vmn.two_state_rates(1e-3, 4.0)
    step = max(int(round(100.0 / 1e3 / dt)), 1)
    for k_in, k_out in ((3.0, 0.7), (3.0, 0.1)):
        reference, _ = vmn.two_state_kinetics(
            drive, dt, k_act, k_inact, k_in, k_out,
            state_step=max(int(round(8.0 / 1e3 / dt)), 1))
        active, inactivated = vmn.two_state_kinetics(drive, dt, k_act, k_inact,
                                                     k_in, k_out, state_step=step)
        error = np.max(np.abs(active - reference)) / np.ptp(reference)
        assert error < 0.06, (k_out, error)
        # `I` moves within a block rather than jumping between them.
        inside = np.abs(np.diff(inactivated[:step]))
        assert np.any(inside > 0), 'I is constant across the block -- not ramped'
        # Occupancies stay physical, which the ramp must not break.
        resting = 1.0 - active - inactivated
        assert active.min() >= -1e-9 and inactivated.min() >= -1e-9
        assert resting.min() >= -1e-9, resting.min()


def test_the_solver_residual_tracks_the_grid_error(cell, drive):
    """`return_residual` is the diagnostic to trust when the rates move.

    100 ms is right for the regime these fits go to (a slow recovery, a large
    in/out ratio) and *not* universally: at `k_slow_out` 2/s -- a 0.5 s pool --
    it is still 25% off. The residual is what says so, so it has to rise with
    the error rather than merely being printed.
    """
    dt = cell.sampling_interval
    k_act, k_inact = vmn.two_state_rates(1e-3, 4.0)
    residuals = []
    for ms in (250.0, 100.0, 50.0):
        step = max(int(round(ms / 1e3 / dt)), 1)
        _, _, residual = vmn.two_state_kinetics(drive, dt, k_act, k_inact,
                                                3.0, 0.7, state_step=step,
                                                return_residual=True)
        residuals.append(residual)
    assert residuals[0] > residuals[1] > residuals[2], residuals


def test_the_grid_error_falls_with_the_step(cell, drive):
    """Against a fixed reference the sequence must be a convergence curve.

    This is the property a sweep needs and the reason `reference_step` exists.
    With each candidate refined against *itself* the sequence is not monotone --
    measured on this cell, 200 ms scored 5.8% and 100 ms scored 8.4%, because
    the two were compared against different references -- so a scan built on
    that could not justify picking anything. Both numbers are individually
    meaningful; only the fixed reference makes them comparable.

    **Monotone where the scheme is asymptotic, which is not everywhere.** At
    the coarse end it still is not: 400 ms scores 6.3% against 200 ms at 7.3%
    for these rates, because a block that long is not resolving anything and
    the max-abs error over the record stops behaving like a truncation term.
    That is why `scan_state_dt` requires two consecutive passing candidates
    rather than trusting one -- and why this test asserts the curve only from
    200 ms down, where the recommendation actually lives.
    """
    dt = cell.sampling_interval
    reference = max(int(round(10.0 / 1e3 / dt)), 1)
    k_act, k_inact = vmn.two_state_rates(1e-3, 4.0)
    rates = dict(k_act=k_act, k_inact=k_inact, k_slow_in=3.0, k_slow_out=0.7)
    errors = [vmn.state_grid_error(drive, dt, max(int(round(ms / 1e3 / dt)), 1),
                                   reference_step=reference,
                                   variant='two_state', **rates)
              for ms in (200.0, 100.0, 50.0, 25.0)]
    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] > 0, errors

    one = [vmn.state_grid_error(drive, dt, max(int(round(ms / 1e3 / dt)), 1),
                                reference_step=reference, variant='one_state',
                                tau_on=0.15, tau_off=0.80)
           for ms in (100.0, 50.0, 25.0)]
    assert one == sorted(one, reverse=True) and one[-1] > 0, one

    # A step at or below the reference has nothing finer to be compared with.
    assert vmn.state_grid_error(drive, dt, reference, reference_step=reference,
                                variant='one_state', tau_on=0.15,
                                tau_off=0.80) == 0.0


def test_the_guard_errs_on_the_conservative_side(cell, drive):
    """The cheap metric must not under-state the error it is guarding against.

    `fit_lnk_two_state` checks itself with the `refine` comparison, because a
    fixed fine reference over the whole record is a solve it would rather not
    pay on every fit. The raw difference under-states the true error by
    construction -- it measures `C S (1 - 1/refine)` where the error is `C S` --
    and a guard that reads low is a guard that passes a grid-limited fit. The
    first-order correction is what makes it read slightly high instead: at this
    cell's fitted rates, 4.14% against a true 3.89% at 100 ms and 12.86%
    against 11.50% at 250, where the raw numbers were 3.15% and 9.75%.
    """
    dt = cell.sampling_interval
    k_act, k_inact = vmn.two_state_rates(0.004, 0.184)
    rates = dict(k_act=k_act, k_inact=k_inact, k_slow_in=20.0, k_slow_out=0.271)
    # One reference solve, shared: `state_grid_error(reference_step=...)` would
    # re-run the expensive one per candidate and this test is in the fast tier.
    truth = vmn._state_at_step(drive, dt, 1, 'two_state', rates)
    for ms in (250.0, 100.0, 50.0):
        step = max(int(round(ms / 1e3 / dt)), 1)
        guard = vmn.state_grid_error(drive, dt, step, refine=4, **rates)
        raw = vmn.state_grid_error(drive, dt, step, refine=4, extrapolate=False,
                                   **rates)
        true = vmn._relative_deviation(
            vmn._state_at_step(drive, dt, step, 'two_state', rates), truth)
        assert raw < true, (ms, raw, true)          # the reason for the fix
        assert guard >= true * 0.95, (ms, guard, true)
        assert guard <= true * 1.6, (ms, guard, true)

    # The correction applies to a refinement and not to a fixed reference,
    # where there is no ratio to correct by.
    step = max(int(round(100.0 / 1e3 / dt)), 1)
    assert (vmn.state_grid_error(drive, dt, step, reference_step=2, **rates)
            == vmn.state_grid_error(drive, dt, step, reference_step=2,
                                    extrapolate=False, **rates))


def test_the_grid_error_needs_the_rates_it_scores(cell, drive):
    """A missing rate must be an error, not a silent default.

    Scoring the grid against the wrong rate constants would produce a number
    that looks fine and means nothing, which is worse than a traceback.
    """
    dt = cell.sampling_interval
    with pytest.raises(TypeError):
        vmn.state_grid_error(drive, dt, 25, variant='two_state', k_act=1.0)
    with pytest.raises(ValueError):
        vmn.state_grid_error(drive, dt, 25, variant='nonsense', tau_on=0.1,
                             tau_off=0.1)


def test_the_scan_recommends_a_step_that_meets_its_tolerance(cell):
    """`scan_state_dt` picks the largest step it measured under tolerance.

    Largest, because the step is pure cost: the only reason to go finer is
    accuracy, and this is what measures it. The scan works coarse to fine and
    stops at the first pass, so the recommendation is the last row it ran.
    """
    for variant, tolerance in (('two_state', 0.05), ('one_state', 0.02)):
        table = vmn.scan_state_dt(cell, variant=variant, tolerance=tolerance,
                                  candidates_ms=(200.0, 100.0, 50.0, 25.0),
                                  show=False)
        assert not table.empty
        best = table.attrs['recommended_state_dt_ms']
        worst = table.groupby('state_dt_ms').error.max()
        assert worst[best] <= tolerance, (variant, best, worst[best])
        # Everything coarser than the pick was tried and failed, or the scan
        # would have stopped there instead.
        for ms, value in worst.items():
            if ms > best:
                assert value > tolerance, (variant, ms, value)


def test_the_scan_is_conservative_against_the_fitted_rates(cell):
    """The pre-flight scan may ask for a smaller step than the cell needs.

    It takes the worst case over a box of plausible rate constants, because the
    rates are what the fit is for and are not known beforehand. On this cell the
    box's worst corner is a 0.5 s slow pool, which wants 50 ms; the two-state
    fit actually lands at `k_slow_out` 0.271/s -- a 3.7 s pool -- where 100 ms
    measures 3.2% and is fine. Both numbers are right, which is why
    `fit_lnk_two_state` re-checks at its own fitted rates through
    `solver_tolerance` rather than trusting the scan.
    """
    dt = cell.sampling_interval
    drive = vmn.probe_drive(cell)
    step = max(int(round(100.0 / 1e3 / dt)), 1)
    reference = max(int(round(10.0 / 1e3 / dt)), 1)
    k_act, k_inact = vmn.two_state_rates(0.004, 0.184)   # the fitted fast state
    # Scored the same way the scan scores, or the two are not comparable.
    fitted = vmn.state_grid_error(drive, dt, step, reference_step=reference,
                                  variant='two_state', k_act=k_act,
                                  k_inact=k_inact, k_slow_in=20.0,
                                  k_slow_out=0.271)
    box = vmn.scan_state_dt(cell, variant='two_state', tolerance=0.05,
                            candidates_ms=(200.0, 100.0, 50.0),
                            show=False).groupby('state_dt_ms').error.max()
    assert fitted < 0.05, fitted
    assert box[100.0] > 0.05 > fitted, (box[100.0], fitted)


def test_probe_drive_spans_a_luminance_step(cell):
    """The probe has to contain the feature whose smearing is being measured.

    A drive taken from one epoch has no step in it, and a grid coarse enough to
    smear the step would score clean. Two epochs of an alternating protocol is
    one dim and one bright by construction.
    """
    drive = vmn.probe_drive(cell, n_epochs=2)
    order = vmn._epoch_order(cell.sequence_epoch)
    width = int((cell.sequence_epoch == order[0]).sum())
    assert drive.size == 2 * width
    levels = [float(cell.sequence_light_mean[cell.sequence_epoch == e][0])
              for e in order[:2]]
    assert levels[0] != levels[1], levels
    assert np.all((drive >= 0) & (drive <= 1)), 'a drive is an occupancy'


@pytest.mark.slow
def test_a_grid_limited_fit_says_so(cell, monkeypatch):
    """`solver_tolerance` has to reach the caller, not just a printed number.

    Before this it was a parameter `fit_lnk_two_state` declared and never read.
    The guard is only worth anything if exceeding it is visible, so this forces
    the measurement high and checks all three ways it surfaces: the warning, the
    flag, and the recorded error. The error is forced rather than provoked with
    real rates because whether a given cell lands in the grid-limited regime is
    a property of that cell -- on this subset it does not, which is the right
    outcome and the wrong test.
    """
    small = vmn.subset_analysis(cell, epochs_per_level=2, decimate=2)
    monkeypatch.setattr(vmn, 'state_grid_error', lambda *a, **k: 0.9)
    with pytest.warns(RuntimeWarning, match='grid-limited'):
        model = vmn.fit_lnk_two_state(small, n_restarts=1, state_dt_ms=400.0,
                                      solver_tolerance=0.05, verbose=False)
    assert model is not None
    assert model.grid_limited is True
    assert model.solver_error == pytest.approx(0.9)
    assert model.solver_tolerance == pytest.approx(0.05)


@pytest.mark.slow
def test_a_converged_fit_is_left_alone(cell):
    """The guard must stay quiet when the grid is fine, or it will be ignored.

    Same subset, measured rather than forced: the fit lands somewhere 400 ms
    resolves, `solver_error` comes out near 0.005, and nothing is raised.
    """
    small = vmn.subset_analysis(cell, epochs_per_level=2, decimate=2)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        model = vmn.fit_lnk_two_state(small, n_restarts=1, state_dt_ms=400.0,
                                      solver_tolerance=0.05, verbose=False)
    assert model is not None
    assert model.grid_limited is False
    assert 0.0 <= model.solver_error < 0.05, model.solver_error


def test_a_decimated_analysis_can_still_be_fitted(cell):
    """Any sample rate has to give an even filter, since decimation is a knob.

    `convolve_filter_with_stim` refuses an odd-length filter, and whether
    `filter_length_s / dt` lands odd is an accident of the rate: 1.0 s at
    125 Hz is 125, which raised "Filter must have an even number of points"
    from inside the fit. `subset_analysis(decimate=...)` makes that reachable
    by ordinary use, so it is guarded at the source.
    """
    for decimate in (1, 2, 3, 4, 5, 7):
        small = vmn.subset_analysis(cell, epochs_per_level=1, decimate=decimate)
        points = vmn._even_filter_pts(small.filter_length_s,
                                      small.sampling_interval)
        assert points % 2 == 0 and points >= 2, (decimate, points)
