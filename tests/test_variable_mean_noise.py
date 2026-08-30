"""VariableMeanNoise: the stimulus must come from MATLAB, not from NumPy.

The noise stimulus is reconstructed from the seed Symphony recorded, and it has
to be the waveform the retina actually saw. MATLAB's
``RandStream('mt19937ar').randn`` does not agree with any NumPy generator -- its
``rand`` does, which is what makes the mistake easy to make -- so the Gaussian
draw is taken from the MATLAB engine and everything after it is NumPy.

These tests pin that. The reference values were produced by running the
transcribed ``GaussianNoiseGeneratorV2.generateStimulus`` in MATLAB R2025b; the
full 20000-sample case agrees with the Python port to 8e-16.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / 'SingCell_Notebooks' / 'rodAdaptation'
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

vmn = pytest.importorskip('variable_mean_noise')

# RandStream('mt19937ar', 'Seed', 42).randn(1, 64), from MATLAB R2025b.
MATLAB_RANDN_SEED42 = np.array([
    -0.53824389372926962, 0.86723215762957329, 0.97598646347253948, 0.33739025237921211,
    -0.99609409661495396, -0.52321403168644098, -1.2974474771673907, 0.91738858914593557,
    0.17660162861754794, 0.75517993567748831, -0.59149995977194048, 1.8443896370005026,
    1.816922249179955, -0.12383335027924647, -1.1106013550606419, -0.68090588026724619,
    0.014169326397636454, -0.059550610459064499, -0.66101101446033916, 0.30595091506898697,
    -0.4090578457905143, -1.2818548228848252, -0.28490282946308648, -0.06478685589122006,
    1.0003831888389663, -0.80132929831121413, 0.091800015251789033, 0.27380479193770302,
    -2.1334675594179631, 0.40397728054942095, -0.95849940934865763, -0.3263168518663801,
    2.4786956296041733, 1.5390760520725197, 0.82701089590509713, -0.15958722588765548,
    -0.8577217490829192, 0.39835396641254128, -0.22480884254365807, -1.3146676437229412,
    -0.021796865945491947, -0.88006034099445507, 1.1086404728571893, -0.46413826093464672,
    0.66456919138124804, -0.53013811821816303, 0.040909329870564795, 0.15505390913724837,
    -0.32108498156234999, 2.1458146711040254, 0.80040655725138687, 1.1962715843610494,
    2.0975463826553962, 0.38771501535540298, 1.0453182014943911, -1.3718263433146096,
    -0.95840715563049539, -2.6689399596406402, -0.80268518189891847, -0.45101594662272571,
    -0.73658154396130016, 1.1147039555125553, -0.85335003027021894, -0.75875185852041416,
])

# First 8 samples of GaussianNoiseGeneratorV2 for that draw: stDev 0.2,
# 60 Hz / 4 poles, mean 1.0, 64 points at 1 kHz. The whole 64-sample draw is
# needed to reproduce them, since the filter is applied in the frequency domain.
MATLAB_STIMULUS_HEAD = np.array([
    0.790875972988326, 0.86349387936666, 0.932651498882655, 0.996042965609082,
    1.0529432372854, 1.10333798944662, 1.14697357984573, 1.18271589004235])


def test_numpy_does_not_reproduce_matlab_randn():
    """The reason the engine is required, stated as a test.

    If some NumPy generator ever did match, this test failing is the signal to
    revisit the dependency -- not a reason to swap one in quietly.
    """
    candidates = {
        'RandomState.standard_normal':
            np.random.RandomState(42).standard_normal(8),
        'Generator(MT19937)':
            np.random.Generator(np.random.MT19937(42)).standard_normal(8),
        'default_rng': np.random.default_rng(42).standard_normal(8),
    }
    reference = MATLAB_RANDN_SEED42[:8]
    for name, values in candidates.items():
        assert not np.allclose(values, reference), (
            f'{name} now matches MATLAB randn; the engine dependency can be '
            f'revisited deliberately')


def test_generator_matches_matlab_given_the_matlab_draw():
    """Everything after the draw is NumPy, and it agrees with MATLAB.

    Passing noise= supplies MATLAB's own draw, so this runs without the
    engine and still checks the FFT, the filter construction, the
    standard-deviation correction and the mean offset against MATLAB's
    arithmetic on the same input.
    """
    stimulus = vmn.gaussian_noise_stimulus(
        seed=42, stim_pts=64, st_dev=0.2, freq_cutoff=60.0, num_filters=4,
        mean=1.0, sample_rate=1000.0, noise=MATLAB_RANDN_SEED42)
    assert stimulus.shape == (64,)
    np.testing.assert_allclose(stimulus[:MATLAB_STIMULUS_HEAD.size],
                               MATLAB_STIMULUS_HEAD, rtol=0, atol=1e-12)


def test_epoch_stimulus_takes_its_draw_from_matlab(monkeypatch):
    """``epoch_stimulus`` must route through :func:`matlab_randn`.

    The generator accepts a ``noise=`` argument for testing, and this guards the
    path that real analysis uses: with no draw supplied it has to ask MATLAB,
    never a NumPy generator.
    """
    import pandas as pd

    calls = {}

    def fake_randn(seed, n):
        calls['seed'], calls['n'] = int(seed), int(n)
        return np.zeros(n)

    monkeypatch.setattr(vmn, 'matlab_randn', fake_randn)
    params = pd.Series({'seed': 12345, 'stimTime': 100.0, 'sampleRate': 1000.0,
                        'stdv': 0.2, 'frequencyCutoff': 60.0,
                        'numberOfFilters': 4, 'lightMean': 0.5})
    vmn.epoch_stimulus(params)
    assert calls == {'seed': 12345, 'n': 100}


def test_missing_engine_raises_rather_than_falling_back():
    """No silent NumPy fallback: without the engine the call fails loudly.

    A stimulus that is not the one presented would corrupt every filter fitted
    from it, and would not look wrong.
    """
    import builtins

    real_import = builtins.__import__

    def block_matlab(name, *args, **kwargs):
        if name.startswith('matlab'):
            raise ImportError('matlab.engine unavailable')
        return real_import(name, *args, **kwargs)

    saved = vmn._MATLAB_ENGINE
    vmn._MATLAB_ENGINE = None
    builtins.__import__ = block_matlab
    try:
        with pytest.raises(RuntimeError, match='MATLAB engine'):
            vmn.matlab_randn(1, 4)
    finally:
        builtins.__import__ = real_import
        vmn._MATLAB_ENGINE = saved


def test_vendored_cascadegraph_is_self_contained():
    """The vendored copy must not fall through to a sibling checkout.

    ``utils/cascadegraph`` exists so this repository does not depend on
    ``~/Documents/GitHub/cascadegraph`` being present and on ``sys.path``. Its
    imports were absolute (``from cascadegraph.nodes.base import ...``), so the
    copy was a facade and the real code came from the sibling. This blocks the
    top-level name outright and imports the copy anyway.
    """
    import builtins
    import importlib

    for name in [n for n in list(sys.modules)
                 if n == 'cascadegraph' or n.startswith('cascadegraph.')
                 or 'utils.cascadegraph' in n]:
        del sys.modules[name]

    real_import = builtins.__import__

    def block_cascadegraph(name, *args, **kwargs):
        if name == 'cascadegraph' or name.startswith('cascadegraph.'):
            raise ImportError('top-level cascadegraph is not importable here')
        return real_import(name, *args, **kwargs)

    saved_path = list(sys.path)
    sys.path = [p for p in sys.path if 'GitHub/cascadegraph' not in p]
    builtins.__import__ = block_cascadegraph
    try:
        cg = importlib.import_module('retinanalysis.utils.cascadegraph')
        assert 'retinanalysis' in cg.__file__
        for name in ('compute_filter', 'convolve_filter_with_stim', 'sample_nl',
                     'compute_variance_explained', 'apply_frequency_cutoff',
                     'SigmoidNlNode'):
            assert hasattr(cg, name), name
        assert 'cascadegraph' not in sys.modules
    finally:
        builtins.__import__ = real_import
        sys.path = saved_path


def test_sigmoid_start_is_read_off_the_data():
    """Each start value is the statistic of the sampled curve it names.

    The parameters of ``alpha*Phi(beta*x + gamma) + epsilon`` are the baseline,
    the rise, the steepness and the midpoint, so a curve with known values must
    produce a start close to them -- otherwise the optimiser is being handed a
    generic guess and left to find its own way, which is what let ``alpha`` run
    to 1.35e7 against an ``epsilon`` of -1.35e7.
    """
    from scipy.stats import norm

    x = np.linspace(-2.0, 2.0, 100)
    alpha, beta, gamma, epsilon = 50.0, 2.0, -1.0, -10.0
    y = alpha * norm.cdf(beta * x + gamma) + epsilon

    guess, lower, upper = vmn.sigmoid_start_and_bounds(x, y)
    assert np.all(lower <= guess) and np.all(guess <= upper)
    assert guess[0] > 0                                   # rising, so alpha > 0
    assert 0.5 * alpha < guess[0] < 2.0 * alpha           # the rise
    assert abs(guess[3] - epsilon) < 0.25 * alpha         # the baseline
    # gamma/beta places the midpoint; check the implied x50 rather than the
    # raw pair, since only their ratio is identifiable from a location.
    assert abs(-guess[2] / guess[1] - (-gamma / beta)) < 0.2 * np.ptp(x)

    # The fit that follows recovers the true parameters and stays conditioned.
    fitted = vmn.fit_sigmoid(x, y)
    assert fitted['r2'] > 0.999
    assert abs(fitted['alpha']) < 10 * np.ptp(y)


def test_sigmoid_start_handles_a_falling_curve():
    """A negative-going nonlinearity must start with a negative alpha."""
    from scipy.stats import norm

    x = np.linspace(-2.0, 2.0, 100)
    y = -40.0 * norm.cdf(1.5 * x) + 5.0
    guess, lower, upper = vmn.sigmoid_start_and_bounds(x, y)
    assert guess[0] < 0
    assert np.all(lower <= guess) and np.all(guess <= upper)
    assert vmn.fit_sigmoid(x, y)['r2'] > 0.999


def test_amplitude_ceiling_follows_the_recording_units():
    """``alpha`` is a response amplitude, so its bound is in the response's units.

    5 nA means nothing to an extracellular recording and 1000 Hz means nothing
    to a voltage clamp, so the ceiling has to come from ``rec_type`` rather than
    from one constant.
    """
    from scipy.stats import norm

    x = np.linspace(-2.0, 2.0, 100)
    # A large-amplitude curve, so the data-relative cap is far above both
    # physiological ceilings and the units decide which one applies.
    y = 4000.0 * norm.cdf(1.5 * x) - 2000.0

    _, low_wc, high_wc = vmn.sigmoid_start_and_bounds(x, y, rec_type='exc')
    _, low_ex, high_ex = vmn.sigmoid_start_and_bounds(x, y, rec_type='extracellular')
    assert high_wc[0] == vmn.SIGMOID_AMPLITUDE_MAX['exc'] == 5000.0
    assert high_ex[0] == vmn.SIGMOID_AMPLITUDE_MAX['extracellular'] == 1000.0
    assert low_wc[0] == -high_wc[0] and low_ex[0] == -high_ex[0]

    # 'inh' shares the whole-cell ceiling; an unknown mode falls back to the
    # data-relative cap alone rather than silently applying a wrong unit.
    _, _, high_inh = vmn.sigmoid_start_and_bounds(x, y, rec_type='inh')
    _, _, high_none = vmn.sigmoid_start_and_bounds(x, y, rec_type=None)
    assert high_inh[0] == 5000.0
    assert high_none[0] > 5000.0


def test_epsilon_is_not_capped_at_an_absolute_current():
    """The baseline carries the holding current, which reaches -14 nA here.

    ``epsilon`` is an absolute level, not an amplitude: one cell in this
    dataset modulates by 946 pA about a holding current of -14.4 nA. A ceiling
    of "a few thousand pA" on ``epsilon`` would refuse that cell outright, so
    it is bounded by the data's own range instead.
    """
    from scipy.stats import norm

    x = np.linspace(-2.0, 2.0, 100)
    y = 946.0 * norm.cdf(1.5 * x) - 14_357.0      # the real cell's scale

    guess, lower, upper = vmn.sigmoid_start_and_bounds(x, y, rec_type='exc')
    assert lower[3] < -14_357.0 < upper[3]
    assert np.all(lower <= guess) and np.all(guess <= upper)

    fitted = vmn.fit_sigmoid(x, y, rec_type='exc')
    assert fitted['r2'] > 0.999
    assert abs(fitted['epsilon'] + 14_357.0) < 200.0
    assert not fitted['at_bounds'], fitted['at_bounds']


def test_generator_axis_bounds_do_not_depend_on_units():
    """``beta`` and ``gamma`` live on the generator axis, which is contrast.

    The generator signal runs to about +/-3 whatever the amplifier was doing,
    so one pair of limits serves every recording mode.
    """
    from scipy.stats import norm

    x = np.linspace(-2.0, 2.0, 100)
    y = 100.0 * norm.cdf(2.0 * x)
    bounds = [vmn.sigmoid_start_and_bounds(x, y, rec_type=r)[2][1:3]
              for r in ('exc', 'inh', 'extracellular', None)]
    for other in bounds[1:]:
        np.testing.assert_allclose(bounds[0], other)
    assert bounds[0][0] == vmn.SIGMOID_SLOPE_MAX == 100.0


def test_fit_sigmoid_reports_a_constrained_fit():
    """A fit resting on its bound must say so rather than report the bound."""
    x = np.linspace(-1.0, 1.0, 100)
    y = 300.0 * x                       # a line: alpha runs away without a bound

    fitted = vmn.fit_sigmoid(x, y, rec_type='extracellular')
    assert 'at_bounds' in fitted
    if fitted['at_bounds']:
        assert abs(fitted['alpha']) <= vmn.SIGMOID_AMPLITUDE_MAX['extracellular']
    # A curve the box comfortably contains reports nothing.
    from scipy.stats import norm
    clean = vmn.fit_sigmoid(x, 80.0 * norm.cdf(2.5 * x) + 5.0,
                            rec_type='extracellular')
    assert clean['at_bounds'] == ()


def test_psth_binning_matches_smoothing_at_the_full_rate():
    """Binning then smoothing must equal smoothing then binning.

    The PSTH used to be laid down at the amplifier's 10 kHz, Gaussian-smoothed
    there, and only then block-averaged to the analysis rate -- a 302k-sample
    array convolved with an 801-tap kernel per epoch, which was most of the
    cost of loading a condition. Convolution commutes with the boxcar that
    block-averaging applies, so doing it at the reduced rate is the same trace
    for ~100x less arithmetic; this pins that they agree.
    """
    from scipy.ndimage import gaussian_filter1d

    rng = np.random.default_rng(0)
    n_samples, sample_rate, step, sigma_ms = 302_000, 10_000.0, 10, 10.0
    spikes = np.sort(rng.choice(n_samples, size=1800, replace=False))

    dense = np.zeros(n_samples)
    dense[spikes] = 1.0
    slow = vmn._block_average(
        gaussian_filter1d(dense, sigma_ms / 1e3 * sample_rate) * sample_rate, step)
    fast = vmn._spike_rate(spikes, n_samples, sample_rate, step, sigma_ms)

    assert fast.shape == slow.shape
    assert np.corrcoef(slow, fast)[0, 1] > 0.999
    # Absolute agreement relative to the peak rate, not to zero: the two orders
    # differ only by where the boxcar falls, which is sub-bin.
    assert np.max(np.abs(slow - fast)) < 0.05 * slow.max()
    # Total spike count is preserved either way.
    assert abs(fast.sum() - slow.sum()) < 0.01 * slow.sum()
