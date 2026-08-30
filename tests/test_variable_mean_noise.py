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
