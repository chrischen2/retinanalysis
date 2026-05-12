"""Regen for ``MEAVariableMeanNoiseCone``.

Source: ``chris-package/+edu/+washington/+riekelab/+chris/+protocols/
MEAVariableMeanNoiseCone.m``

Stimulus structure: full-screen luminance modulated by Gaussian noise about
a per-epoch mean. Each epoch:

    framePerPeriod = ceil(60/frameDwell * stimTime/1e3)
    noiseStream = RandStream('mt19937ar', 'Seed', noiseSeed)
    intensity = currentMean + noiseStdv * currentMean * noiseStream.randn(1, framePerPeriod)
    intensity = clip(intensity, 0, 1)

Per-epoch params saved in the H5:
    noiseSeed, currentMean

Block-level params:
    preTime, stimTime, tailTime, noiseStdv, frameDwell, meanIntensity (list),
    apertureDiameter, numberOfAverages

This protocol is entirely seed-driven and has no external resource
dependencies, so it regenerates whether or not chris-package is cloned.

MATLAB ↔ NumPy ``randn`` parity caveat
--------------------------------------
The *uniform* MT19937 streams produced by MATLAB's ``RandStream('mt19937ar',
'Seed', s).rand(...)`` and NumPy's ``RandomState(s).rand(...)`` are
byte-identical. The *normal* streams are NOT — MATLAB uses Marsaglia's
ziggurat algorithm to transform uniforms into normals, while NumPy uses
polar Marsaglia. The marginal distribution is the same N(0,1), so any
analysis that depends on stimulus *statistics* (STA, LN/GLM fitting,
spectrum estimation) is unaffected. Frame-by-frame byte-exact replay is
not — use the actual saved values from the rig if you need that.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from typing import Any, Literal

from . import register
from ._matlab_rng import matlab_randn, MatlabUnavailableError
from retinanalysis.utils.matlab_engine import is_matlab_engine_available

PROTOCOL_NAME = 'edu.washington.riekelab.chris.protocols.MEAVariableMeanNoiseCone'


def _draw_randn(seed: int, n: int, engine: str) -> tuple[np.ndarray, str]:
    """Draw n standard normals matching MATLAB. Returns (values, engine_used)."""
    if engine == 'numpy':
        return np.random.RandomState(int(seed)).randn(n), 'numpy'
    if engine == 'matlab':
        return matlab_randn(int(seed), 1, n).ravel(), 'matlab'
    # auto
    if is_matlab_engine_available():
        try:
            return matlab_randn(int(seed), 1, n).ravel(), 'matlab'
        except MatlabUnavailableError:
            pass
    return np.random.RandomState(int(seed)).randn(n), 'numpy'


def regen(stim_block: Any, verbose: bool = True,
          frame_rate_hz: float = 60.0,
          engine: Literal['auto', 'matlab', 'numpy'] = 'auto') -> xr.Dataset:
    """Build an xarray.Dataset of regenerated noise intensities.

    Returns a Dataset with:
        intensity (epoch, update) float — clipped intensity per noise update
        current_mean (epoch) float — per-epoch mean intensity drawn from
            ``meanIntensity`` list at run time
        noise_seed (epoch) int — per-epoch RNG seed (saved by Symphony)

    Each ``update`` represents one noise step; with the default
    ``frameDwell=1`` and ``frame_rate_hz=60``, there is one update per
    monitor frame. The ``time_s`` coord on ``update`` gives the time of
    each update in seconds from stim onset.

    Parameters
    ----------
    engine : {'auto', 'matlab', 'numpy'}
        Where to draw the normal samples from.
        * ``'auto'`` (default): use MATLAB engine if installed (byte-exact
          with the rig), else fall back to NumPy.
        * ``'matlab'``: require MATLAB; raise if it can't start.
        * ``'numpy'``: use ``np.random.RandomState(seed).randn(...)``. Same
          distribution as MATLAB but not byte-exact (ziggurat vs polar
          Marsaglia). The result is set on ``ds.attrs['rng_engine']``.
    """
    df = stim_block.df_epochs
    n_epochs = len(df)
    block_params = getattr(stim_block, 'd_epoch_block_params', {}) or {}

    # Block-level params we need to regen the noise. Pull from
    # d_epoch_block_params first, fall back to per-epoch parameters dict.
    def _block(key, default=None):
        v = block_params.get(key, None)
        if v is not None:
            return v
        first = df['epoch_parameters'].iloc[0] if n_epochs else {}
        return first.get(key, default)

    stim_time_ms = float(_block('stimTime'))
    frame_dwell = int(_block('frameDwell', 1))
    noise_stdv = float(_block('noiseStdv', 0.3))

    update_rate = frame_rate_hz / frame_dwell
    frame_per_period = int(np.ceil(update_rate * stim_time_ms / 1e3))

    # Per-epoch arrays
    seeds = np.array([int(p['noiseSeed']) for p in df['epoch_parameters']])
    current_means = np.array([float(p['currentMean']) for p in df['epoch_parameters']])

    intensity = np.zeros((n_epochs, frame_per_period), dtype=np.float64)
    engines_used = set()
    for i in range(n_epochs):
        z, used = _draw_randn(int(seeds[i]), frame_per_period, engine)
        engines_used.add(used)
        ints = current_means[i] + noise_stdv * current_means[i] * z
        intensity[i] = np.clip(ints, 0.0, 1.0)

    rng_engine = 'matlab' if engines_used == {'matlab'} else (
        'numpy' if engines_used == {'numpy'} else 'mixed'
    )

    if verbose:
        unique_means = sorted(set(current_means.tolist()))
        print(f'[regen] MEAVariableMeanNoiseCone: {n_epochs} epochs × {frame_per_period} updates '
              f'(frameDwell={frame_dwell}, noiseStdv={noise_stdv}, rng={rng_engine})')
        print(f'[regen]   currentMean values used: {unique_means}')

    time_s = np.arange(frame_per_period) / update_rate

    ds = xr.Dataset(
        data_vars={
            'intensity': (('epoch', 'update'), intensity),
            'current_mean': (('epoch',), current_means),
            'noise_seed': (('epoch',), seeds),
        },
        coords={
            'epoch': np.arange(n_epochs),
            'time_s': ('update', time_s),
        },
    )
    ds.attrs.update({
        'protocol_name': PROTOCOL_NAME,
        'exp_name': stim_block.exp_name,
        'datafile_name': stim_block.datafile_name,
        'preTime_ms': block_params.get('preTime', None),
        'stimTime_ms': stim_time_ms,
        'tailTime_ms': block_params.get('tailTime', None),
        'noiseStdv': noise_stdv,
        'frameDwell': frame_dwell,
        'frame_rate_hz': frame_rate_hz,
        'update_rate_hz': update_rate,
        'meanIntensity_options': _block('meanIntensity', None),
        'apertureDiameter_um': block_params.get('apertureDiameter', None),
        'rng_engine': rng_engine,
    })
    return ds


register(PROTOCOL_NAME, regen)
