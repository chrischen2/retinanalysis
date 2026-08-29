"""VariableMeanNoise: LN models across a mean-luminance step, fitted in Python.

The protocol
------------
``edu.washington.riekelab.{rieke,turner}.protocols.VariableMeanNoise`` -- the two
packages carry the same protocol and the recorded epoch parameters are identical
(``lightMean``, ``stdv``, ``seed``, ``led``, ``stimTime``, ``frequencyCutoff``,
``numberOfFilters``), so :data:`PROTOCOLS` matches both and :func:`find_blocks`
reports which variant each block came from.

Gaussian noise of constant contrast is delivered **through an LED** while the
mean light level steps periodically. One epoch therefore contains transitions in
both directions, and the question is how the cell's linear-nonlinear model
changes as it adapts to each new mean.

**No filter wheel.** The LED does not sit behind the wheel, so a recorded
``background:FilterWheel:NDF`` does not attenuate this stimulus even though the
block metadata carries one -- on 2021-08-18_B the blocks report ``FW3`` beside
the LED's own ``B1, B12``. :func:`led_attenuation` uses the LED's ``ndfs`` only
and returns the wheel separately, so the double-count cannot happen silently.

How this module is meant to be used
-----------------------------------
The saved MATLAB file is a **data-entry list**, not the data:

1. :func:`load_summary` reads ``matlabSummary/rodVariableMeanNoise.mat`` -- the
   53 cells the MATLAB analysis was run on;
2. :func:`find_blocks` searches the database for VariableMeanNoise recordings,
   the same discovery step as section 1 of the cone-disc notebooks;
3. :func:`match_roster` intersects the two, so the analysis runs on cells that
   were chosen *and* whose raw data is reachable;
4. :func:`analyze_condition` loads those epochs, regenerates each epoch's
   stimulus, and fits the LN model in Python with cascadegraph.

The MATLAB's own fits stay available through :func:`load_cell`, which is what
makes step 4 checkable: the same cell can be refitted here and compared against
what the MATLAB stored.

**Reachability.** Of the 16 dates in the saved file, one (2021-08-18_B) is in
the database. The wider search finds VariableMeanNoise blocks over ~146
experiments, so the pipeline has plenty to run on -- but the overlap with the
saved roster is currently one date, and :func:`match_roster` is where that
shows up rather than in a confusing empty result later.

Stimulus regeneration
---------------------
Symphony stores the noise **generator parameters and seed**, not the waveform,
so the stimulus has to be rebuilt to fit an LN model.
:func:`gaussian_noise_stimulus` ports ``GaussianNoiseGeneratorV2`` step for
step, with one unavoidable dependency: MATLAB's
``RandStream('mt19937ar').randn`` does not match any NumPy generator (its
``rand`` does -- verified -- but ``randn`` uses a different transform from
uniform to normal), so the Gaussian draw comes from the MATLAB engine and every
later step is NumPy. Without the engine the function raises rather than
returning a stimulus that silently is not the one presented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Both packages carry the same protocol; the recorded parameters are identical.
PROTOCOLS = (
    'edu.washington.riekelab.rieke.protocols.VariableMeanNoise',
    'edu.washington.riekelab.turner.protocols.VariableMeanNoise',
)
PROTOCOL_SEARCH = 'VariableMeanNoise'

# The saved analysis, kept beside this module so the notebook needs no absolute
# path. Same file the MATLAB script writes to its `summary/` folder.
SUMMARY_DIR = Path(__file__).resolve().parent / 'matlabSummary'
DEFAULT_SUMMARY_PATH = SUMMARY_DIR / 'rodVariableMeanNoise.mat'
SUMMARY_VARIABLE = 'rodNoiseLNModelSummary'

STEP_DIRECTIONS = ('low', 'high')
STEP_LABELS = {'low': 'high → low', 'high': 'low → high'}

_IDENTITY = {
    'expDate': 'exp_date', 'cellLabel': 'cell_label', 'cellType': 'cell_type',
    'recType': 'rec_type', 'fitMode': 'fit_mode', 'exampleCell': 'example_cell',
    'epochLen': 'epoch_len',
}


# --------------------------------------------------------------------------
# 1. the data-entry list: the saved MATLAB summary
# --------------------------------------------------------------------------
def _scalar(value):
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return np.nan
        if value.size == 1:
            value = value.item()
    if isinstance(value, bytes):
        return value.decode()
    return value


def _text(value) -> str:
    value = _scalar(value)
    return '' if value is None else str(value).strip()


def _numeric(value) -> float:
    value = _scalar(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


@lru_cache(maxsize=4)
def _load_mat(path: str):
    """Load and cache the summary file. ~2.5 s and ~364 MB, so cache it."""
    import scipy.io as sio
    data = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    if SUMMARY_VARIABLE not in data:
        available = [k for k in data if not k.startswith('__')]
        raise KeyError(f'{path} has no {SUMMARY_VARIABLE!r}; found {available}')
    return np.atleast_1d(data[SUMMARY_VARIABLE])


def summary_path(path=None) -> Path:
    return Path(path) if path is not None else DEFAULT_SUMMARY_PATH


def load_summary(path=None, show: bool = False) -> pd.DataFrame:
    """The cells the MATLAB analysis was run on -- the data-entry list.

    One row per saved entry, in file order, with ``index`` as the key
    :func:`load_cell` takes and ``calendar_date`` as the key
    :func:`match_roster` joins on. Scalars only; the saved arrays stay on disk
    until asked for.

    ``duplicate`` marks entries whose (date, cell, mode) triple appears more
    than once -- the MATLAB appended a re-analysis rather than replacing the
    original, so both are kept and flagged.
    """
    entries = _load_mat(str(summary_path(path)))
    rows = []
    for index, entry in enumerate(entries):
        row = {'index': index}
        for field_name, column in _IDENTITY.items():
            row[column] = _text(getattr(entry, field_name, ''))
        for direction in STEP_DIRECTIONS:
            row[f'tau_{direction}'] = _numeric(
                getattr(getattr(entry, 'timeConsts', None), direction, np.nan))
            model = getattr(getattr(entry, 'lnModel', None), direction, None)
            row[f'r2_{direction}'] = _numeric(getattr(model, 'r2', np.nan))
        rows.append(row)
    frame = pd.DataFrame(rows)

    # epochLen is normally the epoch duration in ms, but the MATLAB wrote
    # `selectedNodes{1}.parent.splitValue` -- whatever the parent tree node
    # split on. Where that was the NDF node it holds an NDF list instead.
    frame['epoch_len_ms'] = pd.to_numeric(frame['epoch_len'], errors='coerce')
    # The saved date is `yyyy/mm/dd`; the database writes `yyyy-mm-dd_R`.
    frame['calendar_date'] = frame['exp_date'].str.replace('/', '-', regex=False)
    frame['cell_key'] = (frame['exp_date'] + '/' + frame['cell_label']
                         + '/' + frame['rec_type'])
    frame['duplicate'] = frame['cell_key'].duplicated(keep=False)
    frame['is_example'] = (frame['example_cell'].str.upper()
                           .str.startswith('Y').fillna(False))
    if show:
        print(f'{len(frame)} saved cells | {frame.exp_date.nunique()} dates | '
              f'{int(frame.duplicate.sum())} rows in duplicated groups')
        print(pd.crosstab(frame.cell_type, frame.rec_type).to_string())
    return frame


@dataclass
class LNModel:
    """A linear-nonlinear model: filter, sampled nonlinearity, variance explained.

    Used for both the models read back from the MATLAB file and the ones fitted
    here, so the two can be compared directly. ``filter_time_s`` is in seconds;
    ``source`` says which side produced it.
    """

    label: str
    r2: float
    filter: np.ndarray
    filter_time_s: np.ndarray
    nl_x: np.ndarray
    nl_y: np.ndarray
    params: Dict[str, float] = field(default_factory=dict)
    source: str = 'matlab'

    @property
    def biphasic_index(self) -> float:
        peak, trough = np.nanmax(self.filter), abs(np.nanmin(self.filter))
        total = peak + trough
        return float((peak - trough) / total) if total else np.nan

    @property
    def time_to_peak_ms(self) -> float:
        if self.filter.size == 0:
            return np.nan
        return float(self.filter_time_s[int(np.nanargmax(np.abs(self.filter)))] * 1e3)


def _model_from(struct, label: str) -> Optional[LNModel]:
    if struct is None:
        return None
    filt = np.atleast_1d(np.asarray(getattr(struct, 'filter', []), dtype=float))
    stamps = np.atleast_1d(np.asarray(
        getattr(struct, 'filterTimeStamps', []), dtype=float))
    if stamps.size != filt.size:
        stamps = np.arange(filt.size, dtype=float)
    return LNModel(
        label=label, r2=_numeric(getattr(struct, 'r2', np.nan)),
        filter=filt, filter_time_s=stamps,
        nl_x=np.atleast_1d(np.asarray(getattr(struct, 'nlX', []), dtype=float)),
        nl_y=np.atleast_1d(np.asarray(getattr(struct, 'nlY', []), dtype=float)),
        source='matlab')


def load_cell(index, path=None) -> Dict[str, object]:
    """The MATLAB's saved result for one roster row, for comparison.

    Returns the identity fields, the two step-direction LN models, and the
    binned adaptation traces. This is the reference a Python refit is checked
    against -- it is not the input to :func:`analyze_condition`.

    The stored ``SigmoidNlNode`` object cannot be read back: MATLAB wrote it
    through the MCOS mechanism and ``scipy.io.loadmat`` returns an opaque
    reference, so the stored ``alpha/beta/gamma/epsilon`` are unavailable. The
    measured ``nlX``/``nlY`` are plain arrays and come back intact.
    """
    if isinstance(index, (pd.Series, dict)):
        index = int(index['index'])
    entries = _load_mat(str(summary_path(path)))
    index = int(index)
    if not 0 <= index < entries.size:
        raise IndexError(f'index {index} outside 0..{entries.size - 1}')
    entry = entries[index]

    out: Dict[str, object] = {
        'index': index,
        'exp_date': _text(getattr(entry, 'expDate', '')),
        'cell_label': _text(getattr(entry, 'cellLabel', '')),
        'cell_type': _text(getattr(entry, 'cellType', '')),
        'rec_type': _text(getattr(entry, 'recType', '')),
        'epoch_len': _text(getattr(entry, 'epochLen', '')),
        'example_cell': _text(getattr(entry, 'exampleCell', '')),
        'ln_model': {}, 'bin_time_s': {}, 'bin_average': {}, 'time_const_s': {},
    }
    for direction in STEP_DIRECTIONS:
        for name, key in (('binTimestamps', 'bin_time_s'),
                          ('binAverage', 'bin_average')):
            values = getattr(getattr(entry, name, None), direction, None)
            out[key][direction] = (np.atleast_1d(np.asarray(values, dtype=float))
                                   if values is not None else np.array([]))
        out['time_const_s'][direction] = _numeric(
            getattr(getattr(entry, 'timeConsts', None), direction, np.nan))
        model = _model_from(
            getattr(getattr(entry, 'lnModel', None), direction, None), direction)
        if model is not None:
            out['ln_model'][direction] = model
    return out


# --------------------------------------------------------------------------
# 2. dataset search -- the same discovery step as the cone-disc notebooks
# --------------------------------------------------------------------------
def find_blocks(exp_names: Optional[Sequence[str]] = None,
                protocols: Sequence[str] = PROTOCOLS,
                show: bool = True, height: int = 400) -> pd.DataFrame:
    """VariableMeanNoise epoch blocks in the database.

    Searches on the protocol name and keeps only the variants in ``protocols``,
    so the rieke and turner copies of the same protocol are found together and
    ``protocol_name`` says which is which. ``VariableMeanNoiseCurInject`` and
    the monitor-based variants are different experiments and are excluded.
    """
    from retinanalysis.SCutils import explore as sc

    frame = sc.find_blocks(PROTOCOL_SEARCH, show=False)
    if frame.empty:
        if show:
            print(f'no blocks found for {PROTOCOL_SEARCH}')
        return frame
    frame = frame[frame.protocol_name.isin(list(protocols))].copy()
    if exp_names is not None:
        frame = frame[frame.exp_name.isin(list(exp_names))].copy()
    frame['calendar_date'] = frame.exp_name.astype(str).str.slice(0, 10)
    frame = frame.sort_values(['exp_name', 'block_id']).reset_index(drop=True)
    if show:
        print(f'{len(frame)} blocks | {frame.exp_name.nunique()} experiments')
        print(frame.protocol_name.value_counts().to_string())
        columns = [c for c in ('exp_name', 'block_id', 'protocol_name', 'ndfs',
                               'filter_wheel_ndf', 'fixed_ndf_source')
                   if c in frame.columns]
        sc.scroll_table(frame[columns], height=height,
                        num_cols=('block_id', 'filter_wheel_ndf'))
    return frame


def match_roster(roster: pd.DataFrame, blocks: Optional[pd.DataFrame] = None,
                 show: bool = True) -> pd.DataFrame:
    """Intersect the data-entry list with what the database actually holds.

    One row per saved cell, with the experiment and blocks it maps onto where
    the date is present. Matching is on calendar date: the saved file records
    ``yyyy/mm/dd`` with no rig suffix, so it cannot distinguish two rigs run on
    one day -- where that happens every matching experiment is listed and
    ``n_experiments`` is above one.

    Cell labels are deliberately **not** matched. The saved labels (``cell5``,
    ``Cell2``) come from the riekesuite source tree and the database keeps its
    own, so the date is the reliable join and the cell is chosen by hand.
    """
    blocks = find_blocks(show=False) if blocks is None else blocks
    out = roster.copy()
    if blocks.empty:
        out['n_experiments'] = 0
        out['experiments'] = ''
        out['n_blocks'] = 0
        out['reachable'] = False
        return out

    by_date = (blocks.groupby('calendar_date')
               .agg(n_experiments=('exp_name', 'nunique'),
                    experiments=('exp_name', lambda s: ', '.join(sorted(set(s)))),
                    n_blocks=('block_id', 'nunique')).reset_index())
    out = out.merge(by_date, on='calendar_date', how='left')
    out['n_experiments'] = out['n_experiments'].fillna(0).astype(int)
    out['n_blocks'] = out['n_blocks'].fillna(0).astype(int)
    out['experiments'] = out['experiments'].fillna('')
    out['reachable'] = out.n_blocks > 0
    if show:
        n = int(out.reachable.sum())
        print(f'{n} of {len(out)} saved cells sit on a date the database has '
              f'({out.loc[out.reachable, "calendar_date"].nunique()} of '
              f'{out.calendar_date.nunique()} dates)')
        if n:
            print(out[out.reachable][['index', 'exp_date', 'cell_label',
                                      'cell_type', 'rec_type', 'experiments',
                                      'n_blocks']].to_string(index=False))
    return out


@lru_cache(maxsize=128)
def _epoch_parameters_cached(block_id: int) -> pd.DataFrame:
    import json
    from retinanalysis.config import schema
    epochs = (schema.Epoch() & f'parent_id={int(block_id)}').to_pandas().reset_index()
    if epochs.empty:
        return pd.DataFrame()
    rows = []
    for raw in epochs.parameters:
        value = json.loads(raw) if isinstance(raw, str) else raw
        rows.append(value if isinstance(value, dict) else {})
    return pd.DataFrame(rows)


def epoch_parameters(block_id: int) -> pd.DataFrame:
    """Recorded parameters for every epoch of one block, one row per epoch."""
    return _epoch_parameters_cached(int(block_id)).copy()


def block_conditions(exp_name: str, blocks: Optional[pd.DataFrame] = None,
                     show: bool = True) -> pd.DataFrame:
    """Per-block stimulus conditions for one experiment.

    Reads the recorded epoch parameters rather than the protocol defaults, so
    the ``lightMean`` values listed are the ones actually presented.
    """
    blocks = find_blocks(show=False) if blocks is None else blocks
    subset = blocks[blocks.exp_name.eq(exp_name)]
    rows = []
    for block_id in subset.block_id.astype(int):
        params = epoch_parameters(block_id)
        if params.empty:
            continue
        row = {'exp_name': exp_name, 'block_id': int(block_id),
               'n_epochs': len(params)}
        for key in ('stimTime', 'sampleRate', 'frequencyCutoff',
                    'numberOfFilters', 'led'):
            values = pd.unique(params[key].dropna()) if key in params else []
            row[key] = values[0] if len(values) == 1 else ', '.join(map(str, values))
        for key in ('lightMean', 'stdv'):
            values = (sorted(pd.to_numeric(params[key], errors='coerce')
                             .dropna().unique()) if key in params else [])
            row[key] = ', '.join(f'{v:g}' for v in values)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if show and len(frame):
        print(f'{exp_name}: {len(frame)} VariableMeanNoise block(s)')
        print(frame.to_string(index=False))
    return frame


# --------------------------------------------------------------------------
# 3. light level -- LED filters only, the wheel is not in this path
# --------------------------------------------------------------------------
def led_attenuation(block_row, color: Optional[str] = None) -> Dict[str, object]:
    """Optical density of the LED's own filters, ignoring the filter wheel.

    The LED does not sit behind the wheel, so a recorded FilterWheel NDF must
    not be added to this stimulus's attenuation. It comes back as
    ``filter_wheel_ndf`` with ``wheel_ignored`` set, rather than being dropped
    silently, because the same block metadata is right for a Stage protocol and
    wrong here.

    ``color`` defaults to the LED named in the block (``UV LED`` -> ``uv``).
    """
    from retinanalysis.utils.isomerization import (led_ndf_attenuations,
                                                   parse_ndfs, infer_rig_name)

    row = block_row.iloc[0] if isinstance(block_row, pd.DataFrame) else block_row
    exp_name = str(row['exp_name'])
    rig = infer_rig_name(exp_name)
    tokens = tuple(t for t in parse_ndfs(row.get('ndfs'))
                   if not str(t).upper().startswith('FW'))
    if color is None:
        led = str(row.get('led') or '')
        color = led.split()[0].lower() if led else 'uv'

    total, missing = 0.0, []
    try:
        table = dict(led_ndf_attenuations(rig, color))
    except KeyError:
        table = {}
    for token in tokens:
        if token in table:
            total += float(table[token])
        else:
            missing.append(token)
    wheel = row.get('filter_wheel_ndf')
    return {
        'exp_name': exp_name, 'rig': rig, 'led_color': color,
        'led_ndfs': ', '.join(tokens) if tokens else '(none)',
        'optical_density': np.nan if missing else total,
        'attenuation': np.nan if missing else 10.0 ** -total,
        'unknown_tokens': ', '.join(missing),
        'filter_wheel_ndf': wheel,
        'wheel_ignored': bool(pd.notna(wheel)),
    }


# --------------------------------------------------------------------------
# 4. stimulus regeneration -- GaussianNoiseGeneratorV2
# --------------------------------------------------------------------------
_MATLAB_ENGINE = None


def _matlab_engine():
    """Start (once) and return a MATLAB engine. Only the Gaussian draw needs it."""
    global _MATLAB_ENGINE
    if _MATLAB_ENGINE is None:
        try:
            import matlab.engine
        except ImportError as exc:                       # pragma: no cover
            raise RuntimeError(
                'Regenerating the stimulus needs the MATLAB engine, because '
                "MATLAB's RandStream randn does not match any NumPy generator. "
                'Install it into this env, or pass `noise=` explicitly.'
            ) from exc
        _MATLAB_ENGINE = matlab.engine.start_matlab()
    return _MATLAB_ENGINE


def matlab_randn(seed: int, n: int) -> np.ndarray:
    """``RandStream('mt19937ar', 'Seed', seed).randn(1, n)``, exactly.

    NumPy's Mersenne Twister matches MATLAB's ``rand`` but not its ``randn``
    (verified in this project), because the two use different transforms from
    uniform to normal. The stimulus has to be the one that was presented, so
    this defers to MATLAB rather than approximating it.
    """
    engine = _matlab_engine()
    engine.eval(f"cg_s = RandStream('mt19937ar','Seed',{int(seed)});", nargout=0)
    values = engine.eval(f'cg_s.randn(1,{int(n)})', nargout=1)
    return np.asarray(values, dtype=float).ravel()


def gaussian_noise_stimulus(seed: int, stim_pts: int, st_dev: float,
                            freq_cutoff: float, num_filters: int,
                            mean: float, sample_rate: float,
                            upper_limit: float = np.inf,
                            lower_limit: float = -np.inf,
                            inverted: bool = False,
                            noise: Optional[np.ndarray] = None) -> np.ndarray:
    """Port of ``GaussianNoiseGeneratorV2.generateStimulus``.

    Follows the MATLAB step for step: draw ``stDev * randn``, FFT, apply a
    Butterworth-shaped one-sided filter mirrored to full length, zero the DC
    bin, return to the time domain, rescale so the post-smoothing standard
    deviation is the requested one, add the mean, clip.

    ``noise`` supplies the Gaussian draw directly, which is how this is tested
    without the MATLAB engine.
    """
    stim_pts = int(stim_pts)
    draw = matlab_randn(seed, stim_pts) if noise is None else np.asarray(noise, float)
    if draw.size != stim_pts:
        raise ValueError(f'noise has {draw.size} points, expected {stim_pts}')
    noise_time = float(st_dev) * draw

    noise_freq = np.fft.fft(noise_time)
    freq_step = float(sample_rate) / stim_pts
    if stim_pts % 2 == 0:
        frequencies = np.arange(stim_pts // 2 + 1) * freq_step
        one_sided = 1.0 / (1.0 + (frequencies / float(freq_cutoff))
                           ** (2 * int(num_filters)))
        filt = np.concatenate([one_sided, one_sided[1:-1][::-1]])
    else:
        frequencies = np.arange((stim_pts - 1) // 2 + 1) * freq_step
        one_sided = 1.0 / (1.0 + (frequencies / float(freq_cutoff))
                           ** (2 * int(num_filters)))
        filt = np.concatenate([one_sided, one_sided[1:][::-1]])

    # How much the filter shrinks the standard deviation. Variances of the
    # sinusoidal components add, so this is an RMS over the filter, and the DC
    # bin is excluded because it carries the mean rather than any variance.
    filter_factor = np.sqrt(filt[1:] @ filt[1:] / (stim_pts - 1))

    noise_freq = noise_freq * filt
    noise_freq[0] = 0.0
    noise_time = np.real(np.fft.ifft(noise_freq)) / filter_factor
    if inverted:
        noise_time = -noise_time
    return np.clip(noise_time + float(mean), lower_limit, upper_limit)


def epoch_stimulus(params, sample_rate: Optional[float] = None,
                   noise: Optional[np.ndarray] = None) -> np.ndarray:
    """Rebuild one epoch's stimulus from its recorded parameters."""
    rate = float(sample_rate if sample_rate is not None else params['sampleRate'])
    stim_pts = int(round(float(params['stimTime']) / 1e3 * rate))
    return gaussian_noise_stimulus(
        seed=int(float(params['seed'])), stim_pts=stim_pts,
        st_dev=float(params['stdv']), freq_cutoff=float(params['frequencyCutoff']),
        num_filters=int(float(params['numberOfFilters'])),
        mean=float(params['lightMean']), sample_rate=rate, noise=noise)


# --------------------------------------------------------------------------
# 5. LN model fitting -- vendored cascadegraph
# --------------------------------------------------------------------------
def fit_sigmoid(nl_x, nl_y, optim_iters: int = 5) -> Dict[str, float]:
    """Fit ``alpha * Phi(beta * x + gamma) + epsilon`` with cascadegraph.

    ``SigmoidNlNode`` is the Python port of the node class the MATLAB fitted,
    so parameter names and model are identical. Returns NaNs rather than
    raising when the fit fails, so one bad cell does not stop a loop.
    """
    from retinanalysis.utils.cascadegraph import SigmoidNlNode

    x = np.asarray(nl_x, dtype=float)
    y = np.asarray(nl_y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    failed = {k: np.nan for k in ('alpha', 'beta', 'gamma', 'epsilon', 'r2')}
    if x.size < 5:
        return failed
    node = SigmoidNlNode()
    guess = np.array([2 * float(np.max(np.abs(y))) or 1.0,
                      1.0 / (float(np.std(x)) or 1.0), 0.0, float(np.min(y))])
    try:
        params = node.fit_to_sample(x, y, params0=guess, optim_iters=optim_iters)
    except Exception:
        return failed
    predicted = node.process(x)
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2 = (1.0 - float(np.sum((y - predicted) ** 2)) / denominator
          if denominator else np.nan)
    return dict(zip(('alpha', 'beta', 'gamma', 'epsilon'),
                    (float(p) for p in np.asarray(params).ravel())), r2=r2)


def sigmoid(x, alpha: float, beta: float, gamma: float, epsilon: float):
    """Evaluate the fitted form, for drawing a curve through the points."""
    from retinanalysis.utils.cascadegraph import SigmoidNlNode
    return SigmoidNlNode().process_temp_params(
        np.array([alpha, beta, gamma, epsilon]), np.asarray(x, dtype=float))


def fit_ln_model(stimulus, response, sampling_interval: float,
                 label: str = '', filter_length_s: float = 1.0,
                 frequency_cutoff: Optional[float] = None,
                 correct_stim_power: bool = True,
                 n_bins: int = 100) -> LNModel:
    """Fit an LN model to (epochs x time) stimulus and response matrices.

    Every stage is cascadegraph's, matching ``computeLNmodel.m``:
    ``compute_filter`` for the linear stage, ``convolve_filter_with_stim`` for
    the generator signal, ``sample_nl`` to bin the input-output relation, and
    ``SigmoidNlNode`` for the static nonlinearity.
    """
    from retinanalysis.utils.cascadegraph import (compute_filter,
                                                  convolve_filter_with_stim,
                                                  sample_nl,
                                                  compute_variance_explained)

    stimulus = np.atleast_2d(np.asarray(stimulus, dtype=float))
    response = np.atleast_2d(np.asarray(response, dtype=float))
    if stimulus.shape != response.shape:
        raise ValueError(f'stimulus {stimulus.shape} and response '
                         f'{response.shape} must have the same shape')
    filter_pts = int(round(filter_length_s / sampling_interval))

    # computeLNmodel.m low-passes the response at the same cutoff before
    # fitting, so the model is not asked to explain noise the stimulus could
    # not have driven.
    if frequency_cutoff is not None:
        from retinanalysis.utils.cascadegraph import apply_frequency_cutoff
        response = np.atleast_2d(
            apply_frequency_cutoff(response, frequency_cutoff, sampling_interval))

    # compute_filter wants the cutoff and the interval together, or neither.
    cutoff_kwargs = ({} if frequency_cutoff is None else
                     dict(frequency_cutoff=frequency_cutoff,
                          sampling_interval=sampling_interval))
    filter_causal, _ = compute_filter(
        stimulus, response, filter_pts,
        correct_stim_power=correct_stim_power, **cutoff_kwargs)
    generator = convolve_filter_with_stim(filter_causal, stimulus)
    nl_x, nl_y = sample_nl(generator, response, num_bins=n_bins)
    params = fit_sigmoid(nl_x, nl_y)
    predicted = sigmoid(generator, params['alpha'], params['beta'],
                        params['gamma'], params['epsilon'])
    try:
        r2 = float(compute_variance_explained(predicted, response))
    except Exception:
        r2 = params['r2']
    return LNModel(label=label, r2=r2,
                   filter=np.asarray(filter_causal, dtype=float),
                   filter_time_s=np.arange(filter_pts) * sampling_interval,
                   nl_x=np.asarray(nl_x, dtype=float),
                   nl_y=np.asarray(nl_y, dtype=float),
                   params=params, source='python')


def _block_average(trace: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by averaging within each window, as ``parseData.m`` does.

    Slicing every nth sample would alias: the response is smoothed but the
    stimulus carries power right up to its 60 Hz cutoff, and decimating a
    10 kHz trace to 1 kHz without averaging folds that back in.
    """
    factor = max(int(factor), 1)
    if factor == 1:
        return np.asarray(trace, dtype=float)
    trace = np.asarray(trace, dtype=float)
    width = (trace.size // factor) * factor
    return trace[:width].reshape(-1, factor).mean(axis=1)


# --------------------------------------------------------------------------
# 6. the analysis: raw epochs -> stimulus -> LN model, split by light mean
# --------------------------------------------------------------------------
@dataclass
class ConditionAnalysis:
    """LN models for one recording, one per mean light level."""

    exp_name: str
    block_ids: List[int]
    rec_type: str
    sample_rate: float
    units: str
    light_means: List[float] = field(default_factory=list)
    n_epochs: Dict[float, int] = field(default_factory=dict)
    ln_model: Dict[float, LNModel] = field(default_factory=dict)

    def __repr__(self) -> str:
        means = ', '.join(f'{m:g}' for m in self.light_means)
        return (f'<ConditionAnalysis {self.exp_name} blocks={self.block_ids} '
                f'| {self.rec_type} | lightMean {means}>')


def analyze_condition(exp_name: str, block_ids: Sequence[int],
                      rec_type: str = 'extracellular',
                      filter_length_s: float = 1.0,
                      downsample: int = 10,
                      psth_sigma_ms: float = 10.0,
                      n_bins: int = 100,
                      frequency_cutoff: Optional[float] = None,
                      max_epochs: Optional[int] = None,
                      verbose: bool = True) -> ConditionAnalysis:
    """Load epochs, regenerate their stimuli, and fit one LN model per light mean.

    Epochs are grouped by their recorded ``lightMean``, since a filter fitted
    across two mean levels would describe neither. ``downsample`` reduces the
    10 kHz amplifier rate before fitting -- the noise is cut off at 60 Hz, so
    1 kHz is already generous and the full rate makes the estimate slow without
    making it better.

    For extracellular recordings the response is a smoothed spike rate; for
    voltage clamp it is the baseline-subtracted current.

    ``frequency_cutoff`` defaults to the stimulus's own ``frequencyCutoff``
    (60 Hz here) and is **load-bearing**, not cosmetic. The noise is 4-pole
    filtered at that frequency, so its power above it is ~1e-9 of the power
    below; dividing by that spectrum -- which is what ``correct_stim_power``
    does -- amplifies noise without bound and returns a filter that is pure
    noise. Cutting the filter off at the same frequency the stimulus was cut
    off at is what ``computeLNmodel.m`` does through ``SETTINGS``.
    """
    import retinanalysis as ra
    from scipy.ndimage import gaussian_filter1d

    spiking = rec_type == 'extracellular'
    stimuli: Dict[float, List[np.ndarray]] = {}
    responses: Dict[float, List[np.ndarray]] = {}
    sample_rate = np.nan
    cutoff = frequency_cutoff
    used = 0

    for block_id in block_ids:
        params = epoch_parameters(int(block_id))
        if params.empty:
            continue
        block = ra.SCResponseBlock(exp_name, int(block_id), b_spiking=spiking,
                                   b_LED=True, verbose=False)
        amp = np.asarray(block.amp_data, dtype=float)
        sample_rate = float(block.amp_sample_rate)
        # SCResponseBlock.get_spike_times() populates block.spike_times and
        # returns None, so read the attribute rather than the return value.
        spike_times = None
        if spiking:
            if getattr(block, 'spike_times', None) is None:
                block.get_spike_times()
            spike_times = block.spike_times
        for index in range(min(len(params), amp.shape[0])):
            if max_epochs is not None and used >= max_epochs:
                break
            row = params.iloc[index]
            if cutoff is None and 'frequencyCutoff' in row:
                cutoff = float(row['frequencyCutoff'])
            try:
                stimulus = epoch_stimulus(row, sample_rate=sample_rate)
            except Exception as exc:
                if verbose:
                    print(f'  block {block_id} epoch {index}: '
                          f'stimulus not rebuilt ({exc})')
                continue
            width = min(stimulus.size, amp.shape[1])
            if spiking:
                trace = np.zeros(amp.shape[1], dtype=float)
                times = np.asarray(spike_times[index], dtype=int)
                times = times[(times >= 0) & (times < trace.size)]
                trace[times] = 1.0
                trace = gaussian_filter1d(
                    trace, psth_sigma_ms / 1e3 * sample_rate) * sample_rate
            else:
                baseline = int(min(0.1 * sample_rate, amp.shape[1]))
                trace = amp[index] - float(np.mean(amp[index][:baseline]))
            mean_level = float(row['lightMean'])
            stimuli.setdefault(mean_level, []).append(stimulus[:width])
            responses.setdefault(mean_level, []).append(trace[:width])
            used += 1

    step = max(int(downsample), 1)
    interval = step / sample_rate
    analysis = ConditionAnalysis(
        exp_name=exp_name, block_ids=[int(b) for b in block_ids],
        rec_type=rec_type, sample_rate=sample_rate,
        units='firing rate (Hz)' if spiking else 'current (pA)')
    for mean_level in sorted(stimuli):
        width = min(min(s.size for s in stimuli[mean_level]),
                    min(r.size for r in responses[mean_level]))
        stim = np.vstack([_block_average(s[:width], step)
                          for s in stimuli[mean_level]])
        resp = np.vstack([_block_average(r[:width], step)
                          for r in responses[mean_level]])
        analysis.light_means.append(mean_level)
        analysis.n_epochs[mean_level] = int(stim.shape[0])
        analysis.ln_model[mean_level] = fit_ln_model(
            stim, resp, sampling_interval=interval,
            label=f'lightMean {mean_level:g}',
            filter_length_s=filter_length_s, n_bins=n_bins,
            frequency_cutoff=cutoff)
        if verbose:
            model = analysis.ln_model[mean_level]
            print(f'  lightMean {mean_level:g}: {stim.shape[0]} epochs | '
                  f'r²={model.r2:.3f} | time-to-peak {model.time_to_peak_ms:.0f} ms')
    return analysis


def plot_condition(analysis: ConditionAnalysis,
                   figsize: Tuple[float, float] = (11.0, 4.4)):
    """Filters and nonlinearities for one recording, one line per light mean."""
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    means = analysis.light_means
    colors = style.colors_for_conditions([f'{m:g}' for m in means])
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for mean_level in means:
        model = analysis.ln_model[mean_level]
        color = colors[f'{mean_level:g}']
        axes[0].plot(model.filter_time_s * 1e3, model.filter, lw=1.8, color=color,
                     label=f'lightMean {mean_level:g} '
                           f'(n={analysis.n_epochs[mean_level]}, r²={model.r2:.2f})')
        axes[1].plot(model.nl_x, model.nl_y, 'o', ms=3, alpha=0.6, color=color)
        params = model.params
        if params and np.isfinite(params.get('alpha', np.nan)):
            grid = np.linspace(np.nanmin(model.nl_x), np.nanmax(model.nl_x), 200)
            axes[1].plot(grid, sigmoid(grid, params['alpha'], params['beta'],
                                       params['gamma'], params['epsilon']),
                         lw=1.6, color=color)
    axes[0].axhline(0, color='#888888', lw=0.8, ls='--')
    axes[0].set_xlabel('filter time (ms)')
    axes[0].set_ylabel('filter (a.u.)')
    axes[0].set_title('temporal filter', fontsize=9)
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_xlabel('generator signal')
    axes[1].set_ylabel(analysis.units)
    axes[1].set_title('nonlinearity', fontsize=9)
    fig.suptitle(f'{analysis.exp_name} | blocks {analysis.block_ids} | '
                 f'{analysis.rec_type}', fontsize=11)
    fig.tight_layout()
    return fig


__all__ = [
    'PROTOCOLS', 'PROTOCOL_SEARCH', 'DEFAULT_SUMMARY_PATH', 'SUMMARY_DIR',
    'STEP_DIRECTIONS', 'STEP_LABELS', 'LNModel', 'ConditionAnalysis',
    'summary_path', 'load_summary', 'load_cell',
    'find_blocks', 'match_roster', 'block_conditions', 'epoch_parameters',
    'led_attenuation', 'matlab_randn', 'gaussian_noise_stimulus',
    'epoch_stimulus', 'fit_sigmoid', 'sigmoid', 'fit_ln_model',
    'analyze_condition', 'plot_condition',
]
