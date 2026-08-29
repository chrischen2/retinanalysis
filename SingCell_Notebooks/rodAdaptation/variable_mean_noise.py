"""VariableMeanNoise: LN models across a mean-luminance step, from rod levels up.

Python port of ``analyzeVariableMeanNoise.m`` (rodAdaptation repo). The MATLAB
drove an interactive riekesuite epoch tree; the analysis it produced was saved
one cell at a time into ``summary/rodVariableMeanNoise.mat``, and every
population figure in that script reads from the saved struct array rather than
from the recordings. This module keeps that split and makes the saved file the
primary source:

``load_summary``
    the roster of saved cells -- one row per (date, cell, recording mode);
``load_cell``
    the arrays for one of them (adaptation kinetics, LN filters and
    nonlinearities, the phase-resolved temporal LN model);
``find_blocks`` / ``unanalyzed_dates``
    discovery over the whole database, for the *other* direction -- finding
    recordings that have not been analyzed yet.

**Why the saved file leads.** Of the 16 experiment dates in the summary, one
(2021-08-18_B) currently has raw data reachable on this machine and one is in
the DataJoint database. The analysis those 53 cells represent is therefore only
reproducible from the saved arrays, so this module treats them as the data and
the recordings as the optional extra.

**The stimulus.** Gaussian noise delivered at a mean luminance that steps
part-way through each epoch, so one epoch contains a high-to-low transition and
the next a low-to-high one. Throughout this module and the saved file, ``low``
and ``high`` name the *step direction*: ``low`` is the response after a step
down to the low mean, ``high`` after a step up. Each carries its own LN model,
and ``temporalLNModel`` refits the model in five successive windows after the
step so the filter and nonlinearity can be watched recovering.

**Model fitting is cascadegraph's**, the same library the MATLAB used
(``SigmoidNlNode``, ``compute_filter``, ``sample_nl``), via its Python port at
``cascadegraph/python``. Nothing here reimplements a filter or a sigmoid.

**What cannot be read back.** Each saved model carries a ``node``, the fitted
``SigmoidNlNode`` object. MATLAB stored it through the MCOS object mechanism,
which ``scipy.io.loadmat`` cannot unpack -- it returns an opaque reference, so
the stored ``alpha/beta/gamma/epsilon`` are not recoverable from Python. The
measured nonlinearity (``nlX``/``nlY``) is saved as plain arrays and *is*
readable, so :func:`fit_sigmoid` refits the same node class to those points.
That is the same model fitted to the same data, but the parameters are a refit
rather than the stored values, and :func:`fit_sigmoid` is where that happens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROTOCOL = 'edu.washington.riekelab.grimes.protocols.VariableMeanNoise'

# The saved analysis, kept beside this module so the notebook needs no absolute
# path. This is the same file the MATLAB script writes to its `summary/` folder.
SUMMARY_DIR = Path(__file__).resolve().parent / 'matlabSummary'
DEFAULT_SUMMARY_PATH = SUMMARY_DIR / 'rodVariableMeanNoise.mat'
SUMMARY_VARIABLE = 'rodNoiseLNModelSummary'

# The two step directions, in the order the figures should draw them.
STEP_DIRECTIONS = ('low', 'high')
STEP_LABELS = {'low': 'high → low', 'high': 'low → high'}

# temporalLNModel windows, in order. The MATLAB always wrote five.
PHASES = ('first', 'second', 'third', 'fourth', 'fifth')

# Scalar identity fields on the saved struct, and the roster column they become.
_IDENTITY = {
    'expDate': 'exp_date', 'cellLabel': 'cell_label', 'cellType': 'cell_type',
    'recType': 'rec_type', 'fitMode': 'fit_mode', 'exampleCell': 'example_cell',
    'epochLen': 'epoch_len',
}


# --------------------------------------------------------------------------
# reading the saved MATLAB summary
# --------------------------------------------------------------------------
def _scalar(value):
    """Unwrap the 0-d / 1-element arrays scipy leaves on MATLAB scalars."""
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
    """The roster of analyzed cells saved by the MATLAB script.

    One row per saved entry, in file order, with ``index`` as the key
    :func:`load_cell` takes. Scalars only -- the filters, nonlinearities and
    generator signals stay on disk until :func:`load_cell` asks for one.

    ``duplicate`` marks entries whose (date, cell, mode) triple appears more
    than once. The MATLAB appended to the struct array without checking, so
    re-analyzing a cell added a second entry rather than replacing the first;
    those rows are kept and flagged rather than dropped, since which one is
    current can only be decided by looking.
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
            row[f'half_point_{direction}'] = _numeric(
                getattr(getattr(entry, 'halfPoint', None), direction, np.nan))
            model = getattr(getattr(entry, 'lnModel', None), direction, None)
            row[f'r2_{direction}'] = _numeric(getattr(model, 'r2', np.nan))
        rows.append(row)
    frame = pd.DataFrame(rows)

    # epochLen is normally the epoch duration in ms, but the MATLAB wrote
    # `selectedNodes{1}.parent.splitValue`, which is whatever the parent tree
    # node happened to split on. Where that parent was the NDF node the field
    # holds an NDF list instead, so keep the raw text and add a numeric column
    # that is NaN when it is not a duration.
    frame['epoch_len_ms'] = pd.to_numeric(frame['epoch_len'], errors='coerce')
    frame['cell_key'] = (frame['exp_date'] + '/' + frame['cell_label']
                         + '/' + frame['rec_type'])
    frame['duplicate'] = frame['cell_key'].duplicated(keep=False)
    frame['is_example'] = (frame['example_cell'].str.upper()
                           .str.startswith('Y').fillna(False))
    if show:
        print(f'{len(frame)} saved cells | {frame.exp_date.nunique()} dates | '
              f'{int(frame.duplicate.sum())} rows in duplicated '
              f'(date, cell, mode) groups')
        print(pd.crosstab(frame.cell_type, frame.rec_type).to_string())
    return frame


@dataclass
class LNModel:
    """One fitted linear-nonlinear model: filter, measured nonlinearity, r^2.

    ``filter_time_s`` is in seconds -- the saved ``filterTimeStamps`` run 0.001
    to 1.0 over 1000 points, i.e. the 1000 ms filter the MATLAB configured,
    expressed in seconds.
    """

    direction: str
    r2: float
    filter: np.ndarray
    filter_time_s: np.ndarray
    nl_x: np.ndarray
    nl_y: np.ndarray
    phase: Optional[str] = None

    def sigmoid_fit(self) -> Dict[str, float]:
        """Refit the saved nonlinearity; see the module docstring."""
        return fit_sigmoid(self.nl_x, self.nl_y)

    @property
    def biphasic_index(self) -> float:
        """(peak - |trough|) / (peak + |trough|) of the filter.

        The MATLAB's ``bpi``: +1 for a purely monophasic filter, 0 when the two
        lobes are equal.
        """
        peak, trough = np.nanmax(self.filter), abs(np.nanmin(self.filter))
        total = peak + trough
        return float((peak - trough) / total) if total else np.nan

    @property
    def time_to_peak_ms(self) -> float:
        """Time of the filter's largest excursion, in milliseconds."""
        if self.filter.size == 0:
            return np.nan
        index = int(np.nanargmax(np.abs(self.filter)))
        return float(self.filter_time_s[index] * 1e3)


@dataclass
class CellRecord:
    """Everything saved for one analyzed cell."""

    index: int
    exp_date: str
    cell_label: str
    cell_type: str
    rec_type: str
    fit_mode: str
    example_cell: str
    epoch_len: str
    bin_time_s: Dict[str, np.ndarray] = field(default_factory=dict)
    bin_average: Dict[str, np.ndarray] = field(default_factory=dict)
    bin_sem: Dict[str, np.ndarray] = field(default_factory=dict)
    time_const_s: Dict[str, float] = field(default_factory=dict)
    half_point_s: Dict[str, float] = field(default_factory=dict)
    ln_model: Dict[str, LNModel] = field(default_factory=dict)
    temporal_ln: Dict[str, Dict[str, LNModel]] = field(default_factory=dict)
    phase_time_s: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def key(self) -> str:
        return f'{self.exp_date}/{self.cell_label}/{self.rec_type}'

    @property
    def units(self) -> str:
        return 'firing rate (Hz)' if self.rec_type == 'extracellular' else 'current (pA)'

    def __repr__(self) -> str:
        return (f'<CellRecord {self.key} | {self.cell_type} | '
                f"tau {self.time_const_s.get('low', float('nan')):.1f}/"
                f"{self.time_const_s.get('high', float('nan')):.1f} s>")


def _model_from(struct, direction: str, phase: Optional[str] = None) -> Optional[LNModel]:
    if struct is None:
        return None
    filt = np.atleast_1d(np.asarray(getattr(struct, 'filter', []), dtype=float))
    stamps = np.atleast_1d(np.asarray(
        getattr(struct, 'filterTimeStamps', []), dtype=float))
    if stamps.size != filt.size:
        stamps = np.arange(filt.size, dtype=float)
    return LNModel(
        direction=direction, phase=phase,
        r2=_numeric(getattr(struct, 'r2', np.nan)),
        filter=filt, filter_time_s=stamps,
        nl_x=np.atleast_1d(np.asarray(getattr(struct, 'nlX', []), dtype=float)),
        nl_y=np.atleast_1d(np.asarray(getattr(struct, 'nlY', []), dtype=float)))


def load_cell(index, path=None) -> CellRecord:
    """The saved arrays for one roster row.

    ``index`` is the roster's ``index`` column, or a row/Series carrying it.
    ``generatorSignal`` is deliberately not read: it is 123 MB of the file and
    nothing downstream of the saved analysis uses it.
    """
    if isinstance(index, (pd.Series, dict)):
        index = int(index['index'])
    entries = _load_mat(str(summary_path(path)))
    index = int(index)
    if not 0 <= index < entries.size:
        raise IndexError(f'index {index} outside 0..{entries.size - 1}')
    entry = entries[index]

    record = CellRecord(
        index=index,
        exp_date=_text(getattr(entry, 'expDate', '')),
        cell_label=_text(getattr(entry, 'cellLabel', '')),
        cell_type=_text(getattr(entry, 'cellType', '')),
        rec_type=_text(getattr(entry, 'recType', '')),
        fit_mode=_text(getattr(entry, 'fitMode', '')),
        example_cell=_text(getattr(entry, 'exampleCell', '')),
        epoch_len=_text(getattr(entry, 'epochLen', '')))

    for direction in STEP_DIRECTIONS:
        for name, target in (('binTimestamps', record.bin_time_s),
                             ('binAverage', record.bin_average),
                             ('binSte', record.bin_sem)):
            values = getattr(getattr(entry, name, None), direction, None)
            target[direction] = (np.atleast_1d(np.asarray(values, dtype=float))
                                 if values is not None else np.array([]))
        record.time_const_s[direction] = _numeric(
            getattr(getattr(entry, 'timeConsts', None), direction, np.nan))
        record.half_point_s[direction] = _numeric(
            getattr(getattr(entry, 'halfPoint', None), direction, np.nan))
        model = _model_from(
            getattr(getattr(entry, 'lnModel', None), direction, None), direction)
        if model is not None:
            record.ln_model[direction] = model

        block = getattr(getattr(entry, 'temporalLNModel', None), direction, None)
        phase_models = {}
        for phase in PHASES:
            sub = _model_from(getattr(block, phase, None), direction, phase)
            if sub is not None and sub.filter.size:
                phase_models[phase] = sub
        record.temporal_ln[direction] = phase_models

    steps = getattr(getattr(entry, 'temporalLNModel', None), 'timeSteps', None)
    record.phase_time_s = (np.atleast_1d(np.asarray(steps, dtype=float))
                           if steps is not None else np.array([]))
    return record


# --------------------------------------------------------------------------
# model fitting -- cascadegraph, the library the MATLAB used
# --------------------------------------------------------------------------
def fit_sigmoid(nl_x, nl_y, optim_iters: int = 5) -> Dict[str, float]:
    """Fit ``alpha * Phi(beta * x + gamma) + epsilon`` with cascadegraph.

    Uses ``cascadegraph.SigmoidNlNode``, the Python port of the same node class
    the MATLAB fitted, so the parameter names and the model are identical.
    Returns NaNs rather than raising when the fit fails, so a population loop
    is not stopped by one bad cell.
    """
    from cascadegraph import SigmoidNlNode

    x = np.asarray(nl_x, dtype=float)
    y = np.asarray(nl_y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    failed = {k: np.nan for k in ('alpha', 'beta', 'gamma', 'epsilon', 'r2')}
    if x.size < 5:
        return failed

    node = SigmoidNlNode()
    # The MATLAB's params0 for this protocol: twice the response spread for the
    # maximum, unit steepness, no shift.
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
    from cascadegraph import SigmoidNlNode
    node = SigmoidNlNode()
    return node.process_temp_params(
        np.array([alpha, beta, gamma, epsilon]), np.asarray(x, dtype=float))


def fit_ln_model(stimulus, response, sampling_interval: float,
                 filter_length_s: float = 1.0,
                 frequency_cutoff: Optional[float] = None,
                 correct_stim_power: bool = True,
                 n_bins: int = 100) -> Dict[str, object]:
    """Fit an LN model to raw epochs, the way ``computeLNmodel.m`` does.

    ``stimulus`` and ``response`` are (epochs x time) matrices on the same
    clock. Every step is cascadegraph's: :func:`compute_filter` for the linear
    stage, :func:`convolve_filter_with_stim` for the generator signal,
    :func:`sample_nl` to bin the input-output relation, and
    :class:`SigmoidNlNode` for the static nonlinearity.

    This is the path for recordings that have *not* been through the MATLAB.
    For the 53 saved cells, :func:`load_cell` already has the fitted result and
    this does not need to run.
    """
    from cascadegraph import (compute_filter, convolve_filter_with_stim,
                              sample_nl, compute_variance_explained)

    stimulus = np.atleast_2d(np.asarray(stimulus, dtype=float))
    response = np.atleast_2d(np.asarray(response, dtype=float))
    if stimulus.shape != response.shape:
        raise ValueError(f'stimulus {stimulus.shape} and response '
                         f'{response.shape} must have the same shape')
    filter_pts = int(round(filter_length_s / sampling_interval))

    # compute_filter requires frequency_cutoff and sampling_interval together,
    # so only hand it the pair when a cutoff was actually asked for.
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
    return {
        'filter': np.asarray(filter_causal, dtype=float),
        'filter_time_s': np.arange(filter_pts) * sampling_interval,
        'nl_x': np.asarray(nl_x, dtype=float),
        'nl_y': np.asarray(nl_y, dtype=float),
        'generator': generator, 'params': params, 'r2': r2,
    }


def fit_exponential(time_s, values) -> Tuple[float, np.ndarray]:
    """Single exponential to an adaptation trace; returns ``(tau, fitted)``.

    The MATLAB's ``fitExp`` returns a rate ``k`` and reports ``1/k``; this
    returns that time constant directly, in the units of ``time_s``.
    """
    from scipy.optimize import curve_fit
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(values, dtype=float)
    keep = np.isfinite(t) & np.isfinite(y)
    t, y = t[keep], y[keep]
    if t.size < 4:
        return np.nan, np.full(np.size(time_s), np.nan)

    def model(x, amplitude, rate, offset):
        return amplitude * np.exp(-rate * x) + offset

    try:
        params, _ = curve_fit(
            model, t, y, p0=[float(y[0] - y[-1]) or 1.0, 0.1, float(y[-1])],
            maxfev=20000)
    except Exception:
        return np.nan, np.full_like(t, np.nan)
    rate = params[1]
    tau = float(1.0 / rate) if rate and np.isfinite(rate) else np.nan
    return tau, model(t, *params)


# --------------------------------------------------------------------------
# population tables
# --------------------------------------------------------------------------
def select_population(roster: pd.DataFrame, rec_type: str = 'extracellular',
                      cell_types: Optional[Sequence[str]] = None,
                      drop_duplicates: bool = True,
                      examples_only: bool = False) -> pd.DataFrame:
    """Filter the roster for a population figure.

    ``drop_duplicates`` keeps the *last* entry of each (date, cell, mode)
    group, since the MATLAB appended a re-analysis rather than replacing the
    original. Set it False to see every saved entry.
    """
    frame = roster[roster.rec_type.eq(rec_type)].copy()
    if cell_types is not None:
        frame = frame[frame.cell_type.isin(list(cell_types))]
    if examples_only:
        frame = frame[frame.is_example]
    if drop_duplicates:
        frame = frame.drop_duplicates('cell_key', keep='last')
    return frame.reset_index(drop=True)


def adaptation_traces(population: pd.DataFrame, path=None,
                      normalize: bool = True) -> pd.DataFrame:
    """Per-cell adaptation traces, long form.

    ``normalize`` divides both directions by that cell's peak on the
    low-to-high trace, which is the MATLAB's normalization: it puts the two
    directions of one cell on a shared scale without flattening the difference
    between them.
    """
    rows = []
    for _, row in population.iterrows():
        record = load_cell(row, path=path)
        reference = np.nanmax(record.bin_average.get('high', np.array([np.nan])))
        if normalize and (not np.isfinite(reference) or reference == 0):
            continue
        for direction in STEP_DIRECTIONS:
            time_s = record.bin_time_s.get(direction, np.array([]))
            values = record.bin_average.get(direction, np.array([]))
            if time_s.size == 0 or values.size != time_s.size:
                continue
            if normalize:
                values = values / reference
            for t, v in zip(time_s, values):
                rows.append({'index': record.index, 'cell_key': record.key,
                             'cell_type': record.cell_type,
                             'rec_type': record.rec_type,
                             'direction': direction, 'time_s': float(t),
                             'value': float(v)})
    return pd.DataFrame(rows)


def population_adaptation(population: pd.DataFrame, path=None,
                          normalize: bool = True, n_bins: int = 30) -> pd.DataFrame:
    """Mean adaptation trace per (cell type, direction), pooled over cells.

    Cells were binned on their own time base, which differs between
    recordings, so the traces are re-binned onto one grid before averaging --
    the MATLAB's ``averageInBins``. Returns mean, SEM and the contributing cell
    count per bin.
    """
    traces = adaptation_traces(population, path=path, normalize=normalize)
    if traces.empty:
        return traces
    edges = np.linspace(traces.time_s.min(), traces.time_s.max(), n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    traces = traces.assign(
        bin=np.clip(np.digitize(traces.time_s, edges) - 1, 0, n_bins - 1))
    grouped = (traces.groupby(['cell_type', 'direction', 'bin'], dropna=False)
               .agg(mean=('value', 'mean'),
                    sem=('value', lambda s: float(s.std(ddof=1) / np.sqrt(len(s)))
                         if len(s) > 1 else np.nan),
                    n_cells=('cell_key', 'nunique'))
               .reset_index())
    grouped['time_s'] = centres[grouped['bin'].to_numpy()]
    return grouped.drop(columns='bin')


def time_constant_table(population: pd.DataFrame, path=None,
                        max_tau_s: float = 120.0) -> pd.DataFrame:
    """Saved adaptation time constants per cell, plus a Python refit.

    ``tau_saved_*`` is what the MATLAB stored; ``tau_refit_*`` is a single
    exponential fitted here to the saved trace. They agree closely where the
    fit was sound, so the useful column is ``implausible``: True when either
    direction is negative, non-finite, or beyond ``max_tau_s``, which is how
    a failed exponential fit shows up. The MATLAB kept those values without
    flagging them and its population averages include them.
    """
    rows = []
    for _, row in population.iterrows():
        record = load_cell(row, path=path)
        entry = {'index': record.index, 'cell_key': record.key,
                 'cell_type': record.cell_type, 'rec_type': record.rec_type}
        for direction in STEP_DIRECTIONS:
            entry[f'tau_saved_{direction}'] = record.time_const_s.get(direction, np.nan)
            tau, _ = fit_exponential(record.bin_time_s.get(direction, []),
                                     record.bin_average.get(direction, []))
            entry[f'tau_refit_{direction}'] = tau
        saved = [entry[f'tau_saved_{d}'] for d in STEP_DIRECTIONS]
        entry['implausible'] = any(
            (not np.isfinite(v)) or v <= 0 or v > max_tau_s for v in saved)
        rows.append(entry)
    return pd.DataFrame(rows)


def population_filters(population: pd.DataFrame, path=None,
                       normalize: bool = True) -> pd.DataFrame:
    """Mean temporal filter per (cell type, direction), pooled over cells.

    Both directions are divided by the same per-cell scalar taken from the
    high-to-low filter -- its peak for spikes, the magnitude of its trough for
    voltage-clamp currents, which are inward and therefore negative. Using one
    scalar for both keeps the gain change between directions visible; scaling
    each separately would normalize it away.
    """
    rows = []
    for _, row in population.iterrows():
        record = load_cell(row, path=path)
        reference_model = record.ln_model.get('low')
        if reference_model is None or reference_model.filter.size == 0:
            continue
        if record.rec_type == 'extracellular':
            reference = float(np.nanmax(reference_model.filter))
        else:
            reference = float(abs(np.nanmin(reference_model.filter)))
        if normalize and (not np.isfinite(reference) or reference == 0):
            continue
        for direction, model in record.ln_model.items():
            values = model.filter / reference if normalize else model.filter
            for t, v in zip(model.filter_time_s, values):
                rows.append({'cell_key': record.key, 'cell_type': record.cell_type,
                             'direction': direction, 'time_s': float(t),
                             'value': float(v)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (frame.groupby(['cell_type', 'direction', 'time_s'], dropna=False)
            .agg(mean=('value', 'mean'),
                 sem=('value', lambda s: float(s.std(ddof=1) / np.sqrt(len(s)))
                      if len(s) > 1 else np.nan),
                 n_cells=('cell_key', 'nunique'))
            .reset_index())


def population_nonlinearities(population: pd.DataFrame, path=None,
                              normalize: bool = True,
                              n_bins: int = 40) -> pd.DataFrame:
    """Mean nonlinearity per (cell type, direction), pooled over cells.

    Cells do not share a generator-signal axis, so each curve is resampled onto
    a common quantile grid before averaging. Normalization follows
    :func:`population_filters` -- one scalar from the low-to-high curve,
    applied to both directions.
    """
    rows = []
    for _, row in population.iterrows():
        record = load_cell(row, path=path)
        reference_model = record.ln_model.get('high')
        if reference_model is None or reference_model.nl_y.size == 0:
            continue
        if record.rec_type == 'extracellular':
            reference = float(np.nanmax(reference_model.nl_y))
        else:
            reference = float(abs(np.nanmin(reference_model.nl_y)))
        if normalize and (not np.isfinite(reference) or reference == 0):
            continue
        for direction, model in record.ln_model.items():
            if model.nl_x.size < 5:
                continue
            values = model.nl_y / reference if normalize else model.nl_y
            for x, y in zip(model.nl_x, values):
                rows.append({'cell_key': record.key, 'cell_type': record.cell_type,
                             'direction': direction, 'generator': float(x),
                             'value': float(y)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    out = []
    for (cell_type, direction), block in frame.groupby(['cell_type', 'direction']):
        edges = np.unique(np.nanquantile(block.generator,
                                         np.linspace(0, 1, n_bins + 1)))
        if edges.size < 3:
            continue
        centres = 0.5 * (edges[:-1] + edges[1:])
        binned = np.clip(np.digitize(block.generator, edges) - 1, 0, centres.size - 1)
        summary = (block.assign(bin=binned).groupby('bin')
                   .agg(mean=('value', 'mean'),
                        sem=('value', lambda s: float(s.std(ddof=1) / np.sqrt(len(s)))
                             if len(s) > 1 else np.nan),
                        n_cells=('cell_key', 'nunique'))
                   .reset_index())
        summary['generator'] = centres[summary['bin'].to_numpy()]
        summary['cell_type'] = cell_type
        summary['direction'] = direction
        out.append(summary.drop(columns='bin'))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def temporal_summary(population: pd.DataFrame, path=None) -> pd.DataFrame:
    """Filter time-to-peak and biphasic index per phase after the step.

    This is the phase-resolved model reduced to two numbers per window, which
    is how the MATLAB's kinetics figures read it: recovery shows up as the
    filter speeding up and becoming more biphasic as the phases advance.
    """
    rows = []
    for _, row in population.iterrows():
        record = load_cell(row, path=path)
        for direction, phase_models in record.temporal_ln.items():
            for order, (phase, model) in enumerate(phase_models.items()):
                time_s = (float(record.phase_time_s[order])
                          if order < record.phase_time_s.size else np.nan)
                rows.append({
                    'cell_key': record.key, 'cell_type': record.cell_type,
                    'rec_type': record.rec_type, 'direction': direction,
                    'phase': phase, 'phase_order': order, 'phase_time_s': time_s,
                    'r2': model.r2, 'time_to_peak_ms': model.time_to_peak_ms,
                    'biphasic_index': model.biphasic_index})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
_DIRECTION_COLORS = {'low': '#33517A', 'high': '#B5802A'}


def plot_cell(record: CellRecord, figsize: Tuple[float, float] = (11.0, 7.5)):
    """One saved cell: adaptation trace, filter, nonlinearity, phase kinetics."""
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    colors = _DIRECTION_COLORS
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    ax = axes[0][0]
    for direction in STEP_DIRECTIONS:
        time_s = record.bin_time_s.get(direction, np.array([]))
        values = record.bin_average.get(direction, np.array([]))
        errors = record.bin_sem.get(direction, np.array([]))
        if time_s.size == 0 or values.size != time_s.size:
            continue
        if errors.size == values.size:
            ax.fill_between(time_s, values - errors, values + errors,
                            color=colors[direction], alpha=0.18, lw=0)
        ax.plot(time_s, values, 'o-', ms=3, lw=1.6, color=colors[direction],
                label=f"{STEP_LABELS[direction]}  "
                      f"τ={record.time_const_s.get(direction, float('nan')):.1f} s")
    ax.set_xlabel('time after the step (s)')
    ax.set_ylabel(record.units)
    ax.set_title('adaptation after the mean step', fontsize=9)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0][1]
    for direction, model in record.ln_model.items():
        ax.plot(model.filter_time_s * 1e3, model.filter, lw=1.8,
                color=colors[direction], label=STEP_LABELS[direction])
    ax.axhline(0, color='#888888', lw=0.8, ls='--')
    ax.set_xlabel('filter time (ms)')
    ax.set_ylabel('filter (a.u.)')
    ax.set_title('temporal filter', fontsize=9)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1][0]
    for direction, model in record.ln_model.items():
        ax.plot(model.nl_x, model.nl_y, 'o', ms=2.5, alpha=0.55,
                color=colors[direction],
                label=f'{STEP_LABELS[direction]} (r²={model.r2:.2f})')
        params = model.sigmoid_fit()
        if np.isfinite(params['alpha']):
            grid = np.linspace(np.nanmin(model.nl_x), np.nanmax(model.nl_x), 200)
            ax.plot(grid, sigmoid(grid, params['alpha'], params['beta'],
                                  params['gamma'], params['epsilon']),
                    lw=1.6, color=colors[direction])
    ax.set_xlabel('generator signal')
    ax.set_ylabel(record.units)
    ax.set_title('nonlinearity (points saved, curve refitted)', fontsize=9)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1][1]
    for direction, phase_models in record.temporal_ln.items():
        if not phase_models:
            continue
        times, peaks = [], []
        for order, model in enumerate(phase_models.values()):
            times.append(float(record.phase_time_s[order])
                         if order < record.phase_time_s.size else order)
            peaks.append(model.time_to_peak_ms)
        ax.plot(times, peaks, 'o-', ms=4, lw=1.6, color=colors[direction],
                label=STEP_LABELS[direction])
    ax.set_xlabel('time after the step (s)')
    ax.set_ylabel('filter time-to-peak (ms)')
    ax.set_title('filter kinetics across the step', fontsize=9)
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle(f'{record.key} | {record.cell_type} | example: {record.example_cell}',
                 fontsize=11)
    fig.tight_layout()
    return fig


def plot_population_adaptation(population: pd.DataFrame, path=None,
                               normalize: bool = True, n_bins: int = 30,
                               min_cells: Optional[int] = None,
                               figsize: Tuple[float, float] = (7.6, 5.0)):
    """Population adaptation traces, one panel, colored by cell type.

    Epoch lengths differ between recordings (50-60 s here), so the late bins
    rest on only the longest epochs -- on this dataset the count falls from 18
    cells to 2 after 50 s. ``min_cells`` drops bins below a threshold, and
    defaults to half the maximum for that series, so the drawn curve is one
    sample rather than a full population that thins into a handful of cells.
    Pass ``min_cells=1`` to draw everything.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    table = population_adaptation(population, path=path,
                                  normalize=normalize, n_bins=n_bins)
    if table.empty:
        print('no adaptation traces to plot')
        return None
    cell_types = sorted(table.cell_type.unique())
    colors = style.colors_for_conditions(cell_types)
    fig, ax = plt.subplots(figsize=figsize)
    for cell_type in cell_types:
        for direction in STEP_DIRECTIONS:
            block = table[table.cell_type.eq(cell_type)
                          & table.direction.eq(direction)].sort_values('time_s')
            if block.empty:
                continue
            n = int(block.n_cells.max())
            floor = int(np.ceil(n / 2)) if min_cells is None else int(min_cells)
            block = block[block.n_cells >= floor]
            if block.empty:
                continue
            ax.fill_between(block.time_s, block['mean'] - block['sem'].fillna(0),
                            block['mean'] + block['sem'].fillna(0),
                            color=colors[cell_type], alpha=0.14, lw=0)
            ax.plot(block.time_s, block['mean'],
                    ls='-' if direction == 'low' else '--',
                    lw=1.9, color=colors[cell_type],
                    label=f'{cell_type} {STEP_LABELS[direction]} (n={n})')
    ax.set_xlabel('time after the mean step (s)')
    ax.set_ylabel('normalized response' if normalize else 'response')
    ax.legend(frameon=False, fontsize=7)
    mode = population.rec_type.iloc[0] if len(population) else ''
    fig.suptitle(f'Adaptation after a mean-luminance step — {mode}', fontsize=11)
    fig.tight_layout()
    return fig


def plot_population_ln(population: pd.DataFrame, path=None, normalize: bool = True,
                       figsize: Tuple[float, float] = (11.0, 4.4)):
    """Population filters and nonlinearities side by side."""
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    filters = population_filters(population, path=path, normalize=normalize)
    nonlin = population_nonlinearities(population, path=path, normalize=normalize)
    if filters.empty and nonlin.empty:
        print('no LN models to plot')
        return None
    cell_types = sorted(set(filters.cell_type.unique()) | set(nonlin.cell_type.unique()))
    colors = style.colors_for_conditions(cell_types)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for cell_type in cell_types:
        for direction in STEP_DIRECTIONS:
            block = filters[filters.cell_type.eq(cell_type)
                            & filters.direction.eq(direction)].sort_values('time_s')
            if not block.empty:
                n = int(block.n_cells.max())
                axes[0].fill_between(block.time_s * 1e3,
                                     block['mean'] - block['sem'].fillna(0),
                                     block['mean'] + block['sem'].fillna(0),
                                     color=colors[cell_type], alpha=0.14, lw=0)
                axes[0].plot(block.time_s * 1e3, block['mean'],
                             ls='-' if direction == 'low' else '--',
                             lw=1.8, color=colors[cell_type],
                             label=f'{cell_type} {STEP_LABELS[direction]} (n={n})')
            curve = nonlin[nonlin.cell_type.eq(cell_type)
                           & nonlin.direction.eq(direction)].sort_values('generator')
            if curve.empty:
                continue
            axes[1].fill_between(curve.generator,
                                 curve['mean'] - curve['sem'].fillna(0),
                                 curve['mean'] + curve['sem'].fillna(0),
                                 color=colors[cell_type], alpha=0.14, lw=0)
            axes[1].plot(curve.generator, curve['mean'],
                         ls='-' if direction == 'low' else '--',
                         lw=1.8, color=colors[cell_type])
    axes[0].axhline(0, color='#888888', lw=0.8, ls='--')
    axes[0].set_xlabel('filter time (ms)')
    axes[0].set_ylabel('normalized filter' if normalize else 'filter')
    axes[0].set_title('temporal filter', fontsize=9)
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_xlabel('generator signal')
    axes[1].set_ylabel('normalized response' if normalize else 'response')
    axes[1].set_title('nonlinearity', fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# browsing the wider database (the not-yet-analyzed direction)
# --------------------------------------------------------------------------
def find_blocks(exp_names: Optional[Sequence[str]] = None, show: bool = True,
                height: int = 400) -> pd.DataFrame:
    """Every VariableMeanNoise epoch block in the DataJoint database.

    This is the counterpart to :func:`load_summary`: the saved ``.mat`` holds
    the cells that have been analyzed, and this finds recordings in general.
    The two overlap only where a saved date was also ingested -- most saved
    dates predate this database, so expect little overlap and use this to find
    *new* recordings rather than to re-reach the saved ones.
    """
    from retinanalysis.SCutils import explore as sc

    frame = sc.find_blocks(PROTOCOL, show=False)
    if frame.empty:
        if show:
            print(f'no blocks found for {PROTOCOL}')
        return frame
    if exp_names is not None:
        frame = frame[frame.exp_name.isin(list(exp_names))].copy()
    frame = frame.sort_values(['exp_name', 'block_id']).reset_index(drop=True)
    if show:
        print(f'{len(frame)} blocks | {frame.exp_name.nunique()} experiments')
        columns = [c for c in ('exp_name', 'cell_label', 'cell_type', 'block_id',
                               'ndfs', 'ndf_fw', 'filter_wheel_ndf', 'protocol_name')
                   if c in frame.columns]
        sc.scroll_table(frame[columns], height=height)
    return frame


def unanalyzed_dates(roster: Optional[pd.DataFrame] = None,
                     blocks: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Recorded dates that carry no saved analysis yet.

    Dates are compared on the calendar date alone, since the saved file writes
    ``yyyy/mm/dd`` while the database writes ``yyyy-mm-dd_R`` with a rig suffix.
    """
    roster = load_summary() if roster is None else roster
    blocks = find_blocks(show=False) if blocks is None else blocks
    if blocks.empty:
        return blocks
    analyzed = set(roster.exp_date.str.replace('/', '-', regex=False))
    frame = blocks.copy()
    frame['calendar_date'] = frame.exp_name.astype(str).str.slice(0, 10)
    frame['has_saved_analysis'] = frame.calendar_date.isin(analyzed)
    # sc.find_blocks returns different columns for different protocols -- cell
    # labels are present for some and not others -- so aggregate over whatever
    # this protocol actually has rather than assuming a schema.
    agg = {'blocks': ('block_id', 'nunique')}
    for column, name in (('cell_label', 'cells'), ('ndfs', 'ndf_combinations')):
        if column in frame.columns:
            agg[name] = (column, 'nunique')
    return (frame.groupby(['exp_name', 'has_saved_analysis'], dropna=False)
            .agg(**agg).reset_index()
            .sort_values(['has_saved_analysis', 'exp_name']))


# --------------------------------------------------------------------------
# light level: NDF combinations and unit isomerizations
# --------------------------------------------------------------------------
# computeLedUnitIsom.m keeps its own attenuation tables and reference
# isomerization rates. They are transcribed here so the two can be compared
# rather than silently diverging; `retinanalysis.utils.isomerization` is the
# source this module actually computes from.
MATLAB_UV_OD = {
    'two_photon': {'B1': .29, 'B2': .71, 'B3': 1.21, 'B4': 2.54, 'B5': 4.58,
                   'B6': 2.71, 'B7': 5.13},
    'shared_two_photon': {'G1': 1.0060, 'G2': 1.0524, 'G3': 2.1342, 'G4': 2.6278,
                          'G6': .28, 'G7': .59, 'G8': 1.25, 'G9': 2.23},
}
# The gain setting on rig B is an attenuation too, and computeLedUnitIsom folds
# it into the same sum.
MATLAB_GAIN_OD = {'high': 0.0, 'medium': 1.0, 'low': 2.0}
# Isomerizations per rod per second at LED intensity 1 with no attenuation, as
# computeLedUnitIsom defines it. Rig B's comment reads "UV led, with FW and B2
# NDFS, norm to attn"; rig G's reads "UV led, no NDFs".
MATLAB_REFERENCE_ISOM = {'two_photon': 12822.0, 'shared_two_photon': 1377897.0}

_FILTER_WHEEL_OD = {'FW0': 0.0, 'FW05': 0.5, 'FW1': 1.0, 'FW2': 2.0,
                    'FW3': 3.0, 'FW4': 4.0}


def parse_ndf_combination(value) -> Tuple[str, ...]:
    """Split a saved NDF label into tokens.

    Accepts what the various sources produce: a JSON-ish list
    (``["G1","G3","G7"]``), a comma or space separated string, or a sequence.
    Uses ``retinanalysis.utils.isomerization.parse_ndfs`` so the parsing
    matches the rest of the package.
    """
    from retinanalysis.utils.isomerization import parse_ndfs

    if isinstance(value, (list, tuple, set, np.ndarray)):
        tokens = [str(v) for v in value]
    else:
        tokens = list(parse_ndfs(value))
    return tuple(t.strip().strip('"').strip("'") for t in tokens if str(t).strip())


def ndf_optical_density(tokens: Sequence[str], rig: str, color: str = 'uv',
                        source: str = 'python') -> Tuple[float, List[str]]:
    """Total optical density for a set of NDF tokens.

    ``source='python'`` uses ``utils.isomerization.led_ndf_attenuations``, the
    measured per-rig tables in ``utils.isomerization``. ``source='matlab'``
    uses the transcribed :data:`MATLAB_UV_OD`. Filter-wheel tokens and rig-B
    gain names resolve the same way in both. Returns the total and any tokens
    that had no entry.
    """
    from retinanalysis.utils.isomerization import led_ndf_attenuations

    if source == 'python':
        table = dict(led_ndf_attenuations(rig, color))
    elif source == 'matlab':
        table = dict(MATLAB_UV_OD.get(rig, {}))
    else:
        raise ValueError("source must be 'python' or 'matlab'")
    table.update(_FILTER_WHEEL_OD)
    table.update(MATLAB_GAIN_OD)

    total, missing = 0.0, []
    for token in tokens:
        key = str(token).strip()
        if key in table:
            total += float(table[key])
            continue
        try:                       # a bare number is already an optical density
            total += float(key)
        except ValueError:
            missing.append(key)
    return total, missing


def unit_isomerization(tokens: Sequence[str], rig: str, color: str = 'uv',
                       source: str = 'python',
                       reference: Optional[float] = None) -> float:
    """Isomerizations per rod per second at LED intensity 1, after the NDFs.

    ``reference / 10**total_OD``, the calculation ``computeLedUnitIsom`` does.
    The reference defaults to that function's per-rig value.
    """
    if reference is None:
        reference = MATLAB_REFERENCE_ISOM.get(rig, np.nan)
    total, missing = ndf_optical_density(tokens, rig, color, source=source)
    if missing or not np.isfinite(reference):
        return np.nan
    return float(reference) / 10.0 ** total


def ndf_table_comparison(rig: str, color: str = 'uv') -> pd.DataFrame:
    """Per-token optical density from both tables, side by side.

    The point of this table is to make disagreement visible. Where a token is
    in one table only, the other column is NaN; where both have it, ``delta_od``
    is the difference and ``light_ratio`` the factor it puts on the light level.
    """
    from retinanalysis.utils.isomerization import led_ndf_attenuations

    python_table = dict(led_ndf_attenuations(rig, color))
    matlab_table = dict(MATLAB_UV_OD.get(rig, {}))
    rows = []
    for token in sorted(set(python_table) | set(matlab_table)):
        p = python_table.get(token, np.nan)
        m = matlab_table.get(token, np.nan)
        delta = p - m if np.isfinite(p) and np.isfinite(m) else np.nan
        rows.append({'ndf': token, 'python_od': p, 'matlab_od': m,
                     'delta_od': delta,
                     'light_ratio': 10.0 ** delta if np.isfinite(delta) else np.nan})
    return pd.DataFrame(rows)


def isomerization_audit(combinations: Sequence, rig: str, color: str = 'uv',
                        measured: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """Compute the light level for each NDF combination, both ways.

    ``combinations`` is any sequence of NDF labels; ``measured`` is the recorded
    R* where a recording saved one, compared against the computed value as a
    ratio. A ratio far from 1 means the combination's attenuation and the saved
    number disagree, which is the thing worth chasing.
    """
    rows = []
    measured = list(measured) if measured is not None else [None] * len(combinations)
    for value, saved in zip(combinations, measured):
        tokens = parse_ndf_combination(value)
        python_od, missing = ndf_optical_density(tokens, rig, color, 'python')
        matlab_od, matlab_missing = ndf_optical_density(tokens, rig, color, 'matlab')
        row = {
            'ndf_combination': ', '.join(tokens) if tokens else '(none)',
            'n_filters': len(tokens),
            # A token absent from a table contributes nothing to its sum, which
            # would read as "no attenuation" rather than "not known". Blank the
            # total instead, per table, and name the tokens responsible.
            'python_od': python_od if not missing else np.nan,
            'matlab_od': matlab_od if not matlab_missing else np.nan,
            'unknown_tokens': ', '.join(sorted(set(missing) | set(matlab_missing))),
            'isom_python': unit_isomerization(tokens, rig, color, 'python'),
            'isom_matlab': unit_isomerization(tokens, rig, color, 'matlab'),
        }
        if saved is not None and np.isfinite(float(saved)):
            row['isom_saved'] = float(saved)
            row['saved_over_python'] = (
                float(saved) / row['isom_python']
                if np.isfinite(row['isom_python']) and row['isom_python'] else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    'PROTOCOL', 'DEFAULT_SUMMARY_PATH', 'SUMMARY_DIR', 'STEP_DIRECTIONS',
    'STEP_LABELS', 'PHASES', 'MATLAB_UV_OD', 'MATLAB_REFERENCE_ISOM',
    'LNModel', 'CellRecord', 'summary_path', 'load_summary', 'load_cell',
    'sigmoid', 'fit_sigmoid', 'fit_ln_model', 'fit_exponential',
    'select_population', 'adaptation_traces', 'population_adaptation',
    'time_constant_table', 'population_filters', 'population_nonlinearities',
    'temporal_summary', 'plot_cell', 'plot_population_adaptation',
    'plot_population_ln', 'find_blocks', 'unanalyzed_dates',
    'parse_ndf_combination', 'ndf_optical_density', 'unit_isomerization',
    'ndf_table_comparison', 'isomerization_audit',
]
