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
    # r2 above is the held-out value where one could be computed. The two below
    # are kept apart because they are easy to mistake for it and are both
    # optimistic: r2_train is measured on the epochs the filter was fitted to,
    # and nl_r2 is the sigmoid's fit to the ~100 binned nonlinearity points,
    # which is near 1 almost regardless of how well the model predicts a trace.
    r2_train: float = np.nan
    nl_r2: float = np.nan
    n_train: int = 0
    n_test: int = 0
    example_time_s: np.ndarray = field(default_factory=lambda: np.array([]))
    example_measured: np.ndarray = field(default_factory=lambda: np.array([]))
    example_predicted: np.ndarray = field(default_factory=lambda: np.array([]))

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


def block_cells(blocks: pd.DataFrame) -> pd.DataFrame:
    """Attach ``cell_label`` and ``cell_type`` to a block table.

    ``sc.find_blocks`` returns protocol and light metadata but no cell
    identity, so this walks EpochBlock -> EpochGroup -> Cell once for the whole
    table rather than per block.
    """
    from retinanalysis.config import schema

    if blocks.empty:
        return blocks
    ids = sorted({int(b) for b in blocks.block_id})
    # EpochBlock.parent_id -> EpochGroup.id -> EpochGroup.parent_id -> Cell.id.
    # Every table keys on `id`, and the cell carries `label` and `type`.
    block_rows = (schema.EpochBlock() & [f'id={i}' for i in ids]
                  ).to_pandas().reset_index()[['id', 'parent_id']]
    if block_rows.empty:
        return blocks
    group_ids = sorted({int(g) for g in block_rows.parent_id.dropna()})
    groups = (schema.EpochGroup() & [f'id={g}' for g in group_ids]
              ).to_pandas().reset_index()[['id', 'parent_id']]
    cell_ids = sorted({int(c) for c in groups.parent_id.dropna()})
    cells = (schema.Cell() & [f'id={c}' for c in cell_ids]
             ).to_pandas().reset_index()[['id', 'label', 'type']]

    joined = (block_rows.rename(columns={'id': 'block_id', 'parent_id': 'group_id'})
              .merge(groups.rename(columns={'id': 'group_id', 'parent_id': 'cell_id'}),
                     on='group_id', how='left')
              .merge(cells.rename(columns={'id': 'cell_id', 'label': 'cell_label',
                                           'type': 'cell_type'}),
                     on='cell_id', how='left'))
    out = blocks.merge(joined[['block_id', 'cell_label', 'cell_type']],
                       on='block_id', how='left')
    out['cell_type_short'] = (out['cell_type'].astype(str)
                              .str.split('\\').str[-1].replace('nan', ''))
    return out


def resolve_block_mode(exp_name: str, block_id: int,
                       amp_data: Optional[np.ndarray] = None,
                       sample_rate: float = 1e4) -> Dict[str, object]:
    """Decide how one block was recorded, from the amplifier rather than a label.

    These blocks carry no ``onlineAnalysis`` -- the experimenter's menu choice
    is simply absent -- so the reading determines the mode rather than
    overruling it, which is the ``'none'`` path of
    :func:`SCutils.recording_mode.resolve_recording_mode`:

    * series resistance above zero means the cell was held whole-cell, and the
      polarity comes from the sign of the current: inward (negative mean) is
      ``exc``, outward is ``inh``;
    * a reading of exactly zero is not by itself cell-attached -- it also
      happens when whole-cell compensation was never run -- so it is confirmed
      against the trace, and only a trace that really contains spikes is
      called ``extracellular``.

    ``amp_data`` is optional; without it the block is loaded to get it.
    """
    import retinanalysis as ra
    from retinanalysis.SCutils.recording_mode import (read_series_resistance,
                                                      resolve_recording_mode)

    # Symphony writes seriesResistance into the epoch parameters for this
    # protocol, which is both faster and more reliable than re-reading the h5;
    # fall back to the h5 reader when the parameter is absent.
    rs_median = np.nan
    params = epoch_parameters(int(block_id))
    if 'seriesResistance' in params:
        values = pd.to_numeric(params['seriesResistance'], errors='coerce').dropna()
        if len(values):
            rs_median = float(np.median(values))
    if not np.isfinite(rs_median):
        try:
            rs = np.asarray(read_series_resistance(exp_name, int(block_id)),
                            dtype=float)
            rs_median = float(np.nanmedian(rs)) if rs.size else np.nan
        except Exception:
            rs_median = np.nan

    if amp_data is None:
        block = ra.SCResponseBlock(exp_name, int(block_id), b_spiking=False,
                                   b_LED=True, verbose=False)
        amp_data = np.asarray(block.amp_data, dtype=float)
        sample_rate = float(block.amp_sample_rate)

    mode, note = resolve_recording_mode('none', rs_median, amp_data=amp_data,
                                        sample_rate=sample_rate)
    return {'rec_type': mode, 'rec_note': note,
            'series_resistance_mohm': (rs_median / 1e6
                                       if np.isfinite(rs_median) and rs_median > 1e3
                                       else rs_median),
            'mean_current_pa': float(np.nanmean(amp_data))}


def condition_table(exp_names: Optional[Sequence[str]] = None,
                    blocks: Optional[pd.DataFrame] = None,
                    roster: Optional[pd.DataFrame] = None,
                    resolve_modes: bool = True,
                    show: bool = True, height: int = 400) -> pd.DataFrame:
    """One row per block: cell, stimulus, and the recording type as verified.

    This is the table to pick an ``entry`` from and hand to
    :func:`analyze_condition`. ``lightMean`` and ``stdv`` are read off the
    recorded epochs, so a row listing two means is a block that stepped between
    them.

    ``resolve_modes`` loads each block to check the recording type against the
    amplifier (see :func:`resolve_block_mode`); it is the slow part, so pass
    ``exp_names`` to keep it to the experiments being worked on. With it off the
    stimulus columns still fill in and ``rec_type`` is blank.

    ``roster_index`` links a row back to the saved data-entry list where the
    date matches, so the MATLAB's own result for that cell can be pulled up.
    """
    blocks = find_blocks(show=False) if blocks is None else blocks
    if exp_names is not None:
        blocks = blocks[blocks.exp_name.isin(list(exp_names))]
    if blocks.empty:
        return pd.DataFrame()
    blocks = block_cells(blocks)

    roster_by_date = {}
    if roster is not None:
        for date, group in roster.groupby('calendar_date'):
            roster_by_date[date] = ', '.join(str(i) for i in group['index'])

    rows = []
    for _, block in blocks.iterrows():
        block_id = int(block.block_id)
        params = epoch_parameters(block_id)
        if params.empty:
            continue
        row = {
            'entry': len(rows),
            'exp_name': block.exp_name,
            'cell_label': block.get('cell_label', ''),
            'cell_type_short': block.get('cell_type_short', ''),
            'block_id': block_id,
            'n_epochs': len(params),
        }
        for key, name in (('stimTime', 'stimTime_ms'), ('led', 'led'),
                          ('sampleRate', 'sampleRate')):
            values = pd.unique(params[key].dropna()) if key in params else []
            row[name] = values[0] if len(values) == 1 else ', '.join(map(str, values))
        for key in ('lightMean', 'stdv'):
            values = (sorted(pd.to_numeric(params[key], errors='coerce')
                             .dropna().unique()) if key in params else [])
            row[key] = ', '.join(f'{v:g}' for v in values)
            row[f'n_{key}'] = len(values)
        row['roster_index'] = roster_by_date.get(block.get('calendar_date', ''), '')
        if resolve_modes:
            row.update(resolve_block_mode(block.exp_name, block_id))
        rows.append(row)

    frame = pd.DataFrame(rows)
    if show and len(frame):
        columns = [c for c in ('entry', 'exp_name', 'cell_label', 'cell_type_short',
                               'rec_type', 'series_resistance_mohm', 'mean_current_pa',
                               'stimTime_ms', 'led', 'lightMean', 'stdv',
                               'n_epochs', 'block_id', 'roster_index')
                   if c in frame.columns]
        print(f'{len(frame)} block(s) | {frame.exp_name.nunique()} experiment(s)')
        if 'rec_type' in frame:
            print(frame.rec_type.value_counts().to_string())
        from retinanalysis.SCutils import explore as sc
        sc.scroll_table(frame[columns].round(3), height=height,
                        num_cols=('entry', 'series_resistance_mohm',
                                  'mean_current_pa', 'stimTime_ms', 'n_epochs',
                                  'block_id'))
    return frame


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
# 2b. finding the raw files: the saved dates are two days early
# --------------------------------------------------------------------------
# The MATLAB wrote `expDate` from `datestr(epochList.elements(1).startDate)`,
# and it does not agree with the experiment filenames: of the 16 saved dates,
# one lands on a file and fifteen land two days before one. Matching at +2 and
# then confirming on the cell label and cell type resolves 44 of the 45 saved
# cells, and 39 of them agree on label, type and protocol at once, so the
# offset is real rather than a coincidence of a dense recording calendar.
#
# The offsets are searched in this order and the first *confirmed* match wins,
# so a date that really is exact (2021-08-18) still resolves correctly.
DATE_OFFSETS = (2, 0, 1, 3, -1, -3, 4, -4)

# Where the Symphony h5 and its parsed json metadata live. These are the same
# directories config.ini declares as DataJoint ingest sources.
SINGLE_CELL_ROOT = Path('/Volumes/ChrisNewSSD/single_cell')


def _normalize_cell_type(value) -> str:
    """``RGC\\ON-midget`` and ``OnMidget`` both become ``onmidget``."""
    text = str(value or '').lower().replace('\\', '/').split('/')[-1]
    return text.replace('-', '').replace(' ', '').replace('_', '')


def metadata_files(root=None) -> pd.DataFrame:
    """Every dated json/h5 pair under the single-cell tree.

    One row per experiment file, with the calendar date and the rig suffix
    parsed off the name (``2021-04-27_B`` -> rig ``B``), which is what
    separates two experiments recorded on one day.
    """
    import re

    root = Path(root) if root is not None else SINGLE_CELL_ROOT
    rows = []
    if not root.exists():
        return pd.DataFrame(columns=['exp_name', 'calendar_date', 'rig', 'source',
                                     'json_path', 'h5_path'])
    for source in sorted(p for p in root.iterdir() if p.is_dir()):
        json_dir, h5_dir = source / 'json', source / 'h5'
        if not json_dir.is_dir():
            continue
        for json_path in sorted(json_dir.glob('*.json')):
            match = re.match(r'(\d{4}-\d{2}-\d{2})(?:_([A-Za-z0-9]+))?', json_path.stem)
            if not match:
                continue
            h5_path = h5_dir / f'{json_path.stem}.h5'
            rows.append({
                'exp_name': json_path.stem,
                'calendar_date': match.group(1),
                'rig': (match.group(2) or '')[:1].upper(),
                'source': source.name,
                'json_path': str(json_path),
                'h5_path': str(h5_path) if h5_path.exists() else '',
            })
    return pd.DataFrame(rows)


@lru_cache(maxsize=256)
def _json_cells(json_path: str) -> Dict[str, Tuple[str, bool]]:
    """``{cell_label: (cell_type, has_VariableMeanNoise)}`` for one metadata file."""
    import json as _json

    try:
        data = _json.loads(Path(json_path).read_text())
    except Exception:
        return {}
    out: Dict[str, Tuple[str, bool]] = {}
    for animal in data.get('animals', []):
        for prep in animal.get('preparations', []):
            for cell in prep.get('cells', []):
                protocols = {str(block.get('protocolID', '')).rsplit('.', 1)[-1]
                             for group in cell.get('epoch_groups', [])
                             for block in group.get('epoch_blocks', [])}
                cell_type = str(cell.get('properties', {}).get('type')
                                or cell.get('type') or '')
                out[cell.get('label')] = (
                    cell_type,
                    any('VariableMeanNoise' in name for name in protocols))
    return out


def resolve_roster_files(roster: pd.DataFrame, root=None,
                         offsets: Sequence[int] = DATE_OFFSETS,
                         show: bool = True) -> pd.DataFrame:
    """Locate each saved cell's raw files, correcting the two-day date offset.

    For every roster row the candidate experiments at each offset are opened and
    the cell is looked up **by label, case sensitively** -- the saved labels are
    ``cell5`` for the older entries and ``Cell2`` for the newer ones, and both
    appear in the metadata as written. A candidate is scored on whether the cell
    type also agrees and whether that cell actually has VariableMeanNoise
    blocks; the best-scoring candidate wins, and the search stops early on a
    full match.

    ``match_quality`` is one of ``label+type+protocol`` (all three agree),
    ``label+protocol``, ``label+type``, ``label`` or ``not found``. Only the
    first should be trusted without looking.
    """
    files = metadata_files(root)
    if files.empty:
        out = roster.copy()
        for column in ('exp_name', 'rig', 'json_path', 'h5_path', 'match_quality'):
            out[column] = ''
        out['day_offset'] = np.nan
        return out

    by_date: Dict[str, List[pd.Series]] = {}
    for _, row in files.iterrows():
        by_date.setdefault(row.calendar_date, []).append(row)

    import datetime as _dt
    rows = []
    for _, entry in roster.iterrows():
        try:
            day = _dt.date.fromisoformat(entry.calendar_date)
        except ValueError:
            rows.append({}); continue
        best = None
        for offset in offsets:
            target = (day + _dt.timedelta(days=int(offset))).isoformat()
            for candidate in by_date.get(target, []):
                info = _json_cells(candidate.json_path).get(entry.cell_label)
                if info is None:
                    continue
                cell_type, has_protocol = info
                type_match = (_normalize_cell_type(cell_type)
                              == _normalize_cell_type(entry.cell_type))
                score = 2 * type_match + bool(has_protocol)
                if best is None or score > best[0]:
                    best = (score, candidate, offset, cell_type, has_protocol,
                            type_match)
            if best is not None and best[0] == 3:
                break
        if best is None:
            rows.append({'exp_name': '', 'rig': '', 'json_path': '', 'h5_path': '',
                         'day_offset': np.nan, 'matched_cell_type': '',
                         'type_match': False, 'has_protocol': False,
                         'match_quality': 'not found'})
            continue
        score, candidate, offset, cell_type, has_protocol, type_match = best
        quality = ('label+type+protocol' if score == 3 else
                   'label+type' if score == 2 else
                   'label+protocol' if score == 1 else 'label')
        rows.append({
            'exp_name': candidate.exp_name, 'rig': candidate.rig,
            'source': candidate.source,
            'json_path': candidate.json_path, 'h5_path': candidate.h5_path,
            'day_offset': int(offset), 'matched_cell_type': cell_type,
            'type_match': bool(type_match), 'has_protocol': bool(has_protocol),
            'match_quality': quality})

    out = pd.concat([roster.reset_index(drop=True),
                     pd.DataFrame(rows)], axis=1)
    out['h5_present'] = out['h5_path'].fillna('').ne('')
    if show:
        found = out[out.match_quality.ne('not found')]
        print(f'{len(found)} of {len(out)} saved cells located on disk '
              f'({int(out.h5_present.sum())} with an h5)')
        print(out.match_quality.value_counts().to_string())
        if len(found):
            print('\nday offset applied:')
            print(found.day_offset.value_counts().sort_index().to_string())
    return out


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


def _fit_ln_once(stimulus, response, sampling_interval, filter_pts,
                 frequency_cutoff, correct_stim_power, n_bins):
    """One filter + nonlinearity fit. Returns the pieces, no scoring."""
    from retinanalysis.utils.cascadegraph import (compute_filter,
                                                  convolve_filter_with_stim,
                                                  sample_nl)
    cutoff_kwargs = ({} if frequency_cutoff is None else
                     dict(frequency_cutoff=frequency_cutoff,
                          sampling_interval=sampling_interval))
    filter_causal, _ = compute_filter(
        stimulus, response, filter_pts,
        correct_stim_power=correct_stim_power, **cutoff_kwargs)
    generator = convolve_filter_with_stim(filter_causal, stimulus)
    nl_x, nl_y = sample_nl(generator, response, num_bins=n_bins)
    params = fit_sigmoid(nl_x, nl_y)
    return filter_causal, generator, nl_x, nl_y, params


def _predict(filter_vec, params, stimulus):
    from retinanalysis.utils.cascadegraph import convolve_filter_with_stim
    generator = convolve_filter_with_stim(filter_vec, stimulus)
    return generator, sigmoid(generator, params['alpha'], params['beta'],
                              params['gamma'], params['epsilon'])


def _variance_explained(predicted, measured) -> float:
    """Mean row-wise variance explained.

    ``compute_variance_explained`` returns one value per epoch for a 2-D input,
    so this averages them. Calling ``float()`` on that vector raises, which is
    how an earlier version of this function silently fell back to reporting the
    nonlinearity's own fit quality instead of the model's.
    """
    from retinanalysis.utils.cascadegraph import compute_variance_explained
    measured = np.atleast_2d(measured)
    predicted = np.atleast_2d(predicted)
    # An epoch the cell was silent through is flat after smoothing, so its
    # total variance is zero and its r-squared is -inf. There is nothing to
    # explain in such an epoch, so it is dropped rather than allowed to make
    # the mean -inf; if every held-out epoch is flat the score is NaN.
    varying = np.var(measured, axis=1) > 0
    if not np.any(varying):
        return np.nan
    values = np.atleast_1d(compute_variance_explained(
        predicted[varying], measured[varying]))
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def fit_ln_model(stimulus, response, sampling_interval: float,
                 label: str = '', filter_length_s: float = 1.0,
                 frequency_cutoff: Optional[float] = None,
                 correct_stim_power: bool = True,
                 n_bins: int = 100,
                 test_fraction: float = 0.2,
                 eval_iterations: int = 3,
                 random_state: Optional[int] = 0) -> LNModel:
    """Fit an LN model to (epochs x time) matrices, scored on held-out epochs.

    Every stage is cascadegraph's, matching ``computeLNmodel.m``:
    ``compute_filter`` for the linear stage, ``convolve_filter_with_stim`` for
    the generator signal, ``sample_nl`` to bin the input-output relation, and
    ``SigmoidNlNode`` for the static nonlinearity.

    **Scoring follows ``LNModelWrapper.m``**: on each of ``eval_iterations``
    rounds a random ``test_fraction`` of epochs is held out, the filter and
    nonlinearity are fitted on the rest, and variance explained is measured on
    the held-out epochs only. ``LNModel.r2`` is the mean of those rounds. An
    in-sample number is far higher and means much less -- both are returned, as
    ``r2`` and ``r2_train``, so the gap is visible rather than implied.

    The returned filter and nonlinearity are refitted on **all** epochs once the
    scoring is done, since that is the best estimate to plot; only the score
    comes from the splits. With too few epochs to hold one out, ``r2`` is NaN
    and ``n_test`` is 0.
    """
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

    n_epochs = stimulus.shape[0]
    n_test = int(np.ceil(test_fraction * n_epochs)) if n_epochs > 1 else 0
    n_test = min(n_test, n_epochs - 1) if n_epochs > 1 else 0

    held_out = []
    rng = np.random.default_rng(random_state)
    for _ in range(eval_iterations if n_test else 0):
        test_idx = rng.choice(n_epochs, size=n_test, replace=False)
        train_idx = np.setdiff1d(np.arange(n_epochs), test_idx)
        try:
            filt, _, _, _, params = _fit_ln_once(
                stimulus[train_idx], response[train_idx], sampling_interval,
                filter_pts, frequency_cutoff, correct_stim_power, n_bins)
            if not np.isfinite(params['alpha']):
                continue
            _, predicted = _predict(filt, params, stimulus[test_idx])
            held_out.append(_variance_explained(predicted, response[test_idx]))
        except Exception:
            continue

    # Final model on every epoch -- the one worth plotting.
    filt, generator, nl_x, nl_y, params = _fit_ln_once(
        stimulus, response, sampling_interval, filter_pts,
        frequency_cutoff, correct_stim_power, n_bins)
    _, predicted_all = _predict(filt, params, stimulus)
    r2_train = (_variance_explained(predicted_all, response)
                if np.isfinite(params['alpha']) else np.nan)

    time_s = np.arange(response.shape[1]) * sampling_interval
    return LNModel(
        label=label,
        r2=float(np.nanmean(held_out)) if held_out else np.nan,
        r2_train=r2_train, nl_r2=params.get('r2', np.nan),
        n_train=int(n_epochs - n_test), n_test=int(n_test),
        filter=np.asarray(filt, dtype=float),
        filter_time_s=np.arange(filter_pts) * sampling_interval,
        nl_x=np.asarray(nl_x, dtype=float), nl_y=np.asarray(nl_y, dtype=float),
        params=params, source='python',
        example_time_s=time_s,
        example_measured=np.asarray(response[0], dtype=float),
        example_predicted=np.asarray(np.atleast_2d(predicted_all)[0], dtype=float))


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
            print(f'  lightMean {mean_level:g}: {stim.shape[0]} epochs '
                  f'({model.n_train} train / {model.n_test} test) | '
                  f'r²={model.r2:.3f} held out, {model.r2_train:.3f} in sample | '
                  f'time-to-peak {model.time_to_peak_ms:.0f} ms')
    return analysis


def plot_condition(analysis: ConditionAnalysis,
                   example_seconds: float = 6.0,
                   figsize: Tuple[float, float] = (13.0, 4.2)):
    """Filter, nonlinearity, and measured-versus-predicted, per light mean.

    The third panel is the one that says whether the model works: a filter and
    a nonlinearity can both look reasonable while predicting a trace badly.
    ``example_seconds`` limits how much of the first epoch is drawn, since a
    60 s trace at 1 kHz is unreadable at full width.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    means = analysis.light_means
    colors = style.colors_for_conditions([f'{m:g}' for m in means])
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for mean_level in means:
        model = analysis.ln_model[mean_level]
        color = colors[f'{mean_level:g}']
        axes[0].plot(model.filter_time_s * 1e3, model.filter, lw=1.8, color=color,
                     label=f'lightMean {mean_level:g} '
                           f'(n={analysis.n_epochs[mean_level]}, '
                           f'r²={model.r2:.2f})')
        axes[1].plot(model.nl_x, model.nl_y, 'o', ms=3, alpha=0.6, color=color)
        params = model.params
        if params and np.isfinite(params.get('alpha', np.nan)):
            grid = np.linspace(np.nanmin(model.nl_x), np.nanmax(model.nl_x), 200)
            axes[1].plot(grid, sigmoid(grid, params['alpha'], params['beta'],
                                       params['gamma'], params['epsilon']),
                         lw=1.6, color=color)
        if model.example_measured.size:
            keep = model.example_time_s <= example_seconds
            axes[2].plot(model.example_time_s[keep], model.example_measured[keep],
                         lw=1.0, color=color, alpha=0.55)
            axes[2].plot(model.example_time_s[keep], model.example_predicted[keep],
                         lw=1.6, color=color,
                         label=f'lightMean {mean_level:g} prediction')
    axes[0].axhline(0, color='#888888', lw=0.8, ls='--')
    axes[0].set_xlabel('filter time (ms)')
    axes[0].set_ylabel('filter (a.u.)')
    axes[0].set_title('temporal filter', fontsize=9)
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_xlabel('generator signal')
    axes[1].set_ylabel(analysis.units)
    axes[1].set_title('nonlinearity', fontsize=9)
    axes[2].set_xlabel('time (s)')
    axes[2].set_ylabel(analysis.units)
    axes[2].set_title(f'measured (thin) vs predicted (thick), '
                      f'first {example_seconds:g} s', fontsize=9)
    axes[2].legend(frameon=False, fontsize=7)
    fig.suptitle(f'{analysis.exp_name} | blocks {analysis.block_ids} | '
                 f'{analysis.rec_type}', fontsize=11)
    fig.tight_layout()
    return fig


__all__ = [
    'PROTOCOLS', 'PROTOCOL_SEARCH', 'DEFAULT_SUMMARY_PATH', 'SUMMARY_DIR',
    'STEP_DIRECTIONS', 'STEP_LABELS', 'LNModel', 'ConditionAnalysis',
    'summary_path', 'load_summary', 'load_cell',
    'DATE_OFFSETS', 'SINGLE_CELL_ROOT', 'metadata_files',
    'resolve_roster_files',
    'find_blocks', 'match_roster', 'block_cells', 'resolve_block_mode',
    'condition_table', 'block_conditions', 'epoch_parameters',
    'led_attenuation', 'matlab_randn', 'gaussian_noise_stimulus',
    'epoch_stimulus', 'fit_sigmoid', 'sigmoid', 'fit_ln_model',
    'analyze_condition', 'plot_condition',
]
