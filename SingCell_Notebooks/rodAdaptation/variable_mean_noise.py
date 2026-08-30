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

from collections import OrderedDict
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
# Protocol parameters worth carrying into the discovery table, taken from the
# protocol's own properties (turner-package/.../VariableMeanNoise.m). The
# rieke copy records the same set -- verified against the stored epochs, not
# assumed. `lightMean`, `stdv` and `seed` are written per epoch rather than
# once per block, so they are summarised separately.
PROTOCOL_PARAMETERS = (
    'stimTime',          # noise duration (ms) -- the analysis gate
    'led',               # which LED delivered it
    'ndfs',              # that LED's own filters (no filter wheel in this path)
    'frequencyCutoff',   # noise smoothing cutoff (Hz)
    'numberOfFilters',   # poles in the smoothing cascade
    'Contrast',          # noise contrast(s) configured
    'sampleRate',
    'useRandomSeed',
    'numberOfAverages',
)

# Epochs shorter than this cannot show the adaptation the analysis is after:
# the mean steps part-way through, and a 600 ms epoch has no post-step stretch
# to fit a filter on. The recordings that matter here run 30-60 s.
MIN_STIM_TIME_MS = 30_000

# The primate ganglion-cell types this analysis is about. Everything else the
# rigs recorded -- cones, horizontals, unlabelled cells -- is a different
# experiment and is dropped rather than silently averaged in.
PRIMATE_CELL_TYPES = ('ON-parasol', 'OFF-parasol', 'ON-midget', 'OFF-midget', 'AII')

# Where the per-block recording-mode cache lives. Resolving a mode can need the
# block's raw trace, which is seconds per block, so it is done once and stored.
MODE_CACHE_PATH = Path(__file__).resolve().parent / 'block_modes.csv'


def _match_cell_type(value, allowed=PRIMATE_CELL_TYPES) -> bool:
    """True when a recorded cell type is one of ``allowed``, spelling aside."""
    text = str(value or '').lower().replace('\\', '/').split('/')[-1]
    text = text.replace('-', '').replace(' ', '').replace('_', '')
    return any(text == str(a).lower().replace('-', '').replace(' ', '')
               for a in allowed)


def build_mode_cache(protocol_blocks: pd.DataFrame,
                     path: Optional[Path] = None,
                     verbose: bool = True) -> pd.DataFrame:
    """Resolve every block's recording type and write the cache.

    Reading one block's traces to decide whether it is extracellular, exc or
    inh takes about 5.5 s, so doing it for the whole protocol takes several
    minutes; the answer never changes, so it is written once to
    :data:`MODE_CACHE_PATH` and read back by :func:`load_block_modes`.

    Was a loop in the notebook. Returns the frame it wrote.
    """
    path = MODE_CACHE_PATH if path is None else Path(path)
    rows = []
    total = len(protocol_blocks)
    for n, (_, row) in enumerate(protocol_blocks.iterrows(), 1):
        block_id = int(row['block_id'])
        exp_name = str(row['exp_name'])
        rows.append(dict(exp_name=exp_name, block_id=block_id,
                         n_epochs=len(epoch_parameters(block_id)),
                         **resolve_block_mode(exp_name, block_id)))
        if verbose and n % 50 == 0:
            print(f'  {n}/{total}')
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    if verbose:
        print(f'wrote {len(frame)} blocks to {path.name}')
    return frame


def load_block_modes(path=None) -> pd.DataFrame:
    """The cached per-block recording type, or an empty frame if not built.

    Built by the notebook's mode-cache cell. Resolving a block needs its trace
    whenever the epoch-group ``recordingTechnique`` and the series resistance do
    not already settle it, which is why this is cached rather than recomputed.
    """
    path = Path(path) if path is not None else MODE_CACHE_PATH
    if not path.exists():
        return pd.DataFrame(columns=['exp_name', 'block_id', 'rec_type',
                                     'n_epochs', 'series_resistance',
                                     'mean_current_pa', 'rec_note'])
    frame = pd.read_csv(path)
    frame['block_id'] = pd.to_numeric(frame['block_id'], errors='coerce').astype('Int64')
    return frame


def _first_epoch_metadata(block_ids: Sequence[int]):
    """First-epoch parameters and epoch counts, in two batched queries.

    Copied from ``linear_equivalent_disc``: reading parameters block by block
    is far too slow over the ~1700 blocks this protocol has.
    """
    import datajoint as dj
    from retinanalysis.config import schema

    ids = [int(b) for b in block_ids]
    if not ids:
        return {}, pd.Series(dtype=int)
    epochs = schema.Epoch() & [{'parent_id': b} for b in ids]
    summary = (dj.U('parent_id')
               .aggr(epochs, first_epoch_id='min(id)', n_epochs='count(*)')
               .to_pandas().reset_index())
    if summary.empty:
        return {}, pd.Series(0, index=ids, dtype=int)
    first = (schema.Epoch() & [{'id': int(e)} for e in summary['first_epoch_id']]
             ).to_pandas().reset_index()[['parent_id', 'parameters']]
    import json as _json
    parameters = {}
    for row in first.itertuples():
        value = row.parameters
        if isinstance(value, str):
            try:
                value = _json.loads(value)
            except ValueError:
                value = {}
        parameters[int(row.parent_id)] = value if isinstance(value, dict) else {}
    counts = summary.set_index('parent_id')['n_epochs'].astype(int)
    counts.index = counts.index.astype(int)
    return parameters, counts


def find_blocks(exp_names: Optional[Sequence[str]] = None,
                protocols: Sequence[str] = PROTOCOLS,
                min_stim_time_ms: Optional[float] = MIN_STIM_TIME_MS,
                cell_types: Optional[Sequence[str]] = PRIMATE_CELL_TYPES,
                show: bool = True, height: int = 420) -> pd.DataFrame:
    """Every VariableMeanNoise block, with its cell and its protocol settings.

    One row per epoch block. Cell identity comes from
    EpochBlock -> EpochGroup -> Cell, and the protocol settings from the
    block's first epoch, both in batched queries rather than per block.

    ``min_stim_time_ms`` keeps only blocks whose noise ran at least that long
    (30 s by default). The protocol's own default is 600 ms, and a large share
    of the recorded blocks are short runs that cannot show adaptation to a mean
    step; what is dropped is always reported.

    The two package copies of the protocol are searched together and
    ``protocol_name`` says which one each block came from.
    """
    from retinanalysis.config import schema
    from retinanalysis.SCutils import explore as sc

    protocol_df = schema.Protocol().to_pandas().reset_index()[['protocol_id', 'name']]
    protocol_df = protocol_df[protocol_df['name'].isin(list(protocols))]
    if protocol_df.empty:
        if show:
            print(f'no protocol matched {list(protocols)}')
        return pd.DataFrame()

    blocks = (schema.EpochBlock() & [f'protocol_id={int(i)}'
                                     for i in protocol_df.protocol_id]
              ).to_pandas().reset_index()
    if blocks.empty:
        return pd.DataFrame()
    blocks = blocks.rename(columns={'id': 'block_id', 'parent_id': 'group_id'})
    blocks = blocks.merge(protocol_df.rename(columns={'name': 'protocol_name'}),
                          on='protocol_id', how='left')

    experiments = (schema.Experiment() & [f'id={int(i)}'
                                          for i in blocks.experiment_id.unique()]
                   ).to_pandas().reset_index()[['id', 'exp_name']]
    groups = (schema.EpochGroup() & [f'id={int(i)}'
                                     for i in blocks.group_id.unique()]
              ).to_pandas().reset_index()[['id', 'parent_id']]
    cells = (schema.Cell() & [f'id={int(i)}' for i in groups.parent_id.unique()]
             ).to_pandas().reset_index()[['id', 'label', 'type']]
    frame = (blocks
             .merge(experiments.rename(columns={'id': 'experiment_id'}),
                    on='experiment_id', how='left')
             .merge(groups.rename(columns={'id': 'group_id',
                                           'parent_id': 'cell_id'}),
                    on='group_id', how='left')
             .merge(cells.rename(columns={'id': 'cell_id', 'label': 'cell_label',
                                          'type': 'cell_type'}),
                    on='cell_id', how='left'))
    frame['cell_type_short'] = (frame['cell_type'].astype(str)
                                .str.split('\\').str[-1].replace('nan', 'Unknown'))
    if exp_names is not None:
        frame = frame[frame.exp_name.isin(list(exp_names))]
    if frame.empty:
        return frame

    parameters, counts = _first_epoch_metadata(frame.block_id)
    for name in PROTOCOL_PARAMETERS:
        frame[name] = [parameters.get(int(b), {}).get(name) for b in frame.block_id]
    frame['n_epochs'] = [int(counts.get(int(b), 0)) for b in frame.block_id]

    # lightMean and stdv are written per epoch, so the first epoch only shows
    # one of them. Read every epoch's to see what the block actually stepped
    # between; contrast is stdv / lightMean and should be constant by design.
    means, contrasts = [], []
    for block_id in frame.block_id:
        params = epoch_parameters(int(block_id))
        light = pd.to_numeric(params.get('lightMean', pd.Series(dtype=float)),
                              errors='coerce').dropna()
        stdv = pd.to_numeric(params.get('stdv', pd.Series(dtype=float)),
                             errors='coerce').dropna()
        means.append(sorted(light.unique()))
        ratio = (stdv / light).round(4) if len(light) == len(stdv) else pd.Series(dtype=float)
        contrasts.append(sorted(ratio.dropna().unique()))
    frame['light_means'] = means
    frame['light_contrasts'] = contrasts
    frame['stimTime'] = pd.to_numeric(frame['stimTime'], errors='coerce')
    frame['stim_seconds'] = frame['stimTime'] / 1e3
    frame['led_color'] = (frame['led'].astype(str).str.split().str[0]
                          .str.lower().replace('none', ''))

    dropped = dropped_type = 0
    if min_stim_time_ms is not None:
        keep = frame.stimTime.ge(float(min_stim_time_ms))
        dropped = int((~keep).sum())
        frame = frame[keep]
    if cell_types is not None:
        keep = frame.cell_type_short.map(lambda v: _match_cell_type(v, cell_types))
        dropped_type = int((~keep).sum())
        frame = frame[keep]
    frame = frame.sort_values(['exp_name', 'cell_label', 'block_id']).reset_index(drop=True)

    if show:
        print(f'{len(frame)} blocks | {frame.exp_name.nunique()} experiments | '
              f'{frame.groupby(["exp_name", "cell_label"]).ngroups} cells')
        if dropped:
            print(f'  dropped {dropped} block(s) with stimTime < '
                  f'{min_stim_time_ms / 1e3:g} s')
        if dropped_type:
            print(f'  dropped {dropped_type} block(s) on non-primate-RGC cell types')
        print('  by protocol: ' + ', '.join(
            f'{n.rsplit(".", 1)[-1]} ({c})' if False else f'{n} {c}'
            for n, c in frame.protocol_name.value_counts().items()))
        columns = [c for c in ('exp_name', 'cell_label', 'cell_type_short',
                               'block_id', 'stim_seconds', 'led', 'ndfs',
                               'frequencyCutoff', 'numberOfFilters', 'Contrast',
                               'n_epochs', 'protocol_name') if c in frame]
        sc.scroll_table(frame[columns], height=height,
                        num_cols=('block_id', 'stim_seconds', 'frequencyCutoff',
                                  'numberOfFilters', 'n_epochs'))
    return frame


def find_protocol_cells(blocks: Optional[pd.DataFrame] = None,
                        modes: Optional[pd.DataFrame] = None,
                        require_step: bool = True,
                        single_contrast: bool = True,
                        show: bool = True, height: int = 460) -> pd.DataFrame:
    """One row per cell -- the section 1 table to pick a ``cell_index`` from.

    Per-epoch settings are summarised across the cell's blocks, and the cell is
    kept only if it is worth analysing here:

    ``require_step``
        drops cells that only ever ran **one** light mean. This protocol is
        about the step between means, and a cell that saw one mean has no step;
    ``single_contrast``
        drops cells that ran more than one noise contrast, since a filter
        pooled across contrasts describes neither.

    ``modes`` is the cached per-block recording type from
    :func:`load_block_modes`. When it is available the epoch counts are broken
    out into ``n_extracellular`` / ``n_exc`` / ``n_inh``, which is what says
    whether a cell is worth opening and in which mode.
    """
    from retinanalysis.SCutils import explore as sc

    blocks = find_blocks(show=False) if blocks is None else blocks
    if blocks.empty:
        return blocks
    modes = load_block_modes() if modes is None else modes

    frame = blocks.copy()
    if not modes.empty:
        frame = frame.merge(
            modes[['block_id', 'rec_type']].astype({'block_id': 'int64'}),
            on='block_id', how='left')
    else:
        frame['rec_type'] = ''
    frame['rec_type'] = frame['rec_type'].fillna('')

    rows = []
    for (exp_name, cell_label, cell_type), group in frame.groupby(
            ['exp_name', 'cell_label', 'cell_type_short'], dropna=False, sort=False):
        means = sorted({m for values in group.light_means for m in values})
        contrasts = sorted({c for values in group.light_contrasts for c in values})
        counts = group.groupby('rec_type').n_epochs.sum()
        rows.append({
            'exp_name': exp_name, 'cell_label': cell_label,
            'cell_type': cell_type,
            'n_extracellular': int(counts.get('extracellular', 0)),
            'n_exc': int(counts.get('exc', 0)),
            'n_inh': int(counts.get('inh', 0)),
            'n_unresolved': int(counts.get('', 0)),
            'light_means': ', '.join(f'{m:g}' for m in means),
            'n_light_means': len(means),
            'light_contrast': ', '.join(f'{c:g}' for c in contrasts),
            'n_contrasts': len(contrasts),
            'stim_seconds': ', '.join(
                f'{v:g}' for v in sorted(group.stim_seconds.dropna().unique())),
            'led': ', '.join(sorted({str(v) for v in group.led.dropna()})),
            '_block_ids': [int(b) for b in sorted(group.block_id)],
        })
    cells = pd.DataFrame(rows)
    if cells.empty:
        return cells

    dropped_step = dropped_contrast = 0
    if require_step:
        keep = cells.n_light_means.gt(1)
        dropped_step = int((~keep).sum())
        cells = cells[keep]
    if single_contrast:
        keep = cells.n_contrasts.le(1)
        dropped_contrast = int((~keep).sum())
        cells = cells[keep]

    cells = cells.sort_values(['exp_name', 'cell_label']).reset_index(drop=True)
    cells.insert(0, 'cell_index', np.arange(len(cells)))
    if show:
        print(f'{len(cells)} cells across {cells.exp_name.nunique()} experiments')
        if dropped_step:
            print(f'  dropped {dropped_step} cell(s) that ran only one light mean')
        if dropped_contrast:
            print(f'  dropped {dropped_contrast} cell(s) that ran more than one contrast')
        if modes.empty:
            print('  recording types not resolved yet -- build the mode cache '
                  'to fill n_extracellular / n_exc / n_inh')
        columns = ['exp_name', 'cell_index', 'cell_label', 'cell_type',
                   'n_extracellular', 'n_exc', 'n_inh', 'light_means',
                   'light_contrast', 'stim_seconds', 'led']
        if int(cells.n_unresolved.sum()):
            columns.insert(7, 'n_unresolved')
        sc.tree_table(cells[columns], levels=['exp_name'], height=height,
                      num_cols=('cell_index', 'n_extracellular', 'n_exc',
                                'n_inh', 'n_unresolved'))
    return cells


def cell_blocks(cells: pd.DataFrame, cell_index: int,
                rec_type: Optional[str] = None,
                modes: Optional[pd.DataFrame] = None) -> Tuple[str, List[int], str]:
    """``(exp_name, block_ids, rec_type)`` for one row of the section 1 table.

    Blocks of different recording types are never returned together: a spike
    rate and a synaptic current are different quantities, and one cell often
    has both. ``rec_type`` picks the group; the default is whichever has the
    most epochs.
    """
    row = cells[cells.cell_index.eq(int(cell_index))]
    if row.empty:
        raise ValueError(f'cell_index {cell_index} is not in this table')
    row = row.iloc[0]
    modes = load_block_modes() if modes is None else modes
    block_ids = list(row['_block_ids'])
    if modes.empty:
        return row.exp_name, block_ids, (rec_type or 'extracellular')

    subset = modes[modes.block_id.isin(block_ids)]
    if rec_type is None:
        counts = subset.groupby('rec_type').n_epochs.sum().sort_values(ascending=False)
        counts = counts[counts.index.astype(str).ne('')]
        rec_type = counts.index[0] if len(counts) else 'extracellular'
    chosen = [int(b) for b in subset.loc[subset.rec_type.eq(rec_type), 'block_id']]
    return row.exp_name, (chosen or block_ids), rec_type


def _spike_rate(spike_samples, n_samples: int, sample_rate: float,
                downsample: int, sigma_ms: float) -> np.ndarray:
    """A smoothed PSTH at the *reduced* rate, in Hz.

    Bin first, then smooth. The old order -- lay the spikes down at the
    amplifier's 10 kHz, Gaussian-smooth there, then block-average to 1 kHz --
    convolves a 302k-sample array with an 801-tap kernel per epoch, which cost
    3.4 s for 19 epochs and was most of the time spent loading a condition.
    Binning to the analysis rate first makes it a 30k-sample array and an
    81-tap kernel, ~100x less arithmetic.

    The result is the same to within rounding: block-averaging is a boxcar,
    convolution commutes, and the Gaussian band-limits the train well below the
    reduced Nyquist either way, so decimating before or after the smoothing
    gives the same trace. Binning first is also the textbook PSTH -- a
    histogram of spike counts, then smoothed.
    """
    from scipy.ndimage import gaussian_filter1d

    step = max(int(downsample), 1)
    n_bins = n_samples // step
    if n_bins <= 0:
        return np.zeros(0, dtype=float)
    times = np.asarray(spike_samples, dtype=np.int64)
    times = times[(times >= 0) & (times < n_bins * step)]
    counts = np.bincount(times // step, minlength=n_bins)[:n_bins].astype(float)
    reduced_rate = sample_rate / step
    rate_hz = counts * reduced_rate
    sigma_bins = float(sigma_ms) / 1e3 * reduced_rate
    if sigma_bins > 0:
        rate_hz = gaussian_filter1d(rate_hz, sigma_bins)
    return rate_hz


def plot_traces(exp_name: str, block_ids: Sequence[int], rec_type: str,
                max_epochs: Optional[int] = 12, downsample: int = 50,
                subtract_baseline: bool = False,
                figsize: Tuple[float, float] = (12.0, 5.0)):
    """Every epoch's response, coloured by the light mean it was recorded at.

    Drawn before any fitting: this is where a dead epoch, a lost patch or a
    mislabelled recording type shows up, and none of those are visible in a
    filter. Whole-cell traces are drawn as recorded -- see
    :func:`analyze_condition` for why there is no baseline to subtract -- so
    the holding current is part of what is shown, and a patch that drifts over
    the block is visible here.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style
    from scipy.ndimage import gaussian_filter1d

    style.apply_publication_style()
    spiking = rec_type == 'extracellular'
    traces, labels = [], []
    for block_id in block_ids:
        params = epoch_parameters(int(block_id))
        if params.empty:
            continue
        amp, rate, spike_times = load_block(exp_name, int(block_id), spiking)
        for index in range(min(len(params), amp.shape[0])):
            if max_epochs is not None and len(traces) >= max_epochs:
                break
            factor = max(int(downsample), 1)
            if spiking:
                # Already at the reduced rate: binned, then smoothed.
                reduced = _spike_rate(spike_times[index], amp.shape[1], rate,
                                      factor, 10.0)
            else:
                trace = amp[index]
                if subtract_baseline:
                    trace = trace - float(np.mean(trace[:int(0.1 * rate)]))
                # Block-average, not slicing: a whole-cell trace is unsmoothed
                # at the amplifier rate, so taking every nth sample folds fast
                # events into the drawn line. Same reduction the analysis uses.
                reduced = _block_average(trace, factor)
            traces.append((reduced, rate / factor))
            labels.append(float(params.iloc[index].get('lightMean', np.nan)))

    if not traces:
        print('no epochs to draw')
        return None
    means = sorted({m for m in labels if np.isfinite(m)})
    colors = style.colors_for_conditions([f'{m:g}' for m in means])
    fig, ax = plt.subplots(figsize=figsize)
    offset_step = np.nanpercentile([np.ptp(t) for t, _ in traces], 90) or 1.0
    for index, ((trace, rate), mean_level) in enumerate(zip(traces, labels)):
        time_s = np.arange(trace.size) / rate
        color = colors.get(f'{mean_level:g}', '#888888')
        ax.plot(time_s, trace + index * offset_step, lw=0.7, color=color)
    for mean_level in means:
        ax.plot([], [], lw=2, color=colors[f'{mean_level:g}'],
                label=f'lightMean {mean_level:g}')
    ax.set_xlabel('time in epoch (s)')
    ax.set_ylabel('firing rate (Hz)' if spiking else 'current (pA)')
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(f'{exp_name} | blocks {list(block_ids)} | {rec_type} | '
                 f'{len(traces)} epochs, offset vertically', fontsize=10)
    fig.tight_layout()
    return fig


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


# --------------------------------------------------------------------------
# 2b. finding the raw files: the saved dates are two days early
# --------------------------------------------------------------------------
# The MATLAB wrote `expDate` from `datestr(epochList.elements(1).startDate)`,
# and it is **two days early, across the board**. Whatever that call resolved
# to, it is not the date the experiment was filed under: every saved date that
# has a file at all has it two days later, and the cells confirm it -- matching
# at +2 and then checking the cell label and type resolves the roster with 44 of
# 53 entries agreeing on label, type and protocol at once.
#
# So +2 is applied as a correction rather than searched for. The fallback
# offsets exist only for the two entries where +2 does not land on a file, and
# any row that needed one is called out in `date_note` rather than quietly
# resolved.
SAVED_DATE_OFFSET_DAYS = 2
FALLBACK_OFFSETS = (0, 1, 3, -1, -3, 4, -4)
DATE_OFFSETS = (SAVED_DATE_OFFSET_DAYS,) + FALLBACK_OFFSETS

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


def corrected_dates(roster: pd.DataFrame,
                    offset_days: int = SAVED_DATE_OFFSET_DAYS) -> pd.Series:
    """The saved date shifted by the known offset, as ``yyyy-mm-dd``."""
    import datetime as _dt

    def shift(value):
        try:
            return (_dt.date.fromisoformat(str(value))
                    + _dt.timedelta(days=int(offset_days))).isoformat()
        except ValueError:
            return ''

    return roster['calendar_date'].map(shift)


def resolve_roster_files(roster: pd.DataFrame, root=None,
                         offset_days: int = SAVED_DATE_OFFSET_DAYS,
                         fallback_offsets: Sequence[int] = FALLBACK_OFFSETS,
                         show: bool = True) -> pd.DataFrame:
    """Locate each saved cell's raw files, correcting the two-day date offset.

    The saved ``expDate`` is two days early, so ``corrected_date`` is the date
    actually searched. Within that date the cell is looked up **by label, case
    sensitively** -- the saved labels are ``cell5`` for the older entries and
    ``Cell2`` for the newer ones, and both appear in the metadata as written --
    and the match is then confirmed against the cell type and against whether
    that cell really has VariableMeanNoise blocks. The rig suffix on the
    filename separates two experiments recorded on one day.

    Columns worth reading before using a row:

    ``corrected_date``
        the saved date plus the offset -- the date searched;
    ``day_offset`` / ``date_note``
        what was actually used. ``date_note`` is empty for the corrected date
        and says so loudly when a row only resolved at some other offset,
        which happens for the two entries whose corrected date has no file;
    ``match_quality``
        ``label+type+protocol`` when all three agree. Anything less is listed
        rather than trusted.
    """
    files = metadata_files(root)
    out = roster.reset_index(drop=True).copy()
    out['corrected_date'] = corrected_dates(out, offset_days)
    if files.empty:
        for column in ('exp_name', 'rig', 'json_path', 'h5_path', 'match_quality',
                       'date_note'):
            out[column] = ''
        out['day_offset'] = np.nan
        out['h5_present'] = False
        return out

    by_date: Dict[str, List[pd.Series]] = {}
    for _, row in files.iterrows():
        by_date.setdefault(row.calendar_date, []).append(row)

    import datetime as _dt
    rows = []
    for _, entry in out.iterrows():
        try:
            day = _dt.date.fromisoformat(entry.calendar_date)
        except ValueError:
            rows.append({'exp_name': '', 'rig': '', 'json_path': '', 'h5_path': '',
                         'day_offset': np.nan, 'matched_cell_type': '',
                         'type_match': False, 'has_protocol': False,
                         'match_quality': 'not found',
                         'date_note': 'unreadable saved date'})
            continue
        best = None
        for offset in (int(offset_days), *[int(o) for o in fallback_offsets]):
            target = (day + _dt.timedelta(days=offset)).isoformat()
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
            # Stop as soon as the corrected date produces any match at all: a
            # worse match there still beats a better one on the wrong day.
            if best is not None and offset == int(offset_days):
                break
            if best is not None and best[0] == 3:
                break
        if best is None:
            rows.append({'exp_name': '', 'rig': '', 'json_path': '', 'h5_path': '',
                         'day_offset': np.nan, 'matched_cell_type': '',
                         'type_match': False, 'has_protocol': False,
                         'match_quality': 'not found',
                         'date_note': f'no file holds {entry.cell_label} near '
                                      f'{entry.corrected_date}'})
            continue
        score, candidate, offset, cell_type, has_protocol, type_match = best
        quality = ('label+type+protocol' if score == 3 else
                   'label+type' if score == 2 else
                   'label+protocol' if score == 1 else 'label')
        note = ('' if offset == int(offset_days) else
                f'corrected date {entry.corrected_date} has no matching file; '
                f'resolved at {offset:+d} days instead -- verify')
        rows.append({
            'exp_name': candidate.exp_name, 'rig': candidate.rig,
            'source': candidate.source,
            'json_path': candidate.json_path, 'h5_path': candidate.h5_path,
            'day_offset': int(offset), 'matched_cell_type': cell_type,
            'type_match': bool(type_match), 'has_protocol': bool(has_protocol),
            'match_quality': quality, 'date_note': note})

    out = pd.concat([out, pd.DataFrame(rows)], axis=1)
    out['h5_present'] = out['h5_path'].fillna('').ne('')
    if show:
        found = out[out.match_quality.ne('not found')]
        corrected = int((out.day_offset == int(offset_days)).sum())
        print(f'saved dates corrected by {offset_days:+d} days '
              f'({corrected} of {len(out)} entries resolve there)')
        print(f'{len(found)} of {len(out)} saved cells located on disk '
              f'({int(out.h5_present.sum())} with an h5)')
        print(out.match_quality.value_counts().to_string())
        deviant = out[out.date_note.ne('')]
        if len(deviant):
            print(f'\n{len(deviant)} entr(ies) did not resolve on the corrected date:')
            for _, row in deviant.iterrows():
                print(f'  {row.exp_date} {row.cell_label}: {row.date_note}')
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
    all_tokens = tuple(parse_ndfs(row.get('ndfs')))
    # The wheel is not in the LED's path, so an FW token recorded in the LED's
    # own ndfs list must not be added to this stimulus's attenuation. It is
    # reported rather than dropped silently: the same list is right for a Stage
    # protocol and wrong here.
    wheel_tokens = tuple(t for t in all_tokens if str(t).upper().startswith('FW'))
    tokens = tuple(t for t in all_tokens if t not in wheel_tokens)
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
        'exp_name': exp_name, 'rig': rig, 'led': row.get('led', ''),
        'led_color': color,
        'led_ndfs': ', '.join(tokens) if tokens else '(none)',
        'optical_density': np.nan if missing else total,
        'attenuation': np.nan if missing else 10.0 ** -total,
        'unknown_tokens': ', '.join(missing),
        'wheel_tokens_ignored': ', '.join(wheel_tokens),
        'filter_wheel_ndf': wheel,
        'wheel_ignored': bool(wheel_tokens) or bool(pd.notna(wheel)),
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


# --------------------------------------------------------------------------
# Caches
#
# Re-running one analysis cell is the normal way to work in the notebook, and
# without these every re-run pays the same two costs again: the MATLAB engine
# redraws every epoch's noise, and the amplifier traces come back off disk.
# Both are pure functions of their keys -- the stimulus of the recorded
# generator parameters, the traces of the block -- so both are memoised.
#
# They are bounded because the arrays are large: one 30 s epoch at 10 kHz is
# 2.4 MB and one block of ten is 24 MB, so the caps below are roughly 150 MB
# and 100 MB. `clear_caches()` empties them.
# --------------------------------------------------------------------------
STIMULUS_CACHE_MAX = 64
BLOCK_CACHE_MAX = 4
_STIMULUS_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_BLOCK_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()


def _cache_get(cache: OrderedDict, key):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


def _cache_put(cache: OrderedDict, key, value, max_entries: int):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_entries:
        cache.popitem(last=False)
    return value


def clear_caches() -> None:
    """Empty the stimulus and trace caches.

    Only needed if the underlying H5 has been rewritten under a running
    kernel, or to hand back the memory.
    """
    _STIMULUS_CACHE.clear()
    _BLOCK_CACHE.clear()


def load_block(exp_name: str, block_id: int, spiking: bool):
    """``(amp, sample_rate, spike_times)`` for one block, memoised.

    The arrays are handed out as-is rather than copied -- they are large, and
    every caller here reads them. Do not write into what this returns.
    """
    import retinanalysis as ra

    key = (str(exp_name), int(block_id), bool(spiking))
    hit = _cache_get(_BLOCK_CACHE, key)
    if hit is not None:
        return hit
    block = ra.SCResponseBlock(exp_name, int(block_id), b_spiking=spiking,
                               b_LED=True, verbose=False)
    amp = np.asarray(block.amp_data, dtype=float)
    rate = float(block.amp_sample_rate)
    spike_times = None
    if spiking:
        # SCResponseBlock.get_spike_times() populates block.spike_times and
        # returns None, so read the attribute rather than the return value.
        if getattr(block, 'spike_times', None) is None:
            block.get_spike_times()
        spike_times = block.spike_times
    return _cache_put(_BLOCK_CACHE, key, (amp, rate, spike_times),
                      BLOCK_CACHE_MAX)


def matlab_randn(seed: int, n: int) -> np.ndarray:
    """``RandStream('mt19937ar', 'Seed', seed).randn(1, n)``, exactly.

    NumPy's Mersenne Twister matches MATLAB's ``rand`` but not its ``randn``
    (verified in this project), because the two use different transforms from
    uniform to normal. The stimulus has to be the one that was presented, so
    this defers to MATLAB rather than approximating it.
    """
    engine = _matlab_engine()
    # One `eval` rather than two. Assigning the stream and then drawing from it
    # costs a second round trip through the engine for no benefit -- the
    # functional form draws from exactly the same stream and returns
    # byte-identical values (checked against the pinned reference in
    # tests/test_variable_mean_noise.py), at 40 ms per 302k-sample epoch
    # against 75 ms.
    values = engine.eval(
        f"randn(RandStream('mt19937ar','Seed',{int(seed)}),1,{int(n)})",
        nargout=1)
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
    kwargs = dict(
        seed=int(float(params['seed'])), stim_pts=stim_pts,
        st_dev=float(params['stdv']), freq_cutoff=float(params['frequencyCutoff']),
        num_filters=int(float(params['numberOfFilters'])),
        mean=float(params['lightMean']), sample_rate=rate)
    if noise is not None:
        # A supplied draw is the test path; it is not part of the key, so it
        # must not be served from or written to the cache.
        return gaussian_noise_stimulus(noise=noise, **kwargs)
    key = tuple(sorted(kwargs.items()))
    hit = _cache_get(_STIMULUS_CACHE, key)
    if hit is not None:
        return hit.copy()
    return _cache_put(_STIMULUS_CACHE, key, gaussian_noise_stimulus(**kwargs),
                      STIMULUS_CACHE_MAX).copy()


# --------------------------------------------------------------------------
# 5. LN model fitting -- vendored cascadegraph
# --------------------------------------------------------------------------
# Physiological ceilings for the sigmoid nonlinearity.
#
# `alpha` is a response amplitude, so its ceiling depends on what the response
# is measured in; `beta` and `gamma` live on the generator axis, which is in
# contrast units whatever the amplifier was doing, so one number covers both
# recording modes. Every value below is a ceiling no real cell reaches, not a
# typical value -- the point is to cut off the degenerate ridge where `alpha`
# and `epsilon` grow together and cancel, without touching any shape the data
# can actually show. Measured over 57 fits from 24 cells: `alpha` reached
# 3742 pA whole cell and 492 Hz extracellular, `beta` 32, `gamma` -14.3, and
# the generator spanned -3.0 to +2.6.
SIGMOID_AMPLITUDE_MAX = {
    'exc': 5_000.0,            # pA -- a few nA of synaptic current
    'inh': 5_000.0,            # pA
    'extracellular': 1_000.0,  # Hz -- above any primate RGC's maintained rate
}
# 1 / generator unit. The transition of `alpha*Phi(beta*x + gamma)` takes about
# `4/|beta|` generator units, so 100 is a step spanning 0.04 of an axis that
# runs to about +/-3: sharper than the binned nonlinearity can resolve.
SIGMOID_SLOPE_MAX = 100.0
# The midpoint `-gamma/beta` may sit this many full generator ranges outside
# the sampled data -- enough for a nonlinearity seen only in its tail.
SIGMOID_X50_HEADROOM = 1.0


def sigmoid_start_and_bounds(x, y, rec_type: Optional[str] = None,
                             alpha_max_factor: float = 10.0):
    """Starting parameters and bounds for ``alpha * Phi(beta*x + gamma) + eps``.

    Every start value is read off the sampled nonlinearity, because each
    parameter of this form *is* a summary statistic of it:

    ``epsilon``
        the lower asymptote -- what the curve approaches as ``beta*x + gamma``
        runs to -inf, so the baseline: ``min(y)`` for a rising nonlinearity,
        ``max(y)`` for a falling one.
    ``alpha``
        the total rise, since the curve spans ``epsilon`` to
        ``epsilon + alpha``. So ``+/- (max(y) - min(y))``, signed by the
        direction of the relationship.
    ``beta``
        the steepness. The steepest slope of the form is
        ``alpha * beta / sqrt(2*pi)``, so matching that to the average slope
        ``(max(y) - min(y)) / (max(x) - min(x))`` gives
        ``beta = sqrt(2*pi) / x_range`` -- a transition taking up about the
        whole sampled range, which is what an unsaturated nonlinearity looks
        like.
    ``gamma``
        places the midpoint. The curve is at half height where
        ``beta*x + gamma == 0``, so ``gamma = -beta * x50`` with ``x50``
        interpolated at the half-height crossing.

    **The bounds are physiological**, and the two axes are bounded differently
    because they mean different things.

    ``alpha`` is a response amplitude, so its ceiling is in the units of the
    recording: :data:`SIGMOID_AMPLITUDE_MAX` gives 5 nA for ``exc``/``inh`` and
    1000 Hz for ``extracellular``, passed in through ``rec_type``. It is also
    held within ``alpha_max_factor`` times the observed range, since a curve
    cannot rise by much more than the data it was sampled from; the tighter of
    the two applies. With ``rec_type=None`` only the data-relative cap does.

    ``beta`` and ``gamma`` live on the *generator* axis, which is in contrast
    units whatever the amplifier was doing, so one pair of numbers serves every
    recording mode: :data:`SIGMOID_SLOPE_MAX` for the steepness, and a ``gamma``
    ceiling that lets the midpoint sit :data:`SIGMOID_X50_HEADROOM` full ranges
    outside the sampled data.

    ``epsilon`` is *not* given an absolute ceiling, and this is deliberate: it
    is an absolute level, not an amplitude, so under voltage clamp it carries
    the holding current -- one cell here sits at -14.4 nA while modulating by
    946 pA. Capping it at "a few thousand pA" would refuse that cell's baseline
    outright. It is bounded relative to the data instead, by one ``alpha``
    ceiling either side: the curve spans ``epsilon`` to ``epsilon + alpha`` and
    must pass through the sampled points, so that is exactly how far the
    asymptote can be from them. Either side, because a falling nonlinearity's
    ``epsilon`` is its *upper* asymptote.

    The start point matters as much as the box. The old guess fixed ``gamma``
    at 0 and left every bound infinite, and the optimiser drifted along a ridge
    where ``alpha`` and ``epsilon`` grow together and nearly cancel, using a
    sliver of ``Phi`` as a near-linear segment: on 2026-02-27_G lightMean 0.3 it
    stopped at ``alpha`` 1.35e7 against ``epsilon`` -1.35e7, a difference of 282
    out of 1.35e7, so seven digits cancelled and neither parameter carried any
    information. Letting ``alpha`` run from 3.7e3 to 1.25e11 bought 6e-5 of
    r squared and cost 14x the time.

    That ridge is a real optimum, so cutting it off costs a little fit on
    near-linear nonlinearities -- about 0.01 of r squared -- in exchange for
    parameters that mean what they are named. :func:`fit_sigmoid` reports which
    parameters ended on a bound, so a fit that was actually constrained is
    visible rather than silent.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_range = float(np.max(x) - np.min(x))
    y_range = float(np.max(y) - np.min(y))
    if not np.isfinite(x_range) or x_range <= 0:
        x_range = float(np.std(x)) or 1.0
    if not np.isfinite(y_range) or y_range <= 0:
        y_range = float(np.std(y)) or 1.0

    # Direction of the relationship, from the sign of the least-squares slope.
    x_centered, y_centered = x - x.mean(), y - y.mean()
    denominator = float(np.sum(x_centered ** 2))
    slope = float(np.sum(x_centered * y_centered) / denominator) if denominator else 1.0
    direction = 1.0 if slope >= 0 else -1.0

    alpha0 = direction * y_range
    epsilon0 = float(np.min(y)) if direction > 0 else float(np.max(y))
    beta0 = direction * np.sqrt(2.0 * np.pi) / x_range

    # Half-height crossing. Interpolating on y sorted ascending (carrying x
    # along) is monotone by construction, so this works whichever way the
    # nonlinearity runs and does not care that y is noisy.
    order = np.argsort(y)
    x50 = float(np.interp(float(np.min(y) + 0.5 * y_range), y[order], x[order]))
    gamma0 = -beta0 * x50

    alpha_max = float(alpha_max_factor) * y_range
    physiological = SIGMOID_AMPLITUDE_MAX.get(str(rec_type))
    if physiological is not None:
        alpha_max = min(alpha_max, float(physiological))
    beta_max = SIGMOID_SLOPE_MAX
    x50_max = float(np.max(np.abs(x))) + SIGMOID_X50_HEADROOM * x_range
    gamma_max = beta_max * x50_max

    # The curve runs from `epsilon` to `epsilon + alpha` and has to pass
    # through the data, so the asymptote can sit at most one `alpha` outside
    # it -- on either side, since a falling curve's `epsilon` is its upper
    # asymptote. Deriving it from `alpha_max` rather than picking a multiple of
    # the data range matters: an unsaturated nonlinearity's asymptote is
    # legitimately far outside the sampled points, and a fixed one-range box
    # pinned 11 of 18 windowed fits on this cell at exactly its limit.
    lower = np.array([-alpha_max, -beta_max, -gamma_max,
                      float(np.min(y)) - alpha_max])
    upper = np.array([alpha_max, beta_max, gamma_max,
                      float(np.max(y)) + alpha_max])
    guess = np.clip(np.array([alpha0, beta0, gamma0, epsilon0]), lower, upper)
    return guess, lower, upper


def fit_sigmoid(nl_x, nl_y, optim_iters: int = 5,
                rec_type: Optional[str] = None,
                alpha_max_factor: float = 10.0) -> Dict[str, float]:
    """Fit ``alpha * Phi(beta * x + gamma) + epsilon`` with cascadegraph.

    ``SigmoidNlNode`` is the Python port of the node class the MATLAB fitted,
    so parameter names and model are identical. Returns NaNs rather than
    raising when the fit fails, so one bad cell does not stop a loop.

    The start point and bounds come from :func:`sigmoid_start_and_bounds`,
    which estimates each parameter from the sampled points and bounds it
    physiologically -- pass ``rec_type`` so the amplitude ceiling is the one
    for the units the response is in.

    The returned dict carries ``at_bounds``: the names of any parameters that
    ended on their limit, empty when none did. A constrained fit is a fit whose
    parameters are being reported by the bound rather than by the data, and
    that should be visible rather than silent.
    """
    from retinanalysis.utils.cascadegraph import SigmoidNlNode

    x = np.asarray(nl_x, dtype=float)
    y = np.asarray(nl_y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    failed = {k: np.nan for k in ('alpha', 'beta', 'gamma', 'epsilon', 'r2')}
    failed['at_bounds'] = ()
    if x.size < 5:
        return failed
    node = SigmoidNlNode()
    guess, lower, upper = sigmoid_start_and_bounds(
        x, y, rec_type=rec_type, alpha_max_factor=alpha_max_factor)
    try:
        params = node.fit_to_sample(x, y, params0=guess, lower_bounds=lower,
                                    upper_bounds=upper, optim_iters=optim_iters)
    except Exception:
        return failed
    span = upper - lower
    at_bounds = tuple(
        name for name, value, low, high, width
        in zip(('alpha', 'beta', 'gamma', 'epsilon'),
               np.asarray(params).ravel(), lower, upper, span)
        if np.isfinite(width) and width > 0
        and (abs(value - low) <= 1e-6 * width or abs(value - high) <= 1e-6 * width))
    predicted = node.process(x)
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2 = (1.0 - float(np.sum((y - predicted) ** 2)) / denominator
          if denominator else np.nan)
    return dict(zip(('alpha', 'beta', 'gamma', 'epsilon'),
                    (float(p) for p in np.asarray(params).ravel())),
                r2=r2, at_bounds=at_bounds)


def sigmoid(x, alpha: float, beta: float, gamma: float, epsilon: float):
    """Evaluate the fitted form, for drawing a curve through the points."""
    from retinanalysis.utils.cascadegraph import SigmoidNlNode
    return SigmoidNlNode().process_temp_params(
        np.array([alpha, beta, gamma, epsilon]), np.asarray(x, dtype=float))


def _crop_for_equal_n(*arrays, n_bins: int):
    """Trim columns so the flattened point count divides by ``n_bins``.

    ``sample_nl``'s equal-N binning refuses a count that does not divide
    evenly, and epoch lengths never happen to. Dropping at most ``n_bins``
    samples off the end of each epoch is the smallest change that satisfies it.
    """
    rows, cols = np.atleast_2d(arrays[0]).shape
    while cols > 1 and (rows * cols) % int(n_bins):
        cols -= 1
    return [np.atleast_2d(a)[:, :cols] for a in arrays]


def _fit_ln_once(stimulus, response, sampling_interval, filter_pts,
                 frequency_cutoff, correct_stim_power, n_bins,
                 bin_type='equalN', rec_type=None):
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
    # Normalise the filter by the ratio of contrasts, not by its peak.
    #
    # The stimulus is raw LED intensity, so its contrast is sigma/mean -- the
    # protocol's own `Contrast`. The generator signal is a convolution of the
    # filter with that stimulus, so measuring its contrast against the same
    # mean gives sigma_generator/mean, and
    #
    #     scale = contrast_generator / contrast_stimulus
    #
    # is exactly the factor by which the filter inflates it. Dividing the
    # filter by that scale leaves the generator with the same sigma/mean as the
    # stimulus, which is the invariant that makes two light levels comparable:
    # the filter carries the kinetics, the gain it divided out is returned
    # separately, and the nonlinearity's input is a contrast at every mean.
    filter_causal = np.asarray(filter_causal, dtype=float)
    generator = convolve_filter_with_stim(filter_causal, stimulus)
    stim_mean = float(np.nanmean(stimulus))
    contrast_stimulus = (float(np.nanstd(stimulus)) / stim_mean
                         if stim_mean else np.nan)
    contrast_generator = (float(np.nanstd(generator)) / stim_mean
                          if stim_mean else np.nan)
    scale = (contrast_generator / contrast_stimulus
             if np.isfinite(contrast_generator) and np.isfinite(contrast_stimulus)
             and contrast_stimulus else 1.0)
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    # Both steps are folded into the filter itself -- dividing by the contrast
    # ratio, then by the mean to state the generator as a contrast -- so that
    # convolving this filter with the raw stimulus reproduces exactly the
    # generator the nonlinearity was fitted against. Applying either step
    # outside the filter would leave prediction and fit on different scales.
    filter_causal = filter_causal / scale
    if stim_mean:
        filter_causal = filter_causal / stim_mean
    generator = convolve_filter_with_stim(filter_causal, stimulus)
    # equalN puts the same number of points in every bin, as the MATLAB's
    # SETTINGS.binningType does. Equal-width bins put almost nothing in the
    # tails, where the nonlinearity actually saturates, so the fit is pulled
    # around by a handful of samples out there.
    gen_binned, resp_binned = (_crop_for_equal_n(generator, response,
                                                 n_bins=n_bins)
                               if bin_type == 'equalN' else (generator, response))
    nl_x, nl_y = sample_nl(gen_binned, resp_binned, num_bins=n_bins,
                           bin_type=bin_type)
    params = fit_sigmoid(nl_x, nl_y, rec_type=rec_type)
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
                 n_bins: int = 100, bin_type: str = 'equalN',
                 test_fraction: float = 0.2,
                 eval_iterations: int = 3,
                 rec_type: Optional[str] = None,
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
                filter_pts, frequency_cutoff, correct_stim_power, n_bins,
                bin_type, rec_type)
            if not np.isfinite(params['alpha']):
                continue
            _, predicted = _predict(filt, params, stimulus[test_idx])
            held_out.append(_variance_explained(predicted, response[test_idx]))
        except Exception:
            continue

    # Final model on every epoch -- the one worth plotting.
    filt, generator, nl_x, nl_y, params = _fit_ln_once(
        stimulus, response, sampling_interval, filter_pts,
        frequency_cutoff, correct_stim_power, n_bins, bin_type, rec_type)
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
    # The downsampled (epochs x time) arrays the models were fitted to, kept so
    # the group mean and the windowed models are built from exactly the same
    # data rather than reloaded and re-reduced.
    sampling_interval: float = np.nan
    skip_seconds: float = 0.0
    # The stimulus's own ``frequencyCutoff``, as resolved from the epochs. Kept
    # so anything fitted downstream (the windowed models in particular) cuts
    # off where the stimulus does instead of guessing a number.
    frequency_cutoff: float = np.nan
    # The fit settings the loader was called with, so `fit_condition` and the
    # windowed models reproduce them without being told again.
    filter_length_s: float = 1.0
    n_bins: int = 100
    stimulus: Dict[float, np.ndarray] = field(default_factory=dict)
    response: Dict[float, np.ndarray] = field(default_factory=dict)
    # What the whole-cell salvage did: epochs refused on series resistance,
    # and the per-epoch offsets removed to bring their holding currents onto a
    # common level. Both are kept so a fit can be traced back to the traces.
    dropped_epochs: pd.DataFrame = field(default_factory=pd.DataFrame)
    epoch_adjustments: pd.DataFrame = field(default_factory=pd.DataFrame)

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
                      skip_seconds: float = 2.0,
                      subtract_baseline: bool = False,
                      max_series_resistance: Optional[float] = 30e6,
                      align_epoch_means: bool = True,
                      max_epochs: Optional[int] = None,
                      fit: bool = True,
                      verbose: bool = True) -> ConditionAnalysis:
    """Load epochs, regenerate their stimuli, and fit one LN model per light mean.

    ``fit=False`` stops after the traces are loaded, aligned and downsampled,
    leaving ``ln_model`` empty. That is the point at which the group mean is
    worth looking at -- it is the data the models will be fitted to, and seeing
    it first is what tells you whether a fit is worth trusting.
    :func:`fit_condition` finishes the job on the same object.

    Epochs are grouped by their recorded ``lightMean``, since a filter fitted
    across two mean levels would describe neither. ``downsample`` reduces the
    10 kHz amplifier rate before fitting -- the noise is cut off at 60 Hz, so
    1 kHz is already generous and the full rate makes the estimate slow without
    making it better.

    For extracellular recordings the response is a smoothed spike rate; for
    voltage clamp it is the recorded current, **not** baseline subtracted.

    **Whole-cell drift -- ``exc`` and ``inh`` only.** Neither guard below runs
    for an ``extracellular`` recording, whatever its arguments say: that
    response is a spike rate built from spike times, with no holding current to
    wander and no series resistance to refuse on. They correct a voltage-clamp
    artefact and nothing else, and which way they resolved is printed at the
    top of every run.

    The holding current wanders over a recording, and over epochs this long it
    wanders a lot: on 2021-04-27_B cell1 the epoch means span 1741 pA while the
    modulation within an epoch is about 300 pA, so the offset is roughly five
    times the signal.

    ``max_series_resistance``
        drops voltage-clamp epochs whose recorded series resistance is above it
        (30 MOhm by default); what was dropped is printed and kept in
        ``ConditionAnalysis.dropped_epochs``.
    ``align_epoch_means``
        shifts each epoch so its mean matches the **median of the epoch means
        within its own light mean**. Per light mean, not globally -- the two
        means genuinely differ in holding current and that difference is
        signal. The median rather than the mean, so one badly drifted epoch
        does not drag the level it is being compared against. The shifts are
        printed and kept in ``ConditionAnalysis.epoch_adjustments``.

    A constant offset per epoch is not something the model can explain: it adds
    between-epoch variance to the response while the stimulus says nothing
    about it. Removing it leaves the within-epoch modulation, which is the part
    the filter is fitted to, untouched.

    ``subtract_baseline`` defaults to False because this protocol has no
    baseline to subtract: ``preTime`` and ``tailTime`` are zero -- neither is
    even written to the epoch -- so the noise runs from the first sample and
    the leading 100 ms is stimulus-driven like every other stretch. Removing
    its mean would subtract a response, not a resting level, and would also
    discard the holding current, which is the part of a voltage-clamp trace
    that says how much excitation or inhibition the cell is receiving. Nothing
    downstream needs it removed: ``compute_filter`` zeroes the DC bin,
    ``convolve_filter_with_stim`` mean-subtracts the stimulus, and the
    sigmoid's ``epsilon`` absorbs a constant offset in the response.

    The stimulus stays in **raw intensity** units. Its contrast is
    ``sigma / mean``, and :func:`fit_ln_model` normalises the filter by the
    ratio of that to the generator signal's own contrast, so the generator ends
    up with the same ``sigma / mean`` as the stimulus at every light level.

    ``skip_seconds`` drops the start of each epoch, where the cell is still
    responding to the step onto the epoch's first mean rather than to the
    noise.

    **Downsampling.** Everything is built at the amplifier rate and reduced once,
    at the end, by ``downsample`` (10, so 10 kHz to 1 kHz):

    ==================  ==================================================
    stimulus            regenerated at the amplifier rate from the seed
    spiking response    spike times -> binary train -> Gaussian smoothed at
                        ``psth_sigma_ms`` (10 ms), all at the amplifier rate
    whole-cell response the recorded current as-is
    ==================  ==================================================

    The reduction is a **block average** (:func:`_block_average`), never
    ``trace[::n]``. Averaging is both the anti-alias filter and the decimation:
    a 10-sample boxcar at 10 kHz has its first null at 1 kHz, far above the
    stimulus's 60 Hz cutoff, so nothing the stimulus could have driven is lost,
    while slicing would fold the amplifier's own high-frequency content back
    into the band being fitted. It is also what ``parseData.m`` does.

    Stimulus and response are truncated to a common length *before* the
    average, so they stay sample-aligned, and ``sampling_interval`` handed to
    :func:`fit_ln_model` is ``downsample / sample_rate`` -- the reduced rate,
    not the amplifier's. On top of this :func:`fit_ln_model` low-passes the
    response at the stimulus's own ``frequencyCutoff`` before fitting.

    ``frequency_cutoff`` defaults to the stimulus's own ``frequencyCutoff``
    (60 Hz here) and is **load-bearing**, not cosmetic. The noise is 4-pole
    filtered at that frequency, so its power above it is ~1e-9 of the power
    below; dividing by that spectrum -- which is what ``correct_stim_power``
    does -- amplifies noise without bound and returns a filter that is pure
    noise. Cutting the filter off at the same frequency the stimulus was cut
    off at is what ``computeLNmodel.m`` does through ``SETTINGS``.
    """
    from scipy.ndimage import gaussian_filter1d

    spiking = rec_type == 'extracellular'
    # The drift guards below correct a *holding current*, which only a
    # voltage-clamp recording has. An extracellular response is a spike rate
    # built from spike times: nothing to drift, no series resistance to refuse
    # on, and shifting one would move a firing rate for no reason. Both guards
    # are gated on this rather than on their own arguments, so passing them for
    # an extracellular cell is simply inert.
    whole_cell = not spiking
    stimuli: Dict[float, List[np.ndarray]] = {}
    responses: Dict[float, List[np.ndarray]] = {}
    sources: Dict[float, List[dict]] = {}
    dropped: List[dict] = []
    sample_rate = np.nan
    cutoff = frequency_cutoff
    used = 0

    if verbose:
        if whole_cell:
            gate = ('off' if max_series_resistance is None
                    else f'{max_series_resistance / 1e6:g} MOhm')
            print(f'{rec_type}: whole-cell drift guards apply -- '
                  f'series-resistance limit {gate}, align epoch means '
                  f'{"on" if align_epoch_means else "off"}')
        else:
            print(f'{rec_type}: spike rate, so the whole-cell drift guards do '
                  f'not apply -- nothing to drift and no series resistance')

    for block_id in block_ids:
        params = epoch_parameters(int(block_id))
        if params.empty:
            continue
        amp, sample_rate, spike_times = load_block(exp_name, int(block_id),
                                                   spiking)
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
                # Built at the reduced rate directly -- see `_spike_rate`. It
                # is upsampled back by repetition only so the trimming and
                # alignment below stay in amplifier samples like the stimulus;
                # `_block_average` then returns exactly these values.
                reduced = _spike_rate(spike_times[index], amp.shape[1],
                                      sample_rate, downsample, psth_sigma_ms)
                trace = np.repeat(reduced, max(int(downsample), 1))
                if trace.size < amp.shape[1]:
                    trace = np.concatenate(
                        [trace, np.full(amp.shape[1] - trace.size,
                                        trace[-1] if trace.size else 0.0)])
            else:
                trace = amp[index]
                if subtract_baseline:
                    baseline = int(min(0.1 * sample_rate, amp.shape[1]))
                    trace = trace - float(np.mean(trace[:baseline]))
            mean_level = float(row['lightMean'])
            # The stimulus stays in raw intensity units: its contrast is
            # sigma/mean, and fit_ln_model normalises the filter by the ratio
            # of that to the generator signal's own contrast.
            start = int(round(max(skip_seconds, 0.0) * sample_rate))
            if start >= width:
                continue

            series_resistance = np.nan
            if 'seriesResistance' in row:
                series_resistance = _numeric(row['seriesResistance'])
            if (whole_cell and max_series_resistance is not None
                    and np.isfinite(series_resistance)
                    and series_resistance > float(max_series_resistance)):
                dropped.append({'block_id': int(block_id), 'epoch': index,
                                'light_mean': mean_level,
                                'series_resistance_mohm': series_resistance / 1e6,
                                'reason': f'series resistance above '
                                          f'{max_series_resistance / 1e6:g} MOhm'})
                continue

            stimuli.setdefault(mean_level, []).append(stimulus[:width][start:])
            responses.setdefault(mean_level, []).append(trace[:width][start:])
            sources.setdefault(mean_level, []).append(
                {'block_id': int(block_id), 'epoch': index,
                 'light_mean': mean_level,
                 'series_resistance_mohm': (series_resistance / 1e6
                                            if np.isfinite(series_resistance)
                                            else np.nan)})
            used += 1

    step = max(int(downsample), 1)
    interval = step / sample_rate
    adjustments: List[dict] = []
    analysis = ConditionAnalysis(
        exp_name=exp_name, block_ids=[int(b) for b in block_ids],
        rec_type=rec_type, sample_rate=sample_rate,
        sampling_interval=interval, skip_seconds=float(skip_seconds),
        units='firing rate (Hz)' if spiking else 'current (pA)')
    for mean_level in sorted(stimuli):
        width = min(min(s.size for s in stimuli[mean_level]),
                    min(r.size for r in responses[mean_level]))
        stim = np.vstack([_block_average(s[:width], step)
                          for s in stimuli[mean_level]])
        resp = np.vstack([_block_average(r[:width], step)
                          for r in responses[mean_level]])

        # Bring the epochs of this light mean onto a common holding level. The
        # target is the median of the epoch means, so a single badly drifted
        # epoch cannot drag the level the others are judged against, and it is
        # taken within the light mean because the two means genuinely differ
        # in holding current.
        if whole_cell and align_epoch_means and resp.shape[0] > 1:
            epoch_means = resp.mean(axis=1)
            target = float(np.median(epoch_means))
            offsets = target - epoch_means
            resp = resp + offsets[:, None]
            for record, before, offset in zip(sources.get(mean_level, []),
                                              epoch_means, offsets):
                adjustments.append({**record,
                                    'mean_before_pa': float(before),
                                    'offset_pa': float(offset),
                                    'mean_after_pa': float(target)})

        analysis.light_means.append(mean_level)
        analysis.n_epochs[mean_level] = int(stim.shape[0])
        analysis.stimulus[mean_level] = stim
        analysis.response[mean_level] = resp

    analysis.frequency_cutoff = float(cutoff) if cutoff is not None else np.nan
    analysis.filter_length_s = float(filter_length_s)
    analysis.n_bins = int(n_bins)
    analysis.dropped_epochs = pd.DataFrame(dropped)
    analysis.epoch_adjustments = pd.DataFrame(adjustments)
    if verbose:
        if dropped:
            print(f'\n  dropped {len(dropped)} epoch(s) on series resistance '
                  f'> {max_series_resistance / 1e6:g} MOhm:')
            for record in dropped:
                print(f'    block {record["block_id"]} epoch {record["epoch"]} '
                      f'(lightMean {record["light_mean"]:g}): '
                      f'{record["series_resistance_mohm"]:.1f} MOhm')
        if adjustments:
            frame = analysis.epoch_adjustments
            print(f'\n  aligned {len(frame)} epoch(s) to the median holding '
                  f'current of their light mean:')
            for mean_level, group in frame.groupby('light_mean'):
                print(f'    lightMean {mean_level:g}: target '
                      f'{group.mean_after_pa.iloc[0]:+.0f} pA | offsets '
                      f'{group.offset_pa.min():+.0f} to '
                      f'{group.offset_pa.max():+.0f} pA')
    if fit:
        fit_condition(analysis, verbose=verbose)
    return analysis


def fit_condition(analysis: ConditionAnalysis,
                  filter_length_s: Optional[float] = None,
                  n_bins: Optional[int] = None,
                  verbose: bool = True) -> ConditionAnalysis:
    """Fit one LN model per light mean on an already-loaded condition.

    Split out of :func:`analyze_condition` so the group-mean response can be
    drawn from the same arrays before anything is fitted to them. Defaults come
    off the analysis, so this reproduces what ``analyze_condition`` would have
    done. Fills ``analysis.ln_model`` and returns the same object.
    """
    filter_length_s = (analysis.filter_length_s if filter_length_s is None
                       else float(filter_length_s))
    n_bins = analysis.n_bins if n_bins is None else int(n_bins)
    cutoff = (analysis.frequency_cutoff
              if np.isfinite(analysis.frequency_cutoff) else None)
    for mean_level in analysis.light_means:
        stim = analysis.stimulus[mean_level]
        resp = analysis.response[mean_level]
        model = fit_ln_model(
            stim, resp, sampling_interval=analysis.sampling_interval,
            label=f'lightMean {mean_level:g}',
            filter_length_s=filter_length_s, n_bins=n_bins,
            frequency_cutoff=cutoff, rec_type=analysis.rec_type)
        analysis.ln_model[mean_level] = model
        if verbose:
            bounded = model.params.get('at_bounds') or ()
            print(f'  lightMean {mean_level:g}: {stim.shape[0]} epochs '
                  f'({model.n_train} train / {model.n_test} test) | '
                  f'r²={model.r2:.3f} held out, {model.r2_train:.3f} in sample | '
                  f'time-to-peak {model.time_to_peak_ms:.0f} ms'
                  + (f' | at bound: {", ".join(bounded)}' if bounded else ''))
    return analysis


def mean_response(analysis: ConditionAnalysis,
                  window_s: float = 1.0) -> pd.DataFrame:
    """Group-mean response per light mean, averaged in fixed windows.

    Epochs differ from one another -- different noise seeds, so different
    responses -- but they share the light mean and the time since the mean
    stepped, which is the axis adaptation runs along. Averaging across the
    epochs of one light mean and then within ``window_s`` windows leaves the
    slow course of the rate, with the noise-driven modulation averaged out.

    ``time_s`` is measured from the start of the fitted stretch, so it already
    excludes the ``skip_seconds`` dropped at the head of each epoch.
    """
    rows = []
    for mean_level in analysis.light_means:
        block = analysis.response.get(mean_level)
        if block is None or not block.size:
            continue
        step = max(int(round(window_s / analysis.sampling_interval)), 1)
        width = (block.shape[1] // step) * step
        if width == 0:
            continue
        windows = block[:, :width].reshape(block.shape[0], -1, step).mean(axis=2)
        centres = (np.arange(windows.shape[1]) + 0.5) * step * analysis.sampling_interval
        for index, centre in enumerate(centres):
            values = windows[:, index]
            rows.append({
                'light_mean': mean_level,
                'time_s': float(centre + analysis.skip_seconds),
                'mean': float(np.nanmean(values)),
                'sem': (float(np.nanstd(values, ddof=1) / np.sqrt(values.size))
                        if values.size > 1 else np.nan),
                'n_epochs': int(values.size)})
    return pd.DataFrame(rows)


def plot_mean_response(analysis: ConditionAnalysis, window_s: float = 1.0,
                       figsize: Tuple[float, float] = (9.0, 4.0)):
    """The group mean response over the epoch, one line per light mean."""
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    table = mean_response(analysis, window_s=window_s)
    if table.empty:
        print('no responses to average')
        return None
    colors = style.colors_for_conditions(
        [f'{m:g}' for m in analysis.light_means])
    fig, ax = plt.subplots(figsize=figsize)
    for mean_level in analysis.light_means:
        block = table[table.light_mean.eq(mean_level)].sort_values('time_s')
        if block.empty:
            continue
        color = colors[f'{mean_level:g}']
        ax.fill_between(block.time_s, block['mean'] - block['sem'].fillna(0),
                        block['mean'] + block['sem'].fillna(0),
                        color=color, alpha=0.18, lw=0)
        ax.plot(block.time_s, block['mean'], 'o-', ms=3, lw=1.6, color=color,
                label=f'lightMean {mean_level:g} '
                      f'(n={int(block.n_epochs.max())} epochs)')
    ax.set_xlabel('time in epoch (s)')
    ax.set_ylabel(analysis.units)
    ax.set_title(f'{analysis.exp_name} | mean response in {window_s:g} s windows',
                 fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


# Shortest window that still supports an LN fit, in seconds.
#
# Measured, not assumed: over 18 condition/cell combinations the held-out r2 of
# a windowed fit, as a fraction of the fit on the whole usable stretch, runs
#
#     0.5 s  1.0   1.5   2.0   3.0   4.0   5.0   7.0   10.0  14.0
#     0.64   0.78  0.82  0.85  0.90  0.92  0.94  0.96  0.97  0.98
#
# and the spread across windows of one condition falls from 0.135 to 0.034 over
# the same range. There is no knee -- it is a smooth trade -- so 3 s is the
# point where a window keeps 90% of what the full stretch achieves. Below 2 s
# the scatter between windows is as large as the adaptation being measured, and
# at 0.5 s individual fits diverge outright.
#
# The degrees-of-freedom argument agrees. A filter band-limited to the
# stimulus's cutoff and lasting L seconds has about `2*f_c*L` free parameters,
# and a window of T seconds over n epochs supplies about `2*f_c*T*n`
# independent samples, so the ratio is just `n*T/L`. At 3 s with 7 epochs and a
# 1 s filter that is ~21 samples per parameter, which is the usual floor for a
# regression to reach most of its achievable variance explained.
MIN_LN_WINDOW_S = 3.0


def temporal_windows(usable_seconds: float,
                     min_window_s: float = MIN_LN_WINDOW_S) -> Tuple[int, float]:
    """``(n_windows, window_seconds)`` covering ``usable_seconds`` with no remainder.

    Fitting fixed-width windows and dropping the remainder loses real data --
    a 5 s window on the 29 s left of a 30 s epoch after ``skip_seconds`` gives
    five windows and throws the last 4 s away, which is 14% of the recording
    and the part furthest into the adaptation. Instead take as many windows as
    :data:`MIN_LN_WINDOW_S` allows and divide the stretch evenly between them,
    so every sample is used and the windows stay equal.

    The count depends only on ``usable_seconds``, so two cells recorded with
    the same epoch length and the same ``skip_seconds`` get the same number of
    windows and their windows line up -- which is what makes them poolable.
    """
    usable = float(usable_seconds)
    if not np.isfinite(usable) or usable <= 0:
        return 0, float('nan')
    # A hair of tolerance so an epoch one sample short of an exact multiple
    # does not silently drop a whole window relative to its neighbours.
    n_windows = int(np.floor(usable / float(min_window_s) + 1e-6))
    n_windows = max(n_windows, 1)
    return n_windows, usable / n_windows


def temporal_ln_model(analysis: ConditionAnalysis,
                      window_seconds: Optional[float] = None,
                      min_window_s: float = MIN_LN_WINDOW_S,
                      filter_length_s: float = 1.0,
                      frequency_cutoff: Optional[float] = None,
                      n_bins: int = 100,
                      verbose: bool = True) -> Dict[float, List[LNModel]]:
    """One LN model per successive window of the epoch, per light mean.

    The MATLAB's ``temporalLNModel``: cut the fitted stretch into successive
    windows and fit the model separately in each, so the filter and the
    nonlinearity can be watched changing as the cell adapts. Epochs are pooled
    within a window -- all the first five seconds together, then all the second
    five -- because one epoch's window is far too little data for a filter.

    **The windows are sized to fit the epoch, not fixed.** With
    ``window_seconds=None`` (the default) :func:`temporal_windows` takes as many
    windows of at least ``min_window_s`` as the usable stretch allows and
    divides it evenly between them, so nothing is dropped: a 30 s epoch with
    ``skip_seconds=1`` gives nine windows of 3.22 s rather than five of 5 s
    plus 4 s discarded. Passing ``window_seconds`` explicitly asks for that
    width, but the stretch is still divided evenly into the nearest whole
    number of windows rather than leaving a remainder.

    ``frequency_cutoff`` defaults to ``analysis.frequency_cutoff`` -- the
    stimulus's own ``frequencyCutoff``, which is what ``SETTINGS`` carries into
    ``computeFilter`` in ``LNNodeModelWrapper.m``. It is not a constant across
    this dataset: 60 Hz on 196 of the 373 blocks, but 30, 10, 5, 20 or 1 Hz on
    177 of them. Passing a number here overrides it, which is worth doing only
    to fit below the stimulus's own cutoff, never above it -- above it
    ``correct_stim_power`` divides by a spectrum the 4-pole filter has already
    driven to ~1e-9 of its in-band value, and the filter comes back as noise.
    """
    if frequency_cutoff is None:
        frequency_cutoff = analysis.frequency_cutoff
        if not np.isfinite(frequency_cutoff):
            raise ValueError(
                'no frequency cutoff: analysis.frequency_cutoff is unset, so '
                'pass frequency_cutoff= explicitly (the stimulus parameter is '
                'called frequencyCutoff)')
    # One window count for the whole condition, taken from the shortest epoch
    # stretch, so the light means stay comparable window for window.
    widths = [analysis.stimulus[m].shape[1] for m in analysis.light_means
              if analysis.stimulus.get(m) is not None
              and analysis.stimulus[m].size]
    if not widths:
        return {}
    usable_s = min(widths) * analysis.sampling_interval
    if window_seconds is None:
        n_windows, window_seconds = temporal_windows(usable_s, min_window_s)
    else:
        n_windows = max(int(round(usable_s / float(window_seconds))), 1)
        window_seconds = usable_s / n_windows
    if verbose:
        print(f'  cutoff {frequency_cutoff:g} Hz (the stimulus\'s own) | '
              f'{n_windows} windows of {window_seconds:.2f} s '
              f'covering {usable_s:.2f} s')

    out: Dict[float, List[LNModel]] = {}
    for mean_level in analysis.light_means:
        stim = analysis.stimulus.get(mean_level)
        resp = analysis.response.get(mean_level)
        if stim is None or resp is None or not stim.size:
            continue
        # Split by sample index rather than by rounding each edge, so the
        # windows tile the stretch exactly and none overlap.
        edges = np.linspace(0, stim.shape[1], n_windows + 1).round().astype(int)
        models = []
        # stim[:, lo:hi] keeps every row -- all the epochs of this light mean --
        # and slices only time, so each window is fitted once over every epoch's
        # samples for that stretch. One model per window, not a model per epoch
        # averaged; temporalLNModel.m indexes the same way, `(:,frameRange)`.
        for index in range(n_windows):
            lo, hi = int(edges[index]), int(edges[index + 1])
            if hi - lo < 2:
                continue
            start_s = lo * analysis.sampling_interval + analysis.skip_seconds
            model = fit_ln_model(
                stim[:, lo:hi], resp[:, lo:hi],
                sampling_interval=analysis.sampling_interval,
                label=f'{start_s:.1f}-{hi * analysis.sampling_interval + analysis.skip_seconds:.1f} s',
                filter_length_s=min(filter_length_s, window_seconds / 2),
                frequency_cutoff=frequency_cutoff, n_bins=n_bins,
                rec_type=analysis.rec_type)
            models.append(model)
            if verbose:
                print(f'  lightMean {mean_level:g}  {model.label:>12}: '
                      f'{stim.shape[0]} epochs pooled '
                      f'({model.n_train} train / {model.n_test} test) | '
                      f'r²={model.r2:.3f} | time-to-peak '
                      f'{model.time_to_peak_ms:.0f} ms')
        out[mean_level] = models
    return out


def condition_summary(analysis: ConditionAnalysis,
                      show: bool = False) -> pd.DataFrame:
    """One row per light mean: fit quality and the filter's shape.

    Was a loop in the notebook; it is here because every protocol notebook
    wants the same table and none of it is specific to one cell.
    """
    rows = []
    for mean_level in analysis.light_means:
        model = analysis.ln_model.get(mean_level)
        if model is None:
            continue
        rows.append({
            'lightMean': mean_level,
            'n_epochs': analysis.n_epochs.get(mean_level, 0),
            'n_train': model.n_train, 'n_test': model.n_test,
            'r2': model.r2, 'r2_train': model.r2_train, 'nl_r2': model.nl_r2,
            'time_to_peak_ms': model.time_to_peak_ms,
            'biphasic_index': model.biphasic_index,
            'at_bounds': ','.join(model.params.get('at_bounds') or ()),
        })
    frame = pd.DataFrame(rows)
    if show and len(frame) > 1:
        dim, bright = frame.iloc[0], frame.iloc[-1]
        ratio = (bright.lightMean / dim.lightMean if dim.lightMean else np.nan)
        print(f'lightMean {dim.lightMean:g} -> {bright.lightMean:g} '
              f'({ratio:.0f}x brighter):')
        print(f'  time-to-peak   {dim.time_to_peak_ms:.0f} -> '
              f'{bright.time_to_peak_ms:.0f} ms')
        print(f'  biphasic index {dim.biphasic_index:+.2f} -> '
              f'{bright.biphasic_index:+.2f}')
    return frame


def temporal_summary(models: Dict[float, List[LNModel]]) -> pd.DataFrame:
    """One row per (light mean, window): filter shape and sigmoid parameters.

    The long form behind :func:`plot_temporal_kinetics` -- the same numbers the
    figure draws, for when they are wanted as numbers.
    """
    rows = []
    for mean_level, window_models in models.items():
        for order, model in enumerate(window_models):
            params = model.params or {}
            rows.append({
                'lightMean': mean_level, 'window': model.label, 'order': order,
                'start_s': _window_start(model.label),
                'centre_s': _window_centre(model.label),
                'r2': model.r2, 'nl_r2': model.nl_r2,
                'time_to_peak_ms': model.time_to_peak_ms,
                'biphasic_index': model.biphasic_index,
                'alpha': params.get('alpha', np.nan),
                'beta': params.get('beta', np.nan),
                'gamma': params.get('gamma', np.nan),
                'epsilon': params.get('epsilon', np.nan),
                'at_bounds': ','.join(params.get('at_bounds') or ()),
            })
    return pd.DataFrame(rows)


def _window_bounds(label: str) -> Tuple[float, float]:
    """``(start, end)`` in seconds from a ``'1.0-4.2 s'`` window label."""
    try:
        lo, hi = label.replace(' s', '').split('-')
        return float(lo), float(hi)
    except Exception:
        return np.nan, np.nan


def _window_start(label: str) -> float:
    return _window_bounds(label)[0]


def _window_centre(label: str) -> float:
    lo, hi = _window_bounds(label)
    return (lo + hi) / 2.0


def plot_temporal_kinetics(analysis: ConditionAnalysis,
                           models: Dict[float, List[LNModel]],
                           figsize: Tuple[float, float] = (13.5, 6.5)):
    """How the fitted model's parameters move across the epoch.

    Against time-since-step: the filter's **time-to-peak** and **biphasic
    index** on the top row with the fit quality, and the nonlinearity's **four
    parameters** on the bottom. The windowed
    filters and nonlinearities themselves are curves, and a pile of curves
    hides a trend; these are the numbers that describe them, so a filter
    speeding up or a nonlinearity steepening is a line going somewhere rather
    than something to be read off a legend.

    Held-out r2 is drawn alongside so a parameter that moves only where the fit
    is poor can be recognised as such. Windows whose fit ended on a bound are
    ringed, since those parameters are set by the limit and not by the data.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    frame = temporal_summary(models)
    if frame.empty:
        print('no windowed models to plot')
        return None
    means = [m for m in analysis.light_means if models.get(m)]
    colors = style.colors_for_conditions([f'{m:g}' for m in means])

    unit = analysis.units.split()[-1].strip('()')
    # Row 1 is the linear stage and the fit; row 2 is the nonlinearity's four
    # parameters, kept together so they read as one object changing.
    panels = [('time_to_peak_ms', 'filter time-to-peak (ms)'),
              ('biphasic_index', 'filter biphasic index'),
              ('r2', 'held-out r$^2$'),
              (None, None),
              ('alpha', f'nl alpha -- rise ({unit})'),
              ('beta', 'nl beta -- slope (1/generator)'),
              ('gamma', 'nl gamma -- threshold'),
              ('epsilon', f'nl epsilon -- baseline ({unit})')]
    fig, axes = plt.subplots(2, 4, figsize=figsize, sharex=True)
    for ax, (column, ylabel) in zip(axes.ravel(), panels):
        if column is None:
            ax.axis('off')
            continue
        for mean_level in means:
            block = frame[frame.lightMean.eq(mean_level)].sort_values('centre_s')
            if block.empty:
                continue
            color = colors[f'{mean_level:g}']
            ax.plot(block.centre_s, block[column], 'o-', ms=4, lw=1.6,
                    color=color, label=f'lightMean {mean_level:g}')
            constrained = block[block.at_bounds.ne('')]
            if len(constrained):
                ax.plot(constrained.centre_s, constrained[column], 'o', ms=10,
                        mfc='none', mec=color, mew=1.4)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('time since luminance step (s)', fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=7)
    n_windows = int(frame.groupby('lightMean').size().max())
    fig.suptitle(f'{analysis.exp_name} | {analysis.rec_type} | '
                 f'{n_windows} windows | rings mark a fit on a bound',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# --------------------------------------------------------------------------
# 7. Decoding the stimulus back out of the response
# --------------------------------------------------------------------------
def decoding_filter(response, stimulus, noise_ratio: float = 0.1):
    """The optimal linear filter mapping response back onto stimulus.

    Classic stimulus reconstruction: the decoder is a single linear operator
    ``s_hat = D * r``, with ``D`` the Wiener solution

        D(f) = <S(f) conj(R(f))> / (<|R(f)|^2> + lambda)

    averaged over training epochs, ``lambda`` a fraction of the mean response
    power. Both signals are mean-subtracted per epoch first, so ``D`` describes
    fluctuations about whatever level the cell is sitting at.

    Reconstructing through the *encoding* model instead -- inverting the fitted
    sigmoid and deconvolving its filter -- was tried and abandoned: the
    nonlinearity has no inverse outside its own range, and on a spike rate with
    many empty bins that put 27-44% of samples on the clip, which is a property
    of the clip rather than of the cell. The linear decoder needs no inverse
    and is what "linear decoder" conventionally means here.
    """
    response = np.atleast_2d(np.asarray(response, dtype=float))
    stimulus = np.atleast_2d(np.asarray(stimulus, dtype=float))
    response = response - response.mean(axis=1, keepdims=True)
    stimulus = stimulus - stimulus.mean(axis=1, keepdims=True)
    n_time = response.shape[1]
    fft_r = np.fft.rfft(response, axis=1)
    fft_s = np.fft.rfft(stimulus, axis=1)
    cross = np.mean(fft_s * np.conj(fft_r), axis=0)
    power = np.mean(np.abs(fft_r) ** 2, axis=0)
    lam = float(noise_ratio) * float(np.mean(power)) if np.mean(power) else 1.0
    return cross / (power + lam)


def apply_decoding_filter(decoder, response):
    """Reconstruct the stimulus from the response with a decoding filter."""
    response = np.atleast_2d(np.asarray(response, dtype=float))
    response = response - response.mean(axis=1, keepdims=True)
    n_time = response.shape[1]
    fft_r = np.fft.rfft(response, axis=1)
    if fft_r.shape[1] != np.asarray(decoder).size:
        # Different window length: resample the decoder onto this grid.
        grid_old = np.linspace(0.0, 1.0, np.asarray(decoder).size)
        grid_new = np.linspace(0.0, 1.0, fft_r.shape[1])
        decoder = (np.interp(grid_new, grid_old, np.real(decoder))
                   + 1j * np.interp(grid_new, grid_old, np.imag(decoder)))
    return np.fft.irfft(fft_r * decoder, n_time, axis=1)


def _bin_mean(array, bin_samples: int):
    """Average into non-overlapping bins of ``bin_samples`` columns."""
    array = np.atleast_2d(np.asarray(array, dtype=float))
    n_bins = array.shape[1] // int(bin_samples)
    if n_bins <= 0:
        return array[:, :0]
    trimmed = array[:, :n_bins * int(bin_samples)]
    return trimmed.reshape(array.shape[0], n_bins, int(bin_samples)).mean(axis=2)


def _decode_metrics(estimate, truth_sign) -> dict:
    """Increment/decrement performance of one decoded stretch.

    ``estimate`` is the decoded stimulus, ``truth_sign`` the true polarity per
    sample. Recall is reported per class rather than pooled: a decoder that
    calls everything an increment scores 1.0 on increments and 0.0 on
    decrements, and only the split shows it. ``auc`` is threshold-free, so it
    separates "the response carries the polarity" from "the threshold is set
    where the response happens to sit" -- which is exactly the difference
    between the two decoding modes below.
    """
    estimate = np.asarray(estimate, dtype=float).ravel()
    truth_sign = np.asarray(truth_sign).ravel()
    keep = np.isfinite(estimate) & np.isfinite(truth_sign) & (truth_sign != 0)
    estimate, truth_sign = estimate[keep], truth_sign[keep]
    up = truth_sign > 0
    down = ~up
    n_up, n_down = int(up.sum()), int(down.sum())
    if not n_up or not n_down:
        return dict(n_increment=n_up, n_decrement=n_down, acc_increment=np.nan,
                    acc_decrement=np.nan, accuracy=np.nan, auc=np.nan, bias=np.nan)
    called_up = estimate > 0
    acc_up = float(np.mean(called_up[up]))
    acc_down = float(np.mean(~called_up[down]))
    # Mann-Whitney U / (n_up * n_down) -- the probability that a random
    # increment sample is ranked above a random decrement one.
    order = np.argsort(estimate, kind='mergesort')
    ranks = np.empty(estimate.size, dtype=float)
    ranks[order] = np.arange(1, estimate.size + 1)
    auc = float((ranks[up].sum() - n_up * (n_up + 1) / 2.0) / (n_up * n_down))
    return dict(n_increment=n_up, n_decrement=n_down,
                acc_increment=acc_up, acc_decrement=acc_down,
                accuracy=0.5 * (acc_up + acc_down), auc=auc,
                bias=float(np.mean(called_up)))


# The decoder's window floor, separate from MIN_LN_WINDOW_S and smaller.
#
# A decoding filter is one linear operator estimated by ratio of spectra; an LN
# fit is a filter plus a four-parameter nonlinearity fitted by least squares,
# so it needs far more data per window. The recovery this section exists to
# measure happens in the first 1-3 s, which a 3 s window cannot resolve at all:
# on 2020-06-11_B, 1 s windows show AUC climbing 0.73 -> 0.81 -> 0.86 over the
# first three seconds and then flat, while 3.2 s windows show 0.85 throughout.
MIN_DECODE_WINDOW_S = 1.0


def decode_windows(analysis: ConditionAnalysis,
                   mode: str = 'per_window',
                   steady_state_s: float = 10.0,
                   window_seconds: Optional[float] = None,
                   min_window_s: float = MIN_DECODE_WINDOW_S,
                   decode_bin_ms: float = 25.0,
                   noise_ratio: float = 0.1,
                   max_folds: int = 5,
                   random_state: Optional[int] = 0,
                   verbose: bool = True) -> pd.DataFrame:
    """Decode stimulus polarity from the response, window by window.

    A linear decoder is fitted from response back onto stimulus
    (:func:`decoding_filter`), the reconstruction is averaged into
    ``decode_bin_ms`` bins, and its sign is compared with the true light
    polarity in the same bins -- scored **separately for increments and
    decrements**, because the two are not symmetric in a rectifying cell and
    pooling them hides which one adaptation costs.

    ``decode_bin_ms`` sets the question being asked: at 1 ms it is "was the
    light above its mean in this millisecond", which even a perfect decoder
    answers poorly because a spike rate in one millisecond is nearly all noise.
    25 ms is near the filter's own width, so the question is one the response
    can actually carry. On 2020-06-11_B mean AUC runs 0.79 at 100 ms bins and
    0.85 at 25 ms.

    **``skip_seconds`` decides whether the recovery is visible at all.** The
    step happens at t=0 of every epoch, the rate transient is over within about
    3 s, and an ``analysis`` built with ``skip_seconds=1`` has already thrown
    away the first third of it. Build the analysis passed here with
    ``skip_seconds=0``.

    Two modes, and the comparison between them is the point:

    ``'per_window'``
        Fit the model inside each window on training epochs and decode a
        held-out epoch of that same window. The decoder is recalibrated as the
        cell adapts, so this measures how much polarity information the
        response carries at that moment, independent of any drift.

    ``'steady_state'``
        Fit one model on the last ``steady_state_s`` of the epoch -- the
        adapted state -- and decode the earlier windows with it. **Windows
        overlapping the training stretch are excluded**, so nothing is decoded
        by a model that saw it. This measures what a downstream reader with a
        fixed, adapted calibration would make of the un-adapted response, which
        is the quantity that should recover with time since the step.

    Held-out epochs matter for ``'per_window'`` and would matter for
    ``'steady_state'`` too were the training stretch not disjoint in time; it
    is, which is why the exclusion is enforced rather than assumed.
    """
    if mode not in ('per_window', 'steady_state'):
        raise ValueError("mode must be 'per_window' or 'steady_state'")

    cutoff = (analysis.frequency_cutoff
              if np.isfinite(analysis.frequency_cutoff) else None)
    interval = analysis.sampling_interval
    widths = [analysis.stimulus[m].shape[1] for m in analysis.light_means
              if analysis.stimulus.get(m) is not None and analysis.stimulus[m].size]
    if not widths:
        return pd.DataFrame()
    usable_s = min(widths) * interval
    if window_seconds is None:
        n_windows, window_seconds = temporal_windows(usable_s, min_window_s)
    else:
        n_windows = max(int(round(usable_s / float(window_seconds))), 1)
        window_seconds = usable_s / n_windows

    rng = np.random.default_rng(random_state)
    rows: List[dict] = []
    for mean_level in analysis.light_means:
        stim = analysis.stimulus.get(mean_level)
        resp = analysis.response.get(mean_level)
        if stim is None or resp is None or not stim.size:
            continue
        n_epochs, n_time = stim.shape
        edges = np.linspace(0, n_time, n_windows + 1).round().astype(int)
        # Polarity is defined against each epoch's own mean intensity, which is
        # the light level the cell is adapting to -- not against zero.
        truth = np.sign(stim - stim.mean(axis=1, keepdims=True))

        steady_lo = n_time
        steady_model = None
        if mode == 'steady_state':
            steady_lo = max(int(round((usable_s - steady_state_s) / interval)), 0)
            if n_time - steady_lo < 2:
                continue
            steady_model = decoding_filter(resp[:, steady_lo:],
                                           stim[:, steady_lo:],
                                           noise_ratio=noise_ratio)

        for index in range(n_windows):
            lo, hi = int(edges[index]), int(edges[index + 1])
            if hi - lo < 2:
                continue
            start_s = lo * interval + analysis.skip_seconds
            end_s = hi * interval + analysis.skip_seconds
            label = f'{start_s:.1f}-{end_s:.1f} s'

            if mode == 'steady_state':
                # Refuse any window that touches what the model was fitted on.
                if hi > steady_lo:
                    continue
                folds = [(np.arange(n_epochs), steady_model)]
            else:
                order = rng.permutation(n_epochs)
                folds = []
                for held in order[:min(max_folds, n_epochs)]:
                    train = np.setdiff1d(np.arange(n_epochs), [held])
                    if not train.size:
                        continue
                    folds.append((np.array([held]),
                                  decoding_filter(resp[train, lo:hi],
                                                  stim[train, lo:hi],
                                                  noise_ratio=noise_ratio)))

            bin_samples = max(int(round(decode_bin_ms / 1e3 / interval)), 1)
            for test_idx, decoder in folds:
                if decoder is None:
                    continue
                columns = np.arange(lo, hi)
                estimate = apply_decoding_filter(
                    decoder, resp[np.ix_(test_idx, columns)])
                true_block = (stim[np.ix_(test_idx, columns)]
                              - stim[test_idx].mean(axis=1, keepdims=True))
                # Score at the bin the response can actually resolve, not at
                # the sampling interval.
                metrics = _decode_metrics(
                    _bin_mean(estimate - estimate.mean(axis=1, keepdims=True),
                              bin_samples),
                    np.sign(_bin_mean(true_block, bin_samples)))
                rows.append(dict(lightMean=mean_level, window=label, order=index,
                                 start_s=start_s, centre_s=0.5 * (start_s + end_s),
                                 mode=mode, n_test_epochs=int(test_idx.size),
                                 bin_ms=decode_bin_ms, **metrics))

    frame = pd.DataFrame(rows)
    if frame.empty:
        if verbose:
            print('  no windows decoded')
        return frame
    grouped = (frame.groupby(['lightMean', 'order', 'window', 'centre_s', 'mode'],
                             as_index=False)
               .agg(acc_increment=('acc_increment', 'mean'),
                    acc_decrement=('acc_decrement', 'mean'),
                    accuracy=('accuracy', 'mean'),
                    auc=('auc', 'mean'),
                    bias=('bias', 'mean'),
                    n_folds=('accuracy', 'size'))
               .sort_values(['lightMean', 'order']).reset_index(drop=True))
    if verbose:
        span = (f'{grouped.centre_s.min():.1f}-{grouped.centre_s.max():.1f} s'
                if len(grouped) else 'nothing')
        print(f'  {mode}: {len(grouped)} windows decoded ({span})'
              + (f', last {steady_state_s:g} s held back as the training stretch'
                 if mode == 'steady_state' else ''))
        for mean_level, group in grouped.groupby('lightMean'):
            first, last = group.iloc[0], group.iloc[-1]
            print(f'    lightMean {mean_level:g}: AUC {first.auc:.3f} -> '
                  f'{last.auc:.3f} | increments {first.acc_increment:.3f} -> '
                  f'{last.acc_increment:.3f} | decrements '
                  f'{first.acc_decrement:.3f} -> {last.acc_decrement:.3f}')
    return grouped


def decode_recovery(analysis: ConditionAnalysis, **kwargs) -> pd.DataFrame:
    """Both decoding modes in one frame, for plotting side by side."""
    frames = [decode_windows(analysis, mode=mode, **kwargs)
              for mode in ('per_window', 'steady_state')]
    frames = [f for f in frames if not f.empty]
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())


def plot_decoding(analysis: ConditionAnalysis, decoded: pd.DataFrame,
                  figsize: Tuple[float, float] = (12.0, 7.0)):
    """Decoding performance against time since the step, by polarity.

    Columns are the two modes: a decoder recalibrated in every window, and one
    calibrated on the adapted state and applied backwards. Rows are the
    threshold-free AUC and the per-class recall. If adaptation costs the cell
    information, AUC recovers in both; if it only moves the operating point,
    AUC is flat and the recovery shows up in the fixed-calibration column.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if decoded is None or decoded.empty:
        print('nothing decoded')
        return None
    modes = [m for m in ('per_window', 'steady_state')
             if m in set(decoded['mode'])]
    means = [m for m in analysis.light_means
             if m in set(decoded.lightMean)]
    colors = style.colors_for_conditions([f'{m:g}' for m in means])
    titles = {'per_window': 'recalibrated each window',
              'steady_state': 'calibrated on the adapted state'}

    fig, axes = plt.subplots(2, len(modes), figsize=figsize, squeeze=False,
                             sharex=True, sharey='row')
    for col, mode in enumerate(modes):
        block_mode = decoded[decoded['mode'].eq(mode)]
        top, bottom = axes[0][col], axes[1][col]
        for mean_level in means:
            block = block_mode[block_mode.lightMean.eq(mean_level)].sort_values('centre_s')
            if block.empty:
                continue
            color = colors[f'{mean_level:g}']
            top.plot(block.centre_s, block.auc, 'o-', ms=4, lw=1.7, color=color,
                     label=f'lightMean {mean_level:g}')
            bottom.plot(block.centre_s, block.acc_increment, 'o-', ms=4, lw=1.6,
                        color=color, label=f'{mean_level:g} increment')
            bottom.plot(block.centre_s, block.acc_decrement, 's--', ms=4, lw=1.6,
                        color=color, mfc='none', label=f'{mean_level:g} decrement')
        for ax in (top, bottom):
            ax.axhline(0.5, color='0.6', lw=0.8, ls=':')
            ax.set_xlabel('time since luminance step (s)', fontsize=9)
        top.set_title(titles[mode], fontsize=10)
        if col == 0:
            top.set_ylabel('AUC (threshold-free)', fontsize=9)
            bottom.set_ylabel('recall by polarity', fontsize=9)
            top.legend(frameon=False, fontsize=7)
            bottom.legend(frameon=False, fontsize=6, ncol=2)
    fig.suptitle(f'{analysis.exp_name} | {analysis.rec_type} | '
                 f'decoding light polarity from the response  '
                 f'(dotted line = chance)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def light_level_table(protocol_blocks: pd.DataFrame,
                      show: bool = False) -> pd.DataFrame:
    """One row per (experiment, LED): the resolved attenuation for each.

    Wraps the per-group :func:`led_attenuation` call that was a loop in the
    notebook.
    """
    rows = []
    for (exp_name, led), group in protocol_blocks.groupby(['exp_name', 'led'],
                                                          dropna=False):
        entry = led_attenuation(group.iloc[0])
        entry['n_blocks'] = len(group)
        rows.append(entry)
    frame = pd.DataFrame(rows)
    if show and len(frame):
        unresolved = frame[frame.unknown_tokens.ne('')]
        print(f'{len(frame)} experiment x LED combinations | '
              f'{int(frame.wheel_ignored.sum())} list an FW filter that is not '
              f'in the LED path (stripped, named in wheel_tokens_ignored)')
        if len(unresolved):
            print(f'{len(unresolved)} with filters missing from the rig LED '
                  f'table: {sorted(set(unresolved.unknown_tokens))}')
    return frame


def plot_temporal_ln(analysis: ConditionAnalysis,
                     models: Dict[float, List[LNModel]],
                     figsize: Optional[Tuple[float, float]] = None):
    """Filter and nonlinearity per window, one column per light mean.

    Windows run dark to light within a column, so a filter that speeds up or a
    nonlinearity that steepens as the cell adapts reads as a progression rather
    than a pile of lines.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    means = [m for m in analysis.light_means if models.get(m)]
    if not means:
        print('no windowed models to plot')
        return None
    if figsize is None:
        figsize = (5.2 * len(means), 7.0)
    fig, axes = plt.subplots(2, len(means), figsize=figsize, squeeze=False)
    for column, mean_level in enumerate(means):
        window_models = models[mean_level]
        shades = style.colors_for_conditions(
            [m.label for m in window_models], cmap_name='CubicL'
            if False else 'cividis')
        for model in window_models:
            color = shades[model.label]
            axes[0][column].plot(model.filter_time_s * 1e3, model.filter,
                                 lw=1.6, color=color,
                                 label=f'{model.label} (r²={model.r2:.2f})')
            axes[1][column].plot(model.nl_x, model.nl_y, 'o', ms=2.5,
                                 alpha=0.5, color=color)
            params = model.params
            if params and np.isfinite(params.get('alpha', np.nan)):
                grid = np.linspace(np.nanmin(model.nl_x),
                                   np.nanmax(model.nl_x), 200)
                axes[1][column].plot(
                    grid, sigmoid(grid, params['alpha'], params['beta'],
                                  params['gamma'], params['epsilon']),
                    lw=1.5, color=color, label=model.label)
        axes[0][column].axhline(0, color='#888888', lw=0.8, ls='--')
        axes[0][column].set_xlabel('filter time (ms)')
        axes[0][column].set_title(f'lightMean {mean_level:g} — filter',
                                  fontsize=9)
        axes[0][column].legend(frameon=False, fontsize=6.5)
        axes[1][column].set_xlabel('generator signal (contrast)')
        axes[1][column].set_title(f'lightMean {mean_level:g} — nonlinearity',
                                  fontsize=9)
        axes[1][column].legend(frameon=False, fontsize=6.5)
        if column == 0:
            axes[0][column].set_ylabel('filter')
            axes[1][column].set_ylabel(analysis.units)
    fig.suptitle(f'{analysis.exp_name} | LN model per window, '
                 f'{analysis.rec_type}', fontsize=11)
    fig.tight_layout()
    return fig


def plot_condition(analysis: ConditionAnalysis, window_seconds: float = 10.0,
                   positions: Sequence[str] = ('first', 'middle', 'last'),
                   width: float = 13.0, row_height: float = 2.0):
    """The model on the first row, then one row per light mean per window.

    The top row is the fitted filter and the nonlinearity. The filter is
    divided by ``contrast_generator / contrast_stimulus``, so the generator
    signal carries the same ``sigma / mean`` as the stimulus and the
    nonlinearity's x axis is one contrast axis across light levels.

    Every row below is a single light mean over one stretch of an epoch,
    measured against predicted. Light means are drawn separately rather than
    overlaid -- two rate traces on one axis hide each other -- and
    ``window_seconds`` sets how much of the epoch each row shows.
    ``positions`` chooses which stretches: 'first', 'middle' and 'last' by
    default, so the model can be judged early and late in the same epoch.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from retinanalysis.utils import style

    style.apply_publication_style()
    means = analysis.light_means
    colors = style.colors_for_conditions([f'{m:g}' for m in means])
    positions = tuple(positions)
    panels = [(m, p) for m in means for p in positions]

    fig = plt.figure(figsize=(width, row_height * (1.6 + len(panels)) + 0.4))
    grid = GridSpec(1 + len(panels), 2, figure=fig,
                    height_ratios=[1.6] + [1.0] * len(panels), hspace=0.75)
    ax_filter = fig.add_subplot(grid[0, 0])
    ax_nl = fig.add_subplot(grid[0, 1])

    for mean_level in means:
        model = analysis.ln_model[mean_level]
        color = colors[f'{mean_level:g}']
        ax_filter.plot(
            model.filter_time_s * 1e3, model.filter, lw=1.8, color=color,
            label=f'lightMean {mean_level:g} (n={analysis.n_epochs[mean_level]}, '
                  f'r²={model.r2:.2f})')
        ax_nl.plot(model.nl_x, model.nl_y, 'o', ms=3, alpha=0.6, color=color)
        params = model.params
        if params and np.isfinite(params.get('alpha', np.nan)):
            grid_x = np.linspace(np.nanmin(model.nl_x), np.nanmax(model.nl_x), 200)
            ax_nl.plot(grid_x, sigmoid(grid_x, params['alpha'], params['beta'],
                                       params['gamma'], params['epsilon']),
                       lw=1.6, color=color)
    ax_filter.axhline(0, color='#888888', lw=0.8, ls='--')
    ax_filter.set_xlabel('filter time (ms)')
    ax_filter.set_ylabel('filter')
    ax_filter.set_title('temporal filter', fontsize=9)
    ax_filter.legend(frameon=False, fontsize=7)
    ax_nl.set_xlabel('generator signal (contrast)')
    ax_nl.set_ylabel(analysis.units)
    ax_nl.set_title('nonlinearity', fontsize=9)

    for row, (mean_level, position) in enumerate(panels, start=1):
        ax = fig.add_subplot(grid[row, :])
        model = analysis.ln_model[mean_level]
        color = colors[f'{mean_level:g}']
        if model.example_measured.size:
            time_s = model.example_time_s
            span = time_s[-1] if time_s.size else 0.0
            if position == 'first':
                lo = time_s[0]
            elif position == 'middle':
                lo = max(span / 2 - window_seconds / 2, 0.0)
            else:
                lo = max(span - window_seconds, 0.0)
            keep = (time_s >= lo) & (time_s <= lo + window_seconds)
            ax.plot(time_s[keep], model.example_measured[keep], lw=0.8,
                    color='#555555', alpha=0.75, label='measured')
            ax.plot(time_s[keep], model.example_predicted[keep], lw=1.5,
                    color=color, label='predicted')
            ax.legend(frameon=False, fontsize=7, ncol=2, loc='upper right')
        ax.set_ylabel(analysis.units)
        ax.set_title(f'lightMean {mean_level:g} — {position} '
                     f'{window_seconds:g} s of an epoch', fontsize=9)
        if row == len(panels):
            ax.set_xlabel('time in epoch (s)')

    fig.suptitle(f'{analysis.exp_name} | blocks {analysis.block_ids} | '
                 f'{analysis.rec_type}', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


__all__ = [
    'PROTOCOLS', 'PROTOCOL_SEARCH', 'DEFAULT_SUMMARY_PATH', 'SUMMARY_DIR',
    'STEP_DIRECTIONS', 'STEP_LABELS', 'LNModel', 'ConditionAnalysis',
    'summary_path', 'load_summary', 'load_cell',
    'DATE_OFFSETS', 'SAVED_DATE_OFFSET_DAYS', 'FALLBACK_OFFSETS',
    'SINGLE_CELL_ROOT', 'metadata_files', 'corrected_dates',
    'resolve_roster_files',
    'find_blocks', 'find_protocol_cells', 'cell_blocks', 'plot_traces',
    'epoch_parameters', 'resolve_block_mode', 'load_block_modes',
    'PROTOCOL_PARAMETERS', 'MIN_STIM_TIME_MS', 'PRIMATE_CELL_TYPES',
    'MODE_CACHE_PATH',
    'led_attenuation', 'matlab_randn', 'gaussian_noise_stimulus',
    'epoch_stimulus', 'fit_sigmoid', 'sigmoid', 'fit_ln_model',
    'analyze_condition', 'plot_condition', 'mean_response',
    'plot_mean_response', 'temporal_ln_model', 'plot_temporal_ln',
]
