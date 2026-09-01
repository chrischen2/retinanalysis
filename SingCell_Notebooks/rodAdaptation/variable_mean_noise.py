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
    # The accepted epochs concatenated in **recorded order**, which is not the
    # order `stimulus`/`response` hold them in: those are grouped by light
    # mean, and this protocol alternates means epoch to epoch. Since
    # `interpulseInterval` is 0 the epochs are contiguous in time, so this is
    # one continuous record of the cell stepping between light levels -- the
    # data shape a kinetic model has to be fitted on. Empty until
    # `analyze_condition` fills it.
    sequence_stimulus: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sequence_response: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sequence_light_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sequence_epoch: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
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
    # (light mean, row within that mean's block) per accepted epoch, in the
    # order the rig recorded them.
    recorded_order: List[Tuple[float, int]] = []
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
            recorded_order.append((mean_level, len(stimuli[mean_level]) - 1))
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

    # Walk the recorded order and pull each epoch back out of the per-mean
    # arrays, so the sequence carries the same downsampling and the same
    # holding-current alignment as everything else.
    seq_s, seq_r, seq_m, seq_e = [], [], [], []
    for position, (mean_level, row) in enumerate(recorded_order):
        block = analysis.stimulus.get(mean_level)
        if block is None or row >= block.shape[0]:
            continue
        seq_s.append(block[row])
        seq_r.append(analysis.response[mean_level][row])
        seq_m.append(np.full(block.shape[1], mean_level, dtype=float))
        seq_e.append(np.full(block.shape[1], position, dtype=int))
    if seq_s:
        analysis.sequence_stimulus = np.concatenate(seq_s)
        analysis.sequence_response = np.concatenate(seq_r)
        analysis.sequence_light_mean = np.concatenate(seq_m)
        analysis.sequence_epoch = np.concatenate(seq_e)

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


def reconstruction_metrics(estimate, truth) -> dict:
    """How well a reconstructed stimulus trace matches the real one, by phase.

    ``estimate`` and ``truth`` are both the stimulus about its own mean, in the
    same units, so they can be compared sample for sample. Everything is
    reported three times -- over all samples, over the samples where the true
    light was **above** its mean, and over those where it was **below** -- so
    the question "do we reconstruct increments better than decrements" is
    answered directly rather than inferred from a classification score.

    Three numbers per phase, because reconstruction can fail in three ways:

    ``r``
        correlation between reconstruction and stimulus: does it have the right
        shape and timing, regardless of scale.
    ``gain``
        the through-origin regression slope of reconstruction on stimulus. 1.0
        recovers the stimulus amplitude, below 1 is a compressed
        reconstruction. Correlation cannot see this -- a perfectly shaped
        reconstruction at half amplitude still has r = 1.
    ``nrmse``
        root-mean-square error over the phase, divided by the **overall**
        stimulus standard deviation. Normalising by the overall figure rather
        than by each phase's own keeps the two phases on one scale.

    Conditioning on the sign of the true stimulus restricts its range within
    each phase, which lowers ``r`` relative to the unconditioned value. The
    Gaussian stimulus is symmetric, so it lowers both phases identically and
    the increment-minus-decrement comparison stays fair; the ``_all`` columns
    are there so the absolute level is not read off a conditioned number.
    """
    estimate = np.asarray(estimate, dtype=float).ravel()
    truth = np.asarray(truth, dtype=float).ravel()
    keep = np.isfinite(estimate) & np.isfinite(truth)
    estimate, truth = estimate[keep], truth[keep]
    out: Dict[str, float] = {}
    sigma = float(np.std(truth)) if truth.size else np.nan
    for name, mask in (('all', np.ones(truth.shape, dtype=bool)),
                       ('increment', truth > 0), ('decrement', truth < 0)):
        e, t = estimate[mask], truth[mask]
        n = int(e.size)
        out[f'n_{name}'] = n
        if n < 3 or not np.isfinite(sigma) or sigma == 0:
            out[f'r_{name}'] = out[f'gain_{name}'] = out[f'nrmse_{name}'] = np.nan
            continue
        denom = float(np.sum(t * t))
        out[f'r_{name}'] = (float(np.corrcoef(e, t)[0, 1])
                            if np.std(e) and np.std(t) else np.nan)
        out[f'gain_{name}'] = float(np.sum(e * t) / denom) if denom else np.nan
        out[f'nrmse_{name}'] = float(np.sqrt(np.mean((e - t) ** 2)) / sigma)
    return out


# The reconstruction window floor, separate from MIN_LN_WINDOW_S and smaller.
#
# A decoding filter is one linear operator estimated by a ratio of spectra; an
# LN fit is a filter plus a four-parameter nonlinearity fitted by least
# squares, so it needs far more data per window. The recovery this section
# exists to measure happens in the first 1-3 s, which a 3 s window cannot
# resolve at all: on 2020-06-11_B, 1 s windows show correlation climbing over
# the first three seconds and then flat, while 3.2 s windows show it flat
# throughout.
MIN_DECODE_WINDOW_S = 1.0


def reconstruct_traces(analysis: ConditionAnalysis,
                       mode: str = 'per_window',
                       steady_state_s: float = 10.0,
                       window_seconds: Optional[float] = None,
                       min_window_s: float = MIN_DECODE_WINDOW_S,
                       decode_bin_ms: float = 25.0,
                       noise_ratio: float = 0.1,
                       verbose: bool = True) -> pd.DataFrame:
    """Held-out stimulus reconstructions as a tidy frame, one row per bin.

    ``reconstruct_stimulus`` summarises these traces into three numbers per
    phase; this returns the traces themselves, so the figures and the metrics
    are computed from exactly the same reconstruction rather than from two
    parallel loops that could drift apart.

    Columns: ``lightMean``, ``mode``, ``epoch``, ``window``, ``order``,
    ``time_s`` (bin centre since the step), ``stimulus`` and
    ``reconstruction`` -- both about their own mean, in stimulus units.

    **Every epoch is held out exactly once**, rather than a random sample of
    them: a mean over folds is insensitive to which epochs were drawn, but an
    average aligned on phase onsets is not, and repeated epochs would weight
    their own onsets several times over.
    """
    if mode not in ('per_window', 'steady_state'):
        raise ValueError("mode must be 'per_window' or 'steady_state'")

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
    bin_samples = max(int(round(decode_bin_ms / 1e3 / interval)), 1)

    pieces: List[pd.DataFrame] = []
    for mean_level in analysis.light_means:
        stim = analysis.stimulus.get(mean_level)
        resp = analysis.response.get(mean_level)
        if stim is None or resp is None or not stim.size or stim.shape[0] < 2:
            continue
        n_epochs, n_time = stim.shape
        edges = np.linspace(0, n_time, n_windows + 1).round().astype(int)

        steady_lo, steady_decoder = n_time, None
        if mode == 'steady_state':
            steady_lo = max(int(round((usable_s - steady_state_s) / interval)), 0)
            if n_time - steady_lo < 2:
                continue
            steady_decoder = decoding_filter(resp[:, steady_lo:],
                                             stim[:, steady_lo:],
                                             noise_ratio=noise_ratio)

        for index in range(n_windows):
            lo, hi = int(edges[index]), int(edges[index + 1])
            if hi - lo < 2:
                continue
            start_s = lo * interval + analysis.skip_seconds
            label = f'{start_s:.1f}-{hi * interval + analysis.skip_seconds:.1f} s'
            columns = np.arange(lo, hi)

            if mode == 'steady_state':
                if hi > steady_lo:      # refuse anything the decoder was fitted on
                    continue
                folds = [(epoch, steady_decoder) for epoch in range(n_epochs)]
            else:
                folds = []
                for epoch in range(n_epochs):     # full leave-one-out
                    train = np.setdiff1d(np.arange(n_epochs), [epoch])
                    if train.size:
                        folds.append((epoch,
                                      decoding_filter(resp[train, lo:hi],
                                                      stim[train, lo:hi],
                                                      noise_ratio=noise_ratio)))

            for epoch, decoder in folds:
                if decoder is None:
                    continue
                estimate = apply_decoding_filter(decoder, resp[epoch:epoch + 1, columns])
                estimate = estimate - estimate.mean()
                truth = stim[epoch:epoch + 1, columns] - stim[epoch].mean()
                binned_e = _bin_mean(estimate, bin_samples).ravel()
                binned_t = _bin_mean(truth, bin_samples).ravel()
                if binned_e.size == 0:
                    continue
                time_s = ((np.arange(binned_e.size) + 0.5) * bin_samples * interval
                          + lo * interval + analysis.skip_seconds)
                pieces.append(pd.DataFrame({
                    'lightMean': mean_level, 'mode': mode, 'epoch': epoch,
                    'window': label, 'order': index, 'time_s': time_s,
                    'stimulus': binned_t, 'reconstruction': binned_e}))

    frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if verbose:
        if frame.empty:
            print(f'  {mode}: nothing reconstructed')
        else:
            print(f'  {mode}: {len(frame)} bins from '
                  f'{frame.groupby(["lightMean", "epoch"]).ngroups} '
                  f'(light mean x epoch) reconstructions')
    return frame


def reconstruct_stimulus(analysis: ConditionAnalysis,
                         mode: str = 'per_window',
                         steady_state_s: float = 10.0,
                         window_seconds: Optional[float] = None,
                         min_window_s: float = MIN_DECODE_WINDOW_S,
                         decode_bin_ms: float = 25.0,
                         noise_ratio: float = 0.1,
                         verbose: bool = True) -> pd.DataFrame:
    """Reconstruct the stimulus trace window by window and score it by phase.

    A linear decoding filter is fitted from response back onto stimulus
    (:func:`decoding_filter`) and applied to a held-out epoch, giving a
    reconstructed light trace in stimulus units. :func:`reconstruction_metrics`
    then scores that trace separately over the increment and decrement phases
    of the real stimulus.

    Two train/test regimes, and the comparison between them is the point:

    ``'per_window'``
        Fit the decoder inside each window on training epochs and reconstruct a
        held-out epoch of that same window. The decoder is refitted as the cell
        adapts, so this measures how well the response supports reconstruction
        at that moment, independent of any drift in the operating point.

    ``'steady_state'``
        Fit one decoder on the last ``steady_state_s`` -- the adapted state --
        and reconstruct the earlier windows with it. **Windows overlapping the
        training stretch are refused**, so nothing is scored by a decoder that
        saw it. This measures what a downstream reader with a fixed, adapted
        decoder would recover from the un-adapted response.

    ``decode_bin_ms`` averages the reconstruction and the stimulus into bins
    before scoring. At 1 ms the comparison is dominated by the noise in a
    single millisecond of spike rate; 25 ms is near the filter's own width.

    **``skip_seconds`` decides whether the recovery is visible at all.** The
    step is at t=0 and the transient is over within about 3 s, so an
    ``analysis`` built with ``skip_seconds=1`` has already thrown away the
    first third of it. Build the one passed here with ``skip_seconds=0``.
    """
    traces = reconstruct_traces(
        analysis, mode=mode, steady_state_s=steady_state_s,
        window_seconds=window_seconds, min_window_s=min_window_s,
        decode_bin_ms=decode_bin_ms, noise_ratio=noise_ratio, verbose=False)
    if traces.empty:
        if verbose:
            print(f'  {mode}: no windows reconstructed')
        return pd.DataFrame()

    rows: List[dict] = []
    for (mean_level, order, label), block in traces.groupby(
            ['lightMean', 'order', 'window'], sort=True):
        for epoch, piece in block.groupby('epoch'):
            metrics = reconstruction_metrics(piece.reconstruction.values,
                                             piece.stimulus.values)
            rows.append(dict(lightMean=mean_level, window=label, order=order,
                             start_s=float(piece.time_s.min()),
                             centre_s=float(piece.time_s.mean()),
                             mode=mode, epoch=int(epoch),
                             bin_ms=decode_bin_ms, **metrics))

    frame = pd.DataFrame(rows)
    value_cols = [c for c in frame.columns
                  if c.startswith(('r_', 'gain_', 'nrmse_', 'n_'))]
    grouped = (frame.groupby(['lightMean', 'order', 'window', 'centre_s', 'mode'],
                             as_index=False)[value_cols].mean()
               .sort_values(['lightMean', 'order']).reset_index(drop=True))
    grouped['r_asymmetry'] = grouped.r_increment - grouped.r_decrement
    grouped['gain_asymmetry'] = grouped.gain_increment - grouped.gain_decrement
    if verbose:
        for mean_level, group in grouped.groupby('lightMean'):
            first, last = group.iloc[0], group.iloc[-1]
            print(f'  {mode:<12} lightMean {mean_level:g}: '
                  f'r_all {first.r_all:.3f} -> {last.r_all:.3f} | '
                  f'r_inc {group.r_increment.mean():.3f} vs '
                  f'r_dec {group.r_decrement.mean():.3f} | '
                  f'gain_inc {group.gain_increment.mean():.2f} vs '
                  f'gain_dec {group.gain_decrement.mean():.2f}')
    return grouped


def decode_recovery(analysis: ConditionAnalysis,
                    modes: Sequence[str] = ('per_window', 'steady_state'),
                    verbose: bool = True, **kwargs) -> pd.DataFrame:
    """Reconstruction quality under both train/test regimes, in one frame."""
    frames = [reconstruct_stimulus(analysis, mode=mode, verbose=verbose, **kwargs)
              for mode in modes]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def reconstruction_summary(decoded: pd.DataFrame) -> pd.DataFrame:
    """One row per regime x light mean: recovery, and the phase asymmetry.

    ``r_all_first``/``r_all_last`` are the recovery over time.
    ``r_asymmetry`` and ``gain_asymmetry`` are increment minus decrement, so a
    positive sign means the increment phase is reconstructed better.
    """
    if decoded is None or decoded.empty:
        return pd.DataFrame()
    ordered = decoded.sort_values('centre_s')
    return (ordered.groupby(['mode', 'lightMean'], as_index=False)
            .agg(r_all_first=('r_all', 'first'), r_all_last=('r_all', 'last'),
                 r_inc=('r_increment', 'mean'), r_dec=('r_decrement', 'mean'),
                 gain_inc=('gain_increment', 'mean'),
                 gain_dec=('gain_decrement', 'mean'),
                 nrmse_inc=('nrmse_increment', 'mean'),
                 nrmse_dec=('nrmse_decrement', 'mean'),
                 r_asymmetry=('r_asymmetry', 'mean'),
                 gain_asymmetry=('gain_asymmetry', 'mean'),
                 n_windows=('r_all', 'size'))
            .assign(r_gain=lambda f: f.r_all_last - f.r_all_first))


def plot_reconstruction_trace(analysis: ConditionAnalysis,
                              light_mean: Optional[float] = None,
                              seconds: Tuple[float, float] = (0.0, 4.0),
                              decode_bin_ms: float = 25.0,
                              noise_ratio: float = 0.1,
                              figsize: Tuple[float, float] = (12.0, 4.6)):
    """The reconstructed light trace against the real one, phases shaded.

    This is the analysis before any summary statistic: a decoder is fitted on
    every epoch but one, and the held-out epoch's stimulus is reconstructed
    from its response alone. Increment phases of the true stimulus are shaded,
    so whether the reconstruction tracks the light better above or below the
    mean is visible directly rather than only through the numbers.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    means = ([light_mean] if light_mean is not None else analysis.light_means)
    means = [m for m in means if analysis.stimulus.get(m) is not None]
    if not means:
        print('nothing to reconstruct')
        return None
    interval = analysis.sampling_interval
    bin_samples = max(int(round(decode_bin_ms / 1e3 / interval)), 1)
    colors = style.colors_for_conditions([f'{m:g}' for m in means])

    fig, axes = plt.subplots(len(means), 1, figsize=(figsize[0],
                             figsize[1] * len(means) / 2 + 1.4), squeeze=False,
                             sharex=True)
    for row, mean_level in enumerate(means):
        ax = axes[row][0]
        stim = analysis.stimulus[mean_level]
        resp = analysis.response[mean_level]
        lo = max(int(round((seconds[0] - analysis.skip_seconds) / interval)), 0)
        hi = min(int(round((seconds[1] - analysis.skip_seconds) / interval)),
                 stim.shape[1])
        if hi - lo < 2 or stim.shape[0] < 2:
            continue
        train = np.arange(1, stim.shape[0])
        decoder = decoding_filter(resp[train, lo:hi], stim[train, lo:hi],
                                  noise_ratio=noise_ratio)
        estimate = apply_decoding_filter(decoder, resp[0:1, lo:hi])
        estimate = estimate - estimate.mean()
        truth = stim[0:1, lo:hi] - stim[0].mean()
        e = _bin_mean(estimate, bin_samples).ravel()
        t = _bin_mean(truth, bin_samples).ravel()
        time_s = (np.arange(t.size) + 0.5) * bin_samples * interval + \
            lo * interval + analysis.skip_seconds

        # Shade the increment phases of the real stimulus.
        above = t > 0
        edges = np.flatnonzero(np.diff(above.astype(int)))
        starts = np.r_[0, edges + 1]
        stops = np.r_[edges + 1, above.size]
        for a, b in zip(starts, stops):
            if above[a]:
                ax.axvspan(time_s[a], time_s[min(b, time_s.size - 1)],
                           color=colors[f'{mean_level:g}'], alpha=.10, lw=0)
        ax.axhline(0, color='0.6', lw=0.8)
        ax.plot(time_s, t, color='0.25', lw=1.6, label='true stimulus')
        ax.plot(time_s, e, color=colors[f'{mean_level:g}'], lw=1.5, ls='--',
                label='reconstruction (held-out epoch)')
        m = reconstruction_metrics(e, t)
        ax.set_ylabel(f'lightMean {mean_level:g}\n(stimulus units)', fontsize=9)
        ax.text(0.995, 0.04,
                f"r  inc {m['r_increment']:.2f} / dec {m['r_decrement']:.2f}   "
                f"gain  inc {m['gain_increment']:.2f} / dec {m['gain_decrement']:.2f}",
                transform=ax.transAxes, ha='right', fontsize=8, color='0.35')
        if row == 0:
            ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper left')
    axes[-1][0].set_xlabel('time since luminance step (s)', fontsize=9)
    fig.suptitle(f'{analysis.exp_name} | {analysis.rec_type} | reconstructed '
                 f'light trace (shading = true increment phase)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def phase_slopes(stimulus, reconstruction) -> Dict[str, float]:
    """Through-origin slope of reconstruction on stimulus, each side of zero.

    These are the same quantity as ``gain_increment`` / ``gain_decrement`` in
    :func:`reconstruction_metrics` -- deliberately, so the transfer-function
    figure cannot disagree with the table beside it.
    """
    stimulus = np.asarray(stimulus, dtype=float).ravel()
    reconstruction = np.asarray(reconstruction, dtype=float).ravel()
    keep = np.isfinite(stimulus) & np.isfinite(reconstruction)
    stimulus, reconstruction = stimulus[keep], reconstruction[keep]
    out = {}
    for name, mask in (('increment', stimulus > 0), ('decrement', stimulus < 0)):
        t, e = stimulus[mask], reconstruction[mask]
        denom = float(np.sum(t * t))
        out[name] = float(np.sum(e * t) / denom) if denom else np.nan
    return out


def _phase_onsets(stimulus, pre_bins: int, post_bins: int, min_bins: int):
    """Indices where the stimulus crosses zero, upward and downward.

    Returns ``(increment_onsets, decrement_onsets)``. Two filters, both of
    which matter:

    * a crossing is kept only if the phase it starts lasts ``min_bins`` -- the
      band-limited noise still produces brief excursions, and averaging over
      them contributes noise rather than a phase;
    * a crossing is kept only if the full cut ``pre_bins .. post_bins`` fits
      inside the trace, because the reconstruction is an FFT operation and its
      first and last bins carry edge artefacts.
    """
    stimulus = np.asarray(stimulus, dtype=float).ravel()
    sign = np.sign(stimulus)
    crossings = np.flatnonzero(np.diff(sign) != 0) + 1
    if crossings.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    # A phase runs from one crossing to the next; the last runs to the end.
    ends = np.r_[crossings[1:], stimulus.size]
    durations = ends - crossings
    keep = ((durations >= min_bins)
            & (crossings - pre_bins >= 0)
            & (crossings + post_bins < stimulus.size))
    crossings = crossings[keep]
    rising = stimulus[crossings] > 0
    return crossings[rising], crossings[~rising]


def plot_reconstruction_transfer(analysis: ConditionAnalysis,
                                 traces: pd.DataFrame,
                                 gridsize: int = 44,
                                 figsize: Optional[Tuple[float, float]] = None):
    """Reconstruction against true stimulus: where in stimulus space accuracy goes.

    The decoding transfer function. **The phase split is the x axis** -- samples
    left of zero are the decrement phase and those right of it the increment
    phase -- so no conditioning is needed and gain, compression and noise are
    visible at once. A line on the identity means the stimulus is recovered at
    full amplitude; a line flatter than identity on one side means that phase
    is compressed, which is what the ``gain`` column reports as a number.

    Density is the raw scatter of every held-out bin; the heavy line is the
    mean reconstruction given the stimulus, with an inter-quartile band. The
    two straight segments are through-origin fits either side of zero, and
    their slopes are :func:`phase_slopes`, the same numbers as
    ``gain_increment`` and ``gain_decrement``.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if traces is None or traces.empty:
        print('nothing to plot')
        return None
    modes = [m for m in ('per_window', 'steady_state') if m in set(traces['mode'])]
    means = [m for m in analysis.light_means if m in set(traces.lightMean)]
    if not means or not modes:
        print('nothing to plot')
        return None
    colors = style.colors_for_conditions([f'{m:g}' for m in means])
    titles = {'per_window': 'decoder refitted each window',
              'steady_state': 'decoder fitted on the adapted state'}
    if figsize is None:
        figsize = (4.6 * len(modes) + 0.8, 4.1 * len(means))

    fig, axes = plt.subplots(len(means), len(modes), figsize=figsize,
                             squeeze=False)
    for row, mean_level in enumerate(means):
        for col, mode in enumerate(modes):
            ax = axes[row][col]
            block = traces[traces.lightMean.eq(mean_level) & traces['mode'].eq(mode)]
            if block.empty:
                ax.axis('off')
                continue
            x = block.stimulus.values
            y = block.reconstruction.values
            limit = float(np.nanpercentile(np.abs(x), 99.5)) or 1.0
            ax.hexbin(x, y, gridsize=gridsize, cmap='Greys', mincnt=1,
                      extent=(-limit, limit, -limit, limit), linewidths=0)
            ax.plot([-limit, limit], [-limit, limit], ls='--', lw=1.1,
                    color='0.45', zorder=3)

            # Conditional mean of the reconstruction given the stimulus.
            edges = np.linspace(-limit, limit, 19)
            centres = 0.5 * (edges[:-1] + edges[1:])
            which = np.digitize(x, edges) - 1
            mid, low, high = [], [], []
            for b in range(centres.size):
                vals = y[which == b]
                if vals.size < 8:
                    mid.append(np.nan); low.append(np.nan); high.append(np.nan)
                else:
                    mid.append(np.mean(vals))
                    low.append(np.percentile(vals, 25))
                    high.append(np.percentile(vals, 75))
            color = colors[f'{mean_level:g}']
            ax.fill_between(centres, low, high, color=color, alpha=.22, lw=0, zorder=4)
            ax.plot(centres, mid, '-', lw=2.0, color=color, zorder=5)

            slopes = phase_slopes(x, y)
            for name, span in (('decrement', np.array([-limit, 0.0])),
                               ('increment', np.array([0.0, limit]))):
                if np.isfinite(slopes[name]):
                    # Vermillion, not gold: the cividis condition palette is
                    # gold at the bright end and the slope line would vanish
                    # into the conditional-mean curve it is meant to summarise.
                    ax.plot(span, slopes[name] * span, '-', lw=1.4,
                            color='#D55E00', zorder=6)
            ax.axvline(0, color='0.5', lw=0.9)
            ax.axhline(0, color='0.5', lw=0.9)
            ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
            ax.set_aspect('equal', adjustable='box')
            ax.text(0.03, 0.95, f"decrement\nslope {slopes['decrement']:.2f}",
                    transform=ax.transAxes, va='top', fontsize=8, color='#D55E00')
            ax.text(0.97, 0.06, f"increment\nslope {slopes['increment']:.2f}",
                    transform=ax.transAxes, ha='right', fontsize=8, color='#D55E00')
            if row == 0:
                ax.set_title(titles[mode], fontsize=10)
            if col == 0:
                ax.set_ylabel(f'lightMean {mean_level:g}\nreconstruction', fontsize=9)
            if row == len(means) - 1:
                ax.set_xlabel('true stimulus (about its mean)', fontsize=9)
    fig.suptitle(f'{analysis.exp_name} | {analysis.rec_type} | decoding transfer '
                 f'function (grey dashed = identity, orange = per-phase slope, '
                 f'coloured = mean reconstruction given stimulus)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig



# The two halves of the epoch, for the time-split transfer function. Okabe-Ito
# vermillion and blue: the contrast that carries this figure is early against
# late, not one light mean against the other, which is already the row.
EARLY_LATE_COLORS = {'early': '#D55E00', 'late': '#0072B2'}


def label_early_late(traces: pd.DataFrame, early_s: float = 3.0,
                     late_s: float = 3.0) -> pd.DataFrame:
    """Tag every reconstructed bin ``early``, ``late``, or neither.

    :func:`plot_reconstruction_transfer` pools every window, so the curve it
    draws is a mixture of the un-adapted and adapted response. This splits the
    same traces by time since the step so the two can be drawn against each
    other.

    **The cut points are taken per (mode, light mean), not globally.** The two
    decoding regimes do not cover the same windows: the steady-state decoder
    refuses every window it was fitted on, so its last window ends
    ``steady_state_s`` before the end of the epoch. One global cut would
    compare that regime's *middle* against the other's end and call the
    difference adaptation.

    Bins in neither half get ``half = None``; callers drop them. Raises if the
    two halves would overlap, rather than quietly double-counting bins.
    """
    if traces is None or traces.empty:
        return traces
    frame = traces.copy()
    frame['half'] = None
    for (mode, mean_level), block in frame.groupby(['mode', 'lightMean'], sort=False):
        t = block.time_s.values
        lo, hi = float(np.min(t)), float(np.max(t))
        if lo + float(early_s) > hi - float(late_s):
            raise ValueError(
                f'early_s={early_s} and late_s={late_s} overlap for {mode} '
                f'lightMean {mean_level:g}, which spans only {hi - lo:.1f} s '
                f'({lo:.1f}-{hi:.1f} s). Shorten them.')
        index = block.index
        frame.loc[index[t < lo + float(early_s)], 'half'] = 'early'
        frame.loc[index[t > hi - float(late_s)], 'half'] = 'late'
    return frame


def transfer_early_late(traces: pd.DataFrame, early_s: float = 3.0,
                        late_s: float = 3.0) -> pd.DataFrame:
    """Per-phase reconstruction scores, early against late, as a table.

    One row per (mode, light mean, half). The numbers come from
    :func:`reconstruction_metrics` -- the same function behind the pooled
    summary -- so ``gain_increment`` here is the pooled ``gain_inc`` restricted
    in time, and the figure's slope annotations cannot drift from it.

    ``span_s`` records the seconds each half actually covers, since the halves
    are cut per regime and the steady-state one ends early.
    """
    labelled = label_early_late(traces, early_s=early_s, late_s=late_s)
    if labelled is None or labelled.empty:
        return pd.DataFrame()
    rows = []
    for (mode, mean_level, half), block in labelled.dropna(subset=['half']).groupby(
            ['mode', 'lightMean', 'half'], sort=False):
        metrics = reconstruction_metrics(block.reconstruction.values,
                                         block.stimulus.values)
        rows.append({'mode': mode, 'lightMean': mean_level, 'half': half,
                     'span_s': float(block.time_s.max() - block.time_s.min()),
                     't_start_s': float(block.time_s.min()),
                     'n_bins': int(len(block)),
                     **metrics,
                     'gain_asymmetry': metrics['gain_increment'] - metrics['gain_decrement'],
                     'r_asymmetry': metrics['r_increment'] - metrics['r_decrement']})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    order = pd.Categorical(frame['half'], categories=['early', 'late'], ordered=True)
    return frame.assign(half=order).sort_values(['mode', 'lightMean', 'half'],
                                                ignore_index=True)


def plot_transfer_early_late(analysis: ConditionAnalysis,
                             traces: pd.DataFrame,
                             early_s: float = 3.0,
                             late_s: float = 3.0,
                             n_bins: int = 15,
                             min_count: int = 12,
                             figsize: Optional[Tuple[float, float]] = None):
    """The decoding transfer function, early against late in one panel.

    :func:`plot_reconstruction_transfer` marginalises over time, so a curve
    that bends could be bending because the cell saturates or because the first
    second after the step is mixed in with the tenth. This overlays the two
    halves on shared axes: a **shift in slope** between them is a gain change,
    a **change in where the curve leaves the straight part** is a change in the
    saturating range, and curves that lie on top of each other say the
    compression was there all along and is not the adaptation.

    Both halves are binned on one x grid -- the limit is taken over the whole
    panel, not per half -- so the curves are comparable point for point. The
    raw density is deliberately not drawn; two overlaid hexbins are unreadable,
    and the scatter is in :func:`plot_reconstruction_transfer`.

    **Read the earliest window knowing what is in it.** Each decoding window is
    mean-subtracted, which removes the step's sustained offset but not the
    transient inside the first window, and that transient is response the
    stimulus in that window did not cause. It inflates the apparent noise
    early; it is not a reason for a *slope* to differ.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if traces is None or traces.empty:
        print('nothing to plot')
        return None
    labelled = label_early_late(traces, early_s=early_s, late_s=late_s)
    labelled = labelled.dropna(subset=['half'])
    modes = [m for m in ('per_window', 'steady_state') if m in set(labelled['mode'])]
    means = [m for m in analysis.light_means if m in set(labelled.lightMean)]
    if not means or not modes:
        print('nothing to plot')
        return None
    if figsize is None:
        figsize = (4.6 * len(modes) + 0.8, 4.3 * len(means))

    fig, axes = plt.subplots(len(means), len(modes), figsize=figsize, squeeze=False)
    for row, mean_level in enumerate(means):
        for col, mode in enumerate(modes):
            ax = axes[row][col]
            panel = labelled[labelled.lightMean.eq(mean_level)
                             & labelled['mode'].eq(mode)]
            if panel.empty:
                ax.axis('off')
                continue
            # One grid for both halves, from the whole panel.
            limit = float(np.nanpercentile(np.abs(panel.stimulus.values), 99.5)) or 1.0
            edges = np.linspace(-limit, limit, n_bins + 1)
            centres = 0.5 * (edges[:-1] + edges[1:])
            ax.plot([-limit, limit], [-limit, limit], ls='--', lw=1.1,
                    color='0.45', zorder=2)

            for half in ('early', 'late'):
                block = panel[panel['half'].eq(half)]
                if block.empty:
                    continue
                x = block.stimulus.values
                y = block.reconstruction.values
                color = EARLY_LATE_COLORS[half]
                which = np.digitize(x, edges) - 1
                mid, low, high = [], [], []
                for b in range(centres.size):
                    vals = y[which == b]
                    if vals.size < min_count:
                        mid.append(np.nan); low.append(np.nan); high.append(np.nan)
                    else:
                        mid.append(np.mean(vals))
                        low.append(np.percentile(vals, 25))
                        high.append(np.percentile(vals, 75))
                slopes = phase_slopes(x, y)
                span = (block.time_s.min(), block.time_s.max())
                ax.fill_between(centres, low, high, color=color, alpha=.16,
                                lw=0, zorder=3)
                ax.plot(centres, mid, '-', lw=2.1, color=color, zorder=5,
                        label=f'{half} ({span[0]:.1f}-{span[1]:.1f} s)  '
                              f'inc {slopes["increment"]:.2f} / '
                              f'dec {slopes["decrement"]:.2f}')
                for name, side in (('decrement', np.array([-limit, 0.0])),
                                   ('increment', np.array([0.0, limit]))):
                    if np.isfinite(slopes[name]):
                        ax.plot(side, slopes[name] * side, ':', lw=1.3,
                                color=color, zorder=4)
            ax.axvline(0, color='0.5', lw=0.9)
            ax.axhline(0, color='0.5', lw=0.9)
            ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
            ax.set_aspect('equal', adjustable='box')
            ax.legend(frameon=False, fontsize=7.5, loc='upper left')
            if row == 0:
                ax.set_title(titles_mode(mode), fontsize=10)
            if col == 0:
                ax.set_ylabel(f'lightMean {mean_level:g}\nreconstruction', fontsize=9)
            if row == len(means) - 1:
                ax.set_xlabel('true stimulus (about its mean)', fontsize=9)
    fig.suptitle(f'{analysis.exp_name} | {analysis.rec_type} | decoding transfer '
                 f'function, first {early_s:g} s vs last {late_s:g} s '
                 f'(dashed grey = identity, dotted = per-phase slope)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


def plot_phase_triggered(analysis: ConditionAnalysis, traces: pd.DataFrame,
                         pre_ms: float = 100.0, post_ms: float = 300.0,
                         min_phase_ms: float = 100.0,
                         mode: Optional[str] = None,
                         figsize: Optional[Tuple[float, float]] = None):
    """Onset-aligned averages: when within a phase the accuracy goes.

    Every zero crossing of the true stimulus starts a phase. Averaging the
    stimulus and its reconstruction over all increment onsets, and separately
    over all decrement onsets, shows whether a compressed reconstruction is
    compressed from the start or only fails to keep up -- which the transfer
    function, having no time axis, cannot distinguish.

    The peak ratio annotated on each panel is reconstruction peak over stimulus
    peak, the amplitude recovered for that phase; the latency is the difference
    between the two peak times, quantised to ``decode_bin_ms``.

    ``min_phase_ms`` is load-bearing rather than cosmetic. The stimulus is
    band-limited noise, so it crosses zero constantly and most crossings start
    an excursion lasting one or two bins; averaging over those gives the
    zero-crossing waveform of the noise, not the response to a sustained phase.
    On 2020-06-11_B the surviving onset count runs 884 at 50 ms, 251 at 100 ms,
    77 at 150 ms and 23 at 200 ms, so 100 ms is where the phases are long
    enough to mean something and still numerous enough to average.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if traces is None or traces.empty:
        print('nothing to plot')
        return None
    if mode is None:
        mode = ('per_window' if 'per_window' in set(traces['mode'])
                else sorted(set(traces['mode']))[0])
    frame = traces[traces['mode'].eq(mode)]
    means = [m for m in analysis.light_means if m in set(frame.lightMean)]
    if not means:
        print('nothing to plot')
        return None

    # Bin width from the traces themselves, so this cannot disagree with them.
    sample = frame[frame.lightMean.eq(means[0])]
    first = sample.groupby(['epoch', 'window']).time_s.apply(
        lambda v: np.median(np.diff(np.sort(v.values))) if v.size > 1 else np.nan)
    bin_s = float(np.nanmedian(first.values))
    pre_bins = max(int(round(pre_ms / 1e3 / bin_s)), 1)
    post_bins = max(int(round(post_ms / 1e3 / bin_s)), 1)
    min_bins = max(int(round(min_phase_ms / 1e3 / bin_s)), 1)
    lag_ms = (np.arange(-pre_bins, post_bins + 1)) * bin_s * 1e3

    colors = style.colors_for_conditions([f'{m:g}' for m in means])
    if figsize is None:
        figsize = (11.0, 3.1 * len(means) + 1.0)
    fig, axes = plt.subplots(len(means), 2, figsize=figsize, squeeze=False,
                             sharex=True)
    for row, mean_level in enumerate(means):
        block = frame[frame.lightMean.eq(mean_level)]
        cuts = {'increment': {'stim': [], 'rec': []},
                'decrement': {'stim': [], 'rec': []}}
        for _, piece in block.groupby(['epoch', 'window'], sort=False):
            piece = piece.sort_values('time_s')
            stim = piece.stimulus.values
            rec = piece.reconstruction.values
            up, down = _phase_onsets(stim, pre_bins, post_bins, min_bins)
            for name, onsets in (('increment', up), ('decrement', down)):
                for onset in onsets:
                    sl = slice(onset - pre_bins, onset + post_bins + 1)
                    cuts[name]['stim'].append(stim[sl])
                    cuts[name]['rec'].append(rec[sl])

        for col, name in enumerate(('increment', 'decrement')):
            ax = axes[row][col]
            stim_cuts = cuts[name]['stim']
            if not stim_cuts:
                ax.text(0.5, 0.5, f'no {name} onsets survive the\nedge and '
                        f'duration filters', transform=ax.transAxes, ha='center',
                        va='center', fontsize=8, color='0.45')
                ax.set_xticks([])
                continue
            stim_arr = np.vstack(stim_cuts)
            rec_arr = np.vstack(cuts[name]['rec'])
            n = stim_arr.shape[0]
            color = colors[f'{mean_level:g}']
            for arr, style_kw, label in (
                    (stim_arr, dict(color='0.25', lw=1.8, ls='-'), 'true stimulus'),
                    (rec_arr, dict(color=color, lw=1.7, ls='--'), 'reconstruction')):
                mean_trace = arr.mean(axis=0)
                sem = arr.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean_trace)
                ax.fill_between(lag_ms, mean_trace - sem, mean_trace + sem,
                                color=style_kw['color'], alpha=.20, lw=0)
                ax.plot(lag_ms, mean_trace, label=label, **style_kw)
            ax.axvline(0, color='0.5', lw=0.9)
            ax.axhline(0, color='0.5', lw=0.9)

            peak = np.argmax if name == 'increment' else np.argmin
            s_mean, r_mean = stim_arr.mean(axis=0), rec_arr.mean(axis=0)
            s_i, r_i = peak(s_mean), peak(r_mean)
            ratio = (r_mean[r_i] / s_mean[s_i]) if s_mean[s_i] else np.nan
            ax.text(0.97, 0.06 if name == 'increment' else 0.90,
                    f'n = {n}   peak ratio {ratio:.2f}\n'
                    f'latency {lag_ms[r_i] - lag_ms[s_i]:+.0f} ms',
                    transform=ax.transAxes, ha='right',
                    va='bottom' if name == 'increment' else 'top',
                    fontsize=8, color='0.35')
            if row == 0:
                ax.set_title(f'{name} onsets', fontsize=10)
                if col == 0:
                    ax.legend(frameon=False, fontsize=8, loc='upper left')
            if col == 0:
                ax.set_ylabel(f'lightMean {mean_level:g}\n(stimulus units)', fontsize=9)
            if row == len(means) - 1:
                ax.set_xlabel('time from phase onset (ms)', fontsize=9)
    fig.suptitle(f'{analysis.exp_name} | {analysis.rec_type} | phase-onset '
                 f'triggered average ({titles_mode(mode)})', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


def titles_mode(mode: str) -> str:
    """Human-readable name for a decoding regime."""
    return {'per_window': 'decoder refitted each window',
            'steady_state': 'decoder fitted on the adapted state'}.get(mode, mode)


def plot_decoding(analysis: ConditionAnalysis, decoded: pd.DataFrame,
                  figsize: Tuple[float, float] = (12.0, 7.4)):
    """Reconstruction quality by phase, against time since the step.

    Columns are the two train/test regimes. Rows are the three ways a
    reconstruction can be good or bad: shape (``r``), amplitude (``gain``), and
    total error (``nrmse``). Within a panel, solid is the increment phase and
    dashed the decrement phase, so the vertical gap between a matched pair
    **is** the asymmetry the analysis is about.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if decoded is None or decoded.empty:
        print('nothing decoded')
        return None
    modes = [m for m in ('per_window', 'steady_state') if m in set(decoded['mode'])]
    means = [m for m in analysis.light_means if m in set(decoded.lightMean)]
    colors = style.colors_for_conditions([f'{m:g}' for m in means])
    titles = {'per_window': 'decoder refitted each window',
              'steady_state': 'decoder fitted on the adapted state'}
    rows = [('r', 'correlation with stimulus'),
            ('gain', 'recovered amplitude (gain)'),
            ('nrmse', 'error / stimulus SD')]

    fig, axes = plt.subplots(len(rows), len(modes), figsize=figsize,
                             squeeze=False, sharex=True, sharey='row')
    for row, (stem, ylabel) in enumerate(rows):
        for col, mode in enumerate(modes):
            ax = axes[row][col]
            block_mode = decoded[decoded['mode'].eq(mode)]
            for mean_level in means:
                block = block_mode[block_mode.lightMean.eq(mean_level)].sort_values('centre_s')
                if block.empty:
                    continue
                color = colors[f'{mean_level:g}']
                ax.plot(block.centre_s, block[f'{stem}_increment'], '-', lw=1.7,
                        color=color, label=f'{mean_level:g} increment')
                ax.plot(block.centre_s, block[f'{stem}_decrement'], '--', lw=1.5,
                        color=color, label=f'{mean_level:g} decrement')
            if stem == 'gain':
                ax.axhline(1.0, color='0.6', lw=0.8, ls=':')
            if row == 0:
                ax.set_title(titles[mode], fontsize=10)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == len(rows) - 1:
                ax.set_xlabel('time since luminance step (s)', fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=7, ncol=2, loc='lower right')
    fig.suptitle(f'{analysis.exp_name} | {analysis.rec_type} | stimulus '
                 f'reconstruction by phase (solid = increment, dashed = decrement)',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


# --------------------------------------------------------------------------
# 8. LNK: an LN cascade with one slow adaptive state
#
# Ozuysal & Baccus (2012) put all of contrast adaptation in a four-state
# kinetic block after a filter and nonlinearity. Two of those states carry the
# *fast* dynamics, and our filters show no fast change to explain: over nine
# windows spanning a 30 s epoch, time-to-peak is constant to a millisecond or
# two. Fitting four fast parameters to a flat line is how identifiability
# problems start, so this is the reduction the data licenses -- one slow state,
# seven free parameters against 570 s at 1 kHz rather than 26.
#
# What the kinetics block does to the gain is a scaling (supplement eq 18:
# gain proportional to resting-state occupancy), so a slow state can rescale
# the nonlinearity but cannot restructure the filter. The filter *is* different
# between light levels here -- 32 ms against 59 ms, biphasic index changing
# sign -- so each level gets its own, and the shared nonlinearity plus one
# state is what the fit is being asked to explain.
#
# ONE ARCHITECTURAL DEPARTURE, STATED PLAINLY. In baccuslab/LNKS the kinetics
# block *is* the output stage: `v = X[1,:]` is the active-state occupancy,
# min-max normalised, and the nonlinearity only supplies its drive -- which is
# why two nonlinearity parameters suffice there and four are needed here. In
# this reduction the nonlinearity is the output stage and the slow state
# modulates it. That follows from dropping the fast states: in LNK the block is
# simultaneously the fast output stage and the slow adaptation mechanism, so
# keeping only the slow half leaves nothing to generate the response and the
# nonlinearity has to take over.
#
# The consequence is worth being clear about. This is a *phenomenological*
# model of the observed nonlinearity change -- it imposes a slope change or a
# shift and asks which fits better -- not a mechanistic one that derives either
# from depletion. A two-state variant, one fast state as the output stage and
# one slow state for adaptation, would restore the LNK architecture and let the
# nonlinearity change emerge rather than be imposed. That is the natural next
# step and it is not implemented here.
#
# The state is driven by the cell's own rectified drive and returns to the
# nonlinearity by exactly one of two routes, which is the experiment:
#
#   multiplicative   r = alpha exp(-k a') Phi(beta g + gamma) + eps  -> SLOPE
#   subtractive      r = alpha Phi(beta g + gamma - k a') + eps       -> SHIFT
#
# where a' is the state standardised to zero mean and unit variance. Neither
# its mean nor its scale is identifiable -- the mean is absorbed by `alpha`
# (multiplicatively) or `gamma` (subtractively), and the scale trades off
# against the time constants -- so `k` is fitted as modulation per standard
# deviation of adaptation, which is comparable between couplings and cells.
#
# A gain-only mechanism cannot produce a shift: scaling a rectifying
# nonlinearity compresses it toward zero, it does not translate it. So the two
# couplings are not two settings of one mechanism, and which one a cell needs
# is a model comparison rather than a parameter readout.
# --------------------------------------------------------------------------
LNK_COUPLINGS = ('multiplicative', 'subtractive')


@dataclass
class LNKModel:
    """One fitted LNK, with the static LN it has to beat."""

    coupling: str
    params: Dict[str, float] = field(default_factory=dict)
    filter: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # One filter per light mean: they differ enough between levels that a
    # pooled one describes neither. `filter` is the first for convenience.
    filters: Dict[float, np.ndarray] = field(default_factory=dict)
    # r2 of the 5-parameter form against each level's reverse-correlation
    # estimate, so a filter the form cannot represent is visible not silent.
    filter_r2: Dict[float, float] = field(default_factory=dict)
    filter_params: Dict[float, np.ndarray] = field(default_factory=dict)
    fit_filter: bool = False
    filter_time_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sampling_interval: float = np.nan
    state_dt_s: float = np.nan
    r2: float = np.nan            # held-out epochs, adaptive model
    r2_train: float = np.nan
    r2_static: float = np.nan     # held-out epochs, same model with k = 0
    n_train_epochs: int = 0
    n_test_epochs: int = 0
    predicted: np.ndarray = field(default_factory=lambda: np.zeros(0))
    predicted_static: np.ndarray = field(default_factory=lambda: np.zeros(0))
    state: np.ndarray = field(default_factory=lambda: np.zeros(0))
    generator: np.ndarray = field(default_factory=lambda: np.zeros(0))
    at_bounds: Tuple[str, ...] = ()

    @property
    def r2_gain(self) -> float:
        """What the adaptive state buys over the same cascade without it."""
        return self.r2 - self.r2_static

    def __repr__(self) -> str:
        return (f'<LNKModel {self.coupling} | r2 {self.r2:.3f} held out '
                f'(static {self.r2_static:.3f}) | tau_on '
                f'{self.params.get("tau_on", np.nan):.2f} s, tau_off '
                f'{self.params.get("tau_off", np.nan):.2f} s, k '
                f'{self.params.get("k", np.nan):.3f}>')


def _relax(steady, rate, dt: float, a0: float = 0.0) -> np.ndarray:
    """Integrate ``x' = rate * (steady - x)`` for time-varying ``steady``/``rate``.

    Exponential Euler, exact for values held constant across a step, so the
    step size is a speed choice and not a stability one.

    The recurrence ``x[n] = c[n] x[n-1] + d[n]`` has a closed form,
    ``x[n] = P[n] (x0 + sum_{m<=n} d[m]/P[m])`` with ``P[n] = prod_{j<=n} c[j]``,
    which turns a per-sample Python loop into two cumulative operations. The
    catch is that ``P`` underflows -- ``c`` is just under 1 and there are tens
    of thousands of samples -- so it is applied in blocks short enough that
    ``P`` cannot fall below about 1e-150, restarting each block from the
    previous one's last value. Exact, and about 19x faster than the loop.
    """
    steady = np.asarray(steady, dtype=float)
    rate = np.asarray(rate, dtype=float)
    if steady.size == 0:
        return np.zeros(0)
    decay = np.exp(-dt * rate)
    offsets = steady * (1.0 - decay)
    worst = float(np.max(dt * rate))
    block = 4096 if worst <= 0 else int(np.clip(150 * np.log(10) / worst, 8, 4096))
    out = np.empty_like(steady)
    previous = float(a0)
    for start in range(0, steady.size, block):
        stop = min(start + block, steady.size)
        products = np.cumprod(decay[start:stop])
        out[start:stop] = products * (previous
                                      + np.cumsum(offsets[start:stop] / products))
        previous = out[stop - 1]
    return out


def two_state_kinetics(drive, dt: float, k_act: float, k_inact: float,
                       k_slow_in: float, k_slow_out: float,
                       state_step: int = 250, n_passes: int = 1,
                       relaxation: float = 1.0, return_residual: bool = False):
    """The LNK kinetics block reduced to a fast output state and a slow pool.

    Three occupancies summing to one -- resting ``R``, active ``A``,
    inactivated ``I`` -- with the activation rate driven by ``u(t)``:

        dA/dt = k_act u (1 - A - I) - k_inact A
        dI/dt = k_slow_in A - k_slow_out I

    ``A`` is the output, exactly as in ``baccuslab/LNKS`` where ``v = X[1,:]``
    is the active state. That is the point of this variant: adaptation is not
    imposed on the nonlinearity, it *emerges*, because the instantaneous gain
    is proportional to the resting occupancy ``R = 1 - A - I`` (supplement
    eq 18) and sustained drive fills ``I`` at ``R``'s expense.

    ``k_act`` has to stay free even though it looks like it should be absorbed
    by the output amplitude. It is not: the ratio ``k_act/k_inact`` sets the
    *occupancy* of the active state while their sum sets its *speed*, so
    pinning one forces a choice between fast kinetics and a state with any room
    to deplete. Fixed at 1 with ``k_inact`` free, the fit drove ``k_inact`` to
    192/s for a 5 ms response, leaving ``A`` at 0.005 and no adaptation to find.

    **Integration marches; it does not iterate.** ``A`` is fast and ``I`` is
    slow, so each ``state_step`` block solves ``A`` exactly with ``I`` held
    fixed (a first-order recurrence, hence :func:`_relax`) and then advances
    ``I`` across that block from the block's mean ``A``. Because ``I`` is
    carried forward as the march proceeds, this is unconditionally stable.

    An earlier version swept the whole record repeatedly instead, re-solving
    ``A`` against the previous sweep's ``I``. That converged only where the
    coupling was weak: at ``k_slow_in/k_slow_out`` above about 10 it
    oscillated, and since the data wants a slow recovery and a large ratio, it
    oscillated exactly where the fits wanted to go. Bounds and a penalty were
    tried to keep the optimiser out of that regime -- the wrong fix, since the
    regime was numerically bad rather than physically wrong.
    """
    drive = np.asarray(drive, dtype=float)
    step = max(int(state_step), 1)
    activation = float(k_act) * drive
    k_inact = float(k_inact)
    k_in, k_out = float(k_slow_in), max(float(k_slow_out), 1e-12)

    active = np.empty_like(drive)
    inactivated = np.empty_like(drive)
    slow = 0.0
    initial = 0.0
    decay_slow = np.exp(-k_out * dt * step)
    for start in range(0, drive.size, step):
        stop = min(start + step, drive.size)
        rate = activation[start:stop] + k_inact
        block = _relax(activation[start:stop] * (1.0 - slow)
                       / np.maximum(rate, 1e-12), rate, dt, initial)
        active[start:stop] = block
        inactivated[start:stop] = slow
        initial = block[-1]
        # Advance the slow pool across the block from its mean active value,
        # then cap it at what the active state leaves free so R stays >= 0.
        steady = k_in * float(np.mean(block)) / k_out
        slow = steady + (slow - steady) * decay_slow
        slow = float(np.clip(slow, 0.0, max(1.0 - initial, 0.0)))
    if return_residual:
        # A march has no iteration to converge, so the honest diagnostic is how
        # much the slow pool moves within one block relative to its own size --
        # large values mean the block is too coarse for these rate constants.
        moved = float(np.max(np.abs(np.diff(inactivated[::step])))) if inactivated.size > step else 0.0
        scale = float(np.max(inactivated)) or 1.0
        return active, inactivated, moved / scale
    return active, inactivated


def two_state_convergence(drive, dt: float, k_act: float, k_inact: float,
                          k_slow_in: float, k_slow_out: float,
                          state_step: int = 25,
                          max_passes: int = 6) -> List[float]:
    """Max change in the slow state per fixed-point sweep, for checking."""
    previous = None
    deltas = []
    for passes in range(1, int(max_passes) + 1):
        _, slow = two_state_kinetics(drive, dt, k_act, k_inact, k_slow_in,
                                     k_slow_out, state_step=state_step,
                                     n_passes=passes)
        if previous is not None:
            deltas.append(float(np.max(np.abs(slow - previous))))
        previous = slow
    return deltas


def adaptation_state(drive, dt: float, tau_on: float, tau_off: float,
                     a0: float = 0.0) -> np.ndarray:
    """Integrate ``a' = u(1-a)/tau_on - a/tau_off``.

    Exponential Euler, which is exact for a drive held constant across a step
    and unconditionally stable, so the step size is a speed choice rather than
    a stability one. Written as a decay toward the instantaneous steady state:
    with ``rate = u/tau_on + 1/tau_off``, ``a`` relaxes toward
    ``(u/tau_on)/rate`` with time constant ``1/rate``.

    ``a`` stays in [0, 1] for any non-negative drive, which is what makes the
    coupling strength ``k`` interpretable as a fraction.
    """
    drive = np.asarray(drive, dtype=float)
    rate_on = drive / float(tau_on)
    total = rate_on + 1.0 / float(tau_off)
    steady = rate_on / total
    decay = np.exp(-float(dt) * total)
    if drive.size == 0:
        return np.zeros(0)
    return _relax(steady, total, float(dt), float(a0))


def _lnk_predict(generator, params, coupling: str, dt: float,
                 state_step: int, adaptive: bool = True,
                 cache: Optional[dict] = None):
    """``(prediction, state)`` for one parameter set over the whole sequence.

    The drive is the *unadapted* rectified output ``Phi(beta g + gamma)``, so
    the state depends only on the stimulus and never on the measured response.
    That is what makes holding out epochs sound: the state can be integrated
    across a held-out stretch without having seen its response.

    **``cache`` stages the computation by what each part depends on**, which is
    where the fit's time goes. Measured on a 570k-sample sequence, essentially
    100% of ``fit_lnk``'s wall time is inside this function and 413 of its 481
    calls exist only to finite-difference the Jacobian. But ``drive`` depends
    on ``beta`` and ``gamma`` alone and the state on those plus the two time
    constants -- so perturbing ``alpha``, ``epsilon`` or ``k`` recomputes a
    570k-sample normal CDF that cannot have changed. Passing a dict lets
    consecutive calls reuse both: 1.50x end to end, and bit-identical, since
    nothing is approximated. Keyed on the generator *object* as well as the
    parameters, so ``fit_filter=True`` -- which rebuilds the generator every
    evaluation -- correctly misses rather than returning a stale drive.
    """
    # `ndtr` is the raw normal CDF; `scipy.stats.norm.cdf` is the same function
    # behind argument validation that costs 2.6x on a 570k array and is paid on
    # every one of a few thousand residual evaluations. Identical to 1e-15.
    from scipy.special import ndtr

    alpha = float(params['alpha']); beta = float(params['beta'])
    gamma = float(params['gamma']); epsilon = float(params['epsilon'])
    generator = np.asarray(generator, dtype=float)
    argument = drive = None
    if cache is not None and (cache.get('generator') is generator
                              and cache.get('drive_key') == (beta, gamma)):
        argument, drive = cache['argument'], cache['drive']
    if drive is None:
        argument = beta * generator + gamma
        drive = ndtr(argument)
        if cache is not None:
            cache.update(generator=generator, drive_key=(beta, gamma),
                         argument=argument, drive=drive, state_key=None)
    if not adaptive:
        return alpha * drive + epsilon, np.zeros_like(drive)

    # Integrate the state on a coarser grid: its time constants are seconds, so
    # 1 ms steps buy nothing and cost 40x the arithmetic. The drive is averaged
    # into those bins rather than sampled, since the state responds to the mean
    # drive over the step, not to whichever sample fell on the boundary.
    tau_on = float(params['tau_on']); tau_off = float(params['tau_off'])
    state = centred = None
    if cache is not None and cache.get('state_key') == (beta, gamma, tau_on,
                                                        tau_off, state_step):
        state, centred = cache['state'], cache['centred']
    if state is None:
        coarse = _bin_mean(drive[None, :], state_step).ravel()
        state_coarse = adaptation_state(coarse, dt * state_step, tau_on, tau_off)
        state = np.repeat(state_coarse, state_step)
        if state.size < drive.size:
            state = np.concatenate([state, np.full(drive.size - state.size,
                                                   state[-1] if state.size else 0.0)])
        state = state[:drive.size]

    # Couple the state's *modulation*, not its level. The mean of `a` is
    # degenerate with `alpha` (multiplicative) and with `gamma` (subtractive),
    # so leaving it in makes `k` fight an offset another parameter already
    # absorbs -- on 2020-06-11_B that pinned `k` at its bound while the state
    # swung only 0.08 between light means. Centring removes the degeneracy and
    # leaves `k` as what it is meant to be: how much a unit of adaptation moves
    # the nonlinearity.
    # Standardise, not just centre. The mean is degenerate with `alpha` /
    # `gamma`, and the *scale* is degenerate with the time constants: shrink
    # the state's swing by moving tau and `k` grows to compensate, so only the
    # product k*std(a) is identifiable. Dividing it out leaves `k` as
    # modulation per standard deviation of adaptation -- one number, comparable
    # between couplings and between cells, which is what a pathway comparison
    # needs. Without this `k` pinned at its bound even on synthetic data whose
    # true value was well inside it.
        spread = float(np.std(state))
        centred = (state - float(np.mean(state))) / (spread if spread > 1e-9 else 1.0)
        if cache is not None:
            cache.update(state_key=(beta, gamma, tau_on, tau_off, state_step),
                         state=state, centred=centred)
    k = float(params['k'])
    if coupling == 'multiplicative':
        # Log-gain, so the gain stays positive for any k and there is no
        # ceiling to pin against; exp(0) = 1 leaves alpha meaning what it did.
        return alpha * np.exp(-k * centred) * drive + epsilon, state
    if coupling == 'subtractive':
        return alpha * ndtr(argument - k * centred) + epsilon, state
    raise ValueError(f'coupling must be one of {LNK_COUPLINGS}')


PARAM_FILTER_NAMES = ('numFilt', 'tauR', 'tauD', 'tauP', 'phi')


def param_filter(params, n_points: int, dt: float) -> np.ndarray:
    """cascadegraph's five-parameter temporal filter, peak-normalised."""
    from retinanalysis.utils.cascadegraph import ParamFilterNode

    if not isinstance(params, dict):
        params = dict(zip(PARAM_FILTER_NAMES,
                          (float(v) for v in np.asarray(params).ravel())))
    return ParamFilterNode.get_filter_with_params(params, int(n_points), float(dt))


def fit_param_filter(measured, dt: float, n_starts: int = 40,
                     random_state: Optional[int] = 0) -> Tuple[np.ndarray, float]:
    """Fit the 5-parameter form to a measured filter. Returns ``(params, r2)``.

    ``ParamFilterNode`` describes a filter with five numbers instead of the
    thousand samples reverse correlation returns:

        ((t/tauR)^numFilt / (1 + (t/tauR)^numFilt)) * exp(-t/tauD)
            * cos(2 pi t/tauP + 2 pi phi/360)

    a rising saturation times an exponential decay times a cosine, which covers
    monophasic and biphasic shapes alike.

    **Starting points are scaled to the measured filter's own time-to-peak, and
    that is the whole trick.** With time constants fixed at values suited to a
    fast cell, this form fits a 32 ms filter at r2 0.99 and a 200 ms one at
    r2 ~= 0.0 -- worse than a flat line -- which reads as the form being
    incapable when it is only badly started. Scaled starts fit all four filters
    measured here at r2 0.96-0.99, beating the 8-function orthonormal basis the
    LNKS code uses with three fewer parameters.

    The parameters are partially degenerate: fits reach ``tauP`` at its bound
    (meaning "no oscillation across the window") and trade ``numFilt`` against
    ``tauR``. The *shape* is recovered, the individual values are not
    identifiable, so use this as a shape basis and never quote ``tauD`` as a
    cell's decay constant.
    """
    from scipy.optimize import least_squares

    measured = np.asarray(measured, dtype=float).ravel()
    peak = float(np.max(np.abs(measured)))
    if not np.isfinite(peak) or peak == 0:
        return np.full(len(PARAM_FILTER_NAMES), np.nan), np.nan
    target = measured / peak
    n_points = target.size
    time_to_peak = max(int(np.argmax(np.abs(target))), 1) * float(dt)

    def residual(vector):
        return param_filter(vector, n_points, dt) - target

    lower = np.array([0.1, 1e-4, 1e-4, 1e-3, -720.0])
    upper = np.array([20.0, 5.0, 5.0, 20.0, 720.0])
    rng = np.random.default_rng(random_state)
    best, best_cost = None, np.inf
    for _ in range(max(int(n_starts), 1)):
        start = np.array([rng.uniform(0.5, 6.0),
                          time_to_peak * rng.uniform(0.15, 1.5),
                          time_to_peak * rng.uniform(0.5, 6.0),
                          time_to_peak * rng.uniform(1.5, 12.0),
                          rng.uniform(0.0, 360.0)])
        try:
            candidate = least_squares(residual, np.clip(start, lower, upper),
                                      bounds=(lower, upper), max_nfev=3000)
        except Exception:
            continue
        if candidate.cost < best_cost:
            best, best_cost = candidate, candidate.cost
    if best is None:
        return np.full(len(PARAM_FILTER_NAMES), np.nan), np.nan
    spread = float(np.sum((target - target.mean()) ** 2))
    r2 = (1.0 - float(np.sum(residual(best.x) ** 2)) / spread
          if spread > 0 else np.nan)
    return best.x, r2


def _band_split(values, dt: float, split_hz: float):
    """``(low, high)`` halves of a segment either side of ``split_hz``."""
    spectrum = np.fft.rfft(values)
    freqs = np.fft.rfftfreq(values.size, d=dt)
    low = spectrum.copy(); low[freqs > split_hz] = 0
    high = spectrum - low
    return np.fft.irfft(low, values.size), np.fft.irfft(high, values.size)


def normalized_residual(predicted, measured, dt: float,
                        bin_s: float = 10.0, split_hz: Optional[float] = None):
    """Residual normalised within sections, as ``mse_weighted_loss`` does.

    Plain squared error is the wrong metric for this protocol, and the LNK
    supplement says why for its own: *"at different contrasts, the membrane
    potential shows substantial variation in its amplitude. Thus, using the
    mean squared error would bias the model towards fitting the high contrast
    segment"*. Ours is worse -- the bright epochs' stimulus fluctuates ten
    times harder, so an unweighted fit is effectively a fit to the bright
    epochs with the dim ones along for the ride.

    The default reproduces what the reference implementation actually does,
    ``mse_weighted_loss(y, v, len_section=10000, weight_type="std")``: cut the
    record into ``bin_s`` sections and divide each section's residual by the
    standard deviation of the *measured* response in it, so every section
    contributes comparably whatever its amplitude.

    ``split_hz`` additionally splits each section into two frequency bands,
    which is what the published supplement describes (eq 27) and the repo does
    not do. Kept reachable for anyone reproducing the paper rather than the
    code; the stated motivation is that slow baseline shifts should not be
    swamped by fast fluctuations.
    """
    predicted = np.asarray(predicted, dtype=float)
    measured = np.asarray(measured, dtype=float)
    per_bin = max(int(round(bin_s / dt)), 8)
    pieces = []
    for start in range(0, measured.size - per_bin + 1, per_bin):
        stop = start + per_bin
        if split_hz is None:
            bands = [(measured[start:stop], predicted[start:stop])]
        else:
            bands = list(zip(_band_split(measured[start:stop], dt, split_hz),
                             _band_split(predicted[start:stop], dt, split_hz)))
        for band_m, band_p in bands:
            sigma = float(np.std(band_m))
            pieces.append((band_p - band_m) / (sigma if sigma > 1e-12 else 1.0))
    if not pieces:
        return predicted - measured
    return np.concatenate(pieces)


@dataclass
class _LNKSetup:
    """Everything both LNK variants need before any parameter is fitted."""

    stimulus: np.ndarray
    response: np.ndarray
    epochs: np.ndarray
    generator: np.ndarray
    dt: float
    filter_pts: int
    levels: List[float]
    filters: Dict[float, np.ndarray]
    filter_r2: Dict[float, float]
    shape_params: Dict[float, np.ndarray]
    init_level: float
    build_generator: object


def _prepare_lnk(analysis, filter_mode: str, filter_length_s, random_state,
                 verbose: bool):
    """Filter estimation, parameterisation and the generator, shared by both.

    Split out so the two-state variant cannot drift from the modulated one on
    any of the choices that took several rounds to settle: one filter per light
    mean, shape-only normalisation so the generator's amplitude still carries
    the luminance step, and a single global scaling so the ratio between levels
    survives it.
    """
    from retinanalysis.utils.cascadegraph import (compute_filter,
                                                  convolve_filter_with_stim)

    stimulus = np.asarray(analysis.sequence_stimulus, dtype=float)
    response = np.asarray(analysis.sequence_response, dtype=float)
    epochs = np.asarray(analysis.sequence_epoch, dtype=int)
    if stimulus.size < 1000 or stimulus.size != response.size:
        if verbose:
            print('  no usable sequence: run analyze_condition first')
        return None

    dt = float(analysis.sampling_interval)
    filter_length_s = (analysis.filter_length_s if filter_length_s is None
                       else float(filter_length_s))
    filter_pts = int(round(filter_length_s / dt))
    cutoff = (analysis.frequency_cutoff
              if np.isfinite(analysis.frequency_cutoff) else None)
    cutoff_kwargs = ({} if cutoff is None
                     else dict(frequency_cutoff=cutoff, sampling_interval=dt))
    # --- one filter per light mean, not one for the recording ------------
    #
    # The temporal filter is genuinely different at the two light levels --
    # on 2020-06-11_B time-to-peak is 32 ms dim against 59 ms bright and the
    # biphasic index changes sign -- and the two shapes correlate only 0.71
    # with each other. A filter pooled over both is a compromise describing
    # neither, and it is not even an even-handed one: the bright condition's
    # stimulus fluctuates 10x harder, so it dominates the regression (the
    # pooled filter correlates 0.97 with the bright filter and 0.80 with the
    # dim one). What the slow state is here to explain is the adaptation
    # *within* a light level, which section 2b shows leaves the filter alone;
    # the between-level filter change is a separate, faster effect and giving
    # each level its own filter is how it stays out of the state's way.
    #
    # Each filter is normalised to unit norm, so it carries **shape only**.
    # That is load-bearing: `compute_filter` returns a gain as well, and its
    # gain is inversely proportional to the stimulus amplitude, so keeping it
    # would equalise the generator across light levels and leave the state with
    # no luminance signal to track at all. Stripped to shape, the generator's
    # amplitude follows the stimulus -- 10x larger when bright, since `stdv`
    # scales with `lightMean` -- which is exactly the "mean of a rectified
    # signal" the kinetic block is supposed to adapt to.
    def shape_filter(block_s, block_r):
        one, _ = compute_filter(block_s, block_r, filter_pts,
                                correct_stim_power=True, **cutoff_kwargs)
        one = np.asarray(one, dtype=float)
        norm_one = float(np.linalg.norm(one))
        return None if not np.isfinite(norm_one) or norm_one == 0 else one / norm_one

    per_mean: Dict[float, np.ndarray] = {}
    for mean_level in analysis.light_means:
        block_s = analysis.stimulus.get(mean_level)
        block_r = analysis.response.get(mean_level)
        if block_s is None or not block_s.size:
            continue
        one = shape_filter(block_s, block_r)
        if one is not None:
            per_mean[mean_level] = one
    if not per_mean:
        if verbose:
            print('  no filter could be estimated')
        return None

    # The LNK paper initialises the filter and nonlinearity from an LN model
    # fitted to the **high contrast period** -- one condition, not the pooled
    # record. The analogue here is the brightest mean, whose absolute
    # fluctuations are largest and whose LN fit is therefore best determined.
    init_level = max(per_mean)
    if filter_mode == 'shared':
        measured_filters = {level: per_mean[init_level] for level in per_mean}
    elif filter_mode == 'per_mean':
        measured_filters = dict(per_mean)
    else:
        raise ValueError("filter_mode must be 'shared' or 'per_mean'")

    # Reduce each measured filter to five parameters. LNKS fits its filter as
    # 8 orthonormal basis coefficients rather than freezing the reverse-
    # correlation estimate, because F_LNK is not F_LN; the same is available
    # here at five parameters per level, and the reverse-correlation fit is
    # what initialises them.
    levels = sorted(measured_filters)
    shape_params: Dict[float, np.ndarray] = {}
    filter_r2: Dict[float, float] = {}
    for level in levels:
        found, r2 = fit_param_filter(measured_filters[level], dt,
                                     random_state=random_state)
        if not np.all(np.isfinite(found)):
            if verbose:
                print(f'  lightMean {level:g}: filter could not be parameterised')
            return None
        shape_params[level] = found
        filter_r2[level] = r2
    if verbose:
        summary = ', '.join(f'{level:g}: r²={filter_r2[level]:.3f}' for level in levels)
        print(f'  filter shape fit ({len(levels)} level(s)) -- {summary}')

    light = np.asarray(analysis.sequence_light_mean, dtype=float)
    boundaries = np.r_[0, np.flatnonzero(np.diff(epochs) != 0) + 1, epochs.size]

    def build_generator(by_level: Dict[float, np.ndarray]):
        """Convolve each epoch with the filter for its own light level.

        Per epoch rather than per contiguous run, so a filter never straddles a
        luminance step and each epoch's edge effects stay its own.
        """
        out = np.zeros_like(stimulus)
        built = {level: param_filter(vector, filter_pts, dt)
                 for level, vector in by_level.items()}
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            chosen = built.get(float(light[start]))
            if chosen is None:
                chosen = next(iter(built.values()))
            out[start:stop] = convolve_filter_with_stim(
                chosen, stimulus[start:stop][None, :])[0]
        return out, built

    generator, filters = build_generator(shape_params)
    scale = float(np.std(generator))
    if not np.isfinite(scale) or scale == 0:
        if verbose:
            print('  generator has no variance; filter estimate failed')
        return None
    # One global normalisation, so the amplitude *ratio* between light levels
    # -- the thing that drives the state -- survives it.
    generator = generator / scale
    filters = {level: one / scale for level, one in filters.items()}
    filter_causal = filters[levels[0]]
    return _LNKSetup(
        stimulus=stimulus, response=response, epochs=epochs,
        generator=generator, dt=dt, filter_pts=filter_pts, levels=levels,
        filters=filters, filter_r2=filter_r2, shape_params=shape_params,
        init_level=init_level, build_generator=build_generator)


def fit_lnk(analysis: ConditionAnalysis,
            coupling: str = 'multiplicative',
            filter_mode: str = 'per_mean',
            fit_filter: bool = False,
            weighted: bool = False,
            n_restarts: int = 2,
            filter_length_s: Optional[float] = None,
            state_dt_ms: float = 25.0,
            test_fraction: float = 0.25,
            max_nfev: int = 400,
            random_state: Optional[int] = 0,
            verbose: bool = True) -> Optional[LNKModel]:
    """Fit an LN cascade plus one slow adaptive state to the whole recording.

    Fitted on ``analysis.sequence_*`` -- every accepted epoch concatenated in
    recorded order. Because ``interpulseInterval`` is 0 the epochs are
    contiguous, so this is one continuous record of the cell stepping between
    light levels, which is the data shape a kinetic model needs.

    **One filter per light mean.** LNK itself carries a single filter, but what
    its kinetics block does to the gain is a *scaling*: the supplement's eq 18
    gives the instantaneous gain as proportional to the resting-state occupancy
    ``R(t)``, which multiplies the nonlinearity's output and leaves its input
    untouched. The block does also change its own impulse response ``Fk(t)``
    with mean drive, but that depends on the **fast** rate constants -- their
    Fig. 8C-D makes the time-constant change a function of ``kfi/ka`` and
    ``kfr/ka``, and Fig. 7 measures it on that timescale.

    This reduction keeps only the slow state, because our filters are flat
    within an epoch and fast states would have been four parameters fitted to a
    flat line. A state with time constants in seconds cannot restructure a
    filter whose time-to-peak is 32 ms at one light level and 59 ms at the
    other, whichever way it couples. So the per-level filter difference is not
    something this model could generate and must be supplied. Measured on
    2020-06-11_B, held-out r2:

    ==========  ==========  ==========
    filter      unweighted  weighted
    ==========  ==========  ==========
    shared           0.338       0.153
    per mean         0.766       0.746
    ==========  ==========  ==========

    A shared filter fails here under either error metric, as it has to: it asks
    one filter to stand in for two that correlate 0.71 with each other, with
    nothing in the model able to reconcile them. ``filter_mode='shared'``
    reproduces the comparison.

    Each filter is normalised to shape alone and the generator normalised once,
    globally, so the amplitude ratio between light levels survives -- that
    ratio is what the state tracks. The nonlinearity and the state stay
    **shared** across levels: that is the claim being tested, so letting them
    vary per level would assume the answer.

    **``weighted`` reproduces the supplement's error metric** (eq 27):
    residuals normalised within 10 s time bins and two frequency bands, so a
    high-amplitude stretch cannot dominate the fit. Off by default here because
    ``r2`` is reported unweighted and mixing the two makes the number hard to
    read, and because our domination problem is milder than theirs -- the
    stimulus differs 10x between light levels but the spike rate only about 2x.

    **The filter is five parameters, and freezing them is the default.** Each
    level's reverse-correlation estimate is reduced to cascadegraph's
    ``ParamFilterNode`` form by :func:`fit_param_filter` (r2 0.987-0.993 on
    2020-06-11_B), which replaces about a thousand samples with five numbers at
    a cost of 0.005 held-out r2. ``fit_filter=True`` then puts those five per
    level into the joint search, as LNKS does with its eight basis
    coefficients, on the grounds that F_LNK is not F_LN.

    Measured, that buys nothing here. Frozen scores 0.761 held out against
    0.758 fitted, while the in-sample number moves the other way (0.654 to
    0.663) -- ten extra parameters bought training fit and lost a little
    generalisation, at five times the runtime (53 s against 281 s). One cell
    and one split, and the 0.003 margin is inside the noise, so the honest
    reading is "no measurable benefit", not "freezing is better". Default off
    on cost; turn it on to check a cell whose filter looks suspect.

    Initial values follow the supplement: the filter and nonlinearity come from
    a static LN fit at the brightest mean (their "high contrast period"), and
    the rate constants are redrawn ``n_restarts`` times, since they note the
    optimum is "non-unique and local" and take the best of several starts. On
    2020-06-11_B one start and two reach the same optimum to three decimals, so
    the LN initialisation is already landing in the right basin and the
    restarts are insurance; each extra one costs roughly the first, because a
    random start converges more slowly than the LN one.

    **Held-out epochs, and no leakage.** A fraction of whole epochs is held out
    of the residual; the state is still integrated across them, which is sound
    because the state is driven by the stimulus alone. ``r2_static`` is the
    same model with ``k`` forced to zero -- a nested baseline, so
    ``r2_gain`` is what the adaptive state buys and nothing else.

    Returns ``None`` with a printed reason when the sequence is too short or
    the fit fails, rather than raising.
    """
    from scipy.optimize import least_squares
    from retinanalysis.utils.cascadegraph import (compute_filter,
                                                  convolve_filter_with_stim)

    if coupling not in LNK_COUPLINGS:
        raise ValueError(f'coupling must be one of {LNK_COUPLINGS}')
    stimulus = np.asarray(analysis.sequence_stimulus, dtype=float)
    response = np.asarray(analysis.sequence_response, dtype=float)
    epochs = np.asarray(analysis.sequence_epoch, dtype=int)
    if stimulus.size < 1000 or stimulus.size != response.size:
        if verbose:
            print('  no usable sequence: run analyze_condition first')
        return None

    dt = float(analysis.sampling_interval)
    filter_length_s = (analysis.filter_length_s if filter_length_s is None
                       else float(filter_length_s))
    filter_pts = int(round(filter_length_s / dt))
    cutoff = (analysis.frequency_cutoff
              if np.isfinite(analysis.frequency_cutoff) else None)
    cutoff_kwargs = ({} if cutoff is None
                     else dict(frequency_cutoff=cutoff, sampling_interval=dt))
    # --- one filter per light mean, not one for the recording ------------
    #
    # The temporal filter is genuinely different at the two light levels --
    # on 2020-06-11_B time-to-peak is 32 ms dim against 59 ms bright and the
    # biphasic index changes sign -- and the two shapes correlate only 0.71
    # with each other. A filter pooled over both is a compromise describing
    # neither, and it is not even an even-handed one: the bright condition's
    # stimulus fluctuates 10x harder, so it dominates the regression (the
    # pooled filter correlates 0.97 with the bright filter and 0.80 with the
    # dim one). What the slow state is here to explain is the adaptation
    # *within* a light level, which section 2b shows leaves the filter alone;
    # the between-level filter change is a separate, faster effect and giving
    # each level its own filter is how it stays out of the state's way.
    #
    # Each filter is normalised to unit norm, so it carries **shape only**.
    # That is load-bearing: `compute_filter` returns a gain as well, and its
    # gain is inversely proportional to the stimulus amplitude, so keeping it
    # would equalise the generator across light levels and leave the state with
    # no luminance signal to track at all. Stripped to shape, the generator's
    # amplitude follows the stimulus -- 10x larger when bright, since `stdv`
    # scales with `lightMean` -- which is exactly the "mean of a rectified
    # signal" the kinetic block is supposed to adapt to.
    def shape_filter(block_s, block_r):
        one, _ = compute_filter(block_s, block_r, filter_pts,
                                correct_stim_power=True, **cutoff_kwargs)
        one = np.asarray(one, dtype=float)
        norm_one = float(np.linalg.norm(one))
        return None if not np.isfinite(norm_one) or norm_one == 0 else one / norm_one

    per_mean: Dict[float, np.ndarray] = {}
    for mean_level in analysis.light_means:
        block_s = analysis.stimulus.get(mean_level)
        block_r = analysis.response.get(mean_level)
        if block_s is None or not block_s.size:
            continue
        one = shape_filter(block_s, block_r)
        if one is not None:
            per_mean[mean_level] = one
    if not per_mean:
        if verbose:
            print('  no filter could be estimated')
        return None

    # The LNK paper initialises the filter and nonlinearity from an LN model
    # fitted to the **high contrast period** -- one condition, not the pooled
    # record. The analogue here is the brightest mean, whose absolute
    # fluctuations are largest and whose LN fit is therefore best determined.
    init_level = max(per_mean)
    if filter_mode == 'shared':
        measured_filters = {level: per_mean[init_level] for level in per_mean}
    elif filter_mode == 'per_mean':
        measured_filters = dict(per_mean)
    else:
        raise ValueError("filter_mode must be 'shared' or 'per_mean'")

    # Reduce each measured filter to five parameters. LNKS fits its filter as
    # 8 orthonormal basis coefficients rather than freezing the reverse-
    # correlation estimate, because F_LNK is not F_LN; the same is available
    # here at five parameters per level, and the reverse-correlation fit is
    # what initialises them.
    levels = sorted(measured_filters)
    shape_params: Dict[float, np.ndarray] = {}
    filter_r2: Dict[float, float] = {}
    for level in levels:
        found, r2 = fit_param_filter(measured_filters[level], dt,
                                     random_state=random_state)
        if not np.all(np.isfinite(found)):
            if verbose:
                print(f'  lightMean {level:g}: filter could not be parameterised')
            return None
        shape_params[level] = found
        filter_r2[level] = r2
    if verbose:
        summary = ', '.join(f'{level:g}: r²={filter_r2[level]:.3f}' for level in levels)
        print(f'  filter shape fit ({len(levels)} level(s)) -- {summary}')

    light = np.asarray(analysis.sequence_light_mean, dtype=float)
    boundaries = np.r_[0, np.flatnonzero(np.diff(epochs) != 0) + 1, epochs.size]

    def build_generator(by_level: Dict[float, np.ndarray]):
        """Convolve each epoch with the filter for its own light level.

        Per epoch rather than per contiguous run, so a filter never straddles a
        luminance step and each epoch's edge effects stay its own.
        """
        out = np.zeros_like(stimulus)
        built = {level: param_filter(vector, filter_pts, dt)
                 for level, vector in by_level.items()}
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            chosen = built.get(float(light[start]))
            if chosen is None:
                chosen = next(iter(built.values()))
            out[start:stop] = convolve_filter_with_stim(
                chosen, stimulus[start:stop][None, :])[0]
        return out, built

    generator, filters = build_generator(shape_params)
    scale = float(np.std(generator))
    if not np.isfinite(scale) or scale == 0:
        if verbose:
            print('  generator has no variance; filter estimate failed')
        return None
    # One global normalisation, so the amplitude *ratio* between light levels
    # -- the thing that drives the state -- survives it.
    generator = generator / scale
    filters = {level: one / scale for level, one in filters.items()}
    filter_causal = filters[levels[0]]

    state_step = max(int(round(state_dt_ms / 1e3 / dt)), 1)

    # Start the nonlinearity from an actual static LN fit, as the LNK
    # supplement does, rather than from the bounds helper's opening guess: the
    # optimiser then begins exactly where the non-adaptive model already sits,
    # and only has to find what the state adds.
    _, lower_nl, upper_nl = sigmoid_start_and_bounds(
        generator, response, rec_type=analysis.rec_type)
    init_mask = np.asarray(analysis.sequence_light_mean, dtype=float) == init_level
    static_nl = fit_sigmoid(generator[init_mask], response[init_mask],
                            rec_type=analysis.rec_type)
    guess_nl = np.array([static_nl.get(name, np.nan)
                         for name in ('alpha', 'beta', 'gamma', 'epsilon')])
    if not np.all(np.isfinite(guess_nl)):
        guess_nl, _, _ = sigmoid_start_and_bounds(generator, response,
                                                  rec_type=analysis.rec_type)
    # When `fit_filter` is on, each level's five shape parameters join the
    # search. LNKS optimises its filter for the same reason: the LNK filter is
    # not the LN filter, so freezing at the reverse-correlation estimate keeps
    # a bias that cannot be seen from inside the fit.
    filter_names = tuple(f'{name}_{level:g}' for level in levels
                         for name in PARAM_FILTER_NAMES) if fit_filter else ()
    names = ('alpha', 'beta', 'gamma', 'epsilon', 'tau_on', 'tau_off',
             'k') + filter_names
    # Same bounds for both couplings, so neither is handicapped in the
    # comparison. `k` is signed: positive suppresses the response as the cell
    # adapts, negative would be facilitation, and the data decides which.
    #
    # 15 rather than something tighter because the two couplings measure `k` in
    # different units -- log-gain per SD of adaptation against a shift in
    # nonlinearity-argument units -- and a bound that binds on one is not a
    # fair comparison. On 2020-06-11_B multiplicative settles at 0.47 and
    # subtractive at 7.3; at a limit of 5 the subtractive fit pinned and scored
    # 0.664 instead of 0.672, which would have flattered the winner.
    k_limit = 15.0
    guess = np.r_[guess_nl, 2.0, 5.0, 0.5]
    lower = np.r_[lower_nl, 0.05, 0.05, -k_limit]
    upper = np.r_[upper_nl, 60.0, 60.0, k_limit]
    if fit_filter:
        shape_lower = np.array([0.1, 1e-4, 1e-4, 1e-3, -720.0])
        shape_upper = np.array([20.0, 5.0, 5.0, 20.0, 720.0])
        guess = np.r_[guess, np.concatenate([shape_params[l] for l in levels])]
        lower = np.r_[lower, np.tile(shape_lower, len(levels))]
        upper = np.r_[upper, np.tile(shape_upper, len(levels))]

    rng = np.random.default_rng(random_state)
    unique_epochs = np.unique(epochs)
    n_test = max(int(round(test_fraction * unique_epochs.size)), 1)
    n_test = min(n_test, unique_epochs.size - 1)
    test_epochs = rng.choice(unique_epochs, size=n_test, replace=False)
    is_test = np.isin(epochs, test_epochs)
    train = ~is_test

    n_core = 7

    def unpack(vector):
        return dict(zip(names, (float(v) for v in vector)))

    def generator_for(vector):
        """Generator for one parameter vector, rebuilt only if the filter moves."""
        if not fit_filter:
            return generator
        by_level = {level: np.asarray(vector[n_core + 5 * index:
                                             n_core + 5 * (index + 1)], dtype=float)
                    for index, level in enumerate(levels)}
        rebuilt, _ = build_generator(by_level)
        spread = float(np.std(rebuilt))
        return rebuilt / (spread if spread > 1e-12 else 1.0)

    def score_residual(predicted, mask):
        if weighted:
            return normalized_residual(predicted[mask], response[mask], dt)
        return predicted[mask] - response[mask]

    # One cache for the whole fit: consecutive residual evaluations differ in
    # one parameter at a time (that is what a finite-difference Jacobian is),
    # so most of them reuse the drive, and many reuse the state as well.
    predict_cache: dict = {}

    def residual(vector):
        predicted, _ = _lnk_predict(generator_for(vector), unpack(vector),
                                    coupling, dt, state_step,
                                    cache=predict_cache)
        return score_residual(predicted, train)

    # "the optimum values obtained are assumed to be non-unique and local. To
    # address this issue, we used multiple initial points to converge to
    # different optima and then chose the best solution." The nonlinearity
    # start is the static LN fit every time; only the rate constants and the
    # coupling are redrawn, which is what the supplement randomises.
    starts = [np.clip(guess, lower, upper)]
    for _ in range(max(int(n_restarts) - 1, 0)):
        draw = guess.copy()
        draw[4] = rng.uniform(0.1, 20.0)      # tau_on
        draw[5] = rng.uniform(0.1, 20.0)      # tau_off
        draw[6] = rng.uniform(-k_limit, k_limit)
        starts.append(np.clip(draw, lower, upper))

    result = None
    for start in starts:
        try:
            candidate = least_squares(residual, start, bounds=(lower, upper),
                                      max_nfev=max_nfev)
        except Exception:
            continue
        if result is None or candidate.cost < result.cost:
            result = candidate
    if result is None:
        if verbose:
            print('  LNK fit failed from every starting point')
        return None

    params = unpack(result.x)
    generator = generator_for(result.x)
    if fit_filter:
        # Keep the filters the fit actually settled on, not the ones it started
        # from, so `model.filters` describes the model that was scored.
        filters = {level: param_filter(
            result.x[n_core + 5 * index: n_core + 5 * (index + 1)],
            filter_pts, dt) for index, level in enumerate(levels)}
        filter_causal = filters[levels[0]]
    predicted, state = _lnk_predict(generator, params, coupling, dt, state_step)

    # Nested baseline: same cascade, k = 0, nonlinearity refitted so the
    # comparison is not rigged by leaving it at the adaptive optimum.
    def residual_static(vector):
        flat = dict(zip(names[:4], (float(v) for v in vector)))
        flat.update(tau_on=1.0, tau_off=1.0, k=0.0)
        predicted_flat, _ = _lnk_predict(generator, flat, coupling, dt,
                                         state_step, adaptive=False)  # noqa
        return score_residual(predicted_flat, train)

    try:
        static = least_squares(residual_static, np.clip(guess_nl, lower_nl, upper_nl),
                               bounds=(lower_nl, upper_nl), max_nfev=max_nfev)
        static_params = dict(zip(names[:4], (float(v) for v in static.x)))
        static_params.update(tau_on=1.0, tau_off=1.0, k=0.0)
        predicted_static, _ = _lnk_predict(generator, static_params, coupling,
                                           dt, state_step, adaptive=False)
    except Exception:
        predicted_static = np.full_like(predicted, np.nan)

    span = upper - lower
    at_bounds = tuple(
        name for name, value, low, high, width
        in zip(names, result.x, lower, upper, span)
        if np.isfinite(width) and width > 0
        and (abs(value - low) <= 1e-6 * width or abs(value - high) <= 1e-6 * width))

    model = LNKModel(
        coupling=coupling, params=params,
        filter=np.asarray(filter_causal, dtype=float), filters=filters,
        filter_r2=filter_r2, fit_filter=bool(fit_filter),
        filter_params={level: (np.asarray(result.x[n_core + 5 * i:
                                                   n_core + 5 * (i + 1)], float)
                               if fit_filter else shape_params[level])
                       for i, level in enumerate(levels)},
        filter_time_s=np.arange(filter_pts) * dt,
        sampling_interval=dt, state_dt_s=state_step * dt,
        r2=_variance_explained(predicted[is_test], response[is_test]),
        r2_train=_variance_explained(predicted[train], response[train]),
        r2_static=_variance_explained(predicted_static[is_test], response[is_test]),
        n_train_epochs=int(unique_epochs.size - n_test), n_test_epochs=int(n_test),
        predicted=predicted, predicted_static=predicted_static, state=state,
        generator=generator, at_bounds=at_bounds)
    if verbose:
        print(f'  {coupling:>14}: r²={model.r2:.3f} held out '
              f'(static {model.r2_static:.3f}, gain {model.r2_gain:+.3f}) | '
              f'tau_on {params["tau_on"]:.2f} s, tau_off {params["tau_off"]:.2f} s, '
              f'k {params["k"]:+.3f}'
              + (f' | at bound: {", ".join(at_bounds)}' if at_bounds else ''))
    return model


# Fitted in terms of the two quantities that are actually identifiable.
#
# `k_act` and `k_inact` are not separately determined. Their *ratio* sets the
# active state's occupancy and their *sum* sets its speed, and once the speed
# passes the sampling interval the sum stops mattering: scaling both by 20x,
# holding the ratio, moves held-out r2 from 0.671 to 0.667 and leaves the
# occupancy at 0.0555, so the optimiser is free to drift up that flat
# direction until it hits a bound -- which is exactly what it did. Changing
# the ratio alone, by contrast, has a sharp optimum: 0.624 at 1, 0.671 at 4,
# 0.620 at 16.
#
# So fit `tau_fast` (the relaxation time at full drive, 1/(k_act + k_inact))
# and `occupancy` (k_act/k_inact), and convert. `tau_fast` is floored at the
# sampling interval because nothing below it is resolvable -- and pushing far
# past it makes the relaxation underflow and return NaN.
TWO_STATE_NAMES = ('alpha', 'beta', 'gamma', 'epsilon',
                   'tau_fast', 'occupancy', 'k_slow_in', 'k_slow_out')


def two_state_rates(tau_fast: float, occupancy: float) -> Tuple[float, float]:
    """``(k_act, k_inact)`` from the identifiable pair."""
    total = 1.0 / max(float(tau_fast), 1e-9)
    k_inact = total / (1.0 + float(occupancy))
    return total - k_inact, k_inact


def fit_lnk_two_state(analysis: ConditionAnalysis,
                      filter_mode: str = 'per_mean',
                      filter_length_s: Optional[float] = None,
                      state_dt_ms: float = 250.0,
                      n_passes: int = 8,
                      solver_tolerance: float = 0.01,
                      weighted: bool = False,
                      n_restarts: int = 3,
                      test_fraction: float = 0.25,
                      max_nfev: int = 400,
                      random_state: Optional[int] = 0,
                      verbose: bool = True) -> Optional[LNKModel]:
    """LNK with the kinetics block restored as the output stage.

    The other variant (:func:`fit_lnk`) *imposes* the adaptation: a slow state
    either rescales or shifts the nonlinearity, and the fit reports which fits
    better. This one imposes nothing. The output is the active-state occupancy
    ``A(t)`` -- as in ``baccuslab/LNKS``, where ``v = X[1,:]`` -- and any change
    in the apparent input-output curve has to *emerge* from depletion, because
    the instantaneous gain is proportional to the resting occupancy
    ``R = 1 - A - I`` (supplement eq 18) and sustained drive fills ``I``.

    So there is no coupling to choose here, and that is the point: whether the
    cell's apparent nonlinearity scales or shifts becomes a **prediction** of
    the model rather than an input to it. :func:`apparent_nonlinearity` reads it
    back off the fitted model, which is the analogue of the paper's Fig. 3A.

    Eight free parameters against the modulated variant's seven:
    ``alpha``/``epsilon`` map occupancy into response units, ``beta``/``gamma``
    shape the drive ``u = Phi(beta g + gamma)``, then ``tau_fast`` and
    ``occupancy`` for the fast state and ``k_slow_in``/``k_slow_out`` for the
    slow pool. The fast state is fitted as a time constant and a ratio rather
    than as ``k_act``/``k_inact``, because only those two combinations are
    identifiable -- see :data:`TWO_STATE_NAMES`.
    """
    from scipy.optimize import least_squares
    from scipy.special import ndtr

    setup = _prepare_lnk(analysis, filter_mode, filter_length_s, random_state,
                         verbose)
    if setup is None:
        return None
    generator, response = setup.generator, setup.response
    dt, epochs = setup.dt, setup.epochs
    state_step = max(int(round(state_dt_ms / 1e3 / dt)), 1)

    _, lower_nl, upper_nl = sigmoid_start_and_bounds(
        generator, response, rec_type=analysis.rec_type)
    init_mask = (np.asarray(analysis.sequence_light_mean, dtype=float)
                 == setup.init_level)
    static_nl = fit_sigmoid(generator[init_mask], response[init_mask],
                            rec_type=analysis.rec_type)
    guess_nl = np.array([static_nl.get(name, np.nan)
                         for name in ('alpha', 'beta', 'gamma', 'epsilon')])
    if not np.all(np.isfinite(guess_nl)):
        guess_nl, _, _ = sigmoid_start_and_bounds(generator, response,
                                                  rec_type=analysis.rec_type)
    # `alpha` here is rate (or current) *per unit occupancy*, not a response
    # range: it multiplies `A`, which peaks near 0.3 rather than 1, so the
    # physiological ceiling has to be divided by that headroom or it binds for
    # a reason that has nothing to do with physiology. Five times is generous
    # and still refuses runaway values.
    lower_nl = lower_nl.copy(); upper_nl = upper_nl.copy()
    lower_nl[0] *= 5.0; upper_nl[0] *= 5.0

    # `tau_fast` is floored at the sampling interval: faster than one sample is
    # not resolvable, and the fit that motivated this reparameterisation landed
    # at exactly 0.96 ms with dt = 1 ms. `occupancy` is the ratio.
    #
    # The slow pair covers recovery from 50 ms to 200 s. These are the
    # physiological bounds; an earlier version had to box the in/out ratio at
    # 10 to keep the fixed-point solver from oscillating, which constrained the
    # model to protect the integrator. The march does not need that.
    lower = np.r_[lower_nl, dt, 0.05, 0.005, 0.005]
    upper = np.r_[upper_nl, 1.0, 50.0, 20.0, 20.0]
    guess = np.r_[guess_nl, max(5e-3, dt), 4.0, 0.5, 0.2]

    rng = np.random.default_rng(random_state)
    unique_epochs = np.unique(epochs)
    n_test = min(max(int(round(test_fraction * unique_epochs.size)), 1),
                 unique_epochs.size - 1)
    test_epochs = rng.choice(unique_epochs, size=n_test, replace=False)
    is_test = np.isin(epochs, test_epochs)
    train = ~is_test

    def unpack(vector):
        return dict(zip(TWO_STATE_NAMES, (float(v) for v in vector)))

    def predict(vector):
        p = unpack(vector)
        drive = ndtr(p['beta'] * generator + p['gamma'])
        k_act, k_inact = two_state_rates(p['tau_fast'], p['occupancy'])
        active, inactivated, residual_solver = two_state_kinetics(
            drive, dt, k_act, k_inact, p['k_slow_in'],
            p['k_slow_out'], state_step=state_step, n_passes=n_passes,
            return_residual=True)
        return (p['alpha'] * active + p['epsilon'], active, inactivated,
                residual_solver)

    def score_residual(predicted, mask):
        if weighted:
            return normalized_residual(predicted[mask], response[mask], dt)
        return predicted[mask] - response[mask]

    def residual(vector):
        return score_residual(predict(vector)[0], train)

    starts = [np.clip(guess, lower, upper)]
    for _ in range(max(int(n_restarts) - 1, 0)):
        draw = guess.copy()
        draw[4] = rng.uniform(dt, 0.05)
        draw[5] = rng.uniform(0.5, 20.0)
        draw[6] = rng.uniform(0.02, 5.0)
        draw[7] = rng.uniform(0.02, 5.0)
        starts.append(np.clip(draw, lower, upper))

    result = None
    for start in starts:
        try:
            candidate = least_squares(residual, start, bounds=(lower, upper),
                                      max_nfev=max_nfev)
        except Exception:
            continue
        if result is None or candidate.cost < result.cost:
            result = candidate
    if result is None:
        if verbose:
            print('  two-state fit failed from every starting point')
        return None

    params = unpack(result.x)
    predicted, active, inactivated, solver_residual = predict(result.x)

    # Nested baseline: the same cascade with no slow pool at all, so the only
    # difference is whether depletion is allowed to happen.
    def residual_static(vector):
        flat = dict(zip(TWO_STATE_NAMES[:6], (float(v) for v in vector)))
        drive = ndtr(flat['beta'] * generator + flat['gamma'])
        ka, kfi = two_state_rates(flat['tau_fast'], flat['occupancy'])
        act, _ = two_state_kinetics(drive, dt, ka, kfi, 0.0, 1.0,
                                    state_step=state_step, n_passes=1)
        return score_residual(flat['alpha'] * act + flat['epsilon'], train)

    try:
        static_lo = np.r_[lower_nl, lower[4], lower[5]]
        static_hi = np.r_[upper_nl, upper[4], upper[5]]
        static = least_squares(residual_static,
                               np.clip(np.r_[guess_nl, guess[4], guess[5]],
                                       static_lo, static_hi),
                               bounds=(static_lo, static_hi), max_nfev=max_nfev)
        flat = dict(zip(TWO_STATE_NAMES[:6], (float(v) for v in static.x)))
        drive_flat = ndtr(flat['beta'] * generator + flat['gamma'])
        ka, kfi = two_state_rates(flat['tau_fast'], flat['occupancy'])
        act_flat, _ = two_state_kinetics(drive_flat, dt, ka, kfi, 0.0, 1.0,
                                         state_step=state_step, n_passes=1)
        predicted_static = flat['alpha'] * act_flat + flat['epsilon']
    except Exception:
        predicted_static = np.full_like(predicted, np.nan)

    span = upper - lower
    at_bounds = tuple(
        name for name, value, low, high, width
        in zip(TWO_STATE_NAMES, result.x, lower, upper, span)
        if np.isfinite(width) and width > 0
        and (abs(value - low) <= 1e-6 * width or abs(value - high) <= 1e-6 * width))

    model = LNKModel(
        coupling='two_state', params=params,
        filter=np.asarray(setup.filters[setup.levels[0]], dtype=float),
        filters=setup.filters, filter_r2=setup.filter_r2,
        filter_params=setup.shape_params, fit_filter=False,
        filter_time_s=np.arange(setup.filter_pts) * dt,
        sampling_interval=dt, state_dt_s=state_step * dt,
        r2=_variance_explained(predicted[is_test], response[is_test]),
        r2_train=_variance_explained(predicted[train], response[train]),
        r2_static=_variance_explained(predicted_static[is_test], response[is_test]),
        n_train_epochs=int(unique_epochs.size - n_test), n_test_epochs=int(n_test),
        predicted=predicted, predicted_static=predicted_static,
        state=inactivated, generator=generator, at_bounds=at_bounds)
    model.active = active
    model.solver_residual = solver_residual
    if verbose:
        tau_fast = params['tau_fast'] * 1e3
        print(f"  {'two-state':>14}: r²={model.r2:.3f} held out "
              f"(no-depletion {model.r2_static:.3f}, gain {model.r2_gain:+.3f}) | "
              f"tau_fast {tau_fast:.2f} ms, occupancy {params['occupancy']:.2f}, "
              f"k_in {params['k_slow_in']:.3f}, "
              f"k_out {params['k_slow_out']:.3f} /s | I {inactivated.min():.3f}"
              f"-{inactivated.max():.3f}, solver ±{solver_residual:.4f}"
              + (f' | at bound: {", ".join(at_bounds)}' if at_bounds else ''))
    return model


def apparent_nonlinearity(analysis: ConditionAnalysis, model: LNKModel,
                          n_points: int = 200, quantiles=(25, 75)) -> pd.DataFrame:
    """The input-output curve the two-state model *displays* at two adaptation levels.

    The analogue of the paper's Fig. 3A. Nothing in the two-state model was
    told to rescale or shift the nonlinearity -- its output is the active-state
    occupancy and adaptation is depletion of the resting pool -- so whichever
    it displays is a prediction.

    **Computed analytically, not by binning samples on the state.** Splitting
    the record into low- and high-state halves and binning each against the
    generator is confounded: the state is driven by the stimulus, so
    high-state samples are also high-drive samples, and the contrast picks up
    that selection on top of any gain change. Done that way this cell reported
    a gain *ratio above 1* -- more adaptation, more output -- which is the
    opposite of what depletion does and was an artefact of the split.

    Instead the model's own instantaneous map is evaluated with the slow pool
    held at a low and a high value:

        A_ss(u) = k_act u (1 - I) / (k_act u + k_inact),   r = alpha A_ss + eps

    which is the curve a fast probe would trace at that level of depletion,
    with the stimulus statistics identical on both. ``data`` is the measured
    response binned against the generator over the whole record -- one curve,
    for scale, not split by state.
    """
    params = model.params
    generator = np.asarray(model.generator, dtype=float)
    slow = np.asarray(model.state, dtype=float)
    low_value, high_value = np.percentile(slow, quantiles)
    grid = np.linspace(np.percentile(generator, 1),
                       np.percentile(generator, 99), int(n_points))

    from scipy.special import ndtr
    drive = ndtr(params['beta'] * grid + params['gamma'])
    k_act_value, k_inact_value = two_state_rates(params['tau_fast'],
                                                 params['occupancy'])
    activation = k_act_value * drive

    measured = np.asarray(analysis.sequence_response, dtype=float)
    edges = np.percentile(generator, np.linspace(1, 99, 26))
    centres = 0.5 * (edges[:-1] + edges[1:])
    which = np.digitize(generator, edges) - 1
    binned = np.array([measured[which == i].mean() if (which == i).sum() >= 20
                       else np.nan for i in range(centres.size)])

    rows = []
    for label, occupancy in (('low', low_value), ('high', high_value)):
        steady = (activation * (1.0 - occupancy)
                  / np.maximum(activation + k_inact_value, 1e-12))
        response = params['alpha'] * steady + params['epsilon']
        for value, curve in zip(grid, response):
            rows.append({'adaptation': label, 'slow_state': float(occupancy),
                         'generator': float(value), 'model': float(curve),
                         'data': float(np.interp(value, centres, binned))})
    return pd.DataFrame(rows)


def describe_apparent_change(curves: pd.DataFrame) -> Dict[str, float]:
    """Is the apparent nonlinearity change a scaling or a shift?

    Regresses the high-adaptation curve on the low one over their shared
    generator range. A pure scaling gives slope != 1 with intercept 0 at the
    curve's foot; a pure shift gives slope ~ 1 with the curve displaced along
    the generator axis, which shows up as a horizontal offset. Both are
    reported so neither has to be assumed.
    """
    out = {'gain_ratio': np.nan, 'shift_generator': np.nan, 'n_points': 0}
    if curves is None or curves.empty:
        return out
    low = curves[curves.adaptation.eq('low')].set_index('generator')
    high = curves[curves.adaptation.eq('high')].set_index('generator')
    shared = low.index.intersection(high.index)
    if shared.size < 4:
        return out
    y_low = low.loc[shared, 'model'].values
    y_high = high.loc[shared, 'model'].values
    x = np.asarray(shared, dtype=float)
    base = min(y_low.min(), y_high.min())
    denom = float(np.sum((y_low - base) ** 2))
    out['gain_ratio'] = (float(np.sum((y_high - base) * (y_low - base)) / denom)
                         if denom else np.nan)
    # Horizontal displacement: where each curve reaches its own half height.
    def half_point(y):
        target = 0.5 * (y.min() + y.max())
        order = np.argsort(y)
        return float(np.interp(target, y[order], x[order]))
    out['shift_generator'] = half_point(y_high) - half_point(y_low)
    out['n_points'] = int(shared.size)
    return out


def plot_apparent_nonlinearity(analysis: ConditionAnalysis, model: LNKModel,
                               curves: Optional[pd.DataFrame] = None,
                               figsize: Tuple[float, float] = (12.0, 4.4)):
    """What the two-state model *predicts* the nonlinearity change to be.

    The analogue of the paper's Fig. 3A. Nothing in the two-state model was
    told to rescale or shift the nonlinearity -- the output is the active-state
    occupancy and adaptation is depletion of the resting pool -- so whichever
    the fitted model displays is a prediction, and can be checked against the
    same curves measured from the data.

    Left: the model's input-output curve at low and high slow-state occupancy,
    with the data's beside it. Middle: the state and occupancies over a
    luminance step, which is where depletion is visible. Right: the two
    summary numbers, gain ratio and horizontal shift.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if model is None:
        print('no model to plot')
        return None
    if curves is None:
        curves = apparent_nonlinearity(analysis, model)
    if curves.empty:
        print('not enough samples to bin the nonlinearity')
        return None
    summary = describe_apparent_change(curves)

    fig, axes = plt.subplots(1, 3, figsize=figsize,
                             gridspec_kw={'width_ratios': [1.1, 1.5, 0.8]})
    shades = {'low': '#0072B2', 'high': '#D55E00'}
    ax = axes[0]
    for label in ('low', 'high'):
        block = curves[curves.adaptation.eq(label)].sort_values('generator')
        if block.empty:
            continue
        ax.plot(block.generator, block.model, '-', lw=2.0, color=shades[label],
                label=f'model, {label} adaptation')
        ax.plot(block.generator, block.data, 'o', ms=3.5, mfc='none',
                color=shades[label], label=f'data, {label}')
    ax.set_xlabel('generator (SD units)', fontsize=9)
    ax.set_ylabel(analysis.units, fontsize=9)
    ax.set_title('apparent nonlinearity', fontsize=9.5)
    ax.legend(frameon=False, fontsize=6.8)

    ax = axes[1]
    dt = model.sampling_interval
    light = np.asarray(analysis.sequence_light_mean, dtype=float)
    lo, hi = int(round(25.0 / dt)), int(round(95.0 / dt))
    hi = min(hi, light.size)
    time_s = np.arange(lo, hi) * dt
    active = getattr(model, 'active', None)
    inactivated = np.asarray(model.state, dtype=float)
    means = sorted(set(light[lo:hi]))
    colors = style.colors_for_conditions([f'{m:g}' for m in means])
    edges = np.r_[0, np.flatnonzero(np.diff(light[lo:hi]) != 0) + 1, hi - lo]
    for a0, a1 in zip(edges[:-1], edges[1:]):
        ax.axvspan(time_s[a0], time_s[min(a1, time_s.size - 1)],
                   color=colors[f'{light[lo + a0]:g}'], alpha=.10, lw=0)
    step = max(int(round(0.2 / dt)), 1)
    smooth_t = ((np.arange(_bin_mean(inactivated[None, lo:hi], step).size) + 0.5)
                * step * dt + lo * dt)
    if active is not None:
        resting = 1.0 - np.asarray(active) - inactivated
        ax.plot(smooth_t, _bin_mean(resting[None, lo:hi], step).ravel(),
                color='#009E73', lw=1.6, label='resting R  (gain)')
        ax.plot(smooth_t, _bin_mean(np.asarray(active)[None, lo:hi], step).ravel(),
                color='#8a6512', lw=1.4, label='active A  (output)')
    ax.plot(smooth_t, _bin_mean(inactivated[None, lo:hi], step).ravel(),
            color='#CC79A7', lw=1.6, label='inactivated I  (slow pool)')
    ax.set_xlabel('time in the concatenated recording (s)', fontsize=9)
    ax.set_ylabel('state occupancy', fontsize=9)
    ax.set_title('depletion across a luminance step (200 ms boxcar)', fontsize=9.5)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    ax.axis('off')
    ax.text(0.0, 0.92, 'predicted change', fontsize=9.5, fontweight='bold',
            transform=ax.transAxes)
    lines = [f"gain ratio      {summary['gain_ratio']:.3f}",
             f"shift (gen SD)  {summary['shift_generator']:+.3f}",
             '',
             f"held-out r2     {model.r2:.3f}",
             f"no depletion    {model.r2_static:.3f}",
             f"gain            {model.r2_gain:+.3f}"]
    ax.text(0.0, 0.80, '\n'.join(lines), fontsize=8.5, va='top',
            family='monospace', transform=ax.transAxes)
    verdict = ('a scaling' if abs(summary['gain_ratio'] - 1) > 4 * abs(
        summary['shift_generator']) else 'a shift')
    ax.text(0.0, 0.24, f'reads as {verdict}', fontsize=9, style='italic',
            transform=ax.transAxes)
    fig.suptitle(f'{analysis.exp_name} | {analysis.rec_type} | two-state LNK: '
                 f'the nonlinearity change is predicted, not imposed', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig



def timelapse_windows(epoch_s: float, n_windows: int = 5,
                      first_edge_s: float = 1.0) -> List[Tuple[float, float]]:
    """Window edges that get wider with time, for watching an adaptation.

    Geometric rather than even. The state moves fastest just after the step and
    barely at all by the end, so even windows spend most of their resolution
    where nothing is happening and blur the part that is. Five windows over a
    30 s epoch come out roughly 0-1, 1-2.3, 2.3-5.4, 5.4-12.7, 12.7-30 s.
    """
    epoch_s = float(epoch_s)
    if epoch_s <= first_edge_s or n_windows < 2:
        return [(0.0, epoch_s)]
    inner = np.geomspace(first_edge_s, epoch_s, int(n_windows))
    edges = np.r_[0.0, inner]
    return [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:])]


def nonlinearity_timelapse(analysis: ConditionAnalysis, model: LNKModel,
                           windows_s: Optional[List[Tuple[float, float]]] = None,
                           n_windows: int = 5, n_points: int = 200,
                           min_bin_samples: int = 20,
                           warmup_epochs: int = 1) -> pd.DataFrame:
    """The input-output curve the one-state LNK displays at successive times.

    The point of the model is that one slow state moves one fixed nonlinearity,
    so "how the nonlinearity changes" is not a separate fit per window -- it is
    the same four parameters evaluated at the state the cell has reached by
    that time. This walks the state forward and draws the curve it implies:

    ``multiplicative``  ``r(g) = alpha exp(-k a') Phi(beta g + gamma) + eps``
    ``subtractive``     ``r(g) = alpha Phi(beta g + gamma - k a') + eps``

    with ``a'`` the standardised state averaged over the window -- the same
    standardisation :func:`_lnk_predict` applies, so these curves are the ones
    the fitted model actually used.

    **Binned by time, which is why the empirical curve is safe here.**
    :func:`apparent_nonlinearity` has to compute its two curves analytically
    because splitting the record *on the state* is confounded: the state is
    driven by the stimulus, so high-state samples are also high-drive samples
    and the split selects on the very axis being plotted. Time since the step
    is imposed by the protocol and is not selected by the stimulus, so binning
    the measured response against the generator within a time window is an
    honest comparison rather than a circular one. That is what ``data`` is.

    **Per light mean, and on that level's own generator range.** The filters
    are normalised to shape and the generator globally, so the bright level's
    generator swings about 10x wider; one shared grid would draw the dim level
    as a dot at the origin.

    **``warmup_epochs`` drops the start of the record, and it is not optional
    in practice.** :func:`adaptation_state` integrates from ``a0 = 0``, but the
    state is standardised before it is coupled, so a zero start is not a small
    perturbation -- it is zero against a record whose mean state is far from
    zero and whose spread can be small. On a synthetic record with mean state
    0.46 and SD 0.038 the first samples sit at **-12 standardised units** and
    ``exp(-k a')`` reaches 1530 against 46 for the rest of the record. That
    lands in the earliest window, which is exactly the one this function
    exists to show, so the first epoch is dropped by default. It is an
    artefact of the initial condition, not the cell's history.

    Columns: ``lightMean``, ``window``, ``order``, ``t_mid_s``, ``state``
    (mean standardised state in that window), ``n_samples``, ``generator``,
    ``model``, ``data``.
    """
    from scipy.special import ndtr

    if model is None:
        return pd.DataFrame()
    params = model.params
    generator = np.asarray(model.generator, dtype=float)
    raw_state = np.asarray(model.state, dtype=float)
    response = np.asarray(analysis.sequence_response, dtype=float)
    light = np.asarray(analysis.sequence_light_mean, dtype=float)
    epochs = np.asarray(analysis.sequence_epoch, dtype=int)
    if generator.size == 0 or raw_state.size != generator.size:
        return pd.DataFrame()

    # Standardised exactly as the fit did: only k * a' is identifiable.
    spread = float(np.std(raw_state))
    state = (raw_state - float(np.mean(raw_state))) / (spread if spread > 1e-9 else 1.0)

    # Time since the luminance step = time within the epoch, because the mean
    # is constant within an epoch and changes between them.
    dt = float(analysis.sampling_interval)
    starts = np.r_[0, np.flatnonzero(np.diff(epochs) != 0) + 1]
    since_step = np.arange(generator.size, dtype=float)
    for start, stop in zip(starts, np.r_[starts[1:], generator.size]):
        since_step[start:stop] -= start
    since_step *= dt

    if windows_s is None:
        epoch_s = float(np.median([stop - start for start, stop
                                   in zip(starts, np.r_[starts[1:], generator.size])])) * dt
        windows_s = timelapse_windows(epoch_s, n_windows=n_windows)

    alpha = float(params['alpha']); beta = float(params['beta'])
    gamma = float(params['gamma']); epsilon = float(params['epsilon'])
    k = float(params['k'])

    # Drop the warm-up epochs before anything is measured off the state.
    keep = np.ones(generator.size, dtype=bool)
    if int(warmup_epochs) > 0:
        first = np.unique(epochs)[:int(warmup_epochs)]
        keep &= ~np.isin(epochs, first)

    rows = []
    for mean_level in analysis.light_means:
        level_mask = (light == mean_level) & keep
        if not level_mask.any():
            continue
        # This level's own generator range; the two differ by about 10x.
        lo, hi = np.percentile(generator[level_mask], [1, 99])
        grid = np.linspace(lo, hi, int(n_points))
        edges = np.percentile(generator[level_mask], np.linspace(1, 99, 26))
        centres = 0.5 * (edges[:-1] + edges[1:])
        for order, (start_s, stop_s) in enumerate(windows_s):
            mask = level_mask & (since_step >= start_s) & (since_step < stop_s)
            n = int(mask.sum())
            if n < min_bin_samples:
                continue
            a_bar = float(np.mean(state[mask]))
            if model.coupling == 'multiplicative':
                curve = alpha * np.exp(-k * a_bar) * ndtr(beta * grid + gamma) + epsilon
            elif model.coupling == 'subtractive':
                curve = alpha * ndtr(beta * grid + gamma - k * a_bar) + epsilon
            else:
                raise ValueError(f'unknown coupling {model.coupling!r}')
            # Measured response over the same samples, binned on the generator.
            which = np.digitize(generator[mask], edges) - 1
            block = response[mask]
            binned = np.array([block[which == i].mean()
                               if (which == i).sum() >= min_bin_samples else np.nan
                               for i in range(centres.size)])
            label = f'{start_s:.1f}-{stop_s:.1f} s'
            for value, curve_value in zip(grid, curve):
                rows.append({'lightMean': mean_level, 'window': label,
                             'order': order,
                             't_mid_s': 0.5 * (start_s + stop_s),
                             'state': a_bar, 'n_samples': n,
                             'generator': float(value),
                             'model': float(curve_value),
                             'data': float(np.interp(value, centres, binned))})
    return pd.DataFrame(rows)


def describe_timelapse(curves: pd.DataFrame) -> pd.DataFrame:
    """Is the nonlinearity's change over time a scaling or a shift?

    Each window's curve is compared with the **first** window's, per light
    mean, by the same two measures :func:`describe_apparent_change` uses on the
    two-state model: a through-foot slope (a pure scaling moves this away from
    1) and the displacement of the half-height point along the generator axis
    (a pure shift moves this away from 0).

    For the one-state model the answer is not in doubt -- the coupling was
    imposed, so ``multiplicative`` scales and ``subtractive`` shifts by
    construction -- and that is the use of this table: it puts the *size* of
    the imposed change on the same two axes the two-state model's emergent
    prediction is reported on, so the two variants can be read side by side.
    """
    if curves is None or curves.empty:
        return pd.DataFrame()
    rows = []
    for mean_level, block in curves.groupby('lightMean', sort=True):
        first = block[block.order.eq(block.order.min())]
        x = first.generator.values
        y_first = first.model.values
        base = float(np.min(y_first))
        for order, window in block.groupby('order', sort=True):
            y = window.model.values
            if y.size != y_first.size:
                continue
            denom = float(np.sum((y_first - base) ** 2))

            def half_point(values):
                target = 0.5 * (values.min() + values.max())
                order_by = np.argsort(values)
                return float(np.interp(target, values[order_by], x[order_by]))

            rows.append({
                'lightMean': mean_level, 'order': int(order),
                'window': window.window.iloc[0],
                't_mid_s': float(window.t_mid_s.iloc[0]),
                'state': float(window.state.iloc[0]),
                'gain_ratio': (float(np.sum((y - base) * (y_first - base)) / denom)
                               if denom else np.nan),
                'shift_generator': half_point(y) - half_point(y_first),
                'max_rate': float(np.max(y))})
    return pd.DataFrame(rows)


def plot_nonlinearity_timelapse(analysis: ConditionAnalysis, model: LNKModel,
                                curves: Optional[pd.DataFrame] = None,
                                temporal_models: Optional[Dict] = None,
                                windows_s=None, n_windows: int = 5,
                                warmup_epochs: int = 1,
                                figsize: Optional[Tuple[float, float]] = None):
    """The state, the nonlinearity it implies, and the filter, over one epoch.

    Three columns per light mean, left to right: **the state** running from the
    luminance step with the window edges marked, so the sampling of the
    adaptation is visible rather than assumed; **the nonlinearity** at each of
    those windows, model as a line and the measured response binned on the
    generator as dots; and **the temporal filter** over the same windows, taken
    from the windowed LN fits of §2b when they are passed in.

    **The third column is the model's central assumption, drawn.** This
    reduction keeps only a slow state, which cannot restructure a filter -- so
    the filter is frozen per light mean and every change within a level is
    charged to the nonlinearity. If the windowed filters in the right-hand
    column move over the epoch, that assumption is wrong for this cell and the
    nonlinearity change in the middle column is partly absorbing a filter
    change. Passing ``temporal_models`` from :func:`temporal_ln_model` is
    therefore worth the extra fit; without it the column shows only the frozen
    filter the LNK used.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    from retinanalysis.utils import style

    style.apply_publication_style()
    if model is None:
        print('no model to plot')
        return None
    if curves is None:
        curves = nonlinearity_timelapse(analysis, model, windows_s=windows_s,
                                        n_windows=n_windows,
                                        warmup_epochs=warmup_epochs)
    if curves is None or curves.empty:
        print('nothing to plot')
        return None
    means = [m for m in analysis.light_means if m in set(curves.lightMean)]
    if not means:
        print('nothing to plot')
        return None

    orders = sorted(curves.order.unique())
    shades = cm.get_cmap('viridis')(np.linspace(0.08, 0.92, len(orders)))
    colors = dict(zip(orders, shades))

    dt = float(analysis.sampling_interval)
    raw_state = np.asarray(model.state, dtype=float)
    spread = float(np.std(raw_state))
    state = (raw_state - float(np.mean(raw_state))) / (spread if spread > 1e-9 else 1.0)
    light = np.asarray(analysis.sequence_light_mean, dtype=float)
    epochs = np.asarray(analysis.sequence_epoch, dtype=int)
    starts = np.r_[0, np.flatnonzero(np.diff(epochs) != 0) + 1]
    stops = np.r_[starts[1:], epochs.size]

    if figsize is None:
        figsize = (13.5, 3.6 * len(means) + 0.8)
    fig, axes = plt.subplots(len(means), 3, figsize=figsize, squeeze=False)

    for row, mean_level in enumerate(means):
        block = curves[curves.lightMean.eq(mean_level)]

        # --- the state, averaged over the epochs of this light mean --------
        ax = axes[row][0]
        warm = set(np.unique(epochs)[:int(warmup_epochs)]) if warmup_epochs > 0 else set()
        cuts = [state[start:stop] for start, stop in zip(starts, stops)
                if light[start] == mean_level and epochs[start] not in warm]
        if cuts:
            width = min(piece.size for piece in cuts)
            stack = np.vstack([piece[:width] for piece in cuts])
            t = np.arange(width) * dt
            ax.plot(t, stack.mean(axis=0), color='0.25', lw=1.6)
            ax.fill_between(t, stack.mean(axis=0) - stack.std(axis=0),
                            stack.mean(axis=0) + stack.std(axis=0),
                            color='0.5', alpha=.2, lw=0)
        for order in orders:
            piece = block[block.order.eq(order)]
            if piece.empty:
                continue
            ax.axvline(float(piece.t_mid_s.iloc[0]), color=colors[order],
                       lw=1.4, alpha=.85)
        ax.set_ylabel(f'lightMean {mean_level:g}\nstate a′ (SD)', fontsize=9)
        if row == 0:
            ax.set_title('adaptation state', fontsize=10)
        if row == len(means) - 1:
            ax.set_xlabel('time since the step (s)', fontsize=9)

        # --- the nonlinearity at each window -------------------------------
        ax = axes[row][1]
        for order in orders:
            piece = block[block.order.eq(order)].sort_values('generator')
            if piece.empty:
                continue
            ax.plot(piece.generator, piece['model'], '-', lw=1.9,
                    color=colors[order], label=piece.window.iloc[0])
            thinned = piece.iloc[::12]
            ax.plot(thinned.generator, thinned['data'], 'o', ms=3.0,
                    color=colors[order], alpha=.75, mew=0)
        ax.legend(frameon=False, fontsize=7.5, title='time since step',
                  title_fontsize=7.5, loc='upper left')
        ax.set_ylabel('response', fontsize=9)
        if row == 0:
            ax.set_title(f'nonlinearity ({model.coupling}; '
                         f'lines model, dots measured)', fontsize=10)
        if row == len(means) - 1:
            ax.set_xlabel('generator (SD)', fontsize=9)

        # --- the filter over the same windows ------------------------------
        ax = axes[row][2]
        frozen = model.filters.get(mean_level)
        drawn = 0
        if temporal_models:
            windowed = temporal_models.get(mean_level) or []
            for index, one in enumerate(windowed):
                shade = cm.get_cmap('viridis')(
                    0.08 + 0.84 * index / max(len(windowed) - 1, 1))
                vector = np.asarray(one.filter, dtype=float)
                norm = float(np.linalg.norm(vector))
                ax.plot(np.asarray(one.filter_time_s) * 1e3,
                        vector / (norm if norm else 1.0),
                        '-', lw=1.3, color=shade, alpha=.9,
                        label=one.label if index in (0, len(windowed) - 1) else None)
                drawn += 1
        if frozen is not None and np.size(frozen):
            vector = np.asarray(frozen, dtype=float)
            norm = float(np.linalg.norm(vector))
            ax.plot(np.asarray(model.filter_time_s) * 1e3,
                    vector / (norm if norm else 1.0), '--', lw=2.0,
                    color='#D55E00', label='LNK (frozen)')
        ax.axhline(0, color='0.6', lw=0.8)
        ax.legend(frameon=False, fontsize=7.5, loc='lower right')
        ax.set_ylabel('filter (unit norm)', fontsize=9)
        if row == 0:
            ax.set_title('temporal filter'
                         + ('' if drawn else ' — pass temporal_models for windows'),
                         fontsize=10)
        if row == len(means) - 1:
            ax.set_xlabel('time (ms)', fontsize=9)

    fig.suptitle(f'{analysis.exp_name} | {analysis.rec_type} | one-state LNK '
                 f'({model.coupling}) — the nonlinearity over time, and whether '
                 f'the filter stayed put', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


def compare_lnk_couplings(analysis: ConditionAnalysis, verbose: bool = True,
                          **kwargs) -> Dict[str, Optional[LNKModel]]:
    """Fit both couplings on the same data and say which the cell prefers.

    The comparison is the experiment: a slope change and a shift are different
    mechanisms, not two settings of one, so the held-out difference between
    them is the pathway result rather than a diagnostic.
    """
    models = {coupling: fit_lnk(analysis, coupling=coupling, verbose=verbose,
                                **kwargs)
              for coupling in LNK_COUPLINGS}
    fitted = {name: m for name, m in models.items() if m is not None}
    if verbose and len(fitted) == 2:
        best = max(fitted, key=lambda n: fitted[n].r2)
        margin = abs(fitted['multiplicative'].r2 - fitted['subtractive'].r2)
        print(f'  -> prefers {best} by {margin:.3f} held-out r²'
              + ('  (margin is small; treat as undecided)' if margin < 0.01 else ''))
    return models


def lnk_summary(models: Dict[str, Optional[LNKModel]]) -> pd.DataFrame:
    """One row per coupling: fit quality, time constants, coupling strength."""
    rows = []
    for coupling, model in models.items():
        if model is None:
            continue
        rows.append({'coupling': coupling, 'r2_heldout': model.r2,
                     'r2_static': model.r2_static, 'r2_gain': model.r2_gain,
                     'r2_train': model.r2_train,
                     'tau_on_s': model.params.get('tau_on', np.nan),
                     'tau_off_s': model.params.get('tau_off', np.nan),
                     'k': model.params.get('k', np.nan),
                     'alpha': model.params.get('alpha', np.nan),
                     'beta': model.params.get('beta', np.nan),
                     'gamma': model.params.get('gamma', np.nan),
                     'n_test_epochs': model.n_test_epochs,
                     'at_bounds': ','.join(model.at_bounds)})
    return pd.DataFrame(rows)


def plot_lnk_fit(analysis: ConditionAnalysis,
                 models: Dict[str, Optional[LNKModel]],
                 seconds: Tuple[float, float] = (25.0, 95.0),
                 smooth_ms: float = 200.0,
                 figsize: Tuple[float, float] = (13.0, 9.0)):
    """What the slow state buys, and which coupling the cell prefers.

    Four panels. The top two share a time axis spanning at least one luminance
    step, since that is where an adaptive model and a static one part company:
    the measured response with both predictions over it, and the state ``a(t)``
    underneath with the light level shaded. The bottom row is the mechanism and
    the score -- the fitted nonlinearity drawn at a low and a high value of the
    state, which is where a slope change and a shift look different, and the
    held-out r2 of each coupling against the nested static baseline.

    The static bar is the number to judge the others against, **not** the
    per-window LN r2 from earlier sections: those fit a separate filter and
    nonlinearity per light mean per window, while this fits one of each across
    the whole 570 s. The comparison that means something is like against like.

    The top panel is smoothed with a ``smooth_ms`` boxcar, and says so on the
    axis. Seventy seconds at 1 kHz is 70,000 points in a few hundred pixels,
    where every trace becomes a solid block; what that panel is for is the slow
    divergence between an adaptive prediction and a static one across the step,
    which survives smoothing. The r2 values quoted are always from the
    unsmoothed fit.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style
    from scipy.stats import norm

    style.apply_publication_style()
    fitted = {name: m for name, m in models.items() if m is not None}
    if not fitted:
        print('no LNK fit to plot')
        return None
    reference = fitted.get('multiplicative') or next(iter(fitted.values()))
    dt = reference.sampling_interval
    response = np.asarray(analysis.sequence_response, dtype=float)
    light = np.asarray(analysis.sequence_light_mean, dtype=float)
    lo = max(int(round(seconds[0] / dt)), 0)
    hi = min(int(round(seconds[1] / dt)), response.size)
    if hi - lo < 10:
        print('requested stretch is outside the recording')
        return None
    time_s = np.arange(lo, hi) * dt
    means = sorted(set(light[lo:hi]))
    colors = style.colors_for_conditions([f'{m:g}' for m in means])

    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.5, 0.85, 1.25], hspace=.42,
                            wspace=.26)
    ax_trace = fig.add_subplot(grid[0, :])
    ax_state = fig.add_subplot(grid[1, :], sharex=ax_trace)
    ax_nl = fig.add_subplot(grid[2, 0])
    ax_bar = fig.add_subplot(grid[2, 1])

    # --- light-level shading, shared by the top two panels -----------------
    for ax in (ax_trace, ax_state):
        edges = np.r_[0, np.flatnonzero(np.diff(light[lo:hi]) != 0) + 1, hi - lo]
        for a0, a1 in zip(edges[:-1], edges[1:]):
            level = light[lo + a0]
            ax.axvspan(time_s[a0], time_s[min(a1, time_s.size - 1)],
                       color=colors[f'{level:g}'], alpha=.10, lw=0)

    step = max(int(round(smooth_ms / 1e3 / dt)), 1)

    def smooth(values):
        return _bin_mean(np.asarray(values)[None, lo:hi], step).ravel()

    smooth_t = ((np.arange(smooth(response).size) + 0.5) * step * dt
                + lo * dt)
    palette = {'multiplicative': '#D55E00', 'subtractive': '#0072B2',
               'two_state': '#009E73'}
    ax_trace.plot(smooth_t, smooth(response), color='0.3', lw=1.3,
                  label='response')
    ax_trace.plot(smooth_t, smooth(reference.predicted_static), color='#8c8c8c',
                  lw=1.3, ls=':', label=f'static LN (r²={reference.r2_static:.3f})')
    for name, model in fitted.items():
        ax_trace.plot(smooth_t, smooth(model.predicted), color=palette[name],
                      lw=1.3, label=f'{name} (r²={model.r2:.3f})')
    ax_trace.set_ylabel(analysis.units, fontsize=9)
    ax_trace.legend(frameon=False, fontsize=7.5, ncol=4, loc='upper right')
    ax_trace.set_title(f'response and prediction across a luminance step '
                       f'({smooth_ms:g} ms boxcar; r² is from the unsmoothed '
                       f'fit)', fontsize=10)

    for name, model in fitted.items():
        ax_state.plot(time_s, model.state[lo:hi], color=palette[name], lw=1.4,
                      label=f'{name}  τ_on {model.params["tau_on"]:.1f} s, '
                            f'τ_off {model.params["tau_off"]:.1f} s')
    ax_state.set_ylabel('adaptive state a(t)', fontsize=9)
    ax_state.set_xlabel('time in the concatenated recording (s)', fontsize=9)
    ax_state.legend(frameon=False, fontsize=7.5, loc='upper right')

    # --- the mechanism: nonlinearity at a low and a high state -------------
    grid_g = np.linspace(-3, 3, 200)
    for name, model in fitted.items():
        centred = model.state - float(np.mean(model.state))
        low, high = np.percentile(centred, [10, 90])
        p = model.params
        argument = p['beta'] * grid_g + p['gamma']
        for value, ls, tag in ((low, '-', 'adapted low'), (high, '--', 'adapted high')):
            if name == 'multiplicative':
                curve = p['alpha'] * np.exp(-p['k'] * value) * norm.cdf(argument) + p['epsilon']
            else:
                curve = p['alpha'] * norm.cdf(argument - p['k'] * value) + p['epsilon']
            ax_nl.plot(grid_g, curve, ls=ls, lw=1.6, color=palette[name],
                       label=f'{name}, {tag}')
    ax_nl.set_xlabel('generator (SD units)', fontsize=9)
    ax_nl.set_ylabel(analysis.units, fontsize=9)
    ax_nl.set_title('nonlinearity at low vs high adaptation', fontsize=9.5)
    ax_nl.legend(frameon=False, fontsize=6.8)

    # --- the score ---------------------------------------------------------
    labels = ['static LN'] + list(fitted)
    values = [reference.r2_static] + [fitted[n].r2 for n in fitted]
    bar_colors = ['#8c8c8c'] + [palette[n] for n in fitted]
    ax_bar.bar(range(len(values)), values, color=bar_colors, width=.62)
    for index, value in enumerate(values):
        ax_bar.text(index, value, f'{value:.3f}', ha='center', va='bottom',
                    fontsize=8)
    ax_bar.set_xticks(range(len(labels)))
    ax_bar.set_xticklabels(labels, fontsize=8)
    ax_bar.set_ylabel('held-out r²', fontsize=9)
    ax_bar.set_title('same filter and nonlinearity throughout;\n'
                     'only the state differs', fontsize=9.5)
    ax_bar.set_ylim(0, max(values) * 1.22)

    fig.suptitle(f'{analysis.exp_name} | {analysis.rec_type} | LN cascade with '
                 f'one slow adaptive state', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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
