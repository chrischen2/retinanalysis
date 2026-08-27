"""Linear-equivalent disc / annulus with cone linearization: NLI analysis.

Python port of ``analyzeLinearDiscCone.m`` and ``populationLinConeDisc.m``
(linCone repo), reading from DataJoint + ``SCResponseBlock`` instead of
riekesuite. Shares the light-level and grouping helpers with
:mod:`spot_annular_grating` so the single-cell analyses stay consistent.

**The experiment.** Each natural-image patch is shown three ways, interleaved:

``image``
    the image patch itself;
``intensity``
    a uniform disc at the linear-equivalent intensity — the patch averaged over
    the receptive field with a Gaussian weight;
``linConeIntensity`` / ``lin cone intensity``
    a uniform disc at the *cone-linearized* equivalent intensity, which averages
    the patch after passing it through a Weber cone nonlinearity
    ``I / (I + WeberConstant)``.

Comparing image against each disc asks how much of the cell's preference for the
real image survives when the averaging is done in cone-response space rather
than in intensity space. The measure is the nonlinearity index, per patch::

    NLI = (image - disc) / (|image| + |disc|)

zeroed when neither response clears a recording-mode threshold, exactly as
``computeNLI`` does. Extracellular responses are firing rates (Hz) and
whole-cell responses are mean smoothed current (pA); the MATLAB's per-window
thresholds -- 3 spikes, 10 pA*s, 5 pA*s -- are converted to those units by
:func:`working_thresholds`, so the same patches stay above threshold.

**Three protocols feed this analysis**, and one of them needs filtering:

===========================  ===================================================
``LinearEquivalentDiscConeLin``  disc over the centre; always has ``linearizeCones``
``LinearEquivalentAnnulus``      disc over the surround (ON parasols); always has it
``LinearEquivalentDisc``         **only** blocks that carry a ``linearizeCones``
                                 parameter — the same protocol name was reused for
                                 an older experiment without cone linearization,
                                 and those blocks are not this experiment
===========================  ===================================================

In the current database that filter keeps 58 of 421 ``LinearEquivalentDisc``
blocks; :func:`find_blocks` applies it and reports what it dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from retinanalysis.SCutils.protocols import spot_annular_grating as sag
from retinanalysis.SCutils.protocols.spot_annular_grating import (  # noqa: F401
    apply_rstar_mapping, is_calibrated, light_level_rstar, light_setting,
    read_filter_wheel_ndf,
)
from retinanalysis.SCutils.recording_mode import (  # noqa: F401
    MAX_SERIES_RESISTANCE, check_series_resistance, mode_family,
    read_series_resistance, read_stage_ndfs, resolve_recording_mode,
    series_resistance_table, stage_ndf_table, trace_is_spiking,
)

# Protocol leaf names that feed this analysis. LinearEquivalentDisc is only
# included per-block, when the block carries a linearizeCones parameter.
PROTOCOLS = ('LinearEquivalentDiscConeLin', 'LinearEquivalentAnnulus',
             'LinearEquivalentDisc')
NEEDS_LINEARIZE_FILTER = ('LinearEquivalentDisc',)

# Keep the discovery view compact. The returned DataFrame still carries every
# field needed downstream; these are only the columns shown by find_blocks().
FIND_BLOCKS_DISPLAY_COLUMNS = (
    'exp_name', 'cell_label', 'cell_type_short', 'site',
    'filter_wheel_ndf', 'maxIntensity', 'n_epochs',
)

# stimulusTag spellings: DiscConeLin writes camelCase, the other two use spaces.
TAG_IMAGE = 'image'
TAG_DISC = 'intensity'
TAG_CONE_DISC = ('linConeIntensity', 'lin cone intensity')

# nliThreshold() in the MATLAB: below this the responses are noise and the index
# is meaningless, so it is set to zero rather than dividing tiny by tiny.
#
# These are the MATLAB's values and they are *per-window integrated* quantities:
# spike counts within a window for extracellular, pA*s for whole-cell (it
# multiplied the mean current by the stimulus duration). Responses here are
# rates and mean currents instead, so working_thresholds() divides by the
# relevant window duration to keep the same patches above threshold.
NLI_THRESHOLD = {'extracellular': 3.0, 'exc': 10.0, 'inh': 5.0}

DEFAULTS = dict(
    psth_sigma_ms=10.0,
    wc_offset=100,        # samples, whole-cell response window offset
    spike_offset=300,     # samples, spiking response window offset
    smooth_ms=10.0,
)

CONFIG_KEYS = ['apertureDiameter', 'annulusInnerDiameter', 'annulusOuterDiameter',
               'backgroundIntensity', 'NDF', 'onlineAnalysis', 'linearizeCones',
               'WeberConstant', 'maxIntensity', 'rfSigmaCenter', 'rfSigmaSurround',
               'linearIntegrationFunction', 'currentImageSet', 'noPatches',
               'imageName', 'preTime', 'stimTime', 'tailTime', 'sampleRate',
               'micronsPerPixel']

GROUP_DISPLAY_COLUMNS = (
    'exp_name', 'cell_label', 'cell_type_short', 'onlineAnalysis', 'site',
    'light_settings', 'block_ids', 'image_names', 'protocols', 'maxIntensity',
)

CONDITION_SUMMARY_COLUMNS = (
    'imageName', 'block_ids', 'epochs', 'maxIntensity',
    'backgroundIntensity', 'meanIntensity',
)
CONDITION_OUTPUT_VERSION = 1
PATCH_VARIANCE_POPULATION_VERSION = 4
NATURAL_IMAGE_PATCH_SIZE_PIXELS = 38
NATURAL_IMAGE_PATCH_PIXEL_COUNT = NATURAL_IMAGE_PATCH_SIZE_PIXELS ** 2


def stimulus_site(protocol: str) -> str:
    """'surround' for the annulus protocol, 'centre' for the disc protocols."""
    return 'surround' if 'Annulus' in protocol else 'center'


def category_of(tag) -> str:
    """Map a stimulusTag to 'image' / 'disc' / 'cone_disc', or '' if unknown."""
    t = str(tag).strip()
    if t == TAG_IMAGE:
        return 'image'
    if t == TAG_DISC:
        return 'disc'
    if t in TAG_CONE_DISC or t.replace(' ', '').lower() == 'linconeintensity':
        return 'cone_disc'
    return ''


def working_thresholds(mode: str, stim_s: float, offset_s: float,
                       spiking: bool) -> Tuple[float, float]:
    """MATLAB per-window thresholds converted to this module's units.

    Returns ``(onset, offset)``. Extracellular responses are rates, so a count
    threshold becomes ``count / window duration`` -- and the onset and offset
    windows differ in length, so they get different rate thresholds. Whole-cell
    responses are mean currents in pA while the MATLAB integrated to pA*s using
    the stimulus duration for *both* windows, so both divide by ``stim_s``.
    """
    thresh = NLI_THRESHOLD.get(mode, NLI_THRESHOLD['extracellular'])
    if spiking:
        on = thresh / stim_s if stim_s > 0 else thresh
        off = thresh / offset_s if offset_s > 0 else on
        return on, off
    scale = thresh / stim_s if stim_s > 0 else thresh
    return scale, scale


def compute_nli(image_mean, disc_mean, threshold: float) -> np.ndarray:
    """Port of ``computeNLI``: (image - disc) / (|image| + |disc|).

    Patches where neither response reaches ``threshold`` are set to zero (the
    index is meaningless for noise), and non-finite values are dropped.
    """
    img = np.asarray(image_mean, dtype=float)
    disc = np.asarray(disc_mean, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        nli = (img - disc) / (np.abs(img) + np.abs(disc))
    nli = np.where(np.maximum(np.abs(img), np.abs(disc)) < threshold, 0.0, nli)
    return nli[np.isfinite(nli)]


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

def _protocol_block_rows(protocols: Sequence[str],
                         exp_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Fetch block, experiment, and cell metadata without per-block queries."""
    from retinanalysis.config import schema

    protocols = tuple(protocols)
    unknown = sorted(set(protocols) - set(PROTOCOLS))
    if unknown:
        raise ValueError(f'Unsupported protocol(s): {unknown}. Choose from {list(PROTOCOLS)}.')

    protocol_df = schema.Protocol().to_pandas().reset_index()[['protocol_id', 'name']]
    protocol_df['protocol'] = protocol_df['name'].str.rsplit('.', n=1).str[-1]
    protocol_df = protocol_df[protocol_df['protocol'].isin(protocols)]
    if protocol_df.empty:
        return pd.DataFrame()

    block_query = schema.EpochBlock() & [
        f'protocol_id={int(protocol_id)}' for protocol_id in protocol_df['protocol_id']]
    blocks = block_query.to_pandas().reset_index()
    if blocks.empty:
        return pd.DataFrame()
    blocks = blocks.rename(columns={'id': 'block_id', 'parent_id': 'group_id',
                                    'label': 'block_label'})

    experiments = (schema.Experiment() & [
        f'id={int(exp_id)}' for exp_id in blocks['experiment_id'].unique()
    ]).to_pandas().reset_index()[['id', 'exp_name', 'is_mea']]
    experiments = experiments.rename(columns={'id': 'experiment_id'})
    experiments = experiments[~experiments['is_mea'].astype(bool)]
    if exp_names is not None:
        experiments = experiments[experiments['exp_name'].isin(exp_names)]

    groups = (schema.EpochGroup() & [
        f'id={int(group_id)}' for group_id in blocks['group_id'].unique()
    ]).to_pandas().reset_index()[['id', 'parent_id', 'label', 'properties']]
    groups = groups.rename(columns={'id': 'group_id', 'parent_id': 'cell_id',
                                    'label': 'epoch_group',
                                    'properties': 'group_properties'})
    cells = (schema.Cell() & [
        f'id={int(cell_id)}' for cell_id in groups['cell_id'].unique()
    ]).to_pandas().reset_index()[['id', 'label', 'properties']]
    cells = cells.rename(columns={'id': 'cell_id', 'label': 'cell_label',
                                  'properties': 'cell_properties'})

    df = (blocks.merge(experiments, on='experiment_id')
                .merge(groups, on='group_id')
                .merge(cells, on='cell_id')
                .merge(protocol_df[['protocol_id', 'protocol']], on='protocol_id'))
    df['cell_type'] = df['cell_properties'].apply(
        lambda value: value.get('type', 'Unknown') if isinstance(value, dict) else 'Unknown')
    return df


def _linearized_only(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Drop old same-named disc blocks using block-level parameters."""
    needs_filter = df['protocol'].isin(NEEDS_LINEARIZE_FILTER)
    has_parameter = df['parameters'].apply(
        lambda value: isinstance(value, dict) and 'linearizeCones' in value)
    keep = ~needs_filter | has_parameter
    return df.loc[keep].copy(), int((~keep).sum())


def _first_epoch_metadata(block_ids: Sequence[int]) -> Tuple[Dict[int, dict], pd.Series]:
    """First-epoch parameters and epoch counts, fetched in two batched queries."""
    import datajoint as dj
    from retinanalysis.config import schema

    ids = [int(block_id) for block_id in block_ids]
    if not ids:
        return {}, pd.Series(dtype=int)
    epochs = schema.Epoch() & [{'parent_id': block_id} for block_id in ids]
    summary = (dj.U('parent_id')
               .aggr(epochs, first_epoch_id='min(id)', n_epochs='count(*)')
               .to_pandas().reset_index())
    if summary.empty:
        return {}, pd.Series(0, index=ids, dtype=int)

    first_epochs = (schema.Epoch() & [
        {'id': int(epoch_id)} for epoch_id in summary['first_epoch_id']
    ]).to_pandas().reset_index()[['parent_id', 'parameters']]
    parameters = {int(row.parent_id): row.parameters
                  for row in first_epochs.itertuples()}
    counts = summary.set_index('parent_id')['n_epochs'].astype(int)
    counts.index = counts.index.astype(int)
    return parameters, counts


def find_protocol_cells(protocol: Union[str, Sequence[str]], show: bool = True,
                        height: int = 420) -> pd.DataFrame:
    """Unique experiment dates and cells that ran the requested protocols.

    This is the fast discovery view used in Section 1 of the cone/disc notebooks.
    For ``LinearEquivalentDisc``, older blocks without ``linearizeCones`` are
    excluded using the block's own parameter dictionary.
    """
    from retinanalysis.SCutils import explore as sc

    protocols = (protocol,) if isinstance(protocol, str) else tuple(protocol)
    label = ', '.join(protocols)
    blocks = _protocol_block_rows(protocols)
    if blocks.empty:
        cells = pd.DataFrame(columns=[
            'exp_name', 'cell_label', 'cell_type_short', 'protocol'])
        if show:
            print(f'No single-cell blocks found for {label}.')
        return cells
    blocks, dropped = _linearized_only(blocks)
    if 'cell_type' in blocks:
        blocks['cell_type_short'] = (
            blocks['cell_type'].fillna('Unknown').astype(str).str.split('\\').str[-1])
    else:
        blocks['cell_type_short'] = 'Unknown'
    cells = (blocks[[
        'exp_name', 'cell_label', 'cell_type_short', 'protocol']].drop_duplicates()
             .sort_values(['exp_name', 'cell_label', 'protocol']).reset_index(drop=True))
    if show:
        print(f'{label}: {len(cells)} cells across {cells.exp_name.nunique()} experiments')
        if dropped:
            print(f'  excluded {dropped} older block(s) without linearizeCones')
        sc.scroll_table(cells, height=height)
    return cells


def protocol_cells_from_blocks(blocks: pd.DataFrame, show: bool = True,
                               height: int = 420) -> pd.DataFrame:
    """Compact experiment/cell overview from an existing resolved block table.

    Use this when detailed blocks are already needed downstream, avoiding a
    second database discovery query just to build the Section 1 overview. One
    row represents an experiment/cell/protocol, with recording modes and
    FilterWheel settings consolidated across its blocks.
    """
    from retinanalysis.SCutils import explore as sc

    columns = ['date_index', 'exp_name', 'cell_label', 'cell_type_short',
               'recording_technique', 'onlineAnalysis', 'filter_wheel_values',
               'protocol']
    if blocks.empty:
        cells = pd.DataFrame(columns=columns)
    else:
        frame = blocks.copy()
        frame['recording_technique'] = frame['group_properties'].apply(
            lambda value: value.get('recordingTechnique', '')
            if isinstance(value, dict) else '')

        def text_values(values):
            found = sorted({str(value).strip() for value in values
                            if pd.notna(value) and str(value).strip()
                            and str(value).strip().lower() != 'nan'})
            return ', '.join(found) if found else '?'

        def filter_values(values):
            numeric = sorted({float(value) for value in values if pd.notna(value)})
            return numeric + (['?'] if pd.isna(values).any() else [])

        keys = ['exp_name', 'cell_label', 'cell_type_short', 'protocol']
        cells = (frame.groupby(keys, dropna=False, sort=False)
                 .agg(recording_technique=('recording_technique', text_values),
                      onlineAnalysis=('onlineAnalysis', text_values),
                      filter_wheel_values=('filter_wheel_ndf', filter_values))
                 .reset_index()[columns[1:]]
                 .sort_values(['exp_name', 'cell_label', 'protocol'])
                 .reset_index(drop=True))
        cells.insert(0, 'date_index', pd.factorize(cells['exp_name'], sort=False)[0] + 1)
    if show:
        protocol = ', '.join(sorted(cells['protocol'].unique())) if not cells.empty else 'protocol'
        print(f'{protocol}: {len(cells)} cells across '
              f'{cells.exp_name.nunique() if not cells.empty else 0} experiments')
        sc.scroll_table(cells, height=height)
    return cells


def describe_experiment_protocol(exp_name: str, protocol: str, show: bool = True,
                                 height: int = 500) -> pd.DataFrame:
    """Block-level protocol metadata for one experiment, grouped for display."""
    from retinanalysis.SCutils import explore as sc

    columns = ['exp_name', 'cell_label', 'epoch_group', 'recording_technique',
               'onlineAnalysis', 'block_id', 'protocol', 'filter_wheel_ndf', 'imageName']
    blocks = _protocol_block_rows((protocol,), exp_names=(exp_name,))
    if blocks.empty:
        result = pd.DataFrame(columns=columns)
        if show:
            print(f'No {protocol} blocks found for {exp_name}.')
        return result
    blocks, dropped = _linearized_only(blocks)
    first_parameters, _ = _first_epoch_metadata(blocks['block_id'])

    def parameter(row, name, default=np.nan):
        epoch_value = first_parameters.get(int(row['block_id']), {}).get(name, default)
        if not (pd.isna(epoch_value) if np.isscalar(epoch_value) else False):
            return epoch_value
        block_parameters = row['parameters'] if isinstance(row['parameters'], dict) else {}
        return block_parameters.get(name, default)

    result = pd.DataFrame({
        'exp_name': blocks['exp_name'],
        'cell_label': blocks['cell_label'],
        'epoch_group': blocks['epoch_group'].fillna('?'),
        'recording_technique': blocks['group_properties'].apply(
            lambda value: value.get('recordingTechnique', '?')
            if isinstance(value, dict) else '?'),
        'onlineAnalysis': blocks.apply(lambda row: parameter(row, 'onlineAnalysis', '?'), axis=1),
        'block_id': blocks['block_id'].astype(int),
        'protocol': blocks['protocol'],
        'filter_wheel_ndf': blocks.apply(lambda row: parameter(row, 'NDF'), axis=1),
        'imageName': blocks.apply(lambda row: parameter(row, 'imageName', '?'), axis=1),
        '_start_time': blocks['start_time'],
    }).sort_values(['cell_label', '_start_time', 'block_id']).drop(columns='_start_time')
    result = result.reset_index(drop=True)
    if show:
        print(f'{exp_name} | {protocol} | {result.cell_label.nunique()} cells | '
              f'{len(result)} blocks')
        if dropped:
            print(f'  excluded {dropped} older block(s) without linearizeCones')
        sc.tree_table(result,
                      levels=['exp_name', 'cell_label', 'epoch_group',
                              'recording_technique', 'onlineAnalysis'],
                      height=height, num_cols=('block_id', 'filter_wheel_ndf'))
    return result


def find_blocks(exp_names: Optional[Sequence[str]] = None, show: bool = True,
                height: int = 420, protocols: Optional[Sequence[str]] = None,
                include_stage_ndfs: bool = False) -> pd.DataFrame:
    """Detailed blocks from selected protocols, with ``linearizeCones`` filtering.

    ``LinearEquivalentDisc`` blocks without a ``linearizeCones`` parameter are the
    older, unrelated experiment and are dropped (reported in the output). Database
    metadata and epoch counts are fetched in batches. Slow per-block stage-NDF reads
    are opt-in via ``include_stage_ndfs=True``.
    """
    from retinanalysis.SCutils import explore as sc

    if protocols is None:
        selected_protocols = PROTOCOLS
    elif isinstance(protocols, str):
        selected_protocols = (protocols,)
    else:
        selected_protocols = tuple(protocols)
    blocks = _protocol_block_rows(selected_protocols, exp_names=exp_names)
    if blocks.empty:
        return pd.DataFrame()
    blocks, dropped = _linearized_only(blocks)

    first_parameters, epoch_counts = _first_epoch_metadata(blocks['block_id'])

    df = blocks.copy()
    df['n_epochs'] = df['block_id'].map(epoch_counts).fillna(0).astype(int)
    for key in CONFIG_KEYS:
        df[key] = df.apply(
            lambda row, name=key: first_parameters.get(int(row['block_id']), {}).get(
                name, row['parameters'].get(name, np.nan)
                if isinstance(row['parameters'], dict) else np.nan), axis=1)
    df['site'] = df['protocol'].apply(stimulus_site)
    df['cell_type_short'] = df['cell_type'].astype(str).str.split('\\').str[-1]
    df = df.rename(columns={'NDF': 'filter_wheel_ndf'})
    df['has_filter_wheel'] = df['filter_wheel_ndf'].notna()
    df['light_setting'] = [light_setting(n, b) for n, b in
                           zip(df['filter_wheel_ndf'], df['backgroundIntensity'])]
    rs = [light_level_rstar(n, b) for n, b in
          zip(df['filter_wheel_ndf'], df['backgroundIntensity'])]
    df['rstar'] = [r for r, _ in rs]
    df['light_level'] = [lab for _, lab in rs]
    if include_stage_ndfs:
        df = df.merge(stage_ndf_table(df[['exp_name', 'block_id']], verbose=show),
                      on='block_id', how='left')
        df['stage_ndfs'] = df['stage_ndfs'].fillna('')
    df = df.sort_values(['exp_name', 'cell_label', 'start_time']).reset_index(drop=True)

    if show:
        print(f"{len(df)} blocks | {df['exp_name'].nunique()} experiments | "
              f"{df.groupby(['exp_name', 'cell_label']).ngroups} cells")
        if dropped:
            print(f'  dropped {dropped} LinearEquivalentDisc block(s) with no '
                  f'linearizeCones parameter (the older, unrelated protocol)')
        print('  by protocol: ' + ', '.join(f'{k} {v}' for k, v in
                                            df['protocol'].value_counts().items()))
        for protocol in PROTOCOLS:
            protocol_df = df[df['protocol'].eq(protocol)]
            if protocol_df.empty:
                continue
            print(f'\n{protocol} ({len(protocol_df)} blocks)')
            display_columns = [column for column in FIND_BLOCKS_DISPLAY_COLUMNS
                               if column in protocol_df]
            sc.scroll_table(protocol_df[display_columns], height=height,
                            num_cols=('filter_wheel_ndf', 'maxIntensity', 'n_epochs'))
    return df


def _manual_ndf_setting(value) -> Tuple[str, float, str]:
    """Return fixed-filter label, nominal total OD, and any parsing problem.

    The LightCrafter filter names used for these experiments encode their
    nominal density: ``EL3`` is OD 3 and ``EL06`` is OD 0.6. Embedded ``FW``
    labels are deliberately ignored because the numeric FilterWheel reading is
    authoritative.
    """
    from retinanalysis.utils.isomerization import split_stage_ndfs

    fixed, _embedded_wheel = split_stage_ndfs(value)
    if not fixed:
        return '(not recorded)', np.nan, 'manual NDF not recorded'
    densities = []
    unknown = []
    for name in fixed:
        match = re.fullmatch(r'EL(\d+(?:\.\d+)?)', str(name), flags=re.IGNORECASE)
        if match is None:
            unknown.append(str(name))
            continue
        token = match.group(1)
        densities.append(float(f'0.{token[1:]}') if token.startswith('0')
                         and len(token) > 1 and '.' not in token else float(token))
    label = ', '.join(fixed)
    if unknown:
        return label, np.nan, f"unrecognized manual NDF: {', '.join(unknown)}"
    return label, float(np.sum(densities)), ''


def _qc_numeric_intensity(value) -> Tuple[float, str]:
    """Parse one intensity, choosing the larger value in a comma conflict."""
    if isinstance(value, str) and ',' in value:
        candidates = pd.to_numeric(
            pd.Series([part.strip() for part in value.split(',')]),
            errors='coerce').dropna().to_numpy(dtype=float)
        if candidates.size:
            selected = float(np.max(candidates))
            return selected, f'{value!r} -> {selected:g}'
    numeric = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
    return (float(numeric), '') if pd.notna(numeric) else (np.nan, '')


def estimate_rig_max_intensity(
        protocols: Optional[Sequence[str]] = None,
        blocks: Optional[pd.DataFrame] = None,
        show: bool = True,
        height: int = 360) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate each rig's unattenuated ``maxIntensity`` as a metadata QC.

    For every block, the estimate reverses the nominal manual NDF stack and the
    authoritative numeric FilterWheel OD::

        rig maxIntensity = recorded maxIntensity * 10 ** (manual OD + wheel OD)

    This reads only block metadata and one Stage setting per block; it does not
    load traces or run response analysis. The first returned table is the rig
    map and the second preserves the per-NDF-combination evidence.
    """
    from retinanalysis.SCutils import explore as sc

    if blocks is None:
        frame = find_blocks(
            protocols=protocols, include_stage_ndfs=True, show=False)
    else:
        frame = blocks.copy()
        if 'stage_ndfs' not in frame:
            frame = frame.merge(
                stage_ndf_table(frame[['exp_name', 'block_id']], verbose=show),
                on='block_id', how='left')
    columns = [
        'rig', 'data_source', 'manual_ndfs', 'manual_ndf_od',
        'filter_wheel_ndf', 'recorded_maxIntensity',
        'estimated_rig_maxIntensity', 'estimate_min', 'estimate_max',
        'blocks', 'experiment_dates', 'status',
    ]
    summary_columns = [
        'rig', 'data_source', 'estimated_rig_maxIntensity', 'estimate_min',
        'estimate_max', 'manual_ndfs', 'filter_wheel_ndfs', 'blocks',
        'experiment_dates',
    ]
    if frame.empty:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=columns)

    frame = frame.drop_duplicates('block_id').copy()
    frame['rig'] = frame['exp_name'].astype(str).str.extract(
        r'^\d{4}-\d{2}-\d{2}_([A-Za-z])', expand=False).str.upper()
    frame['data_source'] = frame['rig'].map({'E': 'fred_data', 'G': 'chris_data'}).fillna('other')
    manual = frame.get('stage_ndfs', pd.Series('', index=frame.index)).apply(
        _manual_ndf_setting)
    frame[['manual_ndfs', 'manual_ndf_od', 'status']] = pd.DataFrame(
        manual.tolist(), index=frame.index)
    parsed = frame['maxIntensity'].apply(_qc_numeric_intensity)
    frame['recorded_maxIntensity'] = [value for value, _ in parsed]
    frame['intensity_correction'] = [note for _, note in parsed]
    frame['filter_wheel_ndf'] = pd.to_numeric(
        frame['filter_wheel_ndf'], errors='coerce')
    frame['estimated_rig_maxIntensity'] = (
        frame['recorded_maxIntensity']
        * 10.0 ** (frame['manual_ndf_od'] + frame['filter_wheel_ndf']))

    keys = ['rig', 'data_source', 'manual_ndfs', 'manual_ndf_od',
            'filter_wheel_ndf']

    def values_text(values):
        numeric = pd.to_numeric(values, errors='coerce').dropna().unique()
        return ', '.join(f'{value:g}' for value in sorted(numeric))

    def text_join(values):
        return ' | '.join(sorted({str(value) for value in values if str(value)}))

    evidence = (frame.groupby(keys, dropna=False, sort=True)
                .agg(recorded_maxIntensity=('recorded_maxIntensity', values_text),
                     estimated_rig_maxIntensity=('estimated_rig_maxIntensity', 'median'),
                     estimate_min=('estimated_rig_maxIntensity', 'min'),
                     estimate_max=('estimated_rig_maxIntensity', 'max'),
                     blocks=('block_id', 'nunique'),
                     experiment_dates=('exp_name', 'nunique'),
                     status=('status', text_join))
                .reset_index()[columns])
    inconsistent = (evidence['estimate_min'].gt(0)
                    & evidence['estimate_max'].div(evidence['estimate_min']).gt(1.25))
    evidence.loc[inconsistent & evidence['status'].eq(''), 'status'] = (
        'inconsistent estimates (>25% spread)')
    valid = frame.dropna(subset=['rig', 'estimated_rig_maxIntensity'])
    summary = (valid.groupby(['rig', 'data_source'], sort=True)
               .agg(estimated_rig_maxIntensity=('estimated_rig_maxIntensity', 'median'),
                    estimate_min=('estimated_rig_maxIntensity', 'min'),
                    estimate_max=('estimated_rig_maxIntensity', 'max'),
                    manual_ndfs=('manual_ndfs', text_join),
                    filter_wheel_ndfs=('filter_wheel_ndf', values_text),
                    blocks=('block_id', 'nunique'),
                    experiment_dates=('exp_name', 'nunique'))
               .reset_index()[summary_columns])

    if show:
        print('Formula: recorded maxIntensity × 10^(manual NDF OD + FilterWheel OD)')
        corrections = frame.loc[frame['intensity_correction'].ne(''),
                                ['exp_name', 'block_id', 'intensity_correction']]
        for row in corrections.itertuples(index=False):
            print(f'ALERT: corrected {row.exp_name} block {row.block_id}: '
                  f'{row.intensity_correction}')
        unavailable = int(frame['estimated_rig_maxIntensity'].isna().sum())
        if unavailable:
            print(f'ALERT: {unavailable} block(s) could not be estimated because '
                  'manual NDF, wheel NDF, or maxIntensity was unavailable.')
        for row in evidence.loc[inconsistent].itertuples(index=False):
            print(f'ALERT: rig {row.rig}, {row.manual_ndfs}, '
                  f'FW{row.filter_wheel_ndf:g} spans '
                  f'{row.estimate_min:g}–{row.estimate_max:g}; check its blocks.')
        print('\nRig maxIntensity map (median with full observed range):')
        sc.scroll_table(summary, height=220,
                        num_cols=('estimated_rig_maxIntensity', 'estimate_min',
                                  'estimate_max', 'blocks', 'experiment_dates'))
        print('\nEvidence by manual-NDF and FilterWheel combination:')
        sc.scroll_table(evidence, height=height,
                        num_cols=('manual_ndf_od', 'filter_wheel_ndf',
                                  'estimated_rig_maxIntensity', 'estimate_min',
                                  'estimate_max', 'blocks', 'experiment_dates'))
    return summary, evidence


def group_blocks(df: pd.DataFrame, show: bool = True, height: int = 420) -> pd.DataFrame:
    """One row per recording group: experiment x cell x mode x site x light level.

    Groups on ``onlineAnalysis`` after :func:`check_series_resistance` has
    resolved it against the amplifier. Run that first: nearly half these blocks
    were recorded with the label left at ``'none'``, and grouping on the raw
    label would put every one of them in the same bucket regardless of how the
    cell was actually held.

    Blocks still carrying an unresolved label are reported rather than silently
    pooled; :func:`analyze_group` resolves each one itself, so the analysis is
    right either way, but the grouping is only right once the labels are.
    """
    from retinanalysis.SCutils import explore as sc

    df = df.copy()
    if 'imageName' not in df:
        df['imageName'] = np.nan
    df['mode'] = df['onlineAnalysis'].astype(str).str.lower()
    unresolved = ~df['mode'].isin(['extracellular', 'exc', 'inh'])
    if show and unresolved.any():
        print(f"WARNING: {int(unresolved.sum())} block(s) still have an unresolved "
              f"onlineAnalysis ({', '.join(sorted(df.loc[unresolved, 'mode'].unique()))}). "
              f'Run check_series_resistance() first so they group by how the cell was '
              f'actually recorded rather than by a label nobody set.')

    keys = ['exp_name', 'cell_label', 'cell_type_short', 'mode', 'site',
            'filter_wheel_ndf', 'backgroundIntensity']
    agg = dict(blocks=('block_id', 'size'), epochs=('n_epochs', 'sum'),
               onlineAnalysis=('mode', 'first'),
               protocols=('protocol', lambda s: ', '.join(sorted(set(s)))),
               light_setting=('light_setting', 'first'), rstar=('rstar', 'first'),
               light_level=('light_level', 'first'),
               weber=('WeberConstant', 'first'),
               max_intensity=('maxIntensity', 'first'),
               maxIntensity=('maxIntensity', 'first'),
               light_settings=('light_setting', 'first'),
               block_ids=('block_id', lambda s: ', '.join(str(int(b)) for b in sorted(s))),
               image_names=('imageName',
                            lambda s: sorted({str(value) for value in s if pd.notna(value)})))
    recorded_col = ('onlineAnalysis_recorded' if 'onlineAnalysis_recorded' in df.columns
                    else 'onlineAnalysis')
    agg['recorded_labels'] = (recorded_col,
                              lambda s: ', '.join(sorted({str(v) for v in s})))

    def median_mohm(values):
        numeric = np.asarray(values, dtype=float)
        finite = numeric[np.isfinite(numeric)]
        return np.round(np.median(finite) / 1e6, 2) if finite.size else np.nan

    for name, source in (('stage_ndfs', 'stage_ndfs'),
                         ('rs_mohm', 'series_resistance'),
                         ('epochs_high_rs', 'n_epochs_high_rs')):
        if source not in df.columns:
            continue
        if name == 'rs_mohm':
            agg[name] = (source, median_mohm)
        elif name == 'epochs_high_rs':
            agg[name] = (source, 'sum')
        else:
            agg[name] = (source, lambda s: ' | '.join(sorted({str(v) for v in s})))
    g = df.groupby(keys, dropna=False, sort=False).agg(**agg).reset_index()
    if show:
        print(f'{len(g)} recording groups (experiment x cell x mode x site x light level)')
        display = g[list(GROUP_DISPLAY_COLUMNS)].sort_values(
            ['exp_name', 'cell_label', 'onlineAnalysis', 'site', 'light_settings'])
        sc.tree_table(display, levels=['exp_name', 'cell_label', 'cell_type_short'],
                      height=height, num_cols=('maxIntensity',))
    return g


def condition_image_summary(blocks: pd.DataFrame) -> pd.DataFrame:
    """One row per image in a selected cell/mode/FilterWheel condition.

    ``meanIntensity`` follows the protocol convention requested in the notebook:
    ``maxIntensity * backgroundIntensity``. Patch indices are intentionally not
    part of this aggregation because they restart for every image name.
    """
    if blocks.empty:
        return pd.DataFrame(columns=CONDITION_SUMMARY_COLUMNS)

    frame = blocks.copy()
    frame['imageName'] = frame['imageName'].astype(str)
    frame['meanIntensity'] = (pd.to_numeric(frame['maxIntensity'], errors='coerce')
                              * pd.to_numeric(frame['backgroundIntensity'],
                                              errors='coerce'))

    def numeric_value(values):
        values = pd.to_numeric(values, errors='coerce')
        finite = np.asarray(values[np.isfinite(values)], dtype=float)
        unique = np.unique(finite)
        if unique.size == 0:
            return np.nan
        if unique.size == 1:
            return float(unique[0])
        return ', '.join(f'{value:g}' for value in unique)

    summary = (frame.groupby('imageName', sort=True, dropna=False)
               .agg(block_ids=('block_id', lambda values: [int(v) for v in sorted(values)]),
                    epochs=('n_epochs', 'sum'),
                    maxIntensity=('maxIntensity', numeric_value),
                    backgroundIntensity=('backgroundIntensity', numeric_value),
                    meanIntensity=('meanIntensity', numeric_value))
               .reset_index())
    return summary[list(CONDITION_SUMMARY_COLUMNS)]


def select_condition_blocks(blocks: pd.DataFrame, cell_label: str,
                            online_analysis: str, filter_wheel_ndf: float,
                            show: bool = True, height: int = 360) -> pd.DataFrame:
    """Select one cell x recording mode x FilterWheel condition.

    The input is normally ``selected_blocks`` from Sections 1--2 of a cone/disc
    notebook and therefore already belongs to one experiment.
    The returned rows retain every image-specific background intensity while
    pooling across image names for the requested condition.
    """
    from retinanalysis.SCutils import explore as sc

    required = {'cell_label', 'onlineAnalysis', 'filter_wheel_ndf', 'block_id'}
    missing = sorted(required - set(blocks.columns))
    if missing:
        raise ValueError(f'blocks is missing required column(s): {missing}')

    mode = str(online_analysis).strip().lower()
    wheel = float(filter_wheel_ndf)
    numeric_wheel = pd.to_numeric(blocks['filter_wheel_ndf'], errors='coerce')
    keep = (blocks['cell_label'].astype(str).eq(str(cell_label))
            & blocks['onlineAnalysis'].astype(str).str.strip().str.lower().eq(mode)
            & np.isclose(numeric_wheel, wheel, equal_nan=False))
    sort_columns = [column for column in ('imageName', 'start_time', 'block_id')
                    if column in blocks]
    selected = blocks.loc[keep].sort_values(sort_columns).copy()
    if selected.empty:
        available = (blocks[['cell_label', 'onlineAnalysis', 'filter_wheel_ndf']]
                     .drop_duplicates().sort_values(['cell_label', 'onlineAnalysis',
                                                     'filter_wheel_ndf']))
        choices = '; '.join(
            f"{row.cell_label}/{row.onlineAnalysis}/FW{row.filter_wheel_ndf:g}"
            for row in available.itertuples())
        raise ValueError(
            f'No blocks match cell_label={cell_label!r}, onlineAnalysis={mode!r}, '
            f'FilterWheel={wheel:g}. Available: {choices}')

    experiments = selected['exp_name'].astype(str).unique()
    if len(experiments) != 1:
        raise ValueError('Select blocks from one experiment before analyzing one cell.')

    summary = condition_image_summary(selected)
    selected.attrs['image_summary'] = summary
    if show:
        cell_type_column = ('cell_type_short' if 'cell_type_short' in selected
                            else 'cell_type' if 'cell_type' in selected else None)
        cell_types = ([] if cell_type_column is None else sorted({
            str(value) for value in selected[cell_type_column].dropna()
            if str(value).strip()
        }))
        cell_type = ', '.join(cell_types) if cell_types else 'Unknown'
        block_ids = ', '.join(str(int(value)) for value in selected['block_id'])
        print(f'{experiments[0]}/{cell_label} ({cell_type}) | '
              f'{mode} | FilterWheel {wheel:g}')
        print(f'block_ids: {block_ids}')
        print(f'{len(summary)} imageNames | {int(summary.epochs.sum())} epochs')
        sc.scroll_table(summary, height=height,
                        num_cols=('epochs', 'maxIntensity',
                                  'backgroundIntensity', 'meanIntensity'))
    return selected


# --------------------------------------------------------------------------
# per-group analysis
# --------------------------------------------------------------------------

@dataclass
class DiscRecord:
    """One (experiment, cell, recording mode, site, light level)."""
    exp_name: str
    cell_label: str
    cell_type: str
    online_analysis: str
    site: str
    ndf: float
    background_intensity: float
    rstar: float
    light_setting: str
    weber_constant: float
    image_names: List[str]
    patch_ids: np.ndarray
    # Per-patch mean response, one row per patch, for each stimulus category.
    image_onset: np.ndarray
    image_offset: np.ndarray
    disc_onset: np.ndarray
    disc_offset: np.ndarray
    cone_onset: np.ndarray
    cone_offset: np.ndarray
    # Nonlinearity indices, image vs each disc.
    nli_disc_onset: np.ndarray
    nli_disc_offset: np.ndarray
    nli_cone_onset: np.ndarray
    nli_cone_offset: np.ndarray
    n_epochs: int
    n_patches: int
    threshold_onset: float
    threshold_offset: float
    block_ids: List[int]
    # True when exc/inh came from the sign of the data rather than a recorded
    # onlineAnalysis label, so an inferred polarity is never mistaken for one
    # the experimenter set.
    mode_inferred: bool = False
    # The label as the experimenter set it (often 'none'), kept beside the mode
    # the block was actually analyzed as. Plus the amplifier reading behind that
    # decision, in ohms.
    online_analysis_recorded: str = ''
    series_resistance: float = np.nan
    n_epochs_high_rs: int = 0
    config: Dict = field(default_factory=dict)
    units: str = ''

    @property
    def key(self) -> str:
        return record_key(self.exp_name, self.cell_label, self.online_analysis,
                          self.site, self.ndf, self.background_intensity)

    def summary_row(self) -> Dict:
        def m(x):
            x = np.asarray(x, dtype=float)
            return float(np.nanmean(x)) if x.size else np.nan
        return {
            'key': self.key, 'exp_name': self.exp_name, 'cell_label': self.cell_label,
            'cell_type': self.cell_type, 'online_analysis': self.online_analysis,
            'site': self.site, 'ndf': self.ndf,
            'background_intensity': self.background_intensity, 'rstar': self.rstar,
            'rstar_measured': is_calibrated(self.ndf, self.background_intensity),
            'light_setting': self.light_setting, 'weber_constant': self.weber_constant,
            'image_names': ','.join(self.image_names),
            'n_epochs': self.n_epochs, 'n_patches': self.n_patches,
            'mode_inferred': bool(self.mode_inferred),
            'online_analysis_recorded': self.online_analysis_recorded or self.online_analysis,
            'series_resistance': self.series_resistance,
            'n_epochs_high_rs': self.n_epochs_high_rs,
            'max_intensity': self.config.get('maxIntensity', np.nan),
            'threshold_onset': self.threshold_onset,
            'threshold_offset': self.threshold_offset,
            'nli_disc_onset': m(self.nli_disc_onset),
            'nli_disc_offset': m(self.nli_disc_offset),
            'nli_cone_onset': m(self.nli_cone_onset),
            'nli_cone_offset': m(self.nli_cone_offset),
            'block_ids': ','.join(str(b) for b in self.block_ids), 'units': self.units,
        }

    def describe(self) -> str:
        def m(x):
            x = np.asarray(x, float)
            return np.nanmean(x) if x.size else np.nan
        return (f'{self.exp_name} | {self.cell_type} | {self.cell_label} | '
                f'{self.online_analysis} | disc over {self.site} | '
                f'{self.light_setting} | {self.n_patches} patches, {self.n_epochs} epochs\n'
                f'  NLI standard disc : onset {m(self.nli_disc_onset):+.3f}  '
                f'offset {m(self.nli_disc_offset):+.3f}\n'
                f'  NLI cone-lin disc : onset {m(self.nli_cone_onset):+.3f}  '
                f'offset {m(self.nli_cone_offset):+.3f}')


@dataclass
class ConditionAnalysis:
    """Exact-onset responses for one cell/mode/FilterWheel condition."""
    exp_name: str
    cell_label: str
    cell_type: str
    online_analysis: str
    filter_wheel_ndf: float
    block_ids: List[int]
    protocols: List[str]
    site: str
    image_summary: pd.DataFrame
    epoch_responses: pd.DataFrame
    patch_responses: pd.DataFrame
    units: str
    threshold: float
    loaded_from_saved: bool = False


def summarize_patch_responses(epoch_responses: pd.DataFrame,
                              threshold: float) -> pd.DataFrame:
    """Mean, SEM, and NLI per unique ``(imageName, patchIndex)`` pair."""
    required = {'imageName', 'patchIndex', 'category', 'response'}
    missing = sorted(required - set(epoch_responses.columns))
    if missing:
        raise ValueError(f'epoch_responses is missing required column(s): {missing}')

    def sem(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return (float(np.std(values, ddof=1) / np.sqrt(values.size))
                if values.size > 1 else 0.0)

    stats = (epoch_responses.groupby(['imageName', 'patchIndex', 'category'], sort=True)
             .agg(response_mean=('response', 'mean'),
                  response_sem=('response', sem),
                  n_trials=('response', 'count'))
             .reset_index())
    index = ['imageName', 'patchIndex']
    base = stats[index].drop_duplicates().set_index(index)
    for category in ('image', 'disc', 'cone_disc'):
        category_rows = stats.loc[stats['category'].eq(category)].set_index(index)
        for source, suffix in (('response_mean', 'mean'), ('response_sem', 'sem'),
                               ('n_trials', 'n')):
            base[f'{category}_{suffix}'] = category_rows[source]
    patches = base.reset_index()
    for column in ('image_mean', 'disc_mean', 'cone_disc_mean'):
        if column not in patches:
            patches[column] = np.nan
    keep = (patches['image_mean'].notna()
            & (patches['disc_mean'].notna() | patches['cone_disc_mean'].notna()))
    patches = patches.loc[keep].reset_index(drop=True)

    def nli(disc_column):
        image = patches['image_mean'].to_numpy(dtype=float)
        disc = patches[disc_column].to_numpy(dtype=float)
        with np.errstate(invalid='ignore', divide='ignore'):
            values = (image - disc) / (np.abs(image) + np.abs(disc))
        below = np.maximum(np.abs(image), np.abs(disc)) < float(threshold)
        values[below] = 0.0
        values[~np.isfinite(image) | ~np.isfinite(disc)] = np.nan
        return values

    patches['nli_disc'] = nli('disc_mean')
    patches['nli_cone_disc'] = nli('cone_disc_mean')
    patches['patch_key'] = [f'{image}:{patch:g}' for image, patch in
                            zip(patches['imageName'], patches['patchIndex'])]
    return patches


def analyze_group(exp_name: str, block_ids: Sequence[int],
                  online_analysis: Optional[str] = None,
                  spike_offset: int = DEFAULTS['spike_offset'],
                  wc_offset: int = DEFAULTS['wc_offset'],
                  smooth_ms: float = DEFAULTS['smooth_ms'],
                  detector_kwargs: Optional[dict] = None,
                  max_series_resistance: Optional[float] = MAX_SERIES_RESISTANCE,
                  verbose: bool = True) -> DiscRecord:
    """Port of the per-node body of ``analyzeLinearDiscCone.m``.

    Measures each epoch's onset and offset response, averages within
    (image, patch, stimulus category), and forms the two nonlinearity indices.
    A patch is kept only if it has an image trial and at least one disc trial,
    matching the MATLAB.

    How the block is treated comes from the amplifier, not the label:
    :func:`resolve_recording_mode` reads ``stimulus:Amp1:seriesResistance`` and,
    where that alone cannot decide, the trace itself. That matters more here
    than anywhere else — nearly half these blocks were recorded with
    ``onlineAnalysis`` left at ``'none'``, so for most of the dataset there is
    no label to trust in the first place. A block recorded through more than
    ``max_series_resistance`` ohms is skipped.
    """
    import retinanalysis as ra
    from scipy.ndimage import uniform_filter1d

    per_epoch = []          # (image, patch, category, onset, offset)
    first_params, used_blocks = None, []
    rs_kept, n_high_rs, mode_inferred = [], 0, False
    recorded_mode = ''

    for bid in block_ids:
        sb = ra.StimBlock(exp_name, int(bid), verbose=False)
        ep = sb.df_epochs
        params = list(ep['epoch_parameters'])
        p0 = params[0]
        if first_params is None:
            first_params = p0
        recorded_mode = (online_analysis or p0.get('onlineAnalysis', 'extracellular')).lower()

        # Load before deciding: the label is often absent or wrong, and the
        # amplifier reading plus the trace are what settle it.
        rb = ra.SCResponseBlock(exp_name, int(bid), b_spiking=False, verbose=False)
        sr = float(rb.amp_sample_rate)
        try:
            rs = read_series_resistance(exp_name, int(bid))
        except Exception:
            rs = np.full(len(ep), np.nan)
        rs_median = float(np.nanmedian(rs)) if np.isfinite(rs).any() else np.nan

        if (max_series_resistance is not None and np.isfinite(rs_median)
                and rs_median > max_series_resistance):
            n_high_rs += len(ep)
            if verbose:
                print(f'  block {bid}: series resistance {rs_median / 1e6:.1f} MOhm is above '
                      f'the {max_series_resistance / 1e6:g} MOhm cutoff — block skipped')
            continue

        mode, note = resolve_recording_mode(recorded_mode, rs_median, amp_data=rb.amp_data,
                                            sample_rate=sr,
                                            detector_kwargs=detector_kwargs)
        if note:
            mode_inferred = True
            if verbose:
                print(f'  block {bid}: {note}')
        spiking = mode == 'extracellular'
        if spiking:
            rb.get_spike_times(**(detector_kwargs or {}))

        pre_pts = int(round(float(p0['preTime']) / 1e3 * sr))
        stim_pts = int(round(float(p0['stimTime']) / 1e3 * sr))
        n_samples = rb.amp_data.shape[1]
        rs_kept.append(rs_median)
        used_blocks.append(int(bid))

        if spiking:
            # Firing rate in each window. The onset window is the stimulus, the
            # offset window is everything after it, and they differ in length --
            # hence each count is divided by its own duration.
            on_lo, on_hi = pre_pts + spike_offset, pre_pts + stim_pts + spike_offset
            stim_s = (on_hi - on_lo) / sr
            offset_s = max(n_samples - on_hi, 1) / sr
            onsets, offsets = [], []
            for st in rb.spike_times:
                st = np.asarray(st, dtype=float)
                onsets.append(float(np.sum((st > on_lo) & (st < on_hi))) / stim_s)
                offsets.append(float(np.sum(st > on_hi)) / offset_s)
            units = 'rate (Hz)'
        else:
            if mode not in ('exc', 'inh'):
                # Only reachable when the label was never set *and* the h5 had
                # no series-resistance reading to resolve it with. Excitatory
                # currents are inward (negative), so the sign still names the
                # holding potential -- but it is an inference, hence
                # mode_inferred.
                mode_inferred = True
                mode = 'exc' if float(np.mean(rb.amp_data)) < 0 else 'inh'
            sign = -1.0 if mode == 'exc' else 1.0
            width = max(int(round(smooth_ms / 1e3 * sr)), 1)
            data = uniform_filter1d(np.asarray(rb.amp_data, dtype=float), size=width, axis=1)
            data = data - data[:, :pre_pts].mean(axis=1, keepdims=True)
            lo = pre_pts + wc_offset
            hi = min(pre_pts + stim_pts + wc_offset, n_samples)
            onsets = [sign * float(data[i, lo:hi].mean()) for i in range(data.shape[0])]
            offsets = [sign * float(data[i, hi:].mean()) if hi < n_samples else np.nan
                       for i in range(data.shape[0])]
            stim_s = (hi - lo) / sr
            offset_s = max(n_samples - hi, 1) / sr
            units = 'excitation (pA)' if mode == 'exc' else 'current (pA)'

        for i, p in enumerate(params):
            cat = category_of(p.get('stimulusTag'))
            if not cat or i >= len(onsets):
                continue
            per_epoch.append({'image': str(p.get('imageName')),
                              'patch': float(p.get('imagePatchIndex', np.nan)),
                              'category': cat, 'onset': onsets[i], 'offset': offsets[i]})

    df = pd.DataFrame(per_epoch)
    if df.empty:
        raise ValueError(f'{exp_name} blocks {list(block_ids)}: no usable epochs')

    # Mean response per (image, patch, category).
    means = df.groupby(['image', 'patch', 'category'])[['onset', 'offset']].mean()
    wide = means.unstack('category')

    def col(field, cat):
        return (wide[(field, cat)].to_numpy(dtype=float)
                if (field, cat) in wide.columns else np.full(len(wide), np.nan))

    image_on, image_off = col('onset', 'image'), col('offset', 'image')
    disc_on, disc_off = col('onset', 'disc'), col('offset', 'disc')
    cone_on, cone_off = col('onset', 'cone_disc'), col('offset', 'cone_disc')

    # Keep patches with an image trial and at least one disc trial.
    keep = np.isfinite(image_on) & (np.isfinite(disc_on) | np.isfinite(cone_on))
    idx = wide.index[keep]
    image_on, image_off = image_on[keep], image_off[keep]
    disc_on, disc_off = disc_on[keep], disc_off[keep]
    cone_on, cone_off = cone_on[keep], cone_off[keep]

    thresh_on, thresh_off = working_thresholds(mode, stim_s, offset_s,
                                               spiking=(mode == 'extracellular'))
    ndf = float(first_params.get('NDF', np.nan))
    bg = float(first_params['backgroundIntensity'])
    rstar, _ = light_level_rstar(ndf, bg)

    import retinanalysis as _ra
    summary = _ra.get_exp_summary(exp_name)
    row = summary[summary['block_id'].eq(int(used_blocks[0]))].iloc[0]

    rec = DiscRecord(
        exp_name=exp_name, cell_label=str(row['cell_label']), cell_type=str(row['cell_type']),
        online_analysis=mode, site=stimulus_site(str(row['protocol_name']).split('.')[-1]),
        ndf=ndf, background_intensity=bg, rstar=rstar,
        light_setting=light_setting(ndf, bg),
        weber_constant=float(first_params.get('WeberConstant', np.nan)),
        image_names=sorted({str(i) for i, _ in idx}),
        patch_ids=np.array([p for _, p in idx], dtype=float),
        image_onset=image_on, image_offset=image_off,
        disc_onset=disc_on, disc_offset=disc_off,
        cone_onset=cone_on, cone_offset=cone_off,
        nli_disc_onset=compute_nli(image_on, disc_on, thresh_on),
        nli_disc_offset=compute_nli(image_off, disc_off, thresh_off),
        nli_cone_onset=compute_nli(image_on, cone_on, thresh_on),
        nli_cone_offset=compute_nli(image_off, cone_off, thresh_off),
        n_epochs=int(len(df)), n_patches=int(keep.sum()), mode_inferred=mode_inferred,
        threshold_onset=thresh_on, threshold_offset=thresh_off,
        block_ids=used_blocks,
        online_analysis_recorded=recorded_mode,
        series_resistance=(float(np.nanmedian(rs_kept))
                           if rs_kept and np.isfinite(rs_kept).any() else np.nan),
        n_epochs_high_rs=n_high_rs,
        config={k: first_params.get(k) for k in CONFIG_KEYS}, units=units)
    if verbose:
        print(rec.describe())
    return rec


def analyze_condition(blocks: pd.DataFrame,
                      detector_kwargs: Optional[dict] = None,
                      max_series_resistance: Optional[float] = MAX_SERIES_RESISTANCE,
                      reuse_saved: bool = True,
                      saved_output_dir=None,
                      verbose: bool = True) -> ConditionAnalysis:
    """Analyze one selected cell/mode/FilterWheel condition.

    The onset window is exactly ``preTime`` through ``preTime + stimTime``.
    Extracellular responses are spike counts in that window. Whole-cell traces
    are baseline-subtracted using the preTime samples and integrated over the
    window in pA*s; excitatory currents are sign-flipped so stronger responses
    remain positive. Repeated epochs provide the SEM used for both plot axes.
    """
    import contextlib
    import io
    import warnings

    import retinanalysis as ra
    from retinanalysis.utils.datajoint_utils import get_epochblock_amp_data
    from retinanalysis.utils.spike_detector import detector

    if blocks.empty:
        raise ValueError('No condition blocks were supplied.')
    for column in ('exp_name', 'cell_label', 'onlineAnalysis', 'filter_wheel_ndf',
                   'block_id'):
        if blocks[column].nunique(dropna=False) != 1 and column != 'block_id':
            raise ValueError(f'Condition must contain exactly one {column}.')

    exp_name = str(blocks['exp_name'].iloc[0])
    cell_label = str(blocks['cell_label'].iloc[0])
    mode = str(blocks['onlineAnalysis'].iloc[0]).strip().lower()
    if mode not in ('extracellular', 'exc', 'inh'):
        raise ValueError(f'Unsupported onlineAnalysis {mode!r}.')
    ndf = float(blocks['filter_wheel_ndf'].iloc[0])
    cell_type_column = ('cell_type_short' if 'cell_type_short' in blocks
                        else 'cell_type')
    cell_type = str(blocks[cell_type_column].iloc[0])

    if reuse_saved and detector_kwargs is None:
        saved = load_condition_output(blocks, output_dir=saved_output_dir, verbose=verbose)
        if saved is not None:
            return saved

    responses = []
    used_blocks = []
    for row in blocks.itertuples(index=False):
        block_id = int(row.block_id)
        sb = ra.StimBlock(exp_name, block_id, verbose=False)
        parameters = list(sb.df_epochs['epoch_parameters'])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            amp_data, sample_rate = get_epochblock_amp_data(
                exp_name, block_id, verbose=False)
        amp_data = np.asarray(amp_data, dtype=float)
        sample_rate = float(sample_rate)
        try:
            series_resistance = np.asarray(read_series_resistance(exp_name, block_id),
                                           dtype=float)
            rs_median = (float(np.nanmedian(series_resistance))
                         if np.isfinite(series_resistance).any() else np.nan)
        except Exception:
            rs_median = np.nan
        if (max_series_resistance is not None and np.isfinite(rs_median)
                and rs_median > max_series_resistance):
            if verbose:
                print(f'  block {block_id}: series resistance {rs_median / 1e6:.1f} MOhm '
                      f'exceeds {max_series_resistance / 1e6:g} MOhm; skipped')
            continue

        if mode == 'extracellular':
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning,
                                        module=r'retinanalysis\.utils\.spike_detector')
                options = dict(detector_kwargs or {})
                options.setdefault('verbose', False)
                spike_times, _, _ = detector(
                    amp_data, sample_rate=sample_rate, **options)
        used_blocks.append(block_id)

        for epoch_index, parameters_i in enumerate(parameters):
            category = category_of(parameters_i.get('stimulusTag'))
            patch_index = parameters_i.get('patchIndex')
            if patch_index is None:
                patch_index = parameters_i.get('imagePatchIndex', np.nan)
            try:
                patch_index = float(patch_index)
            except (TypeError, ValueError):
                patch_index = np.nan
            if (not category or epoch_index >= len(amp_data)
                    or not np.isfinite(patch_index)):
                continue
            pre_points = int(round(float(parameters_i['preTime']) / 1e3 * sample_rate))
            stim_points = int(round(float(parameters_i['stimTime']) / 1e3 * sample_rate))
            onset_start = max(pre_points, 0)
            onset_stop = min(pre_points + stim_points, amp_data.shape[1])
            if onset_stop <= onset_start:
                continue

            if mode == 'extracellular':
                spikes = np.asarray(spike_times[epoch_index], dtype=float)
                response = float(np.sum((spikes >= onset_start) & (spikes < onset_stop)))
            else:
                trace = amp_data[epoch_index]
                baseline = float(np.mean(trace[:pre_points])) if pre_points > 0 else 0.0
                sign = -1.0 if mode == 'exc' else 1.0
                response = sign * float(np.sum(trace[onset_start:onset_stop] - baseline)
                                        / sample_rate)

            image_name = str(parameters_i.get('imageName', getattr(row, 'imageName', '?')))
            max_intensity = float(parameters_i.get(
                'maxIntensity', getattr(row, 'maxIntensity', np.nan)))
            background = float(parameters_i.get(
                'backgroundIntensity', getattr(row, 'backgroundIntensity', np.nan)))
            responses.append({
                'block_id': block_id, 'epoch_index': epoch_index,
                'imageName': image_name, 'patchIndex': patch_index,
                'category': category, 'response': response,
                'maxIntensity': max_intensity,
                'backgroundIntensity': background,
                'meanIntensity': max_intensity * background,
            })

    epoch_responses = pd.DataFrame(responses)
    if epoch_responses.empty:
        raise ValueError(f'{exp_name}/{cell_label}: no usable epochs in the selected condition')
    threshold = float(NLI_THRESHOLD[mode])
    patch_responses = summarize_patch_responses(epoch_responses, threshold=threshold)
    if patch_responses.empty:
        raise ValueError(f'{exp_name}/{cell_label}: no image/disc patch pairs were found')

    units = 'spikes' if mode == 'extracellular' else 'charge (pA*s)'
    analysis = ConditionAnalysis(
        exp_name=exp_name, cell_label=cell_label, cell_type=cell_type,
        online_analysis=mode, filter_wheel_ndf=ndf, block_ids=used_blocks,
        protocols=sorted({str(value) for value in blocks['protocol']}),
        site=str(blocks['site'].iloc[0]),
        image_summary=condition_image_summary(blocks.loc[blocks['block_id'].isin(used_blocks)]),
        epoch_responses=epoch_responses, patch_responses=patch_responses,
        units=units, threshold=threshold)
    if verbose:
        print(f'analyzed {len(used_blocks)} blocks, {len(epoch_responses)} epochs, '
              f'{len(patch_responses)} image/patch pairs')
    return analysis


# --------------------------------------------------------------------------
# record store
# --------------------------------------------------------------------------

def store_dir():
    """``<OUTPUT_DIR>/linear_equivalent_disc``."""
    from pathlib import Path
    from retinanalysis.config.settings import OUTPUT_DIR
    return Path(OUTPUT_DIR) / 'linear_equivalent_disc'


def condition_population_table(analysis: ConditionAnalysis) -> pd.DataFrame:
    """Population-ready patch rows for one saved cell condition.

    One row is one unique ``(imageName, patchIndex)`` pair. Image-specific
    intensity metadata is joined without collapsing patch indices reused by a
    different image.
    """
    patches = analysis.patch_responses.copy()
    image_metadata = analysis.image_summary[
        ['imageName', 'block_ids', 'epochs', 'maxIntensity', 'backgroundIntensity',
         'meanIntensity']].copy()
    table = patches.merge(image_metadata, on='imageName', how='left', validate='many_to_one')
    table.insert(0, 'date', analysis.exp_name)
    table.insert(1, 'cell_label', analysis.cell_label)
    table.insert(2, 'cell_type', analysis.cell_type)
    table.insert(3, 'onlineAnalysis', analysis.online_analysis)
    table.insert(4, 'protocol', ', '.join(analysis.protocols))
    table.insert(5, 'site', analysis.site)
    table.insert(6, 'filter_wheel_ndf', analysis.filter_wheel_ndf)
    table['block_ids'] = table['block_ids'].apply(
        lambda values: ','.join(str(int(value)) for value in values)
        if isinstance(values, (list, tuple, np.ndarray)) else str(values))
    table['response_units'] = analysis.units
    table['output_version'] = CONDITION_OUTPUT_VERSION
    table = table.rename(columns={
        'image_mean': 'image_response', 'image_sem': 'image_response_sem',
        'image_n': 'image_trials', 'disc_mean': 'disc_response',
        'disc_sem': 'disc_response_sem', 'disc_n': 'disc_trials',
        'cone_disc_mean': 'cone_disc_response',
        'cone_disc_sem': 'cone_disc_response_sem',
        'cone_disc_n': 'cone_disc_trials',
        'nli_disc': 'nli_image_vs_disc',
        'nli_cone_disc': 'nli_image_vs_cone_disc',
        'epochs': 'image_epochs',
    })
    columns = [
        'date', 'cell_label', 'cell_type', 'onlineAnalysis', 'protocol', 'site',
        'filter_wheel_ndf', 'imageName', 'patchIndex', 'patch_key', 'block_ids',
        'image_epochs', 'maxIntensity', 'backgroundIntensity', 'meanIntensity',
        'response_units',
        'image_response', 'image_response_sem', 'image_trials',
        'disc_response', 'disc_response_sem', 'disc_trials',
        'cone_disc_response', 'cone_disc_response_sem', 'cone_disc_trials',
        'nli_image_vs_disc', 'nli_image_vs_cone_disc', 'output_version',
    ]
    return table[columns].sort_values(['imageName', 'patchIndex']).reset_index(drop=True)


def _condition_protocol_folder(
        protocol: Union[str, Sequence[str]]) -> str:
    """Map saved protocols to the center- or annulus-disc output folder."""
    protocols = {protocol} if isinstance(protocol, str) else set(protocol)
    has_annulus = 'LinearEquivalentAnnulus' in protocols
    has_center = bool(protocols.intersection(
        {'LinearEquivalentDisc', 'LinearEquivalentDiscConeLin'}))
    if has_annulus and has_center:
        raise ValueError('A saved condition cannot mix center- and annulus-disc protocols.')
    if has_annulus:
        return 'annulus_disc'
    if has_center:
        return 'center_disc'
    raise ValueError(f'Unsupported saved condition protocol(s): {sorted(protocols)}')


def condition_output_dir(
        protocol: Optional[Union[str, Sequence[str]]] = None):
    """Return the shared legacy root or a protocol-specific output directory."""
    root = store_dir() / 'condition_outputs'
    return root if protocol is None else root / _condition_protocol_folder(protocol)


def _condition_read_directories(output_dir=None, protocol=None):
    """Directories to search, with protocol-specific storage before legacy root."""
    from pathlib import Path

    if output_dir is not None:
        return [Path(output_dir)]
    root = condition_output_dir()
    if protocol is None:
        directories = [root / 'center_disc', root / 'annulus_disc']
    else:
        directories = [condition_output_dir(protocol)]
    return directories + [root]


def _condition_paths(output_dir=None, protocol=None, suffix='.h5'):
    """Find saved conditions once, preferring routed folders over legacy root."""
    paths = []
    seen_names = set()
    for directory in _condition_read_directories(output_dir, protocol):
        if not directory.exists():
            continue
        for path in sorted(directory.glob(f'*{suffix}')):
            if path.name not in seen_names:
                paths.append(path)
                seen_names.add(path.name)
    return paths


def _condition_output_name_from_values(exp_name: str, cell_label: str,
                                       online_analysis: str, site: str,
                                       protocols: Sequence[str],
                                       filter_wheel_ndf: float,
                                       suffix: str = '.h5') -> str:
    import re

    raw = '__'.join((exp_name, cell_label, online_analysis, site,
                    ','.join(protocols), f'FW{filter_wheel_ndf:g}'))
    return re.sub(r'[^A-Za-z0-9_-]+', '-', raw).strip('-') + suffix


def _condition_output_name(analysis: ConditionAnalysis) -> str:
    return _condition_output_name_from_values(
        analysis.exp_name, analysis.cell_label, analysis.online_analysis,
        analysis.site, analysis.protocols, analysis.filter_wheel_ndf)


def matching_condition_outputs(analysis: ConditionAnalysis, output_dir=None):
    """Saved files matching one date/cell/mode/site/FilterWheel condition.

    Protocol names are deliberately not part of this identity. The two center
    protocol implementations belong to one analysis family, so saving the same
    cell condition once under each protocol name would otherwise duplicate it
    in population analysis. Both routed HDF5 files and legacy HDF5/CSV files
    are checked using their stored metadata rather than their filenames.
    """
    import h5py

    def _text(value):
        return value.decode('utf-8') if isinstance(value, bytes) else str(value)

    target = (
        str(analysis.exp_name), str(analysis.cell_label),
        str(analysis.online_analysis).strip().lower(), str(analysis.site).strip().lower(),
        float(analysis.filter_wheel_ndf),
    )
    matches = []
    seen = set()
    for directory in _condition_read_directories(output_dir, analysis.protocols):
        if not directory.exists():
            continue
        for path in sorted((*directory.glob('*.h5'), *directory.glob('*.csv'))):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                if path.suffix.lower() == '.h5':
                    with h5py.File(path, 'r') as saved:
                        identity = (
                            _text(saved.attrs['exp_name']), _text(saved.attrs['cell_label']),
                            _text(saved.attrs['online_analysis']).strip().lower(),
                            _text(saved.attrs['site']).strip().lower(),
                            float(saved.attrs['filter_wheel_ndf']),
                        )
                else:
                    row = pd.read_csv(path, nrows=1).iloc[0]
                    identity = (
                        str(row['date']), str(row['cell_label']),
                        str(row['onlineAnalysis']).strip().lower(),
                        str(row['site']).strip().lower(), float(row['filter_wheel_ndf']),
                    )
            except (OSError, ValueError, KeyError, IndexError):
                continue
            if identity[:4] == target[:4] and np.isclose(identity[4], target[4]):
                matches.append(path)
    return matches


def save_condition_output(analysis: ConditionAnalysis, output_dir=None,
                          verbose: bool = True, remove_duplicates: bool = True):
    """Idempotently save one cell condition as compressed typed arrays.

    By default, center-disc and annulus-disc conditions are routed to separate
    ``center_disc`` and ``annulus_disc`` directories. An explicit
    ``output_dir`` is used as-is.

    The file is deliberately organized like a MATLAB struct containing arrays:
    scalar condition metadata live in HDF5 attributes, while ``image_summary``
    and ``patch_responses`` are groups of column datasets.  A flat population
    table is only materialized when requested, avoiding a large text CSV on
    disk while retaining the same public loading API.
    """
    import h5py
    from pathlib import Path

    directory = (Path(output_dir) if output_dir is not None
                 else condition_output_dir(analysis.protocols))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _condition_output_name(analysis)
    temporary = path.with_suffix(path.suffix + '.tmp')
    existing = (matching_condition_outputs(analysis, output_dir=output_dir)
                if remove_duplicates else [])
    if verbose:
        label = (f'{analysis.exp_name}/{analysis.cell_label} '
                 f'({analysis.online_analysis}, {analysis.site}, '
                 f'FW{analysis.filter_wheel_ndf:g})')
        if existing:
            print(f'ALERT: found {len(existing)} saved copy/copies of {label}:')
            for existing_path in existing:
                print(f'  {existing_path}')
        else:
            print(f'No existing saved copy found for {label}.')

    with h5py.File(temporary, 'w') as h5:
        h5.attrs['output_version'] = CONDITION_OUTPUT_VERSION
        h5.attrs['exp_name'] = analysis.exp_name
        h5.attrs['cell_label'] = analysis.cell_label
        h5.attrs['cell_type'] = analysis.cell_type
        h5.attrs['online_analysis'] = analysis.online_analysis
        h5.attrs['filter_wheel_ndf'] = analysis.filter_wheel_ndf
        h5.attrs['protocols'] = '\n'.join(analysis.protocols)
        h5.attrs['site'] = analysis.site
        h5.attrs['units'] = analysis.units
        h5.attrs['threshold'] = analysis.threshold
        h5.create_dataset('block_ids', data=np.asarray(analysis.block_ids, dtype=np.int64))
        _write_condition_frame(h5.create_group('image_summary'), analysis.image_summary)
        _write_condition_frame(h5.create_group('patch_responses'), analysis.patch_responses)
    temporary.replace(path)

    removed = []
    failed = []
    for duplicate in existing:
        if duplicate.resolve() == path.resolve():
            continue
        try:
            duplicate.unlink()
            removed.append(duplicate)
        except OSError as error:
            failed.append((duplicate, error))
    if verbose:
        if any(existing_path.resolve() == path.resolve()
               for existing_path in existing):
            print(f'ALERT: replaced the existing canonical saved condition at {path}')
        if removed:
            print(f'ALERT: removed {len(removed)} duplicate saved copy/copies:')
            for duplicate in removed:
                print(f'  {duplicate}')
        for duplicate, error in failed:
            print(f'WARNING: could not remove duplicate {duplicate}: {error}')
        print(f'saved 1 condition ({len(analysis.patch_responses)} patches as arrays) to {path}')
    return path


def _write_condition_frame(group, frame: pd.DataFrame):
    """Write a DataFrame as compressed HDF5 column arrays."""
    import h5py

    group.attrs['columns'] = '\n'.join(frame.columns)
    for column in frame.columns:
        values = frame[column]
        if column == 'block_ids':
            array = np.asarray([
                ','.join(str(int(value)) for value in item)
                if isinstance(item, (list, tuple, np.ndarray)) else str(item)
                for item in values
            ], dtype=object)
        elif pd.api.types.is_numeric_dtype(values.dtype):
            array = values.to_numpy()
        else:
            array = np.asarray(['' if pd.isna(value) else str(value)
                                for value in values], dtype=object)
        options = dict(compression='gzip', shuffle=True) if len(array) else {}
        if array.dtype.kind in 'OUS':
            group.create_dataset(column, data=array,
                                 dtype=h5py.string_dtype(encoding='utf-8'), **options)
        else:
            group.create_dataset(column, data=array, **options)


def _read_condition_frame(group) -> pd.DataFrame:
    """Read a DataFrame written by :func:`_write_condition_frame`."""
    columns_attr = group.attrs.get('columns', '')
    if isinstance(columns_attr, bytes):
        columns_attr = columns_attr.decode()
    columns = str(columns_attr).split('\n') if columns_attr else list(group)
    data = {}
    for column in columns:
        values = group[column][()]
        if values.dtype.kind in 'SO':
            values = np.asarray([value.decode() if isinstance(value, bytes) else str(value)
                                 for value in values], dtype=object)
        data[column] = values
    frame = pd.DataFrame(data, columns=columns)
    if 'block_ids' in frame:
        frame['block_ids'] = frame['block_ids'].apply(
            lambda item: [int(value) for value in str(item).split(',') if value])
    return frame


def _normalize_saved_condition_intensities(frame: pd.DataFrame, path) -> pd.DataFrame:
    """Temporarily resolve comma-joined intensity metadata to its larger value.

    A small number of saved blocks contain two typed ``maxIntensity`` values
    for one FilterWheel condition. ``condition_image_summary`` preserves that
    conflict as a comma-joined string, but population analysis requires one
    numeric x value. Prefer the larger candidate and warn so the source data
    remains visibly abnormal until it can be corrected upstream.
    """
    import warnings

    normalized = frame.copy()
    for column in ('maxIntensity', 'meanIntensity'):
        if column in normalized:
            normalized[column] = normalized[column].astype(object)
    for row_index, row in normalized.iterrows():
        corrections = []
        for column in ('maxIntensity', 'meanIntensity'):
            value = row.get(column, np.nan)
            if not isinstance(value, str) or ',' not in value:
                continue
            candidates = pd.to_numeric(
                pd.Series([part.strip() for part in value.split(',')]),
                errors='coerce').dropna().to_numpy(dtype=float)
            if candidates.size < 2:
                continue
            selected = float(np.max(candidates))
            normalized.at[row_index, column] = selected
            corrections.append(f'{column}={value!r} -> {selected:g}')
        if corrections:
            image_name = row.get('imageName', '?')
            warnings.warn(
                f'{path}: imageName {image_name}: conflicting saved intensity '
                f'metadata; using larger value ({"; ".join(corrections)})',
                RuntimeWarning, stacklevel=2)
    for column in ('maxIntensity', 'meanIntensity'):
        if column in normalized:
            normalized[column] = pd.to_numeric(normalized[column], errors='raise')
    return normalized


def _read_condition_h5(path) -> ConditionAnalysis:
    """Read one complete condition record without DataJoint or raw-data access."""
    import h5py

    def text(value):
        return value.decode() if isinstance(value, bytes) else str(value)

    with h5py.File(path, 'r') as h5:
        if int(h5.attrs.get('output_version', -1)) != CONDITION_OUTPUT_VERSION:
            raise ValueError('condition output version does not match')
        image_summary = _normalize_saved_condition_intensities(
            _read_condition_frame(h5['image_summary']), path)
        patch_responses = _read_condition_frame(h5['patch_responses'])
        protocols = text(h5.attrs['protocols']).split('\n')
        return ConditionAnalysis(
            exp_name=text(h5.attrs['exp_name']),
            cell_label=text(h5.attrs['cell_label']),
            cell_type=text(h5.attrs['cell_type']),
            online_analysis=text(h5.attrs['online_analysis']),
            filter_wheel_ndf=float(h5.attrs['filter_wheel_ndf']),
            block_ids=[int(value) for value in h5['block_ids'][()]],
            protocols=protocols, site=text(h5.attrs['site']),
            image_summary=image_summary, epoch_responses=pd.DataFrame(),
            patch_responses=patch_responses, units=text(h5.attrs['units']),
            threshold=float(h5.attrs['threshold']), loaded_from_saved=True)


def _read_condition_csv(path) -> pd.DataFrame:
    text_columns = {'date': str, 'cell_label': str, 'imageName': str,
                    'patch_key': str, 'block_ids': str}
    return pd.read_csv(path, dtype=text_columns)


def load_condition_output(blocks: pd.DataFrame, output_dir=None,
                          verbose: bool = True) -> Optional[ConditionAnalysis]:
    """Rehydrate a matching saved condition, or return ``None`` if stale/missing."""
    if blocks.empty:
        return None
    exp_names = blocks['exp_name'].astype(str).unique()
    cell_labels = blocks['cell_label'].astype(str).unique()
    modes = blocks['onlineAnalysis'].astype(str).str.strip().str.lower().unique()
    wheels = pd.to_numeric(blocks['filter_wheel_ndf'], errors='coerce').unique()
    sites = blocks['site'].astype(str).unique()
    if any(len(values) != 1 for values in (exp_names, cell_labels, modes, wheels, sites)):
        return None
    protocols = sorted({str(value) for value in blocks['protocol']})
    output_name = _condition_output_name_from_values(
        exp_names[0], cell_labels[0], modes[0], sites[0], protocols, float(wheels[0]))
    directories = _condition_read_directories(output_dir, protocols)
    for path in (directory / output_name for directory in directories):
        if not path.exists():
            continue
        try:
            analysis = _read_condition_h5(path)
            if set(analysis.block_ids) != {int(value) for value in blocks['block_id']}:
                continue
        except (OSError, ValueError, KeyError):
            continue
        if verbose:
            print(f'loaded {len(analysis.patch_responses)} saved patch arrays from {path}')
        return analysis

    # Read pre-HDF5 outputs so existing work remains usable. A subsequent save
    # writes the compact format and takes precedence in population loading.
    legacy_name = _condition_output_name_from_values(
        exp_names[0], cell_labels[0], modes[0], sites[0], protocols,
        float(wheels[0]), suffix='.csv')
    path = next((directory / legacy_name for directory in directories
                 if (directory / legacy_name).exists()), None)
    if path is None:
        return None
    try:
        table = _read_condition_csv(path)
        if ('output_version' not in table
                or not table['output_version'].eq(CONDITION_OUTPUT_VERSION).all()):
            return None
        saved_blocks = {int(value) for values in table['block_ids'].dropna()
                        for value in str(values).split(',') if value}
        expected_blocks = {int(value) for value in blocks['block_id']}
        if saved_blocks != expected_blocks:
            return None
    except (OSError, ValueError, KeyError):
        return None

    patch_responses = table.rename(columns={
        'image_response': 'image_mean', 'image_response_sem': 'image_sem',
        'image_trials': 'image_n', 'disc_response': 'disc_mean',
        'disc_response_sem': 'disc_sem', 'disc_trials': 'disc_n',
        'cone_disc_response': 'cone_disc_mean',
        'cone_disc_response_sem': 'cone_disc_sem',
        'cone_disc_trials': 'cone_disc_n',
        'nli_image_vs_disc': 'nli_disc',
        'nli_image_vs_cone_disc': 'nli_cone_disc',
    })
    patch_columns = [
        'imageName', 'patchIndex', 'image_mean', 'image_sem', 'image_n',
        'disc_mean', 'disc_sem', 'disc_n', 'cone_disc_mean', 'cone_disc_sem',
        'cone_disc_n', 'nli_disc', 'nli_cone_disc', 'patch_key',
    ]
    patch_responses = patch_responses[patch_columns].copy()

    image_summary = (table.groupby('imageName', sort=True)
                     .agg(block_ids=('block_ids', 'first'),
                          epochs=('image_epochs', 'first'),
                          maxIntensity=('maxIntensity', 'first'),
                          backgroundIntensity=('backgroundIntensity', 'first'),
                          meanIntensity=('meanIntensity', 'first'))
                     .reset_index())
    image_summary['block_ids'] = image_summary['block_ids'].apply(
        lambda values: [int(value) for value in str(values).split(',') if value])
    cell_type_column = ('cell_type_short' if 'cell_type_short' in blocks
                        else 'cell_type')
    analysis = ConditionAnalysis(
        exp_name=exp_names[0], cell_label=cell_labels[0],
        cell_type=str(blocks[cell_type_column].iloc[0]),
        online_analysis=modes[0], filter_wheel_ndf=float(wheels[0]),
        block_ids=sorted(saved_blocks), protocols=protocols, site=sites[0],
        image_summary=image_summary, epoch_responses=pd.DataFrame(),
        patch_responses=patch_responses,
        units=str(table['response_units'].iloc[0]),
        threshold=float(NLI_THRESHOLD[modes[0]]), loaded_from_saved=True)
    if verbose:
        print(f'loaded {len(patch_responses)} saved patch rows from {path}')
    return analysis


def load_condition_outputs(
        output_dir=None,
        protocol: Optional[Union[str, Sequence[str]]] = None,
        ) -> pd.DataFrame:
    """Expand all saved condition arrays into one population patch table.

    Legacy CSVs are included only when the same condition has not yet been
    resaved as HDF5.
    """
    h5_paths = _condition_paths(output_dir, protocol, '.h5')
    h5_stems = {path.stem for path in h5_paths}
    legacy_paths = [path for path in _condition_paths(output_dir, protocol, '.csv')
                    if path.stem not in h5_stems]
    analyses = [_read_condition_h5(path) for path in h5_paths]
    tables = [condition_population_table(analysis) for analysis in analyses
              if _saved_condition_matches_protocol(analysis.protocols, protocol)]
    for path in legacy_paths:
        table = _read_condition_csv(path)
        if (protocol is None or ('protocol' in table and not table.empty
                and _saved_condition_matches_protocol(
                    str(table.iloc[0]['protocol']).split(', '), protocol))):
            tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


PATCH_VARIANCE_POPULATION_COLUMNS = [
    'date', 'cell_label', 'cell_type', 'onlineAnalysis', 'protocol', 'site',
    'filter_wheel_ndf', 'imageName', 'patch_uid', 'patchIndex',
    'patch_x_vh', 'patch_y_vh', 'currentStimSet', 'currentImageSet',
    'source_protocol_name', 'library_path', 'image_path', 'source_condition_h5',
    'source_block_ids', 'noPatches_values',
    'seed_values', 'patchSampling_values', 'patchContrast_values',
    'canvas_width_px', 'canvas_height_px', 'micronsPerPixel',
    'apertureDiameter_um', 'annulusInnerDiameter_um',
    'annulusOuterDiameter_um', 'rfSigmaCenter_um', 'rfSigmaSurround_um',
    'apertureDiameter_um_values', 'annulusInnerDiameter_um_values',
    'annulusOuterDiameter_um_values',
    'rfSigmaCenter_um_values', 'rfSigmaSurround_um_values',
    'linearIntegrationFunction', 'WeberConstant',
    'maxIntensity', 'imageMeanIntensity', 'meanIntensity_Rstar_per_s',
    'backgroundIntensity', 'meanIntensity',
    'patchMean', 'patchMeanContrast', 'patchMeanIntensity', 'patchVariance',
    'patchRmsContrast',
    'equivalentIntensity',
    'equivalentIntensity_values', 'equivalentIntensity_recorded_values',
    'equivalentIntensityConeLin', 'equivalentIntensityConeLin_values',
    'equivalentIntensityConeLin_recorded_values',
    'cone_equivalent_metadata_corrected', 'equivalentIntensity_Rstar_per_s',
    'equivalentIntensityConeLin_Rstar_per_s', 'response_units',
    'image_response', 'image_response_sem', 'image_trials',
    'disc_response', 'disc_response_sem', 'disc_trials',
    'cone_disc_response', 'cone_disc_response_sem', 'cone_disc_trials',
    'nli_image_vs_disc', 'nli_image_vs_cone_disc', 'delta_nli_cone_minus_disc',
    'complete_response_triplet', 'stimulus_metadata_mixed_fields',
    'analysis_ready', 'exclusion_reason',
]


def patch_rms_contrast(patch_mean_contrast, patch_variance,
                       n_pixels: int = NATURAL_IMAGE_PATCH_PIXEL_COUNT):
    """Convert stored sample variance to patch-local RMS contrast.

    The natural-image library stores MATLAB ``var(ImagePatch(:))`` after the
    image has been normalized to whole-image Weber contrast. Its fixed patch is
    38 x 38 pixels. Correct the sample variance to population variance and
    renormalize by the patch mean intensity, ``1 + patch_mean_contrast``.
    Invalid negative variances or non-positive patch means return NaN.
    """
    if int(n_pixels) != n_pixels or n_pixels <= 1:
        raise ValueError('n_pixels must be an integer greater than one')
    mean = np.asarray(patch_mean_contrast, dtype=float)
    variance = np.asarray(patch_variance, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        rms = np.sqrt(variance * ((n_pixels - 1) / n_pixels)) / (1 + mean)
    invalid = (~np.isfinite(mean) | ~np.isfinite(variance)
               | (variance < 0) | (mean <= -1))
    rms = np.where(invalid, np.nan, rms)
    return float(rms) if rms.ndim == 0 else rms


def _natural_image_library_path(protocol_name: str, stimulus_set: str):
    """Resolve the exact package resource used by a recorded protocol."""
    from pathlib import Path
    from retinanalysis.utils.protocol_source import protocol_source_path

    source = protocol_source_path(str(protocol_name))
    if source is None:
        raise FileNotFoundError(
            f'no local MATLAB source was found for {protocol_name!r}')
    path = Path(source).parent.parent / '+resources' / f'{stimulus_set}.mat'
    if not path.is_file():
        raise FileNotFoundError(
            f'{protocol_name!r} points to {path.parent}, but '
            f'{stimulus_set}.mat is missing')
    return path


@lru_cache(maxsize=None)
def _load_natural_image_library(path_string: str):
    """Load and cache one NaturalImageFlash MATLAB metadata library."""
    from scipy.io import loadmat

    contents = loadmat(path_string, simplify_cells=True)
    if 'imageData' not in contents:
        raise KeyError(f'{path_string} has no imageData variable')
    return contents['imageData']


def _natural_image_patch_library_row(protocol_name: str, stimulus_set: str,
                                     image_name: str, patch_location) -> Dict:
    """Patch mean/variance for one exact recorded natural-image location."""
    path = _natural_image_library_path(protocol_name, stimulus_set)
    image_data = _load_natural_image_library(str(path))
    field = f'imk{str(image_name).strip()}'
    if field not in image_data:
        raise KeyError(f'{path.name} has no {field} entry')
    record = image_data[field]
    locations = np.asarray(record['location'], dtype=float)
    target = np.asarray(patch_location, dtype=float).reshape(-1)[:2]
    matched = np.flatnonzero(np.all(np.isclose(locations, target), axis=1))
    if matched.size == 0:
        raise ValueError(
            f'{field} location {target.tolist()} is absent from {path.name}')
    # Some historical libraries contain the same physical location twice. It
    # is still one patch when all stored patch statistics agree.
    means = np.asarray(record['PatchMean']).reshape(-1)[matched].astype(float)
    variances = np.asarray(
        record['PatchVariance']).reshape(-1)[matched].astype(float)
    if (not np.allclose(means, means[0], equal_nan=True)
            or not np.allclose(variances, variances[0], equal_nan=True)):
        raise ValueError(
            f'{field} location {target.tolist()} has {matched.size} conflicting '
            f'rows in {path.name}')
    index = int(matched[0])
    return {
        'patchMean': float(np.asarray(record['PatchMean']).reshape(-1)[index]),
        'patchVariance': float(
            np.asarray(record['PatchVariance']).reshape(-1)[index]),
        'library_path': str(path),
    }


def _joined_metadata_values(values) -> str:
    """Stable comma-separated provenance for metadata that can vary by block."""
    unique = []
    for value in values:
        if isinstance(value, (list, tuple, np.ndarray)):
            value = tuple(np.asarray(value).reshape(-1).tolist())
        if not isinstance(value, tuple) and pd.isna(value):
            continue
        text = str(value)
        if text not in unique:
            unique.append(text)
    return ','.join(unique)


def _single_numeric(values, label: str, patch_key: str) -> float:
    """Return one repeated numeric value, rejecting inconsistent metadata."""
    numeric = pd.to_numeric(pd.Series(list(values)), errors='coerce').dropna()
    if numeric.empty:
        return np.nan
    candidates = numeric.to_numpy(dtype=float)
    if not np.allclose(candidates, candidates[0], rtol=1e-7, atol=1e-10):
        raise ValueError(
            f'{patch_key}: recorded {label} has multiple values '
            f'{np.unique(candidates).tolist()}')
    return float(candidates[0])


@lru_cache(maxsize=None)
def _natural_image_weighting(canvas_width: float, canvas_height: float,
                             microns_per_pixel: float, outer_diameter: float,
                             inner_diameter: float, rf_sigma: float,
                             integration_function: str):
    """Protocol-exact RF/aperture weights in van Hateren pixel space."""
    rad_x, rad_y = np.floor(
        np.asarray([canvas_width, canvas_height], dtype=float)
        * float(microns_per_pixel) / VH_MICRONS_PER_PIXEL / 2).astype(int)
    if rad_x <= 0 or rad_y <= 0:
        raise ValueError('recorded canvas geometry produces an empty image patch')

    # MATLAB fspecial('gaussian', 2.*[radX radY], sigma) uses half-pixel
    # centers for these even-sized arrays.
    x = np.arange(2 * rad_x, dtype=float) - (2 * rad_x - 1) / 2
    y = np.arange(2 * rad_y, dtype=float) - (2 * rad_y - 1) / 2
    yy, xx = np.meshgrid(y, x)
    sigma_vh = float(rf_sigma) / VH_MICRONS_PER_PIXEL
    gaussian = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_vh ** 2))
    gaussian[gaussian < np.finfo(float).eps * gaussian.max()] = 0
    gaussian /= gaussian.sum()

    rr, cc = np.meshgrid(
        np.arange(1, 2 * rad_x + 1, dtype=float),
        np.arange(1, 2 * rad_y + 1, dtype=float))
    radius = np.sqrt((rr - rad_x) ** 2 + (cc - rad_y) ** 2).T
    aperture = radius < float(outer_diameter) / 2 / VH_MICRONS_PER_PIXEL
    if float(inner_diameter) > 0:
        aperture &= radius > float(inner_diameter) / 2 / VH_MICRONS_PER_PIXEL

    mode = str(integration_function).strip().lower()
    if mode == 'gaussian':
        weighting = aperture * gaussian
    elif mode == 'uniform':
        weighting = aperture.astype(float)
    else:
        raise ValueError(f'unknown linearIntegrationFunction {integration_function!r}')
    total = float(weighting.sum())
    if total <= 0:
        raise ValueError('recorded aperture produces zero integration weight')
    return weighting / total, int(rad_x), int(rad_y)


def equivalent_intensities_from_epoch(
        params: Dict, image_path=None) -> Tuple[float, float]:
    """Reconstruct both displayed disc intensities from recorded parameters.

    This mirrors ``NaturalImageFlashProtocol.getEquivalentIntensityValues`` and
    ``getEquivalentIntensityValuesConeLin``. It is necessary for the older
    Turner ``LinearEquivalentDisc`` protocol, whose ``prepareEpoch`` displayed
    ``equivalentIntensityConeLin`` correctly but accidentally saved
    ``equivalentIntensity`` under that epoch-parameter name.
    """
    image_name = str(params['imageName']).strip()
    image_set = str(params.get(
        'currentImageSet', 'VHsubsample_20160105')).strip().lstrip('/')
    contrast, _ = (_load_vh_contrast_image_path(str(image_path))
                   if image_path is not None
                   else load_vh_contrast_image(image_name, image_set))
    if contrast is None:
        raise FileNotFoundError(
            f'cannot reconstruct equivalent intensities: imk{image_name}.iml '
            f'was not found in {image_set}')
    canvas = np.asarray(params['canvasSize'], dtype=float).reshape(-1)
    if canvas.size < 2:
        raise ValueError('recorded canvasSize must contain width and height')
    inner = float(params.get('annulusInnerDiameter') or 0.0)
    outer = float(params.get('apertureDiameter')
                  or params.get('annulusOuterDiameter') or 0.0)
    sigma_key = 'rfSigmaSurround' if inner > 0 else 'rfSigmaCenter'
    weighting, rad_x, rad_y = _natural_image_weighting(
        float(canvas[0]), float(canvas[1]), float(params['micronsPerPixel']),
        outer, inner, float(params[sigma_key]),
        str(params.get('linearIntegrationFunction', 'gaussian')))

    x, y = (int(round(value)) for value in
            np.asarray(params['currentPatchLocation'], dtype=float)[:2])
    patch = contrast[x - rad_x:x + rad_x, y - rad_y:y + rad_y]
    if patch.shape != weighting.shape:
        raise ValueError(
            f'imk{image_name} location {[x, y]} produced patch shape '
            f'{patch.shape}; expected {weighting.shape}')

    background = float(params['backgroundIntensity'])
    equivalent_contrast = float(np.sum(weighting * patch))
    equivalent = background * (1 + equivalent_contrast)

    maximum = float(params['maxIntensity'])
    weber = float(params['WeberConstant'])
    patch_isoms = (patch + 1) * background * maximum
    rf_factor = float(np.sum(
        weighting * (patch_isoms - background * maximum)
        / (1 + patch_isoms / weber)))
    cone_contrast = (
        rf_factor * (1 + background * maximum / weber)
        / (1 - rf_factor / weber)
        / (background * maximum))
    cone_equivalent = background * (1 + cone_contrast)
    return float(equivalent), float(cone_equivalent)


def _condition_patch_stimulus_metadata(analysis: ConditionAnalysis) -> pd.DataFrame:
    """Recover physical patch identity and stimulus values from raw epochs.

    Saved responses are indexed by the protocol's run-local ``patchIndex``.
    This function reopens the blocks named in the saved HDF5 and translates
    that ordinal into the stable identity ``(stimulus library, image, x, y)``.
    """
    import contextlib
    import io
    from pathlib import Path
    import retinanalysis as ra

    rows = []
    for block_id in analysis.block_ids:
        # StimBlock can emit frame-monitor diagnostics even though this bridge
        # only reads epoch parameters. Keep the long population build quiet;
        # missing or inconsistent metadata below still raises explicitly.
        with (contextlib.redirect_stdout(io.StringIO()),
              contextlib.redirect_stderr(io.StringIO())):
            stimulus_block = ra.StimBlock(
                analysis.exp_name, int(block_id), verbose=False)
        protocol_name = str(stimulus_block.protocol_name)
        seen_patches = set()
        for params in stimulus_block.df_epochs['epoch_parameters']:
            if category_of(params.get('stimulusTag')) != 'image':
                continue
            patch_index = pd.to_numeric(
                params.get('imagePatchIndex', params.get('patchIndex')),
                errors='coerce')
            location = np.asarray(
                params.get('currentPatchLocation', []), dtype=float).reshape(-1)
            if not np.isfinite(patch_index) or location.size < 2:
                continue
            image_name = str(params.get('imageName', '')).strip()
            stimulus_set = str(params.get('currentStimSet', '')).strip()
            if not image_name or not stimulus_set:
                continue
            x, y = float(location[0]), float(location[1])
            epoch_patch_key = (image_name, float(patch_index), x, y)
            if epoch_patch_key in seen_patches:
                continue
            seen_patches.add(epoch_patch_key)
            library = _natural_image_patch_library_row(
                protocol_name, stimulus_set, image_name, (x, y))
            image_set = str(
                params.get('currentImageSet', '')).strip().lstrip('/')
            image_file = (Path(library['library_path']).parent / image_set
                          / f'imk{image_name}.iml')
            if not image_file.is_file():
                image_file = vh_image_path(image_name, image_set)
            equivalent, cone_equivalent = equivalent_intensities_from_epoch(
                params, image_path=image_file)
            canvas = np.asarray(
                params.get('canvasSize', [np.nan, np.nan]),
                dtype=float).reshape(-1)
            patch_uid = f'{stimulus_set}:imk{image_name}:x{x:g}:y{y:g}'
            rows.append({
                'imageName': image_name,
                'patchIndex': float(patch_index),
                'patch_uid': patch_uid,
                'patch_x_vh': x,
                'patch_y_vh': y,
                'currentStimSet': stimulus_set,
                'currentImageSet': str(
                    params.get('currentImageSet', '')).strip().lstrip('/'),
                'source_protocol_name': protocol_name,
                'image_path': str(image_file) if image_file is not None else '',
                'source_block_id': int(block_id),
                'noPatches': params.get('noPatches', np.nan),
                'seed': params.get('seed', np.nan),
                'patchSampling': params.get('patchSampling', ''),
                'patchContrast': params.get('patchContrast', ''),
                'canvas_width_px': canvas[0] if canvas.size > 0 else np.nan,
                'canvas_height_px': canvas[1] if canvas.size > 1 else np.nan,
                'micronsPerPixel': params.get('micronsPerPixel', np.nan),
                'apertureDiameter_um': params.get('apertureDiameter', np.nan),
                'annulusInnerDiameter_um': params.get(
                    'annulusInnerDiameter', np.nan),
                'annulusOuterDiameter_um': params.get(
                    'annulusOuterDiameter', np.nan),
                'rfSigmaCenter_um': params.get('rfSigmaCenter', np.nan),
                'rfSigmaSurround_um': params.get('rfSigmaSurround', np.nan),
                'linearIntegrationFunction': params.get(
                    'linearIntegrationFunction', ''),
                'WeberConstant': params.get('WeberConstant', np.nan),
                'equivalentIntensity': equivalent,
                'equivalentIntensity_recorded': params.get(
                    'equivalentIntensity', np.nan),
                'equivalentIntensityConeLin': cone_equivalent,
                'equivalentIntensityConeLin_recorded': params.get(
                    'equivalentIntensityConeLin', np.nan),
                **library,
            })
    if not rows:
        raise ValueError(
            f'{analysis.exp_name}/{analysis.cell_label}: no image epochs carried '
            'the patch location and stimulus-library metadata')

    epochs = pd.DataFrame(rows).drop_duplicates()
    key_columns = ['imageName', 'patchIndex']
    identity_counts = epochs.groupby(key_columns)['patch_uid'].nunique()
    collisions = identity_counts.loc[identity_counts.gt(1)]
    if not collisions.empty:
        examples = ', '.join(
            f'{image}:{patch:g}' for image, patch in collisions.index[:8])
        raise ValueError(
            f'{analysis.exp_name}/{analysis.cell_label}: saved response keys '
            f'map to multiple physical locations ({examples}). The saved means '
            'are already mixed and must be recomputed from per-epoch responses.')

    collapsed = []
    for (image_name, patch_index), group in epochs.groupby(key_columns, sort=False):
        patch_key = f'{image_name}:{patch_index:g}'
        mixed_fields = []
        for column in ('canvas_width_px', 'canvas_height_px', 'micronsPerPixel',
                       'apertureDiameter_um', 'annulusInnerDiameter_um',
                       'annulusOuterDiameter_um', 'rfSigmaCenter_um',
                       'rfSigmaSurround_um', 'WeberConstant'):
            values = pd.to_numeric(group[column], errors='coerce').dropna()
            if values.size and not np.allclose(
                    values.to_numpy(float), float(values.iloc[0]),
                    rtol=1e-7, atol=1e-10):
                mixed_fields.append(column)
        fixed_text = {}
        for column in ('patch_uid', 'currentStimSet', 'currentImageSet',
                       'source_protocol_name', 'library_path',
                       'image_path', 'linearIntegrationFunction'):
            values = [str(value) for value in group[column] if str(value)]
            if len(set(values)) > 1:
                raise ValueError(
                    f'{patch_key}: {column} has multiple values {sorted(set(values))}')
            fixed_text[column] = values[0] if values else ''
        collapsed.append({
            'imageName': image_name,
            'patchIndex': float(patch_index),
            **fixed_text,
            'patch_x_vh': _single_numeric(group['patch_x_vh'], 'patch_x_vh', patch_key),
            'patch_y_vh': _single_numeric(group['patch_y_vh'], 'patch_y_vh', patch_key),
            'patchMean': _single_numeric(group['patchMean'], 'patchMean', patch_key),
            'patchVariance': _single_numeric(
                group['patchVariance'], 'patchVariance', patch_key),
            'canvas_width_px': _single_numeric(
                group['canvas_width_px'], 'canvas_width_px', patch_key),
            'canvas_height_px': _single_numeric(
                group['canvas_height_px'], 'canvas_height_px', patch_key),
            'micronsPerPixel': _single_numeric(
                group['micronsPerPixel'], 'micronsPerPixel', patch_key),
            'apertureDiameter_um': float(pd.to_numeric(
                group['apertureDiameter_um'], errors='coerce').mean()),
            'annulusInnerDiameter_um': float(pd.to_numeric(
                group['annulusInnerDiameter_um'], errors='coerce').mean()),
            'annulusOuterDiameter_um': float(pd.to_numeric(
                group['annulusOuterDiameter_um'], errors='coerce').mean()),
            'apertureDiameter_um_values': _joined_metadata_values(
                group['apertureDiameter_um']),
            'annulusInnerDiameter_um_values': _joined_metadata_values(
                group['annulusInnerDiameter_um']),
            'annulusOuterDiameter_um_values': _joined_metadata_values(
                group['annulusOuterDiameter_um']),
            'rfSigmaCenter_um': float(pd.to_numeric(
                group['rfSigmaCenter_um'], errors='coerce').mean()),
            'rfSigmaSurround_um': float(pd.to_numeric(
                group['rfSigmaSurround_um'], errors='coerce').mean()),
            'rfSigmaCenter_um_values': _joined_metadata_values(
                group['rfSigmaCenter_um']),
            'rfSigmaSurround_um_values': _joined_metadata_values(
                group['rfSigmaSurround_um']),
            'WeberConstant': _single_numeric(
                group['WeberConstant'], 'WeberConstant', patch_key),
            # A physical patch can be repeated in blocks whose recorded
            # calibration differs slightly. Responses in the saved file are
            # already pooled across those blocks, so retain every exact value
            # as provenance and use their mean as the representative value.
            'equivalentIntensity': float(pd.to_numeric(
                group['equivalentIntensity'], errors='coerce').mean()),
            'equivalentIntensity_values': _joined_metadata_values(
                group['equivalentIntensity']),
            'equivalentIntensity_recorded_values': _joined_metadata_values(
                group['equivalentIntensity_recorded']),
            'equivalentIntensityConeLin': float(pd.to_numeric(
                group['equivalentIntensityConeLin'], errors='coerce').mean()),
            'equivalentIntensityConeLin_values': _joined_metadata_values(
                group['equivalentIntensityConeLin']),
            'equivalentIntensityConeLin_recorded_values': _joined_metadata_values(
                group['equivalentIntensityConeLin_recorded']),
            'cone_equivalent_metadata_corrected': bool(np.any(~np.isclose(
                pd.to_numeric(group['equivalentIntensityConeLin'], errors='coerce'),
                pd.to_numeric(
                    group['equivalentIntensityConeLin_recorded'], errors='coerce'),
                rtol=1e-7, atol=1e-10, equal_nan=True))),
            'source_block_ids': _joined_metadata_values(
                sorted(group['source_block_id'].unique())),
            'noPatches_values': _joined_metadata_values(group['noPatches']),
            'seed_values': _joined_metadata_values(group['seed']),
            'patchSampling_values': _joined_metadata_values(group['patchSampling']),
            'patchContrast_values': _joined_metadata_values(group['patchContrast']),
            'stimulus_metadata_mixed_fields': ','.join(mixed_fields),
        })
    return pd.DataFrame(collapsed)


def build_center_disc_patch_variance_population(
        output_dir=None, filter_wheel_ndf: Optional[float] = None,
        online_analysis: Optional[str] = None,
        require_analysis_ready: bool = True,
        return_qc: bool = False,
        verbose: bool = True):
    """Enrich saved center-disc responses with physical patch metadata.

    This is a temporary bridge for the spatial-contrast analysis. Response
    values come from the compact saved condition HDF5 files; patch locations,
    equivalent intensities, and protocol settings are recovered from the raw
    epoch metadata. ``PatchMean`` and ``PatchVariance`` are then looked up in
    the exact NaturalImageFlash library belonging to the recorded protocol.

    One output row is one cell/condition/physical patch. By default all
    FilterWheel settings are retained. Incomplete image/disc/cone-disc
    triplets and rows whose saved response pooled conflicting stimulus geometry
    are excluded. Pass a numeric ``filter_wheel_ndf`` only for an explicitly
    restricted rebuild. With ``return_qc=True``, return ``(population,
    excluded_rows)`` so rejected rows remain auditable.
    """
    import h5py

    protocols = ('LinearEquivalentDiscConeLin', 'LinearEquivalentDisc')
    rows = []
    excluded_rows = []
    matched_conditions = 0
    for path in _condition_paths(output_dir, protocols, '.h5'):
        # Filter from scalar attributes before expanding arrays. Besides being
        # faster, this prevents irrelevant FW1 intensity warnings in an FW0 run.
        with h5py.File(path, 'r') as saved:
            saved_fw = float(saved.attrs.get('filter_wheel_ndf', np.nan))
            saved_protocols = str(saved.attrs.get('protocols', '')).split('\n')
            saved_site = str(saved.attrs.get('site', '')).strip().lower()
        wrong_filter = (filter_wheel_ndf is not None
                        and not np.isclose(saved_fw, float(filter_wheel_ndf)))
        if (wrong_filter or saved_site != 'center'
                or not _saved_condition_matches_protocol(
                    saved_protocols, protocols)):
            continue
        analysis = _read_condition_h5(path)
        if (online_analysis is not None
                and analysis.online_analysis.strip().lower()
                != str(online_analysis).strip().lower()):
            continue
        matched_conditions += 1
        stimulus = _condition_patch_stimulus_metadata(analysis)
        population = condition_population_table(analysis)
        enriched = population.merge(
            stimulus, on=['imageName', 'patchIndex'], how='left',
            validate='one_to_one')
        missing = enriched['patch_uid'].isna()
        if missing.any():
            keys = enriched.loc[missing, 'patch_key'].astype(str).tolist()
            raise ValueError(
                f'{path.name}: {len(keys)} saved response rows have no matching '
                f'raw patch metadata: {keys[:8]}')
        enriched['equivalentIntensity_Rstar_per_s'] = (
            enriched['maxIntensity'] * enriched['equivalentIntensity'])
        enriched['equivalentIntensityConeLin_Rstar_per_s'] = (
            enriched['maxIntensity'] * enriched['equivalentIntensityConeLin'])
        # Explicit names prevent the protocol's normalized [0, 1] image mean
        # from being confused with its calibrated mean light level in R*/s.
        enriched['imageMeanIntensity'] = enriched['backgroundIntensity']
        enriched['meanIntensity_Rstar_per_s'] = enriched['meanIntensity']
        enriched['patchMeanContrast'] = enriched['patchMean']
        enriched['patchMeanIntensity'] = (
            enriched['imageMeanIntensity'] * (1 + enriched['patchMeanContrast']))
        enriched['patchRmsContrast'] = patch_rms_contrast(
            enriched['patchMeanContrast'], enriched['patchVariance'])
        enriched['delta_nli_cone_minus_disc'] = (
            enriched['nli_image_vs_cone_disc']
            - enriched['nli_image_vs_disc'])
        response_columns = [
            'image_response', 'disc_response', 'cone_disc_response']
        enriched['complete_response_triplet'] = (
            enriched[response_columns].notna().all(axis=1))
        mixed = enriched['stimulus_metadata_mixed_fields'].astype(str).ne('')
        enriched['analysis_ready'] = (
            enriched['complete_response_triplet'] & ~mixed)
        enriched['exclusion_reason'] = ''
        enriched.loc[
            ~enriched['complete_response_triplet'], 'exclusion_reason'
        ] = 'incomplete response triplet'
        enriched.loc[mixed, 'exclusion_reason'] = (
            'mixed stimulus metadata: '
            + enriched.loc[mixed, 'stimulus_metadata_mixed_fields'].astype(str))
        enriched['source_condition_h5'] = str(path)
        excluded_condition = enriched.loc[~enriched['analysis_ready']].copy()
        if not excluded_condition.empty:
            excluded_rows.append(excluded_condition)
            if verbose:
                print(f'ALERT: {analysis.exp_name}/{analysis.cell_label} | '
                      f'{analysis.online_analysis} | FW{analysis.filter_wheel_ndf:g}: '
                      f'excluding {len(excluded_condition)} patch row(s) that are '
                      'not analysis-ready')
        if require_analysis_ready:
            enriched = enriched.loc[enriched['analysis_ready']].copy()
        rows.append(enriched)
        if verbose:
            print(f'{analysis.exp_name}/{analysis.cell_label} | '
                  f'{analysis.online_analysis} | FW{analysis.filter_wheel_ndf:g} | '
                  f'{len(enriched)} patches')
    excluded = (pd.concat(excluded_rows, ignore_index=True)
                if excluded_rows else
                pd.DataFrame(columns=PATCH_VARIANCE_POPULATION_COLUMNS))
    if not rows:
        table = pd.DataFrame(columns=PATCH_VARIANCE_POPULATION_COLUMNS)
        return (table, excluded) if return_qc else table
    table = pd.concat(rows, ignore_index=True)
    table = table[PATCH_VARIANCE_POPULATION_COLUMNS].sort_values(
        ['date', 'cell_label', 'onlineAnalysis', 'imageName',
         'patch_x_vh', 'patch_y_vh']).reset_index(drop=True)
    if verbose:
        wheel_label = ('all FilterWheel values' if filter_wheel_ndf is None
                       else f'FW{float(filter_wheel_ndf):g}')
        print(f'Built {len(table)} analysis-ready cell-patch rows from '
              f'{matched_conditions} saved {wheel_label} condition(s), '
              f'{table.patch_uid.nunique()} unique physical patches.')
        if len(excluded):
            print(f'Excluded {len(excluded)} non-analysis-ready patch row(s); '
                  'request return_qc=True to inspect them.')
    return (table, excluded) if return_qc else table


def patch_variance_population_path(filter_wheel_ndf: Optional[float] = None):
    """Default consolidated HDF5 path for the temporary patch analysis."""
    label = ('allFW' if filter_wheel_ndf is None
             else f'FW{float(filter_wheel_ndf):g}')
    return store_dir() / 'population' / f'center_disc_patch_variance_{label}.h5'


def save_patch_variance_population(table: pd.DataFrame, path=None,
                                   filter_wheel_ndf: Optional[float] = None,
                                   excluded_qc: Optional[pd.DataFrame] = None,
                                   verbose: bool = True):
    """Save the enriched all-cell patch table as one compressed HDF5 file."""
    import h5py
    from datetime import datetime, timezone
    from pathlib import Path

    missing = sorted(set(PATCH_VARIANCE_POPULATION_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f'patch population is missing columns: {missing}')
    destination = (Path(path) if path is not None
                   else patch_variance_population_path(filter_wheel_ndf))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    with h5py.File(temporary, 'w') as h5:
        h5.attrs['output_version'] = PATCH_VARIANCE_POPULATION_VERSION
        h5.attrs['filter_wheel_selection'] = (
            'all' if filter_wheel_ndf is None else float(filter_wheel_ndf))
        h5.attrs['generated_utc'] = datetime.now(timezone.utc).isoformat()
        h5.attrs['row_identity'] = (
            'date, cell_label, onlineAnalysis, protocol, patch_uid')
        h5.attrs['patch_identity'] = (
            'currentStimSet, imageName, patch_x_vh, patch_y_vh')
        _write_condition_frame(
            h5.create_group('patch_population'),
            table[PATCH_VARIANCE_POPULATION_COLUMNS])
        if excluded_qc is not None and len(excluded_qc):
            _write_condition_frame(
                h5.create_group('excluded_qc'),
                excluded_qc[PATCH_VARIANCE_POPULATION_COLUMNS])
    temporary.replace(destination)
    if verbose:
        print(f'Saved {len(table)} cell-patch rows to {destination}')
    return destination


def load_patch_variance_population(path=None,
                                   filter_wheel_ndf: Optional[float] = None,
                                   excluded_qc: bool = False) -> pd.DataFrame:
    """Load the consolidated temporary spatial-contrast population table."""
    import h5py
    from pathlib import Path

    source = (Path(path) if path is not None
              else patch_variance_population_path(filter_wheel_ndf))
    with h5py.File(source, 'r') as h5:
        version = int(h5.attrs.get('output_version', -1))
        if version not in (3, PATCH_VARIANCE_POPULATION_VERSION):
            raise ValueError(
                f'{source}: output version {version} does not match '
                f'supported versions 3 or {PATCH_VARIANCE_POPULATION_VERSION}')
        group = ('excluded_qc' if excluded_qc
                 else 'patch_population')
        if group not in h5:
            return pd.DataFrame(columns=PATCH_VARIANCE_POPULATION_COLUMNS)
        table = _read_condition_frame(h5[group])
    if 'patchRmsContrast' not in table:
        table['patchRmsContrast'] = patch_rms_contrast(
            table['patchMeanContrast'], table['patchVariance'])
    return table.reindex(columns=PATCH_VARIANCE_POPULATION_COLUMNS)


IMAGE_NLI_SUMMARY_COLUMNS = [
    'exp_name', 'cell_label', 'cell_id', 'cell_type', 'onlineAnalysis',
    'protocol', 'site', 'filter_wheel_ndf', 'imageName', 'meanIntensity',
    'n_patches', 'mean_nli_disc', 'mean_nli_cone_disc',
]

LIGHT_LEVEL_GROUPS = (
    (500.0, 1500.0),
    (1500.0, 3500.0),
    (3500.0, 6000.0),
    (6000.0, 20000.0),
)

LIGHT_LEVEL_SUMMARY_COLUMNS = [
    'cell_type', 'light_level', 'light_min', 'light_max', 'meanIntensity',
    'n_cells', 'n_cell_images', 'mean_nli_disc', 'sem_nli_disc',
    'mean_nli_cone_disc', 'sem_nli_cone_disc',
]

PATCH_NLI_COLUMNS = [
    'exp_name', 'cell_label', 'cell_id', 'cell_type', 'onlineAnalysis',
    'protocol', 'site', 'filter_wheel_ndf', 'imageName', 'patchIndex',
    'patch_key', 'meanIntensity', 'nli_disc', 'nli_cone_disc',
]

CELL_LEVEL_LIGHT_GROUPS = (
    (500.0, 1500.0),
    (6000.0, 20000.0),
)

CELL_PATCH_NLI_COLUMNS = [
    'cell_type', 'cell_id', 'exp_name', 'cell_label', 'light_level',
    'light_min', 'light_max', 'meanIntensity', 'n_images', 'n_patches',
    'mean_nli_disc', 'mean_nli_cone_disc',
]

HIGH_LIGHT_CELL_NLI_COLUMNS = [
    'cell_type', 'cell_id', 'exp_name', 'cell_label', 'min_intensity',
    'meanIntensity', 'n_images', 'n_patches', 'mean_nli_disc',
    'mean_nli_cone_disc',
]


def _saved_condition_matches_protocol(
        saved_protocols: Sequence[str],
        protocol: Optional[Union[str, Sequence[str]]]) -> bool:
    """Return whether a saved condition contains any requested protocol."""
    if protocol is None:
        return True
    requested = {protocol} if isinstance(protocol, str) else set(protocol)
    return bool(requested.intersection(saved_protocols))


def load_condition_image_nli_summary(
        output_dir=None,
        protocol: Optional[Union[str, Sequence[str]]] = 'LinearEquivalentAnnulus',
        ) -> pd.DataFrame:
    """Read saved HDF5 conditions into one row per cell, FW, and image.

    Patch NLIs are averaged only within a single saved condition's
    ``imageName``. No values are pooled across cells, FilterWheel settings, or
    cell types here; that preserves the individual observations needed by the
    population plot. Legacy CSV outputs are intentionally excluded.
    """
    rows = []
    for path in _condition_paths(output_dir, protocol, '.h5'):
        analysis = _read_condition_h5(path)
        if not _saved_condition_matches_protocol(analysis.protocols, protocol):
            continue

        patch_means = (analysis.patch_responses.groupby('imageName', sort=False)
                       .agg(n_patches=('patch_key', 'size'),
                            mean_nli_disc=('nli_disc', 'mean'),
                            mean_nli_cone_disc=('nli_cone_disc', 'mean'))
                       .reset_index())
        image_metadata = (analysis.image_summary[['imageName', 'meanIntensity']]
                          .drop_duplicates('imageName'))
        per_image = image_metadata.merge(
            patch_means, on='imageName', how='inner', validate='one_to_one')

        for row in per_image.itertuples(index=False):
            rows.append({
                'exp_name': analysis.exp_name,
                'cell_label': analysis.cell_label,
                'cell_id': f'{analysis.exp_name}/{analysis.cell_label}',
                'cell_type': analysis.cell_type,
                'onlineAnalysis': analysis.online_analysis,
                'protocol': ', '.join(analysis.protocols),
                'site': analysis.site,
                'filter_wheel_ndf': analysis.filter_wheel_ndf,
                'imageName': str(row.imageName),
                'meanIntensity': float(row.meanIntensity),
                'n_patches': int(row.n_patches),
                'mean_nli_disc': float(row.mean_nli_disc),
                'mean_nli_cone_disc': float(row.mean_nli_cone_disc),
            })

    if not rows:
        return pd.DataFrame(columns=IMAGE_NLI_SUMMARY_COLUMNS)
    return (pd.DataFrame(rows, columns=IMAGE_NLI_SUMMARY_COLUMNS)
            .sort_values(['cell_type', 'exp_name', 'cell_label', 'onlineAnalysis',
                          'filter_wheel_ndf', 'meanIntensity', 'imageName'],
                         ignore_index=True))


def load_condition_patch_nli(
        output_dir=None,
        protocol: Optional[Union[str, Sequence[str]]] = 'LinearEquivalentAnnulus',
        ) -> pd.DataFrame:
    """Read every saved HDF5 patch NLI without image or cell averaging.

    One row remains one unique ``(imageName, patchIndex)`` observation in one
    saved cell condition. Legacy CSV files are intentionally excluded.
    """
    tables = []
    for path in _condition_paths(output_dir, protocol, '.h5'):
        analysis = _read_condition_h5(path)
        if not _saved_condition_matches_protocol(analysis.protocols, protocol):
            continue
        table = condition_population_table(analysis).rename(columns={
            'date': 'exp_name',
            'nli_image_vs_disc': 'nli_disc',
            'nli_image_vs_cone_disc': 'nli_cone_disc',
        })
        table.insert(2, 'cell_id', table['exp_name'] + '/' + table['cell_label'])
        tables.append(table[PATCH_NLI_COLUMNS])
    if not tables:
        return pd.DataFrame(columns=PATCH_NLI_COLUMNS)
    return (pd.concat(tables, ignore_index=True)
            .sort_values(['cell_type', 'exp_name', 'cell_label', 'onlineAnalysis',
                          'filter_wheel_ndf', 'imageName', 'patchIndex'],
                         ignore_index=True))


MATLAB_CENTER_DISC_PROTOCOLS = (
    'LinearEquivalentDiscConeLin', 'LinearEquivalentDisc')
MATLAB_ANNULUS_DISC_PROTOCOLS = ('LinearEquivalentAnnulus',)
MATLAB_RESULT_FIELDS = (
    'NLI', 'NLIConeLin', 'ImageResp', 'DiscResp', 'LinDiscResp',
    'ImageRespSEM', 'DiscRespSEM', 'LinDiscRespSEM', 'imageID', 'date',
    'cell', 'nd', 'cellType', 'maxIntensity', 'meanIntensity',
)


def _matlab_cell_type_name(cell_type: str) -> str:
    """Convert ``OFF-parasol`` to the reference filename label ``OffParasol``."""
    import re

    words = re.findall(r'[A-Za-z0-9]+', str(cell_type))
    return ''.join(word.lower().capitalize() for word in words) or 'Unknown'


def _matlab_result_element(group: pd.DataFrame):
    """Build one nested ``{results: struct}`` element like the reference MAT file."""
    ordered = group.sort_values('patchIndex', kind='stable')

    def vector(column):
        return pd.to_numeric(ordered[column], errors='coerce').to_numpy(
            dtype=float)[None, :]

    def scalar(column):
        values = pd.to_numeric(ordered[column], errors='coerce').dropna().unique()
        if len(values) != 1:
            key = tuple(ordered.iloc[0][
                ['date', 'cell_label', 'filter_wheel_ndf', 'imageName']])
            raise ValueError(
                f'{key}: expected one numeric {column}, found {values.tolist()}')
        return float(values[0])

    first = ordered.iloc[0]
    cell_type = str(first['cell_type'])
    values = {
        'NLI': vector('nli_image_vs_disc'),
        'NLIConeLin': vector('nli_image_vs_cone_disc'),
        'ImageResp': vector('image_response'),
        'DiscResp': vector('disc_response'),
        'LinDiscResp': vector('cone_disc_response'),
        'ImageRespSEM': vector('image_response_sem'),
        'DiscRespSEM': vector('disc_response_sem'),
        'LinDiscRespSEM': vector('cone_disc_response_sem'),
        'imageID': str(first['imageName']),
        'date': str(first['date']),
        'cell': str(first['cell_label']),
        'nd': float(first['filter_wheel_ndf']),
        'cellType': (cell_type if '\\' in cell_type else f'RGC\\{cell_type}'),
        'maxIntensity': scalar('maxIntensity'),
        'meanIntensity': scalar('meanIntensity'),
    }
    results = np.empty((1, 1), dtype=[(name, 'O') for name in MATLAB_RESULT_FIELDS])
    for name in MATLAB_RESULT_FIELDS:
        results[name][0, 0] = values[name]
    element = np.empty((1, 1), dtype=[('results', 'O')])
    element['results'][0, 0] = results
    return element


def _export_population_mat_files(
        protocols, output_label: str, output_dir=None, source_dir=None,
        online_analysis: str = 'extracellular', verbose: bool = True):
    """Write one reference-compatible MAT file per cell type and protocol family.

    The reference ``OffParasolLinEquiv.mat`` contains a ``1 x N`` cell array
    named ``collectedResults``. Each element is one unique date/cell/ND/image
    condition and contains a nested ``results`` struct. This export preserves
    its field order and appends scalar ``maxIntensity`` and ``meanIntensity``.

    The reference schema has no recording-mode field, so the default exports
    extracellular records only. Comma-joined saved intensities are corrected
    to their larger numeric candidate by :func:`_read_condition_h5`; every
    correction is printed for manual review.
    """
    from pathlib import Path
    import warnings
    from scipy.io import savemat

    if output_dir is None:
        output_dir = condition_output_dir(protocols) / 'matlab_exports'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        population = load_condition_outputs(
            output_dir=source_dir, protocol=protocols)
    corrections = [str(item.message) for item in caught
                   if 'conflicting saved intensity metadata' in str(item.message)]
    if verbose:
        if corrections:
            for message in corrections:
                print(f'CORRECTED: {message}')
        else:
            print('No conflicting saved intensity metadata found.')

    if population.empty:
        raise ValueError(
            f'no saved {output_label} population conditions were found')
    mode = str(online_analysis).strip().lower()
    population = population.loc[
        population['onlineAnalysis'].astype(str).str.strip().str.lower().eq(mode)
    ].copy()
    if population.empty:
        raise ValueError(
            f'no saved {output_label} conditions use onlineAnalysis={mode!r}')

    keys = ['date', 'cell_label', 'filter_wheel_ndf', 'imageName']
    duplicate_types = (population[keys + ['cell_type']].drop_duplicates()
                       .groupby(keys, dropna=False)['cell_type'].nunique())
    if duplicate_types.gt(1).any():
        raise ValueError('a date/cell/ND/image condition has multiple cell types')

    written = {}
    for cell_type, cell_rows in population.groupby('cell_type', sort=True):
        elements = [
            _matlab_result_element(group)
            for _, group in cell_rows.groupby(keys, sort=True, dropna=False)
        ]
        collected = np.empty((1, len(elements)), dtype=object)
        for index, element in enumerate(elements):
            collected[0, index] = element
        filename = f'{_matlab_cell_type_name(cell_type)}LinEquiv{output_label}.mat'
        path = output_dir / filename
        savemat(path, {'collectedResults': collected}, do_compression=True)
        written[str(cell_type)] = path
        if verbose:
            print(f'wrote {cell_type}: {len(elements)} conditions to {path}')
    return written


def export_center_disc_population_mat_files(
        output_dir=None, source_dir=None, online_analysis: str = 'extracellular',
        verbose: bool = True):
    """Write one ``*LinEquivCenterDisc.mat`` file per saved cell type."""
    return _export_population_mat_files(
        MATLAB_CENTER_DISC_PROTOCOLS, 'CenterDisc', output_dir=output_dir,
        source_dir=source_dir, online_analysis=online_analysis, verbose=verbose)


def export_annulus_disc_population_mat_files(
        output_dir=None, source_dir=None, online_analysis: str = 'extracellular',
        verbose: bool = True):
    """Write one ``*LinEquivAnnulusDisc.mat`` file per saved cell type."""
    return _export_population_mat_files(
        MATLAB_ANNULUS_DISC_PROTOCOLS, 'AnnulusDisc', output_dir=output_dir,
        source_dir=source_dir, online_analysis=online_analysis, verbose=verbose)


MATLAB_RIG_SOURCES = {'E': 'fred_data', 'G': 'chris_data'}
MATLAB_RIG_SUMMARY_COLUMNS = (
    'scope', 'rig', 'data_source', 'condition_entries', 'unique_cells',
    'experiment_dates')


def summarize_matlab_export_rigs(mat_files) -> pd.DataFrame:
    """Count exported MATLAB conditions and cells from rigs E and G.

    ``mat_files`` is the cell-type-to-path mapping returned by either MATLAB
    exporter. Counts are derived from the written ``collectedResults`` arrays,
    not from the pre-export DataFrame. ``scope='ALL'`` rows provide protocol
    totals, followed by the per-cell-type breakdown.
    """
    import re
    from scipy.io import loadmat

    records = []
    for cell_type, path in mat_files.items():
        collected = loadmat(
            path, squeeze_me=False,
            struct_as_record=False)['collectedResults']
        for raw in collected.ravel(order='F'):
            result = raw.item().results.item()
            date = str(result.date.item())
            match = re.search(r'(?:^|_)([EG])(?:_|$)', date)
            rig = match.group(1) if match else '?'
            records.append({
                'scope': str(cell_type), 'rig': rig,
                'date': date, 'cell': str(result.cell.item()),
            })
    if not records:
        return pd.DataFrame(columns=MATLAB_RIG_SUMMARY_COLUMNS)

    frame = pd.DataFrame(records)

    def aggregate(values, scope):
        return {
            'scope': scope,
            'rig': str(values['rig'].iloc[0]),
            'data_source': MATLAB_RIG_SOURCES.get(
                str(values['rig'].iloc[0]), 'unknown'),
            'condition_entries': int(len(values)),
            'unique_cells': int(values[['date', 'cell']].drop_duplicates().shape[0]),
            'experiment_dates': int(values['date'].nunique()),
        }

    rows = [aggregate(group, 'ALL') for _, group in frame.groupby('rig', sort=True)]
    rows.extend(
        aggregate(group, str(scope))
        for (scope, _), group in frame.groupby(['scope', 'rig'], sort=True))
    return pd.DataFrame(rows, columns=MATLAB_RIG_SUMMARY_COLUMNS)


def summarize_image_nli_light_levels(
        image_summary: pd.DataFrame,
        light_groups: Sequence[Tuple[float, float]] = LIGHT_LEVEL_GROUPS,
        ) -> pd.DataFrame:
    """Aggregate cell/image NLI observations within predefined light ranges.

    Every input row remains one cell/``imageName`` observation. Within each
    cell type and light range, x is the observed mean ``meanIntensity`` and y
    is the mean NLI across those observations; error is SEM across the same
    observations. Ranges are lower-inclusive and upper-exclusive, except the
    final range, which includes its upper bound.
    """
    required = {'cell_type', 'cell_id', 'imageName', 'meanIntensity',
                'mean_nli_disc', 'mean_nli_cone_disc'}
    missing = required.difference(image_summary.columns)
    if missing:
        raise ValueError(f'image_summary is missing columns: {sorted(missing)}')

    groups = [(float(lower), float(upper)) for lower, upper in light_groups]
    if (not groups or any(not np.isfinite([lower, upper]).all() or lower >= upper
                          for lower, upper in groups)
            or any(groups[i][1] > groups[i + 1][0]
                   for i in range(len(groups) - 1))):
        raise ValueError('light_groups must be ordered, finite, non-overlapping ranges')

    frame = image_summary.copy()
    intensity = pd.to_numeric(frame['meanIntensity'], errors='coerce')
    frame['meanIntensity'] = intensity
    group_index = np.full(len(frame), -1, dtype=int)
    for index, (lower, upper) in enumerate(groups):
        upper_keep = intensity.le(upper) if index == len(groups) - 1 else intensity.lt(upper)
        keep = intensity.ge(lower) & upper_keep & (group_index == -1)
        group_index[keep.to_numpy()] = index
    frame['_light_group'] = group_index
    frame = frame.loc[frame['_light_group'].ge(0)].copy()
    if frame.empty:
        return pd.DataFrame(columns=LIGHT_LEVEL_SUMMARY_COLUMNS)

    def sem(values):
        values = pd.to_numeric(values, errors='coerce').dropna()
        return (float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else np.nan)

    summary = (frame.groupby(['cell_type', '_light_group'], sort=True, observed=True)
               .agg(meanIntensity=('meanIntensity', 'mean'),
                    n_cells=('cell_id', 'nunique'),
                    n_cell_images=('imageName', 'size'),
                    mean_nli_disc=('mean_nli_disc', 'mean'),
                    sem_nli_disc=('mean_nli_disc', sem),
                    mean_nli_cone_disc=('mean_nli_cone_disc', 'mean'),
                    sem_nli_cone_disc=('mean_nli_cone_disc', sem))
               .reset_index())
    summary['light_min'] = summary['_light_group'].map(
        lambda index: groups[int(index)][0])
    summary['light_max'] = summary['_light_group'].map(
        lambda index: groups[int(index)][1])
    summary['light_level'] = summary.apply(
        lambda row: f"{row['light_min']:g}-{row['light_max']:g}", axis=1)
    return (summary.sort_values(['cell_type', '_light_group'])
            .drop(columns='_light_group')[LIGHT_LEVEL_SUMMARY_COLUMNS]
            .reset_index(drop=True))


def summarize_cell_patch_nli_light_levels(
        patch_nli: pd.DataFrame,
        light_groups: Sequence[Tuple[float, float]] = CELL_LEVEL_LIGHT_GROUPS,
        ) -> pd.DataFrame:
    """Average all patches and image names within each cell and light level.

    This is deliberately cell-first: patch NLIs are pooled across every
    ``imageName`` in a light range to make one standard-disc and one
    cone-linearized mean per cell. Population error bars must be computed from
    these cell means, not by treating patches as independent cells.
    """
    required = {'cell_type', 'cell_id', 'exp_name', 'cell_label', 'imageName',
                'patch_key', 'meanIntensity', 'nli_disc', 'nli_cone_disc'}
    missing = required.difference(patch_nli.columns)
    if missing:
        raise ValueError(f'patch_nli is missing columns: {sorted(missing)}')

    groups = [(float(lower), float(upper)) for lower, upper in light_groups]
    if (not groups or any(not np.isfinite([lower, upper]).all() or lower >= upper
                          for lower, upper in groups)
            or any(groups[i][1] > groups[i + 1][0]
                   for i in range(len(groups) - 1))):
        raise ValueError('light_groups must be ordered, finite, non-overlapping ranges')

    frame = patch_nli.copy()
    intensity = pd.to_numeric(frame['meanIntensity'], errors='coerce')
    frame['meanIntensity'] = intensity
    group_index = np.full(len(frame), -1, dtype=int)
    for index, (lower, upper) in enumerate(groups):
        upper_keep = intensity.le(upper) if index == len(groups) - 1 else intensity.lt(upper)
        keep = intensity.ge(lower) & upper_keep & (group_index == -1)
        group_index[keep.to_numpy()] = index
    frame['_light_group'] = group_index
    frame = frame.loc[frame['_light_group'].ge(0)].copy()
    if frame.empty:
        return pd.DataFrame(columns=CELL_PATCH_NLI_COLUMNS)

    summary = (frame.groupby(['cell_type', 'cell_id', '_light_group'],
                             sort=True, observed=True)
               .agg(exp_name=('exp_name', 'first'),
                    cell_label=('cell_label', 'first'),
                    meanIntensity=('meanIntensity', 'mean'),
                    n_images=('imageName', 'nunique'),
                    n_patches=('patch_key', 'size'),
                    mean_nli_disc=('nli_disc', 'mean'),
                    mean_nli_cone_disc=('nli_cone_disc', 'mean'))
               .reset_index())
    summary['light_min'] = summary['_light_group'].map(
        lambda index: groups[int(index)][0])
    summary['light_max'] = summary['_light_group'].map(
        lambda index: groups[int(index)][1])
    default_labels = (len(groups) == 2 and groups == list(CELL_LEVEL_LIGHT_GROUPS))
    summary['light_level'] = summary.apply(
        lambda row: (('~1k' if int(row['_light_group']) == 0 else '~10k')
                     if default_labels else
                     f"{row['light_min']:g}-{row['light_max']:g}"), axis=1)
    return (summary.sort_values(['cell_type', '_light_group', 'cell_id'])
            .drop(columns='_light_group')[CELL_PATCH_NLI_COLUMNS]
            .reset_index(drop=True))


def summarize_cell_patch_nli_above(
        patch_nli: pd.DataFrame,
        min_intensity: float = 7000.0,
        ) -> pd.DataFrame:
    """Make one paired standard/cone-disc NLI observation per high-light cell.

    All patches whose image-level ``meanIntensity`` is at least
    ``min_intensity`` are pooled within a cell. This intentionally has no upper
    cutoff: it represents the requested highest-light population rather than a
    finite display bin.
    """
    required = {'cell_type', 'cell_id', 'exp_name', 'cell_label', 'imageName',
                'patch_key', 'meanIntensity', 'nli_disc', 'nli_cone_disc'}
    missing = required.difference(patch_nli.columns)
    if missing:
        raise ValueError(f'patch_nli is missing columns: {sorted(missing)}')
    cutoff = float(min_intensity)
    if not np.isfinite(cutoff):
        raise ValueError('min_intensity must be finite')

    frame = patch_nli.copy()
    intensity = pd.to_numeric(frame['meanIntensity'], errors='coerce')
    frame['meanIntensity'] = intensity
    frame = frame.loc[intensity.ge(cutoff)].copy()
    if frame.empty:
        return pd.DataFrame(columns=HIGH_LIGHT_CELL_NLI_COLUMNS)

    summary = (frame.groupby(['cell_type', 'cell_id'], sort=True, observed=True)
               .agg(exp_name=('exp_name', 'first'),
                    cell_label=('cell_label', 'first'),
                    meanIntensity=('meanIntensity', 'mean'),
                    n_images=('imageName', 'nunique'),
                    n_patches=('patch_key', 'size'),
                    mean_nli_disc=('nli_disc', 'mean'),
                    mean_nli_cone_disc=('nli_cone_disc', 'mean'))
               .reset_index())
    summary.insert(4, 'min_intensity', cutoff)
    return (summary[HIGH_LIGHT_CELL_NLI_COLUMNS]
            .sort_values(['cell_type', 'cell_id'], ignore_index=True))


def load_condition_index(
        output_dir=None,
        protocol: Optional[Union[str, Sequence[str]]] = None,
        ) -> pd.DataFrame:
    """List saved cell conditions by reading metadata only.

    This does not load or expand the patch arrays. Legacy CSVs are listed only
    when the corresponding condition has not yet been resaved as HDF5.
    """
    import h5py
    columns = ['date', 'cell_label', 'cell_type', 'onlineAnalysis',
               'filter_wheel_ndf']

    def text(value):
        return value.decode() if isinstance(value, bytes) else str(value)

    rows = []
    h5_paths = _condition_paths(output_dir, protocol, '.h5')
    h5_stems = {path.stem for path in h5_paths}
    for path in h5_paths:
        with h5py.File(path, 'r') as h5:
            saved_protocols = text(h5.attrs['protocols']).split('\n')
            if not _saved_condition_matches_protocol(saved_protocols, protocol):
                continue
            rows.append({
                'date': text(h5.attrs['exp_name']),
                'cell_label': text(h5.attrs['cell_label']),
                'cell_type': text(h5.attrs['cell_type']),
                'onlineAnalysis': text(h5.attrs['online_analysis']),
                'filter_wheel_ndf': float(h5.attrs['filter_wheel_ndf']),
            })
    for path in _condition_paths(output_dir, protocol, '.csv'):
        if path.stem in h5_stems:
            continue
        legacy = pd.read_csv(path, nrows=1)
        if not legacy.empty and (protocol is None or (
                'protocol' in legacy and _saved_condition_matches_protocol(
                    str(legacy.iloc[0]['protocol']).split(', '), protocol))):
            rows.append(legacy.iloc[0][columns].to_dict())
    return (pd.DataFrame(rows, columns=columns)
            .sort_values(['date', 'cell_label', 'onlineAnalysis', 'filter_wheel_ndf'],
                         ignore_index=True))


def record_key(exp_name: str, cell_label: str, online_analysis: str, site: str,
               ndf: float, background_intensity: float) -> str:
    def num(v):
        return ('NaN' if v is None or (isinstance(v, float) and np.isnan(v))
                else f'{v:g}'.replace('.', 'p'))
    return (f'{exp_name}__{cell_label}__{online_analysis}__{site}__'
            f'FW{num(ndf)}__bg{num(background_intensity)}')


_ARRAY_FIELDS = ('patch_ids', 'image_onset', 'image_offset', 'disc_onset', 'disc_offset',
                 'cone_onset', 'cone_offset', 'nli_disc_onset', 'nli_disc_offset',
                 'nli_cone_onset', 'nli_cone_offset')


def save_records(records: Sequence[DiscRecord], path=None, verbose: bool = True):
    """Upsert records into ``<store>/records.h5`` and refresh ``summary.csv``."""
    import h5py
    from pathlib import Path

    base = Path(path) if path is not None else store_dir()
    base.mkdir(parents=True, exist_ok=True)
    h5_path = base / 'records.h5'
    with h5py.File(h5_path, 'a') as f:
        for rec in records:
            if rec.key in f:
                del f[rec.key]
            g = f.create_group(rec.key)
            for name in _ARRAY_FIELDS:
                g.create_dataset(name, data=np.asarray(getattr(rec, name), dtype=float))
            for k, v in rec.summary_row().items():
                g.attrs[k] = '' if v is None else v
            for k, v in (rec.config or {}).items():
                if v is not None and not isinstance(v, (list, tuple, dict)):
                    g.attrs[f'cfg_{k}'] = v
    summary = load_summary(path=base)
    summary.to_csv(base / 'summary.csv', index=False)
    if verbose:
        print(f'{len(records)} record(s) saved -> {h5_path} ({len(summary)} rows total)')
    return h5_path


def load_summary(path=None) -> pd.DataFrame:
    """Scalar fields for every stored record."""
    import h5py
    from pathlib import Path

    base = Path(path) if path is not None else store_dir()
    h5_path = base / 'records.h5'
    if not h5_path.exists():
        return pd.DataFrame()
    rows = []
    with h5py.File(h5_path, 'r') as f:
        for key in f:
            a = dict(f[key].attrs)
            rows.append({k: (v.decode() if isinstance(v, bytes) else v) for k, v in a.items()})
    return (pd.DataFrame(rows).sort_values(['cell_type', 'exp_name', 'cell_label'],
                                           ignore_index=True) if rows else pd.DataFrame())


def load_records(keys: Optional[Sequence[str]] = None, path=None) -> Dict[str, Dict]:
    """Full records (arrays + scalars) as ``{key: dict}``."""
    import h5py
    from pathlib import Path

    base = Path(path) if path is not None else store_dir()
    h5_path = base / 'records.h5'
    out: Dict[str, Dict] = {}
    if not h5_path.exists():
        return out
    with h5py.File(h5_path, 'r') as f:
        for key in (keys if keys is not None else list(f)):
            if key not in f:
                continue
            g = f[key]
            rec = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in g.attrs.items()}
            rec.update({name: g[name][()] for name in g})
            out[key] = rec
    return out


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

def plot_group(rec: DiscRecord, figsize: Tuple[float, float] = (9.0, 4.2)):
    """Image vs disc response per patch, and the resulting NLI distributions.

    Left: the scatter the online figure shows — each patch's image response
    against its standard disc (grey) and cone-linearized disc (orange). Points
    above the unity line are patches where the real image drove the cell more
    than the equivalent uniform disc. Right: the NLI values those produce.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    fig, (ax_s, ax_n) = plt.subplots(1, 2, figsize=figsize)

    # Orientation follows analyzeLinearDiscCone.m: image on x, disc on y, so
    # points *below* unity are patches the image drove harder than the disc.
    ax_s.scatter(rec.image_onset, rec.disc_onset, s=26, color='#666666',
                 label='standard disc', zorder=3)
    ax_s.scatter(rec.image_onset, rec.cone_onset, s=26, color='#D55E00',
                 label='cone-linearized disc', zorder=3)
    lims = np.array([np.nanmin([rec.disc_onset.min(initial=0), rec.image_onset.min(initial=0)]),
                     np.nanmax([rec.disc_onset.max(initial=1), rec.image_onset.max(initial=1)])])
    ax_s.plot(lims, lims, '--', color='#000000', lw=1, zorder=2, label='unity')
    ax_s.set_xlabel(f'image response ({rec.units})')
    ax_s.set_ylabel(f'disc response ({rec.units})')
    ax_s.set_title(f'{rec.exp_name} {rec.cell_label} ({rec.cell_type})\n'
                   f'{rec.online_analysis} | disc over {rec.site} | {rec.light_setting} | '
                   f'{rec.n_patches} patches', fontsize=8.5)
    ax_s.legend(frameon=False, fontsize=7)

    data = [rec.nli_disc_onset, rec.nli_cone_onset,
            rec.nli_disc_offset, rec.nli_cone_offset]
    labels = ['disc\nonset', 'cone\nonset', 'disc\noffset', 'cone\noffset']
    colors = ['#666666', '#D55E00', '#666666', '#D55E00']
    for i, (d, c) in enumerate(zip(data, colors)):
        d = np.asarray(d, dtype=float)
        if d.size:
            ax_n.scatter(np.full(d.size, i) + np.random.uniform(-.09, .09, d.size), d,
                         s=14, color=c, alpha=0.6, lw=0)
            ax_n.scatter([i], [np.nanmean(d)], marker='_', s=340, color='#0072B2', zorder=4)
    ax_n.axhline(0, color='#000000', ls='--', lw=1)
    ax_n.set_xticks(range(4))
    ax_n.set_xticklabels(labels, fontsize=8)
    ax_n.set_ylabel('NLI  (image - disc) / (|image| + |disc|)')
    ax_n.set_title(f'nonlinearity index (threshold {rec.threshold_onset:.3g} onset / '
                   f'{rec.threshold_offset:.3g} offset {rec.units.split()[-1]})', fontsize=9)
    fig.tight_layout()
    return fig


def plot_condition(analysis: ConditionAnalysis,
                   panel_width: float = 3.2,
                   columns: int = 4):
    """Plot per-image scatters, a pooled scatter, and onset NLI distributions.

    Standard image--disc pairs use grey and cone-linearized pairs use red. Every
    point is one image-specific patch and carries x/y SEM bars. The pooled plot
    preserves the full ``imageName:patchIndex`` key, so reused patch indices are
    never collapsed across images.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    patches = analysis.patch_responses
    images = list(analysis.image_summary['imageName'].astype(str))
    n_columns = max(1, min(int(columns), len(images)))
    n_rows = int(np.ceil(len(images) / n_columns))
    per_image_fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(panel_width * n_columns, panel_width * n_rows),
        squeeze=False)

    colors = {'disc': '#666666', 'cone_disc': '#C44E52'}
    labels = {'disc': 'image vs disc', 'cone_disc': 'image vs cone-lin disc'}

    def draw_scatter(ax, frame, title):
        plotted = []
        for category in ('disc', 'cone_disc'):
            x = frame['image_mean'].to_numpy(dtype=float)
            y = frame[f'{category}_mean'].to_numpy(dtype=float)
            xerr = frame['image_sem'].fillna(0).to_numpy(dtype=float)
            yerr = frame[f'{category}_sem'].fillna(0).to_numpy(dtype=float)
            keep = np.isfinite(x) & np.isfinite(y)
            if not keep.any():
                continue
            ax.errorbar(x[keep], y[keep], xerr=xerr[keep], yerr=yerr[keep],
                        fmt='o', ms=4, color=colors[category], ecolor=colors[category],
                        elinewidth=.8, capsize=2, alpha=.82, label=labels[category],
                        zorder=3)
            plotted.extend((x[keep] - xerr[keep]).tolist())
            plotted.extend((x[keep] + xerr[keep]).tolist())
            plotted.extend((y[keep] - yerr[keep]).tolist())
            plotted.extend((y[keep] + yerr[keep]).tolist())
        finite = np.asarray(plotted, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            lower, upper = min(float(finite.min()), 0.0), max(float(finite.max()), 0.0)
            span = upper - lower
            pad = .06 * span if span else 1.0
            limits = (lower - pad, upper + pad)
            ax.plot(limits, limits, '--', color='black', lw=1, zorder=1)
            ax.set_xlim(limits)
            ax.set_ylim(limits)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f'image response ({analysis.units})')
        ax.set_ylabel(f'disc response ({analysis.units})')
        ax.set_title(title, fontsize=9)

    summaries = analysis.image_summary.set_index('imageName')
    for ax, image_name in zip(axes.flat, images):
        frame = patches.loc[patches['imageName'].astype(str).eq(image_name)]
        summary = summaries.loc[image_name]
        mean_intensity = summary['meanIntensity']
        intensity_text = (f'{mean_intensity:g}' if np.isscalar(mean_intensity)
                          and not isinstance(mean_intensity, str) else str(mean_intensity))
        draw_scatter(
            ax, frame,
            f'{image_name} | {int(summary.epochs)} epochs | '
            f'meanIntensity {intensity_text}\n{len(frame)} patches')
    for ax in axes.flat[len(images):]:
        ax.set_visible(False)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        per_image_fig.legend(handles, legend_labels, loc='upper center', ncol=2,
                             bbox_to_anchor=(.5, .955), frameon=False)
    per_image_fig.suptitle(
        f'{analysis.exp_name}/{analysis.cell_label} | {analysis.online_analysis} | '
        f'FilterWheel {analysis.filter_wheel_ndf:g}', y=.995, fontsize=11)
    per_image_fig.tight_layout(rect=(0, 0, 1, .91))

    pooled_fig, pooled_ax = plt.subplots(figsize=(5.4, 5.0))
    draw_scatter(pooled_ax, patches,
                 f'all image-specific patches ({len(patches)} pairs)')
    pooled_ax.legend(frameon=False, fontsize=8)
    pooled_fig.tight_layout()

    nli_fig, (nli_ax, mean_ax) = plt.subplots(1, 2, figsize=(9.2, 4.2))
    nli_groups = (
        ('nli_disc', colors['disc'], 'image vs disc'),
        ('nli_cone_disc', colors['cone_disc'], 'image vs cone-lin disc'),
    )
    group_stats = []
    for group_index, (column, color, label) in enumerate(nli_groups):
        values = patches[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            sorted_values = np.sort(values)
            cumulative = np.arange(1, values.size + 1, dtype=float) / values.size
            nli_ax.step(np.r_[-1.0, sorted_values], np.r_[0.0, cumulative],
                        where='post', lw=1.6, color=color,
                        label=f'{label} (n={len(values)})')
            mean = float(np.mean(values))
            sem = (float(np.std(values, ddof=1) / np.sqrt(values.size))
                   if values.size > 1 else 0.0)
            nli_ax.axvline(mean, color=color, lw=1, alpha=.75)
            mean_ax.errorbar(group_index, mean, yerr=sem, fmt='o', ms=7,
                             color=color, ecolor=color, elinewidth=1.5, capsize=4)
            group_stats.append((group_index, mean, sem, len(values)))
    nli_ax.axvline(0, color='black', ls='--', lw=1)
    nli_ax.set_xlim(-1, 1)
    nli_ax.set_xlabel('Nonlinear Index  (image - disc) / (|image| + |disc|)')
    nli_ax.set_ylim(0, 1.02)
    nli_ax.set_ylabel('cumulative fraction')
    nli_ax.set_title('onset NLI empirical CDF')
    nli_ax.legend(frameon=False, fontsize=8)
    mean_ax.axhline(0, color='black', ls='--', lw=1)
    mean_ax.set_xticks(range(len(nli_groups)))
    mean_ax.set_xticklabels(['standard disc', 'cone-lin disc'])
    mean_ax.set_ylabel('mean Nonlinear Index ± SEM')
    mean_ax.set_title('across all image-specific patches')
    for group_index, mean, sem, count in group_stats:
        mean_ax.text(group_index, mean + sem, f'n={count}', ha='center', va='bottom',
                     fontsize=8)
    nli_fig.suptitle(
        f'{analysis.exp_name}/{analysis.cell_label} | onset threshold '
        f'{analysis.threshold:g} {analysis.units}', fontsize=10)
    nli_fig.tight_layout()
    return per_image_fig, pooled_fig, nli_fig


def condition_sample_pairs(analysis: ConditionAnalysis,
                           n_pairs: int = 3) -> List[Tuple[str, float]]:
    """Choose reproducible, visibly responsive patch keys for trace checks.

    The strongest standard-disc and cone-disc comparisons are represented when
    both exist. Remaining slots favor distinct images, then response strength.
    Selection always uses the full ``(imageName, patchIndex)`` identity.
    """
    n_pairs = int(n_pairs)
    if n_pairs < 1:
        raise ValueError('n_pairs must be at least 1.')
    patches = analysis.patch_responses.copy()
    required = {'imageName', 'patchIndex', 'image_mean'}
    missing = required.difference(patches.columns)
    if missing:
        raise ValueError(f'patch_responses is missing {sorted(missing)}')
    patches = patches.loc[
        np.isfinite(pd.to_numeric(patches['patchIndex'], errors='coerce'))
        & np.isfinite(pd.to_numeric(patches['image_mean'], errors='coerce'))
    ].copy()
    if patches.empty:
        raise ValueError('No finite image/patch responses are available to sample.')
    patches['_sample_score'] = np.abs(patches['image_mean'].to_numpy(dtype=float))
    patches = patches.sort_values(
        ['_sample_score', 'imageName', 'patchIndex'],
        ascending=[False, True, True], kind='stable')

    selected = []
    selected_keys = set()
    selected_images = set()
    rows = list(patches.itertuples(index=False))
    for comparison_column in ('disc_mean', 'cone_disc_mean'):
        if comparison_column not in patches:
            continue
        for row in rows:
            key = (str(row.imageName), float(row.patchIndex))
            if key in selected_keys:
                continue
            if np.isfinite(float(getattr(row, comparison_column))):
                selected.append(key)
                selected_keys.add(key)
                selected_images.add(key[0])
                break
        if len(selected) == n_pairs:
            return selected
    for distinct_images_only in (True, False):
        for row in rows:
            image_name = str(row.imageName)
            patch_index = float(row.patchIndex)
            key = (image_name, patch_index)
            if key in selected_keys:
                continue
            if distinct_images_only and image_name in selected_images:
                continue
            selected.append(key)
            selected_keys.add(key)
            selected_images.add(image_name)
            if len(selected) == n_pairs:
                return selected
    return selected


def _load_condition_sample_trials(blocks: pd.DataFrame,
                                  pairs: Sequence[Tuple[str, float]],
                                  mode: str,
                                  detector_kwargs: Optional[dict] = None) -> pd.DataFrame:
    """Read raw trials for selected image/patch keys."""
    import contextlib
    import io
    import warnings

    import retinanalysis as ra
    from retinanalysis.utils.datajoint_utils import get_epochblock_amp_data
    from retinanalysis.utils.spike_detector import detector

    pair_set = {(str(image_name), float(patch_index))
                for image_name, patch_index in pairs}
    records = []
    for row in blocks.itertuples(index=False):
        block_id = int(row.block_id)
        exp_name = str(row.exp_name)
        parameters = list(ra.StimBlock(
            exp_name, block_id, verbose=False).df_epochs['epoch_parameters'])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            amp_data, sample_rate = get_epochblock_amp_data(
                exp_name, block_id, verbose=False)
        amp_data = np.asarray(amp_data, dtype=float)
        sample_rate = float(sample_rate)
        spike_times = None
        if mode == 'extracellular':
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning,
                                        module=r'retinanalysis\.utils\.spike_detector')
                options = dict(detector_kwargs or {})
                options.setdefault('verbose', False)
                spike_times, _, _ = detector(
                    amp_data, sample_rate=sample_rate, **options)

        for epoch_index, params in enumerate(parameters):
            if epoch_index >= len(amp_data):
                continue
            category = category_of(params.get('stimulusTag'))
            patch_index = params.get('patchIndex')
            if patch_index is None:
                patch_index = params.get('imagePatchIndex', np.nan)
            try:
                patch_index = float(patch_index)
            except (TypeError, ValueError):
                continue
            image_name = str(params.get('imageName', getattr(row, 'imageName', '?')))
            if not category or (image_name, patch_index) not in pair_set:
                continue
            pre_ms = float(params['preTime'])
            stim_ms = float(params['stimTime'])
            tail_ms = float(params.get('tailTime', 0.0))
            pre_points = max(0, int(round(pre_ms / 1e3 * sample_rate)))
            if mode == 'extracellular':
                values = (np.asarray(spike_times[epoch_index], dtype=float)
                          / sample_rate * 1e3 - pre_ms)
                time_ms = None
            else:
                trace = amp_data[epoch_index]
                baseline = float(np.mean(trace[:pre_points])) if pre_points else 0.0
                sign = -1.0 if mode == 'exc' else 1.0
                values = sign * (trace - baseline)
                time_ms = np.arange(trace.size, dtype=float) / sample_rate * 1e3 - pre_ms
            records.append({
                'imageName': image_name, 'patchIndex': patch_index,
                'category': category, 'block_id': block_id,
                'epoch_index': epoch_index, 'pre_ms': pre_ms,
                'stim_ms': stim_ms, 'tail_ms': tail_ms,
                'time_ms': time_ms, 'values': values,
            })
    return pd.DataFrame(records)


def plot_condition_sample_psths(blocks: pd.DataFrame,
                                analysis: ConditionAnalysis,
                                n_pairs: int = 3,
                                bin_ms: float = 10.0,
                                smooth_ms: float = 20.0,
                                detector_kwargs: Optional[dict] = None):
    """Plot sample PSTHs, or mean current traces for whole-cell conditions."""
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    if blocks.empty:
        raise ValueError('No condition blocks were supplied.')
    mode = str(analysis.online_analysis).strip().lower()
    block_modes = set(blocks['onlineAnalysis'].astype(str).str.strip().str.lower())
    if block_modes != {mode}:
        raise ValueError(f'Block mode {sorted(block_modes)} does not match analysis mode {mode!r}.')
    if bin_ms <= 0 or smooth_ms < 0:
        raise ValueError('bin_ms must be positive and smooth_ms cannot be negative.')

    pairs = condition_sample_pairs(analysis, n_pairs=n_pairs)
    sample_blocks = blocks.loc[
        blocks['block_id'].astype(int).isin(
            [int(block_id) for block_id in analysis.block_ids])]
    if sample_blocks.empty:
        raise ValueError('None of the supplied blocks were used by this analysis.')
    trials = _load_condition_sample_trials(
        sample_blocks, pairs, mode=mode, detector_kwargs=detector_kwargs)
    if trials.empty:
        raise ValueError('No raw trials matched the selected image/patch keys.')

    style.apply_publication_style()
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.1 * len(pairs), 3.35),
                             squeeze=False, sharey=True)
    colors = {'image': '#222222', 'disc': '#777777', 'cone_disc': '#C44E52'}
    labels = {'image': 'image', 'disc': 'standard disc',
              'cone_disc': 'cone-lin disc'}

    for ax, (image_name, patch_index) in zip(axes.flat, pairs):
        panel = trials.loc[
            trials['imageName'].astype(str).eq(image_name)
            & np.isclose(trials['patchIndex'].to_numpy(dtype=float), patch_index)]
        if panel.empty:
            ax.set_visible(False)
            continue
        stim_ms = float(np.nanmedian(panel['stim_ms']))
        ax.axvspan(0, stim_ms, color='#EAEAEA', alpha=.75, lw=0, zorder=0)
        for category in ('image', 'disc', 'cone_disc'):
            category_trials = panel.loc[panel['category'].eq(category)]
            if category_trials.empty:
                continue
            if mode == 'extracellular':
                start = -float(np.nanmax(category_trials['pre_ms']))
                stop = float(np.nanmax(category_trials['stim_ms']
                                      + category_trials['tail_ms']))
                edges = np.arange(start, stop + bin_ms, bin_ms)
                if edges.size < 2:
                    continue
                spikes = [np.asarray(values, dtype=float)
                          for values in category_trials['values']]
                pooled = np.concatenate(spikes) if spikes else np.array([])
                rate = np.histogram(pooled, bins=edges)[0].astype(float)
                rate /= len(category_trials) * (bin_ms / 1e3)
                if smooth_ms > 0 and rate.size > 2:
                    sigma = smooth_ms / bin_ms
                    radius = min(int(np.ceil(3 * sigma)), (rate.size - 1) // 2)
                    x = np.arange(-radius, radius + 1, dtype=float)
                    kernel = np.exp(-.5 * (x / sigma) ** 2)
                    kernel /= kernel.sum()
                    rate = np.convolve(rate, kernel, mode='same')
                centers = (edges[:-1] + edges[1:]) / 2
                ax.plot(centers, rate, color=colors[category], lw=1.5,
                        label=labels[category])
            else:
                start = max(float(np.nanmin(time)) for time in category_trials['time_ms'])
                stop = min(float(np.nanmax(time)) for time in category_trials['time_ms'])
                grid = np.arange(start, stop + .5 * bin_ms, bin_ms)
                if grid.size < 2:
                    continue
                interpolated = np.vstack([
                    np.interp(grid, np.asarray(row.time_ms, dtype=float),
                              np.asarray(row.values, dtype=float))
                    for row in category_trials.itertuples(index=False)
                ])
                mean = np.mean(interpolated, axis=0)
                sem = (np.std(interpolated, axis=0, ddof=1) / np.sqrt(len(interpolated))
                       if len(interpolated) > 1 else np.zeros_like(mean))
                ax.plot(grid, mean, color=colors[category], lw=1.4,
                        label=labels[category])
                ax.fill_between(grid, mean - sem, mean + sem,
                                color=colors[category], alpha=.16, lw=0)
        ax.axvline(0, color='#555555', ls='--', lw=.8)
        ax.axvline(stim_ms, color='#555555', ls='--', lw=.8)
        ax.set_xlabel('time from stimulus onset (ms)')
        counts = panel['category'].value_counts()
        ax.set_title(
            f'{image_name} : patch {patch_index:g}\n'
            f'trials: image {counts.get("image", 0)}, disc {counts.get("disc", 0)}, '
            f'cone {counts.get("cone_disc", 0)}', fontsize=9)
    axes.flat[0].set_ylabel('firing rate (Hz)' if mode == 'extracellular'
                            else 'baseline-subtracted current (pA)')
    legend_entries = {}
    for ax in axes.flat:
        handles, legend_labels = ax.get_legend_handles_labels()
        legend_entries.update(zip(legend_labels, handles))
    if legend_entries:
        fig.legend(legend_entries.values(), legend_entries.keys(),
                   loc='upper center', ncol=3,
                   bbox_to_anchor=(.5, .98), frameon=False, fontsize=8)
    fig.suptitle(
        f'{analysis.exp_name}/{analysis.cell_label} | '
        f'{"sample PSTHs" if mode == "extracellular" else "sample mean traces"}',
        y=1.04, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .91))
    return fig


def plot_population_nli(summary: Optional[pd.DataFrame] = None,
                        window: str = 'onset',
                        figsize: Tuple[float, float] = (9.0, 4.4)):
    """Per-cell paired NLI, standard disc vs cone-linearized disc.

    Section 1 of ``populationLinConeDisc.m``: one line per recording joining its
    mean standard-disc NLI to its mean cone-linearized NLI, one panel per
    recording mode, with a paired Wilcoxon signed-rank test.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if summary is None:
        summary = load_summary()
    a_col, b_col = f'nli_disc_{window}', f'nli_cone_{window}'
    modes = [m for m in ('extracellular', 'exc', 'inh') if m in set(summary['online_analysis'])]
    fig, axes = plt.subplots(1, max(len(modes), 1), figsize=figsize, squeeze=False)

    for ax, mode in zip(axes[0], modes):
        sub = summary[summary['online_analysis'].eq(mode)].dropna(subset=[a_col, b_col])
        for _, r in sub.iterrows():
            ax.plot([0, 1], [r[a_col], r[b_col]], '-', color='#999999', lw=0.9, alpha=0.8)
        ax.scatter(np.zeros(len(sub)), sub[a_col], s=26, color='#666666', zorder=3)
        ax.scatter(np.ones(len(sub)), sub[b_col], s=26, color='#D55E00', zorder=3)
        for x, col in ((0, a_col), (1, b_col)):
            ax.scatter([x], [sub[col].mean()], marker='_', s=420, color='#0072B2', zorder=4)
        stat = ''
        if len(sub) >= 3:
            from scipy.stats import wilcoxon
            try:
                stat = f'\nsigned-rank p = {wilcoxon(sub[a_col], sub[b_col]).pvalue:.3g}'
            except ValueError:
                stat = ''
        ax.axhline(0, color='#000000', ls='--', lw=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['standard\ndisc', 'cone-lin\ndisc'])
        ax.set_ylabel(f'mean NLI ({window})')
        ax.set_title(f'{mode} (n={len(sub)}){stat}', fontsize=9)
    fig.suptitle(f'Per-cell NLI, standard vs cone-linearized disc ({window})',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


def plot_image_nli_by_cell_type(image_summary: Optional[pd.DataFrame] = None,
                                cell_types: Optional[Sequence[str]] = None,
                                light_groups: Sequence[Tuple[float, float]] =
                                LIGHT_LEVEL_GROUPS,
                                log_x: bool = True,
                                columns: int = 2,
                                panel_size: Tuple[float, float] = (4.8, 3.8),
                                title_prefix: str = 'LinearEquivalentAnnulus'):
    """Plot binned population mean NLI and SEM against mean light level.

    The input retains every cell/image observation. This function groups those
    rows with :func:`summarize_image_nli_light_levels`, then draws the standard
    and cone-linearized population mean +/- SEM in one panel per cell type.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    from retinanalysis.utils import style

    style.apply_publication_style()
    if image_summary is None:
        image_summary = load_condition_image_nli_summary()
    grouped = summarize_image_nli_light_levels(image_summary, light_groups=light_groups)
    if cell_types is None:
        preferred = ['ON-parasol', 'OFF-parasol', 'ON-midget', 'OFF-midget']
        present = set(grouped['cell_type'].dropna().astype(str))
        cell_types = [name for name in preferred if name in present]
        cell_types += sorted(present.difference(cell_types))
    else:
        cell_types = [str(name) for name in cell_types]

    n_panels = max(len(cell_types), 1)
    n_columns = max(1, min(int(columns), n_panels))
    n_rows = int(np.ceil(n_panels / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(panel_size[0] * n_columns, panel_size[1] * n_rows),
        squeeze=False, sharey=True)

    series = (
        ('mean_nli_disc', 'sem_nli_disc', '#666666', 'o', 'image vs standard disc'),
        ('mean_nli_cone_disc', 'sem_nli_cone_disc', '#C44E52', 's',
         'image vs cone-lin disc'),
    )

    for ax, cell_type in zip(axes.flat, cell_types):
        sub = grouped.loc[grouped['cell_type'].astype(str).eq(cell_type)].copy()
        for mean_column, sem_column, color, marker, label in series:
            x = sub['meanIntensity'].to_numpy(dtype=float)
            y = sub[mean_column].to_numpy(dtype=float)
            errors = sub[sem_column].to_numpy(dtype=float)
            keep = np.isfinite(x) & np.isfinite(y)
            ax.plot(x[keep], y[keep], '-', color=color, lw=1.3, alpha=.8,
                    label=label, zorder=2)
            ax.scatter(x[keep], y[keep], s=34, marker=marker, color=color,
                       zorder=3)
            with_error = keep & np.isfinite(errors)
            if with_error.any():
                ax.errorbar(x[with_error], y[with_error], yerr=errors[with_error],
                            fmt='none', color=color, elinewidth=1.2, capsize=3,
                            zorder=2)
        ax.axhline(0, color='black', ls='--', lw=1)
        positive_x = pd.to_numeric(sub['meanIntensity'], errors='coerce').dropna()
        if log_x and len(positive_x) and positive_x.gt(0).all():
            ax.set_xscale('log')
            ax.xaxis.set_minor_locator(NullLocator())
        if len(positive_x):
            ax.set_xticks(positive_x)
            ax.set_xticklabels([f'{value:,.0f}' for value in positive_x])
        ax.set_ylim(-1.02, 1.02)
        ax.set_xlabel('mean intensity within light-level group')
        ax.set_ylabel('population mean NLI ± SEM')
        n_cells = image_summary.loc[
            image_summary['cell_type'].astype(str).eq(cell_type), 'cell_id'].nunique()
        ax.set_title(f'{cell_type} | {n_cells} cells', fontsize=9)
        ax.legend(frameon=False, fontsize=7)

    for ax in axes.flat[len(cell_types):]:
        ax.set_visible(False)
    if not cell_types:
        axes.flat[0].text(.5, .5, 'no saved image-level NLI data',
                          ha='center', va='center', transform=axes.flat[0].transAxes)
        axes.flat[0].set_axis_off()

    fig.suptitle(f'{title_prefix}: grouped light-level NLI by cell type',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


def plot_pooled_patch_nli_distributions(
        patch_nli: Optional[pd.DataFrame] = None,
        cell_types: Optional[Sequence[str]] = None,
        bins: int = 50,
        figsize: Tuple[float, float] = (9.2, 4.1),
        title_prefix: str = 'LinearEquivalentAnnulus'):
    """Plot normalized density and empirical CDF for all saved patch NLIs.

    Patches are pooled without image- or cell-level averaging. ``cell_types``
    can restrict the pool; by default every saved annulus condition contributes.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if patch_nli is None:
        patch_nli = load_condition_patch_nli()
    required = {'cell_type', 'nli_disc', 'nli_cone_disc'}
    missing = required.difference(patch_nli.columns)
    if missing:
        raise ValueError(f'patch_nli is missing columns: {sorted(missing)}')
    frame = patch_nli.copy()
    if cell_types is not None:
        wanted = {str(value) for value in cell_types}
        frame = frame.loc[frame['cell_type'].astype(str).isin(wanted)]

    fig, (density_ax, cdf_ax) = plt.subplots(1, 2, figsize=figsize)
    edges = np.linspace(-1, 1, int(bins) + 1)
    series = (
        ('nli_disc', '#666666', 'image vs standard disc'),
        ('nli_cone_disc', '#C44E52', 'image vs cone-lin disc'),
    )
    for column, color, label in series:
        values = pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        full_label = f'{label} (n={values.size})'
        density_ax.hist(values, bins=edges, density=True, histtype='step',
                        lw=1.7, color=color, label=full_label)
        ordered = np.sort(values)
        cumulative = np.arange(1, values.size + 1, dtype=float) / values.size
        cdf_ax.step(np.r_[-1.0, ordered, 1.0], np.r_[0.0, cumulative, 1.0],
                    where='post', lw=1.7, color=color, label=full_label)

    for ax in (density_ax, cdf_ax):
        ax.axvline(0, color='black', ls='--', lw=1)
        ax.set_xlim(-1, 1)
        ax.set_xlabel('NLI  (image - disc) / (|image| + |disc|)')
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, frameon=False, fontsize=8)
        else:
            ax.text(.5, .5, 'no saved patch NLI data', ha='center', va='center',
                    transform=ax.transAxes)
    density_ax.set_ylabel('density')
    density_ax.set_title(f'{int(bins)}-bin pooled patch density')
    cdf_ax.set_ylim(0, 1.02)
    cdf_ax.set_ylabel('cumulative fraction')
    cdf_ax.set_title('pooled patch empirical CDF')
    types = ', '.join(sorted(frame['cell_type'].dropna().astype(str).unique()))
    fig.suptitle(f'{title_prefix} patch NLI | {types}', fontsize=10)
    fig.tight_layout()
    return fig


def plot_patch_nli_distributions_by_cell_type(
        patch_nli: Optional[pd.DataFrame] = None,
        cell_types: Optional[Sequence[str]] = None,
        bins: int = 50,
        panel_size: Tuple[float, float] = (4.5, 3.2),
        title_prefix: str = 'LinearEquivalentAnnulus'):
    """Plot patch-NLI density and empirical CDF separately for each cell type."""
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if patch_nli is None:
        patch_nli = load_condition_patch_nli()
    required = {'cell_type', 'nli_disc', 'nli_cone_disc'}
    missing = required.difference(patch_nli.columns)
    if missing:
        raise ValueError(f'patch_nli is missing columns: {sorted(missing)}')
    frame = patch_nli.copy()
    present = set(frame['cell_type'].dropna().astype(str))
    if cell_types is None:
        preferred = ['ON-parasol', 'OFF-parasol', 'ON-midget', 'OFF-midget']
        cell_types = [name for name in preferred if name in present]
        cell_types += sorted(present.difference(cell_types))
    else:
        cell_types = [str(value) for value in cell_types]

    n_rows = max(len(cell_types), 1)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(panel_size[0] * 2, panel_size[1] * n_rows),
        squeeze=False, sharex=True)
    edges = np.linspace(-1, 1, int(bins) + 1)
    series = (
        ('nli_disc', '#666666', 'image vs standard disc'),
        ('nli_cone_disc', '#C44E52', 'image vs cone-lin disc'),
    )
    for row_index, cell_type in enumerate(cell_types):
        density_ax, cdf_ax = axes[row_index]
        sub = frame.loc[frame['cell_type'].astype(str).eq(cell_type)]
        for column, color, label in series:
            values = pd.to_numeric(sub[column], errors='coerce').to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            full_label = f'{label} (n={values.size})'
            density_ax.hist(values, bins=edges, density=True, histtype='step',
                            lw=1.7, color=color, label=full_label)
            ordered = np.sort(values)
            cumulative = np.arange(1, values.size + 1, dtype=float) / values.size
            cdf_ax.step(np.r_[-1.0, ordered, 1.0],
                        np.r_[0.0, cumulative, 1.0], where='post',
                        lw=1.7, color=color, label=full_label)
        for ax in (density_ax, cdf_ax):
            ax.axvline(0, color='black', ls='--', lw=1)
            ax.set_xlim(-1, 1)
            ax.legend(frameon=False, fontsize=7)
        density_ax.set_ylabel(f'{cell_type}\ndensity')
        cdf_ax.set_ylabel('cumulative fraction')
        cdf_ax.set_ylim(0, 1.02)
        density_ax.set_title(f'{int(bins)}-bin patch density')
        cdf_ax.set_title('patch empirical CDF')
    if not cell_types:
        for ax in axes[0]:
            ax.text(.5, .5, 'no saved patch NLI data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_axis_off()
    else:
        for ax in axes[-1]:
            ax.set_xlabel('NLI  (image - disc) / (|image| + |disc|)')
    fig.suptitle(f'{title_prefix}: patch NLI distributions by cell type',
                 fontsize=11, y=1.0)
    fig.tight_layout()
    return fig


def plot_cell_patch_nli_paired_above(
        cell_summary: Optional[pd.DataFrame] = None,
        patch_nli: Optional[pd.DataFrame] = None,
        min_intensity: float = 7000.0,
        cell_types: Optional[Sequence[str]] = None,
        columns: int = 2,
        panel_size: Tuple[float, float] = (4.2, 3.8),
        title_prefix: str = 'LinearEquivalentAnnulus'):
    """Paired per-cell standard/cone-disc NLI above a light-level cutoff."""
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if cell_summary is None:
        if patch_nli is None:
            patch_nli = load_condition_patch_nli()
        cell_summary = summarize_cell_patch_nli_above(
            patch_nli, min_intensity=min_intensity)
    required = set(HIGH_LIGHT_CELL_NLI_COLUMNS)
    missing = required.difference(cell_summary.columns)
    if missing:
        raise ValueError(f'cell_summary is missing columns: {sorted(missing)}')
    frame = cell_summary.copy()
    present = set(frame['cell_type'].dropna().astype(str))
    if cell_types is None:
        preferred = ['ON-parasol', 'OFF-parasol', 'ON-midget', 'OFF-midget']
        cell_types = [name for name in preferred if name in present]
        cell_types += sorted(present.difference(cell_types))
    else:
        cell_types = [str(value) for value in cell_types]

    n_panels = max(len(cell_types), 1)
    n_columns = max(1, min(int(columns), n_panels))
    n_rows = int(np.ceil(n_panels / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(panel_size[0] * n_columns, panel_size[1] * n_rows),
        squeeze=False, sharey=True)
    for ax, cell_type in zip(axes.flat, cell_types):
        sub = frame.loc[frame['cell_type'].astype(str).eq(cell_type)].dropna(
            subset=['mean_nli_disc', 'mean_nli_cone_disc'])
        for row in sub.itertuples(index=False):
            ax.plot([0, 1], [row.mean_nli_disc, row.mean_nli_cone_disc],
                    '-', color='#9A9A9A', lw=1.0, alpha=.7, zorder=1)
        ax.scatter(np.zeros(len(sub)), sub['mean_nli_disc'], s=30,
                   color='#666666', zorder=3, label='standard disc')
        ax.scatter(np.ones(len(sub)), sub['mean_nli_cone_disc'], s=30,
                   color='#C44E52', zorder=3, label='cone-lin disc')
        ax.axhline(0, color='black', ls='--', lw=1)
        ax.set_xlim(-.35, 1.35)
        ax.set_ylim(-1.02, 1.02)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['standard\ndisc', 'cone-lin\ndisc'])
        ax.set_ylabel('cell mean patch NLI')
        noun = 'cell' if len(sub) == 1 else 'cells'
        ax.set_title(f'{cell_type} | {len(sub)} {noun}', fontsize=9)
    if cell_types:
        for ax in axes.flat[len(cell_types):]:
            ax.set_visible(False)
    else:
        axes.flat[0].text(.5, .5, 'no cells at or above the light cutoff',
                          ha='center', va='center', transform=axes.flat[0].transAxes)
        axes.flat[0].set_axis_off()
    fig.suptitle(
        f'{title_prefix}: paired cell NLI at ≥{float(min_intensity):,.0f} R*',
        fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


def plot_cell_patch_nli_by_light(
        cell_summary: Optional[pd.DataFrame] = None,
        patch_nli: Optional[pd.DataFrame] = None,
        cell_types: Optional[Sequence[str]] = None,
        columns: int = 2,
        panel_size: Tuple[float, float] = (4.6, 3.8),
        title_prefix: str = 'LinearEquivalentAnnulus'):
    """Plot cell-first mean patch NLI at the ~1k and ~10k light levels.

    Faint points are individual cell means across all patches and image names.
    Large points and error bars are the population mean +/- SEM across those
    cell means, so cells with more patches do not receive extra population
    weight.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if cell_summary is None:
        if patch_nli is None:
            patch_nli = load_condition_patch_nli()
        cell_summary = summarize_cell_patch_nli_light_levels(patch_nli)
    required = set(CELL_PATCH_NLI_COLUMNS)
    missing = required.difference(cell_summary.columns)
    if missing:
        raise ValueError(f'cell_summary is missing columns: {sorted(missing)}')

    frame = cell_summary.copy()
    level_rows = (frame.sort_values('light_min')
                  .drop_duplicates('light_level')[
                      ['light_level', 'light_min', 'light_max']])
    levels = level_rows['light_level'].astype(str).tolist()
    positions = {level: index for index, level in enumerate(levels)}
    tick_labels = [
        f'{row.light_level}\n{row.light_min:g}-{row.light_max:g}'
        for row in level_rows.itertuples(index=False)]

    if cell_types is None:
        preferred = ['ON-parasol', 'OFF-parasol', 'ON-midget', 'OFF-midget']
        present = set(frame['cell_type'].dropna().astype(str))
        cell_types = [name for name in preferred if name in present]
        cell_types += sorted(present.difference(cell_types))
    else:
        cell_types = [str(value) for value in cell_types]

    n_panels = max(len(cell_types), 1)
    n_columns = max(1, min(int(columns), n_panels))
    n_rows = int(np.ceil(n_panels / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(panel_size[0] * n_columns, panel_size[1] * n_rows),
        squeeze=False, sharey=True)
    series = (
        ('mean_nli_disc', '#666666', 'o', -.07, 'image vs standard disc'),
        ('mean_nli_cone_disc', '#C44E52', 's', .07, 'image vs cone-lin disc'),
    )

    def sem(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return (float(np.std(values, ddof=1) / np.sqrt(values.size))
                if values.size > 1 else np.nan)

    for ax, cell_type in zip(axes.flat, cell_types):
        sub = frame.loc[frame['cell_type'].astype(str).eq(cell_type)]
        for column, color, marker, offset, label in series:
            mean_x, mean_y = [], []
            for level in levels:
                level_values = pd.to_numeric(
                    sub.loc[sub['light_level'].astype(str).eq(level), column],
                    errors='coerce').dropna().to_numpy(dtype=float)
                x = positions[level] + offset
                if not level_values.size:
                    continue
                ax.scatter(np.full(level_values.size, x), level_values,
                           s=18, marker=marker, color=color, alpha=.28, lw=0,
                           zorder=1)
                mean = float(np.mean(level_values))
                error = sem(level_values)
                ax.scatter([x], [mean], s=58, marker=marker, color=color,
                           edgecolor='white', linewidth=.6, zorder=4)
                if np.isfinite(error):
                    ax.errorbar([x], [mean], yerr=[error], fmt='none',
                                color=color, elinewidth=1.4, capsize=4, zorder=3)
                mean_x.append(x)
                mean_y.append(mean)
            ax.plot(mean_x, mean_y, '-', color=color, lw=1.4, label=label,
                    zorder=2)
        for level in levels:
            n_cells = sub.loc[sub['light_level'].astype(str).eq(level), 'cell_id'].nunique()
            if n_cells:
                ax.text(positions[level], -.96, f'n={n_cells}', ha='center',
                        va='bottom', fontsize=7)
        ax.axhline(0, color='black', ls='--', lw=1)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(-1.02, 1.02)
        ax.set_ylabel('cell mean NLI; population mean ± SEM')
        total_cells = sub.cell_id.nunique()
        noun = 'cell' if total_cells == 1 else 'cells'
        ax.set_title(f'{cell_type} | {total_cells} {noun}', fontsize=9)
        ax.legend(frameon=False, fontsize=7)

    for ax in axes.flat[len(cell_types):]:
        ax.set_visible(False)
    if not cell_types:
        axes.flat[0].text(.5, .5, 'no saved cell-level NLI data',
                          ha='center', va='center', transform=axes.flat[0].transAxes)
        axes.flat[0].set_axis_off()
    fig.suptitle(f'{title_prefix}: cell-level patch NLI by light level',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


def plot_nli_distributions(records: Optional[Dict[str, Dict]] = None,
                           window: str = 'onset',
                           figsize: Tuple[float, float] = (9.0, 4.2)):
    """Pooled per-patch NLI: histogram and CDF, standard vs cone-linearized.

    Section 2 of ``populationLinConeDisc.m`` — every patch from every recording,
    without per-cell averaging.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if records is None:
        records = load_records()
    disc = np.concatenate([np.asarray(r[f'nli_disc_{window}'], float).ravel()
                           for r in records.values()]) if records else np.array([])
    cone = np.concatenate([np.asarray(r[f'nli_cone_{window}'], float).ravel()
                           for r in records.values()]) if records else np.array([])
    disc, cone = disc[np.isfinite(disc)], cone[np.isfinite(cone)]

    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=figsize)
    bins = np.linspace(-1, 1, 41)
    for d, c, lab in ((disc, '#666666', f'standard disc (n={disc.size})'),
                      (cone, '#D55E00', f'cone-linearized (n={cone.size})')):
        if d.size:
            ax_h.hist(d, bins=bins, histtype='step', lw=1.6, color=c, label=lab)
            ax_c.plot(np.sort(d), np.linspace(0, 1, d.size), '-', lw=1.6, color=c, label=lab)
    for ax in (ax_h, ax_c):
        ax.axvline(0, color='#000000', ls='--', lw=1)
        ax.set_xlabel(f'NLI ({window})')
        ax.legend(frameon=False, fontsize=7)
    ax_h.set_ylabel('patches')
    ax_c.set_ylabel('cumulative fraction')
    if disc.size and cone.size:
        from scipy.stats import ks_2samp
        ax_c.set_title(f'KS p = {ks_2samp(disc, cone).pvalue:.3g}', fontsize=9)
    fig.suptitle(f'Pooled per-patch NLI ({window})', fontsize=11, y=1.0)
    fig.tight_layout()
    return fig


def analyze_all(groups: pd.DataFrame, save: bool = True, plot: bool = False,
                on_error: str = 'log', verbose: bool = False,
                skip_existing: bool = False, **kwargs) -> List[DiscRecord]:
    """Run :func:`analyze_group` over every row of :func:`group_blocks` output."""
    records, failures = [], []
    existing = load_summary()
    stored = set(existing['key']) if skip_existing and len(existing) else set()
    skipped = 0
    for _, row in groups.iterrows():
        if skip_existing:
            key = record_key(row['exp_name'], row['cell_label'], row['onlineAnalysis'],
                             row['site'], row['filter_wheel_ndf'], row['backgroundIntensity'])
            if key in stored:
                skipped += 1
                continue
        try:
            rec = analyze_group(row['exp_name'],
                                [int(b) for b in str(row['block_ids']).split(',')],
                                online_analysis=row['onlineAnalysis'], verbose=verbose, **kwargs)
            records.append(rec)
            if save:
                # Save as we go: a batch this long should survive an
                # interruption, and with skip_existing it can then resume.
                save_records([rec], verbose=False)
            if plot:
                plot_group(rec)
        except Exception as e:
            if on_error != 'log':
                raise
            failures.append((row['exp_name'], row['cell_label'], f'{type(e).__name__}: {e}'))
    print(f'analyzed {len(records)}/{len(groups)} groups'
          + (f' ({skipped} already stored, skipped)' if skipped else ''))
    # Two records sharing a key means one overwrote the other in the store --
    # data quietly not making it in, so say so rather than leaving a gap.
    seen = {}
    for rec in records:
        seen.setdefault(rec.key, []).append(f"{rec.exp_name}/{rec.cell_label}")
    clashes = {k: v for k, v in seen.items() if len(v) > 1}
    if clashes:
        print(f'WARNING: {len(clashes)} record key(s) produced by more than one group; '
              f'only the last was kept:')
        for k, who in clashes.items():
            print(f'  {k} <- {len(who)} groups')
    if failures:
        print(f'{len(failures)} failed:')
        for exp, cell, msg in failures[:20]:
            print(f'  {exp} {cell}: {msg[:110]}')
    return records


# --------------------------------------------------------------------------
# per-cell inspection
# --------------------------------------------------------------------------

def list_cells(groups: pd.DataFrame, show: bool = True, height: int = 400) -> pd.DataFrame:
    """Cells available in a group table, with the conditions each was recorded in.

    Use it to find the ``cell_id`` ('<experiment>/<cell label>') for
    :func:`inspect_cell`.
    """
    from retinanalysis.SCutils import explore as _sc
    return _sc.list_cells(groups, show=show, height=height)


def inspect_cell(cell: str, groups: pd.DataFrame, plot: bool = True,
                 show: bool = True, **kwargs):
    """Analyze and plot every recording of one cell, split by condition.

    ``cell`` is '<experiment>/<cell label>', e.g. ``'2026-04-23_E/Cell5'``; a
    bare label works when it is unambiguous. Returns the records.
    """
    from retinanalysis.SCutils import explore as _sc
    return _sc.inspect_cell(groups, cell, analyze=analyze_group,
                            plot=plot_group if plot else None, show=show, **kwargs)


# --------------------------------------------------------------------------
# stimulus rendering (natural image patch + the two discs)
# --------------------------------------------------------------------------

# NaturalImageFlashProtocol.m works in van Hateren pixels at this scale.
VH_MICRONS_PER_PIXEL = 6.6
VH_SHAPE = (1536, 1024)          # as MATLAB reads it: fread(fid, [1536 1024])


@lru_cache(maxsize=None)
def vh_image_path(image_name: str, image_set: str = 'VHsubsample_20160105'):
    """Locate ``imk<name>.iml`` in the turner package resources, or None."""
    from pathlib import Path
    from retinanalysis.config.settings import PROTOCOL_REPOS_ROOT

    if not PROTOCOL_REPOS_ROOT:
        return None
    root = Path(PROTOCOL_REPOS_ROOT)
    name = f'imk{str(image_name).strip()}.iml'
    for cand in root.glob(f'**/+resources/{image_set}/{name}'):
        return cand
    hits = list(root.glob(f'**/{image_set}/{name}')) or list(root.glob(f'**/{name}'))
    return hits[0] if hits else None


@lru_cache(maxsize=None)
def load_vh_contrast_image(image_name: str, image_set: str = 'VHsubsample_20160105'):
    """Load a van Hateren image the way the protocol does.

    Port of the loader in ``NaturalImageFlashProtocol.m``: big-endian uint16 read
    as a 1536x1024 matrix, rescaled so the brightest pixel is 1, then converted
    to contrast about the image mean. Returns ``(contrast_image, background)``
    with the image in MATLAB's (x, y) orientation so patch locations index it
    directly, or ``(None, nan)`` when the file is not on this machine.
    """
    path = vh_image_path(image_name, image_set)
    if path is None:
        return None, np.nan
    return _load_vh_contrast_image_path(str(path))


@lru_cache(maxsize=None)
def _load_vh_contrast_image_path(path_string: str):
    """Load one exact van Hateren image path with protocol normalization."""
    raw = np.fromfile(path_string, dtype='>u2')
    if raw.size < VH_SHAPE[0] * VH_SHAPE[1]:
        return None, np.nan
    # MATLAB fills [1536,1024] column-wise; the row-major read is its transpose.
    img = raw[:VH_SHAPE[0] * VH_SHAPE[1]].reshape(VH_SHAPE[1], VH_SHAPE[0]).T.astype(float)
    img = img / img.max()
    background = float(img.mean())
    return (img - background) / background, background


def image_patch(image_name: str, patch_location, aperture_diameter: float,
                image_set: str = 'VHsubsample_20160105', pad: float = 1.35,
                inner_diameter: float = 0.0):
    """The image patch a cell saw, in intensity units, plus its aperture mask.

    ``patch_location`` is ``currentPatchLocation`` (van Hateren pixels).
    ``inner_diameter`` > 0 makes the mask an annulus, which is what the
    LinearEquivalentAnnulus protocol shows. Returns
    ``(patch, mask, extent_um, background)``; ``patch`` is background *
    (1 + contrast) so it is directly comparable with the discs' equivalent
    intensities, and ``mask`` is True where the stimulus was visible.
    """
    contrast, background = load_vh_contrast_image(image_name, image_set)
    if contrast is None:
        return None, None, np.nan, np.nan
    x, y = (int(round(v)) for v in np.asarray(patch_location, dtype=float)[:2])
    half = int(round(aperture_diameter * pad / 2 / VH_MICRONS_PER_PIXEL))
    x0, x1 = max(x - half, 0), min(x + half, contrast.shape[0])
    y0, y1 = max(y - half, 0), min(y + half, contrast.shape[1])
    patch_contrast = contrast[x0:x1, y0:y1].T      # to (row, col) for imshow
    extent_um = half * VH_MICRONS_PER_PIXEL
    g = np.linspace(-extent_um, extent_um, patch_contrast.shape[1])
    h = np.linspace(-extent_um, extent_um, patch_contrast.shape[0])
    r = np.hypot(*np.meshgrid(g, h))
    mask = (r <= aperture_diameter / 2) & (r >= inner_diameter / 2)
    return background * (1 + patch_contrast), mask, extent_um, background


def stimulus_triplet_frames(
        params: Dict,
        patch_location=None,
        equivalent_intensity: Optional[float] = None,
        equivalent_intensity_cone: Optional[float] = None):
    """Render the three protocol frames on their recorded background.

    ``LinearEquivalentAnnulus.createPresentation`` keeps the full Stage canvas
    at ``backgroundIntensity``, replaces only the annulus with the image or an
    equivalent intensity, and overlays a center spot at
    ``backgroundIntensity * (1 + centerSpotContrast)``. This function mirrors
    that layering rather than making the masked regions transparent.
    """
    if patch_location is None:
        patch_location = params.get('currentPatchLocation')
    if equivalent_intensity is None:
        equivalent_intensity = float(params.get('equivalentIntensity', np.nan))
    if equivalent_intensity_cone is None:
        equivalent_intensity_cone = float(
            params.get('equivalentIntensityConeLin', np.nan))
    if not np.isfinite(equivalent_intensity):
        raise ValueError('the selected image trial has no finite equivalentIntensity')
    if not np.isfinite(equivalent_intensity_cone):
        raise ValueError(
            'the selected block has no cone-linearized equivalent intensity; '
            'use example_patch_params_from_blocks() to select a compatible block')

    outer = float(params.get('apertureDiameter')
                  or params.get('annulusOuterDiameter') or 200.0)
    inner = float(params.get('annulusInnerDiameter') or 0.0)
    patch, patch_mask, extent, image_background = image_patch(
        str(params.get('imageName')), patch_location, outer,
        str(params.get('currentImageSet', 'VHsubsample_20160105')),
        inner_diameter=inner)
    background = pd.to_numeric(params.get('backgroundIntensity'), errors='coerce')
    if not np.isfinite(background):
        background = image_background
    if not np.isfinite(background):
        raise ValueError('the selected image trial has no finite backgroundIntensity')
    background = float(background)

    if patch is None:
        shape = (151, 151)
        extent = outer * .675
    else:
        shape = patch.shape
    x = np.linspace(-extent, extent, shape[1])
    y = np.linspace(-extent, extent, shape[0])
    radius = np.hypot(*np.meshgrid(x, y))
    annulus_mask = (radius <= outer / 2) & (radius >= inner / 2)
    if patch_mask is not None and patch_mask.shape == shape:
        annulus_mask &= patch_mask

    def annulus_frame(value):
        frame = np.full(shape, background, dtype=float)
        if np.ndim(value):
            frame[annulus_mask] = np.asarray(value, dtype=float)[annulus_mask]
        else:
            frame[annulus_mask] = float(value)
        center_diameter = float(params.get('centerSpotDiameter') or 0.0)
        center_contrast = float(params.get('centerSpotContrast') or 0.0)
        if center_diameter > 0:
            frame[radius <= center_diameter / 2] = (
                background * (1 + center_contrast))
        return frame

    image_value = patch if patch is not None else background
    frames = (
        annulus_frame(image_value),
        annulus_frame(equivalent_intensity),
        annulus_frame(equivalent_intensity_cone),
    )
    return frames, float(extent), background


def plot_stimulus_example(params: Dict, patch_location=None,
                          equivalent_intensity: Optional[float] = None,
                          equivalent_intensity_cone: Optional[float] = None,
                          figsize: Tuple[float, float] = (10.0, 3.4)):
    """The three stimuli for one patch: image, standard disc, cone-linearized disc.

    ``params`` is an epoch-parameter dict from an ``image`` trial — it carries
    the patch location and both equivalent intensities. Everything is drawn on
    one grey scale so the discs can be compared with the patch they replace.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if equivalent_intensity is None:
        equivalent_intensity = float(params.get('equivalentIntensity', np.nan))
    if equivalent_intensity_cone is None:
        equivalent_intensity_cone = float(params.get('equivalentIntensityConeLin', np.nan))
    outer = params.get('apertureDiameter') or params.get('annulusOuterDiameter') or 200.0
    inner = float(params.get('annulusInnerDiameter') or 0.0)
    aperture = float(outer)

    frames, extent, background = stimulus_triplet_frames(
        params, patch_location=patch_location,
        equivalent_intensity=equivalent_intensity,
        equivalent_intensity_cone=equivalent_intensity_cone)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    vmax = float(max(np.nanmax(frame) for frame in frames))
    axes[0].imshow(frames[0], cmap='gray', vmin=0, vmax=vmax, origin='lower',
                   extent=[-extent, extent, -extent, extent], interpolation='nearest')
    axes[0].set_xlabel('µm')
    axes[0].set_ylabel('µm')
    axes[0].set_title(f"image patch {params.get('imagePatchIndex')}\n"
                      f"imk{params.get('imageName')}", fontsize=9)

    for ax, frame, value, name in (
            (axes[1], frames[1], equivalent_intensity, 'linear-equivalent disc'),
            (axes[2], frames[2], equivalent_intensity_cone, 'cone-linearized disc')):
        ax.imshow(frame, cmap='gray', vmin=0, vmax=vmax, origin='lower',
                  extent=[-extent, extent, -extent, extent],
                  interpolation='nearest')
        ax.set_title(f'{name}\nintensity {value:.4f}', fontsize=9)
        ax.set_xlabel('µm')
    geom = (f'annulus {inner:g}-{aperture:g} µm' if inner > 0
            else f'aperture {aperture:g} µm')
    fig.suptitle(f'{geom} | image mean {background:.4f}'
                 if np.isfinite(background) else geom, fontsize=9, y=1.02)
    fig.tight_layout()
    return fig


def example_patch_params(exp_name: str, block_id: int,
                         patch_index: Optional[float] = None,
                         image_name: Optional[str] = None):
    """Epoch parameters of one ``image`` trial, for :func:`plot_stimulus_example`."""
    import retinanalysis as ra

    ep = ra.StimBlock(exp_name, int(block_id), verbose=False).df_epochs
    for p in ep['epoch_parameters']:
        if category_of(p.get('stimulusTag')) != 'image':
            continue
        if image_name is not None and str(p.get('imageName')) != str(image_name):
            continue
        if patch_index is None or float(p.get('imagePatchIndex', -1)) == float(patch_index):
            return p
    detail = f'image {image_name}' if image_name is not None else f'patch {patch_index}'
    raise ValueError(f'no image trial for {detail} in block {block_id}')


def example_patch_params_sequence(exp_name: str, block_id: int,
                                  image_name: Optional[str] = None,
                                  count: int = 4,
                                  patch_index: Optional[float] = None):
    """Return several unique image-trial parameter sets from one block."""
    import retinanalysis as ra

    wanted = max(1, int(count))
    ep = ra.StimBlock(exp_name, int(block_id), verbose=False).df_epochs
    found, seen = [], set()
    for params in ep['epoch_parameters']:
        if category_of(params.get('stimulusTag')) != 'image':
            continue
        if image_name is not None and str(params.get('imageName')) != str(image_name):
            continue
        index = pd.to_numeric(params.get('imagePatchIndex'), errors='coerce')
        if patch_index is not None and index != float(patch_index):
            continue
        cone_value = pd.to_numeric(params.get('equivalentIntensityConeLin'),
                                   errors='coerce')
        disc_value = pd.to_numeric(params.get('equivalentIntensity'), errors='coerce')
        key = (str(params.get('imageName')), float(index) if np.isfinite(index) else None)
        if key in seen or not np.isfinite(cone_value) or not np.isfinite(disc_value):
            continue
        seen.add(key)
        found.append(params)
        if len(found) >= wanted:
            break
    detail = f'image {image_name}' if image_name is not None else 'the selected block'
    if not found:
        raise ValueError(f'no cone-linearized image trials are available for {detail}')
    return found


def example_patch_params_from_blocks(blocks: pd.DataFrame,
                                     patch_index: Optional[float] = None,
                                     image_name: Optional[str] = None):
    """Image-trial parameters from a block with a recorded cone intensity.

    Section 2 uses this instead of choosing a positional row: older annulus
    blocks can be present in the discovery table but predate cone linearization.
    Rows whose block metadata carries ``linearizeCones`` are tried first, then
    the returned epoch parameters are verified against the recorded
    ``equivalentIntensityConeLin`` value.
    """
    required = {'exp_name', 'block_id'}
    missing = required.difference(blocks.columns)
    if missing:
        raise ValueError(f'blocks is missing required columns: {sorted(missing)}')
    if blocks.empty:
        raise ValueError('no blocks are available for the Section 2 example')

    candidates = blocks.copy()
    if image_name is not None and 'imageName' in candidates:
        candidates = candidates.loc[
            candidates['imageName'].astype(str).eq(str(image_name))].copy()
        if candidates.empty:
            raise ValueError(f'image {image_name!r} is not available in the selected date')
    if 'linearizeCones' in candidates:
        candidates['_cone_priority'] = pd.to_numeric(
            candidates['linearizeCones'], errors='coerce').notna()
        candidates = candidates.sort_values('_cone_priority', ascending=False,
                                             kind='stable')

    for row in candidates.itertuples(index=False):
        try:
            params = example_patch_params(
                row.exp_name, int(row.block_id), patch_index, image_name=image_name)
        except ValueError as error:
            if str(error).startswith('no image trial for'):
                continue
            raise
        cone_value = pd.to_numeric(params.get('equivalentIntensityConeLin'),
                                   errors='coerce')
        if np.isfinite(cone_value):
            return params
    raise ValueError(
        'none of the selected blocks records equivalentIntensityConeLin; '
        'choose a cone-linearized protocol or experiment')


def example_patch_sequence_from_blocks(blocks: pd.DataFrame,
                                       image_name: Optional[str] = None,
                                       count: int = 4,
                                       patch_index: Optional[float] = None):
    """Select several recorded patch triplets for the Section 2 schematic."""
    required = {'exp_name', 'block_id'}
    missing = required.difference(blocks.columns)
    if missing:
        raise ValueError(f'blocks is missing required columns: {sorted(missing)}')
    candidates = blocks.copy()
    if image_name is not None and 'imageName' in candidates:
        candidates = candidates.loc[
            candidates['imageName'].astype(str).eq(str(image_name))].copy()
    if 'linearizeCones' in candidates:
        candidates['_cone_priority'] = pd.to_numeric(
            candidates['linearizeCones'], errors='coerce').notna()
        candidates = candidates.sort_values('_cone_priority', ascending=False,
                                             kind='stable')
    for row in candidates.itertuples(index=False):
        try:
            return example_patch_params_sequence(
                row.exp_name, int(row.block_id), image_name=image_name,
                count=count, patch_index=patch_index)
        except ValueError as error:
            if str(error).startswith('no cone-linearized image trials'):
                continue
            raise
    raise ValueError(
        'none of the selected blocks has cone-linearized image trials for '
        f'image {image_name!r}')


def plot_stimulus_sequence(params_sequence: Sequence[Dict],
                           figsize: Tuple[float, float] = (12.0, 5.2)):
    """Publication-style tilted time sequence of image, disc, and cone disc."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle
    from matplotlib.transforms import Affine2D
    from retinanalysis.utils import style

    style.apply_publication_style()
    params_sequence = list(params_sequence)
    if not params_sequence:
        raise ValueError('params_sequence must contain at least one patch')

    triplets, finite_values = [], []
    for params in params_sequence:
        frames, _extent, _background = stimulus_triplet_frames(params)
        finite_values.extend(np.concatenate([frame.ravel() for frame in frames]))
        triplets.append((*frames, params))

    vmax = max(finite_values) if finite_values else 1.0
    cmap = plt.get_cmap('gray').copy()
    cmap.set_bad('white')
    frames = []
    for patch_number, (patch, disc, cone, params) in enumerate(triplets, start=1):
        frames.extend([
            (patch, 'image', patch_number, params),
            (disc, 'disc', patch_number, params),
            (cone, 'cone disc', patch_number, params),
        ])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('white')
    width, height = 1.55, 1.55
    dx, dy = .72, .28
    skew = -8
    for index, (frame, label, patch_number, params) in enumerate(frames):
        x, y = index * dx, index * dy
        transform = Affine2D().skew_deg(0, skew) + ax.transData
        ax.imshow(frame, cmap=cmap, vmin=0, vmax=vmax, origin='lower',
                  extent=[x, x + width, y, y + height], interpolation='nearest',
                  transform=transform, zorder=index + 1)
        ax.add_patch(Rectangle((x, y), width, height, fill=False, ec='#333333',
                               lw=.7, transform=transform, zorder=index + 1.2))
        if index < 3:
            ax.text(x + width / 2, y + height + .13, label, ha='center', va='bottom',
                    fontsize=8, transform=transform, zorder=len(frames) + 2)
        if label == 'image':
            ax.text(x + .08, y + .08, f'patch {patch_number}', ha='left', va='bottom',
                    fontsize=7, color='white', transform=transform,
                    zorder=len(frames) + 2,
                    bbox=dict(facecolor='black', alpha=.45, edgecolor='none', pad=1.2))

    transform = Affine2D().skew_deg(0, skew) + ax.transData
    ax.add_patch(FancyArrowPatch(
        (.15, -.18), (width * .62, -.18), arrowstyle='-|>', mutation_scale=11,
        lw=1.3, color='black', transform=transform, zorder=len(frames) + 4))
    ax.text(width * .67, -.18, 'x', va='center', fontsize=9,
            transform=transform, zorder=len(frames) + 4)
    ax.add_patch(FancyArrowPatch(
        (.15, -.18), (.15, height * .42), arrowstyle='-|>', mutation_scale=11,
        lw=1.3, color='black', transform=transform, zorder=len(frames) + 4))
    ax.text(.15, height * .48, 'y', ha='center', fontsize=9, transform=transform)
    last = len(frames) - 1
    time_start = (2 * dx + width * .1, 2 * dy - .7)
    time_end = (last * dx + width * .95, last * dy - .7)
    ax.add_patch(FancyArrowPatch(
        time_start, time_end, arrowstyle='-|>', mutation_scale=12,
        lw=1.5, color='#333333', transform=transform,
        zorder=len(frames) + 4))
    ax.text((time_start[0] + time_end[0]) / 2,
            (time_start[1] + time_end[1]) / 2 - .16, 'time',
            ha='center', va='top', color='#333333', fontsize=9,
            transform=transform, zorder=len(frames) + 4)
    ax.set_xlim(-.35, last * dx + width + .55)
    ax.set_ylim(-.75, last * dy + height + .5)
    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.suptitle('Interleaved natural-image, equivalent-disc, and cone-disc flashes',
                 fontsize=11, y=.98)
    fig.tight_layout()
    return fig


def stimulus_example_widget(blocks: pd.DataFrame,
                            patch_index: Optional[float] = None,
                            sequence_length: int = 4):
    """Dropdown that redraws a tilted sequence of recorded stimulus triplets."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    if 'imageName' not in blocks:
        raise ValueError('blocks is missing required column: imageName')
    image_names = sorted({str(value) for value in blocks['imageName']
                          if pd.notna(value) and str(value).strip()})
    if not image_names:
        raise ValueError('the selected date has no recorded image names')

    dropdown = widgets.Dropdown(
        options=image_names, value=image_names[0], description='Image name:',
        style={'description_width': 'initial'})
    output = widgets.Output()

    def render(_change=None):
        # Replace the model payload directly so one selection cannot leave a
        # second copy of the previous sequence in the notebook front end.
        output.outputs = ()
        params = example_patch_sequence_from_blocks(
            blocks, patch_index=patch_index, image_name=dropdown.value,
            count=sequence_length)
        fig = plot_stimulus_sequence(params)
        output.append_display_data(fig)
        plt.close(fig)

    dropdown.observe(render, names='value')
    render()
    return widgets.VBox([dropdown, output])


def describe_cell(cell: str, groups: pd.DataFrame, show: bool = True, **kwargs):
    """Basic information about one cell before analyzing any of its recordings.

    Cell type, how many conditions it was recorded in, and one row per
    condition. ``cell`` is '<experiment>/<cell label>'.
    """
    from retinanalysis.SCutils import explore as _sc
    return _sc.describe_cell(groups, cell, show=show, **kwargs)
