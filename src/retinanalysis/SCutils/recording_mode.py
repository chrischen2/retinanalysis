"""Shared recording-mode parsing and amplifier metadata for single-cell data.

:func:`parse_single_cell_recording_modes` is the protocol-independent parser.
It treats the normally reliable epoch-group ``recordingTechnique`` as the
primary anchor, positive series resistance as definitive whole-cell evidence,
and the cell's acquisition order as a one-way cell-attached -> whole-cell
sequence. It deliberately does not interpret zero resistance or failure to
detect spikes as proof of a recording family.

The older :func:`resolve_recording_mode` and :func:`check_series_resistance`
APIs remain for protocol modules that perform a block-local amplifier audit.

Also here because it is read from the same place in the h5:
:func:`read_stage_ndfs`, the fixed neutral-density filters in the light path,
which are recorded separately from the filter-wheel setting and which the wheel
setting does not imply.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Whole-cell epochs whose access resistance exceeds this are discarded: the
# series resistance sits between the amplifier and the cell, so above ~20 MOhm
# the recorded current is badly filtered and attenuated. Ohms, so 20 MOhm.
MAX_SERIES_RESISTANCE = 20e6


# --------------------------------------------------------------------------
# series resistance: is this recording what onlineAnalysis says it is?
# --------------------------------------------------------------------------

def _amp_epoch_groups(exp_name: str, block_id: int, amp: str = 'Amp1') -> List[str]:
    """h5 epoch-group paths for a block's amplifier responses, in ``amp_data`` order.

    Built from the same query ``get_epochblock_amp_data`` uses, so element *i*
    of anything read through these paths lines up with row *i* of
    ``SCResponseBlock.amp_data`` and of ``StimBlock.df_epochs``.
    """
    from retinanalysis.utils.datajoint_utils import get_epochblock_response_query

    df = get_epochblock_response_query(exp_name, int(block_id)).fetch(format='frame').reset_index()
    df = df[df['device_name'].astype(str).eq(amp)]
    return [str(p).split('/responses/')[0] for p in df['h5path'].values]


def _amp_response_table(block_ids: Sequence[int], amp: str = 'Amp1') -> pd.DataFrame:
    """Fetch the minimal amplifier response metadata for many blocks at once.

    ``_amp_epoch_groups`` follows the general response loader and performs one
    DataJoint query per block.  That is appropriate for a single recording but
    made a dataset-wide series-resistance audit spend minutes repeatedly
    fetching and decoding response rows. This projection asks only for the
    path, sample rate, and block id, once for the entire block set.
    """
    from retinanalysis.config import schema

    ids = sorted({int(block_id) for block_id in block_ids})
    if not ids:
        return pd.DataFrame(columns=['response_id', 'block_id', 'h5path', 'sample_rate'])
    epochs = (schema.Epoch & [{'parent_id': block_id} for block_id in ids]).proj(
        block_id='parent_id', epoch_id='id')
    responses = epochs * schema.Response.proj(
        ..., epoch_id='parent_id', response_id='id')
    return ((responses & {'device_name': amp})
            .proj('block_id', 'h5path', 'sample_rate').to_pandas().reset_index()
            .sort_values('response_id').reset_index(drop=True))


def _amp_epoch_groups_by_block(block_ids: Sequence[int],
                               amp: str = 'Amp1',
                               response_table: Optional[pd.DataFrame] = None
                               ) -> Dict[int, List[str]]:
    """Fetch amplifier epoch-group paths for many blocks in one query."""
    paths = (_amp_response_table(block_ids, amp=amp)
             if response_table is None else response_table.copy())
    paths['epoch_group'] = paths['h5path'].astype(str).str.split('/responses/').str[0]
    return {int(block_id): group['epoch_group'].tolist()
            for block_id, group in paths.groupby('block_id', sort=False)}


def _amp_trace_samples(df: pd.DataFrame, amp: str = 'Amp1', n_trials: int = 12,
                       verbose: bool = True,
                       response_table: Optional[pd.DataFrame] = None,
                       n_trials_by_block: Optional[Dict[int, int]] = None,
                       trace_seconds: Optional[float] = None
                       ) -> Dict[int, Tuple[np.ndarray, float]]:
    """Load a small raw-trace sample per block without constructing StimBlocks.

    Response paths are fetched once, H5 files are opened once per experiment,
    and only the trials needed by :func:`trace_is_spiking` are read. This avoids
    loading frame-monitor data and repeating DataJoint joins during recording
    mode discovery.
    """
    import h5py
    from retinanalysis.utils.datajoint_utils import (
        get_h5_file,
        read_h5_response_trace,
    )

    paths = (_amp_response_table(df['block_id'], amp=amp)
             if response_table is None else response_table)
    out: Dict[int, Tuple[np.ndarray, float]] = {}
    progress = tqdm(total=len(df), desc='Sampling block traces', unit='block',
                    disable=not verbose or df.empty)
    for exp_name, blocks in df.groupby('exp_name', sort=False):
        try:
            h5 = h5py.File(get_h5_file(str(exp_name)), 'r')
        except Exception as e:
            if verbose:
                print(f'  {exp_name}: cannot open the h5 for trace sampling '
                      f'({type(e).__name__})')
            progress.update(len(blocks))
            continue
        with h5:
            for block_id in blocks['block_id']:
                block_id = int(block_id)
                block_trials = (n_trials if n_trials_by_block is None
                                else n_trials_by_block.get(block_id, n_trials))
                rows = paths[paths['block_id'].eq(block_id)].head(block_trials)
                try:
                    rates = rows['sample_rate'].dropna().astype(float).unique()
                    if len(rates) != 1:
                        raise ValueError(f'expected one sample rate, found {len(rates)}')
                    sample_slice = (None if trace_seconds is None else
                                    slice(0, max(1, int(trace_seconds * rates[0]))))
                    traces = [read_h5_response_trace(h5, path, sample_slice=sample_slice)
                              for path in rows['h5path']]
                    if traces:
                        out[int(block_id)] = (np.asarray(traces), float(rates[0]))
                except Exception as e:
                    if verbose:
                        print(f'  {exp_name} block {int(block_id)}: cannot sample the trace '
                              f'({type(e).__name__}: {e})')
                progress.update(1)
    progress.close()
    return out


def _epoch_series_resistance(epoch_group, amp: str = 'Amp1') -> float:
    """``stimulus:<amp>:seriesResistance`` for one epoch group, in ohms.

    Symphony writes the amplifier's device configuration into
    ``stimuli/<amp>-<uuid>/dataConfigurationSpans/span_0/<amp>``, which is what
    ``epoch.protocolSettings('stimulus:Amp1:seriesResistance')`` returns in the
    MATLAB. Amplifier backgrounds and responses are fallbacks for protocols
    with no amplifier stimulus. NaN when all three lack the attribute.
    """
    # AUISQL response paths can point directly at a root-level trace dataset,
    # not at a Symphony epoch group. Such a node has no configuration tree.
    if not hasattr(epoch_group, 'get'):
        return np.nan
    # LED protocols can carry only a constant amplifier background, with no
    # amplifier stimulus. Read that same epoch's background/response config
    # as a fallback, without traversing device links to the whole experiment.
    device_nodes = []
    for section in ('stimuli', 'backgrounds', 'responses'):
        container = epoch_group.get(section)
        if container is not None:
            for name in container:
                device_nodes.append((name, container[name]))
    for dev, device_node in device_nodes:
        if str(dev).split('-')[0] != amp:
            continue
        # Some legacy responses are stored directly as datasets. They have no
        # configuration children, so continue to a matching stimulus or
        # background rather than failing the entire block.
        if not hasattr(device_node, 'get'):
            continue
        spans = device_node.get('dataConfigurationSpans')
        if spans is None:
            continue
        for span in spans:
            span_node = spans[span]
            # A few legacy files store a response or configuration span as a
            # dataset. Like a dataset-valued device node above, it cannot have
            # an amplifier child and should not abort the rest of the scan.
            if not hasattr(span_node, 'get'):
                continue
            node = span_node.get(amp)
            if node is not None and 'seriesResistance' in node.attrs:
                return float(node.attrs['seriesResistance'])
    return np.nan


def read_series_resistance(exp_name: str, block_id: int, amp: str = 'Amp1',
                           h5=None) -> np.ndarray:
    """Per-epoch ``stimulus:<amp>:seriesResistance`` for a block, in ohms.

    One value per epoch, ordered to match ``SCResponseBlock.amp_data``. In
    practice the amplifier configuration is set once per block so the array is
    constant, but it is read per epoch because that is where Symphony stores it
    and because the cutoff is applied per epoch.

    A cell-attached recording has no access resistance and reads exactly 0; a
    whole-cell recording reads the value the experimenter entered on the
    amplifier. Pass an open :class:`h5py.File` as ``h5`` to read many blocks of
    one experiment without reopening the file.
    """
    import h5py
    from retinanalysis.utils.datajoint_utils import get_h5_file

    groups = _amp_epoch_groups(exp_name, int(block_id), amp=amp)
    if not groups:
        return np.zeros(0, dtype=float)

    def _read(f):
        out = []
        for g in groups:
            node = f.get(g)
            out.append(_epoch_series_resistance(node, amp) if node is not None else np.nan)
        return np.asarray(out, dtype=float)

    if h5 is not None:
        return _read(h5)
    with h5py.File(get_h5_file(exp_name), 'r') as f:
        return _read(f)


def mode_family(online_analysis) -> str:
    """The recording mode behind an ``onlineAnalysis`` label.

    ``'extracellular'`` is a cell-attached spike recording; ``'exc'`` and
    ``'inh'`` are whole-cell voltage clamp at two holding potentials. Anything
    else (``'none'``, missing) gives ``''`` — unknown, not a mismatch.
    """
    m = str(online_analysis).strip().lower()
    if m == 'extracellular':
        return 'cell-attached'
    if m in ('exc', 'inh'):
        return 'whole-cell'
    return ''


def _recording_technique_family(value) -> str:
    """Normalize an epoch-group recording-technique label to one family."""
    if value is None or (np.isscalar(value) and pd.isna(value)):
        return ''
    text = str(value).strip().lower().replace('_', '-').replace(' ', '-')
    if text in ('cell-attached', 'cellattached'):
        return 'cell-attached'
    if text in ('whole-cell', 'wholecell'):
        return 'whole-cell'
    return ''


def parse_single_cell_recording_modes(
        records: pd.DataFrame, *,
        cell_columns: Sequence[str] = ('exp_name', 'cell_label'),
        block_column: str = 'block_id',
        time_column: str = 'start_time',
        technique_column: str = 'recording_technique',
        online_analysis_column: str = 'onlineAnalysis',
        series_resistance_column: str = 'series_resistance',
        mean_current_column: str = 'mean_current',
        spike_evidence_column: Optional[str] = None,
        classifier_family_column: Optional[str] = None) -> pd.DataFrame:
    """Parse recording mode once per epoch block, in acquisition order.

    This is the protocol-independent parser for single-cell block/epoch tables.
    It uses the information that is normally documented correctly, while
    making the two acquisition invariants explicit:

    * every epoch in an epoch block has one recording family; and
    * when one cell has both families, cell-attached recording comes first and
      whole-cell recording follows after a single irreversible transition.

    Evidence is applied in this order:

    1. any positive series-resistance reading forces the whole-cell family;
    2. epoch-group ``recordingTechnique`` supplies the family;
    3. an optional high-confidence block classifier can correct or fill the
       family;
    4. a valid ``onlineAnalysis`` label is a fallback when both technique and
       classifier evidence are absent; and
    5. remaining gaps inherit the cell's chronological family/transition.

    The caller should pass classifier families only after applying its desired
    confidence threshold. For blocks used to train that classifier, these must
    be out-of-fold predictions made without that cell in the training fold.

    Zero or missing resistance never establishes cell-attached, and absence of
    detected spikes never establishes whole-cell. Once whole-cell is known,
    ``onlineAnalysis`` supplies ``exc``/``inh`` when documented; otherwise the
    sign of ``mean_current`` does (negative -> ``exc``, nonnegative -> ``inh``).

    The returned frame has the same rows and order as ``records``. A decision
    is made on the pooled evidence for a block and broadcast to every one of
    its rows, so per-epoch callers cannot split one block across modes. Added
    columns are ``recording_order``, ``recording_family``,
    ``recording_family_source``, ``rec_type``, and ``rec_note``.
    """
    frame = records.copy()
    cell_columns = tuple(cell_columns)
    required = set(cell_columns) | {block_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f'records is missing identity columns: {sorted(missing)}')

    output_columns = (
        'recording_order', 'recording_family', 'recording_family_source',
        'rec_type', 'rec_note')
    frame = frame.drop(columns=[c for c in output_columns if c in frame],
                       errors='ignore')
    if frame.empty:
        for column in output_columns:
            frame[column] = pd.Series(dtype='Int64' if column == 'recording_order'
                                      else object)
        return frame

    work = frame.copy()
    if technique_column not in work:
        if 'group_properties' in work:
            work[technique_column] = work['group_properties'].apply(
                lambda value: value.get('recordingTechnique', '')
                if isinstance(value, dict) else '')
        else:
            work[technique_column] = ''
    for column, default in ((online_analysis_column, ''),
                            (series_resistance_column, np.nan),
                            (mean_current_column, np.nan)):
        if column not in work:
            work[column] = default
    if spike_evidence_column is not None and spike_evidence_column not in work:
        raise ValueError(f'records is missing spike evidence column '
                         f'{spike_evidence_column!r}')
    if (classifier_family_column is not None
            and classifier_family_column not in work):
        raise ValueError(f'records is missing classifier family column '
                         f'{classifier_family_column!r}')

    keys = list(cell_columns) + [block_column]

    def unique_nonempty(values, normalize):
        return sorted({normalized for value in values
                       if (normalized := normalize(value))})

    def normalized_online_label(value):
        if value is None or (np.isscalar(value) and pd.isna(value)):
            return ''
        label = str(value).strip().lower()
        return label if label in ('extracellular', 'exc', 'inh') else ''

    rows = []
    for key, block in work.groupby(keys, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        techniques = unique_nonempty(
            block[technique_column], _recording_technique_family)
        labels = unique_nonempty(
            block[online_analysis_column], normalized_online_label)
        label_families = sorted({mode_family(value) for value in labels})
        resistance = pd.to_numeric(block[series_resistance_column], errors='coerce')
        means = pd.to_numeric(block[mean_current_column], errors='coerce').dropna()
        positive_rs = resistance[resistance.gt(0)]
        classifier_families = (
            [] if classifier_family_column is None else
            unique_nonempty(block[classifier_family_column],
                            _recording_technique_family))
        if (spike_evidence_column is not None
                and block[spike_evidence_column].fillna(False).astype(bool).any()
                and 'cell-attached' not in classifier_families):
            classifier_families.append('cell-attached')
            classifier_families.sort()
        notes = []

        if len(techniques) > 1:
            notes.append('conflicting recordingTechnique values within one block')
        if len(label_families) > 1:
            notes.append('conflicting onlineAnalysis families within one block')
        elif len(labels) > 1:
            notes.append('conflicting whole-cell polarities within one block')

        if not positive_rs.empty:
            family = 'whole-cell'
            source = 'positive series resistance'
            notes.append(f'positive series resistance '
                         f'({float(positive_rs.median()) / 1e6:.2f} MOhm) '
                         'forces whole-cell')
            if techniques == ['cell-attached']:
                notes.append('overrides recordingTechnique=cell-attached')
        elif len(techniques) == 1:
            if (len(classifier_families) == 1
                    and classifier_families[0] != techniques[0]):
                family = classifier_families[0]
                source = 'high-confidence block classifier'
                notes.append(f'high-confidence block classifier corrects '
                             f'recordingTechnique={techniques[0]} to {family}')
            else:
                family = techniques[0]
                source = 'recordingTechnique'
                notes.append(f'recordingTechnique is {family}')
        elif len(classifier_families) == 1:
            family = classifier_families[0]
            source = 'high-confidence block classifier'
            notes.append(f'high-confidence block classifier establishes {family}')
        elif len(techniques) == 0 and len(label_families) == 1:
            family = label_families[0]
            source = 'onlineAnalysis fallback'
            notes.append(f'recordingTechnique missing; onlineAnalysis '
                         f'establishes {family}')
        else:
            family = ''
            source = ''

        start = (pd.to_datetime(block[time_column], errors='coerce').min()
                 if time_column in block else pd.NaT)
        numeric_block = pd.to_numeric(
            pd.Series([key[-1]]), errors='coerce').iloc[0]
        rows.append({
            **dict(zip(keys, key)), '_recording_time': start,
            '_numeric_block': numeric_block,
            '_text_block': str(key[-1]),
            '_labels': labels,
            '_mean_current': float(means.mean()) if not means.empty else np.nan,
            'recording_family': family,
            'recording_family_source': source,
            '_notes': notes,
        })

    blocks = pd.DataFrame(rows)
    blocks['_numeric_block_missing'] = blocks['_numeric_block'].isna()
    sort_columns = list(cell_columns) + [
        '_recording_time', '_numeric_block_missing', '_numeric_block', '_text_block']
    blocks = (blocks.sort_values(sort_columns, na_position='last', kind='stable')
              .reset_index(drop=True))
    blocks['recording_order'] = blocks.groupby(
        list(cell_columns), dropna=False, sort=False).cumcount()

    for _, positions in blocks.groupby(
            list(cell_columns), dropna=False, sort=False).groups.items():
        ordered = list(positions)
        families = blocks.loc[ordered, 'recording_family']
        whole_positions = [position for position in ordered
                           if families.loc[position] == 'whole-cell']
        cell_positions = [position for position in ordered
                          if families.loc[position] == 'cell-attached']
        if whole_positions:
            first_whole = whole_positions[0]
            transition_order = int(blocks.loc[first_whole, 'recording_order'])
            has_cell_before = any(
                int(blocks.loc[position, 'recording_order']) < transition_order
                for position in cell_positions)
            for position in ordered:
                order = int(blocks.loc[position, 'recording_order'])
                previous = blocks.loc[position, 'recording_family']
                if order >= transition_order:
                    if previous != 'whole-cell':
                        blocks.at[position, 'recording_family'] = 'whole-cell'
                        blocks.at[position, 'recording_family_source'] = (
                            'chronology after whole-cell transition')
                        blocks.at[position, '_notes'].append(
                            'whole-cell transition already occurred; '
                            'a later block cannot be cell-attached')
                elif not previous:
                    inferred = 'cell-attached' if has_cell_before else 'whole-cell'
                    blocks.at[position, 'recording_family'] = inferred
                    blocks.at[position, 'recording_family_source'] = (
                        'chronology before whole-cell transition' if has_cell_before
                        else 'single documented family in cell')
                    blocks.at[position, '_notes'].append(
                        f'chronology infers {inferred}')
        elif cell_positions:
            for position in ordered:
                if not blocks.loc[position, 'recording_family']:
                    blocks.at[position, 'recording_family'] = 'cell-attached'
                    blocks.at[position, 'recording_family_source'] = (
                        'single documented family in cell')
                    blocks.at[position, '_notes'].append(
                        'all documented blocks for this cell are cell-attached')

    rec_types, notes = [], []
    for _, row in blocks.iterrows():
        family = row['recording_family']
        labels = row['_labels']
        note_parts = list(row['_notes'])
        if family == 'cell-attached':
            rec_type = 'extracellular'
        elif family == 'whole-cell':
            subtypes = sorted({value for value in labels if value in ('exc', 'inh')})
            if len(subtypes) == 1:
                rec_type = subtypes[0]
                note_parts.append(f'polarity from onlineAnalysis={rec_type}')
            elif np.isfinite(row['_mean_current']):
                rec_type = 'exc' if row['_mean_current'] < 0 else 'inh'
                note_parts.append(
                    f"polarity from mean current ({row['_mean_current']:.3g})")
            else:
                rec_type = ''
                note_parts.append('whole-cell polarity unresolved')
        else:
            rec_type = ''
            note_parts.append('recording family unresolved')
        rec_types.append(rec_type)
        notes.append('; '.join(dict.fromkeys(note_parts)))
    blocks['rec_type'] = rec_types
    blocks['rec_note'] = notes

    decision_columns = keys + list(output_columns)
    decisions = blocks[decision_columns]
    result = frame.merge(decisions, on=keys, how='left', validate='many_to_one',
                         sort=False)
    return result




def trace_is_spiking(amp_data, sample_rate: float, n_trials: int = 12,
                     min_fraction: float = 0.7, detector_kwargs: Optional[dict] = None) -> bool:
    """Does this block's raw data actually contain spikes?

    Runs the spike detector over the first ``n_trials`` trials and asks whether
    at least ``min_fraction`` of them come back with any spikes. The detector's
    own spike-factor test is what decides per trial, so this is the same
    judgement the analysis would make, just sampled — a subsample is enough to
    tell a cell-attached recording from a voltage-clamp one, and it costs a
    twentieth of a full pass.

    The defaults sit in a gap the data actually leaves. Measured over every
    spotWithAnnularGrating block whose label and reading disagree, a genuinely
    cell-attached recording has spikes in 100% of its trials and a whole-cell
    one in 0-30%, with two borderline blocks at ~43%. 0.7 separates those
    cleanly; 0.5 would sit on top of the borderline pair and split one cell
    across two recording modes.
    """
    from retinanalysis.utils.spike_detector import detector

    data = np.asarray(amp_data, dtype=float)
    if data.ndim != 2 or data.shape[0] == 0:
        return False
    sample = data[:max(1, min(n_trials, data.shape[0]))]
    spike_times, _, _ = detector(sample, sample_rate=float(sample_rate),
                                 **(detector_kwargs or {}))
    return float(np.mean([len(s) > 0 for s in spike_times])) >= min_fraction


def prominent_event_width_ms(amp_data, sample_rate: float) -> float:
    """Lower-quartile half-prominence width of prominent raw events.

    Check both polarities without high-pass filtering: broad synaptic currents
    can ring through the spike detector's high-pass filter. This independent
    shape check is used only by the block-level discovery policy. Use the
    lower quartile of the 20 largest events per trial so real narrow spikes
    can coexist with slower fluctuations without being voted away.
    """
    from scipy.signal import find_peaks, peak_widths

    widths = []
    for trace in np.asarray(amp_data, dtype=float):
        candidates = []
        for sign in (-1, 1):
            peaks, props = find_peaks(sign * trace, prominence=0,
                                     wlen=max(3, int(.02 * sample_rate)))
            if not len(peaks):
                continue
            selected = np.argsort(props['prominences'])[-20:]
            prominence_data = tuple(props[k][selected] for k in (
                'prominences', 'left_bases', 'right_bases'))
            w = peak_widths(sign * trace, peaks[selected],
                            prominence_data=prominence_data)[0]
            candidates.extend(zip(props['prominences'][selected], w))
        if candidates:
            strongest = sorted(candidates, reverse=True)[:20]
            widths.extend(width for _, width in strongest)
    return float(np.quantile(widths, .25) * 1000 / sample_rate) if widths else np.nan


def resolve_recording_mode(online_analysis, series_resistance, amp_data=None,
                           sample_rate: float = 1e4,
                           detector_kwargs: Optional[dict] = None) -> Tuple[str, str]:
    """The mode a block should be analyzed as, after checking its label against the amp.

    ``onlineAnalysis`` is a menu item the experimenter picks and can get wrong;
    ``stimulus:Amp1:seriesResistance`` is what the rig recorded. Where they
    disagree the reading wins, but the two directions are not symmetric:

    * **Rs > 0 against an ``extracellular`` label** is unambiguous — a
      cell-attached patch has no access resistance, so the cell was held
      whole-cell. The holding potential is not in the reading, so the polarity
      comes from the sign of the current, as
      ``linear_equivalent_disc.analyze_group`` already does: inward (negative)
      is 'exc', outward is 'inh'.
    * **Rs == 0 against an ``exc``/``inh`` label** is *not* unambiguous. It
      means either cell-attached or that the experimenter never filled the
      field in, and both occur in these datasets — sometimes on the same date,
      for different cells. So the relabel is confirmed against the data first
      (:func:`trace_is_spiking`) and only applied to a trace that really does
      contain spikes. A trace with none keeps its whole-cell label and says why.

    A label of ``'none'`` — which the experimenter simply never set, and which
    covers a large share of the linear-equivalent-disc blocks — has nothing to
    contradict, so the reading *determines* the mode rather than overruling it:
    positive Rs gives whole-cell with the polarity from the sign, and a 0 gives
    cell-attached if the trace has spikes and whole-cell if it does not.

    Returns ``(mode, note)``: ``mode`` is one of 'extracellular', 'exc', 'inh',
    and ``note`` is empty when the recorded label stood, else what was decided
    or changed and on what evidence.
    """
    recorded = str(online_analysis or 'none').strip().lower()
    family = mode_family(recorded)
    rs = np.nan if series_resistance is None else float(series_resistance)

    if not np.isfinite(rs):
        return recorded, ''

    if family == '':
        # Nothing recorded to agree or disagree with, so the amplifier and the
        # trace decide outright.
        if amp_data is None:
            return recorded, ''
        if rs > 0:
            polarity = 'exc' if float(np.mean(np.asarray(amp_data, dtype=float))) < 0 else 'inh'
            return polarity, (f"'{recorded}' resolved to '{polarity}': series resistance is "
                              f'{rs / 1e6:.1f} MOhm, so whole-cell; polarity from the sign '
                              f'of the current')
        if trace_is_spiking(amp_data, sample_rate, detector_kwargs=detector_kwargs):
            return 'extracellular', (f"'{recorded}' resolved to 'extracellular': series "
                                     f'resistance is 0 and the trace contains spikes')
        polarity = 'exc' if float(np.mean(np.asarray(amp_data, dtype=float))) < 0 else 'inh'
        return polarity, (f"'{recorded}' resolved to '{polarity}': the trace has no spikes, so "
                          f'whole-cell with the series resistance never set; polarity from the '
                          f'sign of the current')

    if rs > 0 and family == 'cell-attached':
        if amp_data is None:
            return recorded, ('series resistance is '
                              f'{rs / 1e6:.1f} MOhm, so this is whole-cell, but the trace '
                              'was not available to read the holding potential from')
        polarity = 'exc' if float(np.mean(np.asarray(amp_data, dtype=float))) < 0 else 'inh'
        return polarity, (f"relabelled '{recorded}' -> '{polarity}': series resistance is "
                          f'{rs / 1e6:.1f} MOhm, so the cell was held whole-cell; polarity '
                          f'from the sign of the current')

    if rs == 0 and family == 'whole-cell':
        if amp_data is None:
            return recorded, ('series resistance is 0, which would make this cell-attached, '
                              'but the trace was not available to confirm it')
        if trace_is_spiking(amp_data, sample_rate, detector_kwargs=detector_kwargs):
            return 'extracellular', (f"relabelled '{recorded}' -> 'extracellular': series "
                                     f'resistance is 0 and the trace contains spikes')
        return recorded, ('series resistance is 0 but the trace has no spikes, so the field '
                          'was never set rather than the recording being cell-attached; '
                          'label kept')

    return recorded, ''


def series_resistance_table(df: pd.DataFrame, amp: str = 'Amp1',
                            max_series_resistance: float = MAX_SERIES_RESISTANCE,
                            verbose: bool = True,
                            sample_one_per_block: bool = False,
                            groups_by_block: Optional[Dict[int, List[str]]] = None
                            ) -> pd.DataFrame:
    """Read the series resistance of every block in ``df``, one h5 open per date.

    Returns one row per ``block_id`` with the median / min / max reading and how
    many of its epochs sit above ``max_series_resistance``. Blocks whose h5 is
    missing come back with NaN rather than raising, so one absent file does not
    stop the audit. With ``sample_one_per_block=True``, treat the first epoch as
    representative of the block. This is suitable for quick discovery when the
    full analysis will still audit the recording itself.
    """
    import h5py
    from retinanalysis.utils.datajoint_utils import get_h5_file

    if groups_by_block is None:
        groups_by_block = _amp_epoch_groups_by_block(df['block_id'], amp=amp)
    rows = []
    progress = tqdm(total=len(df), desc='Reading block metadata', unit='block',
                    disable=not verbose)
    for exp, sub in df.groupby('exp_name', sort=True):
        try:
            f = h5py.File(get_h5_file(str(exp)), 'r')
        except Exception as e:
            if verbose:
                print(f'  {exp}: cannot open the h5 ({type(e).__name__}) — '
                      f'{len(sub)} block(s) have no series-resistance reading')
            f = None
        for bid in sub['block_id']:
            rs = np.zeros(0, dtype=float)
            if f is not None:
                try:
                    groups = groups_by_block.get(int(bid), ())
                    groups_to_read = groups[:1] if sample_one_per_block else groups
                    rs = np.asarray([
                        _epoch_series_resistance(f[group], amp)
                        for group in groups_to_read if group in f
                    ], dtype=float)
                except Exception as e:
                    if verbose:
                        print(f'  {exp} block {bid}: {type(e).__name__}: {e}')
            good = rs[np.isfinite(rs)]
            n_epochs = len(groups_by_block.get(int(bid), ()))
            n_high = int(np.sum(good > max_series_resistance))
            if sample_one_per_block and good.size:
                n_read = n_epochs
                n_high = n_epochs if n_high else 0
            else:
                n_read = int(good.size)
            rows.append({
                'block_id': int(bid),
                'series_resistance': float(np.median(good)) if good.size else np.nan,
                'series_resistance_min': float(good.min()) if good.size else np.nan,
                'series_resistance_max': float(good.max()) if good.size else np.nan,
                'n_epochs_rs': n_read,
                'n_epochs_high_rs': n_high,
            })
            progress.update(1)
        if f is not None:
            f.close()
    progress.close()
    return pd.DataFrame(rows)


def check_series_resistance(df: pd.DataFrame, amp: str = 'Amp1',
                            max_series_resistance: float = MAX_SERIES_RESISTANCE,
                            drop: bool = True, show: bool = True,
                            sample_series_resistance: bool = False,
                            detector_kwargs: Optional[dict] = None,
                            block_level_evidence: bool = False,
                            max_spike_width_ms: float = 1.5,
                            infer_from_raw_trace: bool = True,
                            trace_seconds: Optional[float] = None) -> pd.DataFrame:
    """Cross-check every block's ``onlineAnalysis`` label against the amplifier.

    ``onlineAnalysis`` is a menu item the experimenter picks; the amplifier's
    ``stimulus:Amp1:seriesResistance`` is what the rig actually recorded, and it
    is exactly 0 for cell-attached and positive for whole-cell. Where they
    disagree the reading wins and the block is **relabelled** — it is not thrown
    away — so a cell recorded cell-attached but labelled ``exc`` gets spike
    sorted, and one recorded whole-cell but labelled ``extracellular`` gets
    treated as current. :func:`resolve_recording_mode` makes the call, and
    :func:`analyze_group` applies the same rule itself, so a block analyzed
    directly is corrected too.

    Only two things are dropped: a block whose every epoch sits above
    ``max_series_resistance``, and blocks the caller drops via ``drop``.

    Adds these columns:

    ``series_resistance``
        Median reading over the block's epochs, in ohms.
    ``rs_mode`` / ``label_mode``
        'cell-attached', 'whole-cell' or '' (unknown), from the amplifier and
        from the ``onlineAnalysis`` label respectively.
    ``n_epochs_high_rs``
        Epochs above ``max_series_resistance`` (:func:`analyze_group` drops
        these individually; a block where *every* epoch is above it is dropped
        here since nothing would be left).
    ``onlineAnalysis``
        Rewritten where the amplifier overrules the recorded label.
    ``onlineAnalysis_recorded``
        The label as the experimenter set it, always kept.
    ``rs_flag``
        '' when the recorded label stood, else what changed and on what
        evidence.

    ``sample_series_resistance=True`` makes discovery substantially faster by
    using one epoch's representative amplifier setting. The per-recording
    analysis still audits the raw recording when it runs.

    Resolving a contradiction can need the block's raw trace, so this reads it
    only when the label, series resistance, and epoch-group
    ``recordingTechnique`` metadata do not already settle the mode.

    ``block_level_evidence=True`` disables epoch-group shortcuts and shared
    trace decisions. Unlabelled blocks are sampled independently even when
    resistance is missing; group annotations remain provenance only.
    A spike classification must also have prominent raw events narrower than
    ``max_spike_width_ms``; broader events are treated as synaptic current.

    ``infer_from_raw_trace=False`` uses metadata only and never samples raw
    responses. Unresolved labels remain explicit. ``trace_seconds`` limits
    each sampled epoch to its first seconds; None reads the full epoch.
    """
    if trace_seconds is not None and (not np.isfinite(trace_seconds) or trace_seconds <= 0):
        raise ValueError('trace_seconds must be positive or None (full epoch)')
    out = df.copy()
    response_table = _amp_response_table(out['block_id'], amp=amp)
    groups_by_block = _amp_epoch_groups_by_block(
        out['block_id'], amp=amp, response_table=response_table)
    table = series_resistance_table(
        out[['exp_name', 'block_id']].drop_duplicates(), amp=amp,
        max_series_resistance=max_series_resistance, verbose=show,
        sample_one_per_block=sample_series_resistance,
        groups_by_block=groups_by_block)
    out = out.merge(table, on='block_id', how='left')

    out['series_resistance_source'] = np.where(
        out.series_resistance.notna(), 'raw H5', 'unavailable')
    if block_level_evidence and 'epoch_series_resistance' in out:
        fallback = pd.to_numeric(out.epoch_series_resistance, errors='coerce')
        use_fallback = out.series_resistance.isna() & fallback.notna()
        out.loc[use_fallback, 'series_resistance'] = fallback[use_fallback]
        out.loc[use_fallback, 'series_resistance_source'] = 'epoch parameters'

    rs = out['series_resistance'].to_numpy(dtype=float)
    out['rs_mode'] = np.where(np.isnan(rs), '', np.where(rs > 0, 'whole-cell', 'cell-attached'))
    out['label_mode'] = out['onlineAnalysis'].apply(mode_family)
    out['onlineAnalysis_recorded'] = out['onlineAnalysis']

    # Two kinds of block need the trace: one whose label the reading contradicts,
    # and one that was never labelled at all -- the latter is most of the
    # linear-equivalent-disc dataset, where 'none' is the commonest entry.
    if 'recording_technique' in out:
        technique = out['recording_technique']
    elif 'group_properties' in out:
        technique = out['group_properties'].apply(
            lambda value: value.get('recordingTechnique', '')
            if isinstance(value, dict) else '')
    else:
        technique = pd.Series('', index=out.index)
    technique = technique.astype(str).str.strip().str.lower()
    if block_level_evidence and infer_from_raw_trace:
        technique = pd.Series('', index=out.index)
    contested = (out['rs_mode'].ne('') & out['label_mode'].ne('')
                 & out['rs_mode'].ne(out['label_mode']))
    unlabelled = (out['label_mode'].eq('')
                  & (out['rs_mode'].ne('')
                     | technique.isin(['cell-attached', 'whole-cell'])))
    needs_resolving = contested | unlabelled
    if block_level_evidence:
        needs_resolving |= out['label_mode'].eq('')
    all_high = (out['n_epochs_rs'] > 0) & out['n_epochs_high_rs'].eq(out['n_epochs_rs'])

    # When epoch-group metadata says cell-attached and the amplifier does not
    # contradict it, no raw response is needed merely to prove that it spikes.
    metadata_cell_attached = (out['label_mode'].ne('cell-attached')
                              & technique.eq('cell-attached')
                              & ~out['rs_mode'].eq('whole-cell'))
    out.loc[metadata_cell_attached, 'onlineAnalysis'] = 'extracellular'
    needs_resolving = (needs_resolving | metadata_cell_attached) & ~metadata_cell_attached

    if show and infer_from_raw_trace and needs_resolving.any():
        print(f'reading the trace for {int(needs_resolving.sum())} block(s) whose mode the '
              f'label does not settle ({int((unlabelled & needs_resolving).sum())} '
              f'never labelled, {int((contested & needs_resolving).sum())} contradicted)')

    flags = pd.Series('', index=out.index, dtype=object)
    flags.loc[metadata_cell_attached] = (
        "resolved to 'extracellular': recordingTechnique is cell-attached "
        'and no positive series resistance contradicts it')

    if not infer_from_raw_trace:
        flags.loc[needs_resolving] = 'raw-trace inference disabled; metadata cannot resolve mode'
        # Positive resistance establishes whole-cell, but cannot distinguish
        # excitation from inhibition without a block label or current trace.
        out.loc[contested & out.rs_mode.eq('whole-cell'), 'onlineAnalysis'] = 'none'
        needs_resolving[:] = False

    # Blocks in one epoch group belong to the same recording. Resolve a zero-Rs
    # ambiguity from one representative trace, then reuse that decision across
    # matching blocks in the group. Positive-Rs whole-cell polarity may change
    # with holding potential, so those blocks remain independent.
    resolution_keys = {}
    representatives = {}
    for i in out.index[needs_resolving]:
        row = out.loc[i]
        if block_level_evidence or row['rs_mode'] == 'whole-cell' or 'group_id' not in out:
            key = ('block', int(row['block_id']))
        else:
            group_id = row['group_id']
            group_id = int(group_id) if pd.notna(group_id) else int(row['block_id'])
            key = (str(row['exp_name']), group_id,
                   str(row['onlineAnalysis']), str(row['rs_mode']))
        resolution_keys[i] = key
        representatives.setdefault(key, i)

    representative_indices = list(representatives.values())
    n_trials_by_block = {
        int(out.loc[i, 'block_id']): 1 for i in representative_indices
        if (technique.loc[i] == 'whole-cell'
            and out.loc[i, 'rs_mode'] != 'cell-attached')
    }
    trace_samples = _amp_trace_samples(
        out.loc[representative_indices, ['exp_name', 'block_id']], amp=amp,
        verbose=show, response_table=response_table,
        n_trials_by_block=n_trials_by_block,
        trace_seconds=trace_seconds) if infer_from_raw_trace and representative_indices else {}
    resolved_groups = {}
    for key, i in tqdm(representatives.items(), total=len(representatives),
                       desc='Inferring recording mode', unit='block',
                       disable=not show or not representatives):
        row = out.loc[i]
        sample = trace_samples.get(int(row['block_id']))
        if sample is None:
            resolved_groups[key] = (None, 'could not read the trace to resolve the label')
            continue
        amp_data, sample_rate = sample
        if technique.loc[i] == 'whole-cell' and row['rs_mode'] != 'cell-attached':
            mode = 'exc' if float(np.mean(amp_data)) < 0 else 'inh'
            note = (f"'{row['onlineAnalysis']}' resolved to '{mode}': "
                    'recordingTechnique is whole-cell; polarity from the sign '
                    'of the current')
        else:
            resistance = row['series_resistance']
            missing_resistance = not np.isfinite(resistance)
            # The zero-Rs branch tests spikes rather than assuming attachment.
            # Use that same trace decision when Rs is unavailable, while
            # retaining NaN in the audit and explicitly reporting its absence.
            if block_level_evidence and missing_resistance:
                resistance = 0.0
            mode, note = resolve_recording_mode(
                row['onlineAnalysis'], resistance, amp_data=amp_data,
                sample_rate=sample_rate, detector_kwargs=detector_kwargs)
            if block_level_evidence and missing_resistance:
                note = (f"resolved to '{mode}' from this block's trace "
                        '(series resistance unavailable; epoch-group label ignored)')
            if block_level_evidence and mode == 'extracellular':
                width = prominent_event_width_ms(amp_data, sample_rate)
                if np.isfinite(width) and width > max_spike_width_ms:
                    mode = 'exc' if float(np.mean(amp_data)) < 0 else 'inh'
                    note = (f"resolved to '{mode}': prominent raw events have "
                            f'lower-quartile width {width:.2f} ms > {max_spike_width_ms:g} ms; '
                            'broad currents triggered the spike detector; '
                            'polarity from the sign of the current')
        resolved_groups[key] = (mode, note)

    for i, key in resolution_keys.items():
        mode, note = resolved_groups[key]
        if mode is not None:
            out.loc[i, 'onlineAnalysis'] = mode
        flags[i] = note
    flags[all_high & flags.eq('')] = (
        f'series resistance above {max_series_resistance / 1e6:g} MOhm')
    out['rs_flag'] = flags
    relabelled = out['onlineAnalysis'].ne(out['onlineAnalysis_recorded'])

    if show:
        n_read = int((out['n_epochs_rs'] > 0).sum())
        print(f'series resistance read for {n_read}/{len(out)} blocks')
        print()
        print('onlineAnalysis vs the amplifier (as recorded, before any relabelling)')
        print(pd.crosstab(out['label_mode'].replace('', '(none)'),
                          out['rs_mode'].replace('', '(no reading)'),
                          rownames=['onlineAnalysis says'],
                          colnames=['series resistance says']).to_string())

        flagged = out[out['rs_flag'].ne('')]
        print()
        if flagged.empty:
            print('every block with a reading agrees with its onlineAnalysis label')
        else:
            print(f'{len(flagged)} block(s) where the amplifier disagrees:')
            cols = [c for c in ('exp_name', 'cell_label', 'cell_type_short',
                                'onlineAnalysis_recorded', 'onlineAnalysis',
                                'series_resistance', 'n_epochs', 'block_id', 'rs_flag')
                    if c in flagged.columns]
            show_df = flagged[cols].copy()
            show_df['series_resistance'] = (show_df['series_resistance'] / 1e6).round(2)
            show_df = show_df.rename(columns={'series_resistance': 'Rs (MOhm)',
                                              'onlineAnalysis_recorded': 'recorded',
                                              'onlineAnalysis': 'analyzed as'})
            print(show_df.to_string(index=False))
        if relabelled.any():
            print(f'\n{int(relabelled.sum())} block(s) relabelled; they are analyzed as the '
                  f'amplifier says, and their recorded label is kept in '
                  f'onlineAnalysis_recorded')

    if drop:
        n_dropped = int(all_high.sum())
        if show and n_dropped:
            print(f'\ndropping {n_dropped} block(s) whose every epoch is above the '
                  f'{max_series_resistance / 1e6:g} MOhm cutoff; pass drop=False to keep them')
        out = out[~all_high].reset_index(drop=True)
    return out

# --------------------------------------------------------------------------
# fixed neutral-density filters in the light path
# --------------------------------------------------------------------------

def _stage_ndfs_from_group(epoch_group) -> str:
    """The Stage device's ``ndfs`` list for one epoch group, as 'EL06, EL2'."""
    import json

    backgrounds = epoch_group.get('backgrounds')
    if backgrounds is None:
        return ''
    for dev in backgrounds:
        spans = backgrounds[dev].get('dataConfigurationSpans')
        if spans is None:
            continue
        for span in spans:
            for node in spans[span]:
                # The Stage node carries the filters actually in the light path;
                # the LED devices each carry their own (empty) list.
                if 'Stage' not in str(node):
                    continue
                raw = spans[span][node].attrs.get('ndfs')
                if raw is None:
                    continue
                text = raw.decode() if isinstance(raw, bytes) else str(raw)
                try:
                    return ', '.join(str(v) for v in json.loads(text))
                except Exception:
                    return text
    return ''


def read_stage_ndfs(exp_name: str, block_id: int, amp: str = 'Amp1', h5=None) -> str:
    """The fixed neutral-density filters in the light path for a block.

    Distinct from ``background:FilterWheel:NDF``, which is the *wheel* setting:
    this is the stack of filters physically in the path, which Symphony records
    on the Stage device as e.g. ``["EL06","EL2"]``. Both attenuate, so neither
    alone gives the light level — a block at wheel 0 behind an EL3 is three log
    units darker than the same wheel setting with nothing in the path.

    It is not constant within an experiment: on 2026-06-04_G most blocks ran
    behind ``EL06, EL2, FW1`` and four behind ``EL06, EL2``. Hence a per-block
    column rather than a per-date note. Empty string when no filter was in the
    path or the field is absent.
    """
    import h5py
    from retinanalysis.utils.datajoint_utils import get_h5_file

    groups = _amp_epoch_groups(exp_name, int(block_id), amp=amp)
    if not groups:
        return ''

    def _read(f):
        node = f.get(groups[0])
        return _stage_ndfs_from_group(node) if node is not None else ''

    if h5 is not None:
        return _read(h5)
    with h5py.File(get_h5_file(exp_name), 'r') as f:
        return _read(f)


def stage_ndf_table(df: pd.DataFrame, amp: str = 'Amp1', verbose: bool = True) -> pd.DataFrame:
    """Read fixed filter stacks with one DB query and one H5 open per date."""
    import h5py
    from retinanalysis.utils.datajoint_utils import get_h5_file

    wanted = df[['exp_name', 'block_id']].drop_duplicates().copy()
    wanted['block_id'] = wanted['block_id'].astype(int)
    if wanted.empty:
        return pd.DataFrame(columns=['block_id', 'stage_ndfs'])
    try:
        paths_by_block = _amp_epoch_groups_by_block(
            wanted['block_id'].tolist(), amp=amp)
    except Exception as exc:
        if verbose:
            print(f'  cannot batch-read amplifier paths ({exc}); fixed-filter '
                  'readings are unavailable')
        paths_by_block = {}

    rows = []
    for exp, sub in wanted.groupby('exp_name', sort=True):
        try:
            f = h5py.File(get_h5_file(str(exp)), 'r')
        except Exception:
            if verbose:
                print(f'  {exp}: cannot open the h5 — no fixed-filter reading for '
                      f'{len(sub)} block(s)')
            f = None
        for bid in sub['block_id']:
            value = ''
            if f is not None:
                try:
                    paths = paths_by_block.get(int(bid), [])
                    node = f.get(paths[0]) if paths else None
                    value = _stage_ndfs_from_group(node) if node is not None else ''
                except Exception:
                    value = ''
            rows.append({'block_id': int(bid), 'stage_ndfs': value})
        if f is not None:
            f.close()
    return pd.DataFrame(rows)
