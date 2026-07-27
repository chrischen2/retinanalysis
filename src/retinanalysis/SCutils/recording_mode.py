"""Which kind of recording a single-cell block actually is, from the amplifier.

``onlineAnalysis`` is a menu item the experimenter picks before a block runs,
and it is wrong often enough to matter: mislabelled, or left at ``'none'``
entirely. Symphony records what the rig actually did in
``stimulus:Amp1:seriesResistance`` — exactly 0 for a cell-attached patch, which
has no access resistance, and positive for whole-cell — and that reading, with
the trace itself as a tie-breaker, is enough to settle it.

Shared by the single-cell protocol modules so they all resolve the mode the
same way. :func:`resolve_recording_mode` is the decision;
:func:`check_series_resistance` applies it across a block table.

Also here because it is read from the same place in the h5:
:func:`read_stage_ndfs`, the fixed neutral-density filters in the light path,
which are recorded separately from the filter-wheel setting and which the wheel
setting does not imply.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

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


def _epoch_series_resistance(epoch_group, amp: str = 'Amp1') -> float:
    """``stimulus:<amp>:seriesResistance`` for one epoch group, in ohms.

    Symphony writes the amplifier's device configuration into
    ``stimuli/<amp>-<uuid>/dataConfigurationSpans/span_0/<amp>``, which is what
    ``epoch.protocolSettings('stimulus:Amp1:seriesResistance')`` returns in the
    MATLAB. NaN when the attribute is absent (older rigs never wrote it).
    """
    stimuli = epoch_group.get('stimuli')
    if stimuli is None:
        return np.nan
    for dev in stimuli:
        if str(dev).split('-')[0] != amp:
            continue
        spans = stimuli[dev].get('dataConfigurationSpans')
        if spans is None:
            continue
        for span in spans:
            node = spans[span].get(amp)
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
    recorded = str(online_analysis or 'extracellular').strip().lower()
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
                            verbose: bool = True) -> pd.DataFrame:
    """Read the series resistance of every block in ``df``, one h5 open per date.

    Returns one row per ``block_id`` with the median / min / max reading and how
    many of its epochs sit above ``max_series_resistance``. Blocks whose h5 is
    missing come back with NaN rather than raising, so one absent file does not
    stop the audit.
    """
    import h5py
    from retinanalysis.utils.datajoint_utils import get_h5_file

    rows = []
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
                    rs = read_series_resistance(str(exp), int(bid), amp=amp, h5=f)
                except Exception as e:
                    if verbose:
                        print(f'  {exp} block {bid}: {type(e).__name__}: {e}')
            good = rs[np.isfinite(rs)]
            rows.append({
                'block_id': int(bid),
                'series_resistance': float(np.median(good)) if good.size else np.nan,
                'series_resistance_min': float(good.min()) if good.size else np.nan,
                'series_resistance_max': float(good.max()) if good.size else np.nan,
                'n_epochs_rs': int(good.size),
                'n_epochs_high_rs': int(np.sum(good > max_series_resistance)),
            })
        if f is not None:
            f.close()
    return pd.DataFrame(rows)


def check_series_resistance(df: pd.DataFrame, amp: str = 'Amp1',
                            max_series_resistance: float = MAX_SERIES_RESISTANCE,
                            drop: bool = True, show: bool = True,
                            detector_kwargs: Optional[dict] = None) -> pd.DataFrame:
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

    Resolving a contradiction needs the block's raw trace, so this reads the h5
    for those blocks only; blocks whose label already agrees with the reading
    cost nothing beyond the reading itself.
    """
    out = df.copy()
    table = series_resistance_table(out[['exp_name', 'block_id']], amp=amp,
                                    max_series_resistance=max_series_resistance,
                                    verbose=show)
    out = out.merge(table, on='block_id', how='left')

    rs = out['series_resistance'].to_numpy(dtype=float)
    out['rs_mode'] = np.where(np.isnan(rs), '', np.where(rs > 0, 'whole-cell', 'cell-attached'))
    out['label_mode'] = out['onlineAnalysis'].apply(mode_family)
    out['onlineAnalysis_recorded'] = out['onlineAnalysis']

    # Two kinds of block need the trace: one whose label the reading contradicts,
    # and one that was never labelled at all -- the latter is most of the
    # linear-equivalent-disc dataset, where 'none' is the commonest entry.
    contested = (out['rs_mode'].ne('') & out['label_mode'].ne('')
                 & out['rs_mode'].ne(out['label_mode']))
    unlabelled = out['rs_mode'].ne('') & out['label_mode'].eq('')
    needs_resolving = contested | unlabelled
    all_high = (out['n_epochs_rs'] > 0) & out['n_epochs_high_rs'].eq(out['n_epochs_rs'])

    if show and needs_resolving.any():
        print(f'reading the trace for {int(needs_resolving.sum())} block(s) whose mode the '
              f'label does not settle ({int(unlabelled.sum())} never labelled, '
              f'{int(contested.sum())} contradicted)')

    flags = pd.Series('', index=out.index, dtype=object)
    for i in out.index[needs_resolving]:
        row = out.loc[i]
        amp_data = sample_rate = None
        try:
            import retinanalysis as ra
            rb = ra.SCResponseBlock(str(row['exp_name']), int(row['block_id']),
                                    b_spiking=False, verbose=False)
            amp_data, sample_rate = rb.amp_data, float(rb.amp_sample_rate)
        except Exception as e:
            flags[i] = f'could not read the trace to resolve the label ({type(e).__name__})'
            continue
        mode, note = resolve_recording_mode(row['onlineAnalysis'], row['series_resistance'],
                                            amp_data=amp_data, sample_rate=sample_rate,
                                            detector_kwargs=detector_kwargs)
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
    """Read the fixed filter stack of every block in ``df``, one h5 open per date."""
    import h5py
    from retinanalysis.utils.datajoint_utils import get_h5_file

    rows = []
    for exp, sub in df.groupby('exp_name', sort=True):
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
                    value = read_stage_ndfs(str(exp), int(bid), amp=amp, h5=f)
                except Exception:
                    value = ''
            rows.append({'block_id': int(bid), 'stage_ndfs': value})
        if f is not None:
            f.close()
    return pd.DataFrame(rows)