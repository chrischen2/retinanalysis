"""Cell deduplication — protocol-agnostic.

A common Kilosort failure mode is the same physical cell getting
sorted twice (e.g. drifting waveform crossing a clustering boundary).
When that happens, both clusters share the same EI footprint and, on
the noise side, the same STA. Downstream population analyses then
double-count that one cell.

This module finds and merges such duplicates as a *normal step in the
pipeline build*, not as a one-off diagnostic. The heavy lifting (EI /
STA correlation matrices, transitive-closure grouping) already lives
in ``retinanalysis.classes.dedup``; this is a thin, decision-oriented
wrapper that:

- Restricts pairing to the SAME cell type (default) so an OnP and an
  OnM that happen to share an electrode are never merged.
- Picks one *representative* per duplicate group (highest EI peak
  amplitude, by default) and drops the others' rows from
  ``df_spike_times``.
- Returns a tidy log of what was merged, so the user can audit.

Designed to be called in any protocol notebook right after the
pipeline build — see ``dedup_pipeline``.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..classes.dedup import (get_ei_autocorrelation, get_sm_autocorrelation,
                               generate_extended_pairings)


__all__ = [
    'find_duplicate_groups',
    'apply_dedup',
    'dedup_pipeline',
]


# Cell-type strings treated as "no typing" — duplicates among these
# are NOT auto-merged even with restrict_to_same_type=True, because
# we don't actually know they're the same biological cell type.
_UNTYPED_LABELS = frozenset({
    '', 'Unmatched', 'Unknown', 'No Typing File',
    'no typing file', 'unknown', 'unmatched', 'untyped',
})


def _resolve_block(pipeline_or_block, side: str):
    """Resolve to the actual block carrying df_spike_times + d_EIs.

    Accepts either an MEAPipeline, an MEAResponseBlock, or an
    AnalysisChunk. ``side`` selects the protocol side ('protocol') or
    the noise/analysis side ('noise') when handed a pipeline.
    """
    if hasattr(pipeline_or_block, 'resp') and hasattr(pipeline_or_block, 'analysis_chunk'):
        # An MEAPipeline
        return (pipeline_or_block.resp if side == 'protocol'
                else pipeline_or_block.analysis_chunk)
    # Otherwise assume the caller passed the block directly.
    return pipeline_or_block


def _build_type_map(block) -> Dict[int, str]:
    """Map cell_id → cell_type from df_spike_times (or empty if missing)."""
    out: Dict[int, str] = {}
    df = getattr(block, 'df_spike_times', None)
    if df is None or 'cell_type' not in df.columns:
        return out
    for _, r in df.iterrows():
        try:
            out[int(r['cell_id'])] = str(r['cell_type'])
        except (TypeError, ValueError):
            continue
    return out


def find_duplicate_groups(
    pipeline_or_block,
    *,
    side: str = 'protocol',
    ei_threshold: float = 0.85,
    sta_threshold: Optional[float] = None,
    ei_method: str = 'full',
    restrict_to_same_type: bool = True,
    pair_combine: str = 'intersect',
    skip_untyped: bool = True,
    verbose: bool = True,
) -> List[Tuple[int, ...]]:
    """Find groups of cells whose EI (and optionally STA) signatures match.

    Parameters
    ----------
    pipeline_or_block : MEAPipeline | MEAResponseBlock | AnalysisChunk
        Source of cells + EIs. If a pipeline, ``side`` selects which
        block to inspect.
    side : ``'protocol'`` (default) or ``'noise'``
        Which side of an ``MEAPipeline`` to inspect. ``'noise'`` is
        required to use ``sta_threshold`` (the analysis chunk is the
        one carrying ``d_spatial_maps``).
    ei_threshold : float
        Pairwise EI correlation cutoff. Default ``0.85`` — strict
        enough to avoid coincidental high correlations between
        nearby cells of the same type, loose enough to catch real
        duplicates.
    sta_threshold : float, optional
        Pairwise STA correlation cutoff. Only meaningful on the
        ``'noise'`` side. ``None`` (default) → ignore STA.
    ei_method : str
        Forwarded to ``ei_corr``: ``'full'`` / ``'space'`` / ``'power'``.
    restrict_to_same_type : bool
        Drop pairs whose two cells have different ``cell_type`` labels
        (default ``True``). Without this, an OnP and an OnM
        accidentally sharing an electrode could end up in the same
        "duplicate" group.
    pair_combine : ``'intersect'`` (default) or ``'union'``
        How to combine EI and STA pairs when both are computed.
        Intersection is conservative (require both metrics to flag
        the pair); union is aggressive.
    skip_untyped : bool
        When ``restrict_to_same_type=True``, also skip pairs whose
        type is in the "no typing" sentinel set (``''``, ``'Unmatched'``,
        ``'Unknown'``, ``'No Typing File'``). Default ``True``.
    verbose : bool
        Print a one-line summary.

    Returns
    -------
    list of tuple[int, ...]
        Each tuple is a duplicate group, sorted ascending. Singletons
        (cells with no duplicates) are omitted.
    """
    block = _resolve_block(pipeline_or_block, side)
    if not hasattr(block, 'd_EIs'):
        raise ValueError('Block has no d_EIs — cannot compute EI correlations.')

    _, ei_pairs = get_ei_autocorrelation(
        block, ei_method=ei_method, ei_threshold=ei_threshold)
    pairs = set(ei_pairs)

    if sta_threshold is not None:
        if side != 'noise' or not hasattr(block, 'd_spatial_maps'):
            if verbose:
                print(f'  (sta_threshold ignored — no spatial maps available on '
                      f"side={side!r}; pass side='noise' to use STA)")
        else:
            _, sm_pairs = get_sm_autocorrelation(
                block, sm_threshold=sta_threshold)
            if pair_combine == 'intersect':
                pairs = pairs & set(sm_pairs)
            elif pair_combine == 'union':
                pairs = pairs | set(sm_pairs)
            else:
                raise ValueError(
                    f"pair_combine must be 'intersect' or 'union', "
                    f"got {pair_combine!r}")

    if restrict_to_same_type:
        type_map = _build_type_map(block)

        def _is_typed(label: str) -> bool:
            return bool(label) and label not in _UNTYPED_LABELS

        def _pair_ok(a, b):
            ta = type_map.get(int(a), '')
            tb = type_map.get(int(b), '')
            a_typed = _is_typed(ta)
            b_typed = _is_typed(tb)
            if a_typed and b_typed:
                # Two typed cells: must match.
                return ta == tb
            if not a_typed and not b_typed:
                # Two untyped cells: allow only if skip_untyped is False.
                return not skip_untyped
            # One typed + one untyped: this is the common "good cluster
            # + its sloppy split" case. Always allowed — the typed cell
            # will be chosen as the representative by amplitude / typed
            # preference.
            return True
        pairs = {(a, b) for (a, b) in pairs if _pair_ok(a, b)}

    if not pairs:
        if verbose:
            n_cells = len(getattr(block, 'cell_ids', []))
            print(f'find_duplicate_groups: 0 duplicate pairs '
                  f'(from {n_cells} cells, EI≥{ei_threshold:.2f}'
                  f'{", STA≥%.2f" % sta_threshold if sta_threshold else ""}'
                  f'{", same-type" if restrict_to_same_type else ""})')
        return []

    groups_set = generate_extended_pairings(pairs)
    groups = sorted(
        (tuple(sorted(int(c) for c in g)) for g in groups_set),
        key=lambda g: (len(g), g))
    if verbose:
        n_affected = sum(len(g) for g in groups)
        n_extra = n_affected - len(groups)
        print(f'find_duplicate_groups: {len(groups)} group(s), '
              f'{n_affected} cells affected, '
              f'{n_extra} would be dropped as duplicates'
              f' (EI≥{ei_threshold:.2f}'
              f'{", STA≥%.2f" % sta_threshold if sta_threshold else ""}'
              f'{", same-type" if restrict_to_same_type else ""})')
    return groups


def _pick_representative(block, group: Tuple[int, ...],
                          strategy: str) -> int:
    """Pick the cell in ``group`` to keep.

    All strategies prefer a typed cell over an untyped one as the
    primary key. The named strategy is used only to break ties
    *within* the typed set (or *within* the untyped set, if no typed
    cell is present).
    """
    type_map = _build_type_map(block)
    typed = [int(c) for c in group if type_map.get(int(c), '') not in _UNTYPED_LABELS
             and type_map.get(int(c), '')]
    candidates = typed if typed else [int(c) for c in group]

    if strategy == 'highest_amplitude':
        amps: Dict[int, float] = {}
        for cid in candidates:
            try:
                ei = block.d_EIs[int(cid)]
                amps[int(cid)] = float(np.max(np.abs(ei)))
            except (KeyError, AttributeError):
                amps[int(cid)] = 0.0
        return max(amps, key=lambda c: amps[c])
    if strategy == 'most_spikes':
        df = block.df_spike_times
        counts: Dict[int, int] = {}
        for cid in candidates:
            sub = df[df['cell_id'].astype(int) == int(cid)]
            if sub.empty:
                counts[int(cid)] = 0
            else:
                sts = sub['spike_times'].iloc[0]
                counts[int(cid)] = sum(len(s) for s in sts)
        return max(counts, key=lambda c: counts[c])
    if strategy == 'first':
        return int(candidates[0])
    raise ValueError(f'unknown representative strategy: {strategy!r}')


def _merge_spike_trains(
    train_lists: List[List[np.ndarray]],
    refractory_ms: float,
) -> List[np.ndarray]:
    """Per-epoch union of spike-time arrays, with refractory-window dedup.

    ``train_lists`` is a list (one per cell) of lists (one per epoch)
    of float-ms arrays. Returns a single list-per-epoch of merged
    arrays. Within each epoch we concatenate every contributing cell's
    spikes, sort ascending, then drop any event closer than
    ``refractory_ms`` to its predecessor — both cells occasionally
    claim the same physical action potential when their template
    waveforms overlap, and unioning naively double-counts those.
    """
    if not train_lists:
        return []
    n_epochs = max(len(tl) for tl in train_lists)
    merged: List[np.ndarray] = []
    for ep in range(n_epochs):
        parts = []
        for tl in train_lists:
            if ep < len(tl):
                a = np.asarray(tl[ep], dtype=float).ravel()
                if a.size:
                    parts.append(a)
        if not parts:
            merged.append(np.array([], dtype=float))
            continue
        cat = np.sort(np.concatenate(parts))
        if refractory_ms > 0 and cat.size > 1:
            keep = np.concatenate([[True], np.diff(cat) > refractory_ms])
            cat = cat[keep]
        merged.append(cat.astype(float))
    return merged


def apply_dedup(
    pipeline_or_block,
    groups: List[Tuple[int, ...]],
    *,
    side: str = 'protocol',
    representative: str = 'highest_amplitude',
    merge_strategy: str = 'union',
    refractory_ms: float = 0.5,
    inplace: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Collapse each duplicate group to a single cell.

    Two ``merge_strategy`` options:

    - **``'union'``** (default, biologically correct): the
      representative's ``spike_times`` is replaced by the per-epoch
      union of every group member's trains, with events within
      ``refractory_ms`` of each other collapsed (since both clusters
      sometimes claim the same physical AP when their template
      waveforms overlap). Other members' rows are then dropped.
      This recovers spikes that Kilosort split between two clusters.
    - **``'drop'``** (conservative): keep only the representative's
      original train; drop the other rows. Use this when you suspect
      one cluster is actually a different cell (e.g. a neighbor with
      a similar EI footprint) and you'd rather not pool its spikes.

    For each group the representative is the typed cell with the
    highest EI peak amplitude by default (see ``_pick_representative``).
    The merged representative keeps its existing ``cell_type`` and
    ``noise_id``; ``d_EIs[representative]`` is unchanged.

    Parameters
    ----------
    merge_strategy : ``'union'`` (default) | ``'drop'``
    refractory_ms : float
        Window for refractory-period dedup in the unioned train.
        Default 0.5 ms. Set to 0 to skip the dedup (raw union).
        Only used when ``merge_strategy='union'``.
    representative : ``'highest_amplitude'`` | ``'most_spikes'`` | ``'first'``
    inplace : bool
        Mutate the block in place. ``False`` previews without applying.
    verbose : bool

    Returns
    -------
    pandas.DataFrame
        One row per group with columns:
        ``group, representative, dropped, representative_amp,
        cell_type, n_spikes_rep_before, n_spikes_dropped_total,
        n_spikes_rep_after, n_spikes_added_to_rep`` —
        the spike-count columns let you audit how aggressive the merge was.
        Empty when ``groups`` is empty.
    """
    if merge_strategy not in ('union', 'drop'):
        raise ValueError(
            f"merge_strategy must be 'union' or 'drop', got {merge_strategy!r}")

    block = _resolve_block(pipeline_or_block, side)
    df = getattr(block, 'df_spike_times', None)
    if df is None and inplace:
        raise ValueError('Block has no df_spike_times to modify.')

    type_map = _build_type_map(block)
    # Index df by cell_id once for O(1) lookups during the loop.
    df_by_cid = df.set_index(df['cell_id'].astype(int), drop=False) \
        if df is not None else None

    log_rows: List[Dict] = []
    drop_ids: set = set()
    # Stash merged trains keyed by rep id so we can write them out at
    # the end (avoids mutating the DataFrame mid-loop).
    merged_trains: Dict[int, List[np.ndarray]] = {}

    for group in groups:
        rep = _pick_representative(block, group, representative)
        others = tuple(c for c in group if c != rep)
        drop_ids.update(others)
        try:
            rep_amp = float(np.max(np.abs(block.d_EIs[int(rep)])))
        except (KeyError, AttributeError):
            rep_amp = float('nan')

        # Spike-count accounting + (optionally) the union merge.
        n_rep_before = 0
        n_dropped_total = 0
        n_rep_after = 0
        if df_by_cid is not None:
            try:
                rep_sts = list(df_by_cid.at[int(rep), 'spike_times'])
                n_rep_before = int(sum(len(s) for s in rep_sts))
            except KeyError:
                rep_sts = None
                n_rep_before = 0
            for cid in others:
                try:
                    sts = df_by_cid.at[int(cid), 'spike_times']
                    n_dropped_total += int(sum(len(s) for s in sts))
                except KeyError:
                    pass

            if merge_strategy == 'union' and rep_sts is not None:
                trains: List[List[np.ndarray]] = [rep_sts]
                for cid in others:
                    try:
                        trains.append(list(df_by_cid.at[int(cid), 'spike_times']))
                    except KeyError:
                        continue
                merged = _merge_spike_trains(trains, refractory_ms=refractory_ms)
                merged_trains[int(rep)] = merged
                n_rep_after = int(sum(len(s) for s in merged))
            else:
                n_rep_after = n_rep_before

        log_rows.append({
            'group': group,
            'representative': int(rep),
            'dropped': others,
            'representative_amp': rep_amp,
            'cell_type': type_map.get(int(rep), ''),
            'n_spikes_rep_before': n_rep_before,
            'n_spikes_dropped_total': n_dropped_total,
            'n_spikes_rep_after': n_rep_after,
            'n_spikes_added_to_rep': max(0, n_rep_after - n_rep_before),
        })

    if inplace and df is not None:
        if merged_trains:
            # Write merged spike trains back to the representative rows.
            # Use a copy to avoid SettingWithCopyWarning on the indexed view.
            df = df.copy()
            for rep_id, mt in merged_trains.items():
                mask = df['cell_id'].astype(int) == int(rep_id)
                # Assigning a list-of-arrays into a single cell of an
                # object-dtype Series is fiddly; do it via .at on the
                # matching row index.
                idx = df.index[mask]
                if len(idx) == 1:
                    df.at[idx[0], 'spike_times'] = mt
        if drop_ids:
            df = df[~df['cell_id'].astype(int).isin(drop_ids)].reset_index(drop=True)
        block.df_spike_times = df
        block.cell_ids = np.array(
            [int(c) for c in df['cell_id']], dtype=int)

    if verbose:
        kept = len(groups)
        dropped = len(drop_ids)
        n_recovered = sum(r['n_spikes_added_to_rep'] for r in log_rows)
        suffix = '' if inplace else '  (preview — no changes applied)'
        if merge_strategy == 'union':
            print(f'apply_dedup: kept {kept} representative(s), '
                  f'dropped {dropped} duplicate cell(s); '
                  f'recovered {n_recovered:,} spikes via union '
                  f'({refractory_ms} ms refractory){suffix}')
        else:
            print(f'apply_dedup: kept {kept} representative(s), '
                  f'dropped {dropped} duplicate cell(s){suffix}')

    return pd.DataFrame(log_rows, columns=[
        'group', 'representative', 'dropped',
        'representative_amp', 'cell_type',
        'n_spikes_rep_before', 'n_spikes_dropped_total',
        'n_spikes_rep_after', 'n_spikes_added_to_rep',
    ])


def dedup_pipeline(
    pipeline,
    *,
    ei_threshold: float = 0.85,
    sta_threshold: Optional[float] = None,
    restrict_to_same_type: bool = True,
    skip_untyped: bool = True,
    representative: str = 'highest_amplitude',
    merge_strategy: str = 'union',
    refractory_ms: float = 0.5,
    side: str = 'protocol',
    also_dedup_noise: bool = False,
    inplace: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """End-to-end dedup: find groups + apply merge. Protocol-agnostic.

    Designed to be called once right after building the pipeline. By
    default it only deduplicates the protocol side (``pipeline.resp``);
    set ``also_dedup_noise=True`` to also dedup the noise chunk (which
    additionally enables STA-based matching when ``sta_threshold`` is
    given).

    Parameters
    ----------
    pipeline : MEAPipeline
    ei_threshold : float
        EI corr cutoff. Default 0.85.
    sta_threshold : float, optional
        STA corr cutoff. Only used on the noise side.
    restrict_to_same_type : bool
        Default True. Pairs across cell types are ignored.
    representative : str
        Strategy for which cell to keep per group.
    side : ``'protocol'`` or ``'noise'``
        Default ``'protocol'`` (the protocol's response block).
    also_dedup_noise : bool
        Also run dedup on ``pipeline.analysis_chunk`` (uses STA too
        if ``sta_threshold`` is set).
    inplace : bool
        Mutate the pipeline in place.
    verbose : bool

    Returns
    -------
    dict
        ``{'protocol': log_df, 'noise': log_df_or_None}``
    """
    out: Dict[str, pd.DataFrame] = {'protocol': pd.DataFrame(),
                                      'noise': None}

    if verbose:
        print(f'Deduplicating protocol side ({side})…')
    proto_groups = find_duplicate_groups(
        pipeline, side='protocol',
        ei_threshold=ei_threshold,
        sta_threshold=None,           # protocol side has no STA
        restrict_to_same_type=restrict_to_same_type,
        skip_untyped=skip_untyped,
        verbose=verbose,
    )
    if proto_groups:
        out['protocol'] = apply_dedup(
            pipeline, proto_groups,
            side='protocol',
            representative=representative,
            merge_strategy=merge_strategy,
            refractory_ms=refractory_ms,
            inplace=inplace, verbose=verbose,
        )

    if also_dedup_noise:
        if verbose:
            print('Deduplicating noise side (analysis chunk)…')
        noise_groups = find_duplicate_groups(
            pipeline, side='noise',
            ei_threshold=ei_threshold,
            sta_threshold=sta_threshold,
            restrict_to_same_type=restrict_to_same_type,
            skip_untyped=skip_untyped,
            verbose=verbose,
        )
        if noise_groups:
            out['noise'] = apply_dedup(
                pipeline, noise_groups,
                side='noise',
                representative=representative,
                merge_strategy=merge_strategy,
                refractory_ms=refractory_ms,
                inplace=inplace, verbose=verbose,
            )

    return out
