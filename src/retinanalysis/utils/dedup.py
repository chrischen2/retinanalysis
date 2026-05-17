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


def apply_dedup(
    pipeline_or_block,
    groups: List[Tuple[int, ...]],
    *,
    side: str = 'protocol',
    representative: str = 'highest_amplitude',
    inplace: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Collapse each duplicate group to one representative cell.

    For each group: pick a representative (default = highest EI peak
    amplitude), drop the other cells' rows from ``df_spike_times`` and
    from ``cell_ids``. ``d_EIs`` is left alone (not pruned) so downstream
    EI diagnostics remain available; the dropped cells just don't appear
    in any cell-by-cell iteration anymore.

    Parameters
    ----------
    representative : ``'highest_amplitude'`` | ``'most_spikes'`` | ``'first'``
    inplace : bool
        Mutate the block in place. ``False`` leaves the block untouched
        and only returns the merge log (useful for previewing).
    verbose : bool

    Returns
    -------
    pandas.DataFrame
        One row per group: ``group`` (tuple), ``representative`` (int),
        ``dropped`` (tuple of int), ``representative_amp`` (float),
        ``cell_type`` (str). Empty DataFrame when ``groups`` is empty.
    """
    block = _resolve_block(pipeline_or_block, side)
    df = getattr(block, 'df_spike_times', None)
    if df is None and inplace:
        raise ValueError('Block has no df_spike_times to modify.')

    type_map = _build_type_map(block)
    log_rows: List[Dict] = []
    drop_ids: set = set()
    for group in groups:
        rep = _pick_representative(block, group, representative)
        others = tuple(c for c in group if c != rep)
        drop_ids.update(others)
        try:
            rep_amp = float(np.max(np.abs(block.d_EIs[int(rep)])))
        except (KeyError, AttributeError):
            rep_amp = float('nan')
        log_rows.append({
            'group': group,
            'representative': int(rep),
            'dropped': others,
            'representative_amp': rep_amp,
            'cell_type': type_map.get(int(rep), ''),
        })

    if inplace and df is not None and drop_ids:
        new_df = df[~df['cell_id'].astype(int).isin(drop_ids)].reset_index(drop=True)
        block.df_spike_times = new_df
        block.cell_ids = np.array(
            [int(c) for c in new_df['cell_id']], dtype=int)

    if verbose:
        kept = len(groups)
        dropped = len(drop_ids)
        suffix = '' if inplace else '  (preview — no changes applied)'
        print(f'apply_dedup: kept {kept} representative(s), '
              f'dropped {dropped} duplicate cell(s){suffix}')

    return pd.DataFrame(log_rows, columns=[
        'group', 'representative', 'dropped',
        'representative_amp', 'cell_type'])


def dedup_pipeline(
    pipeline,
    *,
    ei_threshold: float = 0.85,
    sta_threshold: Optional[float] = None,
    restrict_to_same_type: bool = True,
    representative: str = 'highest_amplitude',
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
        verbose=verbose,
    )
    if proto_groups:
        out['protocol'] = apply_dedup(
            pipeline, proto_groups,
            side='protocol',
            representative=representative,
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
            verbose=verbose,
        )
        if noise_groups:
            out['noise'] = apply_dedup(
                pipeline, noise_groups,
                side='noise',
                representative=representative,
                inplace=inplace, verbose=verbose,
            )

    return out
