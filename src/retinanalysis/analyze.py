"""End-to-end per-experiment analysis driver.

``analyze_experiment(exp_name)`` does everything the chrisMain notebook
does manually — datafile + ss-version + typing-file detection, pipeline
build, optional rig calibration, type normalization, protocol QC, mosaic
+ TC + ISI overview, per-cell raster/PSTH archive, manifest update — in
one call. ``analyze_experiments(exp_names)`` loops over many dates and
keeps going past per-experiment failures.

The function is intentionally read-only on database state (no DataJoint
writes); it only persists figures and the manifest to ``OUTPUT_DIR``.
Re-running on the same date is incremental: existing per-cell PNGs are
skipped, the mosaic is refreshed, and the manifest accumulates new
protocol entries.
"""

from __future__ import annotations

import os
import traceback
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config.settings import ANALYSIS_DIR, DATA_DIR, find_path
from .utils.cell_type_utils import map_cell_type, filter_available_types
from .utils.style import MAIN_CELL_TYPES
from .utils.protocol_qc import (
    QCThresholds, block_qc_metrics, filter_cells_by_qc,
    save_protocol_qc,
)
from .utils import rig_calibration as rc
from .utils.cell_plot_archive import (
    experiment_root, save_experiment_mosaic, save_per_cell_plots,
)
from .utils.cell_match import save_cell_match


__all__ = ['analyze_experiment', 'analyze_experiments',
           'detect_ss_version', 'pick_typing_file', 'resolve_noise_chunk',
           'add_analysis_chunk_columns', 'find_available_datasets',
           'summarize_batch_results',
           'SS_VERSION_PRIORITY']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Kilosort version priority, highest-preferred first. When more than one
# version exists side-by-side (the lab's analysis tree sometimes carries
# all three after a re-sort), 2.5 wins, then 2, then 4. The user picked
# this order based on which sorter currently produces the cleanest output
# for their array; revisit when that changes.
SS_VERSION_PRIORITY = ('kilosort2.5', 'kilosort2', 'kilosort4')


def _pick_ss_version_from(candidates):
    """Highest-priority match in ``candidates``, or the first item, or None."""
    # macOS AppleDouble dotfiles are not real directories — strip them so
    # they don't sneak into auto-selection.
    candidates = [c for c in candidates
                  if c.startswith('kilosort') and not c.startswith('.')]
    for pref in SS_VERSION_PRIORITY:
        if pref in candidates:
            return pref
    return candidates[0] if candidates else None


def detect_ss_version(
    exp_name: str,
    datafile_name: Optional[str] = None,
    *,
    chunk_name: Optional[str] = None,
    kind: str = 'data',
) -> str:
    """Pick the preferred kilosort version for a given (exp, datafile|chunk).

    Priority: :data:`SS_VERSION_PRIORITY` (``kilosort2.5 → kilosort2 →
    kilosort4``), then any other ``kilosort*`` dir alphabetically.

    Parameters
    ----------
    exp_name : str
    datafile_name : str, optional
        Use with ``kind='data'`` (default) to inspect the protocol
        datafile's kilosort output dirs under ``DATA_DIR``.
    chunk_name : str, optional
        Use with ``kind='analysis'`` to inspect the noise chunk's
        kilosort output dirs under ``ANALYSIS_DIR``.
    kind : ``'data'`` or ``'analysis'``
        Which volume to look in.

    Returns
    -------
    str
        The preferred ``kilosort*`` subdir name.

    Raises
    ------
    FileNotFoundError
        Source directory doesn't exist, or contains no ``kilosort*``
        subdirs.
    """
    # Inspect every configured tier.  Looking only below ``find_path(...)``
    # is insufficient when the first tier contains an incomplete copy of the
    # chunk while a lower-priority tier contains its Kilosort output.
    if kind == 'data':
        if datafile_name is None:
            raise ValueError("kind='data' requires datafile_name")
        parts = (exp_name, datafile_name)
        location_desc = f'{exp_name}/{datafile_name} (data)'
    elif kind == 'analysis':
        if chunk_name is None:
            raise ValueError("kind='analysis' requires chunk_name")
        parts = (exp_name, chunk_name)
        location_desc = f'{exp_name}/{chunk_name} (analysis)'
    else:
        raise ValueError(f"kind must be 'data' or 'analysis', got {kind!r}")

    from .config.settings import tier_dirs

    sort_dirs = []
    first = find_path(kind, *parts)
    if first and os.path.isdir(first):
        sort_dirs.append(first)
    for root in tier_dirs(kind):
        candidate = os.path.join(root, *parts)
        if os.path.isdir(candidate) and candidate not in sort_dirs:
            sort_dirs.append(candidate)

    if not sort_dirs:
        raise FileNotFoundError(
            f'No sort directory for {location_desc} on any configured tier: '
            f'{first}')

    versions = {name for directory in sort_dirs for name in os.listdir(directory)}
    chosen = _pick_ss_version_from(versions)
    if chosen is None:
        raise FileNotFoundError(
            f'No kilosort* subdirs for {location_desc} under {sort_dirs}')
    return chosen


# Backward-compat alias used internally.
def _detect_ss_version(exp_name: str, datafile_name: str) -> str:
    return detect_ss_version(exp_name, datafile_name, kind='data')


def add_ss_version_column(df, kind: str = 'analysis',
                          column: str = 'ss_version'):
    """Annotate a dataset search frame with the sort version present on disk.

    Adds ``column`` to a copy of ``df``, resolved per row with
    :func:`detect_ss_version` — ``kilosort2.5`` when present, else
    ``kilosort2``. ``kind='analysis'`` reads the noise chunk's tree (needs a
    ``chunk_name`` column); ``kind='data'`` reads the protocol datafile's.

    Rows whose directory isn't on any configured volume get ``'not found'``
    rather than raising, so the column is safe to compute over a whole search
    result. Answers per row are memoized, so a date appearing several times
    only costs one directory listing.
    """
    df = df.copy()
    cache = {}

    def _lookup(row):
        key = (row['exp_name'],
               row.get('chunk_name') if kind == 'analysis'
               else row.get('datafile_name'))
        if key not in cache:
            try:
                if kind == 'analysis':
                    cache[key] = detect_ss_version(
                        key[0], chunk_name=key[1], kind='analysis')
                else:
                    cache[key] = detect_ss_version(
                        key[0], datafile_name=key[1], kind='data')
            except (FileNotFoundError, ValueError):
                cache[key] = 'not found'
        return cache[key]

    df[column] = [_lookup(r) for _, r in df.iterrows()]
    return df


def add_analysis_chunk_columns(
    df,
    *,
    chunk_column: str = 'analysis_chunk_name',
    version_column: str = 'ss_version',
    distance_column: str = 'chunk_distance_min',
):
    """Resolve each protocol row to its nearest loadable typed noise chunk.

    The ``chunk_name`` stored on a protocol row is the sorting chunk assigned
    at acquisition time; it is not necessarily the name of a noise-analysis
    directory.  Newer experiments commonly carry labels such as ``dynamics``
    or ``eye_move`` there while their Vision analysis lives in
    ``chunk1/kilosort2.5``.  Consequently, applying
    :func:`add_ss_version_column` directly to a protocol-search frame can
    report ``'not found'`` even though the dataset is loadable.

    This helper follows the same rule used by the MEA pipeline: rank noise
    chunks by recording time (preceding chunks first), then select the first
    candidate that has both a Kilosort directory and a classification file.
    It adds the resolved chunk, Kilosort version, and time distance while
    leaving the database's original ``chunk_name`` untouched.
    """
    from .utils.datajoint_utils import (get_exp_summary,
                                        get_noise_chunks_sorted_by_distance,
                                        get_noise_name_by_exp)

    required = {'exp_name', 'datafile_name'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'dataset frame is missing columns: {sorted(missing)}')

    out = df.copy()
    summaries = {}
    resolved = {}

    def _resolve(exp_name, datafile_name):
        key = (str(exp_name), str(datafile_name))
        if key in resolved:
            return resolved[key]

        if key[0] not in summaries:
            summaries[key[0]] = get_exp_summary(key[0])
        summary = summaries[key[0]]
        candidates, distances = get_noise_chunks_sorted_by_distance(
            summary, key[1], noise_protocol_name=get_noise_name_by_exp(key[0]))

        answer = (None, 'not found', np.nan)
        for chunk_name, distance in zip(candidates, distances):
            try:
                version = detect_ss_version(
                    key[0], chunk_name=str(chunk_name), kind='analysis')
                typing_file = pick_typing_file(
                    key[0], str(chunk_name), version, strict=False)
            except (FileNotFoundError, ValueError, OSError):
                continue
            if typing_file is not None:
                answer = (str(chunk_name), version, float(distance))
                break
        resolved[key] = answer
        return answer

    values = [_resolve(row['exp_name'], row['datafile_name'])
              for _, row in out.iterrows()]
    out[chunk_column] = [value[0] for value in values]
    out[version_column] = [value[1] for value in values]
    out[distance_column] = [value[2] for value in values]
    return out


def pick_typing_file(
    exp_name: str, chunk_name: str, ss_version: str,
    preferred: Optional[str] = None,
    *,
    strict: bool = True,
) -> Optional[str]:
    """Pick a ``*.classification.txt`` from the analysis chunk directory.

    Walks ``find_path('analysis', ...)`` (local-cache → SSD → NAS),
    lists the chunk dir, and filters out macOS AppleDouble dotfiles
    (``._*``). If ``preferred`` is provided and present, returns it;
    otherwise returns the first remaining candidate.

    Parameters
    ----------
    exp_name, chunk_name, ss_version : str
        Identifies the analysis chunk directory.
    preferred : str, optional
        Preferred filename (must be one of the available candidates).
    strict : bool, default True
        When ``True``, raise ``FileNotFoundError`` if the chunk dir is
        missing or no typing file is found. When ``False``, return
        ``preferred`` (or ``None``) silently — the soft mode used by
        :func:`analyze_experiment`'s fallback path.
    """
    chunk_dir = find_path('analysis', exp_name, chunk_name, ss_version)
    if not chunk_dir or not os.path.isdir(chunk_dir):
        if strict:
            raise FileNotFoundError(
                f'No analysis chunk dir for '
                f'{exp_name}/{chunk_name}/{ss_version} on any configured '
                f'tier (last tried: {chunk_dir!r}).')
        return preferred
    candidates = [
        f for f in os.listdir(chunk_dir)
        if f.endswith('.classification.txt') and not f.startswith('.')
    ]
    if preferred and preferred in candidates:
        return preferred
    if not candidates:
        if strict:
            raise FileNotFoundError(
                f'No .classification.txt in {chunk_dir}. '
                f'Pick a different chunk_name.')
        return None
    if preferred and preferred not in candidates:
        if strict:
            raise FileNotFoundError(
                f'{preferred!r} not found in {chunk_dir}. '
                f'Available: {candidates}')
        return None
    return candidates[0]


def _pick_typing_file(
    exp_name: str, chunk_name: str, ss_version: str,
    preferred: Optional[str] = None,
) -> Optional[str]:
    """Soft wrapper around :func:`pick_typing_file` (returns None on miss)."""
    return pick_typing_file(exp_name, chunk_name, ss_version,
                              preferred=preferred, strict=False)


def resolve_noise_chunk(
    exp_name: str,
    datafile_name: str,
    override: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Resolve the noise chunk to use, with a DB-record sanity check.

    Used by the notebook's §3b cell and by other callers that want to
    pin a noise chunk while still flagging mismatches against what the
    experimenter declared at the rig.

    Parameters
    ----------
    exp_name, datafile_name : str
    override : str, optional
        When set, used as-is (no auto-pick). When ``None``, falls back
        to ``MEAStimBlock.nearest_noise_chunk`` (closest in time).

    Returns
    -------
    (chunk, db_chunk, warning) : tuple
        ``chunk`` is the resolved chunk name. ``db_chunk`` is whatever
        the database row carries (may be ``None``). ``warning`` is a
        one-line message when the two differ, else ``None`` — let the
        caller print it however they like.
    """
    # Lazy DJ import.
    from .utils.datajoint_utils import get_exp_summary
    from .classes.stim import MEAStimBlock

    db_chunk: Optional[str] = None
    try:
        _exp_df = get_exp_summary(exp_name)
        _row = _exp_df.query('datafile_name == @datafile_name')
        if not _row.empty:
            db_chunk = str(_row['chunk_name'].iloc[0])
    except Exception:
        db_chunk = None

    if override is not None:
        chunk = override
    else:
        chunk = MEAStimBlock(exp_name, datafile_name,
                              verbose=False).nearest_noise_chunk

    warning: Optional[str] = None
    if db_chunk is not None and db_chunk != chunk:
        warning = (
            f'DB chunk {db_chunk!r} differs from chunk being used '
            f'({chunk!r}). If the DB record is correct, pin '
            f'noise_chunk_name={db_chunk!r}.'
        )
    return chunk, db_chunk, warning


def find_available_datasets(protocol_search: str):
    """Protocol-registry rows filtered to dates with sort output on disk.

    Calls :func:`get_datasets_from_protocol_names` and intersects with the
    dates present under *any* configured analysis root, so registry rows
    whose Kilosort output isn't reachable are silently dropped. One row per
    ``(exp_name, datafile_name)``.

    The intersect spans every mounted volume rather than just the
    top-priority ``ANALYSIS_DIR``: a date archived only on a secondary SSD
    is just as analyzable as one on the top tier.
    """
    from .utils.datajoint_utils import get_datasets_from_protocol_names
    from .config.settings import tier_dirs
    df = get_datasets_from_protocol_names(protocol_search)
    available = {exp for root in tier_dirs('analysis')
                 for exp in os.listdir(root)}
    return df[df['exp_name'].isin(available)].reset_index(drop=True)


def summarize_batch_results(results):
    """Compact summary DataFrame for a list of ``analyze_experiment`` dicts.

    Keeps the columns most useful at the end of a batch run:
    ``exp_name, datafile_name, chunk_name, n_cells_total,
    n_cells_passed_qc, ndf, error``.
    """
    import pandas as pd  # local import keeps top-level cheap
    cols = ['exp_name', 'datafile_name', 'chunk_name',
            'n_cells_total', 'n_cells_passed_qc', 'ndf', 'error']
    return pd.DataFrame([{k: r.get(k) for k in cols} for r in results])


def _resolve_datafile(exp_name: str, protocol_search: Optional[str],
                      datafile_name: Optional[str]) -> str:
    """Resolve ``datafile_name`` either from a direct argument or via search."""
    if datafile_name:
        return datafile_name
    if not protocol_search:
        raise ValueError(
            f'analyze_experiment({exp_name!r}): pass either datafile_name= '
            'or protocol_search= so we know which datafile to analyze.'
        )
    # Lazy import to avoid pulling DJ at module load time.
    from .utils.datajoint_utils import get_datasets_from_protocol_names
    df = get_datasets_from_protocol_names(protocol_search)
    df = df[df['exp_name'] == exp_name]
    if df.empty:
        raise LookupError(
            f'{exp_name!r}: no datafile found for protocol search '
            f'{protocol_search!r}'
        )
    return str(df['datafile_name'].iloc[0])


def _normalize_cell_types(response_block) -> None:
    """In-place: re-map cell_type via ra.map_cell_type (case-insensitive)."""
    if 'cell_type' not in response_block.df_spike_times.columns:
        return
    def _norm(t):
        c = map_cell_type(t)
        return c if c is not None else t
    response_block.df_spike_times['cell_type'] = (
        response_block.df_spike_times['cell_type'].apply(_norm)
    )


def _extract_ndf(stim_block) -> Optional[float]:
    """Consolidate protected FilterWheel NDF across every epoch."""
    from .utils.light_levels import filter_wheel_ndf_from_epoch_parameters

    try:
        parameters = stim_block.df_epochs['epoch_parameters'].tolist()
        value = filter_wheel_ndf_from_epoch_parameters(
            parameters, context='stimulus block epochs')
        return float(value) if np.isfinite(value) else None
    except (KeyError, IndexError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Single-experiment driver
# ---------------------------------------------------------------------------

def analyze_experiment(
    exp_name: str,
    datafile_name: Optional[str] = None,
    *,
    protocol_search: Optional[str] = None,
    ss_version: Optional[str] = None,
    analysis_chunk_name: Optional[str] = None,
    typing_file: Optional[str] = None,
    cell_types: Sequence[str] = MAIN_CELL_TYPES,
    minimum_n: int = 3,
    psth_sigma_ms: float = 10.0,
    sample_rate_hz: float = 500.0,
    fit_calibration: bool = False,
    save_calibration: bool = True,
    qc_thresholds: Optional[QCThresholds] = None,
    condition_keys: Optional[Sequence[str]] = None,
    output_root: Optional[str] = None,
    overwrite: bool = False,
    psth_ncols: int = 1,
    n_jobs: int = -1,
    cell_ids: Optional[Sequence[int]] = None,
    respect_visual_qc: bool = True,
    prune_stale: bool = True,
    protocol_subdir: Optional[str] = None,
    append_datafile_to_subdir: bool = False,
    dedup: bool = True,
    dedup_ei_threshold: float = 0.85,
    dedup_skip_untyped: bool = True,
    dedup_merge_strategy: str = 'union',
    dedup_refractory_ms: float = 0.5,
    verbose: bool = True,
) -> Dict:
    """Run the full archive pipeline for one experiment date.

    Parameters
    ----------
    exp_name : str
        Date+rig string, e.g. ``'20221123C'``.
    datafile_name : str, optional
        Specific protocol datafile (e.g. ``'data027'``). When ``None``,
        resolved via ``protocol_search``.
    protocol_search : str, optional
        Substring matched against protocol names (e.g. ``'AlternatingBackground'``)
        — picks the first datafile of any matching protocol for this date.
    ss_version, typing_file : str, optional
        Auto-detected when not provided.
    analysis_chunk_name : str, optional
        Pin the noise chunk used for RF params / cell typing (and the
        typing-file lookup when ``typing_file`` is not given). When
        ``None`` (default), the nearest-in-time chunk is auto-picked.
        Set this to the same value you pinned as ``noise_chunk_name``
        in the notebook's §3b so the archive doesn't silently fall back
        to a different chunk than the in-notebook pipeline.
    cell_types : sequence
        Restrict per-cell PNGs to these types (default = MAIN_CELL_TYPES).
    fit_calibration : bool
        Run a fresh rig calibration fit for the noise chunk. When False
        (default), the existing saved calibration is loaded if present,
        otherwise the geometric fallback is used.
    save_calibration : bool
        Persist a freshly-fit calibration to ``rig_calibrations/``.
    condition_keys : sequence[str], optional
        Per-epoch keys to split raster/PSTH by. Auto-detected for known
        protocols (e.g. EyeMovement → [imageName, backgroundScale]).
    output_root : str, optional
        Override ``OUTPUT_DIR`` for testing.
    overwrite : bool
        Re-render all per-cell PNGs and the mosaic.
    psth_ncols : int
        Column count for the per-cell PSTH grid (default 1 = vertical stack).
    cell_ids : sequence[int], optional
        Restrict the per-cell PNG step to these cells (intersected with
        the QC-pass set). When ``None`` (default) every QC-passing cell
        is rendered. Use this to re-archive against a visual-QC ``good``
        subset; ``cell_match.csv`` is unaffected and stays comprehensive.
    respect_visual_qc : bool
        When ``True`` (default) and no explicit ``cell_ids`` is passed,
        look for ``<OUTPUT_DIR>/<exp>/<protocol>/visual_qc.csv`` and
        restrict the archive to cells tagged ``'good'``. When no such
        file exists, every QC-passing cell is rendered (first-pass
        default). Set ``False`` to ignore visual QC and always render
        the full QC-pass set.
    prune_stale : bool
        When ``True`` (default), delete per-cell PNGs left over from a
        prior run whose ``cell_id`` is no longer in the kept set. This
        is the standard §17/§18 re-archive flow: tag cells ``bad`` in
        §16, re-run §17/§18, and the bad cells' PNGs disappear from
        ``cells/<celltype>/``. Set ``False`` to keep stale PNGs.
    protocol_subdir : str, optional
        Override the per-protocol subdirectory name under
        ``<OUTPUT_DIR>/<exp_name>/``. Default (``None``) uses
        ``protocol_short_name(protocol_name)``, e.g.
        ``eye_movement_alt_bg``. Set when the experiment has multiple
        datafiles of the **same protocol** in the same chunk — each
        would otherwise write to the same subdir and overwrite the
        previous run. For example, pass
        ``protocol_subdir='eye_movement_alt_bg_data032'`` for the
        second datafile.
    append_datafile_to_subdir : bool
        Convenience shortcut: when ``True`` and ``protocol_subdir``
        is ``None``, set the subdir to ``{protocol_short}_{datafile}``
        automatically. Pairs well with batch runs over many datafiles
        of one protocol.

    Notes
    -----
    This function **reads** ``visual_qc.csv`` but never writes to it.
    Manual tags from the GUI are preserved across re-runs; the only
    writer is ``visual_qc._save_tag`` (invoked per click inside
    ``browse_cells_qc``).

    Returns
    -------
    dict
        Summary of what was done: ``exp_name``, ``datafile_name``,
        ``ss_version``, ``chunk_name``, ``cell_types_used``,
        ``n_cells_total``, ``n_cells_passed_qc``, ``calibration``,
        ``output_dir``, plus ``index_df`` (DataFrame of per-cell PNG paths).
    """
    if verbose:
        print(f'\n=== analyze_experiment({exp_name}) ===')

    # 1. Resolve datafile + sort version
    datafile_name = _resolve_datafile(exp_name, protocol_search, datafile_name)
    if ss_version is None:
        ss_version = _detect_ss_version(exp_name, datafile_name)
    if verbose:
        print(f'  datafile={datafile_name}  ss_version={ss_version}')

    # 2. Build pipeline (lazy import: avoid DataJoint cost on module load)
    from .classes.mea_pipeline import create_mea_pipeline
    from .classes.stim import MEAStimBlock

    if typing_file is None:
        # Need the noise chunk first to know where typing files live. Honor
        # an explicit pin; otherwise fall back to the nearest-in-time chunk.
        if analysis_chunk_name is not None:
            chunk_for_typing = analysis_chunk_name
        else:
            chunk_for_typing = MEAStimBlock(
                exp_name, datafile_name, verbose=False).nearest_noise_chunk
        typing_file = _pick_typing_file(exp_name, chunk_for_typing, ss_version)
        if verbose:
            print(f'  typing_file={typing_file}')

    pipeline = create_mea_pipeline(
        exp_name, datafile_name,
        analysis_chunk_name=analysis_chunk_name,
        ss_version=ss_version, typing_file=typing_file,
        verbose=False,
    )
    analysis_chunk = pipeline.analysis_chunk
    response_block = pipeline.resp
    stim_block = pipeline.stim
    _normalize_cell_types(response_block)

    # Dedup before QC: split clusters that are the same physical cell
    # get their spike trains merged (union with refractory dedup) into
    # one representative cell, so downstream QC + per-cell archive +
    # offline analyses see each cell exactly once. Type-aware: an OnP
    # never merges with an OffM. Same for ALL protocols using this
    # driver — there's no protocol-specific logic.
    if dedup:
        from .utils.dedup import dedup_pipeline as _dedup_pipeline
        n_before = len(response_block.df_spike_times)
        _dedup_pipeline(
            pipeline,
            ei_threshold=dedup_ei_threshold,
            skip_untyped=dedup_skip_untyped,
            merge_strategy=dedup_merge_strategy,
            refractory_ms=dedup_refractory_ms,
            verbose=False,
        )
        n_after = len(response_block.df_spike_times)
        if verbose and n_after < n_before:
            print(f'  dedup: {n_before - n_after} duplicate '
                  f'cell(s) merged into representatives '
                  f'(EI≥{dedup_ei_threshold:.2f}, {dedup_merge_strategy})')

    # 3. Filter requested cell_types to those actually present with n >= minimum_n
    type_counts = response_block.df_spike_times['cell_type'].value_counts()
    present = [t for t, n in type_counts.items() if n >= minimum_n]
    cell_types_used = filter_available_types(list(cell_types), present)
    if verbose:
        print(f'  cell_types_used={cell_types_used}  '
              f'(skipped {[t for t in cell_types if t not in cell_types_used]})')

    # 4. Calibration (lazy or on-demand)
    if fit_calibration:
        try:
            calib = rc.fit_calibration_for_chunk(analysis_chunk, verbose=False)
            if save_calibration:
                rc.save_rig_calibration(calib)
            if verbose:
                print(f'  calibration (fit): rotation={calib.rotation_deg:.2f}° '
                      f'residual={calib.residual_um_rms:.1f} µm  '
                      f'n_cells={calib.n_cells}')
        except Exception as exc:
            calib = None
            if verbose:
                print(f'  calibration fit failed ({exc!r}); '
                      f'falling back to saved/geometric')
    else:
        calib = rc.load_rig_calibration(exp_name)
        if verbose:
            if calib is not None:
                print(f'  calibration (loaded): rotation={calib.rotation_deg:.2f}° '
                      f'residual={calib.residual_um_rms:.1f} µm')
            else:
                print('  calibration: none on disk; using geometric fallback')

    # 5. QC
    timing = response_block.d_timing or {}
    t_total_ms = (
        float(timing.get('pre_time_ms', 0))
        + float(timing.get('stim_time_ms', 0))
        + float(timing.get('tail_time_ms', 0))
    )
    qc = filter_cells_by_qc(
        block_qc_metrics(
            response_block, t_start_ms=0.0, t_end_ms=t_total_ms,
            sample_rate_hz=sample_rate_hz,
        ),
        thresholds=qc_thresholds,
    )
    n_pass = int(qc['passes'].sum())
    if verbose:
        print(f'  QC: {n_pass}/{len(qc)} cells pass ({100*qc["passes"].mean():.0f}%)')

    # 5b. Resolve protocol subdir. By default we use the protocol short
    # name. When multiple datafiles share the same protocol on one date
    # they would otherwise collide in <exp>/<protocol_short>/, so callers
    # can override via protocol_subdir= or the
    # append_datafile_to_subdir=True shortcut to get
    # <protocol_short>_<datafile_name>/.
    from .utils.cell_plot_archive import protocol_short_name
    _proto_short = protocol_short_name(response_block.protocol_name)
    if protocol_subdir is not None:
        _proto_short = protocol_subdir
    elif append_datafile_to_subdir:
        _proto_short = f'{_proto_short}_{datafile_name}'
    if verbose and _proto_short != protocol_short_name(response_block.protocol_name):
        print(f'  protocol_subdir → {_proto_short!r}')

    # Persist QC results so downstream tools can filter without re-running
    # the metrics. Lives at <exp>/<_proto_short>/qc.csv next to
    # index.csv / cell_match.csv / visual_qc.csv.
    try:
        _qc_path = save_protocol_qc(
            qc, exp_name, protocol=_proto_short, output_root=output_root,
        )
        if verbose:
            print(f'  qc → {_qc_path}')
    except Exception as exc:
        if verbose:
            print(f'  qc save: skipped ({exc!r})')

    # 5c. Visual QC integration: when caller hasn't pinned cell_ids,
    # see if a tagged visual_qc.csv exists and prefer those tags. This
    # makes the batch driver inherit the same behavior as the single-
    # date archive cell, with no extra plumbing per date.
    if cell_ids is None and respect_visual_qc:
        try:
            from .utils.visual_qc import visual_qc_csv_path
            import pandas as _pd
            _vqc_path = visual_qc_csv_path(
                exp_name, _proto_short, output_root=output_root,
            )
            if _vqc_path.exists():
                _vqc = _pd.read_csv(_vqc_path)
                _good = (_vqc.loc[_vqc['tag'] == 'good', 'cell_id']
                              .astype(int).tolist())
                _n_bad = int((_vqc['tag'] == 'bad').sum())
                cell_ids = _good
                if verbose:
                    print(f'  visual_qc → {len(_vqc)} tags '
                          f'({len(_good)} good, {_n_bad} bad); '
                          f'archive restricted to good set')
        except Exception as exc:
            if verbose:
                print(f'  visual_qc check: skipped ({exc!r})')

    # 6. Persist the noise↔protocol match table + per-cell EI stats.
    # Cheap; safe to redo on every run since it just snapshots the live
    # pipeline. Lives next to index.csv at <exp>/<protocol>/cell_match.csv.
    try:
        cm_path = save_cell_match(pipeline, output_root=output_root,
                                  qc_pass_only=qc,
                                  protocol_subdir=_proto_short)
        if verbose:
            print(f'  cell_match → {cm_path}')
    except Exception as exc:
        if verbose:
            print(f'  cell_match: skipped ({exc!r})')

    # 7. Save mosaic + per-cell PNGs
    ndf = _extract_ndf(stim_block)
    idx = save_per_cell_plots(
        analysis_chunk, response_block,
        stim_block=stim_block,
        protocol_name=response_block.protocol_name,
        protocol_short_name_=_proto_short,
        cell_types=cell_types_used,
        cell_ids=list(cell_ids) if cell_ids is not None else None,
        qc_pass_only=qc,
        condition_keys=condition_keys,
        main_types=MAIN_CELL_TYPES,
        typing_file=typing_file,
        psth_sigma_ms=psth_sigma_ms,
        sample_rate_hz=sample_rate_hz,
        psth_ncols=psth_ncols,
        output_root=output_root,
        overwrite=overwrite,
        n_jobs=n_jobs,
        verbose=verbose,
        ndf=ndf,
        prune_stale=prune_stale,
    )

    return {
        'exp_name': exp_name,
        'datafile_name': datafile_name,
        'ss_version': ss_version,
        'typing_file': typing_file,
        'chunk_name': analysis_chunk.chunk_name,
        'cell_types_used': list(cell_types_used),
        'n_cells_total': int(len(qc)),
        'n_cells_passed_qc': n_pass,
        'calibration': (calib.to_dict() if calib is not None else None),
        'output_dir': experiment_root(exp_name, output_root=output_root),
        'index_df': idx,
        'ndf': ndf,
    }


def analyze_experiments(
    exp_names: Iterable[str],
    *,
    datafile_names: Optional[Dict[str, str]] = None,
    on_error: str = 'log',
    **kwargs,
) -> List[Dict]:
    """Run :func:`analyze_experiment` over many dates.

    Parameters
    ----------
    exp_names : iterable[str]
        Experiments to analyze.
    datafile_names : dict, optional
        ``{exp_name: datafile_name}`` overrides; otherwise the
        ``protocol_search`` kwarg (passed through) is used.
    on_error : ``'log'`` (default) or ``'raise'``
        ``'log'`` records failures in the returned summaries and continues;
        ``'raise'`` re-raises so the loop stops.
    **kwargs
        Forwarded to :func:`analyze_experiment`.

    Returns
    -------
    list[dict]
        One summary dict per experiment (with ``error`` set on failures).
    """
    results = []
    datafile_names = datafile_names or {}
    for exp in exp_names:
        try:
            summary = analyze_experiment(
                exp, datafile_name=datafile_names.get(exp), **kwargs,
            )
            results.append(summary)
        except Exception as exc:
            if on_error == 'raise':
                raise
            tb = traceback.format_exc(limit=2)
            print(f'[analyze_experiments] {exp}: FAILED — {exc!r}')
            results.append({
                'exp_name': exp, 'error': repr(exc), 'traceback': tb,
            })
    return results
