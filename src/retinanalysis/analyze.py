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
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .config.settings import ANALYSIS_DIR, DATA_DIR
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


__all__ = ['analyze_experiment', 'analyze_experiments']


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
    if kind == 'data':
        if datafile_name is None:
            raise ValueError("kind='data' requires datafile_name")
        sort_dir = os.path.join(DATA_DIR, exp_name, datafile_name)
        location_desc = f'{exp_name}/{datafile_name} (data)'
    elif kind == 'analysis':
        if chunk_name is None:
            raise ValueError("kind='analysis' requires chunk_name")
        from .config.settings import ANALYSIS_DIR as _ANALYSIS_DIR
        sort_dir = os.path.join(_ANALYSIS_DIR, exp_name, chunk_name)
        location_desc = f'{exp_name}/{chunk_name} (analysis)'
    else:
        raise ValueError(f"kind must be 'data' or 'analysis', got {kind!r}")

    if not os.path.isdir(sort_dir):
        raise FileNotFoundError(
            f'No sort directory for {location_desc}: {sort_dir}')
    chosen = _pick_ss_version_from(os.listdir(sort_dir))
    if chosen is None:
        raise FileNotFoundError(f'No kilosort* subdirs under {sort_dir}')
    return chosen


# Backward-compat alias used internally.
def _detect_ss_version(exp_name: str, datafile_name: str) -> str:
    return detect_ss_version(exp_name, datafile_name, kind='data')


def _pick_typing_file(
    exp_name: str, chunk_name: str, ss_version: str,
    preferred: Optional[str] = None,
) -> Optional[str]:
    """Pick the first non-dotfile ``*.classification.txt`` in the chunk dir."""
    chunk_dir = os.path.join(ANALYSIS_DIR, exp_name, chunk_name, ss_version)
    if not os.path.isdir(chunk_dir):
        return preferred
    candidates = [
        f for f in os.listdir(chunk_dir)
        if f.endswith('.classification.txt') and not f.startswith('.')
    ]
    if preferred and preferred in candidates:
        return preferred
    return candidates[0] if candidates else None


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
    """Pull NDF from the first epoch's parameters if present."""
    try:
        v = stim_block.df_epochs['epoch_parameters'].iloc[0].get('NDF')
        return float(v) if v is not None else None
    except (KeyError, IndexError, ValueError, TypeError):
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
        # Need the nearest noise chunk first to know where typing files live.
        tmp = MEAStimBlock(exp_name, datafile_name, verbose=False)
        typing_file = _pick_typing_file(exp_name, tmp.nearest_noise_chunk, ss_version)
        if verbose:
            print(f'  typing_file={typing_file}')

    pipeline = create_mea_pipeline(
        exp_name, datafile_name,
        ss_version=ss_version, typing_file=typing_file,
        verbose=False,
    )
    analysis_chunk = pipeline.analysis_chunk
    response_block = pipeline.resp
    stim_block = pipeline.stim
    _normalize_cell_types(response_block)

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
