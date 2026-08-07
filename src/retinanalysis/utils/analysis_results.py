"""Structured persistence for incremental, cross-date protocol analyses.

Layout::

    <OUTPUT_DIR>/protocol_analysis/<proc>/<date>/
        analysis.pkl
        meta.json
        plots/*.png
    <OUTPUT_DIR>/protocol_analysis/<proc>/summary/
        analysis.pkl
        meta.json
        plots/*.png

Pickle preserves pandas/xarray/numpy objects without flattening them into CSV.
JSON keeps the provenance and basic statistics readable without Python. Only
load bundles created locally by this package; pickle is not a safe interchange
format for untrusted files.
"""
from __future__ import annotations

import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


__all__ = [
    'analysis_output_dir',
    'list_analysis_dates',
    'saved_analysis_stats',
    'save_analysis_bundle',
    'save_analysis_summary',
    'load_analysis_bundle',
    'load_analysis_many',
]


_ROOT_FOLDER = 'protocol_analysis'
_ANALYSIS_FILE = 'analysis.pkl'
_META_FILE = 'meta.json'


def _root(output_root=None) -> Path:
    if output_root is None:
        from retinanalysis.config.settings import OUTPUT_DIR
        return Path(OUTPUT_DIR) / _ROOT_FOLDER
    return Path(output_root) / _ROOT_FOLDER


def _safe_name(value: str) -> str:
    name = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('._')
    if not name:
        raise ValueError(f'cannot make a safe output name from {value!r}')
    return name


def analysis_output_dir(
    proc: str,
    exp_name: Optional[str] = None,
    *,
    summary: bool = False,
    output_root=None,
) -> Path:
    """Resolve a date or cross-date summary directory without creating it."""
    if summary and exp_name is not None:
        raise ValueError('pass either exp_name or summary=True, not both')
    if not summary and exp_name is None:
        return _root(output_root) / _safe_name(proc)
    leaf = 'summary' if summary else _safe_name(str(exp_name))
    return _root(output_root) / _safe_name(proc) / leaf


def list_analysis_dates(proc: str, *, output_root=None) -> list[str]:
    """Sorted dates that contain a complete analysis bundle."""
    parent = analysis_output_dir(proc, output_root=output_root)
    if not parent.is_dir():
        return []
    return [
        path.name for path in sorted(parent.iterdir())
        if path.is_dir() and path.name != 'summary'
        and (path / _ANALYSIS_FILE).is_file()
        and (path / _META_FILE).is_file()
    ]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _object_stats(value) -> Dict:
    stats = {'type': type(value).__name__}
    if isinstance(value, pd.DataFrame):
        stats.update(rows=int(len(value)), columns=list(value.columns))
        if 'cell_id' in value:
            stats['n_cells'] = int(value['cell_id'].nunique())
        elif 'n_cells' in value and len(value):
            stats['n_cells'] = int(pd.to_numeric(
                value['n_cells'], errors='coerce').max())
        if 'cell_type' in value:
            stats['cell_types'] = sorted(
                str(v) for v in value['cell_type'].dropna().unique())
    elif isinstance(value, pd.Series):
        stats.update(rows=int(len(value)), name=str(value.name))
    elif hasattr(value, 'shape'):
        stats['shape'] = [int(v) for v in value.shape]
    elif isinstance(value, Mapping):
        stats['n_items'] = int(len(value))
    elif hasattr(value, '__len__') and not isinstance(value, str):
        stats['n_items'] = int(len(value))
    return stats


def _read_meta(path: Path) -> Dict:
    try:
        with path.open(encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def saved_analysis_stats(proc: str, *, output_root=None) -> pd.DataFrame:
    """Basic statistics for date bundles, read from lightweight JSON only."""
    rows = []
    for exp_name in list_analysis_dates(proc, output_root=output_root):
        folder = analysis_output_dir(proc, exp_name, output_root=output_root)
        meta = _read_meta(folder / _META_FILE)
        objects = meta.get('objects', {})
        row_counts = [v.get('rows') for v in objects.values()
                      if isinstance(v, dict) and v.get('rows') is not None]
        cell_counts = [v.get('n_cells') for v in objects.values()
                       if isinstance(v, dict) and v.get('n_cells') is not None]
        rows.append({
            'exp_name': exp_name,
            'saved_utc': meta.get('saved_utc', ''),
            'n_outputs': len(objects),
            'outputs': ', '.join(sorted(objects)),
            'total_rows': int(sum(row_counts)) if row_counts else 0,
            'max_cells': int(max(cell_counts)) if cell_counts else 0,
            'n_plots': len(meta.get('plots', [])),
        })
    return pd.DataFrame(rows, columns=[
        'exp_name', 'saved_utc', 'n_outputs', 'outputs',
        'total_rows', 'max_cells', 'n_plots',
    ])


def _save_bundle(
    proc: str,
    analysis: Mapping[str, object],
    *,
    exp_name: Optional[str],
    summary: bool,
    metadata: Optional[Mapping] = None,
    figures: Optional[Mapping[str, object]] = None,
    output_root=None,
    dpi: int = 200,
) -> Dict[str, Path]:
    folder = analysis_output_dir(
        proc, exp_name, summary=summary, output_root=output_root)
    plots_dir = folder / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = folder / _ANALYSIS_FILE
    meta_path = folder / _META_FILE
    replaced_existing = analysis_path.exists() or meta_path.exists()

    plot_paths = []
    for name, figure in (figures or {}).items():
        path = plots_dir / f'{_safe_name(name)}.png'
        figure.savefig(path, dpi=int(dpi), bbox_inches='tight')
        plot_paths.append(path)

    meta = {
        'proc': _safe_name(proc),
        'scope': 'summary' if summary else 'date',
        'exp_name': None if summary else str(exp_name),
        'write_mode': ('overwrite_existing' if replaced_existing else 'new'),
        'saved_utc': datetime.now(timezone.utc).isoformat(),
        'objects': {str(k): _object_stats(v) for k, v in analysis.items()},
        'plots': [str(path.relative_to(folder)) for path in plot_paths],
        **dict(metadata or {}),
    }
    meta = _jsonable(meta)

    folder.mkdir(parents=True, exist_ok=True)
    analysis_tmp = folder / f'{_ANALYSIS_FILE}.tmp'
    meta_tmp = folder / f'{_META_FILE}.tmp'
    with analysis_tmp.open('wb') as handle:
        pickle.dump(dict(analysis), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with meta_tmp.open('w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write('\n')
    analysis_tmp.replace(analysis_path)
    meta_tmp.replace(meta_path)

    # A rerun is an exact replacement for this date. Remove plots recorded by
    # an older run but omitted from the new figure mapping, so stale figures
    # cannot be mistaken for part of the current analysis.
    current_plots = {path.name for path in plot_paths}
    for old_plot in plots_dir.glob('*.png'):
        if old_plot.name not in current_plots:
            old_plot.unlink()
    return {
        'folder': folder,
        'analysis': analysis_path,
        'metadata': meta_path,
        'plots': plots_dir,
    }


def save_analysis_bundle(
    proc: str,
    exp_name: str,
    analysis: Mapping[str, object],
    *,
    metadata: Optional[Mapping] = None,
    figures: Optional[Mapping[str, object]] = None,
    output_root=None,
    dpi: int = 200,
    verbose: bool = True,
) -> Dict[str, Path]:
    """Print existing dates, then exactly save/replace one date's bundle.

    Reusing an ``exp_name`` overwrites its pickle and JSON and removes stale
    PNGs from its plot folder. Bundles for every other experiment are left
    untouched.
    """
    exists = str(exp_name) in list_analysis_dates(
        proc, output_root=output_root)
    if verbose:
        existing = saved_analysis_stats(proc, output_root=output_root)
        print(f'{proc} dates saved before this update:')
        print('  none' if existing.empty else existing.to_string(index=False))
    paths = _save_bundle(
        proc, analysis, exp_name=str(exp_name), summary=False,
        metadata=metadata, figures=figures, output_root=output_root, dpi=dpi)
    if verbose:
        action = 'Replacing existing' if exists else 'Adding new'
        print(f'\n{action} {proc}/{exp_name}')
        print(f'Saved {proc}/{exp_name} -> {paths["folder"]}')
        current = saved_analysis_stats(proc, output_root=output_root)
        current = current[current['exp_name'] == str(exp_name)]
        if not current.empty:
            print(current.to_string(index=False))
    return paths


def save_analysis_summary(
    proc: str,
    analysis: Mapping[str, object],
    *,
    metadata: Optional[Mapping] = None,
    figures: Optional[Mapping[str, object]] = None,
    output_root=None,
    dpi: int = 200,
    verbose: bool = True,
) -> Dict[str, Path]:
    """Save the combined dataset, cross-date metadata, and summary plots."""
    paths = _save_bundle(
        proc, analysis, exp_name=None, summary=True,
        metadata=metadata, figures=figures, output_root=output_root, dpi=dpi)
    if verbose:
        print(f'Saved {proc} cross-date summary -> {paths["folder"]}')
    return paths


def load_analysis_bundle(
    proc: str,
    exp_name: Optional[str] = None,
    *,
    summary: bool = False,
    output_root=None,
) -> Dict:
    """Load a trusted local pickle bundle plus its JSON metadata."""
    folder = analysis_output_dir(
        proc, exp_name, summary=summary, output_root=output_root)
    with (folder / _ANALYSIS_FILE).open('rb') as handle:
        analysis = pickle.load(handle)
    return {'analysis': analysis, 'meta': _read_meta(folder / _META_FILE),
            'folder': folder}


def load_analysis_many(
    proc: str,
    exp_names: Optional[Iterable[str]] = None,
    *,
    output_root=None,
) -> Dict[str, Dict]:
    """Load date bundles as ``{exp_name: {'analysis', 'meta', 'folder'}}``."""
    dates = (list(exp_names) if exp_names is not None
             else list_analysis_dates(proc, output_root=output_root))
    return {
        str(exp): load_analysis_bundle(
            proc, str(exp), output_root=output_root)
        for exp in dates
    }
