"""Local file-cache for Vision analysis files.

When the user is working off a remote NAS mount, every kernel restart that
rebuilds the pipeline pays the same bandwidth bill — hundreds of MB of
``.ei`` / ``.neurons`` / ``.params`` / ``.classification.txt`` files pulled
over the wire just to look at the same experiment again.

This module copies those files into ``LOCAL_CACHE_ROOT`` (default
``~/.cache/retinanalysis``). The ``local_cache`` tier sits at the top of
``find_path()``'s priority list, so once a file is mirrored, every
subsequent ``get_protocol_vcd`` / ``get_analysis_vcd`` call transparently
reads from local disk.

Typical use::

    # Once per (date, datafile, chunk), before the first ra.create_mea_pipeline
    summary = ra.mirror_to_local_cache(exp_name, datafile_name, chunk_name)
    # Now build the pipeline — all Vision reads are local, no NAS traffic.
    pipeline = ra.create_mea_pipeline(exp_name, datafile_name, ...)
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..config.settings import (
    LOCAL_CACHE_ROOT, mea_config, find_path, _TIER_PRIORITY,
)


def _find_canonical_source(kind: str, *parts) -> str:
    """Like ``find_path`` but **skips the local_cache tier**.

    The cache is a destination, not a source. When a mirror needs to
    pull fresh files, it must read from the real upstream tier — not
    from a partially-populated cache (which would silently leave new
    upstream files behind, e.g. a freshly-rerun ``.classification.txt``
    that postdates the last mirror).
    """
    fallback = None
    for tier in _TIER_PRIORITY:
        if tier == 'local_cache':
            continue
        if tier not in mea_config:
            continue
        root = mea_config[tier].get(kind, '')
        if not root:
            continue
        candidate = os.path.join(root, *parts)
        if fallback is None:
            fallback = candidate
        if os.path.exists(candidate):
            return candidate
    return fallback


__all__ = [
    'mirror_to_local_cache',
    'local_cache_status',
    'clear_local_cache',
    'create_mea_pipeline_cached',
    'load_cached_pipeline',
    'pipeline_cache_path',
    'LOCAL_CACHE_ROOT',
]


# Files Vision actually reads. The Kilosort ``.npy`` outputs and ``.tsv``
# Phy cluster tables are NOT in this list — they're only needed at sorting
# time, not at downstream pipeline-build time, so we save the disk space.
# ``.sta`` (raw STA movies, often ~9 GB) is also skipped by default; pass
# ``include_sta=True`` to bring it along.
_VISION_NEED_SUFFIXES: Tuple[str, ...] = (
    '.ei',
    '.neurons',
    '.params',
    '.globals',
    '.noise',
    '.classification.txt',
    '_params.mat',
)


def _human_mb(n_bytes: int) -> str:
    if n_bytes < 1e6:
        return f'{n_bytes/1e3:.1f} KB'
    if n_bytes < 1e9:
        return f'{n_bytes/1e6:.1f} MB'
    return f'{n_bytes/1e9:.2f} GB'


def _vision_files_in(src_dir: str, include_sta: bool) -> List[str]:
    """Return basenames of Vision files we care about in ``src_dir``."""
    if not os.path.isdir(src_dir):
        return []
    keep = []
    suffixes = _VISION_NEED_SUFFIXES + (('.sta',) if include_sta else ())
    for name in os.listdir(src_dir):
        # Skip macOS AppleDouble dotfile siblings (._foo) — they're
        # metadata, not real data, and break Vision readers when copied.
        if name.startswith('.'):
            continue
        if any(name.endswith(s) for s in suffixes):
            keep.append(name)
    return keep


def _copy_if_newer(src: str, dst: str) -> Tuple[bool, int]:
    """Copy src→dst when missing or stale. Returns (copied?, bytes_copied)."""
    if os.path.exists(dst):
        sst = os.stat(src)
        dst_st = os.stat(dst)
        # Same size + mtime ≥ src mtime → trust the cache copy.
        if sst.st_size == dst_st.st_size and dst_st.st_mtime >= sst.st_mtime:
            return (False, 0)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # copy2 preserves mtime so the staleness check above works on reruns.
    shutil.copy2(src, dst)
    return (True, os.path.getsize(dst))


def _mirror_dir(src_dir: str, dst_dir: str, include_sta: bool,
                 verbose: bool, label: str) -> Dict:
    """Mirror Vision files from ``src_dir`` to ``dst_dir``.

    Returns a per-call report dict so the caller can roll up totals.
    """
    if not os.path.isdir(src_dir):
        if verbose:
            print(f'  [{label}] source missing: {src_dir}  — skip')
        return {'label': label, 'src': src_dir, 'dst': dst_dir,
                'n_files': 0, 'n_copied': 0,
                'bytes_total': 0, 'bytes_copied': 0, 'seconds': 0.0}
    names = _vision_files_in(src_dir, include_sta=include_sta)
    if not names:
        if verbose:
            print(f'  [{label}] no Vision files in {src_dir}')
    t0 = time.time()
    bytes_total = 0
    bytes_copied = 0
    n_copied = 0
    for name in names:
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        try:
            sz = os.path.getsize(src)
            bytes_total += sz
            copied, b = _copy_if_newer(src, dst)
            bytes_copied += b
            if copied:
                n_copied += 1
                if verbose:
                    print(f'  [{label}] copied {name}  ({_human_mb(sz)})')
            else:
                if verbose:
                    print(f'  [{label}] cached  {name}  ({_human_mb(sz)})')
        except Exception as exc:
            if verbose:
                print(f'  [{label}] FAILED {name}: {exc!r}')
    elapsed = time.time() - t0
    return {
        'label': label, 'src': src_dir, 'dst': dst_dir,
        'n_files': len(names), 'n_copied': n_copied,
        'bytes_total': bytes_total, 'bytes_copied': bytes_copied,
        'seconds': elapsed,
    }


def mirror_to_local_cache(
    exp_name: str,
    datafile_name: Optional[str] = None,
    chunk_name: Optional[str] = None,
    ss_version: str = 'kilosort2.5',
    *,
    include_sta: bool = False,
    verbose: bool = True,
) -> Dict:
    """Copy Vision files for one experiment into ``LOCAL_CACHE_ROOT``.

    After this call, the ``local_cache`` tier — which sits at the top of
    ``find_path()``'s priority list — owns the relevant files. Subsequent
    ``ra.create_mea_pipeline(exp_name, datafile_name, ...)`` calls read
    them from local disk; the NAS sees zero traffic for those files.

    Parameters
    ----------
    exp_name : str
        Experiment id (e.g. ``'20251112C'``).
    datafile_name : str, optional
        Protocol datafile (e.g. ``'data019'``) to mirror under
        ``data/<exp>/<datafile>/<ss_version>/``. Pass ``None`` to skip
        the protocol-side mirror.
    chunk_name : str, optional
        Noise/sorting chunk (e.g. ``'chunk3'``) to mirror under
        ``analysis/<exp>/<chunk>/<ss_version>/``. Pass ``None`` to skip.
    ss_version : str
        Kilosort version. Default ``'kilosort2.5'``.
    include_sta : bool
        Also mirror the (potentially huge — ~9 GB) ``.sta`` raw STA
        movie. Default False — STA *fits* live in ``.params`` and are
        the only thing the rest of the pipeline reads.
    verbose : bool
        Per-file log lines.

    Returns
    -------
    dict
        ``{'exp_name', 'datafile_name', 'chunk_name', 'ss_version',
        'protocol', 'chunk', 'bytes_copied_total', 'bytes_total',
        'seconds_total'}``. ``'protocol'`` and ``'chunk'`` are the
        per-side report dicts from ``_mirror_dir``.
    """
    if datafile_name is None and chunk_name is None:
        raise ValueError(
            'Pass at least one of datafile_name / chunk_name to mirror.')
    if verbose:
        print(f'Mirroring Vision files for {exp_name} into {LOCAL_CACHE_ROOT}')
    proto_report = None
    chunk_report = None

    # Protocol datafile side (DATA_DIR). Use _find_canonical_source so
    # we don't read from the cache we're about to write into.
    if datafile_name is not None:
        src = _find_canonical_source(
            'data', exp_name, datafile_name, ss_version)
        dst = os.path.join(mea_config['local_cache']['data'],
                           exp_name, datafile_name, ss_version)
        proto_report = _mirror_dir(src, dst, include_sta=include_sta,
                                     verbose=verbose,
                                     label=f'data/{datafile_name}')

    # Noise/sorting chunk side (ANALYSIS_DIR). Same canonical-source
    # rule applies.
    if chunk_name is not None:
        src = _find_canonical_source(
            'analysis', exp_name, chunk_name, ss_version)
        dst = os.path.join(mea_config['local_cache']['analysis'],
                           exp_name, chunk_name, ss_version)
        chunk_report = _mirror_dir(src, dst, include_sta=include_sta,
                                     verbose=verbose,
                                     label=f'analysis/{chunk_name}')

    bytes_copied = sum(r['bytes_copied'] for r in (proto_report, chunk_report)
                        if r is not None)
    bytes_total = sum(r['bytes_total'] for r in (proto_report, chunk_report)
                       if r is not None)
    seconds = sum(r['seconds'] for r in (proto_report, chunk_report)
                   if r is not None)
    if verbose:
        print(f'Total: {_human_mb(bytes_copied)} copied, '
              f'{_human_mb(bytes_total - bytes_copied)} already cached, '
              f'in {seconds:.1f} s')
    return {
        'exp_name': exp_name,
        'datafile_name': datafile_name,
        'chunk_name': chunk_name,
        'ss_version': ss_version,
        'protocol': proto_report,
        'chunk': chunk_report,
        'bytes_copied_total': bytes_copied,
        'bytes_total': bytes_total,
        'seconds_total': seconds,
    }


def local_cache_status() -> Dict:
    """Summarize what is currently in the local cache.

    Returns a dict ``{'root': LOCAL_CACHE_ROOT, 'exists': bool,
    'n_experiments': int, 'total_bytes': int, 'experiments': [...]}``
    where each experiment entry lists the datafiles and chunks present
    plus their byte counts.
    """
    root = Path(LOCAL_CACHE_ROOT)
    out: Dict = {
        'root': str(root),
        'exists': root.exists(),
        'n_experiments': 0,
        'total_bytes': 0,
        'experiments': [],
    }
    if not root.exists():
        return out

    exp_seen = set()
    by_exp: Dict[str, Dict] = {}
    for kind in ('data', 'analysis'):
        kind_dir = root / kind
        if not kind_dir.is_dir():
            continue
        for exp_dir in sorted(kind_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            exp_seen.add(exp_dir.name)
            ent = by_exp.setdefault(exp_dir.name, {
                'exp_name': exp_dir.name,
                'datafiles': [],
                'chunks': [],
                'bytes': 0,
            })
            for sub in sorted(exp_dir.iterdir()):
                if not sub.is_dir():
                    continue
                size = 0
                for dirpath, _dirs, files in os.walk(sub):
                    for f in files:
                        try:
                            size += os.path.getsize(os.path.join(dirpath, f))
                        except OSError:
                            pass
                ent['bytes'] += size
                target = 'datafiles' if kind == 'data' else 'chunks'
                ent[target].append({'name': sub.name, 'bytes': size})
    out['experiments'] = [by_exp[k] for k in sorted(by_exp)]
    out['n_experiments'] = len(exp_seen)
    out['total_bytes'] = sum(e['bytes'] for e in out['experiments'])
    return out


def clear_local_cache(
    exp_name: Optional[str] = None,
    *,
    verbose: bool = True,
) -> Dict:
    """Delete cached Vision files. ``exp_name=None`` wipes the whole cache."""
    root = Path(LOCAL_CACHE_ROOT)
    removed_bytes = 0
    removed_paths: List[str] = []
    if not root.exists():
        if verbose:
            print(f'Local cache does not exist at {root}')
        return {'root': str(root), 'removed_bytes': 0, 'paths': []}

    if exp_name is None:
        # Wipe everything under root, but preserve the root dir itself.
        for kind in ('data', 'analysis', 'raw'):
            kd = root / kind
            if kd.exists():
                for dirpath, _dirs, files in os.walk(kd):
                    for f in files:
                        p = os.path.join(dirpath, f)
                        try:
                            removed_bytes += os.path.getsize(p)
                        except OSError:
                            pass
                shutil.rmtree(kd, ignore_errors=True)
                removed_paths.append(str(kd))
    else:
        for kind in ('data', 'analysis', 'raw'):
            ed = root / kind / exp_name
            if ed.exists():
                for dirpath, _dirs, files in os.walk(ed):
                    for f in files:
                        try:
                            removed_bytes += os.path.getsize(
                                os.path.join(dirpath, f))
                        except OSError:
                            pass
                shutil.rmtree(ed, ignore_errors=True)
                removed_paths.append(str(ed))

    if verbose:
        print(f'Removed {_human_mb(removed_bytes)} from local cache '
              f'({len(removed_paths)} dirs).')
    return {'root': str(root),
            'removed_bytes': removed_bytes,
            'paths': removed_paths}


# ---------------------------------------------------------------------------
# Pipeline-object cache (one .pkl per build).
#
# Step 3a (mirror_to_local_cache) eliminates *bandwidth* — but every kernel
# restart still re-runs DataJoint queries, cluster_match, and the per-cell
# attribute pulls. This second-tier cache saves the built MEAPipeline
# object itself, so a kernel restart turns into a single pickle.load().
# ---------------------------------------------------------------------------
import hashlib
import json


_PIPELINE_CACHE_KIND = 'pipelines'

# These build-time knobs change the contents of the pipeline. Two builds
# with the same key always produce the same object; changing any key
# invalidates the cache. Keep this list in sync with create_mea_pipeline().
_PIPELINE_CACHE_KEYS = (
    'exp_name', 'datafile_name', 'ss_version',
    'analysis_chunk_name', 'typing_file',
    'ei_corr_cutoff', 'ei_match_method',
    'ei_use_isi', 'ei_use_timecourse', 'ei_n_removed_channels',
)


def _normalize_kwargs(kwargs: Dict) -> Dict:
    """Canonicalize for hashing — lists → tuples, ints/floats kept as-is."""
    out: Dict = {}
    for k in _PIPELINE_CACHE_KEYS:
        v = kwargs.get(k, None)
        if isinstance(v, list):
            v = tuple(v)
        out[k] = v
    return out


def pipeline_cache_path(
    exp_name: str,
    datafile_name,
    *,
    ss_version: str = 'kilosort2.5',
    analysis_chunk_name: Optional[str] = None,
    typing_file: Optional[str] = None,
    ei_corr_cutoff: float = 0.8,
    ei_match_method: str = 'all',
    ei_use_isi: bool = False,
    ei_use_timecourse: bool = False,
    ei_n_removed_channels: int = 1,
    cache_root: Optional[str] = None,
) -> Path:
    """Deterministic local path for the cached pipeline pkl.

    Two builds with the same inputs map to the same file; changing any
    of the EI-match knobs (cutoff, method, …) changes the hash, so the
    user can never silently load a stale build.
    """
    kwargs = dict(
        exp_name=exp_name, datafile_name=datafile_name,
        ss_version=ss_version, analysis_chunk_name=analysis_chunk_name,
        typing_file=typing_file,
        ei_corr_cutoff=float(ei_corr_cutoff),
        ei_match_method=ei_match_method,
        ei_use_isi=bool(ei_use_isi),
        ei_use_timecourse=bool(ei_use_timecourse),
        ei_n_removed_channels=int(ei_n_removed_channels),
    )
    norm = _normalize_kwargs(kwargs)
    payload = json.dumps(norm, sort_keys=True, default=str).encode()
    short = hashlib.sha1(payload).hexdigest()[:10]

    # Datafile in the filename for human readability; the hash absorbs
    # all other knobs. Tolerate a list-of-datafiles (MEAStimGroup).
    df_tag = datafile_name if isinstance(datafile_name, str) else 'multi'
    fname = f'{exp_name}__{df_tag}__{ss_version}__{short}.pkl'

    root = Path(cache_root) if cache_root else Path(LOCAL_CACHE_ROOT)
    return root / _PIPELINE_CACHE_KIND / fname


def load_cached_pipeline(
    exp_name: str,
    datafile_name,
    *,
    ss_version: str = 'kilosort2.5',
    analysis_chunk_name: Optional[str] = None,
    typing_file: Optional[str] = None,
    ei_corr_cutoff: float = 0.8,
    ei_match_method: str = 'all',
    ei_use_isi: bool = False,
    ei_use_timecourse: bool = False,
    ei_n_removed_channels: int = 1,
    cache_root: Optional[str] = None,
    verbose: bool = True,
):
    """Return a cached ``MEAPipeline`` or ``None`` if not cached.

    Reloads via the existing ``MEAPipeline(pkl_file=...)`` path, which
    re-initialises the per-block ``vcd`` objects through
    ``find_path()``. With the file-level mirror from
    ``mirror_to_local_cache`` in place, those reads are local too.
    """
    from ..classes.mea_pipeline import MEAPipeline
    path = pipeline_cache_path(
        exp_name, datafile_name,
        ss_version=ss_version,
        analysis_chunk_name=analysis_chunk_name,
        typing_file=typing_file,
        ei_corr_cutoff=ei_corr_cutoff,
        ei_match_method=ei_match_method,
        ei_use_isi=ei_use_isi,
        ei_use_timecourse=ei_use_timecourse,
        ei_n_removed_channels=ei_n_removed_channels,
        cache_root=cache_root,
    )
    if not path.exists():
        if verbose:
            print(f'No cached pipeline at {path}')
        return None
    if verbose:
        size_mb = path.stat().st_size / 1e6
        print(f'Loading cached pipeline ({size_mb:.1f} MB) from {path}')
    t0 = time.time()
    pipeline = MEAPipeline(pkl_file=str(path), verbose=verbose)
    if verbose:
        print(f'  loaded in {time.time() - t0:.1f} s')
    return pipeline


def create_mea_pipeline_cached(
    exp_name: str,
    datafile_name,
    *,
    overwrite: bool = False,
    cache_root: Optional[str] = None,
    verbose: bool = True,
    ss_version: str = 'kilosort2.5',
    analysis_chunk_name: Optional[str] = None,
    typing_file: Optional[str] = None,
    ei_corr_cutoff: float = 0.8,
    ei_match_method: str = 'all',
    ei_use_isi: bool = False,
    ei_use_timecourse: bool = False,
    ei_n_removed_channels: int = 1,
    **extra_kwargs,
):
    """Drop-in for ``ra.create_mea_pipeline`` with a local pkl cache.

    On first call: builds the pipeline normally, then pickles it under
    ``<LOCAL_CACHE_ROOT>/pipelines/`` keyed by the build kwargs.
    Subsequent calls with the same kwargs return the cached object.

    The cache file is keyed on the EI-match knobs (cutoff, method,
    use_isi, use_timecourse, n_removed_channels), so changing any of
    them produces a fresh build — the user can never silently load a
    pipeline that was built with different parameters.

    Pass ``overwrite=True`` to rebuild + overwrite an existing cache.

    Extra kwargs (e.g. ``ls_params``, ``b_LED``, ``b_load_fd``) are
    forwarded to ``create_mea_pipeline`` but **not** part of the cache
    key. If you change one, pass ``overwrite=True``.
    """
    from ..classes.mea_pipeline import create_mea_pipeline as _build

    build_kwargs = dict(
        ss_version=ss_version, analysis_chunk_name=analysis_chunk_name,
        typing_file=typing_file,
        ei_corr_cutoff=ei_corr_cutoff,
        ei_match_method=ei_match_method,
        ei_use_isi=ei_use_isi,
        ei_use_timecourse=ei_use_timecourse,
        ei_n_removed_channels=ei_n_removed_channels,
    )
    path = pipeline_cache_path(
        exp_name, datafile_name, cache_root=cache_root, **build_kwargs)

    if not overwrite and path.exists():
        return load_cached_pipeline(
            exp_name, datafile_name,
            cache_root=cache_root, verbose=verbose, **build_kwargs)

    if verbose:
        print(f'Building pipeline (no cache hit at {path.name}) …')
    t0 = time.time()
    pipeline = _build(exp_name, datafile_name,
                       verbose=verbose,
                       **build_kwargs, **extra_kwargs)
    if verbose:
        print(f'  built in {time.time() - t0:.1f} s')

    path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.export_to_pkl(str(path))
    # Sidecar JSON makes the cache contents human-inspectable.
    sidecar = path.with_suffix('.json')
    norm = _normalize_kwargs(dict(exp_name=exp_name,
                                    datafile_name=datafile_name,
                                    **build_kwargs))
    sidecar.write_text(json.dumps(norm, indent=2, default=str))
    if verbose:
        sz = path.stat().st_size / 1e6
        print(f'Cached pipeline → {path}  ({sz:.1f} MB)')
    return pipeline
