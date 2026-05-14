"""Offline per-experiment data store for downstream analysis.

Rationale
---------
Building the pipeline + running protocol QC for a single date takes
~30–60s and depends on the SSD being mounted, the DataJoint database
being up, and the typing file existing on disk. Once a date has been
through QC + visual review, the *interesting* quantities for downstream
analysis are small (spike times + condition table + STA/EI summaries).
This module serializes that subset to a single HDF5 file so future
sessions can re-load in <1s and operate without the database.

Layout
------
``<OUTPUT_DIR>/<exp_name>/<protocol_short>/offline.h5``

Top-level groups:

- ``meta/``       — attrs: ``exp_name, datafile_name, protocol_name,
                    protocol_short, ss_version, chunk_name, typing_file,
                    n_cells, ndf, created, source_version``.
- ``timing/``     — attrs: ``preTime_ms, stimTime_ms, tailTime_ms,
                    sample_rate_hz``.
- ``epochs``      — dataset (compound) with one row per epoch; columns
                    are the union of ``condition_keys`` plus
                    ``epoch_index``.
- ``epoch_block_params/`` — group of scalar/array attrs mirroring
                    ``stim_block.d_epoch_block_params``.
- ``cells/<cell_id>/`` — one group per cell that passes the saved-cells
                    filter. Attrs: ``cell_id, cell_type, noise_cell_id,
                    ei_corr`` + every column from ``cell_match.csv`` +
                    STA params (``center_x, center_y, std_x, std_y, rot,
                    pixels_per_stixel, microns_per_stixel``).
                    Sub-datasets:
                      * ``spike_times/epoch_<i>`` — 1-D ms array
                      * ``psth`` — ``(n_epochs, n_bins)`` Hz
                      * ``psth_time_ms`` — ``(n_bins,)``

Why HDF5 and not pickle / npz: cross-language readable, hierarchical
access (load one cell without paying for the rest), and ragged spike
times fit naturally as per-epoch sub-datasets.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd

from ..config.settings import OUTPUT_DIR
from .psth import epoch_spikes_to_psth, psth_time_axis
from .cell_plot_archive import experiment_root, protocol_short_name


__all__ = [
    'offline_h5_path',
    'save_offline_data',
    'load_offline_data',
    'load_or_build_offline',
    'load_offline_many',
    'OfflineDataset',
]


SOURCE_VERSION = 1  # bump when schema changes


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def offline_h5_path(exp_name: str, protocol: str,
                    output_root: Optional[str] = None) -> Path:
    """Return ``<OUTPUT_DIR>/<exp>/<protocol>/offline.h5``."""
    root = Path(output_root) if output_root else Path(OUTPUT_DIR)
    return root / exp_name / protocol / 'offline.h5'


# ---------------------------------------------------------------------------
# OfflineDataset wrapper
# ---------------------------------------------------------------------------

class OfflineDataset:
    """Lightweight, lazy view of an offline HDF5 file.

    Loads small things eagerly (meta, timing, epochs, cell-level attrs)
    and pulls spike times / PSTHs on demand via :meth:`spike_times` /
    :meth:`psth_matrix`.
    """

    def __init__(self, h5_path: Union[str, Path]):
        self.path = Path(h5_path)
        if not self.path.exists():
            raise FileNotFoundError(f'No offline file at {self.path}')
        with h5py.File(self.path, 'r') as f:
            self.meta = {k: _decode(v) for k, v in f['meta'].attrs.items()}
            self.timing = {k: _decode(v) for k, v in f['timing'].attrs.items()}

            # Epoch table
            epochs_ds = f['epochs']
            self.epochs = pd.DataFrame({
                name: epochs_ds[name][:] for name in epochs_ds.dtype.names
            })
            for col in self.epochs.columns:
                if self.epochs[col].dtype.kind in {'S', 'O'}:
                    self.epochs[col] = self.epochs[col].apply(_decode)

            # Epoch block params (full original protocol params)
            ebp_grp = f.get('epoch_block_params')
            if ebp_grp is None:
                self.epoch_block_params = {}
            else:
                self.epoch_block_params = {
                    k: _decode(v) for k, v in ebp_grp.attrs.items()
                }

            # Per-cell attrs (eager) — assemble cells DataFrame
            cells_grp = f['cells']
            cell_rows = []
            self._cell_ids = []
            for cid_str in sorted(cells_grp.keys(), key=lambda s: int(s)):
                cid = int(cid_str)
                self._cell_ids.append(cid)
                row = {k: _decode(v) for k, v in cells_grp[cid_str].attrs.items()}
                row['cell_id'] = cid
                cell_rows.append(row)
            self.cells = (pd.DataFrame(cell_rows)
                          if cell_rows else pd.DataFrame(columns=['cell_id']))
            if not self.cells.empty:
                self.cells = self.cells.set_index('cell_id', drop=False)

    # ---------- convenience accessors ----------

    @property
    def exp_name(self) -> str:
        return str(self.meta.get('exp_name', ''))

    @property
    def protocol_short(self) -> str:
        return str(self.meta.get('protocol_short', ''))

    @property
    def cell_ids(self) -> List[int]:
        return list(self._cell_ids)

    def cell_types(self) -> List[str]:
        if 'cell_type' not in self.cells.columns:
            return []
        return sorted(self.cells['cell_type'].dropna().unique().tolist())

    def spike_times(self, cell_id: int) -> List[np.ndarray]:
        """Return list of 1-D spike-time arrays (ms), one per epoch."""
        with h5py.File(self.path, 'r') as f:
            g = f[f'cells/{int(cell_id)}/spike_times']
            n_epochs = len(g)
            out = [None] * n_epochs
            for k in g:
                idx = int(k.split('_')[-1])
                out[idx] = g[k][:]
        return out

    def psth_matrix(self, cell_id: int) -> np.ndarray:
        """Return ``(n_epochs, n_bins)`` PSTH in Hz."""
        with h5py.File(self.path, 'r') as f:
            return f[f'cells/{int(cell_id)}/psth'][:]

    def psth_time_ms(self, cell_id: Optional[int] = None) -> np.ndarray:
        """Return the time axis. Single ``psth_time_ms`` dataset shared across cells."""
        with h5py.File(self.path, 'r') as f:
            # All cells share the same axis — read from the first one.
            cid = self._cell_ids[0] if cell_id is None else int(cell_id)
            return f[f'cells/{cid}/psth_time_ms'][:]

    def __repr__(self) -> str:
        return (f"OfflineDataset(exp={self.exp_name}, "
                f"protocol={self.protocol_short}, "
                f"n_cells={len(self._cell_ids)}, "
                f"n_epochs={len(self.epochs)})")


def _decode(v):
    """h5py returns bytes for strings; decode to str. Also unwrap 0-d arrays."""
    if isinstance(v, bytes):
        return v.decode('utf-8')
    if isinstance(v, np.ndarray):
        if v.shape == ():
            return _decode(v.item())
        if v.dtype.kind == 'S':
            return np.array([_decode(x) for x in v.tolist()])
    return v


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _stim_param_value(v):
    """Coerce an epoch_parameters value to something HDF5 can store."""
    if v is None:
        return ''
    if isinstance(v, (str, bool, int, float)):
        return v
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, (list, tuple)):
        try:
            return np.asarray(v)
        except Exception:
            return str(v)
    return str(v)


def _good_cell_ids(
    response_block,
    qc: Optional[pd.DataFrame],
    visual_qc_df: Optional[pd.DataFrame],
    cell_ids_override: Optional[Sequence[int]],
) -> List[int]:
    """Resolve which cells to save (QC ∩ visual-QC-good ∩ override)."""
    all_ids = response_block.df_spike_times['cell_id'].astype(int).tolist()
    if cell_ids_override is not None:
        return [int(c) for c in cell_ids_override if int(c) in all_ids]

    passing = set(all_ids)
    if qc is not None and 'passes' in qc.columns:
        passing &= set(qc.loc[qc['passes'], 'cell_id'].astype(int))
    if visual_qc_df is not None and not visual_qc_df.empty:
        good = set(visual_qc_df.loc[visual_qc_df['tag'] == 'good',
                                    'cell_id'].astype(int))
        passing &= good
    return sorted(passing)


def _build_epochs_dataframe(stim_block,
                            condition_keys: Sequence[str]) -> pd.DataFrame:
    """One row per epoch with ``epoch_index`` and every requested condition key."""
    df = stim_block.df_epochs.copy()
    out = pd.DataFrame({'epoch_index': df['epoch_index'].astype(int).values})
    for k in condition_keys:
        if k in df.columns:
            vals = df[k].tolist()
        else:
            vals = [p.get(k) for p in df['epoch_parameters']]
        # Coerce Nones → 'NA' string so HDF5 can store. Preserve numerics.
        out[k] = [_stim_param_value(v) for v in vals]
    return out


def _epochs_to_compound_array(df: pd.DataFrame) -> np.ndarray:
    """Build a structured array matching ``df``'s columns. Strings → S-dtype."""
    fields = []
    cols = {}
    for col in df.columns:
        series = df[col]
        if series.dtype.kind in {'i', 'u'}:
            arr = series.to_numpy(dtype=np.int64)
            fields.append((col, 'i8'))
        elif series.dtype.kind == 'f':
            arr = series.to_numpy(dtype=np.float64)
            fields.append((col, 'f8'))
        else:
            # Mixed / object — try numeric coerce, else stringify
            try:
                arr = series.astype(np.float64).to_numpy()
                fields.append((col, 'f8'))
            except (ValueError, TypeError):
                arr = np.asarray([str(v).encode('utf-8') for v in series])
                fields.append((col, h5py.string_dtype('utf-8')))
        cols[col] = arr
    rec = np.zeros(len(df), dtype=np.dtype(fields))
    for col, arr in cols.items():
        rec[col] = arr
    return rec


def _cell_match_lookup(cell_match_df: Optional[pd.DataFrame]) -> Dict[int, Dict]:
    """Index ``cell_match.csv`` by ``protocol_cell_id`` for per-cell attrs."""
    if cell_match_df is None or cell_match_df.empty:
        return {}
    out: Dict[int, Dict] = {}
    for _, row in cell_match_df.iterrows():
        try:
            cid = int(row['protocol_cell_id'])
        except (KeyError, ValueError, TypeError):
            continue
        out[cid] = {k: row[k] for k in row.index
                    if k not in {'protocol_cell_id', 'exp_name',
                                 'datafile_name', 'chunk_name'}
                    and not pd.isna(row[k])}
    return out


def _rf_params_for_match(
    analysis_chunk,
    match_dict: Dict[int, int],
) -> Dict[int, Dict]:
    """Return ``{protocol_cell_id: rf_params dict}`` keyed by *protocol* id.

    ``analysis_chunk.rf_params`` is keyed by the *noise* cell id; we
    invert ``match_dict = {noise_id: proto_id}`` to attach the right RF
    fit to each protocol-side cell.
    """
    rf = getattr(analysis_chunk, 'rf_params', {}) or {}
    out = {}
    for noise_id, proto_id in match_dict.items():
        p = rf.get(int(noise_id))
        if p is None:
            continue
        out[int(proto_id)] = dict(p)
    return out


def save_offline_data(
    pipeline,
    *,
    protocol_short: Optional[str] = None,
    qc: Optional[pd.DataFrame] = None,
    visual_qc_df: Optional[pd.DataFrame] = None,
    cell_ids: Optional[Sequence[int]] = None,
    cell_match_df: Optional[pd.DataFrame] = None,
    condition_keys: Optional[Sequence[str]] = None,
    psth_sigma_ms: float = 10.0,
    sample_rate_hz: float = 1000.0,
    output_root: Optional[str] = None,
    overwrite: bool = True,
    verbose: bool = True,
) -> Path:
    """Persist a per-experiment offline HDF5 for downstream analysis.

    Saves only cells that pass automated QC ∩ visual-QC ``'good'``
    (or the explicit ``cell_ids`` override).

    Parameters
    ----------
    pipeline : MEAPipeline
        Built pipeline (with ``analysis_chunk``, ``resp``, ``stim``,
        ``match_dict``, ``corr_dict``).
    protocol_short : str, optional
        Subdir name (default: derived from ``pipeline.resp.protocol_name``).
    qc, visual_qc_df : DataFrame, optional
        ``filter_cells_by_qc`` output and the raw ``visual_qc.csv`` —
        used to filter ``cells/`` to QC-pass ∩ ``good`` set.
    cell_ids : sequence[int], optional
        Override the QC filter and save exactly these cells.
    cell_match_df : DataFrame, optional
        Output of ``build_cell_match_table`` — folded into per-cell attrs.
        If ``None``, built from ``pipeline`` on the fly.
    condition_keys : sequence[str], optional
        Auto-detected for known protocols when ``None`` (e.g.
        EyeMovement → ``[currentImageName, currentBackgroundScale]``).
    psth_sigma_ms, sample_rate_hz : float
        Pre-compute and store the smoothed PSTH so reload is plot-ready.
    overwrite : bool
        Re-create the file even if it exists.

    Returns
    -------
    Path
        Path to the written ``offline.h5``.
    """
    rb = pipeline.resp
    sb = pipeline.stim
    ac = pipeline.analysis_chunk

    short = protocol_short or protocol_short_name(rb.protocol_name)
    out_path = offline_h5_path(rb.exp_name, short, output_root=output_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        if verbose:
            print(f'  offline → exists, skipping ({out_path})')
        return out_path

    # --- Resolve condition keys
    if condition_keys is None:
        from .cell_plot_archive import _PROTOCOL_DEFAULT_CONDITION_KEYS
        condition_keys = _PROTOCOL_DEFAULT_CONDITION_KEYS.get(
            rb.protocol_name, ['currentBackgroundScale'])
    condition_keys = list(condition_keys)

    # --- Resolve cell list
    good_ids = _good_cell_ids(rb, qc, visual_qc_df, cell_ids)
    if verbose:
        print(f'  offline → {len(good_ids)} cells')

    # --- Build the epoch table
    epochs_df = _build_epochs_dataframe(sb, condition_keys)

    # --- Cell-match lookup (per-cell EI/match stats)
    if cell_match_df is None:
        try:
            from .cell_match import build_cell_match_table
            cell_match_df = build_cell_match_table(pipeline, qc_pass_only=qc)
        except Exception:
            cell_match_df = None
    match_lookup = _cell_match_lookup(cell_match_df)

    # --- RF (STA) params, keyed by protocol cell id
    rf_lookup = _rf_params_for_match(ac, pipeline.match_dict)

    # --- Timing
    bp = sb.d_epoch_block_params or {}
    pre_ms = float(bp.get('preTime', 0))
    stim_ms = float(bp.get('stimTime', 0))
    tail_ms = float(bp.get('tailTime', 0))
    t_end_ms = pre_ms + stim_ms + tail_ms

    # --- Write
    with h5py.File(out_path, 'w') as f:
        meta = f.create_group('meta')
        meta.attrs['exp_name'] = rb.exp_name
        meta.attrs['datafile_name'] = rb.datafile_name
        meta.attrs['protocol_name'] = rb.protocol_name
        meta.attrs['protocol_short'] = short
        meta.attrs['ss_version'] = getattr(rb, 'ss_version', '') or ''
        meta.attrs['chunk_name'] = ac.chunk_name
        meta.attrs['typing_file'] = getattr(ac, 'typing_file', '') or ''
        meta.attrs['n_cells'] = len(good_ids)
        ndf = None
        try:
            ndf = sb.df_epochs['epoch_parameters'].iloc[0].get('NDF')
        except Exception:
            pass
        meta.attrs['ndf'] = float(ndf) if ndf is not None else float('nan')
        meta.attrs['created'] = _dt.datetime.now().isoformat(timespec='seconds')
        meta.attrs['source_version'] = SOURCE_VERSION
        meta.attrs['condition_keys'] = np.array(
            [k.encode('utf-8') for k in condition_keys])

        timing = f.create_group('timing')
        timing.attrs['preTime_ms'] = pre_ms
        timing.attrs['stimTime_ms'] = stim_ms
        timing.attrs['tailTime_ms'] = tail_ms
        timing.attrs['sample_rate_hz'] = float(sample_rate_hz)
        timing.attrs['psth_sigma_ms'] = float(psth_sigma_ms)

        # Epoch table
        rec = _epochs_to_compound_array(epochs_df)
        f.create_dataset('epochs', data=rec)

        # Epoch block params (scalar primitives only — everything else stringified)
        ebp = f.create_group('epoch_block_params')
        for k, v in bp.items():
            try:
                if isinstance(v, (str, int, float, bool)):
                    ebp.attrs[k] = v
                elif isinstance(v, np.ndarray) and v.dtype.kind in {'i', 'u', 'f', 'b'}:
                    ebp.attrs[k] = v
                else:
                    ebp.attrs[k] = json.dumps(_jsonify(v))
            except Exception:
                continue

        # Per-cell groups
        cells_grp = f.create_group('cells')
        time_axis = psth_time_axis(t_end_ms, sample_rate_hz, 0.0)

        df_cells = rb.df_spike_times.set_index('cell_id', drop=False)
        for cid in good_ids:
            if cid not in df_cells.index:
                continue
            row = df_cells.loc[cid]
            sts_list = row['spike_times']  # list of arrays (ms)
            cell_type = row.get('cell_type', None)
            noise_id = row.get('noise_id', None)

            g = cells_grp.create_group(str(int(cid)))
            g.attrs['cell_id'] = int(cid)
            g.attrs['cell_type'] = (str(cell_type)
                                    if cell_type is not None and not (
                                        isinstance(cell_type, float) and np.isnan(cell_type)
                                    ) else '')
            if noise_id is not None and not (
                    isinstance(noise_id, float) and np.isnan(noise_id)):
                g.attrs['noise_cell_id'] = int(noise_id)

            # cell_match EI stats
            for k, v in match_lookup.get(int(cid), {}).items():
                try:
                    if isinstance(v, (str, int, float, bool, np.integer, np.floating)):
                        g.attrs[k] = v
                except Exception:
                    continue

            # STA / RF fit
            rf = rf_lookup.get(int(cid))
            if rf is not None:
                for k in ('center_x', 'center_y', 'std_x', 'std_y', 'rot'):
                    if k in rf and rf[k] is not None:
                        try:
                            g.attrs[f'sta_{k}'] = float(rf[k])
                        except Exception:
                            pass
            try:
                g.attrs['pixels_per_stixel'] = float(ac.pixels_per_stixel)
                g.attrs['microns_per_stixel'] = float(ac.microns_per_stixel)
            except Exception:
                pass

            # Spike times (one dataset per epoch — ragged-array friendly)
            st_grp = g.create_group('spike_times')
            for i, sts in enumerate(sts_list):
                arr = np.asarray(sts, dtype=np.float64)
                ds = st_grp.create_dataset(f'epoch_{i:03d}', data=arr,
                                           compression='gzip',
                                           compression_opts=4)
                ds.attrs['epoch_index'] = int(i)

            # PSTH (n_epochs, n_bins) — Gaussian-smoothed Hz
            psth = epoch_spikes_to_psth(
                sts_list, t_end_ms,
                psth_sigma_ms=psth_sigma_ms,
                sample_rate_hz=sample_rate_hz,
            )
            g.create_dataset('psth', data=psth, compression='gzip',
                             compression_opts=4)
            g.create_dataset('psth_time_ms', data=time_axis)

    if verbose:
        print(f'  offline → {out_path}')
    return out_path


def _jsonify(v):
    """Best-effort conversion of nested params to JSON-able types."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonify(x) for k, x in v.items()}
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# ---------------------------------------------------------------------------
# Load + load-or-build
# ---------------------------------------------------------------------------

def load_offline_data(
    exp_name: str,
    protocol: str = 'eye_movement_alt_bg',
    output_root: Optional[str] = None,
) -> OfflineDataset:
    """Load an existing offline HDF5; raise ``FileNotFoundError`` otherwise."""
    return OfflineDataset(offline_h5_path(exp_name, protocol, output_root))


def load_or_build_offline(
    exp_name: str,
    *,
    protocol: str = 'eye_movement_alt_bg',
    protocol_search: Optional[str] = None,
    output_root: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = True,
    **build_kwargs,
) -> OfflineDataset:
    """Load the offline file, or build + save + load it if missing.

    Parameters
    ----------
    exp_name : str
        Experiment date (e.g. ``'20221123C'``).
    protocol : str
        Protocol short-name (subdir). Default ``'eye_movement_alt_bg'``.
    protocol_search : str, optional
        Forwarded to :func:`analyze_experiment` when we have to build —
        e.g. ``'AlternatingBackground'``.
    overwrite : bool
        Force a rebuild even if the file exists.
    **build_kwargs
        Forwarded to :func:`analyze_experiment` (e.g. ``datafile_name``,
        ``ss_version``, ``cell_types``).

    Returns
    -------
    OfflineDataset
    """
    h5_path = offline_h5_path(exp_name, protocol, output_root=output_root)
    if h5_path.exists() and not overwrite:
        if verbose:
            print(f'[load_or_build_offline] {exp_name}: loading {h5_path}')
        return OfflineDataset(h5_path)

    if verbose:
        print(f'[load_or_build_offline] {exp_name}: building (file missing)')

    # Defer imports — pulls DataJoint
    from ..classes.mea_pipeline import create_mea_pipeline
    from ..classes.stim import MEAStimBlock
    from ..analyze import _detect_ss_version, _pick_typing_file, _resolve_datafile
    from .protocol_qc import (block_qc_metrics, filter_cells_by_qc,
                               QCThresholds, save_protocol_qc)
    from .visual_qc import visual_qc_csv_path

    datafile_name = build_kwargs.pop('datafile_name', None)
    datafile_name = _resolve_datafile(exp_name, protocol_search, datafile_name)
    ss_version = build_kwargs.pop('ss_version', None) or _detect_ss_version(
        exp_name, datafile_name)
    typing_file = build_kwargs.pop('typing_file', None)
    if typing_file is None:
        tmp = MEAStimBlock(exp_name, datafile_name, verbose=False)
        typing_file = _pick_typing_file(exp_name, tmp.nearest_noise_chunk, ss_version)

    pipeline = create_mea_pipeline(
        exp_name, datafile_name,
        ss_version=ss_version, typing_file=typing_file,
        verbose=False,
    )

    # QC
    timing = pipeline.resp.d_timing or {}
    t_total_ms = (float(timing.get('pre_time_ms', 0))
                  + float(timing.get('stim_time_ms', 0))
                  + float(timing.get('tail_time_ms', 0)))
    qc = filter_cells_by_qc(
        block_qc_metrics(pipeline.resp, t_start_ms=0.0, t_end_ms=t_total_ms,
                          sample_rate_hz=1000.0),
        thresholds=build_kwargs.get('qc_thresholds'),
    )

    # Visual QC (read-only)
    vqc_path = visual_qc_csv_path(exp_name, protocol, output_root=output_root)
    visual_qc_df = pd.read_csv(vqc_path) if vqc_path.exists() else None

    save_offline_data(
        pipeline,
        protocol_short=protocol,
        qc=qc,
        visual_qc_df=visual_qc_df,
        output_root=output_root,
        overwrite=True,
        verbose=verbose,
        **{k: v for k, v in build_kwargs.items()
           if k in {'condition_keys', 'psth_sigma_ms', 'sample_rate_hz',
                    'cell_ids', 'cell_match_df'}},
    )
    return OfflineDataset(h5_path)


# ---------------------------------------------------------------------------
# Cross-date loading
# ---------------------------------------------------------------------------

def load_offline_many(
    exp_names: Optional[Iterable[str]] = None,
    *,
    protocol: str = 'eye_movement_alt_bg',
    output_root: Optional[str] = None,
    on_missing: str = 'skip',
) -> Dict[str, OfflineDataset]:
    """Load all available offline files into ``{exp_name: OfflineDataset}``.

    Parameters
    ----------
    exp_names : iterable[str], optional
        Restrict to these dates. Default: every subdir of ``output_root``.
    on_missing : ``'skip'`` (default) or ``'raise'``
        Behavior when ``exp_names`` is given but a file is missing.
    """
    root = Path(output_root) if output_root else Path(OUTPUT_DIR)
    if exp_names is None:
        if not root.is_dir():
            return {}
        exp_names = [p.name for p in sorted(root.iterdir()) if p.is_dir()]
    out = {}
    for exp in exp_names:
        h5_path = offline_h5_path(exp, protocol, output_root=output_root)
        if not h5_path.exists():
            if on_missing == 'raise':
                raise FileNotFoundError(f'No offline file for {exp}: {h5_path}')
            continue
        out[exp] = OfflineDataset(h5_path)
    return out
