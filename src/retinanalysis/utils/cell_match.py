"""Persistent noise↔protocol cell matching + per-cell EI stats.

The MEA pipeline performs an EI-based cluster match between the noise
chunk's cells and the protocol datafile's cells. The mapping
(``match_dict``) and the per-cell EI correlation (``corr_dict``) only
live in memory; they are recomputed every time the pipeline is built.
That recomputation is expensive (loads every cell's EI) and not
deterministic-free (e.g. typing-file changes shift cell-type labels).

This module persists the mapping plus a small batch of useful EI
summary stats to disk so downstream / future sessions can:

  * load the table instead of rebuilding the pipeline,
  * filter cells by EI quality (amplitude, footprint, SNR), and
  * join the noise-chunk cell IDs back into protocol-side analyses.

The CSV lives next to ``index.csv`` at
``<OUTPUT_DIR>/<exp_name>/<protocol_short>/cell_match.csv``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config.settings import OUTPUT_DIR


__all__ = [
    'compute_ei_stats', 'build_cell_match_table', 'save_cell_match',
    'load_cell_match', 'cell_match_csv_path',
]


CELL_MATCH_COLUMNS = [
    'exp_name', 'datafile_name', 'chunk_name',
    'protocol_cell_id', 'noise_cell_id', 'cell_type',
    'ei_corr',
    'ei_peak_amp', 'ei_peak_electrode',
    'ei_peak_x_um', 'ei_peak_y_um', 'ei_peak_sample',
    'ei_n_active_electrodes', 'ei_spread_um', 'ei_snr',
]


def cell_match_csv_path(exp_name: str, protocol: str,
                        output_root: Optional[str] = None) -> Path:
    """Return ``<OUTPUT_DIR>/<exp>/<protocol>/cell_match.csv``."""
    root = Path(output_root) if output_root else Path(OUTPUT_DIR)
    return root / exp_name / protocol / 'cell_match.csv'


def compute_ei_stats(ei: np.ndarray, electrode_xy_um: np.ndarray,
                     active_thresh_frac: float = 0.2) -> dict:
    """Summary stats for one cell's EI.

    Parameters
    ----------
    ei : ndarray of shape (n_electrodes, n_samples)
        Electrical image for one cell.
    electrode_xy_um : ndarray of shape (n_electrodes, 2)
        Electrode positions in MEA-chip microns.
    active_thresh_frac : float
        An electrode counts as "active" when its peak |EI| exceeds this
        fraction of the global peak. Default ``0.2`` (i.e. 20%).

    Returns
    -------
    dict
        Keys: ``ei_peak_amp, ei_peak_electrode, ei_peak_x_um,
        ei_peak_y_um, ei_peak_sample, ei_n_active_electrodes,
        ei_spread_um, ei_snr``.

        - ``ei_peak_amp``: max |EI| anywhere in the array.
        - ``ei_peak_electrode``: index of the electrode with the peak
          (a soma proxy).
        - ``ei_peak_x_um``, ``ei_peak_y_um``: chip-µm coords of that
          electrode.
        - ``ei_peak_sample``: sample index of the peak on the peak
          electrode (rough spike-timing offset within the EI window).
        - ``ei_n_active_electrodes``: number of electrodes whose peak
          |EI| exceeds ``active_thresh_frac × global peak``. This is
          the cell's footprint size.
        - ``ei_spread_um``: RMS distance of active-electrode positions
          from the peak electrode — the footprint's spatial extent.
        - ``ei_snr``: peak amp / median of inactive-electrode peaks
          — a coarse signal-to-noise estimate.
    """
    abs_ei = np.abs(ei)
    elec_peak = abs_ei.max(axis=1)
    peak_elec = int(np.argmax(elec_peak))
    peak_amp = float(elec_peak[peak_elec])
    peak_sample = int(np.argmax(abs_ei[peak_elec, :]))

    thresh = active_thresh_frac * peak_amp
    active = elec_peak >= thresh
    n_active = int(active.sum())

    if n_active > 1:
        d = electrode_xy_um[active] - electrode_xy_um[peak_elec]
        spread = float(np.sqrt(np.mean((d ** 2).sum(axis=1))))
    else:
        spread = 0.0

    inactive = ~active
    if inactive.sum() > 0:
        noise = float(np.median(elec_peak[inactive]) + 1e-12)
        snr = peak_amp / noise
    else:
        snr = float('inf')

    return {
        'ei_peak_amp': peak_amp,
        'ei_peak_electrode': peak_elec,
        'ei_peak_x_um': float(electrode_xy_um[peak_elec, 0]),
        'ei_peak_y_um': float(electrode_xy_um[peak_elec, 1]),
        'ei_peak_sample': peak_sample,
        'ei_n_active_electrodes': n_active,
        'ei_spread_um': spread,
        'ei_snr': snr,
    }


def build_cell_match_table(
    pipeline,
    cell_types: Optional[List[str]] = None,
    qc_pass_only: Optional[pd.DataFrame] = None,
    active_thresh_frac: float = 0.2,
) -> pd.DataFrame:
    """Build the (one-row-per-matched-cell) match+EI-stats table.

    Reads from the live pipeline:
      * ``pipeline.match_dict`` — noise_id → protocol_id
      * ``pipeline.corr_dict``  — noise_id → EI correlation
      * ``pipeline.analysis_chunk.d_EIs`` — per-cell EIs
      * ``pipeline.analysis_chunk.vcd.get_electrode_map()`` — electrodes (µm)
      * ``pipeline.resp.df_spike_times`` — cell-type per protocol cell

    Parameters
    ----------
    cell_types : list[str], optional
        Restrict to these types. Default: include every matched cell
        regardless of type.
    qc_pass_only : pandas.DataFrame, optional
        DataFrame from ``filter_cells_by_qc`` — restrict to rows where
        ``passes`` is True.
    active_thresh_frac : float
        Forwarded to :func:`compute_ei_stats`.

    Returns
    -------
    pandas.DataFrame
        Columns listed in :data:`CELL_MATCH_COLUMNS`. Empty (but with
        the right columns) when the pipeline has zero matches.
    """
    ac = pipeline.analysis_chunk
    rb = pipeline.resp
    match_dict: Dict[int, int] = pipeline.match_dict
    corr_dict: Dict[int, float] = getattr(pipeline, 'corr_dict', {}) or {}

    electrode_xy = ac.vcd.get_electrode_map()

    proto_types = (rb.df_spike_times
                     .drop_duplicates('cell_id')
                     .set_index('cell_id')['cell_type']
                     .to_dict()
                   if 'cell_type' in rb.df_spike_times.columns else {})

    passing = None
    if qc_pass_only is not None:
        passing = set(qc_pass_only.loc[qc_pass_only['passes'], 'cell_id']
                      .astype(int))

    rows = []
    for noise_id, proto_id in match_dict.items():
        ct = proto_types.get(int(proto_id))
        if cell_types is not None and ct not in cell_types:
            continue
        if passing is not None and int(proto_id) not in passing:
            continue
        ei = ac.d_EIs.get(int(noise_id))
        if ei is None:
            continue
        stats = compute_ei_stats(ei, electrode_xy, active_thresh_frac)
        rows.append({
            'exp_name': ac.exp_name,
            'datafile_name': rb.datafile_name,
            'chunk_name': ac.chunk_name,
            'protocol_cell_id': int(proto_id),
            'noise_cell_id': int(noise_id),
            'cell_type': ct,
            'ei_corr': float(corr_dict.get(int(noise_id), np.nan)),
            **stats,
        })

    if not rows:
        return pd.DataFrame(columns=CELL_MATCH_COLUMNS)
    df = pd.DataFrame(rows)[CELL_MATCH_COLUMNS]
    return df.sort_values(['cell_type', 'protocol_cell_id'],
                          na_position='last').reset_index(drop=True)


def save_cell_match(
    pipeline,
    output_root: Optional[str] = None,
    cell_types: Optional[List[str]] = None,
    qc_pass_only: Optional[pd.DataFrame] = None,
    active_thresh_frac: float = 0.2,
) -> Path:
    """Build the cell-match table and write it to disk; return the CSV path."""
    from .cell_plot_archive import experiment_root, protocol_short_name

    short = protocol_short_name(pipeline.resp.protocol_name)
    exp_root = Path(experiment_root(pipeline.resp.exp_name, output_root=output_root))
    csv_path = exp_root / short / 'cell_match.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_cell_match_table(
        pipeline, cell_types=cell_types, qc_pass_only=qc_pass_only,
        active_thresh_frac=active_thresh_frac,
    )
    df.to_csv(csv_path, index=False)
    return csv_path


def load_cell_match(
    exp_names: Optional[List[str]] = None,
    output_root: Optional[str] = None,
    protocol: str = 'eye_movement_alt_bg',
) -> pd.DataFrame:
    """Concat ``cell_match.csv`` across experiments.

    Parameters
    ----------
    exp_names : list[str], optional
        Restrict to these dates. Default: every subdir of ``output_root``.
    output_root : str, optional
        Override ``OUTPUT_DIR``.
    protocol : str
        Subdir name. Default ``'eye_movement_alt_bg'``.

    Returns
    -------
    pandas.DataFrame
        Concatenated table. Empty DataFrame with :data:`CELL_MATCH_COLUMNS`
        when nothing is on disk.
    """
    root = Path(output_root) if output_root else Path(OUTPUT_DIR)
    if exp_names is None:
        if not root.is_dir():
            return pd.DataFrame(columns=CELL_MATCH_COLUMNS)
        exp_names = [p.name for p in sorted(root.iterdir()) if p.is_dir()]

    dfs = []
    for exp in exp_names:
        csv_path = root / exp / protocol / 'cell_match.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if 'exp_name' not in df.columns or df['exp_name'].isna().any():
            df['exp_name'] = exp
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=CELL_MATCH_COLUMNS)
    out = pd.concat(dfs, ignore_index=True)
    for col in CELL_MATCH_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[CELL_MATCH_COLUMNS]
