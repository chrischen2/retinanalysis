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
    'match_diagnostics', 'plot_match_qc',
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
    protocol_subdir: Optional[str] = None,
) -> Path:
    """Build the cell-match table and write it to disk; return the CSV path.

    Parameters
    ----------
    protocol_subdir : str, optional
        Subdir name under ``<OUTPUT_DIR>/<exp>/``. Default (``None``) is
        ``protocol_short_name(protocol_name)``. Override to disambiguate
        when multiple datafiles of the same protocol live on one date.
    """
    from .cell_plot_archive import experiment_root, protocol_short_name

    short = protocol_subdir or protocol_short_name(pipeline.resp.protocol_name)
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


# ---------------------------------------------------------------------------
# Match QC: is the EI cluster match believable?
# ---------------------------------------------------------------------------

# Every way cluster_match can reject a candidate, in the order it tests them.
# Kept here so the plot legend stays stable even when a run happens not to
# produce one of the outcomes.
REJECT_REASONS = [
    'below_cutoff',
    'ambiguous_forward',
    'ambiguous_reverse',
    'claimed_by_other',
    'not_reciprocal',
    'isi_or_timecourse_gate',
]


def match_diagnostics(pipeline) -> pd.DataFrame:
    """One row per noise cell: the candidate it settled on, and the verdict.

    Re-runs the pipeline's own cluster match with diagnostics turned on, using
    the config cached on the pipeline, so the numbers describe the match the
    pipeline is actually carrying. Costs one more EI correlation pass — the
    matrices aren't kept after the pipeline is built.

    Columns: ``noise_id``, ``best_test_id``, ``best_corr``,
    ``runner_up_corr``, ``matched``, ``reason``.
    """
    from retinanalysis.utils.vision_utils import cluster_match

    config = dict(getattr(pipeline, 'ei_match_config', {}))
    _, _, df = cluster_match(pipeline.analysis_chunk, pipeline.resp,
                             verbose=False, return_diagnostics=True, **config)
    return df


def _electrode_amplitude(ei: np.ndarray) -> np.ndarray:
    """Peak |EI| per electrode — the vector the 'space' correlation compares."""
    return np.max(np.abs(np.asarray(ei, dtype=float)), axis=1)


def plot_match_qc(pipeline, df_diag: Optional[pd.DataFrame] = None,
                  n_examples: int = 3, bins: int = 40, dpi_hint: bool = False):
    """How well the noise chunk and the protocol datafile line up, cell by cell.

    Top panel: the distribution of each noise cell's best EI correlation
    against the protocol datafile, split into cells that matched and cells
    that didn't, with the acceptance cutoff drawn on. A healthy pairing is
    strongly bimodal — a bulk of real matches up near 1 and a separate bulk of
    cells with no counterpart down near 0. Mass piled up just under the cutoff
    means the threshold, not the data, is deciding.

    Below it, example pairs: each panel overlays the peak-|EI|-per-electrode
    profile of a noise cell and of the protocol cell it was compared against.
    Matched examples span the accepted correlation range; rejected examples are
    the near misses — the highest-correlation cells that still failed a gate —
    since those are the ones worth arguing about. The panel titles name the
    gate that rejected each.

    Note the panels draw raw EIs, while the correlation behind the number was
    computed on a denoised version with the largest electrode dropped, so a
    pair can look slightly more alike here than its coefficient suggests.

    Returns ``(fig, df_diag)``.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils.style import (NEUTRAL_GRAY, OKABE_ITO,
                                           apply_publication_style)

    apply_publication_style()

    if df_diag is None:
        df_diag = match_diagnostics(pipeline)
    if df_diag.empty:
        print('No diagnostics to plot — cluster matching was skipped '
              '(the protocol is part of the sorting chunk).')
        return None, df_diag

    matched_color, rejected_color = OKABE_ITO[5], OKABE_ITO[6]
    cutoff = float(getattr(pipeline, 'ei_match_config', {}).get('corr_cutoff', 0.8))

    matched = df_diag[df_diag['matched']]
    rejected = df_diag[~df_diag['matched']]

    fig = plt.figure(figsize=(10.5, 3.0 + 2.1 * n_examples))
    gs = fig.add_gridspec(1 + n_examples, 2, height_ratios=[1.5] + [1] * n_examples,
                          hspace=0.75, wspace=0.22)

    # ---- distribution -----------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    edges = np.linspace(0, 1, bins + 1)
    ax.hist([rejected['best_corr'].dropna(), matched['best_corr'].dropna()],
            bins=edges, stacked=True, color=[rejected_color, matched_color],
            label=[f'unmatched (n={len(rejected)})', f'matched (n={len(matched)})'])
    ax.axvline(cutoff, color=NEUTRAL_GRAY, linestyle='--', linewidth=1.2)
    ax.text(cutoff, ax.get_ylim()[1], f' cutoff {cutoff:g}', color=NEUTRAL_GRAY,
            va='top', ha='left', fontsize=8)
    ax.set_xlabel("Best EI correlation to the protocol datafile, per noise cell")
    ax.set_ylabel('Cells')
    ax.set_xlim(0, 1)
    ax.legend(loc='upper left')
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)

    n_ref = len(df_diag)
    counts = rejected['reason'].value_counts()
    subtitle = ', '.join(f'{r} {counts[r]}' for r in REJECT_REASONS if r in counts)
    ax.set_title(f'{pipeline.analysis_chunk.exp_name} '
                 f'{pipeline.analysis_chunk.chunk_name} → '
                 f'{getattr(pipeline.resp, "datafile_name", "protocol")}  •  '
                 f'{len(matched)}/{n_ref} matched\nrejected: {subtitle or "none"}',
                 fontsize=9)

    # ---- examples ---------------------------------------------------------
    # Matched: spread over the accepted range, so the panels show a strong
    # match and a marginal one rather than three near-identical good ones.
    m_sorted = matched.sort_values('best_corr', ascending=False)
    if len(m_sorted):
        picks = np.unique(np.linspace(0, len(m_sorted) - 1, n_examples).astype(int))
        m_examples = m_sorted.iloc[picks]
    else:
        m_examples = m_sorted

    # Rejected: one example per failure mode, commonest mode first, each the
    # highest-correlation cell that mode threw out. Picking purely by
    # correlation would show the same gate three times and hide the one doing
    # most of the rejecting.
    by_reason = []
    for reason in rejected['reason'].value_counts().index:
        worst = rejected[rejected['reason'] == reason].sort_values(
            'best_corr', ascending=False)
        by_reason.append(worst.iloc[0])
    r_examples = (pd.DataFrame(by_reason).head(n_examples) if by_reason
                  else rejected.head(0))

    for row in range(n_examples):
        for col, (examples, color, kind) in enumerate(
                ((m_examples, matched_color, 'matched'),
                 (r_examples, rejected_color, 'rejected'))):
            ax = fig.add_subplot(gs[row + 1, col])
            if row >= len(examples):
                ax.set_axis_off()
                continue

            rec = examples.iloc[row]
            noise_id, test_id = int(rec['noise_id']), rec['best_test_id']
            ref_ei = pipeline.analysis_chunk.d_EIs.get(noise_id)
            test_ei = (None if pd.isna(test_id)
                       else pipeline.resp.d_EIs.get(int(test_id)))

            if ref_ei is None or test_ei is None:
                ax.text(0.5, 0.5, '(EI unavailable)', transform=ax.transAxes,
                        ha='center', va='center', fontsize=8)
                ax.set_axis_off()
                continue

            # Noise cell drawn wide underneath, protocol cell thin on top: for
            # a good match the two are nearly identical, and equal weights
            # would just hide one under the other.
            ref_amp, test_amp = _electrode_amplitude(ref_ei), _electrode_amplitude(test_ei)
            ax.plot(ref_amp, color=NEUTRAL_GRAY, linewidth=2.4, alpha=0.5,
                    solid_capstyle='round', label=f'noise {noise_id}')
            ax.plot(test_amp, color=color, linewidth=0.9,
                    label=f'protocol {int(test_id)}')
            ax.set_xlim(0, max(len(ref_amp), len(test_amp)))
            ax.set_xlabel('Electrode')
            if col == 0:
                ax.set_ylabel('Peak |EI|')
            ax.legend(fontsize=7, loc='upper right', frameon=False)

            note = ('' if kind == 'matched' else f'  •  {rec["reason"]}')
            ax.set_title(f'r = {rec["best_corr"]:.3f}  '
                         f'(runner-up {rec["runner_up_corr"]:.3f}){note}',
                         fontsize=8, loc='left', color=color)

    return fig, df_diag
