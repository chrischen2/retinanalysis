"""Save a per-cell figure archive (mosaic + raster + PSTH) for a protocol.

Layout (one tree per experiment day; protocols accumulate under it)::

    {OUTPUT_DIR}/{exp_name}/
        manifest.txt           # human-readable summary
        manifest.json          # machine-readable source of truth
        mosaic.png             # whole-experiment mosaic (main types only)
        {protocol_short_name}/
            index.csv          # one row per saved cell (path + qc + cond meta)
            cells/{celltype}/cell_{id}.png

Design choices:

- **Per-experiment root**: outputs live next to the day's data (under
  ``OUTPUT_DIR/{exp_name}/``), so a date folder gathers everything we know
  about that recording day — calibration, mosaic, protocols, QC tables.
- **Manifest is append-only by protocol**: re-running another protocol on
  the same date adds a section without touching prior ones.
- **Incremental render**: PNGs that already exist are skipped (``overwrite=False``)
  so adding new analyses doesn't redo the cell-level work.
- **Condition coloring**: when ``stim_block`` + ``condition_key`` are
  supplied (auto-detected for known protocols), raster rows and the PSTH
  are split by condition value using a sequential colormap.
- **Speed**: ellipses go through a single :class:`PatchCollection`; raster
  ticks through a :class:`LineCollection`. The whole-experiment mosaic
  background is rasterized once and re-pasted per cell.
"""

from __future__ import annotations

import datetime as _dt
import io
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse

from .psth import epoch_spikes_to_psth, psth_time_axis
from .raster import plot_single_cell_raster
from .vision_utils import get_ells
from .mosaic_overlay import plot_stim_with_mosaic, electrode_positions_canvas_px
from .style import (
    MAIN_CELL_TYPES, NEUTRAL_GRAY,
    color_for_celltype, colors_for_celltypes, colors_for_conditions,
    apply_publication_style,
)
from ..config.settings import OUTPUT_DIR


__all__ = [
    'experiment_root',
    'protocol_short_name',
    'save_per_cell_plots',
    'save_experiment_mosaic',
    'update_manifest',
]


# ---------------------------------------------------------------------------
# Protocol → short-name mapping
# ---------------------------------------------------------------------------

# Known-protocol → short-name registry. Anything not listed falls through
# to the auto-derived name (last dotted segment, CamelCase → snake_case).
_PROTOCOL_SHORT_NAMES: Dict[str, str] = {
    'edu.washington.riekelab.turner.protocols.EyeMovementTrajectoryAlternatingBackground':
        'eye_movement_alt_bg',
    'edu.washington.riekelab.chris.protocols.VariableMeanSpatialNoise':
        'variable_mean_spatial_noise',
    'manookinlab.protocols.VariableMeanSpatialNoise':
        'variable_mean_spatial_noise',
    'edu.washington.riekelab.chris.protocols.monitorVariableMeanNoiseEpochs':
        'one_d_noise',
    'edu.washington.riekelab.vyom.protocols.monitorVariableMeanNoiseEpochs':
        'one_d_noise',
}

# Condition keys auto-detect per protocol (used when caller doesn't pass any).
# Order matters: the FIRST key is the "primary" axis (e.g. natural image —
# each gets its own subplot), and subsequent keys are overlaid inside each
# subplot (e.g. background scale — colored traces within a panel).
_PROTOCOL_DEFAULT_CONDITION_KEYS: Dict[str, List[str]] = {
    'edu.washington.riekelab.turner.protocols.EyeMovementTrajectoryAlternatingBackground':
        ['currentImageName', 'currentBackgroundScale'],
    'edu.washington.riekelab.chris.protocols.monitorVariableMeanNoiseEpochs':
        ['currentMean'],
    'edu.washington.riekelab.vyom.protocols.monitorVariableMeanNoiseEpochs':
        ['currentMean'],
}

# Per-epoch *array* parameters to additionally persist into offline.h5
# (regular condition_keys are scalar per-epoch values that go into the
# compound `epochs` table; arrays don't fit there and live under their
# own `epoch_arrays/<key>` group). Used by `save_offline_data` to stash
# things like the per-frame intensity trace for LN-model fitting.
_PROTOCOL_EXTRA_EPOCH_ARRAYS: Dict[str, List[str]] = {
    'edu.washington.riekelab.chris.protocols.monitorVariableMeanNoiseEpochs':
        ['intensityOverFrame'],
    'edu.washington.riekelab.vyom.protocols.monitorVariableMeanNoiseEpochs':
        ['intensityOverFrame'],
}


def protocol_short_name(full_name: str) -> str:
    """Return a filesystem-friendly short name for a protocol class.

    Falls back to the last dotted segment with CamelCase → snake_case.
    """
    if full_name in _PROTOCOL_SHORT_NAMES:
        return _PROTOCOL_SHORT_NAMES[full_name]
    tail = full_name.rsplit('.', 1)[-1]
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', tail).lower()
    return s or 'protocol'


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def experiment_root(exp_name: str, output_root: Optional[str] = None) -> str:
    """Return ``{output_root}/{exp_name}/`` (created)."""
    root = output_root or OUTPUT_DIR
    p = os.path.join(root, exp_name)
    os.makedirs(p, exist_ok=True)
    return p


def _safe_celltype(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return 'unclassified'
    return str(s).replace('/', '_').replace(' ', '_')


# Matches cells/<celltype>/cell_<id>_<suffix>.png (raster | psth | …)
_CELL_PNG_RE = re.compile(r'^cell_(\d+)_[A-Za-z0-9]+\.png$')


def _prune_stale_cell_pngs(cells_root: str, kept_ids: set) -> int:
    """Delete per-cell PNGs whose ``cell_id`` is not in ``kept_ids``.

    Walks ``cells/<celltype>/cell_<id>_*.png`` and removes any file
    whose id is now outside the kept set. Returns the count removed.
    Also drops empty cell-type directories. Skips anything that doesn't
    match the canonical filename pattern, so a stray user file in the
    archive won't be touched.
    """
    if not os.path.isdir(cells_root):
        return 0
    n_removed = 0
    for ct_name in os.listdir(cells_root):
        ct_dir = os.path.join(cells_root, ct_name)
        if not os.path.isdir(ct_dir):
            continue
        for fname in os.listdir(ct_dir):
            m = _CELL_PNG_RE.match(fname)
            if m is None:
                continue
            cid = int(m.group(1))
            if cid not in kept_ids:
                try:
                    os.remove(os.path.join(ct_dir, fname))
                    n_removed += 1
                except OSError:
                    pass
        # Drop the celltype dir if it ended up empty
        try:
            if not os.listdir(ct_dir):
                os.rmdir(ct_dir)
        except OSError:
            pass
    return n_removed


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _read_manifest_json(exp_root: str) -> Dict:
    path = os.path.join(exp_root, 'manifest.json')
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _write_manifest_files(exp_root: str, data: Dict):
    """Persist the manifest as both JSON (canonical) and TXT (human-readable)."""
    json_path = os.path.join(exp_root, 'manifest.json')
    txt_path = os.path.join(exp_root, 'manifest.txt')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)

    # Render TXT
    lines = []
    lines.append(f'# {data.get("exp_name", "?")}')
    lines.append(f'exp_name: {data.get("exp_name", "?")}')
    lines.append(f'rig: {data.get("rig", "?")}')
    lines.append(f'noise_chunk: {data.get("noise_chunk", "?")}')
    lines.append(f'updated_utc: {data.get("updated_utc", "?")}')
    lines.append('')
    for short, info in sorted(data.get('protocols', {}).items()):
        lines.append(f'[protocol: {short}]')
        for k, v in info.items():
            if isinstance(v, list):
                v_str = ', '.join(str(x) for x in v)
                lines.append(f'  {k}: [{v_str}]')
            else:
                lines.append(f'  {k}: {v}')
        lines.append('')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines).rstrip() + '\n')


def update_manifest(
    exp_name: str,
    output_root: Optional[str] = None,
    rig: Optional[str] = None,
    noise_chunk: Optional[str] = None,
    protocol_entry: Optional[Dict] = None,
) -> Dict:
    """Append-or-update one protocol entry in the experiment manifest.

    Returns the full manifest dict after the update.
    """
    exp_root = experiment_root(exp_name, output_root=output_root)
    data = _read_manifest_json(exp_root)
    data.setdefault('exp_name', exp_name)
    if rig is not None:
        data['rig'] = rig
    if noise_chunk is not None:
        data['noise_chunk'] = noise_chunk
    data.setdefault('protocols', {})
    if protocol_entry is not None:
        short = protocol_entry.get('short_name') or 'protocol'
        # Always overwrite the entry's keys with the latest values.
        existing = data['protocols'].get(short, {})
        existing.update(protocol_entry)
        data['protocols'][short] = existing
    data['updated_utc'] = _dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    _write_manifest_files(exp_root, data)
    return data


# ---------------------------------------------------------------------------
# Mosaic precompute (shared background image)
# ---------------------------------------------------------------------------

def _mosaic_ellipse_params(
    analysis_chunk,
    cell_types: Sequence[str],
    typing_file: Optional[str],
    std_scaling: float = 1.6,
) -> Tuple[List[Tuple[int, str, Tuple[float, float], float, float, float]],
           Dict[int, Tuple]]:
    """Return ellipse params for plotting + a lookup by cell_id.

    Each tuple is ``(cell_id, cell_type, (cx, cy), width, height, angle)``
    in canvas pixels. The second return is the same data keyed by cell_id
    for fast highlight lookup.
    """
    if typing_file is None:
        typing_file = analysis_chunk.typing_files[0] if analysis_chunk.typing_files else None
    if typing_file is None:
        return [], {}
    idx = analysis_chunk.typing_files.index(typing_file)
    col = f'typing_file_{idx}'
    df = analysis_chunk.df_cell_params

    # Build {ct: [ids]} once, then go through get_ells
    by_type: Dict[str, List[int]] = {}
    for ct in cell_types:
        ids = df.loc[df[col] == ct, 'cell_id'].astype(int).tolist()
        if ids:
            by_type[ct] = ids
    if not by_type:
        return [], {}
    d_ells, _ = get_ells(analysis_chunk, by_type, std_scaling=std_scaling, units='pixels')

    out: List[Tuple] = []
    lookup: Dict[int, Tuple] = {}
    for ct, ells in d_ells.items():
        for cid, ell in ells.items():
            entry = (int(cid), ct, ell.center, ell.width, ell.height, ell.angle)
            out.append(entry)
            lookup[int(cid)] = entry
    return out, lookup


def _draw_mosaic_into(
    ax,
    ellipse_params: Sequence[Tuple],
    type_colors: Dict[str, str],
    canvas_size: Tuple[float, float],
    em_canvas: Optional[np.ndarray] = None,
    highlight_id: Optional[int] = None,
    highlight_color: str = '#D55E00',
    peer_alpha: float = 0.35,
    peer_lw: float = 0.7,
    legend: bool = False,
):
    """Draw all ellipses into ``ax`` as a single PatchCollection + highlight."""
    canvas_w, canvas_h = canvas_size
    # Gray background
    ax.imshow(
        np.full((max(int(canvas_h), 2), max(int(canvas_w), 2)), 0.94),
        extent=(0, canvas_w, canvas_h, 0),
        cmap='gray', vmin=0, vmax=1,
    )

    patches = []
    edgecolors = []
    linewidths = []
    alphas = []
    for cid, ct, center, w, h, angle in ellipse_params:
        if cid == highlight_id:
            continue  # drawn separately on top
        patches.append(Ellipse(xy=center, width=w, height=h, angle=angle))
        edgecolors.append(type_colors.get(ct, NEUTRAL_GRAY))
        linewidths.append(peer_lw)
        alphas.append(peer_alpha)
    if patches:
        pc = PatchCollection(patches, match_original=False)
        pc.set_facecolor('none')
        pc.set_edgecolors(edgecolors)
        pc.set_linewidths(linewidths)
        pc.set_alpha(alphas)
        ax.add_collection(pc)

    if em_canvas is not None:
        ax.scatter(em_canvas[:, 0], em_canvas[:, 1],
                   s=2, c='white', edgecolors='black',
                   linewidths=0.2, alpha=0.6, zorder=4)

    # Highlight on top (after electrodes so it's never hidden).
    if highlight_id is not None and highlight_id in {e[0] for e in ellipse_params}:
        _, ct, center, w, h, angle = next(e for e in ellipse_params if e[0] == highlight_id)
        ax.add_patch(Ellipse(
            xy=center, width=w, height=h, angle=angle,
            facecolor='none', edgecolor=highlight_color, linewidth=2.0,
            zorder=6,
        ))

    if legend:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color=c, lw=1.6, label=ct)
            for ct, c in type_colors.items()
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.7)

    ax.set_xlim(0, canvas_w)
    ax.set_ylim(canvas_h, 0)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def _typed_cell_ids(analysis_chunk, cell_type: str,
                    typing_file: Optional[str]) -> List[int]:
    typing_file = typing_file or (
        analysis_chunk.typing_files[0] if analysis_chunk.typing_files else None
    )
    if typing_file is None:
        return []
    idx = analysis_chunk.typing_files.index(typing_file)
    return analysis_chunk.df_cell_params.query(
        f'typing_file_{idx} == @cell_type'
    )['cell_id'].astype(int).tolist()


def _isi_bin_centers(analysis_chunk) -> np.ndarray:
    edges = np.asarray(analysis_chunk.isi_bin_edges)
    return 0.5 * (edges[:-1] + edges[1:])


def _draw_tc_panel(ax, analysis_chunk, ids, color):
    """Mean temporal filter (green channel) + per-cell traces dim."""
    tcs = []
    for cid in ids:
        tc = analysis_chunk.d_timecourses.get(cid)
        if tc is None:
            continue
        tcs.append(tc['green'])
    if not tcs:
        ax.text(0.5, 0.5, '(no timecourses)', transform=ax.transAxes,
                ha='center', va='center', fontsize=8)
        return
    L = min(len(t) for t in tcs)
    mat = np.stack([t[:L] for t in tcs])
    x = np.arange(L)
    for row in mat:
        ax.plot(x, row, color=color, alpha=0.12, linewidth=0.6)
    mean = mat.mean(axis=0)
    sem = mat.std(axis=0) / np.sqrt(max(mat.shape[0], 1))
    ax.plot(x, mean, color=color, linewidth=1.6)
    ax.fill_between(x, mean - sem, mean + sem,
                    color=color, alpha=0.25, linewidth=0)
    ax.axhline(0, color='gray', lw=0.4, alpha=0.5)


def _draw_isi_panel(ax, analysis_chunk, ids, color, xlim_ms=200.0):
    """Mean ISI density + per-cell traces dim."""
    centers = _isi_bin_centers(analysis_chunk)
    rows = []
    for cid in ids:
        h = analysis_chunk.d_ISIs.get(cid)
        if h is None:
            continue
        h = np.asarray(h, dtype=float)
        s = h.sum()
        rows.append(h / s if s > 0 else h)
    if not rows:
        ax.text(0.5, 0.5, '(no ISI data)', transform=ax.transAxes,
                ha='center', va='center', fontsize=8)
        return
    mat = np.stack(rows)
    for row in mat:
        ax.plot(centers, row, color=color, alpha=0.12, linewidth=0.6)
    mean = mat.mean(axis=0)
    sem = mat.std(axis=0) / np.sqrt(max(mat.shape[0], 1))
    ax.plot(centers, mean, color=color, linewidth=1.6)
    ax.fill_between(centers, mean - sem, mean + sem,
                    color=color, alpha=0.25, linewidth=0)
    ax.set_xlim(0, xlim_ms)


def save_experiment_mosaic(
    analysis_chunk,
    out_path: str,
    main_types: Sequence[str] = MAIN_CELL_TYPES,
    show_electrodes: bool = True,
    use_calibration: bool = True,
    typing_file: Optional[str] = None,
    include_tc_isi: bool = True,
    isi_xlim_ms: float = 200.0,
    minimum_n: int = 3,
    dpi: int = 130,
) -> str:
    """Composite mosaic figure: mosaic on top + per-type tc + ISI rows below.

    With ``include_tc_isi=False`` reverts to the legacy mosaic-only layout.
    Cell types absent from the chunk or below ``minimum_n`` are skipped.
    """
    apply_publication_style()
    canvas_w, canvas_h = analysis_chunk.canvas_size
    type_colors = colors_for_celltypes(main_types)
    ellipses, _lookup = _mosaic_ellipse_params(
        analysis_chunk, main_types, typing_file=typing_file,
    )
    em_canvas = electrode_positions_canvas_px(
        analysis_chunk, use_calibration=use_calibration,
    ) if show_electrodes else None

    if not include_tc_isi:
        fig_w = min(8.0, canvas_w / 100.0)
        fig_h = fig_w * canvas_h / canvas_w
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        _draw_mosaic_into(
            ax, ellipses, type_colors, (canvas_w, canvas_h),
            em_canvas=em_canvas, legend=True,
        )
        ax.set_title(f'{analysis_chunk.exp_name} / {analysis_chunk.chunk_name}')
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return out_path

    # Composite layout:
    #   row 0  — full mosaic spanning all columns (height ≈ canvas aspect)
    #   row 1+ — one row per cell type, columns = [TC, ISI]
    types_with_ids = []
    for ct in main_types:
        ids = _typed_cell_ids(analysis_chunk, ct, typing_file)
        if len(ids) >= minimum_n:
            types_with_ids.append((ct, ids))
    n_types = len(types_with_ids)

    fig_w = 11.0
    mosaic_h = fig_w * (canvas_h / canvas_w) * 0.55
    row_h = 2.0
    fig_h = mosaic_h + row_h * max(n_types, 1) + 0.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        1 + max(n_types, 1), 2,
        height_ratios=[mosaic_h] + [row_h] * max(n_types, 1),
        hspace=0.5, wspace=0.2,
    )
    ax_mos = fig.add_subplot(gs[0, :])
    _draw_mosaic_into(
        ax_mos, ellipses, type_colors, (canvas_w, canvas_h),
        em_canvas=em_canvas, legend=True,
    )
    ax_mos.set_title(
        f'{analysis_chunk.exp_name} / {analysis_chunk.chunk_name} — '
        f'{sum(1 for _ in ellipses)} cells '
        f'({", ".join(ct for ct, _ in types_with_ids)})'
    )

    for r, (ct, ids) in enumerate(types_with_ids, start=1):
        color = type_colors.get(ct, NEUTRAL_GRAY)
        ax_tc = fig.add_subplot(gs[r, 0])
        _draw_tc_panel(ax_tc, analysis_chunk, ids, color)
        ax_tc.set_xlabel('STA frame', fontsize=8)
        ax_tc.set_ylabel('contrast', fontsize=8)
        ax_tc.set_title(f'{ct}  •  mean temporal filter  (n={len(ids)})',
                        fontsize=9, loc='left', color=color, pad=2)

        ax_isi = fig.add_subplot(gs[r, 1])
        _draw_isi_panel(ax_isi, analysis_chunk, ids, color, xlim_ms=isi_xlim_ms)
        ax_isi.set_xlabel('ISI (ms)', fontsize=8)
        ax_isi.set_ylabel('density', fontsize=8)
        ax_isi.set_title(f'{ct}  •  mean ISI',
                         fontsize=9, loc='left', color=color, pad=2)

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Condition mapping
# ---------------------------------------------------------------------------

def _resolve_conditions_per_epoch(
    stim_block,
    condition_keys: Optional[Sequence[str]],
    n_epochs: int,
) -> Tuple[Optional[List[List]], Optional[List[str]]]:
    """Return per-epoch condition values for each key + the key list used.

    Output: ``(values, keys)``:
      - ``values`` is a list of length len(keys); each entry is a list of
        length n_epochs giving that key's value per epoch.
      - ``keys`` is the same list of keys (auto-resolved from the protocol
        if the caller passed None).
      - ``(None, None)`` when no usable condition can be resolved.

    A key that resolves to a single unique value is silently dropped (no
    meaningful split). If no keys remain, returns ``(None, None)``.
    """
    if stim_block is None:
        return None, None
    keys = list(condition_keys) if condition_keys else None
    if keys is None:
        proto = getattr(stim_block, 'protocol_name', None)
        keys = list(_PROTOCOL_DEFAULT_CONDITION_KEYS.get(proto, []))
    if not keys:
        return None, None

    df = getattr(stim_block, 'df_epochs', None)
    if df is None or len(df) == 0:
        return None, None

    have_idx = 'epoch_index' in df.columns
    out_values: List[List] = []
    out_keys: List[str] = []
    for key in keys:
        if key in df.columns:
            raw = df[key].tolist()
        else:
            raw = [p.get(key) for p in df['epoch_parameters']]
        if have_idx:
            idx2val = dict(zip(df['epoch_index'].astype(int).tolist(), raw))
            per_epoch = [idx2val.get(i) for i in range(n_epochs)]
        else:
            per_epoch = raw[:n_epochs] + [None] * max(0, n_epochs - len(raw))
        uniq = {v for v in per_epoch if v is not None}
        if len(uniq) <= 1:
            continue  # degenerate; drop
        out_values.append(per_epoch)
        out_keys.append(key)
    if not out_keys:
        return None, None
    return out_values, out_keys


def _sorted_unique(values: Iterable):
    """Sort a list of mixed types (numeric + str) in a stable, intuitive order."""
    return sorted({v for v in values if v is not None},
                  key=lambda x: (isinstance(x, str), x))


# ---------------------------------------------------------------------------
# Per-cell triptych (uses precomputed mosaic geometry + condition coloring)
# ---------------------------------------------------------------------------

def _epoch_psths(
    spike_times_by_epoch: Sequence[np.ndarray],
    epoch_indices: Sequence[int],
    t_end_ms: float,
    psth_sigma_ms: float,
    sample_rate_hz: float,
) -> Optional[np.ndarray]:
    """PSTH stack for a subset of epochs. ``None`` if the subset is empty."""
    if not epoch_indices:
        return None
    epochs = [spike_times_by_epoch[i] for i in epoch_indices
              if i < len(spike_times_by_epoch)]
    if not epochs:
        return None
    return epoch_spikes_to_psth(
        epochs, t_end_ms,
        psth_sigma_ms=psth_sigma_ms, sample_rate_hz=sample_rate_hz,
    )


def _layout_grid(n: int, max_cols: int = 5) -> Tuple[int, int]:
    """Return ``(nrows, ncols)`` for ``n`` panels (roughly square, cap on cols)."""
    if n <= 1:
        return 1, 1
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def _draw_mosaic_panel(ax, *, cell_id, cell_type, ellipse_params,
                        type_colors, em_canvas, canvas_size):
    cell_color = type_colors.get(cell_type, color_for_celltype(cell_type))
    _draw_mosaic_into(
        ax, ellipse_params, type_colors, canvas_size,
        em_canvas=em_canvas, highlight_id=cell_id,
        highlight_color=cell_color,
    )
    ax.set_title(f'cell {cell_id} ({cell_type})')


def _save_cell_raster_png(
    cell_id: int,
    cell_type: str,
    spike_times_by_epoch: Sequence[np.ndarray],
    out_path: str,
    *,
    ellipse_params: Sequence[Tuple],
    type_colors: Dict[str, str],
    em_canvas: Optional[np.ndarray],
    canvas_size: Tuple[float, float],
    pre_ms: float,
    stim_ms: float,
    tail_ms: float,
    cond_values_per_epoch: Optional[List[List]],
    cond_keys: Optional[List[str]],
    dpi: int,
) -> bool:
    """Render one cell's raster (mosaic on left, grouped epochs on right)."""
    n_epochs = len(spike_times_by_epoch)
    t_end_ms = pre_ms + stim_ms + tail_ms
    if t_end_ms <= 0:
        m = 0.0
        for arr in spike_times_by_epoch:
            if len(arr):
                m = max(m, float(np.asarray(arr).max()))
        t_end_ms = max(m, 1.0)

    # Build groupby dict (preserve key order: outer → inner)
    groupby = None
    if cond_keys and cond_values_per_epoch:
        groupby = {k: v for k, v in zip(cond_keys, cond_values_per_epoch)}

    cell_color = type_colors.get(cell_type, color_for_celltype(cell_type))

    fig_h = max(4.0, n_epochs * 0.16)
    fig = plt.figure(figsize=(13.5, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 9.0], wspace=0.18)
    ax_mos = fig.add_subplot(gs[0, 0])
    ax_ras = fig.add_subplot(gs[0, 1])
    _draw_mosaic_panel(
        ax_mos, cell_id=cell_id, cell_type=cell_type,
        ellipse_params=ellipse_params, type_colors=type_colors,
        em_canvas=em_canvas, canvas_size=canvas_size,
    )
    plot_single_cell_raster(
        spike_times_by_epoch,
        t_start_ms=0.0, t_end_ms=t_end_ms,
        groupby_conditions=groupby,
        ax=ax_ras,
        default_color=cell_color,
        pre_time_ms=pre_ms, stim_time_ms=stim_ms,
        title=(f'{n_epochs} epochs'
               + (f' • grouped by {", ".join(cond_keys)}' if cond_keys else '')),
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return True


def _save_cell_psth_png(
    cell_id: int,
    cell_type: str,
    spike_times_by_epoch: Sequence[np.ndarray],
    out_path: str,
    *,
    ellipse_params: Sequence[Tuple],
    type_colors: Dict[str, str],
    em_canvas: Optional[np.ndarray],
    canvas_size: Tuple[float, float],
    pre_ms: float,
    stim_ms: float,
    tail_ms: float,
    cond_values_per_epoch: Optional[List[List]],
    cond_keys: Optional[List[str]],
    psth_sigma_ms: float,
    sample_rate_hz: float,
    psth_ncols: int,
    dpi: int,
) -> bool:
    """Render one cell's PSTH (mosaic on left, vertically stacked panels on right).

    For 1 condition key: single PSTH panel with one trace per condition.
    For 2 condition keys: one panel per primary value (e.g. image), with
    one trace per secondary value (e.g. bg scale). Panels stack vertically
    by default (``psth_ncols=1``) so each x-axis has room.
    """
    n_epochs = len(spike_times_by_epoch)
    t_end_ms = pre_ms + stim_ms + tail_ms
    if t_end_ms <= 0:
        m = 0.0
        for arr in spike_times_by_epoch:
            if len(arr):
                m = max(m, float(np.asarray(arr).max()))
        t_end_ms = max(m, 1.0)
    t_axis = psth_time_axis(t_end_ms, sample_rate_hz)
    cell_color = type_colors.get(cell_type, color_for_celltype(cell_type))
    n_keys = len(cond_keys) if cond_keys else 0

    if n_keys <= 1:
        primary_values = cond_values_per_epoch[0] if n_keys == 1 else None
        primary_key = cond_keys[0] if n_keys == 1 else None
        unique_vals = _sorted_unique(primary_values) if primary_values else None
        color_map = colors_for_conditions(unique_vals) if unique_vals else {}

        fig = plt.figure(figsize=(13.5, 4.0))
        gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 9.0], wspace=0.18)
        ax_mos = fig.add_subplot(gs[0, 0])
        ax_psth = fig.add_subplot(gs[0, 1])
        _draw_mosaic_panel(
            ax_mos, cell_id=cell_id, cell_type=cell_type,
            ellipse_params=ellipse_params, type_colors=type_colors,
            em_canvas=em_canvas, canvas_size=canvas_size,
        )

        if primary_values and unique_vals:
            for v in unique_vals:
                idxs = [i for i, x in enumerate(primary_values) if x == v]
                stack = _epoch_psths(spike_times_by_epoch, idxs, t_end_ms,
                                     psth_sigma_ms, sample_rate_hz)
                if stack is None:
                    continue
                c = color_map[v]
                mu = stack.mean(axis=0)
                sem = stack.std(axis=0) / np.sqrt(max(stack.shape[0], 1))
                ax_psth.plot(t_axis, mu, color=c, linewidth=1.2,
                             label=f'{primary_key}={v}  (n={stack.shape[0]})')
                ax_psth.fill_between(t_axis, mu - sem, mu + sem,
                                     color=c, alpha=0.2, linewidth=0)
            ax_psth.legend(loc='upper right', fontsize=8, framealpha=0.7)
        else:
            psth = epoch_spikes_to_psth(
                spike_times_by_epoch, t_end_ms,
                psth_sigma_ms=psth_sigma_ms, sample_rate_hz=sample_rate_hz,
            )
            mu = psth.mean(axis=0)
            sem = psth.std(axis=0) / np.sqrt(max(psth.shape[0], 1))
            ax_psth.plot(t_axis, mu, color=cell_color, linewidth=1.2)
            ax_psth.fill_between(t_axis, mu - sem, mu + sem,
                                 color=cell_color, alpha=0.2, linewidth=0)
        if stim_ms > 0:
            for x in (pre_ms, pre_ms + stim_ms):
                ax_psth.axvline(x, color='red', lw=0.5, ls='--', alpha=0.6)
        ax_psth.set_xlim(0, t_end_ms)
        ax_psth.set_xlabel('time (ms)')
        ax_psth.set_ylabel('rate (Hz)')
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return True

    # ---- 2-key: stack panels vertically (or grid if psth_ncols > 1)
    primary_per_epoch = cond_values_per_epoch[0]
    secondary_per_epoch = cond_values_per_epoch[1]
    primary_key, secondary_key = cond_keys[0], cond_keys[1]
    primary_values = _sorted_unique(primary_per_epoch)
    secondary_values = _sorted_unique(secondary_per_epoch)
    sec_colors = colors_for_conditions(secondary_values)

    n_p = len(primary_values)
    ncols = max(1, psth_ncols)
    nrows = int(np.ceil(n_p / ncols))

    # Mosaic on left; on right, a vertical stack of panels (taller than wide).
    panel_h = 1.6
    fig_h = max(4.0, panel_h * nrows + 0.8)
    fig_w = 13.5 if ncols == 1 else (4.5 + 4.5 * ncols)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs_outer = fig.add_gridspec(
        1, 2,
        width_ratios=[4.5, fig_w - 4.5],
        wspace=0.18,
    )
    ax_mos = fig.add_subplot(gs_outer[0, 0])
    _draw_mosaic_panel(
        ax_mos, cell_id=cell_id, cell_type=cell_type,
        ellipse_params=ellipse_params, type_colors=type_colors,
        em_canvas=em_canvas, canvas_size=canvas_size,
    )
    gs_inner = gs_outer[0, 1].subgridspec(nrows, ncols, hspace=0.45, wspace=0.2)

    # Precompute per-(p,s) stacks so we share y-limits across panels.
    panel_data: Dict[Tuple, Dict] = {}
    y_max = 0.0
    for p in primary_values:
        for s in secondary_values:
            idxs = [i for i in range(n_epochs)
                    if primary_per_epoch[i] == p and secondary_per_epoch[i] == s]
            stack = _epoch_psths(spike_times_by_epoch, idxs, t_end_ms,
                                 psth_sigma_ms, sample_rate_hz)
            if stack is None:
                continue
            mu = stack.mean(axis=0)
            sem = stack.std(axis=0) / np.sqrt(max(stack.shape[0], 1))
            panel_data[(p, s)] = {'mu': mu, 'sem': sem, 'n': stack.shape[0]}
            y_max = max(y_max, float((mu + sem).max()))

    handles = []
    last_row = nrows - 1
    for i, p in enumerate(primary_values):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs_inner[r, c])
        for s in secondary_values:
            d = panel_data.get((p, s))
            if d is None:
                continue
            color = sec_colors[s]
            line, = ax.plot(t_axis, d['mu'], color=color, linewidth=1.0,
                            label=f'{secondary_key}={s}')
            ax.fill_between(t_axis, d['mu'] - d['sem'], d['mu'] + d['sem'],
                            color=color, alpha=0.2, linewidth=0)
            if i == 0:
                handles.append(line)
        if stim_ms > 0:
            for x in (pre_ms, pre_ms + stim_ms):
                ax.axvline(x, color='red', lw=0.4, ls='--', alpha=0.5)
        ax.set_ylim(0, y_max * 1.05 if y_max > 0 else 1.0)
        ax.set_xlim(0, t_end_ms)
        ax.set_title(f'{primary_key}={p}', fontsize=9, loc='left', pad=2)
        ax.set_ylabel('rate (Hz)', fontsize=8)
        if r != last_row:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('time (ms)', fontsize=8)
    # Hide unused trailing panels in last row when ncols > 1
    for j in range(n_p, nrows * ncols):
        r, c = divmod(j, ncols)
        # Note: subplot already absent; nothing to hide explicitly with subgridspec
        pass
    if handles:
        fig.legend(handles=handles, loc='upper right',
                   bbox_to_anchor=(0.99, 0.99), fontsize=8,
                   framealpha=0.7, title=secondary_key)

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Top-level: save the whole protocol's archive
# ---------------------------------------------------------------------------

def _render_one_cell_worker(per_cell, shared):
    """Top-level worker so joblib's loky backend can pickle it.

    ``per_cell`` carries per-cell data (id, spikes, output paths). ``shared``
    carries everything that is identical across cells (geometry, colors,
    timing, condition mapping, render flags). Splitting the args this way
    keeps the IPC payload small even though we have many cells.
    """
    # Worker processes inherit no matplotlib state — force a headless backend
    # so we don't drag in a display.
    import matplotlib
    matplotlib.use('Agg', force=True)

    cid = per_cell['cell_id']
    ct = per_cell['cell_type']
    spikes = per_cell['spike_times']
    raster_path = per_cell['raster_path']
    psth_path = per_cell['psth_path']
    overwrite = shared['overwrite']

    rendered_raster = False
    if (overwrite or not os.path.exists(raster_path)) and spikes is not None:
        rendered_raster = _save_cell_raster_png(
            cid, ct, spikes, raster_path,
            ellipse_params=shared['ellipse_params'],
            type_colors=shared['type_colors'],
            em_canvas=shared['em_canvas'],
            canvas_size=shared['canvas_size'],
            pre_ms=shared['pre_ms'], stim_ms=shared['stim_ms'],
            tail_ms=shared['tail_ms'],
            cond_values_per_epoch=shared['cond_values_per_epoch'],
            cond_keys=shared['cond_keys'],
            dpi=shared['dpi'],
        )

    rendered_psth = False
    if (overwrite or not os.path.exists(psth_path)) and spikes is not None:
        rendered_psth = _save_cell_psth_png(
            cid, ct, spikes, psth_path,
            ellipse_params=shared['ellipse_params'],
            type_colors=shared['type_colors'],
            em_canvas=shared['em_canvas'],
            canvas_size=shared['canvas_size'],
            pre_ms=shared['pre_ms'], stim_ms=shared['stim_ms'],
            tail_ms=shared['tail_ms'],
            cond_values_per_epoch=shared['cond_values_per_epoch'],
            cond_keys=shared['cond_keys'],
            psth_sigma_ms=shared['psth_sigma_ms'],
            sample_rate_hz=shared['sample_rate_hz'],
            psth_ncols=shared['psth_ncols'],
            dpi=shared['dpi'],
        )

    return {
        'cell_id': cid,
        'cell_type': per_cell['cell_type_raw'],
        'raster_path': raster_path if os.path.exists(raster_path) else None,
        'psth_path': psth_path if os.path.exists(psth_path) else None,
        'rendered_raster': rendered_raster,
        'rendered_psth': rendered_psth,
    }


def save_per_cell_plots(
    analysis_chunk,
    response_block,
    stim_block=None,
    protocol_name: Optional[str] = None,
    protocol_short_name_: Optional[str] = None,
    cell_types: Optional[Iterable[str]] = None,
    cell_ids: Optional[Iterable[int]] = None,
    qc_pass_only: Optional[pd.DataFrame] = None,
    condition_keys: Optional[Sequence[str]] = None,
    main_types: Sequence[str] = MAIN_CELL_TYPES,
    typing_file: Optional[str] = None,
    psth_sigma_ms: float = 10.0,
    sample_rate_hz: float = 500.0,
    show_electrodes: bool = True,
    use_calibration: bool = True,
    output_root: Optional[str] = None,
    dpi: int = 90,
    psth_ncols: int = 1,
    overwrite: bool = False,
    n_jobs: int = -1,
    verbose: bool = True,
    ndf: Optional[float] = None,
    prune_stale: bool = True,
) -> pd.DataFrame:
    """Write the per-cell archive for one protocol into the experiment tree.

    Side effects:
      - Creates ``{OUTPUT_DIR}/{exp_name}/`` (and protocol subdir).
      - Writes / refreshes ``mosaic.png`` for the experiment (main types only).
      - Renders each requested cell to ``cells/{celltype}/cell_{id}.png``,
        skipping any PNG that already exists when ``overwrite=False``.
      - Appends/updates a protocol section in ``manifest.json`` / ``manifest.txt``.

    Parameters
    ----------
    stim_block : optional
        Used to pull per-epoch ``condition_key`` values for raster/PSTH
        coloring. Auto-detected for known protocols (e.g.
        ``EyeMovementTrajectoryAlternatingBackground`` → ``currentBackgroundScale``).
    main_types : sequence
        Cell types to include in the experiment-level ``mosaic.png`` (the
        per-cell mosaic panels follow the same restriction).
    ndf : float, optional
        Annotate the manifest entry with the day's NDF setting.
    prune_stale : bool
        When ``True`` (default), delete any pre-existing PNG under
        ``cells/<celltype>/cell_<id>_*.png`` whose ``cell_id`` is *not*
        in the kept set (i.e. cells filtered out by ``cell_ids`` or
        ``qc_pass_only``). This is how §17/§18 collapses the archive
        to the visual-QC ``good`` subset on re-run — without it, stale
        PNGs from cells later tagged ``bad`` linger on disk. Set
        ``False`` to leave existing PNGs untouched.

    Returns the index DataFrame (``cell_id, cell_type, png_path, rendered``).
    """
    apply_publication_style()

    if protocol_name is None:
        protocol_name = response_block.protocol_name
    short = protocol_short_name_ or protocol_short_name(protocol_name)

    exp_root = experiment_root(analysis_chunk.exp_name, output_root=output_root)
    proto_root = os.path.join(exp_root, short)
    os.makedirs(os.path.join(proto_root, 'cells'), exist_ok=True)

    # --- Experiment-level mosaic (re-written so it reflects current calibration)
    mosaic_path = os.path.join(exp_root, 'mosaic.png')
    if overwrite or not os.path.exists(mosaic_path):
        save_experiment_mosaic(
            analysis_chunk, mosaic_path,
            main_types=main_types,
            show_electrodes=show_electrodes,
            use_calibration=use_calibration,
            typing_file=typing_file,
            dpi=130,
        )
        if verbose:
            print(f'[cell_plot_archive] mosaic → {mosaic_path}')

    # --- Precompute geometry shared across all per-cell plots
    ellipse_params, _lookup = _mosaic_ellipse_params(
        analysis_chunk, main_types, typing_file=typing_file,
    )
    type_colors = colors_for_celltypes(main_types)
    em_canvas = electrode_positions_canvas_px(
        analysis_chunk, use_calibration=use_calibration,
    ) if show_electrodes else None
    canvas_size = tuple(analysis_chunk.canvas_size)

    # --- Per-cell selection
    df = response_block.df_spike_times
    if cell_types is not None:
        df = df[df['cell_type'].isin(set(cell_types))]
    else:
        df = df[df['cell_type'].isin(set(main_types))]
    if cell_ids is not None:
        wanted = set(int(c) for c in cell_ids)
        df = df[df['cell_id'].isin(wanted)]
    if qc_pass_only is not None:
        passing = set(qc_pass_only.loc[qc_pass_only['passes'], 'cell_id'].astype(int))
        df = df[df['cell_id'].isin(passing)]

    # --- Prune stale PNGs from prior runs (cells now outside the kept set)
    if prune_stale:
        kept_ids = set(int(c) for c in df['cell_id'].tolist())
        n_pruned = _prune_stale_cell_pngs(
            os.path.join(proto_root, 'cells'), kept_ids,
        )
        if verbose and n_pruned:
            print(f'[cell_plot_archive] pruned {n_pruned} stale per-cell PNG(s) '
                  f'(cells now outside kept set)')

    # --- Stim timing + condition mapping
    timing = getattr(response_block, 'd_timing', {}) or {}
    pre_ms = float(np.atleast_1d(timing.get('pre_time_ms', 0.0)).flat[0])
    stim_ms = float(np.atleast_1d(timing.get('stim_time_ms', 0.0)).flat[0])
    tail_ms = float(np.atleast_1d(timing.get('tail_time_ms', 0.0)).flat[0])

    # Pick n_epochs from first row
    n_epochs = 0
    if len(df):
        first = df.iloc[0]['spike_times']
        n_epochs = len(first)
    cond_values_per_epoch, cond_keys_used = _resolve_conditions_per_epoch(
        stim_block, condition_keys, n_epochs,
    )

    # --- Build per-cell job list (small + picklable); shared data once
    shared = {
        'ellipse_params': ellipse_params,
        'type_colors': type_colors,
        'em_canvas': em_canvas,
        'canvas_size': canvas_size,
        'pre_ms': pre_ms, 'stim_ms': stim_ms, 'tail_ms': tail_ms,
        'cond_values_per_epoch': cond_values_per_epoch,
        'cond_keys': cond_keys_used,
        'psth_sigma_ms': psth_sigma_ms,
        'sample_rate_hz': sample_rate_hz,
        'psth_ncols': psth_ncols,
        'dpi': dpi,
        'overwrite': overwrite,
    }
    jobs = []
    skipped_no_spikes = []
    for _, r in df.iterrows():
        cid = int(r['cell_id'])
        ct = _safe_celltype(r.get('cell_type', None))
        cell_dir = os.path.join(proto_root, 'cells', ct)
        os.makedirs(cell_dir, exist_ok=True)
        raster_path = os.path.join(cell_dir, f'cell_{cid}_raster.png')
        psth_path = os.path.join(cell_dir, f'cell_{cid}_psth.png')
        spikes = r['spike_times']
        if not spikes or all(len(np.asarray(s)) == 0 for s in spikes):
            skipped_no_spikes.append({
                'cell_id': cid, 'cell_type': r.get('cell_type', None),
                'raster_path': None, 'psth_path': None,
                'rendered_raster': False, 'rendered_psth': False,
            })
            continue
        # Fast-path: when both PNGs already on disk and overwrite=False,
        # skip dispatching this cell to a worker at all (saves IPC).
        if (not overwrite and os.path.exists(raster_path)
                and os.path.exists(psth_path)):
            skipped_no_spikes.append({
                'cell_id': cid, 'cell_type': r.get('cell_type', None),
                'raster_path': raster_path, 'psth_path': psth_path,
                'rendered_raster': False, 'rendered_psth': False,
            })
            continue
        jobs.append({
            'cell_id': cid, 'cell_type': ct,
            'cell_type_raw': r.get('cell_type', None),
            'spike_times': r['spike_times'],
            'raster_path': raster_path, 'psth_path': psth_path,
        })

    # --- Dispatch: parallel via joblib when there's enough work
    if not jobs:
        results = []
    elif n_jobs == 1 or len(jobs) <= 2:
        results = [_render_one_cell_worker(j, shared) for j in jobs]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
            delayed(_render_one_cell_worker)(j, shared) for j in jobs
        )

    rows = skipped_no_spikes + list(results)
    n_done_raster = sum(1 for r in results if r['rendered_raster'])
    n_done_psth = sum(1 for r in results if r['rendered_psth'])
    n_skipped_raster = sum(
        1 for r in skipped_no_spikes if r['raster_path'] is not None
    )
    n_skipped_psth = sum(
        1 for r in skipped_no_spikes if r['psth_path'] is not None
    )

    idx_df = pd.DataFrame(
        rows,
        columns=['cell_id', 'cell_type',
                 'raster_path', 'psth_path',
                 'rendered_raster', 'rendered_psth'],
    )
    idx_df.to_csv(os.path.join(proto_root, 'index.csv'), index=False)

    # --- Manifest update (per-protocol entry)
    rig = analysis_chunk.exp_name[8] if len(analysis_chunk.exp_name) >= 9 else None
    protocol_entry = {
        'short_name': short,
        'full_name': protocol_name,
        'datafile': getattr(response_block, 'datafile_name', None),
        'noise_chunk': analysis_chunk.chunk_name,
        'n_epochs': int(timing.get('n_epochs', n_epochs) or n_epochs),
        'pre_time_ms': pre_ms,
        'stim_time_ms': stim_ms,
        'tail_time_ms': tail_ms,
        'condition_keys': cond_keys_used or [],
        'n_cells_in_archive': int(
            (idx_df['raster_path'].notna() | idx_df['psth_path'].notna()).sum()
        ),
    }
    if cond_keys_used and cond_values_per_epoch:
        for k, vals in zip(cond_keys_used, cond_values_per_epoch):
            protocol_entry[f'unique_{k}'] = _sorted_unique(vals)
    if ndf is not None:
        protocol_entry['NDF'] = ndf
    update_manifest(
        analysis_chunk.exp_name,
        output_root=output_root,
        rig=rig,
        noise_chunk=analysis_chunk.chunk_name,
        protocol_entry=protocol_entry,
    )

    if verbose:
        print(f'[cell_plot_archive] {short}: '
              f'raster: {n_done_raster} rendered, {n_skipped_raster} skipped • '
              f'psth: {n_done_psth} rendered, {n_skipped_psth} skipped '
              f'→ {proto_root}')
    return idx_df
