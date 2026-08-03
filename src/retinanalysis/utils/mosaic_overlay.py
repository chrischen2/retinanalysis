"""Overlay an RGC mosaic on a stimulus frame.

The mosaic comes from STA fits on a spatial-noise chunk
(:class:`retinanalysis.classes.analysis_chunk.AnalysisChunk`). Each cell's
RF parameters are stored in *stixel/check* units of that noise grid, with
a y-axis flip already applied so the coords agree with ``imshow``'s
top-left origin (see ``analysis_chunk.get_rf_params``).

To overlay the mosaic on any other stimulus presented on the same rig, we
just need to scale the RF coords from stixels → canvas pixels via
``analysis_chunk.pixels_per_stixel`` and display the stim frame at the
same canvas extent. ``get_ells(..., units='pixels')`` already returns
``matplotlib.patches.Ellipse`` objects in canvas-pixel coords, so we
reuse it directly.
"""

from __future__ import annotations

import copy
from typing import Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse

from retinanalysis.utils.vision_utils import get_ells
from retinanalysis.utils.style import colors_for_celltypes


def electrode_positions_canvas_px(
    analysis_chunk,
    mea_center_canvas_px: Optional[Tuple[float, float]] = None,
    microns_per_pixel: Optional[float] = None,
    flip_y: bool = True,
    rotation_deg: Optional[float] = None,
    use_calibration: bool = True,
    recompute_alignment: bool = False,
    save_recomputed: bool = False,
) -> np.ndarray:
    """Convert MEA electrode positions (microns) to canvas-pixel coords.

    Three modes, in priority order:

    1. ``recompute_alignment=True`` — fit a fresh similarity transform from
       this chunk's EI-soma electrode vs. STA-center pairs and use it
       (optionally persisting the result with ``save_recomputed=True``).
    2. ``use_calibration=True`` (default) — load a saved per-rig calibration
       from ``rig_calibrations/rig_<rig_id>.json`` if one exists.
    3. Geometric fallback — assume the chip is centered on the canvas and
       map µm → px through ``microns_per_pixel``, applying ``rotation_deg``
       (which itself defaults to ``mea_rotation_deg`` from the rig config).

    Parameters
    ----------
    analysis_chunk : AnalysisChunk
        Source of ``vcd.get_electrode_map()``, ``canvas_size``,
        ``microns_per_pixel`` and (for calibration) the cells' EIs + RFs.
    mea_center_canvas_px : (x, y), optional
        Canvas-pixel coords of the MEA center (geometric fallback only).
    microns_per_pixel : float, optional
        Display scale (geometric fallback only).
    flip_y : bool
        Geometric fallback only. Negate y after rotation (math-y-up →
        image-y-down).
    rotation_deg : float, optional
        Geometric fallback only. Falls back further to ``mea_rotation_deg``
        from the rig config when ``None``.
    use_calibration : bool
        Try to load and apply a saved per-rig calibration. When False,
        the geometric fallback is used unconditionally.
    recompute_alignment : bool
        Override saved calibration with one fit from this chunk only.
    save_recomputed : bool
        When ``recompute_alignment=True``, persist the new fit to
        ``rig_calibrations/`` (default False — preview without writing).

    Returns
    -------
    np.ndarray
        ``(n_electrodes, 2)`` array of (x_canvas, y_canvas) coordinates.
    """
    em_um = analysis_chunk.vcd.get_electrode_map()  # (n, 2) in µm

    # Mode 1 / 2: similarity transform via a learned calibration
    if recompute_alignment or use_calibration:
        from retinanalysis.utils import rig_calibration as _rc
        calib = None
        if recompute_alignment:
            calib = _rc.fit_calibration_for_chunk(analysis_chunk)
            if save_recomputed:
                _rc.save_rig_calibration(calib)
        else:
            calib = _rc.load_rig_calibration(analysis_chunk.exp_name)
        if calib is not None:
            return _rc.apply_similarity_transform(em_um, calib)

    # Mode 3: geometric fallback (centered chip + rig-config rotation)
    if microns_per_pixel is None:
        microns_per_pixel = analysis_chunk.microns_per_pixel
    if mea_center_canvas_px is None:
        canvas_w, canvas_h = analysis_chunk.canvas_size
        mea_center_canvas_px = (canvas_w / 2.0, canvas_h / 2.0)
    if rotation_deg is None:
        # Lazy import to avoid pulling datajoint at module load time.
        from retinanalysis.utils.datajoint_utils import get_display_params_by_exp
        try:
            rotation_deg = float(get_display_params_by_exp(
                analysis_chunk.exp_name, verbose=False).get('mea_rotation_deg', 0.0))
        except Exception:
            rotation_deg = 0.0

    # Rotate chip coords by rotation_deg CCW (still in chip µm, centered on 0).
    if rotation_deg % 360 != 0:
        theta = np.deg2rad(rotation_deg)
        c, s = np.cos(theta), np.sin(theta)
        x_rot = em_um[:, 0] * c - em_um[:, 1] * s
        y_rot = em_um[:, 0] * s + em_um[:, 1] * c
    else:
        x_rot = em_um[:, 0]
        y_rot = em_um[:, 1]

    cx, cy = mea_center_canvas_px
    em_canvas = np.empty_like(em_um, dtype=float)
    em_canvas[:, 0] = cx + x_rot / microns_per_pixel
    em_canvas[:, 1] = cy + (-y_rot if flip_y else y_rot) / microns_per_pixel
    return em_canvas


def electrode_geometry(analysis_chunk) -> dict:
    """Empirical recording-array geometry, derived from ``vcd.get_electrode_map()``.

    The returned values come from the actual electrode positions Vision
    stored — no per-array hardcoding (e.g. "60 µm pitch", "512 channels")
    — so this works across array types and is the right primitive when
    you need to express thresholds in physical units.

    Returns
    -------
    dict
        ``n_electrodes`` : int
        ``pitch_um``     : float — median nearest-neighbor distance (µm)
        ``width_um``     : float — chip span in x (µm)
        ``height_um``    : float — chip span in y (µm)
        ``center_x_um``, ``center_y_um`` : float — mean electrode position
    """
    em = analysis_chunk.vcd.get_electrode_map()
    n = int(em.shape[0])
    if n < 2:
        pitch = float('nan')
    else:
        # Median nearest-neighbor distance — robust to chip edges and
        # any row jitter, no scipy dependency.
        diffs = em[:, None, :] - em[None, :, :]            # (n, n, 2)
        d2 = (diffs ** 2).sum(axis=-1)
        d2[np.eye(n, dtype=bool)] = np.inf
        pitch = float(np.median(np.sqrt(d2.min(axis=1))))
    return {
        'n_electrodes': n,
        'pitch_um': pitch,
        'width_um': float(em[:, 0].max() - em[:, 0].min()),
        'height_um': float(em[:, 1].max() - em[:, 1].min()),
        'center_x_um': float(em[:, 0].mean()),
        'center_y_um': float(em[:, 1].mean()),
    }


def cells_inside_array(
    analysis_chunk,
    max_dist_to_array_um: Optional[float] = None,
    max_dist_pitch: float = 2.0,
    *,
    use_calibration: bool = True,
    recompute_alignment: bool = False,
) -> List[int]:
    """Cell IDs whose RF center is within reach of the electrode array.

    Useful for restricting a mosaic plot to cells the array could have
    actually recorded — many STAs in a typing file are noisy or
    misclassified and have RF centers far off-array, which makes the
    mosaic *look* spatially much larger than the electrode footprint
    even though the calibration is correct.

    Threshold semantics: pass an absolute distance via
    ``max_dist_to_array_um`` to override the pitch-based default. When
    that's ``None`` (default), the cutoff is
    ``max_dist_pitch × electrode_geometry(analysis_chunk)['pitch_um']``
    so the same call works across array types without hardcoded pitch.

    Parameters
    ----------
    analysis_chunk : AnalysisChunk
    max_dist_to_array_um : float, optional
        Absolute cutoff in µm. Default ``None`` → use ``max_dist_pitch``.
    max_dist_pitch : float
        Cutoff in multiples of the empirical electrode pitch.
        Default 2.0.
    use_calibration, recompute_alignment :
        Forwarded to :func:`electrode_positions_canvas_px` so the
        on-array footprint reflects the same geometry the mosaic plot
        uses.
    """
    if max_dist_to_array_um is None:
        pitch = electrode_geometry(analysis_chunk)['pitch_um']
        max_dist_to_array_um = max_dist_pitch * pitch
    em_canvas = electrode_positions_canvas_px(
        analysis_chunk,
        use_calibration=use_calibration,
        recompute_alignment=recompute_alignment,
    )
    # Convert the µm threshold to canvas px so we can do all distance
    # math in one coord system. The calibrated electrode footprint
    # encodes µm → canvas-px scale already.
    if use_calibration or recompute_alignment:
        from retinanalysis.utils import rig_calibration as _rc
        calib = (_rc.load_rig_calibration(analysis_chunk.exp_name)
                 if not recompute_alignment
                 else _rc.fit_calibration_for_chunk(analysis_chunk))
        scale_px_per_um = calib.scale_px_per_um if calib is not None else (
            1.0 / analysis_chunk.microns_per_pixel)
    else:
        scale_px_per_um = 1.0 / analysis_chunk.microns_per_pixel
    cutoff_px = max_dist_to_array_um * scale_px_per_um

    pps = analysis_chunk.pixels_per_stixel
    keep: List[int] = []
    for cid in analysis_chunk.cell_ids:
        p = analysis_chunk.rf_params.get(int(cid))
        if p is None:
            continue
        cx = float(p['center_x']) * pps
        cy = float(p['center_y']) * pps
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        d = np.sqrt(((em_canvas - np.array([cx, cy])) ** 2).sum(axis=1)).min()
        if d <= cutoff_px:
            keep.append(int(cid))
    return keep


def _select_cells_by_type(analysis_chunk,
                          cell_types: Optional[Iterable[str]],
                          minimum_n: int,
                          typing_file: Optional[str]) -> Tuple[dict, list]:
    """Return ``({cell_type: [ids]}, sorted_cell_types)`` honoring filters.

    Mirrors the cell-selection logic in
    :meth:`AnalysisChunk.plot_rfs` but with no plotting side-effects, so
    it can be reused here without touching that method.
    """
    if typing_file is None:
        typing_file = analysis_chunk.typing_files[0] if analysis_chunk.typing_files else None
    if typing_file is None:
        return {}, []
    typing_idx = analysis_chunk.typing_files.index(typing_file)
    type_col = f'typing_file_{typing_idx}'

    df = analysis_chunk.df_cell_params
    if cell_types is None:
        present = sorted(df[type_col].unique())
    else:
        present = sorted(set(cell_types) & set(df[type_col].unique()))

    d_by_type = {}
    for ct in present:
        ids = df.query(f'{type_col} == @ct')['cell_id'].tolist()
        if len(ids) >= minimum_n:
            d_by_type[ct] = ids
    return d_by_type, sorted(d_by_type.keys())


def cell_activity_in_window(pipeline, epoch_index: int,
                            window_s: Tuple[float, float],
                            cell_types: Optional[Iterable[str]] = None,
                            cell_ids: Optional[Iterable[int]] = None,
                            std_scaling: float = 1.6):
    """Firing rate per cell in one epoch and time window, placed on the canvas.

    The join every "activity over the mosaic" figure needs: spike times are
    keyed by the protocol block's cell ids, receptive fields by the noise
    chunk's, and the two are different numbering. The ``noise_id`` coordinate
    that ``get_spike_xarr`` carries is the cluster match between them — the
    same relation as ``pipeline.match_dict``, already aligned to the spike
    array — so the rate and the receptive field of a row always describe one
    cell. Cells that never matched a noise cluster have nothing to place and
    are returned with ``center_x`` NaN rather than dropped, so a caller can
    report how many it could not draw instead of quietly showing fewer cells
    than it was asked for.

    Parameters
    ----------
    pipeline : MEAPipeline
        Supplies the response block, the analysis chunk holding ``rf_params``,
        and the id mapping between them.
    epoch_index : int
        Position of the epoch in the block, as the epoch table numbers them.
    window_s : (start, end)
        Time window within the epoch, in seconds from the epoch start. Rates
        are spikes in the window divided by its duration.
    cell_types : sequence[str], optional
        Restrict to these types. Default: every type in the block.
    cell_ids : sequence[int], optional
        Restrict to these protocol cell ids — pass the QC survivors here.
    std_scaling : float
        Receptive-field ellipse size in σ units, matching :func:`get_ells`.

    Returns
    -------
    pandas.DataFrame
        One row per cell: ``cell_id`` (protocol), ``noise_id``, ``cell_type``,
        ``n_spikes``, ``rate_hz``, ``center_x``/``center_y`` (canvas pixels),
        ``width``/``height``/``angle`` (the ellipse, canvas pixels), and
        ``spike_times_s`` — the spikes inside the window, in seconds from the
        epoch start, so a raster can be drawn from the same table that colored
        the mosaic.
    """
    import pandas as pd

    from retinanalysis.utils.vision_utils import get_spike_xarr

    t0, t1 = (float(w) for w in window_s)
    if not t1 > t0:
        raise ValueError(f'window_s must be (start, end) with end > start; '
                         f'got {window_s}')
    duration_s = t1 - t0

    response_block = pipeline.resp
    analysis_chunk = pipeline.analysis_chunk

    cell_ids = list(cell_ids) if cell_ids is not None else None
    cell_types = list(cell_types) if cell_types is not None else None
    xarr = get_spike_xarr(response_block, protocol_ids=cell_ids,
                          cell_types=cell_types)
    if epoch_index not in xarr.coords['epoch'].values:
        raise IndexError(f'epoch {epoch_index} is not in this block '
                         f'(0..{int(xarr.coords["epoch"].values.max())})')
    epoch = xarr.sel(epoch=epoch_index)

    rf_params = getattr(analysis_chunk, 'rf_params', {}) or {}
    pps = analysis_chunk.pixels_per_stixel

    rows = []
    for idx in range(epoch.sizes['cell_id']):
        one = epoch.isel(cell_id=idx)
        spikes_ms = np.asarray(one.values.item(), dtype=float)
        # Spike times are milliseconds from the epoch start throughout the
        # package; the window is stated in seconds because that is what a
        # person reads off a raster.
        in_window = spikes_ms[(spikes_ms >= t0 * 1000.0)
                              & (spikes_ms <= t1 * 1000.0)]

        noise_id = int(one.coords['noise_id'].item())
        p = rf_params.get(noise_id)
        row = {
            'cell_id': int(one.coords['cell_id'].item()),
            'noise_id': noise_id,
            'cell_type': str(one.coords['cell_type'].item()),
            'n_spikes': int(in_window.size),
            'rate_hz': in_window.size / duration_s,
            'spike_times_s': in_window / 1000.0,
        }
        if p is None:
            row.update(center_x=np.nan, center_y=np.nan,
                       width=np.nan, height=np.nan, angle=np.nan)
        else:
            row.update(
                center_x=float(p['center_x']) * pps,
                center_y=float(p['center_y']) * pps,
                width=float(p['std_x']) * std_scaling * pps,
                height=float(p['std_y']) * std_scaling * pps,
                angle=float(p['rot']),
            )
        rows.append(row)

    return pd.DataFrame(rows)


def plot_mosaic_activity(pipeline, epoch_index: int,
                         window_s: Tuple[float, float],
                         stim_frame: Optional[np.ndarray] = None,
                         cell_types: Optional[Iterable[str]] = None,
                         cell_ids: Optional[Iterable[int]] = None,
                         std_scaling: float = 1.6,
                         cmap: str = 'plasma',
                         stim_cmap: str = 'gray',
                         stim_vmin: Optional[float] = None,
                         stim_vmax: Optional[float] = None,
                         rate_vmax: Optional[float] = None,
                         rate_pct: float = 95.0,
                         aperture_diameter_px: Optional[float] = None,
                         zoom: bool = True,
                         pad_frac: float = 0.06,
                         raster_marker_size: float = 1.2,
                         title: Optional[str] = None,
                         figsize: Optional[Tuple[float, float]] = None):
    """Where the population fired, on the stimulus that drove it.

    Two panels of the same epoch and the same seconds. On the left, the
    mosaic over a reconstructed stimulus frame, each receptive field filled
    by its firing rate in the window and outlined in its cell type's color.
    On the right, the spikes those rates were counted from, one row per cell,
    sorted within each type by rate so the fill gradient on the left and the
    density gradient on the right run the same way.

    **The two panels are the argument for each other.** A rate map alone
    cannot show whether a bright cell fired steadily or in one burst, and a
    raster alone cannot show whether the responding cells sit under the
    grating or out in the black surround. Read together they answer both, for
    a window short enough to hold a few stimulus cycles.

    The stimulus frame is passed in rather than rendered here, because what a
    frame *is* differs per protocol. For ``variableMeanDriftingGrating`` it
    comes from :func:`retinanalysis.regen.variable_mean_drifting_grating
    .grating_frame`, which returns exactly the canvas-pixel array this
    expects. Omit it and the panel is just the mosaic.

    Parameters
    ----------
    pipeline : MEAPipeline
    epoch_index : int
        The epoch to show, numbered as the epoch table numbers it.
    window_s : (start, end)
        Seconds from the epoch start. This is the whole time axis of the
        raster and the interval the rates are computed over.
    stim_frame : np.ndarray, optional
        A canvas-pixel image, indexed ``[y, x]`` with y downwards. Drawn at
        the canvas extent, which is the coordinate system the receptive
        fields are already in.
    cell_types, cell_ids : optional
        Restrict the population, as in :func:`cell_activity_in_window`.
    cmap : str
        Sequential colormap for firing rate. The default is bright at the top
        end, which is what keeps cells legible over a dark stimulus frame.
    stim_vmin, stim_vmax : float, optional
        Display range for the frame. Default: the frame's own range. These
        protocols run at mean intensities as low as 0.03, so a fixed 0–1
        scale renders the stimulus as an unbroken black rectangle; scaling to
        the frame shows the geometry, and the title still reports the true
        intensities. Pass ``0`` and ``1`` for absolute luminance.
    rate_vmax : float, optional
        Top of the rate color scale. Default: the ``rate_pct`` percentile of
        the rates in this window.
    rate_pct : float
        Percentile setting the top of the color scale when ``rate_vmax`` is
        not given. Cells above it are drawn in the top color and the colorbar
        is marked with an arrow.
    aperture_diameter_px : float, optional
        Draw a circle of this diameter at the canvas center, marking the
        stimulus aperture. Cells outside it saw the background rather than
        the stimulus, which is the first thing to check before reading a low
        rate as a weak response.
    zoom : bool
        Frame the mosaic panel on the cells rather than the whole canvas.
    raster_marker_size : float
        Point size for the raster ticks. Lower it for dense populations.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.lines import Line2D

    from retinanalysis.utils.style import (NEUTRAL_GRAY, apply_publication_style,
                                           colors_for_celltypes)

    apply_publication_style()

    activity = cell_activity_in_window(
        pipeline, epoch_index, window_s,
        cell_types=cell_types, cell_ids=cell_ids, std_scaling=std_scaling,
    )
    if activity.empty:
        raise ValueError('No cells matched the requested types and ids.')

    t0, t1 = (float(w) for w in window_s)

    # Panel order follows the caller's cell_types when given, so a figure
    # keeps the same type order as the tables beside it.
    if cell_types is not None:
        order = [ct for ct in cell_types if (activity['cell_type'] == ct).any()]
    else:
        order = sorted(activity['cell_type'].unique())
    type_colors = colors_for_celltypes(order)

    drawable = activity[activity['center_x'].notna()]
    n_no_rf = len(activity) - len(drawable)

    # Firing rates across a population are strongly skewed — a handful of
    # cells fire an order of magnitude above the median — so scaling the
    # colormap to the maximum leaves almost every cell in the bottom of the
    # range and the map reads as uniformly dark. The default tops out at a
    # high percentile instead and the colorbar is drawn with an arrow, so the
    # cells above it are marked as clipped rather than passed off as equal to
    # the ceiling.
    saturated = False
    if rate_vmax is None:
        rate_vmax = float(np.percentile(activity['rate_hz'], rate_pct))
        saturated = bool((activity['rate_hz'] > rate_vmax).any())
    else:
        saturated = bool((activity['rate_hz'] > float(rate_vmax)).any())
    if rate_vmax <= 0:                    # a window where almost nothing fired
        rate_vmax = max(float(activity['rate_hz'].max()), 1.0)
        saturated = False
    norm = plt.Normalize(vmin=0.0, vmax=float(rate_vmax))
    rate_cmap = plt.get_cmap(cmap)

    canvas_w, canvas_h = pipeline.analysis_chunk.canvas_size

    if figsize is None:
        figsize = (13.5, 5.4)
    fig, (ax_mosaic, ax_raster) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1.0, 1.15]})

    # --- Left: stimulus + mosaic ------------------------------------------
    if stim_frame is not None:
        ax_mosaic.imshow(stim_frame, cmap=stim_cmap,
                         vmin=stim_vmin, vmax=stim_vmax,
                         extent=(0.0, float(canvas_w), float(canvas_h), 0.0),
                         interpolation='nearest', zorder=0)

    if aperture_diameter_px:
        ax_mosaic.add_patch(Ellipse(
            xy=(canvas_w / 2.0, canvas_h / 2.0),
            width=float(aperture_diameter_px), height=float(aperture_diameter_px),
            facecolor='none', edgecolor='white', linestyle='--',
            linewidth=0.8, alpha=0.6, zorder=1))

    # Quietest first, so the cells that carry the response end up on top
    # rather than hidden under a neighbor that did nothing.
    for _, cell in drawable.sort_values('rate_hz').iterrows():
        edge = type_colors.get(cell['cell_type'], NEUTRAL_GRAY)
        ax_mosaic.add_patch(Ellipse(
            xy=(cell['center_x'], cell['center_y']),
            width=cell['width'], height=cell['height'], angle=cell['angle'],
            facecolor=rate_cmap(norm(cell['rate_hz'])),
            edgecolor=edge, linewidth=0.7, alpha=0.85, zorder=2))

    if zoom and len(drawable):
        radii = np.maximum(drawable['width'], drawable['height']).to_numpy() / 2
        x_lo = float((drawable['center_x'].to_numpy() - radii).min())
        x_hi = float((drawable['center_x'].to_numpy() + radii).max())
        y_lo = float((drawable['center_y'].to_numpy() - radii).min())
        y_hi = float((drawable['center_y'].to_numpy() + radii).max())
        pad = pad_frac * max(x_hi - x_lo, y_hi - y_lo)
        x_lo, x_hi, y_lo, y_hi = x_lo - pad, x_hi + pad, y_lo - pad, y_hi + pad
    else:
        x_lo, x_hi, y_lo, y_hi = 0.0, float(canvas_w), 0.0, float(canvas_h)
    ax_mosaic.set_xlim(x_lo, x_hi)
    ax_mosaic.set_ylim(y_hi, y_lo)          # canvas y runs downwards
    ax_mosaic.set_aspect('equal')
    ax_mosaic.set_xlabel('canvas x (pix)')
    ax_mosaic.set_ylabel('canvas y (pix)')

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=rate_cmap),
                        ax=ax_mosaic, fraction=0.046, pad=0.02,
                        extend='max' if saturated else 'neither')
    cbar.set_label(f'rate over {t0:g}–{t1:g} s (Hz)')

    # Under the panel rather than inside it: a corner legend lands on the
    # mosaic, and these populations reach the corners.
    ax_mosaic.legend(
        handles=[Line2D([0], [0], marker='o', linestyle='none', markersize=6,
                        markerfacecolor='none', markeredgecolor=type_colors[ct],
                        label=f'{ct} (n={int((activity["cell_type"] == ct).sum())})')
                 for ct in order],
        loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=min(len(order), 4),
        fontsize=8, frameon=False)

    # --- Right: the spikes those rates came from --------------------------
    ordered = pd.concat(
        [activity[activity['cell_type'] == ct].sort_values('rate_hz')
         for ct in order],
        ignore_index=True) if order else activity

    boundaries, row = [], 0
    for ct in order:
        block = ordered[ordered['cell_type'] == ct]
        color = type_colors.get(ct, NEUTRAL_GRAY)
        for _, cell in block.iterrows():
            spikes = cell['spike_times_s']
            if len(spikes):
                ax_raster.plot(spikes, np.full(len(spikes), row), '|',
                               color=color, markersize=raster_marker_size * 3,
                               markeredgewidth=raster_marker_size)
            row += 1
        boundaries.append((ct, row, len(block)))
        if ct != order[-1]:
            ax_raster.axhline(row - 0.5, color=NEUTRAL_GRAY,
                              linewidth=0.5, alpha=0.5)

    ax_raster.set_xlim(t0, t1)
    ax_raster.set_ylim(-0.5, max(row - 0.5, 0.5))
    ax_raster.set_xlabel('time in epoch (s)')
    ax_raster.set_ylabel('cell, by type then rate')
    # Label each type block at its middle rather than every cell on the axis.
    ticks, labels, start = [], [], 0
    for ct, end, n in boundaries:
        if n:
            ticks.append((start + end - 1) / 2.0)
            labels.append(ct)
        start = end
    ax_raster.set_yticks(ticks)
    ax_raster.set_yticklabels(labels, fontsize=8)

    if title is None:
        title = (f'epoch {epoch_index}, {t0:g}–{t1:g} s — '
                 f'{len(drawable)} cells on the mosaic')
        if n_no_rf:
            title += f' ({n_no_rf} more with no matched RF, in the raster only)'
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_stim_with_mosaic(
    stim_frame: np.ndarray,
    analysis_chunk,
    cell_types: Optional[List[str]] = None,
    minimum_n: int = 1,
    std_scaling: float = 1.6,
    frame_extent: Optional[Tuple[float, float, float, float]] = None,
    ax: Optional[Axes] = None,
    typing_file: Optional[str] = None,
    title: str = '',
    cmap: str = 'gray',
    fill_alpha: float = 0.0,
    edge_alpha: float = 0.9,
    edge_linewidth: float = 1.2,
    legend: bool = True,
    show_electrodes: bool = False,
    electrode_kwargs: Optional[dict] = None,
    mea_center_canvas_px: Optional[Tuple[float, float]] = None,
    flip_electrode_y: bool = True,
    electrode_rotation_deg: Optional[float] = None,
    use_calibration: bool = True,
    recompute_alignment: bool = False,
    save_recomputed: bool = False,
    restrict_to_array_um: Optional[float] = None,
    restrict_to_array_pitch: Optional[float] = None,
    zoom_to_array: bool = False,
    zoom_pad_um: Optional[float] = None,
    zoom_pad_pitch: float = 2.0,
    **imshow_kwargs,
) -> Axes:
    """Overlay the cell-type mosaic on a stimulus frame.

    The frame is drawn at the rig's canvas extent (``analysis_chunk.canvas_size``
    in pixels), and RF ellipses (1.6 σ by default) are placed in matching
    canvas-pixel coordinates. Each cell type gets its own color; cells with
    fewer than ``minimum_n`` instances are skipped.

    Parameters
    ----------
    stim_frame : np.ndarray
        2D image (H, W) — typically one frame from a ``regen`` Dataset, or
        a base image from ``EyeMovementTrajectoryAlternatingBackground``.
    analysis_chunk : AnalysisChunk
        Source of the mosaic. Its ``rf_params``, ``pixels_per_stixel``,
        ``canvas_size`` and ``typing_files`` are all used.
    cell_types : list[str], optional
        Restrict to these types. Default: every type in the typing file.
    minimum_n : int
        Skip types with fewer cells than this.
    std_scaling : float
        Ellipse size in σ units (1.6 σ ≈ 1 RF radius — the same convention
        as :func:`get_ells` / :meth:`AnalysisChunk.plot_rfs`).
    frame_extent : (xmin, xmax, ymax, ymin), optional
        Matplotlib ``imshow`` extent. Default = ``(0, canvas_w, canvas_h, 0)``
        so the frame fills the canvas at the rig's pixel resolution.
    ax : matplotlib Axes, optional
        Axis to draw into. A new figure is created if omitted.
    typing_file : str, optional
        Pick a specific classification file. Default: first one in
        ``analysis_chunk.typing_files``.
    fill_alpha, edge_alpha, edge_linewidth : float
        Visual controls for the ellipses. The defaults draw outlines only
        (``fill_alpha=0``) so the stim shows through.
    legend : bool
        Add a legend mapping color → cell type.
    show_electrodes : bool
        Overlay the MEA electrode grid (Litke 512 array) on the plot.
    electrode_kwargs : dict, optional
        Forwarded to ``ax.scatter`` for the electrode dots. Defaults to
        ``dict(s=4, c='white', edgecolors='black', linewidths=0.3,
        alpha=0.6, zorder=5)`` so dots stay readable on either a
        natural image or a noise frame.
    mea_center_canvas_px : (x, y), optional
        Canvas-pixel coords of the MEA chip center. Defaults to canvas
        center, which is the standard co-registration.
    flip_electrode_y : bool
        Flip electrode y when overlaying (MEA chip uses math-up y, canvas
        uses image-down y). Default True. Set False if your rig stores
        electrode positions already in image-down y.
    electrode_rotation_deg : float, optional
        Geometric-fallback rotation (degrees CCW). Only used when no
        learned calibration is available. When ``None`` (default) the
        value is resolved per rig via :func:`get_display_params_by_exp`.
    use_calibration : bool
        Try the learned per-rig calibration first
        (``rig_calibrations/rig_<id>.json``). When False, always use the
        geometric fallback. Default True.
    recompute_alignment : bool
        Fit a fresh per-chunk calibration from EI-soma vs. STA centers
        and use it for this plot. Overrides any saved calibration.
        Default False.
    save_recomputed : bool
        When ``recompute_alignment=True``, also persist the new fit to
        ``rig_calibrations/``. Default False (preview only).
    **imshow_kwargs : dict
        Forwarded to ``ax.imshow`` (e.g. ``vmin``, ``vmax``).
    """
    # --- Select cells by type
    d_by_type, present_types = _select_cells_by_type(
        analysis_chunk, cell_types, minimum_n, typing_file,
    )

    # Optional: drop cells whose RF center isn't near any electrode.
    # Many typed cells have noisy STAs that put their RF far off-array;
    # plotting them makes the mosaic look much larger than the chip
    # footprint even though the geometry is correct.
    if (restrict_to_array_um is not None or restrict_to_array_pitch is not None) and d_by_type:
        keep_ids = set(cells_inside_array(
            analysis_chunk,
            max_dist_to_array_um=restrict_to_array_um,
            max_dist_pitch=(restrict_to_array_pitch
                            if restrict_to_array_pitch is not None
                            else 2.0),
            use_calibration=use_calibration,
            recompute_alignment=recompute_alignment,
        ))
        d_by_type = {ct: [c for c in ids if c in keep_ids]
                     for ct, ids in d_by_type.items()}
        d_by_type = {ct: ids for ct, ids in d_by_type.items()
                     if len(ids) >= minimum_n}
        present_types = sorted(d_by_type.keys())

    if not d_by_type:
        if cell_types:
            print(f'[plot_stim_with_mosaic] no cells matched cell_types={cell_types} '
                  f'with minimum_n={minimum_n} in {analysis_chunk.exp_name}/{analysis_chunk.chunk_name}')
        # Still draw the frame so the caller sees the stim.

    # --- Ellipses in canvas pixels (already y-flipped during rf_params build)
    d_ells_by_type, _ = get_ells(
        analysis_chunk, d_by_type, std_scaling=std_scaling, units='pixels',
    )

    # --- Axis + frame
    canvas_w, canvas_h = analysis_chunk.canvas_size
    if ax is None:
        # Aspect-preserving figure size; cap at ~10 inches wide.
        fig_w = min(10.0, canvas_w / 80.0)
        fig_h = fig_w * canvas_h / canvas_w
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if frame_extent is None:
        # imshow extent convention: (left, right, bottom, top). For
        # origin='upper' (default), bottom > top so the y-axis points
        # downward. We pass (0, canvas_w, canvas_h, 0) to match the
        # rf_params y-flip convention.
        frame_extent = (0.0, float(canvas_w), float(canvas_h), 0.0)

    ax.imshow(stim_frame, extent=frame_extent, cmap=cmap, **imshow_kwargs)

    # --- Overlay ellipses with per-type colors (Okabe-Ito canonical map)
    type_color_map = colors_for_celltypes(present_types)
    legend_handles = []
    for idx, ct in enumerate(present_types):
        color = type_color_map[ct]
        for cell_id, ell in d_ells_by_type[ct].items():
            # get_ells returns one Ellipse per cell already configured with
            # facecolor=alpha. We re-style here so user-supplied alpha/lw
            # take effect; the original objects are reused (one ax per call).
            patch = Ellipse(
                xy=ell.center,
                width=ell.width, height=ell.height, angle=ell.angle,
                facecolor=color if fill_alpha > 0 else 'none',
                edgecolor=color,
                alpha=max(fill_alpha, edge_alpha),
                linewidth=edge_linewidth,
                fill=fill_alpha > 0,
            )
            ax.add_patch(patch)
        legend_handles.append(Ellipse(
            xy=(0, 0), width=1, height=1, angle=0,
            facecolor=color if fill_alpha > 0 else 'none',
            edgecolor=color, linewidth=edge_linewidth,
            label=f'{ct} (n={len(d_ells_by_type[ct])})',
        ))

    # --- Optional electrode overlay (MEA chip → canvas pixels)
    if show_electrodes:
        em_canvas = electrode_positions_canvas_px(
            analysis_chunk,
            mea_center_canvas_px=mea_center_canvas_px,
            flip_y=flip_electrode_y,
            rotation_deg=electrode_rotation_deg,
            use_calibration=use_calibration,
            recompute_alignment=recompute_alignment,
            save_recomputed=save_recomputed,
        )
        defaults = dict(s=4, c='white', edgecolors='black',
                        linewidths=0.3, alpha=0.6, zorder=5)
        if electrode_kwargs:
            defaults.update(electrode_kwargs)
        ax.scatter(em_canvas[:, 0], em_canvas[:, 1], **defaults)

    # --- Limits / labels
    if zoom_to_array:
        # Crop axes to the electrode footprint + pad. Useful when the
        # array covers only part of the canvas (typical) and the rest is
        # empty background pulling the plot scale away from the cells.
        em_z = electrode_positions_canvas_px(
            analysis_chunk,
            mea_center_canvas_px=mea_center_canvas_px,
            flip_y=flip_electrode_y,
            rotation_deg=electrode_rotation_deg,
            use_calibration=use_calibration,
            recompute_alignment=recompute_alignment,
            save_recomputed=False,
        )
        # Convert µm padding → canvas px using the rig scale.
        if zoom_pad_um is None:
            zoom_pad_um = (zoom_pad_pitch
                           * electrode_geometry(analysis_chunk)['pitch_um'])
        pad_px = float(zoom_pad_um) / float(analysis_chunk.microns_per_pixel)
        x0 = max(0.0, em_z[:, 0].min() - pad_px)
        x1 = min(float(canvas_w), em_z[:, 0].max() + pad_px)
        y0 = max(0.0, em_z[:, 1].min() - pad_px)
        y1 = min(float(canvas_h), em_z[:, 1].max() + pad_px)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
    else:
        ax.set_xlim(0, canvas_w)
        ax.set_ylim(canvas_h, 0)
    ax.set_aspect('equal')
    ax.set_xlabel('canvas x (pix)')
    ax.set_ylabel('canvas y (pix)')
    if title:
        ax.set_title(title)
    elif d_by_type:
        ax.set_title(f'{analysis_chunk.exp_name} / {analysis_chunk.chunk_name} — '
                     f'{sum(len(v) for v in d_by_type.values())} cells')

    if legend and legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.7)

    return ax
