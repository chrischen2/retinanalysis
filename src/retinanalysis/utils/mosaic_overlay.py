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
