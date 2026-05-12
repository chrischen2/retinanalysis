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

    # --- Overlay ellipses with per-type colors
    legend_handles = []
    for idx, ct in enumerate(present_types):
        color = f'C{idx}'
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
