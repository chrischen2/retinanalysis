"""Regen for ``EyeMovementTrajectoryAlternatingBackground``.

Source: ``turner-package/+edu/+washington/+riekelab/+turner/+protocols/
EyeMovementTrajectoryAlternatingBackground.m``

Per-epoch parameters stored in the H5 (so always available):
    currentImageName, currentP0, xTraj, yTraj,
    backgroundIntensity, currentImageMean, currentBackgroundScale,
    currentStimSet, currentImageSet, randomSeed
Block-level parameters:
    preTime, stimTime, tailTime, D, backgroundScale, apertureDiameter,
    patchMean

Because xTraj / yTraj are already in the H5, the only external resource
needed is the natural-image library (``imk{name}.iml`` files under
``turner-package/resources/VHsubsample_20160105/``). When the repo is found
but the .iml files are missing — which is the common case, since the VH
images are large and aren't committed — we still return trajectories +
metadata, just with an empty ``base_images`` dict.
"""

from __future__ import annotations

import os
import numpy as np
import xarray as xr
from typing import Any, Optional

from retinanalysis.config.settings import find_protocol_repo
from . import register
from ._io import load_iml_image, find_image_in_library

PROTOCOL_NAME = 'edu.washington.riekelab.turner.protocols.EyeMovementTrajectoryAlternatingBackground'

# VH images: physical pixel scale on the prep (microns per VH pixel), from the
# MATLAB source (lines 90-91): "3.3 for 1 arcmin/VH pixel based on 198 um/degree
# on monkey retina".
VH_UM_PER_PIXEL = 3.3
VH_IMAGE_SHAPE = (1024, 1536)  # rows, cols — post-transpose


def regen(stim_block: Any, verbose: bool = True,
          repo_name: str = 'turner-package',
          image_set: Optional[str] = None) -> xr.Dataset:
    """Build an xarray.Dataset of trajectories + metadata for this protocol.

    Returns a Dataset with per-epoch trajectories (``xTraj_um``, ``yTraj_um``)
    plus image-positioning metadata (``currentP0_x``, ``currentP0_y``,
    ``image_name``, ``backgroundIntensity``, ...). When the turner-package
    repo + .iml image files are findable, also attaches loaded base images
    as ``ds.attrs['base_images']`` (mapping image_name → uint8 array).

    The Dataset is always returned — missing resources only affect what's
    in ``base_images``.
    """
    df = stim_block.df_epochs
    n_epochs = len(df)

    # --- Per-epoch arrays straight from H5 ---
    image_names = [str(p.get('currentImageName', '')) for p in df['epoch_parameters']]
    bg_intensity = df['backgroundIntensity'].to_numpy(dtype=float) \
        if 'backgroundIntensity' in df.columns \
        else np.array([p.get('backgroundIntensity', np.nan) for p in df['epoch_parameters']], dtype=float)
    bg_scale = df['currentBackgroundScale'].to_numpy(dtype=float) \
        if 'currentBackgroundScale' in df.columns \
        else np.array([p.get('currentBackgroundScale', np.nan) for p in df['epoch_parameters']], dtype=float)

    # currentImageMean isn't always saved as an epoch parameter (varies by
    # protocol version). Derive it from the algebraic identity that the
    # protocol enforces: backgroundIntensity = imageMean * currentBackgroundScale.
    raw_image_mean = np.array(
        [p.get('currentImageMean') if p.get('currentImageMean') is not None else np.nan
         for p in df['epoch_parameters']],
        dtype=float,
    )
    derived = np.where(bg_scale != 0, bg_intensity / np.where(bg_scale != 0, bg_scale, 1.0), np.nan)
    image_mean = np.where(np.isnan(raw_image_mean), derived, raw_image_mean)
    p0_arr = np.array(
        [np.asarray(p.get('currentP0', (np.nan, np.nan)), dtype=float).ravel()
         for p in df['epoch_parameters']],
    )  # shape (n_epochs, 2)

    # xTraj / yTraj are stored per epoch in microns (despite a misleading
    # comment in the protocol source — the values come from cumsum of
    # D*randn where D is in um).
    traj_x = [np.asarray(p['xTraj'], dtype=float).ravel() for p in df['epoch_parameters']]
    traj_y = [np.asarray(p['yTraj'], dtype=float).ravel() for p in df['epoch_parameters']]
    traj_len = max(len(t) for t in traj_x)

    # Pad to the max length (within a single block these should all match,
    # but be defensive in case epochs were dropped mid-run with different
    # stimTime).
    xTraj_um = np.full((n_epochs, traj_len), np.nan, dtype=float)
    yTraj_um = np.full((n_epochs, traj_len), np.nan, dtype=float)
    for i, (tx, ty) in enumerate(zip(traj_x, traj_y)):
        xTraj_um[i, :len(tx)] = tx
        yTraj_um[i, :len(ty)] = ty

    # Time axis for trajectory samples (one sample per monitor frame). The
    # per-epoch param monitorRefreshRate is the most authoritative source.
    first_params = df['epoch_parameters'].iloc[0] if n_epochs else {}
    refresh_hz = float(first_params.get('monitorRefreshRate', 60.0) or 60.0)
    microns_per_pixel = first_params.get('micronsPerPixel', None)
    canvas_size = first_params.get('canvasSize', None)
    time_s = np.arange(traj_len) / refresh_hz

    # --- Build the Dataset ---
    ds = xr.Dataset(
        data_vars={
            'xTraj_um': (('epoch', 'time'), xTraj_um),
            'yTraj_um': (('epoch', 'time'), yTraj_um),
            'currentP0_x_vhpix': (('epoch',), p0_arr[:, 0]),
            'currentP0_y_vhpix': (('epoch',), p0_arr[:, 1]),
            'currentImageMean': (('epoch',), image_mean),
            'currentBackgroundScale': (('epoch',), bg_scale),
            'backgroundIntensity': (('epoch',), bg_intensity),
            'image_name': (('epoch',), np.array(image_names, dtype=object)),
        },
        coords={
            'epoch': np.arange(n_epochs),
            'time_s': ('time', time_s),
        },
    )

    # Block-level / display-level metadata
    block_params = getattr(stim_block, 'd_epoch_block_params', {}) or {}
    ds.attrs.update({
        'protocol_name': PROTOCOL_NAME,
        'exp_name': stim_block.exp_name,
        'datafile_name': stim_block.datafile_name,
        'preTime_ms': block_params.get('preTime', None),
        'stimTime_ms': block_params.get('stimTime', None),
        'tailTime_ms': block_params.get('tailTime', None),
        'D_um': block_params.get('D', None),
        'apertureDiameter_um': block_params.get('apertureDiameter', None),
        'backgroundScale_list': block_params.get('backgroundScale', None),
        'patchMean': block_params.get('patchMean', None),
        'monitor_refresh_rate_hz': refresh_hz,
        'vh_um_per_pixel': VH_UM_PER_PIXEL,
        'microns_per_pixel': microns_per_pixel,
        'canvas_size_pix': canvas_size,
    })

    # --- Attempt to attach base images ---
    repo_root = find_protocol_repo(repo_name)
    base_images: dict = {}
    if repo_root is None:
        if verbose:
            print(f'[regen] repo "{repo_name}" not found under PROTOCOL_REPOS_ROOT — '
                  f'returning trajectories + metadata only (no base images).')
    else:
        if verbose:
            print(f'[regen] found repo: {repo_root}')
        # Pick image set: prefer per-epoch value, fall back to MATLAB default.
        if image_set is None:
            sample = df['epoch_parameters'].iloc[0].get('currentImageSet', '') if n_epochs else ''
            image_set = sample.strip('\\/') or 'VHsubsample_20160105'

        unique_names = sorted(set(n for n in image_names if n))
        missing = []
        for name in unique_names:
            path = find_image_in_library(repo_root, name, image_set=image_set)
            if path is None:
                missing.append(name)
                continue
            base_images[name] = load_iml_image(path)

        if verbose:
            if base_images:
                print(f'[regen] loaded {len(base_images)} base image(s) from '
                      f'{os.path.join(repo_root, "resources", image_set)} '
                      f'(missing: {missing if missing else "none"})')
            else:
                print(f'[regen] no .iml files found under {repo_root}/resources/{image_set}/ — '
                      f'returning trajectories + metadata only.')
                print(f'[regen]   (looked for: {[f"imk{n}.iml" for n in unique_names]})')

    ds.attrs['base_images'] = base_images
    ds.attrs['image_set'] = image_set if 'image_set' in locals() else None
    return ds


register(PROTOCOL_NAME, regen)


def render_displayed_canvas(
    stim_ds: xr.Dataset,
    epoch: int,
    frame: Optional[int] = None,
) -> np.ndarray:
    """Render the actual canvas-pixel frame as displayed on the monitor.

    Composites the per-epoch natural image (scaled by
    ``vh_um_per_pixel/microns_per_pixel`` to canvas pixels), centered at
    ``currentP0`` translated to canvas, optionally shifted by the eye
    trajectory at ``frame``. Pixels outside the image extent are filled
    with ``backgroundIntensity``. The result is clipped to the rig's
    ``canvasSize`` — i.e. exactly what was visible to the retina.

    Parameters
    ----------
    stim_ds : xr.Dataset
        Output of :func:`regen` for this protocol. Must include
        ``base_images`` in attrs (i.e. the turner-package resources were
        found).
    epoch : int
        Index along the ``epoch`` dimension.
    frame : int, optional
        Frame index along ``time``. ``None`` (default) returns the
        pre-trajectory frame (image at ``currentP0`` only). For an
        actual frame use ``frame=k`` to apply ``xTraj_um[k]``,
        ``yTraj_um[k]`` (in microns) as the eye-position shift.

    Returns
    -------
    np.ndarray
        ``(canvas_h, canvas_w)`` uint8 array. Plug this into
        :func:`plot_stim_with_mosaic` with the default ``frame_extent``.
    """
    base_images = stim_ds.attrs.get('base_images') or {}
    if not base_images:
        raise ValueError(
            'render_displayed_canvas: stim_ds has no base_images — the '
            'protocol repo or .iml files were not findable at regen time.'
        )

    canvas_size = stim_ds.attrs.get('canvas_size_pix')
    mu_per_pix = stim_ds.attrs.get('microns_per_pixel')
    vh_um_per_pix = stim_ds.attrs.get('vh_um_per_pixel', VH_UM_PER_PIXEL)
    if canvas_size is None or mu_per_pix is None:
        raise ValueError(
            'render_displayed_canvas: stim_ds attrs missing canvas_size_pix '
            'or microns_per_pixel; regen() must have been called against a '
            'StimBlock built on a known rig.'
        )
    canvas_w, canvas_h = int(round(canvas_size[0])), int(round(canvas_size[1]))
    scale = float(vh_um_per_pix) / float(mu_per_pix)  # canvas pix per VH pix

    img_name = str(stim_ds.image_name.values[epoch])
    img = base_images[img_name]
    img_h, img_w = img.shape
    img_h_canvas = img_h * scale
    img_w_canvas = img_w * scale

    # Image-center in canvas pixels. Mirrors the MATLAB createPresentation
    # logic (lines 150-160): center on canvas/2 then shift by the
    # P0-relative-to-image-center offset, with x sign-flipped to match
    # the protocol's "scene right ⇒ position left" convention.
    p0x = float(stim_ds.currentP0_x_vhpix.values[epoch])
    p0y = float(stim_ds.currentP0_y_vhpix.values[epoch])
    cx_base = canvas_w / 2.0 - (p0x - img_w / 2.0) * scale
    cy_base = canvas_h / 2.0 + (p0y - img_h / 2.0) * scale

    # Apply eye-movement shift if a frame index is given. xTraj_um is in
    # microns; convert to canvas pixels (negate to match the protocol's
    # "scene right ⇒ position left" — see line 124 of the .m).
    if frame is not None:
        dx_canvas = -float(stim_ds.xTraj_um.values[epoch, int(frame)]) / float(mu_per_pix)
        dy_canvas = -float(stim_ds.yTraj_um.values[epoch, int(frame)]) / float(mu_per_pix)
    else:
        dx_canvas = 0.0
        dy_canvas = 0.0

    cx = cx_base + dx_canvas
    cy = cy_base + dy_canvas

    # Scale the source image to canvas resolution via NEAREST/BILINEAR. We
    # prefer scipy.ndimage.zoom to avoid a hard OpenCV dep here.
    from scipy.ndimage import zoom
    img_scaled = zoom(img.astype(np.float32), zoom=scale, order=1)
    # zoom() rounding may produce a slightly-off shape; reconcile.
    new_h, new_w = img_scaled.shape

    # Background fill = backgroundIntensity * 255, clipped to [0,255].
    bg_intensity = float(stim_ds.backgroundIntensity.values[epoch])
    bg_uint8 = int(np.clip(round(255.0 * bg_intensity), 0, 255))

    canvas = np.full((canvas_h, canvas_w), bg_uint8, dtype=np.uint8)

    # Place the scaled image with its center at (cx, cy). Compute source
    # and destination slices that lie within both arrays.
    img_left = cx - new_w / 2.0
    img_top = cy - new_h / 2.0

    src_left = int(max(0, np.floor(-img_left)))
    src_top = int(max(0, np.floor(-img_top)))
    src_right = int(min(new_w, np.ceil(canvas_w - img_left)))
    src_bottom = int(min(new_h, np.ceil(canvas_h - img_top)))

    if src_right > src_left and src_bottom > src_top:
        dst_left = int(max(0, round(img_left)))
        dst_top = int(max(0, round(img_top)))
        dst_right = dst_left + (src_right - src_left)
        dst_bottom = dst_top + (src_bottom - src_top)
        # Clamp again in case rounding pushed past canvas edge.
        dst_right = min(dst_right, canvas_w)
        dst_bottom = min(dst_bottom, canvas_h)
        sub = np.clip(img_scaled[src_top:src_top + (dst_bottom - dst_top),
                                 src_left:src_left + (dst_right - dst_left)], 0, 255)
        canvas[dst_top:dst_bottom, dst_left:dst_right] = sub.astype(np.uint8)

    return canvas
