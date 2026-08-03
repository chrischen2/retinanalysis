"""Canvas-space stimulus primitives shared by Stage protocols.

Rebuilding what was on the display is a per-protocol job — which parameters
exist and what they are called differs every time. But the *drawing* is not:
Symphony protocols in these packages compose a small, repeating vocabulary of
Stage objects, and gratings behind a circular aperture account for a large
share of them. This module holds that vocabulary in **canvas pixels**, so a
protocol module is left doing only the part that is genuinely its own —
reading its parameters out of the recorded epochs.

Canvas pixels are the useful frame because the mosaic is already in them.
``AnalysisChunk`` stores RF centers in stixels of the noise grid, y-flipped to
agree with ``imshow``'s top-left origin, and ``pixels_per_stixel`` scales those
to canvas pixels — the same units MATLAB specifies stimulus geometry in. An
array from here therefore overlays a mosaic with no fitted parameter; see
:mod:`retinanalysis.utils.mosaic_overlay`, whose *electrode* overlay is the one
that does need a calibration.

Everything returns arrays indexed ``[y, x]`` with y increasing downwards, ready
for ``imshow(..., extent=(0, canvas_w, canvas_h, 0))``.

This module is deliberately numpy-only. It is imported by protocol
reconstruction code that has no reason to pull in OpenCV or the DataJoint
classes, which is what depending on :mod:`retinanalysis.utils.regen` would cost.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


__all__ = [
    'um_to_pixels',
    'canvas_offsets',
    'project_on_axis',
    'grating_luminance',
    'apply_circular_aperture',
]


def um_to_pixels(um, microns_per_pixel: float):
    """Microns to display pixels, **rounded**, as the rig's device converts.

    This is ``um2pix`` on the LcrVideo stage device: ``round(um /
    micronsPerPixel)``. The rounding is not incidental and must be reproduced.
    A 50 µm bar at 3.8 µm/pixel is 13 pixels, not 13.16, and the spatial
    frequency a protocol derives from it follows the rounded value — carry the
    exact quotient instead and the reconstruction drifts out of register by
    the edge of a large aperture.

    Accepts a scalar or an array. ``retinanalysis.utils.regen
    .lcr_video_device_um_to_pix`` is the same function under its older name and
    delegates here.
    """
    out = np.round(np.asarray(um, dtype=float) / float(microns_per_pixel))
    return int(out) if out.ndim == 0 else out.astype(int)


def canvas_offsets(canvas_w: int, canvas_h: int, downsample: int = 1,
                   center: Optional[Tuple[float, float]] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel ``(dx, dy)`` offsets from the canvas center.

    Sampling is at pixel *centers* (``0.5, 1.5, …``), so a feature the
    protocol places at the canvas center falls between the two middle columns
    rather than on an arbitrary one of them.

    Parameters
    ----------
    canvas_w, canvas_h : int
        Canvas size in pixels, as the rig reports it (``canvasSize``).
    downsample : int
        Take every ``downsample``-th pixel. 1 (default) renders full
        resolution, which is 800×600 on these rigs and cheap.
    center : (x, y), optional
        Origin to measure from. Defaults to the canvas center, which is where
        Stage protocols position things (``p.setBackgroundColor``-relative
        geometry is almost always ``canvasSize/2``).

    Returns
    -------
    (dx, dy)
        Two ``[y, x]``-indexed arrays of offsets in pixels.
    """
    step = max(1, int(downsample))
    cx, cy = center if center is not None else (canvas_w / 2.0, canvas_h / 2.0)
    xs = np.arange(0, canvas_w, step, dtype=float) + 0.5 * step
    ys = np.arange(0, canvas_h, step, dtype=float) + 0.5 * step
    x, y = np.meshgrid(xs, ys)
    return x - cx, y - cy


def project_on_axis(dx: np.ndarray, dy: np.ndarray,
                    orientation_deg: float) -> np.ndarray:
    """Signed distance along an oriented axis, for a pattern that varies in 1-D.

    A grating is constant along its bars and varies only across them, so this
    reduces the 2-D canvas to the one coordinate that matters.

    Canvas y runs *downwards*, so a positive ``orientation_deg`` reads as a
    clockwise rotation on the display. That is the same sense Stage's
    ``orientation`` property has, so a protocol's value can be passed straight
    through.
    """
    theta = np.deg2rad(float(orientation_deg))
    return dx * np.cos(theta) + dy * np.sin(theta)


def grating_luminance(axis_px: np.ndarray, spatial_freq_cyc_per_px: float,
                      mean_intensity: float, contrast: float,
                      phase_rad: float = 0.0,
                      profile: str = 'sinewave') -> np.ndarray:
    """Stage ``Grating`` luminance at each point, in display units.

    Reproduces ``stage.builtin.stimuli.Grating``: the texture spans
    ``(1 ± contrast)/2`` and is scaled by ``color``, and the Rieke protocols
    set ``color = 2 * meanIntensity``. The two combine to

    ``L = mean · (1 + contrast · wave(2π·f·x + phase))``

    which is a grating of the requested contrast about the requested mean.

    **On phase.** Protocols in these packages typically compute a
    ``phaseShift`` whose stated purpose is to put a zero crossing — the
    "contrast reversing boundary" — at the center of the grating quad. Rather
    than reproduce that arithmetic against Stage's texture-coordinate
    convention, which is not visible from here, pass ``axis_px`` measured from
    the center and leave ``phase_rad`` at 0: the result is the pattern the
    shift was chosen to produce, and it does not depend on Stage internals.
    Use ``phase_rad`` for drift, ``2π·F·t`` at time ``t``.

    Values are not clipped — compose the frame first, then clip once.
    """
    wave = np.sin(2 * np.pi * float(spatial_freq_cyc_per_px) * axis_px
                  + float(phase_rad))
    if str(profile).lower().startswith('square'):
        wave = np.sign(wave)
    return float(mean_intensity) * (1.0 + float(contrast) * wave)


def apply_circular_aperture(pattern: np.ndarray, dx: np.ndarray, dy: np.ndarray,
                            diameter_px: float, surround_intensity: float,
                            background_intensity: float = 0.0,
                            surround_extent_px: Optional[float] = None
                            ) -> np.ndarray:
    """Show ``pattern`` inside a circle, surround outside it, background beyond.

    The Rieke idiom this reproduces is a ``Rectangle`` carrying
    ``Mask.createCircularAperture`` drawn on top of the stimulus: the mask is
    transparent inside the circle and opaque outside, so the rectangle paints
    its own color everywhere except the circle.

    **The rectangle is usually no bigger than the aperture**, and the
    presentation background is usually ``0``. So there are three regions, not
    two, and the outermost one is *black rather than mean* — a cell whose
    receptive field lands out there spent the epoch in darkness, and reading
    its rate as a response to the stimulus is a mistake. That is easy to miss
    from the ``.m``, where the background is one line far from the aperture.

    Parameters
    ----------
    pattern : np.ndarray
        The stimulus, already evaluated over the whole canvas.
    dx, dy : np.ndarray
        Offsets from the aperture center, from :func:`canvas_offsets`.
    diameter_px : float
        Aperture diameter in pixels. ``0`` or less returns ``pattern``
        untouched, matching the ``if apertureDiameter > 0`` guard these
        protocols wrap the aperture in.
    surround_intensity : float
        Color of the masking rectangle — the mean, in every protocol here.
    background_intensity : float
        The presentation background, outside the rectangle. Default 0.
    surround_extent_px : float, optional
        Half-width of the masking rectangle. Defaults to the aperture radius,
        which is the common case (``aperture.size = [d d]``). Pass a larger
        value, or ``np.inf`` for a rectangle covering the whole canvas, when
        the protocol sizes it independently.
    """
    if diameter_px is None or diameter_px <= 0:
        return pattern

    radius = float(diameter_px) / 2.0
    extent = radius if surround_extent_px is None else float(surround_extent_px)

    frame = np.full_like(pattern, float(background_intensity))
    if np.isinf(extent):
        frame[:] = float(surround_intensity)
    else:
        frame[(np.abs(dx) <= extent) & (np.abs(dy) <= extent)] = float(surround_intensity)
    inside = (dx ** 2 + dy ** 2) <= radius ** 2
    frame[inside] = pattern[inside]
    return frame
