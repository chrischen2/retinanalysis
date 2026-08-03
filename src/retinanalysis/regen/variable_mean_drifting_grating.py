"""Reconstruct what ``variableMeanDriftingGrating`` put on the display.

The protocol drifts a sinewave grating inside a circular aperture, and
alternates the background mean intensity and the bar width across epochs.
This module rebuilds a single frame of it in **canvas pixels**, which is the
coordinate system the mosaic already lives in: ``AnalysisChunk`` stores RF
centers in stixels of the noise grid, and ``pixels_per_stixel`` scales them to
the same canvas the stimulus was drawn on. So a reconstructed frame and the
mosaic co-register by construction — no rig calibration is involved, unlike
the electrode overlay in :mod:`retinanalysis.utils.mosaic_overlay`, which maps
a physical chip into that frame and does need one.

**Everything is read from the recorded epochs.** The ``.m`` declares 4 Hz and
an 800 µm aperture; the block in ``demos/variableMeanDriftingGrating.ipynb``
ran 2 Hz and 2000 µm. The source is used for one thing the data cannot supply:
the units. Two of its comments are wrong and the code beside them is right ---
``barWidths % Center bar width (pix)`` and ``apertureDiameter % Surround
radius (pix)`` are both passed through ``um2pix``, so both are **microns**,
and the aperture is a diameter rather than a radius.

Three details of the port are worth stating, because each is a place the
reconstruction could silently drift from the display:

``um2pix`` rounds.
    The device converts with ``round(um / micronsPerPixel)``, so a 50 µm bar
    at 3.8 µm/pixel is 13 pixels, not 13.16, and the spatial frequency that
    follows is the rounded one. Skipping the rounding puts the bars visibly
    out of register by the edge of a 526-pixel aperture.

The phase offset is what pins the pattern to the center.
    ``createPresentation`` computes ``phaseShift`` so that "the contrast
    reversing boundary" sits at the middle of the grating quad. Rather than
    reproduce that arithmetic against Stage's own phase reference --- which
    is a texture-coordinate convention this package cannot see --- the frame
    is written in the form the offset was chosen to produce: a sine measured
    from the canvas center, so a zero crossing lands there at ``t=0``. The two
    agree, and this one does not depend on Stage internals.

Outside the aperture is black, not mean.
    The presentation background is ``0``, and the mean-colored rectangle
    carrying the circular aperture mask is only as large as the aperture
    itself. So a cell whose receptive field falls outside the circle spent the
    epoch in darkness --- which is worth knowing before reading its firing
    rate as a response to the grating.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from retinanalysis.utils.stimulus_frame import (
    apply_circular_aperture,
    canvas_offsets,
    grating_luminance,
    project_on_axis,
    um_to_pixels,
)


__all__ = [
    'grating_geometry',
    'grating_frame',
]


def _epoch_params(stim_block, epoch_index: int) -> Dict:
    """The recorded parameter dict for one epoch of the block."""
    df = stim_block.df_epochs
    if epoch_index < 0 or epoch_index >= len(df):
        raise IndexError(f'epoch {epoch_index} is outside the block '
                         f'(0..{len(df) - 1})')
    return dict(df['epoch_parameters'].iloc[epoch_index])


def grating_geometry(stim_block, epoch_index: int) -> Dict:
    """What the grating was, for one epoch, in canvas pixels.

    Every value comes from that epoch's recorded parameters, converted the
    way ``createPresentation`` converts them.

    Returns
    -------
    dict
        ``canvas_w``, ``canvas_h`` : int — canvas size in pixels.
        ``center_x``, ``center_y`` : float — canvas center, where the
            grating and aperture are positioned.
        ``mean_intensity`` : float — this epoch's ``currentMeanIntensity``.
        ``bar_width_um``, ``bar_width_px`` : this epoch's bar width, before
            and after the device's rounding conversion.
        ``aperture_diameter_um``, ``aperture_diameter_px`` : the circular
            aperture, ``round(um2pix(d/2)) * 2`` as MATLAB computes it.
        ``spatial_freq_cyc_per_px`` : ``1 / (2 * bar_width_px)``.
        ``temporal_freq_hz``, ``contrast``, ``orientation_deg`` : as recorded.
        ``spatial_class`` : ``'sinewave'`` or ``'squarewave'``.
        ``microns_per_pixel`` : the display scale used for the conversions.
        ``pre_time_ms``, ``stim_time_ms`` : epoch timing, so a spike time
            measured from the epoch start can be turned into a stimulus time.
    """
    p = _epoch_params(stim_block, epoch_index)

    mpp = float(p['micronsPerPixel'])
    canvas_w, canvas_h = (int(round(v)) for v in p['canvasSize'])

    # The protocol writes this epoch parameter misspelled ('Wdith'); that is
    # the name in the database, so it is the name to read. Fall back to the
    # correct spelling in case a later revision fixes it.
    bar_um = float(p.get('currentBarWdith', p.get('currentBarWidth')))
    aperture_um = float(p['apertureDiameter'])

    bar_px = um_to_pixels(bar_um, mpp)
    if bar_px <= 0:
        raise ValueError(f'bar width {bar_um} µm rounds to {bar_px} pixels at '
                         f'{mpp} µm/pixel — too fine for this display')
    # MATLAB: round(um2pix(apertureDiameter/2))*2 — halved, converted, doubled,
    # so the diameter in pixels is always even.
    aperture_px = um_to_pixels(aperture_um / 2.0, mpp) * 2

    return {
        'canvas_w': canvas_w,
        'canvas_h': canvas_h,
        'center_x': canvas_w / 2.0,
        'center_y': canvas_h / 2.0,
        'mean_intensity': float(p['currentMeanIntensity']),
        'bar_width_um': bar_um,
        'bar_width_px': bar_px,
        'aperture_diameter_um': aperture_um,
        'aperture_diameter_px': aperture_px,
        'spatial_freq_cyc_per_px': 1.0 / (2.0 * bar_px),
        'temporal_freq_hz': float(p['temporalFrequency']),
        'contrast': float(p['spatialContrast']),
        'orientation_deg': float(p['orientation']),
        'spatial_class': str(p.get('spatialClass', 'sinewave')),
        'microns_per_pixel': mpp,
        'pre_time_ms': float(p.get('preTime', 0.0) or 0.0),
        'stim_time_ms': float(p.get('stimTime', 0.0) or 0.0),
    }


def grating_frame(stim_block, epoch_index: int, time_s: float = 0.0,
                  geometry: Optional[Dict] = None,
                  downsample: int = 1) -> Tuple[np.ndarray, Dict]:
    """One frame of the drifting grating, in canvas-pixel coordinates.

    Parameters
    ----------
    stim_block : MEAStimBlock
        The protocol block. Parameters are read from ``epoch_index``'s own
        recorded epoch, so alternating conditions come out right.
    epoch_index : int
        Position of the epoch in the block — the same index the epoch table
        and ``EPOCH_INDICES`` use.
    time_s : float
        Time within the stimulus, in seconds from stimulus onset (not from
        the epoch start; they differ by ``preTime`` when that is nonzero).
        The grating drifts, so this picks which frame you get.
    geometry : dict, optional
        Output of :func:`grating_geometry` for this epoch, if you already
        have it. Computed here when omitted.
    downsample : int
        Render every ``downsample``-th canvas pixel. The default renders at
        full canvas resolution, which is 800×600 on these rigs and cheap;
        raise it only if you are making many frames.

    Returns
    -------
    (frame, geometry)
        ``frame`` is a float array of display intensities in 0–1, indexed
        ``[y, x]`` with y increasing downwards, so it draws directly with
        ``imshow(..., extent=(0, canvas_w, canvas_h, 0))`` — the same extent
        and orientation the mosaic's canvas-pixel coordinates assume.
        ``geometry`` is the dict :func:`grating_geometry` returned, so the
        caller can label the frame without recomputing it.
    """
    g = geometry or grating_geometry(stim_block, epoch_index)

    # Everything is measured from the canvas center: that is where the grating
    # and the aperture are positioned, and where the protocol's phase offset
    # puts a zero crossing — which is why no phase term is needed here beyond
    # the drift.
    dx, dy = canvas_offsets(g['canvas_w'], g['canvas_h'], downsample=downsample)
    axis = project_on_axis(dx, dy, g['orientation_deg'])

    grating = grating_luminance(
        axis,
        spatial_freq_cyc_per_px=g['spatial_freq_cyc_per_px'],
        mean_intensity=g['mean_intensity'],
        contrast=g['contrast'],
        phase_rad=2 * np.pi * g['temporal_freq_hz'] * float(time_s),
        profile=g['spatial_class'],
    )

    # Background, then the mean-colored aperture square, then the grating
    # inside the circle — the order the presentation stacks them.
    frame = apply_circular_aperture(
        grating, dx, dy,
        diameter_px=g['aperture_diameter_px'],
        surround_intensity=g['mean_intensity'],
        background_intensity=0.0,
    )

    # The display cannot show what it cannot show; a high mean and high
    # contrast together clip at the top.
    return np.clip(frame, 0.0, 1.0), g
