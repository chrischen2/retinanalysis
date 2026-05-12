"""Regen for ``VariableMeanSpatialNoise``.

Source: ``chris-package/+edu/+washington/+riekelab/+chris/+protocols/
VariableMeanSpatialNoise.m``

Stimulus structure: each frame is a ``(numYStixels, numXStixels)`` binary
contrast pattern (±1), multiplied by a mean intensity that switches between
``meanIntensities`` every ``framesPerSwitch`` frames. Each pattern is drawn
fresh from a MATLAB MT19937 stream seeded with the per-epoch ``seed``.

Per-epoch params saved in the H5 (all of these are recoverable):
    seed, numXChecks, numYChecks, numXStixels, numYStixels, stixelSize,
    stepsPerStixel, frameDwell, meanSwitchInterval, meanIntensities,
    totalFrames, framesPerSwitch.

Block-level params:
    preTime, stimTime, tailTime, contrast, gridSize, filterSdStixels,
    meanIntensities (default), frameDwells (default).

Notes on MATLAB ↔ NumPy RNG parity
----------------------------------
MATLAB ``RandStream('mt19937ar', 'Seed', seed).rand(rows, cols)`` and NumPy
``np.random.RandomState(seed).rand(rows, cols)`` use the same MT19937 algorithm
*and* the same legacy seeding scheme, so the produced uniform sequences agree.
MATLAB fills the (rows, cols) matrix in column-major order, so we draw a
flat 1-D vector of ``rows*cols`` values and reshape with ``order='F'`` to
match exactly.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from typing import Any, Literal

from . import register
from ._matlab_rng import matlab_rand as _ml_rand, MatlabUnavailableError
from retinanalysis.utils.matlab_engine import is_matlab_engine_available

PROTOCOL_NAME = 'edu.washington.riekelab.chris.protocols.VariableMeanSpatialNoise'


def _matlab_rand(rng: np.random.RandomState, rows: int, cols: int) -> np.ndarray:
    """``RandStream.rand(rows, cols)`` parity: fill (rows, cols) column-major."""
    return rng.rand(rows * cols).reshape((rows, cols), order='F')


def _resolve_engine(engine: str) -> str:
    """Translate 'auto' into 'matlab' or 'numpy' (does not start MATLAB)."""
    if engine in ('matlab', 'numpy'):
        return engine
    return 'matlab' if is_matlab_engine_available() else 'numpy'


def _draw_uniform_block(seed: int, n_rows: int, n_cols: int, engine: str) -> np.ndarray:
    """Draw an entire (rows*cols) MT19937 uniform block from MATLAB or NumPy.

    Returns the values reshaped (n_rows, n_cols) in MATLAB column-major
    layout so that calling code can index ``arr[:, k]`` to get the
    k-th column drawn (equivalent to a single ``rs.rand(n_rows)`` call
    for that column).
    """
    if engine == 'matlab':
        return _ml_rand(seed, n_rows, n_cols)
    rng = np.random.RandomState(int(seed))
    return rng.rand(n_rows * n_cols).reshape((n_rows, n_cols), order='F')


def _generate_epoch_frames(
    seed: int,
    total_frames: int,
    frame_dwell: int,
    n_y: int,
    n_x: int,
    contrast: float,
    mean_sequence: np.ndarray,
    steps_per_stixel: int,
    engine: str = 'numpy',
) -> tuple[np.ndarray, np.ndarray | None]:
    """Replicate ``precomputeTraces`` for one epoch.

    Returns ``(intensity_uint8, position_pix_or_None)``.
    ``intensity_uint8`` has shape ``(total_frames, n_y, n_x)``.
    ``position_pix`` is ``(total_frames, 2)`` (in grid steps, not pixels)
    and only present when ``steps_per_stixel > 1``.

    Each draw matches one MATLAB call:
        currentNoisePattern = noiseStream.rand(n_y, n_x)
        currentPosition     = positionStream.rand(1, 2)
    The MT19937 stream is sequential, so we batch all draws for the epoch
    into a single call to whichever backend (numpy or MATLAB engine).
    """
    # Number of *fresh* draws (one per frame_dwell block of frames).
    n_unique = int(np.ceil(total_frames / max(frame_dwell, 1)))

    # ---- Noise stream: matches `noiseStream.rand(n_y, n_x)` repeated n_unique times.
    # In MATLAB those n_unique calls consume n_unique*(n_y*n_x) uniforms in
    # sequence; a single rand(n_y, n_x*n_unique) draws them in the same
    # column-major order. Reshape with order='F' so that
    # u3[:, :, k] is the k-th MATLAB pattern.
    u_all = _draw_uniform_block(seed, n_y, n_x * n_unique, engine=engine)
    u3 = u_all.reshape(n_y, n_x, n_unique, order='F')
    patterns = (2 * (u3 > 0.5) - 1).astype(np.int8).transpose(2, 0, 1)
    # patterns shape: (n_unique, n_y, n_x)

    intensity = np.zeros((total_frames, n_y, n_x), dtype=np.uint8)
    for frame in range(total_frames):
        idx = frame // frame_dwell
        m = float(mean_sequence[frame])
        M = contrast * patterns[idx] * m + m
        intensity[frame] = np.clip(np.round(255.0 * M), 0, 255).astype(np.uint8)

    position = None
    if steps_per_stixel > 1:
        # MATLAB: positionStream = RandStream('mt19937ar', 'Seed', seed+1000)
        # Repeated `positionStream.rand(1, 2)` consumes 2*n_unique uniforms
        # → matches a single rand(2, n_unique).
        u_pos = _draw_uniform_block(int(seed) + 1000, 2, n_unique, engine=engine)
        pos_unique = np.round((steps_per_stixel - 1) * u_pos)  # (2, n_unique)
        position = np.zeros((total_frames, 2), dtype=np.float64)
        for frame in range(total_frames):
            position[frame] = pos_unique[:, frame // frame_dwell]
        # NB: caller must multiply by gridSizePix to get displayed pixel offset.

    return intensity, position


def _build_mean_sequence(meanIntensities: np.ndarray, total_frames: int,
                        frames_per_switch: int) -> np.ndarray:
    """Replicate the deterministic mean-switch sequence."""
    n_switches = int(np.ceil(total_frames / frames_per_switch))
    seq = np.zeros(total_frames, dtype=np.float64)
    n_means = len(meanIntensities)
    for s in range(n_switches):
        start = s * frames_per_switch
        end = min((s + 1) * frames_per_switch, total_frames)
        seq[start:end] = meanIntensities[s % n_means]
    return seq


def regen(stim_block: Any, verbose: bool = True,
          contrast: float | None = None,
          frame_rate_hz: float = 60.0,
          render_position: bool = True,
          engine: Literal['auto', 'matlab', 'numpy'] = 'auto') -> xr.Dataset:
    """Build an xarray.Dataset of regenerated stixel frames for this protocol.

    Returns a Dataset with:
        intensity (epoch, frame, y, x) uint8 — displayed stixel intensity
        mean_sequence (epoch, frame) float — mean intensity per frame
        position_jitter_stixels (epoch, frame, xy) float — only when
            stepsPerStixel > 1; the integer position offset (in *grid* steps,
            not pixels). Multiply by gridSizePix to recover displayed pixels.

    This protocol doesn't depend on external resources (no .iml/.mat files),
    so it regenerates regardless of whether chris-package is cloned. The
    ``verbose`` flag still controls log output.
    """
    df = stim_block.df_epochs
    n_epochs = len(df)
    block_params = getattr(stim_block, 'd_epoch_block_params', {}) or {}

    # `contrast` is a block-level property (not always saved per-epoch). The
    # protocol's default is 1; let the caller override if needed.
    if contrast is None:
        contrast = float(block_params.get('contrast', 1.0))

    # Per-epoch arrays
    seeds = np.array([int(p['seed']) for p in df['epoch_parameters']])
    total_frames_arr = np.array([int(p['totalFrames']) for p in df['epoch_parameters']])
    frames_per_switch_arr = np.array([int(p['framesPerSwitch']) for p in df['epoch_parameters']])
    nx_arr = np.array([int(p['numXStixels']) for p in df['epoch_parameters']])
    ny_arr = np.array([int(p['numYStixels']) for p in df['epoch_parameters']])
    frame_dwell_arr = np.array([int(p['frameDwell']) for p in df['epoch_parameters']])
    steps_arr = np.array([int(p['stepsPerStixel']) for p in df['epoch_parameters']])
    means_per_epoch = [np.asarray(p['meanIntensities'], dtype=float).ravel()
                       for p in df['epoch_parameters']]

    # Within an epoch block, grid shape is uniform; assert that and use the
    # first epoch's shape for the xarray dims.
    if not (np.all(nx_arr == nx_arr[0]) and np.all(ny_arr == ny_arr[0])):
        raise ValueError(
            f'VariableMeanSpatialNoise regen: epoch grid shapes vary within block '
            f'({set(zip(nx_arr.tolist(), ny_arr.tolist()))}) — not supported yet.'
        )
    if not np.all(total_frames_arr == total_frames_arr[0]):
        raise ValueError('Per-epoch totalFrames varies within block — not supported yet.')

    n_y, n_x = int(ny_arr[0]), int(nx_arr[0])
    total_frames = int(total_frames_arr[0])

    # Pull display-side params (constant within a block — pick from epoch 0).
    first_params = df['epoch_parameters'].iloc[0] if n_epochs else {}
    canvas_size = first_params.get('canvasSize', None)
    microns_per_pixel = first_params.get('micronsPerPixel', None)
    grid_size_um = first_params.get('gridSize', block_params.get('gridSize', None))

    # Resolve engine choice. For .rand() both numpy and matlab are byte-exact,
    # so 'auto' is allowed to silently use numpy here — the option is exposed
    # for callers who want one consistent source of truth or who don't trust
    # numpy parity on a future numpy release.
    resolved_engine = _resolve_engine(engine)
    if engine == 'matlab' and resolved_engine != 'matlab':
        raise MatlabUnavailableError(
            'engine="matlab" requested but the MATLAB engine is not available.'
        )

    if verbose:
        print(f'[regen] VariableMeanSpatialNoise: {n_epochs} epochs × {total_frames} frames '
              f'× {n_y}×{n_x} stixels (contrast={contrast}, rng={resolved_engine})')

    intensity = np.zeros((n_epochs, total_frames, n_y, n_x), dtype=np.uint8)
    mean_seq = np.zeros((n_epochs, total_frames), dtype=np.float64)
    has_jitter = render_position and np.any(steps_arr > 1)
    position = np.zeros((n_epochs, total_frames, 2), dtype=np.float64) if has_jitter else None

    for i in range(n_epochs):
        m_seq = _build_mean_sequence(means_per_epoch[i], total_frames,
                                     int(frames_per_switch_arr[i]))
        mean_seq[i] = m_seq
        ints, pos = _generate_epoch_frames(
            seed=int(seeds[i]),
            total_frames=total_frames,
            frame_dwell=int(frame_dwell_arr[i]),
            n_y=n_y, n_x=n_x,
            contrast=contrast,
            mean_sequence=m_seq,
            steps_per_stixel=int(steps_arr[i]),
            engine=resolved_engine,
        )
        intensity[i] = ints
        if has_jitter and pos is not None:
            position[i] = pos

    time_s = np.arange(total_frames) / frame_rate_hz

    data_vars = {
        'intensity': (('epoch', 'frame', 'y', 'x'), intensity),
        'mean_sequence': (('epoch', 'frame'), mean_seq),
        'seed': (('epoch',), seeds),
        'frame_dwell': (('epoch',), frame_dwell_arr),
        'steps_per_stixel': (('epoch',), steps_arr),
    }
    if has_jitter and position is not None:
        data_vars['position_jitter_stixels'] = (('epoch', 'frame', 'xy'), position)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'epoch': np.arange(n_epochs),
            'time_s': ('frame', time_s),
            'xy': ['x', 'y'],
        },
    )
    ds.attrs.update({
        'protocol_name': PROTOCOL_NAME,
        'exp_name': stim_block.exp_name,
        'datafile_name': stim_block.datafile_name,
        'preTime_ms': block_params.get('preTime', None),
        'stimTime_ms': block_params.get('stimTime', None),
        'tailTime_ms': block_params.get('tailTime', None),
        'contrast': contrast,
        'frame_rate_hz': frame_rate_hz,
        'meanIntensities_default': block_params.get('meanIntensities', None),
        'gridSize_um': block_params.get('gridSize', None),
        'filterSdStixels': block_params.get('filterSdStixels', None),
        'meanSwitchInterval_ms': block_params.get('meanSwitchInterval', None),
        'n_y_stixels': n_y,
        'n_x_stixels': n_x,
        'rng_engine': resolved_engine,
        'canvas_size_pix': canvas_size,
        'microns_per_pixel': microns_per_pixel,
        'grid_size_um': grid_size_um,
        # MATLAB computes ``gridSizePix = um2pix(gridSize) = round(gridSize/µpp)``
        # and ``stixelSizePix = gridSizePix * stepsPerStixel``. Each stixel covers
        # ``stixelSizePix × stixelSizePix`` canvas pixels via GL.NEAREST sampling.
        'grid_size_pix': (round(grid_size_um / microns_per_pixel)
                          if (grid_size_um and microns_per_pixel) else None),
    })
    return ds


def render_displayed_canvas(stim_ds: xr.Dataset, epoch: int, frame: int) -> np.ndarray:
    """Render the actual canvas-pixel frame as displayed on the monitor.

    Upscales the stixel grid (``intensity[epoch, frame]``) by
    ``stixelSizePix = grid_size_pix * steps_per_stixel`` via nearest-neighbor
    (matching ``GL.NEAREST`` in the protocol), centers the grid on
    ``canvasSize/2`` plus this frame's positional jitter, fills any margin
    with ``255 * mean_sequence[epoch, frame]``, and clips to
    ``canvasSize``. The result is what the retina actually saw.

    Parameters
    ----------
    stim_ds : xr.Dataset
        Output of :func:`regen` for this protocol.
    epoch, frame : int
        Indices along the ``epoch`` and ``frame`` dimensions.

    Returns
    -------
    np.ndarray
        ``(canvas_h, canvas_w)`` uint8 array.
    """
    canvas_size = stim_ds.attrs.get('canvas_size_pix')
    grid_size_pix = stim_ds.attrs.get('grid_size_pix')
    if canvas_size is None or grid_size_pix is None:
        raise ValueError(
            'render_displayed_canvas: stim_ds.attrs missing canvas_size_pix or '
            'grid_size_pix; regen() must have been called against a StimBlock '
            'built on a known rig.'
        )
    canvas_w = int(round(canvas_size[0]))
    canvas_h = int(round(canvas_size[1]))

    steps = int(stim_ds.steps_per_stixel.values[epoch])
    stixel_size_pix = int(grid_size_pix * steps)

    # 1. Upscale the stixel grid via nearest-neighbor (matches GL.NEAREST).
    stixels = stim_ds.intensity.values[epoch, frame]  # (n_y, n_x) uint8
    grid_canvas = np.repeat(np.repeat(stixels, stixel_size_pix, axis=0),
                            stixel_size_pix, axis=1)
    grid_h, grid_w = grid_canvas.shape

    # 2. Per-frame jitter, if any. ``position_jitter_stixels`` is in
    #    *grid-step* units (integer in [0, steps-1]); MATLAB multiplies by
    #    ``stixelShiftPix = stixelSizePix / stepsPerStixel = grid_size_pix``
    #    to get canvas-pixel offset.
    if 'position_jitter_stixels' in stim_ds.data_vars:
        jx, jy = stim_ds.position_jitter_stixels.values[epoch, frame] * grid_size_pix
    else:
        jx, jy = 0.0, 0.0

    # 3. Compose on a canvas-sized buffer filled with the current mean.
    bg_intensity = float(stim_ds.mean_sequence.values[epoch, frame])
    bg_uint8 = int(np.clip(round(255.0 * bg_intensity), 0, 255))
    canvas = np.full((canvas_h, canvas_w), bg_uint8, dtype=np.uint8)

    # Grid centered on (canvas_w/2 + jx, canvas_h/2 + jy)
    grid_left = canvas_w / 2.0 - grid_w / 2.0 + jx
    grid_top = canvas_h / 2.0 - grid_h / 2.0 + jy

    src_left = int(max(0, np.floor(-grid_left)))
    src_top = int(max(0, np.floor(-grid_top)))
    src_right = int(min(grid_w, np.ceil(canvas_w - grid_left)))
    src_bottom = int(min(grid_h, np.ceil(canvas_h - grid_top)))

    if src_right > src_left and src_bottom > src_top:
        dst_left = int(max(0, round(grid_left)))
        dst_top = int(max(0, round(grid_top)))
        dst_right = min(canvas_w, dst_left + (src_right - src_left))
        dst_bottom = min(canvas_h, dst_top + (src_bottom - src_top))
        canvas[dst_top:dst_bottom, dst_left:dst_right] = grid_canvas[
            src_top:src_top + (dst_bottom - dst_top),
            src_left:src_left + (dst_right - dst_left),
        ]

    return canvas


register(PROTOCOL_NAME, regen)
