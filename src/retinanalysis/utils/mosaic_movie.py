"""The population response as a movie, over the stimulus that drove it.

:func:`retinanalysis.utils.mosaic_overlay.plot_mosaic_activity` draws one
window: each receptive field filled by its firing rate over a stretch of
seconds, on a single frame of the stimulus. That is a still of something that
moves, and it answers the spatial question — did the cells the grating covered
respond — while leaving the temporal one alone. A drifting grating drives
cells half a spatial period apart in antiphase, so the thing worth seeing is
the response *travelling across the mosaic in step with the bars*, and no
still shows that.

This module animates it. The mosaic keeps §5's coordinates exactly — the
receptive fields were measured in the stimulus's own frame, so nothing is
fitted to make the two line up — and adds the one axis the still was missing.

**Three clocks have to agree, and only one of them is the spike clock.**

- Spike times are milliseconds from the **epoch start**. Everything in the
  package uses that clock.
- ``grating_frame(time_s=...)`` wants seconds from **stimulus onset**, which
  is the epoch start plus ``preTime``. They coincide only when ``preTime`` is
  zero, which it happens to be for the block §5 draws and is not in general —
  so :func:`animate_mosaic_activity` takes ``pre_time_ms`` and does the
  conversion rather than letting the two clocks be silently confused.
- The retina answers late. A rate at time *t* reports a stimulus from some
  tens of milliseconds earlier, so the response visibly trails the bars. That
  lag is a measurement, not an error, and the default leaves it in
  (``latency_ms=0``). §7 measures it — ``describe_phase_alignment`` reports
  the per-type residual as a latency — and passing that value here shifts the
  rate sampling forward so the two lock together. Whether the response then
  tracks the bars is the check; building the alignment in by default would
  destroy the evidence for it.

Two things are held fixed across every frame, because an animation that
rescales per frame encodes nothing: the rate colour scale, and the stimulus
display range. Otherwise the background pulses and a colour means a different
firing rate in each frame.
"""

from __future__ import annotations

import os
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np


__all__ = [
    'cell_rate_timeseries',
    'animate_mosaic_activity',
]


def cell_rate_timeseries(
    activity,
    t_grid: np.ndarray,
    *,
    sigma_s: float = 0.03,
    bin_s: float = 0.001,
):
    """Smoothed firing rate per cell, sampled on ``t_grid``.

    One row per row of ``activity``, built from its ``spike_times_s`` column —
    so the rates and the receptive fields come from the same table and cannot
    drift out of correspondence.

    ``sigma_s`` is the whole temporal resolution of the movie, and the value
    that decides whether it shows anything. The response to be seen is
    modulation at the drift frequency, so the kernel has to be short against
    the drift period: at 2 Hz the period is 500 ms, and a 30 ms sigma resolves
    it while a 150 ms one averages it flat. Too short and every cell flickers
    with its own spike train instead.

    Parameters
    ----------
    activity : pandas.DataFrame
        From :func:`~retinanalysis.utils.mosaic_overlay.cell_activity_in_window`,
        computed over a window at least ``3 * sigma_s`` wider than ``t_grid``
        at both ends — spikes outside the window are not in the table, so a
        rate near its edge is biased low without that margin.
    t_grid : np.ndarray
        Times to sample, in seconds from the epoch start.
    sigma_s : float
        Gaussian kernel sigma, seconds.
    bin_s : float
        Bin width the spikes are histogrammed into before smoothing.

    Returns
    -------
    np.ndarray
        ``(n_cells, len(t_grid))`` of rates in Hz.
    """
    from .psth import gaussian_filter_1d

    t_grid = np.asarray(t_grid, dtype=float)
    if t_grid.size == 0:
        raise ValueError('t_grid is empty.')
    if sigma_s <= 0:
        raise ValueError(f'sigma_s must be positive; got {sigma_s}')

    spike_lists = list(activity['spike_times_s'])
    all_spikes = [np.asarray(s, dtype=float) for s in spike_lists]
    finite = [s for s in all_spikes if s.size]

    # Histogram over a span that covers the grid and every spike the caller
    # loaded, so the convolution has the margin it needs at both ends.
    lo = float(min([t_grid[0]] + [s.min() for s in finite])) - 5 * sigma_s
    hi = float(max([t_grid[-1]] + [s.max() for s in finite])) + 5 * sigma_s
    n_bins = max(int(np.ceil((hi - lo) / bin_s)), 2)
    edges = lo + bin_s * np.arange(n_bins + 1)
    centers = edges[:-1] + bin_s / 2.0

    kernel = gaussian_filter_1d(sigma_s / bin_s)
    rates = np.zeros((len(all_spikes), t_grid.size), dtype=float)
    for i, spikes in enumerate(all_spikes):
        if spikes.size == 0:
            continue
        counts, _ = np.histogram(spikes, bins=edges)
        # Counts per bin -> Hz, then smoothed. The kernel has unit area, so
        # convolving the rate directly keeps the units.
        smoothed = np.convolve(counts / bin_s, kernel, mode='same')
        rates[i] = np.interp(t_grid, centers, smoothed)
    return rates


def _frame_writer(path: str, fps: float, size: Tuple[int, int]):
    """Open a video writer for ``path``, returning ``(write, close, note)``.

    MP4 goes through OpenCV, because there is no ffmpeg here — neither on the
    PATH nor inside this OpenCV wheel, whose only video backend is macOS
    AVFoundation. Which codecs that accepts is a property of the machine, not
    of OpenCV, so rather than hard-code one this tries the usable ones in
    order and keeps the first that opens: ``mp4v`` is the conventional choice
    and is exactly the one AVFoundation refuses. GIF accumulates quantized
    frames and writes them on close.
    """
    ext = os.path.splitext(path)[1].lower()
    width, height = size

    if ext in ('.mp4', '.m4v', '.mov', '.avi'):
        try:
            import cv2
        except ImportError as err:
            raise RuntimeError(
                f'Writing {ext} needs OpenCV (cv2), which is not importable: '
                f'{err}. Use a .gif path instead — pillow is always present.'
            ) from err

        # AVFoundation refuses to open a path that already exists, and reports
        # it the same way it reports an unsupported codec — so re-running a
        # notebook cell would otherwise fail as "no usable codec" on the
        # second run and work on the first.
        if os.path.exists(path):
            os.remove(path)

        # H.264 first: far smaller than MJPG and playable everywhere. MJPG is
        # the fallback that essentially always opens.
        tried = []
        vw = None
        for code in ('avc1', 'H264', 'mp4v', 'MJPG'):
            candidate = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*code),
                                        float(fps), (width, height))
            if candidate.isOpened():
                vw, fourcc = candidate, code
                break
            candidate.release()
            tried.append(code)
        if vw is None:
            raise RuntimeError(
                f'OpenCV could not open {path} for writing with any of '
                f'{", ".join(tried)} at {width}x{height}. Write a .gif '
                f'instead — pillow needs no codec.')

        def write(rgb):
            vw.write(rgb[:, :, ::-1])          # OpenCV wants BGR

        def close():
            vw.release()

        return write, close, f'{ext.lstrip(".")} via OpenCV ({fourcc})'

    if ext == '.gif':
        from PIL import Image
        frames = []

        def write(rgb):
            frames.append(Image.fromarray(rgb).convert(
                'P', palette=Image.ADAPTIVE, colors=128))

        def close():
            if not frames:
                raise RuntimeError('No frames were rendered.')
            frames[0].save(path, save_all=True, append_images=frames[1:],
                           duration=int(round(1000.0 / fps)), loop=0,
                           optimize=True)

        return write, close, 'gif via pillow'

    raise ValueError(f'Unsupported movie extension {ext!r}; use .mp4 or .gif')


def animate_mosaic_activity(
    pipeline,
    epoch_index: int,
    window_s: Tuple[float, float],
    path: str,
    *,
    stim_frame_fn: Optional[Callable[[float], np.ndarray]] = None,
    pre_time_ms: float = 0.0,
    fps: float = 20.0,
    speed: float = 1.0,
    rate_sigma_s: float = 0.03,
    latency_ms: float = 0.0,
    cell_types: Optional[Iterable[str]] = None,
    cell_ids: Optional[Iterable[int]] = None,
    std_scaling: float = 1.6,
    cmap: str = 'plasma',
    stim_cmap: str = 'gray',
    stim_alpha: float = 0.55,
    rate_vmax: Optional[float] = None,
    rate_pct: float = 99.0,
    aperture_diameter_px: Optional[float] = None,
    zoom: bool = True,
    pad_frac: float = 0.06,
    raster_sort_by: str = 'position',
    raster_axis_deg: Optional[float] = None,
    raster_marker_size: float = 1.0,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (13.0, 5.2),
    dpi: int = 100,
    verbose: bool = True,
) -> str:
    """Animate the population response over the drifting stimulus.

    The moving version of
    :func:`~retinanalysis.utils.mosaic_overlay.plot_mosaic_activity`, in the
    same coordinates and the same two panels. **Left**: the mosaic over the
    stimulus, each receptive field filled by that cell's instantaneous firing
    rate, the stimulus dimmed behind it. **Right**: the spikes those rates
    were smoothed from, with a cursor at the current time and the stimulus
    luminance at the canvas centre drawn through them — so whether the
    response follows the drift is readable off the panel rather than inferred
    from the mosaic flickering.

    Like ``plot_mosaic_activity``, the stimulus is supplied rather than
    rendered here, because what a frame *is* differs per protocol. For
    ``variableMeanDriftingGrating``::

        geom = ra.grating_geometry(stim_block, epoch)
        frame_fn = lambda t: ra.grating_frame(stim_block, epoch, time_s=t,
                                              geometry=geom, downsample=2)[0]

    Note what ``frame_fn`` is called with: **seconds from stimulus onset**.
    Spike times run from the epoch start, and the two differ by ``preTime``;
    pass ``pre_time_ms=geom['pre_time_ms']`` and this handles the conversion.

    Parameters
    ----------
    pipeline : MEAPipeline
    epoch_index : int
        The epoch to animate, numbered as the epoch table numbers it.
    window_s : (start, end)
        Seconds from the epoch start. Keep it short — a few drift cycles is
        what the eye can follow, and every second is ``fps`` rendered frames.
    path : str
        Output file. ``.mp4`` (OpenCV) or ``.gif`` (pillow). The extension
        picks the writer.
    stim_frame_fn : callable, optional
        ``t_from_stimulus_onset_s -> 2-D canvas array``. Omit for the mosaic
        on its own.
    pre_time_ms : float
        Epoch start to stimulus onset. ``grating_geometry`` reports it.
    fps : float
        Frames per second of the output file.
    speed : float
        Playback rate relative to real time. ``1.0`` is real time; ``0.25``
        renders four times as many frames and plays back in slow motion,
        which is what a 2 Hz drift needs to be followed by eye.
    rate_sigma_s : float
        Gaussian sigma for the rate estimate — see
        :func:`cell_rate_timeseries`, where the trade-off is described. It has
        to be short against the drift period or the modulation averages away.
    latency_ms : float
        Sample each cell's rate this far *after* the stimulus time being
        drawn. ``0`` (default) leaves the retina's own lag visible, which is
        the honest picture and the thing worth measuring; set it to the
        latency §7 reports to bring stimulus and response into register.
    cell_types, cell_ids : optional
        Restrict the population, as in ``cell_activity_in_window``.
    stim_alpha : float
        Opacity of the stimulus behind the mosaic. The receptive fields are
        the subject; the grating is there to be tracked against, so it is
        drawn faint by default.
    rate_vmax : float, optional
        Top of the rate colour scale, held fixed for the whole movie. Default:
        the ``rate_pct`` percentile over every cell and every frame — computed
        once, because a scale that moved per frame would make the colours
        meaningless.
    aperture_diameter_px : float, optional
        Marks the stimulus aperture. Cells outside it saw the background.
    zoom : bool
        Frame the mosaic on the cells rather than the whole canvas.
    raster_sort_by : ``'position'`` (default), ``'rate'`` or ``'cell_id'``
        Row order in the raster. ``'position'`` sorts along
        ``raster_axis_deg`` so a drifting stimulus reads as a diagonal band
        moving down the panel, which is what makes the alignment visible.
    raster_axis_deg : float, optional
        Axis to sort positions along; pass ``geom['orientation_deg']``.
        ``None`` means 0 (canvas x).

    Returns
    -------
    str
        ``path``, for convenience.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    from .mosaic_overlay import cell_activity_in_window, raster_sort_order
    from .style import NEUTRAL_GRAY, apply_publication_style, colors_for_celltypes

    apply_publication_style()

    t0, t1 = (float(w) for w in window_s)
    if not t1 > t0:
        raise ValueError(f'window_s must be (start, end) with end > start; '
                         f'got {window_s}')
    if speed <= 0:
        raise ValueError(f'speed must be positive; got {speed}')

    latency_s = float(latency_ms) / 1000.0
    pre_s = float(pre_time_ms) / 1000.0

    # Grid of stimulus times to draw. `speed` stretches wall-clock duration
    # without changing what is shown: the same seconds of epoch, sampled finer.
    n_frames = max(int(round((t1 - t0) * fps / speed)), 2)
    t_stim_grid = np.linspace(t0, t1, n_frames)
    # The rates are read at the stimulus time plus the latency being assumed.
    t_rate_grid = t_stim_grid + latency_s

    # Load spikes over a margin so smoothing is unbiased at both ends, and so
    # the latency shift cannot read past what was loaded.
    margin = 5 * rate_sigma_s + abs(latency_s)
    activity = cell_activity_in_window(
        pipeline, epoch_index, (max(t0 - margin, 0.0), t1 + margin),
        cell_types=cell_types, cell_ids=cell_ids, std_scaling=std_scaling,
    )
    if activity.empty:
        raise ValueError('No cells matched the requested types and ids.')

    rates = cell_rate_timeseries(activity, t_rate_grid, sigma_s=rate_sigma_s)

    if cell_types is not None:
        order = [ct for ct in cell_types if (activity['cell_type'] == ct).any()]
    else:
        order = sorted(activity['cell_type'].unique())
    type_colors = colors_for_celltypes(order)

    drawable_mask = activity['center_x'].notna().to_numpy()
    drawable = activity[drawable_mask]
    n_no_rf = int((~drawable_mask).sum())
    if drawable.empty:
        raise ValueError('No cell in the selection has a matched receptive '
                         'field, so there is no mosaic to animate.')

    # One colour scale for the whole movie. Computed over every cell and every
    # frame, so a colour means one firing rate from the first frame to the last.
    drawable_rates = rates[drawable_mask]
    if rate_vmax is None:
        rate_vmax = float(np.percentile(drawable_rates, rate_pct))
    if rate_vmax <= 0:
        rate_vmax = max(float(drawable_rates.max()), 1.0)
    saturated = bool((drawable_rates > rate_vmax).any())
    norm = plt.Normalize(vmin=0.0, vmax=float(rate_vmax))
    rate_cmap = plt.get_cmap(cmap)

    canvas_w, canvas_h = pipeline.analysis_chunk.canvas_size

    # Stimulus frames, and a fixed display range taken across the movie rather
    # than per frame — otherwise the background brightness pulses on its own.
    frames = None
    center_luminance = None
    if stim_frame_fn is not None:
        frames = [np.asarray(stim_frame_fn(t - pre_s), dtype=float)
                  for t in t_stim_grid]
        stim_vmin = float(min(f.min() for f in frames))
        stim_vmax = float(max(f.max() for f in frames))
        if stim_vmax <= stim_vmin:
            stim_vmax = stim_vmin + 1e-6
        # Luminance at the canvas centre: one number per frame, and the
        # simplest honest read-out of the stimulus's own phase.
        center_luminance = np.array(
            [float(f[f.shape[0] // 2, f.shape[1] // 2]) for f in frames])

    fig, (ax_mosaic, ax_raster) = plt.subplots(
        1, 2, figsize=figsize, dpi=dpi,
        gridspec_kw={'width_ratios': [1.0, 1.15]})
    canvas = FigureCanvasAgg(fig)

    # --- Left panel, built once and updated in place -----------------------
    im = None
    if frames is not None:
        im = ax_mosaic.imshow(
            frames[0], cmap=stim_cmap, vmin=stim_vmin, vmax=stim_vmax,
            extent=(0.0, float(canvas_w), float(canvas_h), 0.0),
            interpolation='nearest', alpha=stim_alpha, zorder=0)

    if aperture_diameter_px:
        ax_mosaic.add_patch(Ellipse(
            xy=(canvas_w / 2.0, canvas_h / 2.0),
            width=float(aperture_diameter_px),
            height=float(aperture_diameter_px),
            facecolor='none', edgecolor='white', linestyle='--',
            linewidth=0.8, alpha=0.6, zorder=1))

    # Patch order is fixed for the whole movie: sorting by rate the way the
    # still does would reshuffle the z-order every frame and make the mosaic
    # shimmer for reasons that have nothing to do with the response.
    patches, patch_rows = [], []
    for row_i, (_, cell) in zip(np.flatnonzero(drawable_mask),
                                drawable.iterrows()):
        patch = Ellipse(
            xy=(cell['center_x'], cell['center_y']),
            width=cell['width'], height=cell['height'], angle=cell['angle'],
            facecolor=rate_cmap(norm(0.0)),
            edgecolor=type_colors.get(cell['cell_type'], NEUTRAL_GRAY),
            linewidth=0.7, alpha=0.85, zorder=2)
        ax_mosaic.add_patch(patch)
        patches.append(patch)
        patch_rows.append(row_i)
    patch_rows = np.asarray(patch_rows)

    if zoom:
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
    ax_mosaic.set_ylim(y_hi, y_lo)
    ax_mosaic.set_aspect('equal')
    ax_mosaic.set_xlabel('canvas x (pix)')
    ax_mosaic.set_ylabel('canvas y (pix)')

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=rate_cmap),
                        ax=ax_mosaic, fraction=0.046, pad=0.02,
                        extend='max' if saturated else 'neither')
    cbar.set_label(f'rate (Hz), {int(round(rate_sigma_s * 1000))} ms sigma')

    ax_mosaic.legend(
        handles=[Line2D([0], [0], marker='o', linestyle='none', markersize=6,
                        markerfacecolor='none', markeredgecolor=type_colors[ct],
                        label=f'{ct} (n={int((activity["cell_type"] == ct).sum())})')
                 for ct in order],
        loc='upper center', bbox_to_anchor=(0.5, -0.13),
        ncol=min(len(order), 4), fontsize=8, frameon=False)

    # --- Right panel: the spikes, the stimulus phase, and a cursor ---------
    # Default to spatial order along the drift axis. That is the ordering in
    # which a drifting grating shows itself: successive positions are reached
    # at successively later phases, so the response is a diagonal band moving
    # down the panel, and the cursor crossing it is the alignment made visible.
    # Sorted by rate the same spikes look like noise.
    axis_deg = 0.0 if raster_axis_deg is None else float(raster_axis_deg)
    row = 0
    boundaries = []
    cell_type_col = activity['cell_type'].to_numpy()
    for ct in order:
        # Positions, not index labels: `rates` is built row-by-row from
        # `activity`, so everything downstream has to address it positionally.
        block_pos = np.flatnonzero(cell_type_col == ct)
        sub = activity.iloc[block_pos]
        block_pos = block_pos[raster_sort_order(
            sub, sort_by=raster_sort_by, axis_deg=axis_deg,
            center_xy=(canvas_w / 2.0, canvas_h / 2.0),
            rates=rates[block_pos].mean(axis=1))]
        for pos in block_pos:
            spikes = np.asarray(activity.iloc[pos]['spike_times_s'], dtype=float)
            spikes = spikes[(spikes >= t0) & (spikes <= t1)]
            if spikes.size:
                ax_raster.plot(spikes, np.full(spikes.size, row), '|',
                               color=type_colors.get(ct, NEUTRAL_GRAY),
                               markersize=raster_marker_size * 3,
                               markeredgewidth=raster_marker_size)
            row += 1
        boundaries.append((ct, row, block_pos.size))
        if ct != order[-1]:
            ax_raster.axhline(row - 0.5, color=NEUTRAL_GRAY,
                              linewidth=0.5, alpha=0.5)

    ax_raster.set_xlim(t0, t1)
    ax_raster.set_ylim(-0.5, max(row - 0.5, 0.5))
    ax_raster.set_xlabel('time in epoch (s)')
    ax_raster.set_ylabel(
        {'position': 'cell, by type then position across bars',
         'rate': 'cell, by type then mean rate',
         'cell_id': 'cell, by type then id'}[raster_sort_by])
    ticks, labels, start = [], [], 0
    for ct, end, n in boundaries:
        if n:
            ticks.append((start + end - 1) / 2.0)
            labels.append(ct)
        start = end
    ax_raster.set_yticks(ticks)
    ax_raster.set_yticklabels(labels, fontsize=8)

    if center_luminance is not None:
        ax_stim = ax_raster.twinx()
        ax_stim.plot(t_stim_grid, center_luminance, color='tab:cyan',
                     linewidth=1.1, alpha=0.45, zorder=0)
        ax_stim.set_ylabel('luminance at canvas centre', color='tab:cyan',
                           fontsize=8)
        ax_stim.tick_params(axis='y', labelcolor='tab:cyan', labelsize=7)
        ax_stim.set_zorder(ax_raster.get_zorder() - 1)
        ax_raster.patch.set_visible(False)

    cursor = ax_raster.axvline(t_stim_grid[0], color='white', linewidth=1.6,
                               alpha=0.9, zorder=5)
    cursor_dark = ax_raster.axvline(t_stim_grid[0], color='black',
                                    linewidth=0.6, alpha=0.9, zorder=6)

    if title is None:
        title = (f'epoch {epoch_index} — {len(drawable)} cells'
                 + (f' ({n_no_rf} with no matched RF, raster only)'
                    if n_no_rf else ''))
    lag_note = ('' if latency_ms == 0
                else f', rates read {latency_ms:g} ms after the frame')
    suptitle = fig.suptitle(f'{title}\nt = {t0:.3f} s{lag_note}', fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.92))

    # --- Render -------------------------------------------------------------
    canvas.draw()
    width, height = canvas.get_width_height()
    width, height = width - width % 2, height - height % 2   # mp4 wants even
    write, close, how = _frame_writer(path, fps, (width, height))

    try:
        for k, t in enumerate(t_stim_grid):
            if im is not None:
                im.set_data(frames[k])
            frame_rates = rates[patch_rows, k]
            for patch, r in zip(patches, frame_rates):
                patch.set_facecolor(rate_cmap(norm(r)))
            cursor.set_xdata([t, t])
            cursor_dark.set_xdata([t, t])
            suptitle.set_text(f'{title}\nt = {t:.3f} s{lag_note}')
            canvas.draw()
            rgb = np.asarray(canvas.buffer_rgba())[:height, :width, :3]
            write(rgb)
        close()
    finally:
        plt.close(fig)

    if verbose:
        dur = n_frames / fps
        print(f'{n_frames} frames -> {path} ({how})\n'
              f'  {t0:g}–{t1:g} s of epoch {epoch_index} played over '
              f'{dur:.1f} s at {fps:g} fps ({speed:g}x real time)\n'
              f'  {len(drawable)} cells, rate sigma {rate_sigma_s * 1000:.0f} ms, '
              f'colour scale fixed at 0–{rate_vmax:.1f} Hz'
              + (f', latency {latency_ms:g} ms applied' if latency_ms else
                 ', no latency correction (the response lag is left visible)'))
    return path
