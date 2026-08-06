"""The drifting-grating response, folded onto one cycle and aligned by position.

Between looking at single cells (§5–§6) and measuring the population (§8)
there is a view worth having on its own: what the response *looks like*, per
cell and per type, once it is folded onto the stimulus cycle.

Three steps, each of which needs the one before it:

1. **The trial-averaged PSTH per cell** (:func:`cell_mean_psth`). One trace
   per cell over the whole epoch, averaged across epochs of one condition.
   This is where the slow adaptation is visible — as a decaying envelope —
   and where a cell that is not driven at all announces itself.
2. **The cycle average per cell** (:func:`cycle_average`). The drift frequency
   is known, so the epoch can be folded onto one cycle and the modulation
   read directly instead of inferred from an F1 amplitude. Folding 120 cycles
   into one is also what makes a 10 Hz cell's modulation visible at all.
3. **The type average, aligned by position** (:func:`aligned_cycle_average`).
   This is the step that cannot be skipped. Cells half a spatial period apart
   are driven in **antiphase**, so averaging their cycle responses directly
   cancels the modulation and returns a flat line — the population looks
   unmodulated precisely when it is most organised. Each cell's fold is first
   rotated by the phase the grating puts at its own receptive field, and only
   then averaged.

   The rotation uses no fitted parameter: the predicted phase is
   ``pi/2 - 2*pi*f_s*a`` at position ``a`` along the drift axis, the same
   prediction §7 measures the registration with. :func:`plot_cycle_alignment`
   draws the before and after as heat maps of cells against phase, where the
   diagonal stripes of the unaligned panel — one cycle of phase per spatial
   period — become vertical in the aligned one. That the diagonal exists at
   all *is* the spatial code; that it straightens is the check that the
   alignment is right rather than merely applied.

4. **Both, over time** (:func:`cycle_evolution`). Cut the epoch into short
   windows and repeat, and the aligned cycle average becomes an image of
   phase against time since the luminance step — the same recovery §8
   quantifies, in the form where you can see it.

Everything here reads a :class:`~retinanalysis.utils.spatial_recovery
.PhaseBinnedResponse`, so the conventions, the cell selection and the
refusal to pool epochs of different geometry are shared with §8 rather than
re-implemented beside it.
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


__all__ = [
    'cell_mean_psth',
    'browse_cell_psths',
    'cycle_average',
    'position_phase_shift',
    'aligned_cycle_average',
    'plot_cycle_alignment',
    'browse_cycle_alignment',
    'cycle_evolution',
    'plot_cycle_evolution',
]


# ---------------------------------------------------------------------------
# 1. Trial-averaged PSTH per cell
# ---------------------------------------------------------------------------

def cell_mean_psth(pipeline, epoch_indices: Iterable[int], *,
                   duration_s: Optional[float] = None,
                   pre_time_s: float = 0.0,
                   bin_s: float = 0.05,
                   sigma_s: float = 0.1,
                   cell_types: Optional[Iterable[str]] = None,
                   cell_ids: Optional[Iterable[int]] = None,
                   geometry: Optional[Dict] = None):
    """Per-cell firing rate over the epoch, averaged across epochs.

    Averaging across epochs of **one condition** only — the caller picks
    them. Epochs of different bar width are a different stimulus, and epochs
    of different mean are a different adaptation state, so a mean over both
    describes neither.

    Parameters
    ----------
    pipeline : MEAPipeline
    epoch_indices : sequence[int]
        Epochs to average. Time is measured from each epoch's own start, so
        the average is aligned on the luminance step.
    duration_s : float, optional
        Length of the PSTH. Defaults to ``geometry['stim_time_ms']`` when a
        geometry is given, else the longest spike time seen.
    bin_s, sigma_s : float
        Histogram bin and Gaussian smoothing width. ``sigma_s`` of 0.1 s is
        deliberately *wider* than one drift cycle would need: this panel is
        for the slow envelope, and the cycle-locked modulation belongs in
        :func:`cycle_average` where it is folded rather than smoothed away.

    Returns
    -------
    (cells, t, psth)
        ``cells`` is a DataFrame with ``cell_id``, ``cell_type`` and the RF
        columns; ``t`` is the bin centres in seconds; ``psth`` is
        ``(n_cells, n_bins)`` in Hz.
    """
    from .mosaic_overlay import cell_activity_in_window
    from .psth import gaussian_filter_1d

    epochs = [int(e) for e in epoch_indices]
    if not epochs:
        raise ValueError('epoch_indices is empty')

    if duration_s is None:
        duration_s = (float(geometry['stim_time_ms']) / 1000.0
                      if geometry is not None else None)

    tables = {}
    for e in epochs:
        hi = (pre_time_s + duration_s) if duration_s else 1e6
        tables[e] = cell_activity_in_window(
            pipeline, e, (pre_time_s, hi),
            cell_types=cell_types, cell_ids=cell_ids)

    base = tables[epochs[0]]
    if duration_s is None:
        duration_s = max(
            (float(np.max(s)) for t in tables.values()
             for s in t['spike_times_s'] if len(s)), default=1.0)

    edges = np.arange(0.0, float(duration_s) + bin_s, bin_s)
    centers = edges[:-1] + bin_s / 2.0
    kernel = gaussian_filter_1d(max(sigma_s / bin_s, 1e-6))

    ids = base['cell_id'].astype(int).tolist()
    psth = np.zeros((len(ids), centers.size))
    for e in epochs:
        by_id = tables[e].set_index('cell_id')
        for i, cid in enumerate(ids):
            s = np.asarray(by_id.at[cid, 'spike_times_s'], dtype=float) - pre_time_s
            if s.size:
                psth[i] += np.histogram(s, bins=edges)[0]
    psth = psth / (len(epochs) * bin_s)
    psth = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode='same'),
                               1, psth)

    cells = base[['cell_id', 'noise_id', 'cell_type', 'center_x', 'center_y',
                  'width', 'height', 'angle']].reset_index(drop=True)
    return cells, centers, psth


def browse_cell_psths(cells, t, psth, *, cell_types: Optional[Sequence[str]] = None,
                      geometry: Optional[Dict] = None,
                      n_cols: int = 6, panel_size: Tuple[float, float] = (2.0, 1.3),
                      share_y: bool = False, description: str = 'Cell type:'):
    """Dropdown over cell types; a grid of one trial-averaged PSTH per cell.

    Cells are ordered by position along the drift axis when a ``geometry`` is
    given, so neighbouring panels are neighbouring retina — the ordering that
    makes a systematic phase progression visible rather than scattered.

    ``share_y=False`` scales every panel to itself, which is what you want for
    spotting *shape*; ``True`` puts them on one scale, which is what you want
    for comparing amplitude across cells and makes the quiet ones vanish.
    """
    import matplotlib.pyplot as plt
    from .browse import figure_to_png, png_browser
    from .style import NEUTRAL_GRAY, apply_publication_style, colors_for_celltypes

    apply_publication_style()
    present = ([ct for ct in cell_types if (cells['cell_type'] == ct).any()]
               if cell_types is not None
               else sorted(cells['cell_type'].dropna().unique()))
    if not present:
        print('No cells to show.')
        return None
    colors = colors_for_celltypes(present)

    def _render(ct):
        idx = np.flatnonzero((cells['cell_type'] == ct).to_numpy())
        if geometry is not None:
            a = _axis_of(cells.iloc[idx], geometry)
            idx = idx[np.argsort(a)]
        n = idx.size
        n_rows = int(np.ceil(n / n_cols)) + 1        # +1 for the type mean
        fig, axes = plt.subplots(
            n_rows, n_cols, squeeze=False, sharex=True, sharey=share_y,
            figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows))
        flat = axes.ravel()

        # The type mean first, spanning the top row, as the reference every
        # single cell is read against.
        for ax in flat[:n_cols]:
            ax.remove()
        top = fig.add_subplot(n_rows, 1, 1)
        m = psth[idx].mean(axis=0)
        top.plot(t, m, color=colors.get(ct, NEUTRAL_GRAY), lw=1.4)
        top.set_title(f'{ct}: mean of {n} cells', fontsize=9)
        top.set_ylabel('Hz')

        for k, ax in enumerate(flat[n_cols:]):
            if k >= n:
                ax.axis('off')
                continue
            i = idx[k]
            ax.plot(t, psth[i], color=colors.get(ct, NEUTRAL_GRAY), lw=0.7)
            ax.set_title(f'{int(cells.iloc[i]["cell_id"])}', fontsize=7, pad=1.5)
            ax.tick_params(labelsize=6)
        for ax in flat[-n_cols:]:
            if ax.axes is not None:
                ax.set_xlabel('s', fontsize=7)
        fig.suptitle(f'{ct} — trial-averaged PSTH, '
                     + ('ordered across the bars' if geometry is not None
                        else 'unordered'), fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return None, figure_to_png(fig)

    return png_browser([(ct, ct) for ct in present], _render,
                       description=description)


def _axis_of(cells, geometry):
    theta = np.deg2rad(float(geometry['orientation_deg']))
    dx = cells['center_x'].to_numpy(dtype=float) - float(geometry['center_x'])
    dy = cells['center_y'].to_numpy(dtype=float) - float(geometry['center_y'])
    return dx * np.cos(theta) + dy * np.sin(theta)


# ---------------------------------------------------------------------------
# 2-3. Cycle average, and the position alignment
# ---------------------------------------------------------------------------

def cycle_average(pbr, *, window: Optional[int] = None):
    """Per-cell firing rate against drift phase, folded over every cycle.

    Parameters
    ----------
    pbr : PhaseBinnedResponse
    window : int, optional
        Index into ``pbr.windows``. Default pools every window, which
        double-counts any cycle that two overlapping windows both contain —
        fine for non-overlapping windows and worth knowing otherwise.

    Returns
    -------
    (rates, n_cycles)
        ``rates`` is ``(n_cells, n_phase_bins)`` in Hz; ``n_cycles`` is how
        many drift cycles went into it.
    """
    sel = (np.ones(pbr.counts.shape[0], dtype=bool) if window is None
           else pbr.sample_window == int(window))
    n_cycles = int(sel.sum())
    if n_cycles == 0:
        raise ValueError(f'no cycles in window {window}')
    # The frequency the fold was built at, which is not always the geometry's
    # nominal one — reading it back off `pbr` keeps the bin width consistent
    # with the binning.
    cycle_s = pbr.cycle_s
    bin_s = cycle_s / pbr.n_phase_bins
    # counts is (samples, phase, cell) -> (cell, phase)
    return pbr.counts[sel].sum(axis=0).T / (n_cycles * bin_s), n_cycles


def position_phase_shift(pbr) -> np.ndarray:
    """Bins to rotate each cell's fold by, from its position across the bars.

    The grating puts phase ``pi/2 - 2*pi*f_s*a`` at position ``a`` along the
    drift axis — the same prediction :func:`~retinanalysis.utils.response_phase
    .drift_phase_response` measures residuals against, so the two agree by
    construction rather than by coincidence. Rotating each cell's fold by
    *minus* that puts every cell on a common axis where what remains is the
    response latency, shared across the type.

    Returns integer bin shifts, so alignment is exact only to one bin;
    ``n_phase_bins`` of 24 or more keeps the rounding well under the width of
    any real response peak.
    """
    f_s = float(pbr.geometry['spatial_freq_cyc_per_px'])
    axis = pbr.cells['axis_px'].to_numpy(dtype=float)
    stim_phase = np.pi / 2 - 2 * np.pi * f_s * axis
    return np.rint(stim_phase / (2 * np.pi) * pbr.n_phase_bins).astype(int)


def aligned_cycle_average(pbr, *, window: Optional[int] = None,
                          align: bool = True,
                          normalize: str = 'none',
                          min_spikes: int = 5):
    """Cycle averages rotated onto a common phase, and the mean per cell type.

    Parameters
    ----------
    align : bool
        ``False`` returns the unrotated folds, which is what makes the
        cancellation visible: the per-type mean of unaligned folds is close to
        flat wherever the population is well organised in space.
    normalize : one of ``'none'``, ``'demean'``, ``'fraction'``, ``'peak'``,
        ``'zscore'``
        Per-cell normalization before the type mean, and the choice decides
        what the answer is *about*:

        - ``'none'`` — rate in Hz. High-rate cells dominate the mean, which is
          honest about who carries the signal.
        - ``'demean'`` — Hz around each cell's own mean. Removes the DC, so a
          recovery of baseline firing cannot read as a recovery of modulation.
        - ``'fraction'`` — ``(r - mean) / mean``, modulation relative to the
          cell's own rate. This is the one that tracks F1/F0, and on an
          adapting response it is the only one that shows the recovery:
          absolute modulation *falls* here because rate falls faster than
          F1/F0 rises, and z-scored modulation is flat by construction.
        - ``'peak'`` / ``'zscore'`` — shape only, every cell weighted equally.
          Right when the question is about phase rather than amplitude.

    Returns
    -------
    (per_cell, by_type, phase)
        ``per_cell`` ``(n_cells, n_phase_bins)`` after rotation and
        normalization; ``by_type`` a dict of cell type to the mean over its
        cells; ``phase`` the bin centres in radians.
    """
    rates, _ = cycle_average(pbr, window=window)
    K = pbr.n_phase_bins

    # Every per-cell normalization divides by something the cell itself
    # supplies, so a cell that was silent in this window divides by ~0 and
    # returns either NaN or an enormous fraction — which then takes the whole
    # type mean with it. Those cells are dropped from the mean instead.
    sel = (np.ones(pbr.counts.shape[0], dtype=bool) if window is None
           else pbr.sample_window == int(window))
    spikes_per_cell = pbr.counts[sel].sum(axis=(0, 1))
    too_quiet = spikes_per_cell < int(min_spikes)

    out = rates.astype(float).copy()
    if align:
        shifts = position_phase_shift(pbr)
        for i, s in enumerate(shifts):
            out[i] = np.roll(out[i], -int(s) % K)

    mu = out.mean(axis=1, keepdims=True)
    if normalize == 'peak':
        peak = np.nanmax(np.abs(out), axis=1, keepdims=True)
        out = out / np.where(peak == 0, 1, peak)
    elif normalize == 'zscore':
        sd = out.std(axis=1, keepdims=True)
        out = (out - mu) / np.where(sd == 0, 1, sd)
    elif normalize == 'demean':
        out = out - mu
    elif normalize == 'fraction':
        out = (out - mu) / np.where(mu == 0, np.nan, mu)
    elif normalize != 'none':
        raise ValueError(
            f"normalize must be 'none', 'demean', 'fraction', 'peak' or "
            f"'zscore'; got {normalize!r}")

    if normalize != 'none':
        out[too_quiet] = np.nan

    types = pbr.cells['cell_type'].to_numpy()
    with warnings.catch_warnings():
        # A type whose every cell was too quiet in this window is an empty
        # slice; NaN is the right answer for it, not a warning.
        warnings.simplefilter('ignore', RuntimeWarning)
        by_type = {ct: np.nanmean(out[types == ct], axis=0)
                   for ct in sorted(set(types)) if (types == ct).any()}
    phase = (np.arange(K) + 0.5) * 2 * np.pi / K
    return out, by_type, phase


def _modulation_retained(per_cell, K):
    """How much of the single-cell modulation survives averaging across cells.

    The point of the alignment in one number: the F1 of the mean over cells,
    against the mean of the cells' own F1 magnitudes. 1 means the folds added
    coherently; near 0 means they cancelled, which is what unaligned averaging
    does to a spatially organised population.
    """
    w = np.exp(-1j * 2 * np.pi * np.arange(K) / K)
    ok = np.isfinite(per_cell).all(axis=1)
    if not ok.any():
        return np.nan
    per = np.abs(per_cell[ok] @ w)
    mean_wave = per_cell[ok].mean(axis=0)
    denom = np.mean(per)
    return float(np.abs(mean_wave @ w) / denom) if denom else np.nan


def plot_cycle_alignment(pbr, *, window: Optional[int] = None,
                         cell_types: Optional[Sequence[str]] = None,
                         normalize: str = 'zscore',
                         n_cycles_shown: int = 2,
                         cmap: str = 'RdBu_r',
                         figsize: Tuple[float, float] = (13.0, 5.0)):
    """Cells against phase, before and after the position rotation, plus means.

    Left and middle are the same cells in the same order — sorted by position
    across the bars — folded onto one drift cycle. **Left** is unrotated, and
    a spatially organised population shows a diagonal: one full cycle of
    response phase per spatial period of the grating. **Middle** applies the
    rotation the cells' positions predict, and the diagonal should stand up
    vertical. Nothing is fitted between the two panels, so the straightening
    is a measurement of the registration and not a consequence of it.

    **Right** is what the two orderings cost: the type mean of the aligned
    folds against the type mean of the unaligned ones. The unaligned mean is
    near flat — averaging antiphase cells cancels them — and the number in the
    legend is the fraction of single-cell modulation each retains.
    """
    import matplotlib.pyplot as plt
    from .style import NEUTRAL_GRAY, apply_publication_style, colors_for_celltypes

    apply_publication_style()
    K = pbr.n_phase_bins
    types = pbr.cells['cell_type'].to_numpy()
    present = ([ct for ct in cell_types if (types == ct).any()]
               if cell_types is not None else sorted(set(types)))
    colors = colors_for_celltypes(present)

    raw, raw_by_type, phase = aligned_cycle_average(
        pbr, window=window, align=False, normalize=normalize)
    ali, ali_by_type, _ = aligned_cycle_average(
        pbr, window=window, align=True, normalize=normalize)

    # Blocked by cell type, then by position inside each block. Sorting the
    # whole population by position alone interleaves ON and OFF, which are
    # driven in antiphase, and the two diagonals cross-hatch into noise — the
    # spatial progression is only legible within one polarity.
    axis = pbr.cells['axis_px'].to_numpy()
    blocks, bounds, row = [], [], 0
    for ct in present:
        idx = np.flatnonzero(types == ct)
        idx = idx[np.argsort(axis[idx])]
        blocks.append(idx)
        row += idx.size
        bounds.append((ct, row, idx.size))
    order = np.concatenate(blocks) if blocks else np.array([], dtype=int)
    raw_m, ali_m = raw[order], ali[order]

    # Repeat the cycle so a feature at the wrap-around is not cut in half.
    def _tile(x):
        return np.tile(x, (1, n_cycles_shown))
    deg = np.linspace(0, 360 * n_cycles_shown, K * n_cycles_shown, endpoint=False)

    fig, axes = plt.subplots(1, 3, figsize=figsize,
                             gridspec_kw={'width_ratios': [1, 1, 1.15]})
    vmax = float(np.nanpercentile(np.abs(np.concatenate([raw_m, ali_m])), 98))
    for ax, img, title in ((axes[0], raw_m, 'as recorded'),
                           (axes[1], ali_m, 'rotated by RF position')):
        ax.imshow(_tile(img), aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax,
                  extent=(0, 360 * n_cycles_shown, img.shape[0], 0),
                  interpolation='nearest')
        ax.set_xlabel('drift phase (deg)')
        ax.set_title(title, fontsize=10)
        for _, end, n in bounds[:-1]:
            ax.axhline(end, color='k', lw=0.8)
    ticks = [(end - n / 2.0) for _, end, n in bounds]
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels([ct for ct, _, _ in bounds], fontsize=8)
    axes[0].set_ylabel('cell type, then position across the bars')
    axes[1].set_yticks(ticks)
    axes[1].set_yticklabels([])

    ax = axes[2]
    for ct in present:
        r = _modulation_retained(ali[types == ct], K)
        u = _modulation_retained(raw[types == ct], K)
        ax.plot(np.degrees(phase), ali_by_type[ct], '-', lw=1.6,
                color=colors.get(ct, NEUTRAL_GRAY),
                label=f'{ct} aligned ({r:.2f})')
        ax.plot(np.degrees(phase), raw_by_type[ct], '--', lw=1.0, alpha=0.55,
                color=colors.get(ct, NEUTRAL_GRAY),
                label=f'{ct} as recorded ({u:.2f})')
    ax.axhline(0, color=NEUTRAL_GRAY, lw=0.5)
    ax.set_xlabel('drift phase (deg)')
    ax.set_ylabel({'zscore': 'z-scored rate', 'peak': 'rate / peak',
                   'none': 'rate (Hz)'}[normalize])
    ax.set_title('type mean (fraction of single-cell F1 kept)', fontsize=10)
    ax.legend(fontsize=6.5, ncol=2, loc='upper center',
              bbox_to_anchor=(0.5, -0.22), frameon=False)

    win = ('all windows' if window is None
           else f'{pbr.windows[window][0]:g}–{pbr.windows[window][1]:g} s')
    fig.suptitle(f'cycle average, {win} — '
                 f'{pbr.condition.get("bar_width_um"):g} µm bars at mean '
                 f'{pbr.condition.get("mean_intensity"):g}')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def browse_cycle_alignment(pbr, **kwargs):
    """Dropdown over ``pbr``'s windows, each drawn by :func:`plot_cycle_alignment`."""
    from .browse import figure_to_png, png_browser

    def _render(w):
        fig = plot_cycle_alignment(pbr, window=None if w < 0 else w, **kwargs)
        return None, figure_to_png(fig)

    options = [('all windows pooled', -1)]
    options += [(lab, i) for i, lab in enumerate(pbr.window_labels)]
    return png_browser(options, _render, description='Window:')


# ---------------------------------------------------------------------------
# 4. Over time
# ---------------------------------------------------------------------------

def cycle_evolution(pbr, *, cell_types: Optional[Sequence[str]] = None,
                    align: bool = True, normalize: str = 'fraction') -> Dict:
    """The aligned cycle average of each type, one row per window.

    Returns ``{cell_type: (n_windows, n_phase_bins)}`` plus ``'_t'`` (window
    centres) and ``'_phase'``. Use with short, non-overlapping windows —
    :func:`~retinanalysis.utils.spatial_recovery.sliding_windows` with
    ``step_s == width_s``.

    ``normalize='fraction'`` is the default because it is the only setting in
    which this panel shows the recovery. The response is adapting, so rate
    falls faster than F1/F0 rises: in Hz the modulation *shrinks* over the
    epoch, and z-scored it is flat by construction. Expressed as a fraction of
    each cell's own mean, it grows — which is the same quantity §8 fits a
    timescale to, here as a picture instead of a number.
    """
    types = pbr.cells['cell_type'].to_numpy()
    present = ([ct for ct in cell_types if (types == ct).any()]
               if cell_types is not None else sorted(set(types)))
    out = {ct: [] for ct in present}
    for wi in range(len(pbr.windows)):
        if not (pbr.sample_window == wi).any():
            continue
        _, by_type, phase = aligned_cycle_average(
            pbr, window=wi, align=align, normalize=normalize)
        for ct in present:
            out[ct].append(by_type.get(ct, np.full(pbr.n_phase_bins, np.nan)))
    result = {ct: np.asarray(v) for ct, v in out.items()}
    result['_t'] = pbr.window_centers
    result['_phase'] = phase
    return result


def plot_cycle_evolution(evolution: Dict, *,
                         cmap: str = 'RdBu_r',
                         n_cycles_shown: int = 2,
                         cbar_label: str = 'modulation, fraction of own mean',
                         figsize: Optional[Tuple[float, float]] = None,
                         title: str = ''):
    """Phase against time since the step, one panel per cell type.

    A vertical band that darkens downward is the recovery: the response stays
    at one phase — it has to, the alignment put it there — while its depth
    grows. A band that *drifts* sideways instead would mean the latency is
    changing with adaptation, which is a different and more interesting claim
    and is what this panel is able to show that a single F1 number is not.
    """
    import matplotlib.pyplot as plt
    from .style import apply_publication_style

    apply_publication_style()
    types = [k for k in evolution if not k.startswith('_')]
    if not types:
        raise ValueError('no cell types in this evolution dict')
    t = np.asarray(evolution['_t'], dtype=float)
    K = np.asarray(evolution[types[0]]).shape[1]

    if figsize is None:
        figsize = (3.2 * len(types) + 1.2, 3.6)
    fig, axes = plt.subplots(1, len(types), squeeze=False, figsize=figsize,
                             sharey=True)
    stacked = np.concatenate([np.asarray(evolution[ct]).ravel() for ct in types])
    vmax = float(np.nanpercentile(np.abs(stacked), 98)) or 1.0

    # Time is a categorical row index, not a linear axis: the windows are
    # deliberately uneven in a recovery, and drawing them to scale would give
    # the first seconds a sliver while the question is about them.
    for ax, ct in zip(axes[0], types):
        img = np.tile(np.asarray(evolution[ct]), (1, n_cycles_shown))
        ax.imshow(img, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax,
                  extent=(0, 360 * n_cycles_shown, len(t), 0),
                  interpolation='nearest')
        ax.set_xticks(np.arange(0, 360 * n_cycles_shown + 1, 180))
        ax.set_xlabel('drift phase (deg)')
        ax.set_title(ct, fontsize=10)
    axes[0][0].set_yticks(np.arange(len(t)) + 0.5)
    axes[0][0].set_yticklabels([f'{v:g}' for v in t], fontsize=7)
    axes[0][0].set_ylabel('time since step (s, window centre)')
    fig.colorbar(axes[0][-1].images[0], ax=axes[0][-1], fraction=0.05, pad=0.03,
                 label=cbar_label)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94) if title else None)
    return fig
