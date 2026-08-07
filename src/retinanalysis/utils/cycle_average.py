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
    'response_template',
    'flag_template_outliers',
    'summarize_template_outliers',
    'plot_template_match',
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
                      share_y: bool = False, description: str = 'Cell type:',
                      outlier_cell_ids: Optional[Iterable[int]] = None):
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
    outlier_ids = {int(v) for v in (() if outlier_cell_ids is None
                                    else outlier_cell_ids)}

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
        accepted = np.asarray([
            i for i in idx
            if int(cells.iloc[i]['cell_id']) not in outlier_ids], dtype=int)
        template_idx = accepted if accepted.size else idx
        m = psth[template_idx].mean(axis=0)
        top.plot(t, m, color=colors.get(ct, NEUTRAL_GRAY), lw=1.4)
        n_bad = int(sum(int(cells.iloc[i]['cell_id']) in outlier_ids
                        for i in idx))
        top.set_title(f'{ct}: retained-cell population template '
                      f'(n={len(template_idx)}; {n_bad} outlier'
                      f'{"s" if n_bad != 1 else ""} excluded)', fontsize=9)
        top.set_ylabel('Hz')

        for k, ax in enumerate(flat[n_cols:]):
            if k >= n:
                ax.axis('off')
                continue
            i = idx[k]
            cid = int(cells.iloc[i]['cell_id'])
            is_outlier = cid in outlier_ids
            ax.plot(t, psth[i], color=('crimson' if is_outlier else
                                       colors.get(ct, NEUTRAL_GRAY)),
                    lw=1.0 if is_outlier else 0.7)
            title_kwargs = {'color': 'crimson'} if is_outlier else {}
            ax.set_title(f'{cid}' + (' — OUTLIER' if is_outlier else ''),
                         fontsize=7, pad=1.5, **title_kwargs)
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


def _pick_cross_sections(n_windows: int, n_wanted: int) -> np.ndarray:
    """Evenly spaced window indices, always including the first and the last."""
    n_wanted = int(max(2, min(n_wanted, n_windows)))
    return np.unique(np.linspace(0, n_windows - 1, n_wanted).round().astype(int))


def plot_cycle_evolution(evolution: Dict, *,
                         cmap: str = 'RdBu_r',
                         n_cycles_shown: int = 2,
                         cbar_label: str = 'modulation, fraction of own mean',
                         cross_sections: Optional[Sequence[int]] = None,
                         n_cross_sections: int = 5,
                         cross_section_cmap: str = 'viridis',
                         share_curve_y: bool = True,
                         figsize: Optional[Tuple[float, float]] = None,
                         title: str = ''):
    """Phase against time since the step, one column per cell type.

    **Top row, the heat map.** A vertical band that darkens downward is the
    recovery: the response stays at one phase — it has to, the alignment put
    it there — while its depth grows. A band that *drifts* sideways instead
    would mean the latency is changing with adaptation, which is a different
    and more interesting claim, and is what this panel shows that a single F1
    number cannot.

    **Bottom row, cross-sections through it.** The heat map is good at "does
    the band stay put" and bad at "by how much did it grow" — colour is hard
    to read quantitatively, and the 98th-percentile scale hides whatever sits
    above it. These are the same rows drawn as tuning curves, a few time
    snippets from early (dark) to late (bright), so amplitude is on an axis
    with numbers on it. The curves share a y-axis across cell types, so a type
    that barely modulates looks like one — but a thinly sampled type divides
    by a near-zero mean and can reach values that squash everything else onto
    a sliver of the axis (OffP, 3 cells, does exactly that here). That is
    itself worth seeing once; ``share_curve_y=False`` then gives each type its
    own scale.

    Which rows they come from is marked by a coloured tick at the left edge of
    the heat map, in the same colour as the curve — otherwise the two panels
    are two unrelated pictures. Pass ``cross_sections`` explicitly to choose
    the windows, or ``n_cross_sections`` to change how many the automatic
    choice takes (evenly spaced, always including the first and the last).
    """
    import matplotlib.pyplot as plt
    from .style import NEUTRAL_GRAY, apply_publication_style

    apply_publication_style()
    types = [k for k in evolution if not k.startswith('_')]
    if not types:
        raise ValueError('no cell types in this evolution dict')
    t = np.asarray(evolution['_t'], dtype=float)
    K = np.asarray(evolution[types[0]]).shape[1]

    if cross_sections is None:
        picks = _pick_cross_sections(len(t), n_cross_sections)
    else:
        picks = np.asarray([int(i) for i in cross_sections], dtype=int)
        bad = picks[(picks < 0) | (picks >= len(t))]
        if bad.size:
            raise IndexError(f'cross_sections {bad.tolist()} are outside the '
                             f'{len(t)} windows in this evolution')
    pick_colors = plt.get_cmap(cross_section_cmap)(
        np.linspace(0.05, 0.9, picks.size))

    if figsize is None:
        figsize = (3.2 * len(types) + 1.2, 6.4)
    fig, axes = plt.subplots(2, len(types), squeeze=False, figsize=figsize,
                             gridspec_kw={'height_ratios': [1.35, 1.0]})
    stacked = np.concatenate([np.asarray(evolution[ct]).ravel() for ct in types])
    vmax = float(np.nanpercentile(np.abs(stacked), 98)) or 1.0

    # Phase axis for the curves, tiled to match the heat map above it so the
    # two rows read against one shared x.
    deg = np.arange(K * n_cycles_shown) * 360.0 / K

    # Time is a categorical row index, not a linear axis: the windows are
    # deliberately uneven in a recovery, and drawing them to scale would give
    # the first seconds a sliver while the question is about them.
    for col, ct in enumerate(types):
        rows = np.asarray(evolution[ct], dtype=float)
        ax = axes[0][col]
        ax.imshow(np.tile(rows, (1, n_cycles_shown)),
                  aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax,
                  extent=(0, 360 * n_cycles_shown, len(t), 0),
                  interpolation='nearest')
        ax.set_xticks(np.arange(0, 360 * n_cycles_shown + 1, 180))
        ax.set_xlim(0, 360 * n_cycles_shown)
        ax.set_title(ct, fontsize=10)
        # Rows are windows, and only the first column carries their times.
        # Left as default the other columns tick at raw row indices, which
        # read as seconds and are not.
        ax.set_yticks(np.arange(len(t)) + 0.5)
        ax.tick_params(labelbottom=False, labelleft=False)
        # Tie each curve below to the row it was taken from.
        for c, i in zip(pick_colors, picks):
            ax.plot([0], [i + 0.5], marker='>', markersize=5, color=c,
                    clip_on=False, zorder=5)

        ax = axes[1][col]
        for c, i in zip(pick_colors, picks):
            ax.plot(deg, np.tile(rows[i], n_cycles_shown), lw=1.3, color=c,
                    label=f'{t[i]:g} s')
        ax.axhline(0, color=NEUTRAL_GRAY, lw=0.5)
        ax.set_xticks(np.arange(0, 360 * n_cycles_shown + 1, 180))
        ax.set_xlim(0, 360 * n_cycles_shown)
        ax.set_xlabel('drift phase (deg)')
        if col:
            ax.tick_params(labelleft=False)

    axes[0][0].tick_params(labelleft=True)
    axes[0][0].set_yticklabels([f'{v:g}' for v in t], fontsize=7)
    axes[0][0].set_ylabel('time since step (s, window centre)')
    axes[1][0].set_ylabel(cbar_label)

    if share_curve_y:
        # From the rows actually drawn, not all of them: the limit only has to
        # hold the curves on the page.
        drawn = np.concatenate([np.asarray(evolution[ct], dtype=float)[picks].ravel()
                                for ct in types])
        lo, hi = float(np.nanmin(drawn)), float(np.nanmax(drawn))
        pad = 0.08 * max(hi - lo, 1e-9)
        for ax in axes[1]:
            ax.set_ylim(lo - pad, hi + pad)
    else:
        for col in range(1, len(types)):
            axes[1][col].tick_params(labelleft=True)
    axes[1][0].legend(fontsize=6.5, ncol=2, title='window centre',
                      title_fontsize=6.5, frameon=False)

    fig.colorbar(axes[0][-1].images[0], ax=axes[0][-1], fraction=0.05, pad=0.03,
                 label=cbar_label)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95) if title else None)
    return fig


# ---------------------------------------------------------------------------
# 5. A response template per cell type, and cells that do not match it
# ---------------------------------------------------------------------------

def _zscore_rows(x):
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / np.where(sd == 0, np.nan, sd)


def _aligned_folds(pbr, cycle_mask):
    """Position-rotated, z-scored fold per cell over the selected cycles."""
    K = pbr.n_phase_bins
    x = pbr.counts[cycle_mask].sum(axis=0).T.astype(float)      # (cells, K)
    shifts = position_phase_shift(pbr)
    out = np.empty_like(x)
    for i, s in enumerate(shifts):
        out[i] = np.roll(x[i], -int(s) % K)
    return _zscore_rows(out)


def _row_corr(a, b):
    """Pearson correlation row by row, NaN-safe."""
    a = a - np.nanmean(a, axis=1, keepdims=True)
    b = b - np.nanmean(b, axis=1, keepdims=True)
    na = np.sqrt(np.nansum(a * a, axis=1))
    nb = np.sqrt(np.nansum(b * b, axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.nansum(a * b, axis=1) / np.where((na * nb) == 0, np.nan, na * nb)


def response_template(pbr, *, window: Optional[int] = None,
                      cell_types: Optional[Sequence[str]] = None,
                      min_cells: int = 5,
                      n_splits: int = 25,
                      min_spikes: int = 50,
                      random_seed: Optional[int] = 0):
    """A canonical cycle response per cell type, and how well each cell matches it.

    Every cell of a type saw the same grating — shifted in space by where its
    receptive field sits, and in time by the drift. §8's rotation removes the
    spatial shift, so what is left should be one waveform shared by the type,
    differing between cells only in amplitude and in whatever latency spread
    there is. That makes a **template** well defined, and a cell that does not
    match it is worth looking at.

    **The template is built leave-one-out.** A cell that contributed to the
    template it is scored against is partly correlated with itself, which
    matters enormously at n = 4 and still matters at n = 80. Each cell is
    scored against the mean of every *other* cell of its type.

    **Two ways to fail, and they mean different things.** ``template_r``
    compares the cell as aligned; ``shape_r`` is the best correlation over all
    circular rotations, with ``phase_offset_deg`` saying which one won.

    - Low on both — the cell's response is not this type's response. Bad sort,
      wrong type, or not driven.
    - Low ``template_r`` but high ``shape_r`` — the right waveform at the wrong
      phase. That is not a broken cell, it is a cell in the wrong *place*: an
      offset near 180° is a polarity flip (suspect the type label), and an
      arbitrary offset points at the receptive-field position being wrong,
      which usually means a bad cluster match between the noise chunk and the
      protocol block rather than anything about the cell.

    **``self_r`` is the ceiling.** Split the cycles in half at random and
    correlate the two halves: that is how repeatable the cell's own response
    is, and no match to any template can exceed it. A cell with low
    ``template_r`` *and* low ``self_r`` is merely noisy; low ``template_r``
    with high ``self_r`` is reliably different, which is the interesting case
    and the one worth inspecting rather than deleting.
    ``*_corrected`` divides by ``sqrt(self_r)`` so a quiet cell is not
    penalised for being quiet.

    Returns
    -------
    (templates, match, summary)
        ``templates`` maps cell type to its full (not leave-one-out) mean
        fold, z-scored, length ``n_phase_bins``. ``match`` is one row per
        cell. ``summary`` is one row per type, with ``pc1_frac`` — the share
        of variance the first principal component of the folds explains,
        which is the honest answer to *is there a template here at all*:
        near 1 means the type has one waveform, near 1/n means it does not
        and no per-cell verdict from it is worth anything.
    """
    rng = np.random.default_rng(random_seed)
    K = pbr.n_phase_bins
    sel = (np.ones(pbr.counts.shape[0], dtype=bool) if window is None
           else pbr.sample_window == int(window))
    n_cyc = int(sel.sum())
    if n_cyc < 4:
        raise ValueError(f'need at least 4 cycles to split; got {n_cyc}')

    folds = _aligned_folds(pbr, sel)
    spikes = pbr.counts[sel].sum(axis=(0, 1))
    types = pbr.cells['cell_type'].to_numpy()
    present = ([ct for ct in cell_types if (types == ct).any()]
               if cell_types is not None else sorted(set(types)))

    # Split-half reliability, averaged over random halves of the cycles.
    idx = np.flatnonzero(sel)
    acc = np.zeros(folds.shape[0])
    n_ok = np.zeros(folds.shape[0])
    for _ in range(int(n_splits)):
        perm = rng.permutation(idx.size)
        a = np.zeros(pbr.counts.shape[0], bool)
        b = np.zeros(pbr.counts.shape[0], bool)
        a[idx[perm[:idx.size // 2]]] = True
        b[idx[perm[idx.size // 2:]]] = True
        r = _row_corr(_aligned_folds(pbr, a), _aligned_folds(pbr, b))
        ok = np.isfinite(r)
        acc[ok] += r[ok]
        n_ok[ok] += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        self_r = acc / np.where(n_ok == 0, np.nan, n_ok)

    templates, rows, summary_rows = {}, [], []
    for ct in present:
        m = np.flatnonzero(types == ct)
        X = folds[m]
        usable = m.size >= int(min_cells)

        full = np.nanmean(X, axis=0)
        templates[ct] = ((full - np.nanmean(full)) / np.nanstd(full)
                         if np.nanstd(full) > 0 else full)

        # Leave-one-out template: (total - self) / (n - 1).
        if m.size >= 2:
            tot = np.nansum(X, axis=0)
            loo = _zscore_rows((tot - X) / (m.size - 1))
            fixed = _row_corr(X, loo)
            best = np.full(m.size, np.nan)
            off = np.zeros(m.size, dtype=int)
            for k in range(K):
                c = _row_corr(np.roll(X, k, axis=1), loo)
                upd = np.isfinite(c) & ~(c <= best)
                best[upd] = c[upd]
                off[upd] = k
        else:
            fixed = np.full(m.size, np.nan)
            best = np.full(m.size, np.nan)
            off = np.zeros(m.size, dtype=int)

        # How one-dimensional is this type's set of folds?
        good = np.isfinite(X).all(axis=1)
        if good.sum() >= 2:
            sv = np.linalg.svd(X[good], compute_uv=False)
            pc1 = float(sv[0] ** 2 / np.sum(sv ** 2))
        else:
            pc1 = np.nan

        deg = ((off * 360.0 / K) + 180.0) % 360.0 - 180.0
        sr = self_r[m]
        with np.errstate(invalid='ignore', divide='ignore'):
            denom = np.sqrt(np.clip(sr, 1e-6, None))
        for j, i in enumerate(m):
            rows.append({
                'cell_id': int(pbr.cells['cell_id'].iloc[i]),
                'cell_type': ct,
                'axis_px': float(pbr.cells['axis_px'].iloc[i]),
                'n_spikes': float(spikes[i]),
                'self_r': float(sr[j]),
                'template_r': float(fixed[j]),
                'shape_r': float(best[j]),
                'phase_offset_deg': float(deg[j]),
                'template_r_corrected': float(fixed[j] / denom[j]),
                'shape_r_corrected': float(best[j] / denom[j]),
                'enough_spikes': bool(spikes[i] >= int(min_spikes)),
                'type_usable': bool(usable),
            })
        summary_rows.append({
            'cell_type': ct, 'n_cells': int(m.size),
            'median_self_r': float(np.nanmedian(sr)),
            'median_template_r': float(np.nanmedian(fixed)),
            'median_shape_r': float(np.nanmedian(best)),
            'pc1_frac': pc1,
            'usable': usable,
        })

    import pandas as pd
    match = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    match.attrs['condition'] = pbr.condition
    summary.attrs['condition'] = pbr.condition
    summary.attrs['n_cycles'] = n_cyc
    return templates, match, summary


def flag_template_outliers(match, *, min_shape_r: float = 0.5,
                           max_phase_offset_deg: Optional[float] = None,
                           use_corrected: bool = True,
                           min_self_r: float = 0.0,
                           verbose: bool = True):
    """Mark cells whose response does not have their type's waveform.

    **One criterion decides it**, so a rejection has one cause: the cell's
    reliable response must correlate with its type's leave-one-out template at
    ``min_shape_r`` **allowing any phase** — that is, it must have the right
    *shape*. Phase disagreement is reported next to it rather than folded into
    the same number, because the two have different causes and only one of
    them is about the cell (see :func:`response_template`). Set
    ``max_phase_offset_deg`` to reject on phase as well, knowing that at a
    type whose phases do not track position it will reject most of the type
    for a population-level reason.

    Cells of a type flagged ``type_usable=False`` — too few cells for a
    template — are **kept and marked**, not dropped. There is no evidence
    about them either way, and silently deleting a whole small type is worse
    than carrying it with a caveat.

    Returns ``match`` with ``passes_template``, ``reject_reason`` and
    ``phase_ok`` added. It does not filter anything; the caller decides.
    """
    out = match.copy()
    col = 'shape_r_corrected' if use_corrected else 'shape_r'
    shape = out[col].to_numpy(dtype=float)
    self_r = out['self_r'].to_numpy(dtype=float)
    usable = out['type_usable'].to_numpy(dtype=bool)
    enough = out['enough_spikes'].to_numpy(dtype=bool)

    phase_ok = (np.ones(len(out), dtype=bool) if max_phase_offset_deg is None
                else np.abs(out['phase_offset_deg'].to_numpy(dtype=float))
                <= float(max_phase_offset_deg))

    reason = np.full(len(out), '', dtype=object)
    bad_shape = usable & enough & np.isfinite(shape) & (shape < min_shape_r)
    reason[bad_shape] = 'shape'
    quiet = usable & ~enough
    reason[quiet] = 'too few spikes'
    noisy = usable & enough & (self_r < float(min_self_r))
    reason[noisy] = 'unreliable'
    bad_phase = usable & enough & ~bad_shape & ~phase_ok
    reason[bad_phase] = 'phase'
    reason[~usable] = 'type too small to template'

    passes = usable & enough & phase_ok & np.isfinite(shape) \
        & (shape >= min_shape_r) & (self_r >= float(min_self_r))
    evaluable = (usable & enough & np.isfinite(shape)
                 & (self_r >= float(min_self_r)))
    out['template_evaluable'] = evaluable
    out['passes_template'] = passes
    out['phase_ok'] = phase_ok
    out['reject_reason'] = reason

    if verbose:
        print(f'template QC on {col} >= {min_shape_r}'
              + ('' if max_phase_offset_deg is None
                 else f', |phase offset| <= {max_phase_offset_deg:g} deg'))
        for ct, sub in out.groupby('cell_type'):
            if not sub['type_usable'].iloc[0]:
                print(f'  {ct:5} {len(sub):3d} cells — no template '
                      f'(type too small); all kept, none endorsed')
                continue
            kept = int(sub['passes_template'].sum())
            counts = sub.loc[~sub['passes_template'], 'reject_reason'] \
                        .value_counts().to_dict()
            detail = ', '.join(f'{v} {k}' for k, v in counts.items())
            print(f'  {ct:5} {kept:3d}/{len(sub):3d} kept'
                  + (f'   ({detail})' if detail else ''))
        # The cells worth a human look: reliably different, not merely noisy.
        odd = out[(~out['passes_template']) & (out['reject_reason'] == 'shape')
                  & (out['self_r'] > 0.8)]
        if len(odd):
            print(f'  {len(odd)} rejected cells are highly repeatable '
                  f'(self_r > 0.8) — reliably different rather than noisy, '
                  f'so look at them before deleting them')
    return out


def summarize_template_outliers(match, *, candidate_cell_ids=None,
                                min_conditions: int = 2,
                                min_pass_fraction: float = 0.5,
                                verbose: bool = True):
    """Turn per-condition template matches into one downstream cell filter.

    A cell is excluded only when it could be judged in at least
    ``min_conditions`` and it matches its type template in less than
    ``min_pass_fraction`` of those conditions.  Cells outside the aperture,
    cells belonging to a type too small to template, and cells without enough
    spikes are retained: absence of evidence is not treated as an outlier.

    Parameters
    ----------
    match : DataFrame
        Concatenated output of :func:`flag_template_outliers`, with a
        ``condition`` column.
    candidate_cell_ids : iterable[int], optional
        The cells entering template QC. Supplying this preserves candidates
        that were not evaluable in any condition in the returned summary.

    Returns
    -------
    (condition_match, cell_summary)
        The first table adds ``excluded_downstream`` to every condition row.
        The second has one row per candidate cell and is the compact QC table
        intended for date-level persistence.
    """
    required = {'cell_id', 'cell_type', 'condition', 'passes_template',
                'template_evaluable'}
    missing = required.difference(match.columns)
    if missing:
        raise KeyError(f'template match missing columns: {sorted(missing)}')
    if int(min_conditions) < 1:
        raise ValueError('min_conditions must be at least 1')
    if not 0 <= float(min_pass_fraction) <= 1:
        raise ValueError('min_pass_fraction must be between 0 and 1')

    rows = []
    for (cid, ct), sub in match.groupby(['cell_id', 'cell_type'], sort=True):
        scored = sub[sub['template_evaluable'].astype(bool)]
        n_eval = len(scored)
        n_pass = int(scored['passes_template'].sum())
        fraction = n_pass / n_eval if n_eval else np.nan
        excluded = (n_eval >= int(min_conditions)
                    and fraction < float(min_pass_fraction))
        rows.append({
            'cell_id': int(cid),
            'cell_type': ct,
            'n_conditions_seen': int(sub['condition'].nunique()),
            'n_conditions_evaluable': int(n_eval),
            'n_conditions_passed': n_pass,
            'template_pass_fraction': float(fraction),
            'excluded_downstream': bool(excluded),
            'qc_status': ('outlier' if excluded else
                          ('kept_unscored' if n_eval == 0 else 'kept')),
        })
    summary = pd.DataFrame(rows)

    if candidate_cell_ids is not None:
        candidates = pd.DataFrame({
            'cell_id': sorted({int(v) for v in candidate_cell_ids})})
        summary = candidates.merge(summary, on='cell_id', how='left')
        unseen = summary['n_conditions_seen'].isna()
        summary.loc[unseen, 'cell_type'] = 'unknown'
        for col in ('n_conditions_seen', 'n_conditions_evaluable',
                    'n_conditions_passed'):
            summary[col] = summary[col].fillna(0).astype(int)
        summary['excluded_downstream'] = \
            summary['excluded_downstream'].fillna(False).astype(bool)
        summary.loc[unseen, 'qc_status'] = 'kept_unscored'

    excluded_ids = set(summary.loc[summary['excluded_downstream'],
                                   'cell_id'].astype(int))
    annotated = match.copy()
    annotated['excluded_downstream'] = \
        annotated['cell_id'].astype(int).isin(excluded_ids)
    annotated.attrs.update(match.attrs)
    summary.attrs.update({
        'min_conditions': int(min_conditions),
        'min_pass_fraction': float(min_pass_fraction),
    })

    if verbose:
        n_unscored = int((summary['qc_status'] == 'kept_unscored').sum())
        print(f'cross-condition template QC: {len(summary) - len(excluded_ids)}/'
              f'{len(summary)} cells retained; {len(excluded_ids)} outliers '
              f'excluded downstream; {n_unscored} retained without enough '
              'template evidence')
        if excluded_ids:
            print('  outlier cell IDs: '
                  + ', '.join(str(v) for v in sorted(excluded_ids)))
    return annotated, summary


def plot_template_match(templates, match, *, cell_types=None,
                        min_shape_r: float = 0.5,
                        figsize: Optional[Tuple[float, float]] = None):
    """Templates, the reliability-vs-match plane, and where the misfits sit.

    **Left**: each type's template. **Middle**: every cell as
    ``self_r`` (how repeatable it is) against ``template_r`` (how well it
    matches as aligned), which separates *noisy* from *reliably different* —
    the bottom-left is noise, the bottom-right is a cell that reproduces a
    response its type does not share. **Right**: the phase offset that would
    rescue each poorly-matched cell; a spike at ±180° is polarity, a spread is
    receptive-field positions that are wrong.
    """
    import matplotlib.pyplot as plt
    from .style import NEUTRAL_GRAY, apply_publication_style, colors_for_celltypes

    apply_publication_style()
    present = ([ct for ct in cell_types if ct in templates]
               if cell_types is not None else sorted(templates))
    colors = colors_for_celltypes(present)
    if figsize is None:
        figsize = (13.0, 3.8)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    K = len(next(iter(templates.values())))
    deg = np.arange(K) * 360.0 / K
    for ct in present:
        n = int((match['cell_type'] == ct).sum())
        axes[0].plot(deg, templates[ct], lw=1.6, color=colors.get(ct, NEUTRAL_GRAY),
                     label=f'{ct} (n={n})')
    axes[0].axhline(0, color=NEUTRAL_GRAY, lw=0.5)
    axes[0].set_xlabel('drift phase (deg)')
    axes[0].set_ylabel('z-scored rate')
    axes[0].set_title('type template', fontsize=10)
    axes[0].legend(fontsize=7)

    excluded = (match['excluded_downstream'].astype(bool)
                if 'excluded_downstream' in match else
                ~match.get('passes_template', pd.Series(True, index=match.index))
                    .astype(bool))
    for ct in present:
        s = match[match['cell_type'] == ct]
        axes[1].scatter(s['self_r'], s['template_r'], s=14, alpha=0.75,
                        color=colors.get(ct, NEUTRAL_GRAY), label=ct)
        bad = s[excluded.loc[s.index]]
        axes[1].scatter(bad['self_r'], bad['template_r'], s=34, marker='x',
                        color='crimson', linewidths=1.0)
        for row in bad.itertuples():
            axes[1].annotate(str(int(row.cell_id)),
                             (row.self_r, row.template_r), xytext=(3, 3),
                             textcoords='offset points', fontsize=6,
                             color='crimson')
    axes[1].axhline(min_shape_r, color=NEUTRAL_GRAY, ls='--', lw=0.8)
    axes[1].set_xlabel('self_r (split-half reliability)')
    axes[1].set_ylabel('template_r (as aligned)')
    axes[1].set_title('noisy (left) vs reliably different (lower right)',
                      fontsize=10)

    poor = match[match['template_r'] < min_shape_r]
    for ct in present:
        s = poor[poor['cell_type'] == ct]
        if len(s):
            axes[2].scatter(s['phase_offset_deg'], s['shape_r'], s=16, alpha=0.8,
                            color=colors.get(ct, NEUTRAL_GRAY), label=ct)
    for x in (-180, 0, 180):
        axes[2].axvline(x, color=NEUTRAL_GRAY, lw=0.6,
                        ls='-' if x == 0 else '--')
    axes[2].set_xlim(-190, 190)
    axes[2].set_xlabel('phase offset that best matches (deg)')
    axes[2].set_ylabel('shape_r at that offset')
    axes[2].set_title(f'the {len(poor)} cells below template_r '
                      f'{min_shape_r}', fontsize=10)
    if len(poor):
        axes[2].legend(fontsize=7)
    n_excluded = int(excluded.sum())
    if n_excluded:
        fig.suptitle(f'population-template QC — downstream outliers labelled '
                     f'in red (n={n_excluded})', color='crimson', fontsize=10)
    else:
        fig.suptitle('population-template QC — no downstream outliers',
                     fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
