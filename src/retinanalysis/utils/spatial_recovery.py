"""Recovery of the population's sensitivity to spatial structure after a step.

Every epoch of ``variableMeanDriftingGrating`` is a luminance step: the mean
alternates between epochs, so an epoch at mean 0.3 follows 60 s of adaptation
to 0.03. The question this module is built for is what the population does
over the following minute — and specifically whether it recovers a *spatial*
representation on a different timescale than it recovers firing.

On 20230502C/data017 those two go opposite ways, which is the whole point:
population rate **falls** from ~15 to ~2–5 Hz/cell over ~30 s while phase
locking to the drift **rises** (F1/F0 from 0.41 to 0.60 at 150 µm). Reporting
either alone would describe the adaptation backwards.

The layout follows the nesting of the question:

1. :func:`phase_binned_response` — one pass over the spikes that produces
   everything else: per-cycle phase-binned counts for decoding, and exact
   complex Fourier sums for modulation and coherence.
2. :func:`phase_modulation` — per cell, F1/F2 against F0.
3. :func:`decode_phase` — the primary endpoint: cross-validated decoding of
   the grating's phase from the population, full-cycle and modulo π.
4. :func:`mosaic_coherence` — whether the *phases* agree with the cells'
   positions, which single-cell modulation cannot tell you.
5. :func:`split_half_reliability` — whether decoding moved because responses
   got bigger, better aligned, or less variable.
6. :func:`pathway_decoding` — the same, per cell type at matched cell count.
7. :func:`fit_recovery` — a timescale, as τ and t50.
8. :func:`spatial_structure_index` — decoding normalized to its own shuffle
   and its own late-adapted value, so conditions are comparable.

Three biases will manufacture this result if they are not handled, and all
three are handled here rather than left to the caller:

- **Vector strength is biased upward at low spike counts** by about
  ``sqrt(pi/4n)``. Rate falls five-fold across the epoch, so the bias grows
  exactly where the effect is claimed. Everything reports the bias-corrected
  resultant (:func:`corrected_resultant`); on this block the correction
  accounts for roughly a third of the apparent F1 rise, and the rest survives.
- **A spike-count threshold applied per window selects cells.** Apply one late
  in the epoch and you keep only the cells that stayed active, which raises
  every average. The cell set is chosen **once**, over the whole epoch, and
  held fixed across windows.
- **Decoders see more spikes in some windows than others.** Pass
  ``match_spike_counts=True`` to subsample every window down to the sparsest
  before decoding, which is the control that separates "more information" from
  "more spikes".

The biological replicate for any claim about a timescale is the **retina**,
not the cell and not the epoch. One preparation gives a τ with an interval
around it and no inference beyond itself; :func:`fit_recovery` bootstraps over
whatever grouping it is handed, and it is the caller's job to hand it
preparations once there is more than one.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


__all__ = [
    'PhaseBinnedResponse',
    'corrected_resultant',
    'estimate_drift_frequency',
    'recovery_windows',
    'phase_binned_response',
    'phase_modulation',
    'decode_phase',
    'mosaic_coherence',
    'split_half_reliability',
    'pathway_decoding',
    'fit_recovery',
    'spatial_structure_index',
    'analyze_recovery_conditions',
    'recovery_summary_table',
    'normalize_recovery_summary',
    'saved_recovery_stats',
    'save_recovery_summary',
    'save_recovery_cross_date_summary',
    'load_recovery_many',
    'plot_recovery_across_dates',
    'plot_recovery_summary',
]


# ---------------------------------------------------------------------------
# Small shared pieces
# ---------------------------------------------------------------------------

def _wrap_pi(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def corrected_resultant(vector, n):
    """Bias-corrected vector strength from a complex sum and its spike count.

    The naive resultant ``|Z|/n`` is not zero for random phases — it averages
    about ``sqrt(pi/(4n))``, so a cell with 25 spikes looks 0.18 modulated
    while firing at random, and one with 400 looks 0.04. Any analysis that
    compares windows of *different firing rate* therefore has a built-in trend
    in whichever direction the rate moved, and here the rate moves five-fold.

    This returns ``sqrt(max(0, (n R^2 - 1) / (n - 1)))``, the standard
    correction: an unbiased estimate of the squared resultant, floored at zero
    and rooted. Cells with fewer than 2 spikes come back NaN.
    """
    vector = np.asarray(vector)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        r2 = (np.abs(vector) / n) ** 2
        unbiased = (n * r2 - 1.0) / (n - 1.0)
    out = np.sqrt(np.clip(unbiased, 0.0, None))
    return np.where(n >= 2, out, np.nan)


def estimate_drift_frequency(pipeline, stim_block, epoch_indices, *,
                             geometry: Optional[Dict] = None,
                             cell_types: Optional[Iterable[str]] = None,
                             cell_ids: Optional[Iterable[int]] = None,
                             rel_range: float = 0.02,
                             n_candidates: int = 1201,
                             min_spikes: int = 400,
                             verbose: bool = True) -> Dict:
    """Recover the grating's true drift frequency from the spikes.

    **The nominal frequency is usually not the real one, and on a 60 s epoch
    the difference is not small.** Stage advances the grating's phase once per
    rendered frame, by an increment computed from the *declared* refresh rate;
    the display then runs at whatever rate it actually runs at. On this rig
    the recorded frame times give 60.31 Hz against a declared 60, so a
    nominal 2 Hz grating drifts at 2.01 Hz — and folding 60 s of spikes at
    2.0000 Hz accumulates 214° of phase error from start to end, which
    destroys about half the measured modulation and, in a per-window cycle
    average, imitates a latency that drifts with adaptation.

    This scans candidate frequencies for the one maximising pooled vector
    strength. Each cell's F1 is computed at every candidate and the magnitudes
    averaged across cells, so cells at different positions — which are in
    different phases and would cancel if summed — all contribute.

    Parameters
    ----------
    epoch_indices : sequence[int]
        Epochs of one condition. Longer epochs sharpen the estimate: the
        resolvable frequency difference is roughly ``1 / total_duration``.
    rel_range : float
        Search ``nominal * (1 +/- rel_range)``. The default 2% comfortably
        covers a frame-rate mismatch, which is a fraction of a percent.
    min_spikes : int
        Cells below this (pooled over the epochs) are left out of the average.

    Returns
    -------
    dict
        ``drift_freq_hz`` (the estimate), ``nominal_hz``, ``ratio``,
        ``strength_at_estimate``, ``strength_at_nominal``,
        ``implied_frame_rate_hz``, ``phase_error_deg_per_epoch``, plus the
        ``candidates`` and ``strength`` arrays for plotting.
    """
    from ..regen.variable_mean_drifting_grating import grating_geometry
    from .mosaic_overlay import cell_activity_in_window

    epochs = [int(e) for e in epoch_indices]
    g = geometry or grating_geometry(stim_block, epochs[0])
    nominal = float(g['temporal_freq_hz'])
    pre_s = float(g['pre_time_ms']) / 1000.0
    stim_s = float(g['stim_time_ms']) / 1000.0

    per_cell: Dict[int, List[np.ndarray]] = {}
    for e in epochs:
        table = cell_activity_in_window(pipeline, e, (pre_s, pre_s + stim_s),
                                        cell_types=cell_types, cell_ids=cell_ids)
        for row in table.itertuples():
            per_cell.setdefault(int(row.cell_id), []).append(
                np.asarray(row.spike_times_s, dtype=float) - pre_s)

    usable = [v for v in per_cell.values()
              if sum(x.size for x in v) >= int(min_spikes)]
    if not usable:
        raise ValueError(f'no cell reached {min_spikes} spikes across '
                         f'{len(epochs)} epochs; lower min_spikes')

    fs = np.linspace(nominal * (1 - rel_range), nominal * (1 + rel_range),
                     int(n_candidates))
    score = np.zeros(fs.size)
    for spikes in usable:
        acc = np.zeros(fs.size, dtype=complex)
        n = 0
        for s in spikes:
            if s.size:
                acc += np.exp(1j * 2 * np.pi * np.outer(fs, s)).sum(axis=1)
                n += s.size
        if n:
            score += np.abs(acc) / n
    score /= len(usable)

    best = float(fs[int(np.argmax(score))])
    at_nominal = float(np.interp(nominal, fs, score))
    out = {
        'drift_freq_hz': best,
        'nominal_hz': nominal,
        'ratio': best / nominal,
        'strength_at_estimate': float(score.max()),
        'strength_at_nominal': at_nominal,
        'implied_frame_rate_hz': (float(g.get('monitor_refresh_hz', 60.0))
                                  * best / nominal),
        'phase_error_deg_per_epoch': 360.0 * (best - nominal) * stim_s,
        'n_cells': len(usable),
        'candidates': fs,
        'strength': score,
    }
    if verbose:
        print(f'drift frequency {best:.4f} Hz against a nominal {nominal:g} '
              f'({out["ratio"]:.5f}x), from {len(usable)} cells\n'
              f'  pooled vector strength {at_nominal:.3f} at nominal, '
              f'{score.max():.3f} at the estimate\n'
              f'  folding at the nominal value would accumulate '
              f'{out["phase_error_deg_per_epoch"]:.0f} deg over one '
              f'{stim_s:g} s epoch')
    return out


def recovery_windows(edges_s: Sequence[float] = (0, 2, 5, 10, 20, 30, 45, 60)):
    """Consecutive windows from a list of edges, as ``[(t0, t1), ...]``.

    The default is the log-ish spacing a recovery wants: fine where it is
    changing fastest and coarse where it has settled, so every window carries
    a usable number of drift cycles without smearing the early transient.
    """
    edges = [float(e) for e in edges_s]
    if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f'edges_s must be increasing with >= 2 entries; '
                         f'got {edges_s}')
    return [(a, b) for a, b in zip(edges, edges[1:])]


def sliding_windows(duration_s: float = 60.0, width_s: float = 5.0,
                    step_s: float = 2.5):
    """Overlapping windows, for fitting a continuous recovery curve."""
    starts = np.arange(0.0, duration_s - width_s + 1e-9, step_s)
    return [(float(s), float(s + width_s)) for s in starts]


# ---------------------------------------------------------------------------
# 1. The one pass over spikes that everything else reads
# ---------------------------------------------------------------------------

@dataclass
class PhaseBinnedResponse:
    """Population response to one condition, binned by drift phase and window.

    Attributes
    ----------
    counts : np.ndarray
        ``(n_samples, n_phase_bins, n_cells)`` spike counts. A *sample* is one
        drift cycle of one epoch — the repeat unit for decoding, since the
        grating returns to the same configuration every cycle.
    sample_window, sample_epoch, sample_cycle : np.ndarray
        ``(n_samples,)`` provenance. ``sample_epoch`` is what
        cross-validation must be grouped on: cycles within an epoch share an
        adaptation state and are not independent repeats of it.
    z1, z2 : np.ndarray
        ``(n_windows, n_cells)`` complex Fourier sums at the drift frequency
        and its second harmonic, accumulated from exact spike times rather
        than from ``counts`` — binning attenuates the harmonics and there is
        no reason to pay that where the spike times are in hand.
    n_spikes : np.ndarray
        ``(n_windows, n_cells)``, the ``F0`` those sums divide by.
    exposure_s : np.ndarray
        ``(n_windows,)`` seconds pooled per window, summed over epochs.
    cells : pandas.DataFrame
        One row per cell, in the column order of ``counts``: ``cell_id``,
        ``noise_id``, ``cell_type``, RF centre and ellipse in canvas pixels,
        ``axis_px`` (position across the bars) and ``inside_aperture``.
    """

    counts: np.ndarray
    sample_window: np.ndarray
    sample_epoch: np.ndarray
    sample_cycle: np.ndarray
    z1: np.ndarray
    z2: np.ndarray
    n_spikes: np.ndarray
    exposure_s: np.ndarray
    cells: pd.DataFrame
    windows: List[Tuple[float, float]]
    epochs: List[int]
    geometry: Dict
    n_phase_bins: int
    condition: Dict = field(default_factory=dict)
    # The frequency the folding actually used. Not always the geometry's
    # nominal value — see estimate_drift_frequency — so it is recorded here
    # rather than re-derived, and every consumer reads this field.
    drift_freq_hz: float = float('nan')

    @property
    def cycle_s(self) -> float:
        return 1.0 / float(self.drift_freq_hz)

    @property
    def window_centers(self) -> np.ndarray:
        return np.array([(a + b) / 2.0 for a, b in self.windows])

    @property
    def window_labels(self) -> List[str]:
        return [f'{a:g}–{b:g}s' for a, b in self.windows]

    def __repr__(self):
        return (f'<PhaseBinnedResponse {len(self.cells)} cells x '
                f'{len(self.windows)} windows x {self.n_phase_bins} phase bins, '
                f'{self.counts.shape[0]} cycles from {len(self.epochs)} epochs, '
                f'{self.condition}>')


def phase_binned_response(
    pipeline,
    stim_block,
    epoch_indices: Iterable[int],
    *,
    windows_s: Optional[Sequence[Tuple[float, float]]] = None,
    n_phase_bins: int = 12,
    cell_types: Optional[Iterable[str]] = None,
    cell_ids: Optional[Iterable[int]] = None,
    min_spikes_per_epoch: float = 30.0,
    inside_aperture_only: bool = True,
    std_scaling: float = 1.6,
    geometry: Optional[Dict] = None,
    drift_freq_hz: Optional[float] = None,
    verbose: bool = True,
) -> PhaseBinnedResponse:
    """Bin one condition's spikes by drift phase, per cycle and per window.

    **Only epochs of one condition may be pooled.** Drift phase is measured
    from stimulus onset, so epochs that ran the same geometry are in register
    and their cycles are repeats of each other; epochs of a different bar
    width are a different stimulus and their phases mean something else. This
    checks and raises rather than averaging them together.

    Parameters
    ----------
    pipeline : MEAPipeline
    stim_block : MEAStimBlock
        Read for each epoch's own recorded parameters.
    epoch_indices : sequence[int]
        Epochs of a single condition. Each contributes its own cycles.
    windows_s : sequence of (t0, t1), optional
        Time since epoch onset. Default :func:`recovery_windows`. Windows may
        overlap. Only whole drift cycles falling inside a window are used, so
        a window shorter than one cycle contributes nothing.
    n_phase_bins : int
        Bins per drift cycle. Must be even for modulo-π decoding to split it.
    cell_types, cell_ids : optional
        Restrict the population.
    min_spikes_per_epoch : float
        Cells averaging fewer spikes than this **over the whole epoch** are
        dropped. Chosen on the whole epoch and applied to every window, so the
        population is the same one at 1 s and at 55 s — a per-window threshold
        would silently select the cells that stayed active and manufacture a
        rise in every average that follows.
    inside_aperture_only : bool
        Drop cells whose RF centre lies outside the aperture. They saw the
        black surround, not the grating, so they carry no phase.

    Returns
    -------
    PhaseBinnedResponse
    """
    from ..regen.variable_mean_drifting_grating import grating_geometry
    from .mosaic_overlay import cell_activity_in_window
    from .response_phase import _axis_position

    epochs = [int(e) for e in epoch_indices]
    if not epochs:
        raise ValueError('epoch_indices is empty')

    geoms = [grating_geometry(stim_block, e) for e in epochs]
    g = geometry or geoms[0]
    keys = ('bar_width_px', 'temporal_freq_hz', 'orientation_deg',
            'mean_intensity', 'aperture_diameter_px')
    for e, gi in zip(epochs, geoms):
        differing = {k: (g[k], gi[k]) for k in keys if gi[k] != g[k]}
        if differing:
            raise ValueError(
                f'epoch {e} ran a different condition than epoch {epochs[0]} '
                f'({differing}). Drift phase is measured from stimulus onset, '
                f'so only epochs of one condition are in register — group '
                f'them by condition and call this once per group.')

    # The nominal frequency is the default, not the truth. Stage advances the
    # grating once per rendered frame using the declared refresh rate, so a
    # display running fast drifts the grating fast — 2 Hz becomes 2.01 Hz here,
    # which is 214 deg of accumulated phase over a 60 s epoch and about half
    # the modulation thrown away. Pass the output of
    # estimate_drift_frequency() to fold at the frequency the retina saw.
    freq = float(g['temporal_freq_hz'] if drift_freq_hz is None
                 else drift_freq_hz)
    pre_s = float(g['pre_time_ms']) / 1000.0
    stim_s = float(g['stim_time_ms']) / 1000.0
    cycle_s = 1.0 / freq

    windows = ([tuple(float(v) for v in w) for w in windows_s]
               if windows_s is not None else recovery_windows())
    for (a, b) in windows:
        if b <= a:
            raise ValueError(f'window {(a, b)} is not increasing')
        if b > stim_s + 1e-9:
            raise ValueError(f'window {(a, b)} runs past the {stim_s:g} s '
                             f'stimulus')

    # One read per epoch over the whole stimulus, then windowed in memory.
    per_epoch = {}
    for e in epochs:
        per_epoch[e] = cell_activity_in_window(
            pipeline, e, (pre_s, pre_s + stim_s),
            cell_types=cell_types, cell_ids=cell_ids, std_scaling=std_scaling)

    base = per_epoch[epochs[0]]
    keep = np.ones(len(base), dtype=bool)
    keep &= base['center_x'].notna().to_numpy()

    total = np.zeros(len(base))
    for e in epochs:
        total += per_epoch[e]['n_spikes'].to_numpy()
    keep &= (total / len(epochs)) >= float(min_spikes_per_epoch)

    cells = base.loc[keep, ['cell_id', 'noise_id', 'cell_type', 'center_x',
                            'center_y', 'width', 'height', 'angle']].copy()
    cells['axis_px'] = _axis_position(cells['center_x'], cells['center_y'], g)
    cells['radius_px'] = np.hypot(cells['center_x'] - float(g['center_x']),
                                  cells['center_y'] - float(g['center_y']))
    cells['inside_aperture'] = (cells['radius_px']
                                <= float(g['aperture_diameter_px']) / 2.0)
    if inside_aperture_only:
        cells = cells[cells['inside_aperture']]
    cells = cells.reset_index(drop=True)
    if cells.empty:
        raise ValueError('No cell survived the selection — loosen '
                         'min_spikes_per_epoch or inside_aperture_only.')

    keep_ids = cells['cell_id'].astype(int).tolist()
    n_cells = len(keep_ids)
    n_win = len(windows)

    z1 = np.zeros((n_win, n_cells), dtype=complex)
    z2 = np.zeros((n_win, n_cells), dtype=complex)
    nspk = np.zeros((n_win, n_cells))
    exposure = np.zeros(n_win)

    counts_rows, s_win, s_ep, s_cyc = [], [], [], []

    for e in epochs:
        table = per_epoch[e].set_index('cell_id')
        spikes_by_cell = [
            np.asarray(table.at[cid, 'spike_times_s'], dtype=float) - pre_s
            for cid in keep_ids]

        for wi, (w0, w1) in enumerate(windows):
            exposure[wi] += (w1 - w0)
            # Cycles are counted from stimulus onset so that phase bin k means
            # the same stimulus configuration in every cycle and every epoch.
            first = int(np.ceil(w0 / cycle_s - 1e-9))
            last = int(np.floor(w1 / cycle_s + 1e-9)) - 1
            for ci, spikes in enumerate(spikes_by_cell):
                s = spikes[(spikes >= w0) & (spikes < w1)]
                if s.size:
                    ph = 2 * np.pi * freq * s
                    z1[wi, ci] += np.exp(1j * ph).sum()
                    z2[wi, ci] += np.exp(2j * ph).sum()
                nspk[wi, ci] += s.size

            for cyc in range(first, last + 1):
                c0 = cyc * cycle_s
                block = np.zeros((n_phase_bins, n_cells))
                for ci, spikes in enumerate(spikes_by_cell):
                    s = spikes[(spikes >= c0) & (spikes < c0 + cycle_s)]
                    if s.size:
                        k = np.minimum(
                            ((s - c0) / cycle_s * n_phase_bins).astype(int),
                            n_phase_bins - 1)
                        np.add.at(block[:, ci], k, 1.0)
                counts_rows.append(block)
                s_win.append(wi)
                s_ep.append(e)
                s_cyc.append(cyc)

    if not counts_rows:
        raise ValueError('No whole drift cycle fell inside any window — the '
                         f'cycle is {cycle_s:g} s, so windows must be at '
                         f'least that wide.')

    pbr = PhaseBinnedResponse(
        counts=np.stack(counts_rows),
        sample_window=np.asarray(s_win),
        sample_epoch=np.asarray(s_ep),
        sample_cycle=np.asarray(s_cyc),
        z1=z1, z2=z2, n_spikes=nspk, exposure_s=exposure,
        cells=cells, windows=windows, epochs=epochs, geometry=g,
        n_phase_bins=int(n_phase_bins),
        condition={'bar_width_um': g['bar_width_um'],
                   'mean_intensity': g['mean_intensity']},
        drift_freq_hz=freq,
    )

    if verbose:
        dropped = int(keep.sum()) - n_cells
        per_win = np.bincount(pbr.sample_window, minlength=n_win)
        if drift_freq_hz is not None:
            print(f'folding at {freq:.4f} Hz (nominal '
                  f'{g["temporal_freq_hz"]:g})')
        print(f'{n_cells} cells x {n_win} windows, '
              f'{g["bar_width_um"]:g} um bars at mean {g["mean_intensity"]:g}, '
              f'{len(epochs)} epochs\n'
              f'  cells: {int(keep.sum())} above '
              f'{min_spikes_per_epoch:g} spikes/epoch'
              + (f', {dropped} of them outside the aperture and dropped'
                 if dropped else '')
              + f'\n  cycles per window: '
              + ', '.join(f'{lab} {n}' for lab, n
                          in zip(pbr.window_labels, per_win)))
    return pbr


# ---------------------------------------------------------------------------
# 2. Single-cell modulation
# ---------------------------------------------------------------------------

def phase_modulation(pbr: PhaseBinnedResponse, *,
                     late_windows: int = 1,
                     debias: bool = True) -> pd.DataFrame:
    """Per cell and window: F0, and F1/F2 normalized two ways.

    ``m1``/``m2`` are the bias-corrected resultants — modulation as a fraction
    of the cell's own firing, which is the normalization that keeps a rate
    change from reading as a sensitivity change. ``r1``/``r2`` renormalize
    those to the cell's late-adapted value, which is the recovery trajectory:
    1.0 means "back to where this cell ends up".

    ``debias=False`` returns the naive ``|Z|/n`` instead, which is what makes
    the correction's size visible — on this block it accounts for about a
    third of the apparent F1 rise.

    ``f2_over_f1`` is the frequency-doubling measure. A receptive field
    spanning several bars is driven twice per cycle, so a fine grating should
    push this up relative to a coarse one; it is the honest form of that
    comparison, because raw F2 also rises whenever F1 does.
    """
    n = pbr.n_spikes
    if debias:
        m1 = corrected_resultant(pbr.z1, n)
        m2 = corrected_resultant(pbr.z2, n)
    else:
        with np.errstate(invalid='ignore', divide='ignore'):
            m1, m2 = np.abs(pbr.z1) / n, np.abs(pbr.z2) / n
        m1 = np.where(n >= 2, m1, np.nan)
        m2 = np.where(n >= 2, m2, np.nan)

    late = slice(len(pbr.windows) - int(late_windows), len(pbr.windows))
    with warnings.catch_warnings():
        # A cell too quiet to have a resultant in every late window is NaN
        # there; that is a legitimate "no estimate", not a condition to warn on.
        warnings.simplefilter('ignore', RuntimeWarning)
        base1 = np.nanmean(m1[late], axis=0)
        base2 = np.nanmean(m2[late], axis=0)

    rows = []
    for wi, (w0, w1) in enumerate(pbr.windows):
        rate = n[wi] / pbr.exposure_s[wi]
        # F2/F1 is only meaningful where F1 itself is established. Below the
        # Rayleigh 5% threshold (R ~ sqrt(3/n)) the denominator is noise, and
        # dividing by it turns unmodulated cells into ratios of 2 and 50 that
        # then dominate any average. Those cells get NaN instead.
        with np.errstate(invalid='ignore', divide='ignore'):
            f1_floor = np.sqrt(3.0 / np.where(n[wi] > 0, n[wi], np.nan))
            resolved = m1[wi] > f1_floor
        with np.errstate(invalid='ignore', divide='ignore'):
            rows.append(pd.DataFrame({
                'window': pbr.window_labels[wi],
                't_mid': (w0 + w1) / 2.0,
                't_start': w0, 't_end': w1,
                'cell_id': pbr.cells['cell_id'].to_numpy(),
                'cell_type': pbr.cells['cell_type'].to_numpy(),
                'axis_px': pbr.cells['axis_px'].to_numpy(),
                'n_spikes': n[wi],
                'f0_hz': rate,
                'm1': m1[wi], 'm2': m2[wi],
                'r1': m1[wi] / base1, 'r2': m2[wi] / base2,
                'f1_resolved': resolved,
                'f2_over_f1': np.where(resolved, m2[wi] / m1[wi], np.nan),
                'resp_phase_rad': np.angle(pbr.z1[wi]),
            }))
    out = pd.concat(rows, ignore_index=True)
    out.attrs['debiased'] = bool(debias)
    out.attrs['condition'] = pbr.condition
    return out


# ---------------------------------------------------------------------------
# 3. Population phase decoding — the primary endpoint
# ---------------------------------------------------------------------------

def _fold_templates(x, y, n_classes):
    """Class-mean templates, NaN for classes absent from the training fold."""
    t = np.full((n_classes, x.shape[1]), np.nan)
    for k in range(n_classes):
        m = y == k
        if m.any():
            t[k] = x[m].mean(axis=0)
    return t


def _classify(x, templates, metric):
    ok = ~np.isnan(templates).any(axis=1)
    t = templates[ok]
    idx = np.flatnonzero(ok)
    if t.shape[0] == 0:
        return np.zeros(x.shape[0], dtype=int)
    if metric == 'correlation':
        # Correlation ignores overall gain, so a window is not favoured merely
        # for having more spikes in it.
        xc = x - x.mean(axis=1, keepdims=True)
        tc = t - t.mean(axis=1, keepdims=True)
        xn = np.linalg.norm(xc, axis=1, keepdims=True)
        tn = np.linalg.norm(tc, axis=1, keepdims=True)
        score = (xc / np.where(xn == 0, 1, xn)) @ (tc / np.where(tn == 0, 1, tn)).T
    elif metric == 'euclidean':
        score = -((x[:, None, :] - t[None, :, :]) ** 2).sum(axis=2)
    elif metric == 'poisson':
        lam = np.clip(t, 1e-6, None)
        score = x @ np.log(lam).T - lam.sum(axis=1)[None, :]
    else:
        raise ValueError(f'unknown metric {metric!r}')
    return idx[np.argmax(score, axis=1)]


def _circular_stats(pred, true, n_classes, period_classes=None):
    """Accuracy plus circular error over a space of ``period_classes`` bins."""
    period = period_classes or n_classes
    d = 2 * np.pi * (pred - true) / period
    resultant = np.abs(np.mean(np.exp(1j * d)))
    return {
        'accuracy': float(np.mean(pred == true)),
        'circular_resultant': float(resultant),
        'circular_error': float(1.0 - resultant),
        'abs_circular_error_rad': float(np.mean(np.abs(_wrap_pi(d)))),
        'chance_accuracy': 1.0 / period,
        'n_test': int(true.size),
    }


def _window_design(pbr, wi, mode, subtract_mean, rng, target_spikes=None):
    """Observations for one window: (X, labels, groups) ready for CV."""
    sel = np.flatnonzero(pbr.sample_window == wi)
    if sel.size == 0:
        return None
    counts = pbr.counts[sel]                      # (n_cyc, K, n_cells)
    K = pbr.n_phase_bins

    if target_spikes is not None:
        # Thin every spike independently to a common expected count, so that a
        # window is not decodable merely for being denser. Binomial thinning
        # keeps Poisson data Poisson.
        have = counts.sum()
        if have > 0 and target_spikes < have:
            counts = rng.binomial(counts.astype(int),
                                  float(target_spikes) / have).astype(float)

    x = counts.reshape(-1, counts.shape[2])       # (n_cyc*K, n_cells)
    labels = np.tile(np.arange(K), counts.shape[0])
    groups = np.repeat(pbr.sample_epoch[sel], K)

    if subtract_mean:
        # Remove each cell's phase-averaged rate within its own cycle. Without
        # this a decoder can ride the slow recovery of baseline firing, which
        # is precisely the thing the endpoint is supposed to exclude.
        per_cycle_mean = counts.mean(axis=1, keepdims=True)
        x = (counts - per_cycle_mean).reshape(-1, counts.shape[2])
    used_counts = counts

    if mode in ('mod_pi', 'polarity_blind') and K % 2:
        raise ValueError(f'{mode} decoding needs an even n_phase_bins; got {K}')
    if mode == 'polarity_blind':
        # Antiphase bins are pooled into one class *before* training, so the
        # decoder is forced to be blind to polarity. When F1 dominates this
        # averages two opposite responses into one template and collapses —
        # which is the point: a high score here means a genuinely
        # polarity-invariant (frequency-doubled) code, not merely that phase
        # is decodable.
        labels = labels % (K // 2)
    elif mode not in ('full', 'mod_pi'):
        raise ValueError(f"mode must be 'full', 'mod_pi' or 'polarity_blind'; "
                         f"got {mode!r}")
    return x, labels, groups, used_counts


def _cv_decode(x, labels, groups, n_classes, metric, reduce_mod=None):
    """Leave-one-epoch-out decoding. Returns pooled predictions and truth.

    ``reduce_mod`` scores in a coarser space than the decoder was trained in:
    the templates stay at full phase resolution and only the prediction and
    the truth are folded. That is what makes ``mod_pi`` an honest upper bound
    rather than a handicap — pooling antiphase bins into one template first
    would average two opposite responses and lose the very signal being
    tested.
    """
    preds, truths = [], []
    for held in np.unique(groups):
        tr, te = groups != held, groups == held
        if not tr.any() or not te.any():
            continue
        templates = _fold_templates(x[tr], labels[tr], n_classes)
        preds.append(_classify(x[te], templates, metric))
        truths.append(labels[te])
    if not preds:
        return None, None
    pred, true = np.concatenate(preds), np.concatenate(truths)
    if reduce_mod:
        pred, true = pred % reduce_mod, true % reduce_mod
    return pred, true


def decode_phase(
    pbr: PhaseBinnedResponse,
    *,
    mode: str = 'full',
    metric: str = 'correlation',
    subtract_mean: bool = True,
    cell_mask: Optional[np.ndarray] = None,
    match_spike_counts: bool = False,
    n_shuffles: int = 50,
    random_seed: Optional[int] = 0,
    verbose: bool = False,
) -> pd.DataFrame:
    """Cross-validated decoding of grating phase from the population, per window.

    The endpoint the rest of the module supports. For each time window, every
    drift cycle contributes one population vector per phase bin; a
    nearest-template decoder trained on the other epochs guesses which phase
    bin each came from. **Cross-validation is leave-one-epoch-out**, because
    cycles inside one epoch share an adaptation state and scoring on a
    held-out cycle of a trained-on epoch would be scoring on the same state.

    ``mode='mod_pi'`` collapses phases half a cycle apart, asking whether the
    population locates the bars without distinguishing light from dark. A
    frequency-doubled response carries that and not the full phase, so the two
    modes separate "knows where the edges are" from "knows which side is
    bright" — and they can recover on different timescales.

    ``metric='correlation'`` is the default because it is scale-free: a window
    cannot win by having more spikes in it, only by having a better-shaped
    pattern. ``'poisson'`` is the likelihood-based alternative and does use
    magnitude; ``'euclidean'`` sits between.

    Returns
    -------
    pandas.DataFrame
        One row per window: ``accuracy``, ``circular_resultant``,
        ``circular_error``, ``abs_circular_error_rad``, ``chance_accuracy``,
        ``n_test``, plus ``shuffle_*`` columns and ``p_value`` when
        ``n_shuffles > 0``. The shuffle rolls each cycle's phase labels by a
        random amount, which destroys consistency *across* cycles while
        leaving each cycle's own structure alone — the null for "is there a
        stable phase code", not for "is there any structure at all".
    """
    rng = np.random.default_rng(random_seed)
    K = pbr.n_phase_bins
    # 'mod_pi' trains at full resolution and only scores modulo pi, so its
    # accuracy is by construction at least the full-phase accuracy; what makes
    # it informative is the comparison of each against *its own* chance level.
    n_classes = K // 2 if mode == 'polarity_blind' else K
    reduce_mod = K // 2 if mode == 'mod_pi' else None
    score_classes = K // 2 if mode in ('mod_pi', 'polarity_blind') else K

    pbr_use = pbr
    if cell_mask is not None:
        cell_mask = np.asarray(cell_mask, dtype=bool)
        pbr_use = PhaseBinnedResponse(
            counts=pbr.counts[:, :, cell_mask],
            sample_window=pbr.sample_window, sample_epoch=pbr.sample_epoch,
            sample_cycle=pbr.sample_cycle,
            z1=pbr.z1[:, cell_mask], z2=pbr.z2[:, cell_mask],
            n_spikes=pbr.n_spikes[:, cell_mask], exposure_s=pbr.exposure_s,
            cells=pbr.cells[cell_mask].reset_index(drop=True),
            windows=pbr.windows, epochs=pbr.epochs, geometry=pbr.geometry,
            n_phase_bins=K, condition=pbr.condition)

    target = None
    if match_spike_counts:
        totals = [pbr_use.counts[pbr_use.sample_window == wi].sum()
                  for wi in range(len(pbr_use.windows))]
        # Per-cycle density, matched to the sparsest window, then scaled back
        # up by each window's own cycle count.
        per_cycle = [t / max((pbr_use.sample_window == wi).sum(), 1)
                     for wi, t in enumerate(totals)]
        floor = min(p for p in per_cycle if p > 0)

    rows = []
    for wi, (w0, w1) in enumerate(pbr_use.windows):
        if match_spike_counts:
            n_cyc = int((pbr_use.sample_window == wi).sum())
            target = floor * n_cyc
        design = _window_design(pbr_use, wi, mode, subtract_mean, rng, target)
        if design is None:
            continue
        x, labels, groups, used_counts = design
        pred, true = _cv_decode(x, labels, groups, n_classes, metric,
                                reduce_mod)
        if pred is None:
            continue
        stats = _circular_stats(pred, true, score_classes)
        n_cyc_w = max(int((pbr_use.sample_window == wi).sum()), 1)
        stats.update(window=pbr_use.window_labels[wi], t_mid=(w0 + w1) / 2.0,
                     t_start=w0, t_end=w1, mode=mode,
                     n_cells=int(pbr_use.counts.shape[2]),
                     n_cycles=n_cyc_w,
                     # Post-thinning, so the column reports what the decoder
                     # actually saw rather than what was available to it.
                     mean_spikes_per_cycle=float(used_counts.sum() / n_cyc_w))
        chance = stats['chance_accuracy']
        stats['accuracy_above_chance'] = float(
            (stats['accuracy'] - chance) / (1.0 - chance))

        if n_shuffles:
            null_acc, null_res = [], []
            n_cyc = x.shape[0] // K
            for _ in range(int(n_shuffles)):
                shifts = rng.integers(0, K, size=n_cyc)
                rolled = np.concatenate(
                    [(np.arange(K) + s) % K for s in shifts])
                sl = rolled % n_classes if mode == 'polarity_blind' else rolled
                p, t = _cv_decode(x, sl, groups, n_classes, metric, reduce_mod)
                if p is None:
                    continue
                s = _circular_stats(p, t, score_classes)
                null_acc.append(s['accuracy'])
                null_res.append(s['circular_resultant'])
            if null_acc:
                null_acc = np.array(null_acc)
                stats.update(
                    shuffle_accuracy=float(null_acc.mean()),
                    shuffle_accuracy_sd=float(null_acc.std()),
                    shuffle_accuracy_hi=float(np.percentile(null_acc, 95)),
                    shuffle_resultant=float(np.mean(null_res)),
                    p_value=float((1 + (null_acc >= stats['accuracy']).sum())
                                  / (1 + null_acc.size)))
        rows.append(stats)
        if verbose:
            print(f'  {stats["window"]:>9}  acc {stats["accuracy"]:.3f}  '
                  f'R {stats["circular_resultant"]:.3f}')

    out = pd.DataFrame(rows)
    out.attrs['condition'] = pbr.condition
    out.attrs['mode'] = mode
    return out


# ---------------------------------------------------------------------------
# 4. Position-corrected mosaic coherence
# ---------------------------------------------------------------------------

def mosaic_coherence(pbr: PhaseBinnedResponse, *,
                     orders: Sequence[int] = (1, 2),
                     by_cell_type: bool = True,
                     min_cells: int = 3,
                     weight: str = 'amplitude') -> pd.DataFrame:
    """Do the cells' response phases agree with where the cells sit?

    Single-cell modulation says each cell is driven at the drift frequency.
    It cannot say the population represents *the grating*, because a set of
    strongly modulated cells with unrelated phases carries no spatial pattern.
    This corrects each cell's phase by the phase the grating puts at its own
    position — the prediction §7 measures the registration with, nothing
    fitted — and asks whether what remains agrees:

    ``C_k = |sum_i w_i Z_ki exp(-i k 2pi f_s a_i)| / sum_i w_i |Z_ki|``

    1 is perfect agreement, 0 is scattered. Computed **per cell type** by
    default: ON and OFF cells sit half a cycle apart, and pooling across that
    offset cancels a real alignment into nothing.

    For ``k=2`` the position correction is applied at twice the spatial
    frequency, which is the right prediction for a frequency-doubled response
    — but note it is invariant to adding π to the true phase, so a high
    ``C_2`` establishes locking to the bar *structure* and not to its polarity.
    """
    f_s = float(pbr.geometry['spatial_freq_cyc_per_px'])
    axis = pbr.cells['axis_px'].to_numpy()
    types = pbr.cells['cell_type'].to_numpy()
    n = pbr.n_spikes

    groups = ([(ct, types == ct) for ct in sorted(set(types))]
              if by_cell_type else [('all', np.ones(len(types), bool))])

    rows = []
    for wi, (w0, w1) in enumerate(pbr.windows):
        for order in orders:
            z = pbr.z1[wi] if order == 1 else pbr.z2[wi]
            # Amplitude-weight by the *corrected* resultant times spike count,
            # so a cell whose apparent modulation is all small-sample bias does
            # not get a vote proportional to that bias.
            amp = corrected_resultant(z, n[wi]) * n[wi]
            # `drift_phase_response` measures response phase as
            # angle(sum exp(+i 2pi f t)) against a stimulus phase of
            # (pi/2 - 2pi f_s a), so the position correction that cancels the
            # position term carries a **plus** sign. Getting it negative
            # doubles the spatial term instead of removing it and scatters a
            # perfectly aligned population to near zero.
            corrected = z / np.where(np.abs(z) == 0, 1, np.abs(z))
            corrected = corrected * np.exp(1j * order * 2 * np.pi * f_s * axis)
            w = amp if weight == 'amplitude' else np.ones_like(amp)
            for name, mask in groups:
                # A cell whose corrected resultant floors at zero carries no
                # vote, so counting it toward `min_cells` lets one surviving
                # cell return a coherence of 1.0 — which is what a single
                # vector always gives.
                m = (mask & np.isfinite(amp) & (n[wi] >= 2)
                     & np.isfinite(w) & (w > 0))
                if m.sum() < min_cells:
                    continue
                ww = w[m]
                num = np.abs(np.nansum(ww * corrected[m]))
                den = np.nansum(ww)
                # Kish effective sample size: how many cells the weighted mean
                # is really averaging. Well below n_cells means one or two are
                # carrying it and the coherence is about them.
                n_eff = float(den ** 2 / np.nansum(ww ** 2)) if den > 0 else np.nan
                rows.append({
                    'window': pbr.window_labels[wi],
                    't_mid': (w0 + w1) / 2.0, 't_start': w0, 't_end': w1,
                    'order': order, 'cell_type': name, 'n_cells': int(m.sum()),
                    'n_effective': n_eff,
                    'coherence': float(num / den) if den > 0 else np.nan,
                    'mean_resultant': float(np.nanmean(
                        corrected_resultant(z[m], n[wi][m]))),
                })
    out = pd.DataFrame(rows)
    out.attrs['condition'] = pbr.condition
    return out


# ---------------------------------------------------------------------------
# 5. Reliability
# ---------------------------------------------------------------------------

def split_half_reliability(pbr: PhaseBinnedResponse, *,
                           n_splits: int = 50,
                           random_seed: Optional[int] = 0) -> pd.DataFrame:
    """Correlation between phase-binned population responses of two half-sets.

    Decoding can improve because responses grew, because their phases lined
    up, or because they stopped varying from cycle to cycle. This measures the
    third: cycles are split at random into halves, each half averaged into a
    (phase x cell) response, and the two flattened and correlated. Repeated
    over random splits.
    """
    rng = np.random.default_rng(random_seed)
    rows = []
    for wi, (w0, w1) in enumerate(pbr.windows):
        sel = np.flatnonzero(pbr.sample_window == wi)
        if sel.size < 4:
            continue
        c = pbr.counts[sel]
        vals = []
        for _ in range(int(n_splits)):
            perm = rng.permutation(sel.size)
            a, b = perm[:sel.size // 2], perm[sel.size // 2:]
            ma, mb = c[a].mean(axis=0).ravel(), c[b].mean(axis=0).ravel()
            if ma.std() == 0 or mb.std() == 0:
                continue
            vals.append(np.corrcoef(ma, mb)[0, 1])
        if vals:
            rows.append({'window': pbr.window_labels[wi],
                         't_mid': (w0 + w1) / 2.0, 't_start': w0, 't_end': w1,
                         'reliability': float(np.mean(vals)),
                         'reliability_sd': float(np.std(vals)),
                         'n_cycles': int(sel.size)})
    out = pd.DataFrame(rows)
    out.attrs['condition'] = pbr.condition
    return out


# ---------------------------------------------------------------------------
# 6. Pathway comparison at matched cell count
# ---------------------------------------------------------------------------

def pathway_decoding(pbr: PhaseBinnedResponse, *,
                     cell_types: Optional[Sequence[str]] = None,
                     n_cells: Optional[int] = None,
                     min_cells: int = 5,
                     n_resamples: int = 20,
                     random_seed: Optional[int] = 0,
                     verbose: bool = True,
                     **decode_kwargs) -> pd.DataFrame:
    """Decode from each cell type separately, with the cell count matched.

    Decoding accuracy grows with population size, so a pathway with more
    recorded cells decodes better for a reason that has nothing to do with
    the pathway. Every type is subsampled to the same ``n_cells`` — by default
    the smallest usable type — and the subsampling repeated, so what is
    compared is per-cell quality rather than how many were held.

    Cell count is the confound that always applies; it is not the only one.
    Spike count, RF coverage of the aperture, and late-adapted reliability all
    differ between midget and parasol, and matching those too is a further
    control this leaves to the caller via ``cell_mask``-style prefiltering.
    """
    rng = np.random.default_rng(random_seed)
    types = pbr.cells['cell_type'].to_numpy()
    present = ([ct for ct in cell_types if (types == ct).any()]
               if cell_types is not None else sorted(set(types)))
    available = {ct: int((types == ct).sum()) for ct in present}
    if not available:
        raise ValueError('no requested cell type is present')

    # Matching to the smallest type is only sensible while that type has
    # enough cells to decode from at all. One thinly sampled type otherwise
    # drags every other down to its own count and the comparison becomes a
    # statement about three cells; dropping it and saying so is more useful.
    too_small = {ct: n for ct, n in available.items() if n < int(min_cells)}
    sizes = {ct: n for ct, n in available.items() if n >= int(min_cells)}
    if not sizes:
        raise ValueError(
            f'no cell type reaches min_cells={min_cells} (available '
            f'{available}); lower min_cells to compare anyway, knowing the '
            f'match will be to {min(available.values())} cells')
    if n_cells is None:
        n_cells = min(sizes.values())
    n_cells = int(min(n_cells, min(sizes.values())))
    present = list(sizes)
    if verbose and too_small:
        print(f'matching to {n_cells} cells; dropped '
              + ', '.join(f'{ct} ({n} cells)' for ct, n in too_small.items())
              + f' for having fewer than min_cells={min_cells}')

    rows = []
    for ct in present:
        idx = np.flatnonzero(types == ct)
        for rep in range(int(n_resamples)):
            pick = rng.choice(idx, size=n_cells, replace=False)
            mask = np.zeros(len(types), dtype=bool)
            mask[pick] = True
            d = decode_phase(pbr, cell_mask=mask,
                             random_seed=int(rng.integers(1 << 31)),
                             **decode_kwargs)
            d['cell_type'] = ct
            d['resample'] = rep
            d['n_available'] = sizes[ct]
            rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out.attrs['matched_n_cells'] = n_cells
    out.attrs['available'] = available
    out.attrs['compared'] = sizes
    out.attrs['dropped_too_small'] = too_small
    out.attrs['condition'] = pbr.condition
    return out


# ---------------------------------------------------------------------------
# 7. Recovery timescale
# ---------------------------------------------------------------------------

def fit_recovery(t, y, *, model: str = 'single', skip_first_s: float = 2.0,
                 n_boot: int = 1000, groups=None,
                 random_seed: Optional[int] = 0) -> Dict:
    """Fit ``D(t) = D_inf - A exp(-t/tau)`` and report tau and t50.

    ``skip_first_s`` defaults to 2 s because the step's onset transient is a
    different process from the recovery being timed, and it is large: on this
    block the first two seconds carry three times the steady-state rate and a
    modulation estimate contaminated by the transient itself. Left in, it
    drags the fit to a sub-second tau that describes the onset and not the
    recovery. Set it to 0 deliberately, not by default.

    ``model='double'`` adds a second exponential. Do not reach for it because
    the residuals look structured at three time points — it needs the data to
    support it, and the returned ``delta_aic`` says whether they do (negative
    favours the double).

    ``t50`` is the summary worth quoting: the time at which the metric has
    covered half the distance from its value at the first fitted point to its
    asymptote. It is read off the fitted curve, so it is defined even when no
    window sits near it.

    **Bootstrap and what it licenses.** ``groups`` names the resampling unit —
    pass the preparation when there is more than one, and the interval is a
    statement about retinas. Passing cells or epochs (or nothing, which
    resamples points) gives an interval that describes *this* preparation and
    supports no inference beyond it. The distinction is the difference between
    "tau is 12 s in this retina" and "tau is 12 s", and only the first is
    available from one preparation.
    """
    from scipy.optimize import curve_fit

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y) & (t >= float(skip_first_s))
    t, y = t[ok], y[ok]
    g = None if groups is None else np.asarray(groups)[ok]
    if t.size < 3:
        raise ValueError(f'need at least 3 usable points; got {t.size}')

    def single(tt, d_inf, a, tau):
        return d_inf - a * np.exp(-tt / np.clip(tau, 1e-3, None))

    def double(tt, d_inf, af, tauf, a_s, taus):
        return (d_inf - af * np.exp(-tt / np.clip(tauf, 1e-3, None))
                - a_s * np.exp(-tt / np.clip(taus, 1e-3, None)))

    span = max(y.max() - y.min(), 1e-9)
    t_range = max(t.max() - t.min(), 1e-9)

    def _fit(tt, yy, which):
        if which == 'single':
            p0 = [yy.max(), span, t_range / 3]
            bounds = ([-np.inf, -np.inf, 1e-2], [np.inf, np.inf, 10 * t_range])
            return curve_fit(single, tt, yy, p0=p0, bounds=bounds, maxfev=20000)[0]
        p0 = [yy.max(), span / 2, t_range / 20, span / 2, t_range / 3]
        bounds = ([-np.inf, -np.inf, 1e-2, -np.inf, 1e-2],
                  [np.inf, np.inf, 10 * t_range, np.inf, 10 * t_range])
        return curve_fit(double, tt, yy, p0=p0, bounds=bounds, maxfev=40000)[0]

    popt = _fit(t, y, model)
    fn = single if model == 'single' else double
    resid = y - fn(t, *popt)

    def _aic(res, k):
        rss = float(np.sum(res ** 2))
        n = res.size
        return n * np.log(max(rss, 1e-300) / n) + 2 * k

    delta_aic = np.nan
    try:
        p_s = _fit(t, y, 'single')
        p_d = _fit(t, y, 'double')
        delta_aic = float(_aic(y - double(t, *p_d), 5)
                          - _aic(y - single(t, *p_s), 3))
    except Exception:
        pass

    d_inf = float(popt[0])
    y0 = float(fn(t.min(), *popt))
    half = y0 + (d_inf - y0) / 2.0
    grid = np.linspace(t.min(), t.min() + 10 * t_range, 20000)
    curve = fn(grid, *popt)
    crossed = np.flatnonzero(np.sign(curve - half) != np.sign(curve[0] - half))
    t50 = float(grid[crossed[0]]) if crossed.size else np.nan

    rng = np.random.default_rng(random_seed)
    boot = {'tau': [], 't50': [], 'd_inf': []}
    units = np.unique(g) if g is not None else None
    for _ in range(int(n_boot)):
        if units is not None:
            take = rng.choice(units, size=units.size, replace=True)
            idx = np.concatenate([np.flatnonzero(g == u) for u in take])
        else:
            idx = rng.integers(0, t.size, size=t.size)
        try:
            pb = _fit(t[idx], y[idx], model)
        except Exception:
            continue
        tau_b = float(pb[2]) if model == 'single' else float(max(pb[2], pb[4]))
        d_b = float(pb[0])
        y0b = float(fn(t.min(), *pb))
        hb = y0b + (d_b - y0b) / 2.0
        cb = fn(grid, *pb)
        cr = np.flatnonzero(np.sign(cb - hb) != np.sign(cb[0] - hb))
        boot['tau'].append(tau_b)
        boot['d_inf'].append(d_b)
        boot['t50'].append(float(grid[cr[0]]) if cr.size else np.nan)

    def ci(key):
        v = np.asarray(boot[key], dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 10:
            return (np.nan, np.nan)
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

    tau = float(popt[2]) if model == 'single' else float(max(popt[2], popt[4]))
    # The fit's own upper bound on tau is 10x the time range. A bootstrap
    # interval that reaches it means the data do not bound the timescale from
    # above at all — the curve is consistent with "still rising at the end of
    # the epoch" — which is a different statement from a merely wide interval
    # and the one worth refusing to quote a tau for.
    tau_ceiling = 10 * t_range
    tau_hi = ci('tau')[1]
    tau_bounded = bool(np.isfinite(tau_hi) and tau_hi < 0.9 * tau_ceiling)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        'model': model, 'params': popt, 'tau_s': tau, 't50_s': t50,
        'd_inf': d_inf, 'd_start': y0, 'amplitude': d_inf - y0,
        'tau_ci': ci('tau'), 't50_ci': ci('t50'), 'd_inf_ci': ci('d_inf'),
        'tau_bounded': tau_bounded,
        'r_squared': float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        'delta_aic_double_minus_single': delta_aic,
        'n_points': int(t.size),
        'bootstrap_unit': ('none (points)' if groups is None
                           else f'{np.unique(g).size} groups'),
        'predict': (lambda tt, _fn=fn, _p=popt: _fn(np.asarray(tt, float), *_p)),
    }


# ---------------------------------------------------------------------------
# 9. Normalized spatial-structure index
# ---------------------------------------------------------------------------

def spatial_structure_index(decoding: pd.DataFrame, *,
                            column: str = 'accuracy',
                            shuffle_column: Optional[str] = None,
                            late_windows: int = 1) -> pd.DataFrame:
    """Rescale decoding to 0 = its own shuffle, 1 = its own late-adapted value.

    ``S = (D - D_shuffle) / (D_late - D_shuffle)``. Both ends are taken from
    the same condition and the same population, which is what makes 50 µm
    comparable to 150 µm and midget to parasol: chance depends on the number
    of phase bins, and the achievable ceiling depends on how many cells fired,
    so neither raw accuracy nor accuracy-above-chance compares across them.

    Values above 1 are a transient overshoot; below 0 means the window carried
    no more phase information than its shuffle, which is a real answer for the
    first seconds after a step and not a failure of the fit.
    """
    df = decoding.copy()
    shuffle_column = shuffle_column or f'shuffle_{column}'
    if shuffle_column in df.columns:
        base = df[shuffle_column].to_numpy(dtype=float)
    elif 'chance_accuracy' in df.columns and column == 'accuracy':
        base = df['chance_accuracy'].to_numpy(dtype=float)
    else:
        raise KeyError(f'no {shuffle_column!r} column and no usable fallback; '
                       f'run decode_phase with n_shuffles > 0')

    d = df[column].to_numpy(dtype=float)
    late = float(np.nanmean(d[-int(late_windows):]))
    late_base = float(np.nanmean(base[-int(late_windows):]))
    denom = late - late_base

    # The index divides by how far the late-adapted state sits above its own
    # shuffle. When a condition never beats its shuffle — 50 um here, at any
    # time — that denominator is noise around zero, and dividing by it turns
    # an honest "no signal" into values like -9 and +4 that look like wild
    # recovery. Require the late value to clear the shuffle by two shuffle
    # SDs before the index means anything; otherwise return NaN and say so.
    sd_col = f'shuffle_{column}_sd'
    noise = (float(np.nanmean(df[sd_col].to_numpy(dtype=float)[-int(late_windows):]))
             if sd_col in df.columns else 0.0)
    resolvable = denom > max(2.0 * noise, 1e-12)

    df['index_baseline'] = base
    df['spatial_structure_index'] = ((d - base) / denom if resolvable else np.nan)
    df.attrs['late_value'] = late
    df.attrs['late_baseline'] = late_base
    df.attrs['late_margin'] = denom
    df.attrs['shuffle_sd'] = noise
    df.attrs['resolvable'] = bool(resolvable)
    df.attrs['index_column'] = column
    if not resolvable:
        df.attrs['undefined_reason'] = (
            f'late {column} ({late:.3f}) is {denom:.3f} above its shuffle '
            f'({late_base:.3f}), within {2 * noise:.3f} = 2 shuffle SD, so '
            f'there is no late-adapted signal to normalise to')
    return df


# ---------------------------------------------------------------------------
# 10. Per-date summary, persistence, and cross-date pooling
# ---------------------------------------------------------------------------

_RECOVERY_PROTOCOL = 'vmdg'


def analyze_recovery_conditions(
    pipeline,
    stim_block,
    epochs: pd.DataFrame,
    *,
    condition_keys: Sequence[str],
    windows_s: Sequence[Tuple[float, float]],
    cell_types: Optional[Iterable[str]] = None,
    cell_ids: Optional[Iterable[int]] = None,
    drift_freq_hz: Optional[float] = None,
    n_phase_bins: int = 12,
    n_shuffles: int = 50,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Run the complete spatial-recovery analysis once per condition.

    Conditions are never pooled before analysis: their geometry and phase are
    distinct. The returned dictionary is the stable input to
    :func:`recovery_summary_table` and the single-date diagnostic plots.
    """
    from .protocol_source import condition_label

    out: Dict[str, Dict] = {}
    for values, rows in epochs.groupby(list(condition_keys), sort=True):
        values = values if isinstance(values, tuple) else (values,)
        label = condition_label(condition_keys, values)
        epoch_ids = rows['epoch'].astype(int).tolist()
        if len(epoch_ids) < 2:
            if verbose:
                print(f'skipping {label}: {len(epoch_ids)} epoch; '
                      'need at least 2 for leave-one-epoch-out decoding')
            continue
        if verbose:
            print(f'\n{label}: epochs {epoch_ids}')
        pbr = phase_binned_response(
            pipeline, stim_block, epoch_ids,
            windows_s=windows_s,
            n_phase_bins=n_phase_bins,
            cell_types=cell_types,
            cell_ids=cell_ids,
            drift_freq_hz=drift_freq_hz,
            verbose=verbose,
        )
        out[label] = {
            'pbr': pbr,
            'modulation': phase_modulation(pbr),
            'modulation_naive': phase_modulation(pbr, debias=False),
            'full': decode_phase(pbr, mode='full', n_shuffles=n_shuffles),
            'matched': decode_phase(
                pbr, mode='full', n_shuffles=0,
                match_spike_counts=True,
            ),
            'polarity_blind': decode_phase(
                pbr, mode='polarity_blind', n_shuffles=n_shuffles,
            ),
            'coherence': mosaic_coherence(pbr),
            'reliability': split_half_reliability(pbr),
        }
    return out


def recovery_summary_table(
    recovery: Dict[str, Dict],
    *,
    exp_name: Optional[str] = None,
) -> pd.DataFrame:
    """Collapse condition results into one row per condition and time window.

    Raw endpoints and their cross-date-normalized counterparts are retained.
    The raw values remain available for auditing; pooled plots use only the
    normalized columns created by :func:`normalize_recovery_summary`.
    """
    rows = []
    for condition, result in recovery.items():
        pbr = result['pbr']
        mod = result['modulation']
        naive = result['modulation_naive']
        grouped = mod.groupby(['window', 't_start', 't_end', 't_mid'], sort=False)
        base = grouped.agg(
            rate_hz=('f0_hz', 'mean'),
            f1=('m1', 'mean'),
            f2=('m2', 'mean'),
            n_f1_resolved=('f1_resolved', 'sum'),
        ).reset_index()
        base['f2_over_f1'] = base['f2'] / base['f1']
        naive_mean = (naive.groupby('window', sort=False)['m1'].mean()
                      .rename('f1_naive'))
        base = base.merge(naive_mean, left_on='window', right_index=True,
                          how='left')

        for key, prefix in (
            ('full', 'decode_full'),
            ('matched', 'decode_matched'),
            ('polarity_blind', 'decode_polblind'),
        ):
            dec = result[key].copy()
            keep = ['window', 'accuracy', 'chance_accuracy']
            rename = {
                'accuracy': prefix,
                'chance_accuracy': f'{prefix}_chance',
            }
            for col in ('shuffle_accuracy', 'shuffle_accuracy_sd'):
                if col in dec:
                    keep.append(col)
                    rename[col] = f'{prefix}_{col}'
            base = base.merge(dec[keep].rename(columns=rename),
                              on='window', how='left')

        reliability = result['reliability']
        if not reliability.empty:
            base = base.merge(
                reliability[['window', 'reliability', 'reliability_sd']],
                on='window', how='left',
            )
        geometry = getattr(pbr, 'geometry', {}) or {}
        base.insert(0, 'condition', condition)
        base.insert(0, 'exp_name', exp_name or '')
        base['bar_width_um'] = geometry.get('bar_width_um', np.nan)
        base['mean_intensity'] = geometry.get('mean_intensity', np.nan)
        base['drift_freq_hz'] = float(getattr(
            pbr, 'drift_freq_hz', geometry.get('temporal_freq_hz', np.nan)))
        base['n_cells'] = int(pbr.counts.shape[2])
        base['n_epochs'] = int(len(pbr.epochs))
        rows.append(base)
    if not rows:
        return pd.DataFrame()
    return normalize_recovery_summary(pd.concat(rows, ignore_index=True))


def normalize_recovery_summary(
    summary: pd.DataFrame,
    *,
    late_windows: int = 1,
) -> pd.DataFrame:
    """Normalize each retina and condition before cross-date pooling.

    ``rate_late_fraction`` and ``f1_late_fraction`` divide by that retina's
    own late-adapted value. Decoder indices subtract each window's own
    shuffle (or analytical chance for spike-count-matched decoding) and scale
    by that retina's late shuffle-to-signal margin. Thus 0 means null-level
    phase information and 1 means that preparation's late state.
    """
    if summary.empty:
        return summary.copy()
    required = {'exp_name', 'condition', 't_mid', 'rate_hz', 'f1'}
    missing = required - set(summary.columns)
    if missing:
        raise KeyError(f'recovery summary missing columns: {sorted(missing)}')

    pieces = []
    for _, group in summary.groupby(['exp_name', 'condition'], sort=False,
                                     dropna=False):
        g = group.sort_values('t_mid').copy()
        late = g.tail(int(late_windows))
        for raw, normalized in (
            ('rate_hz', 'rate_late_fraction'),
            ('f1', 'f1_late_fraction'),
        ):
            denominator = float(np.nanmean(late[raw]))
            g[normalized] = (g[raw] / denominator
                             if np.isfinite(denominator) and denominator != 0
                             else np.nan)

        for metric in ('decode_full', 'decode_matched', 'decode_polblind'):
            if metric not in g:
                continue
            shuffle = f'{metric}_shuffle_accuracy'
            chance = f'{metric}_chance'
            baseline_col = shuffle if shuffle in g else chance
            if baseline_col not in g:
                continue
            baseline = g[baseline_col].to_numpy(dtype=float)
            late_margin = float(np.nanmean(
                late[metric].to_numpy(dtype=float)
                - late[baseline_col].to_numpy(dtype=float)))
            sd_col = f'{metric}_shuffle_accuracy_sd'
            resolution_floor = (2.0 * float(np.nanmean(late[sd_col]))
                                if sd_col in late else 0.0)
            target = f'{metric}_index'
            g[target] = ((g[metric].to_numpy(dtype=float) - baseline)
                         / late_margin
                         if (np.isfinite(late_margin)
                             and late_margin > max(resolution_floor, 1e-12))
                         else np.nan)
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def load_recovery_many(
    exp_names: Optional[Iterable[str]] = None,
    *,
    protocol: str = _RECOVERY_PROTOCOL,
    output_root=None,
) -> pd.DataFrame:
    """Load and concatenate every saved per-date pickle summary."""
    from .analysis_results import load_analysis_many

    bundles = load_analysis_many(
        protocol, exp_names=exp_names, output_root=output_root)
    tables = []
    for exp_name, bundle in bundles.items():
        table = bundle['analysis'].get('recovery_summary')
        if isinstance(table, pd.DataFrame):
            table = table.copy()
            table['exp_name'] = str(exp_name)
            tables.append(table)
    if not tables:
        return pd.DataFrame()
    return normalize_recovery_summary(pd.concat(tables, ignore_index=True))


def saved_recovery_stats(
    *,
    protocol: str = _RECOVERY_PROTOCOL,
    output_root=None,
) -> pd.DataFrame:
    """One audit row per date already present in the recovery dataset."""
    saved = load_recovery_many(protocol=protocol, output_root=output_root)
    columns = [
        'exp_name', 'n_rows', 'n_conditions', 'n_windows',
        'n_cells_min', 'n_cells_max', 'n_epochs',
    ]
    if saved.empty:
        return pd.DataFrame(columns=columns)
    return (saved.groupby('exp_name', as_index=False)
            .agg(n_rows=('condition', 'size'),
                 n_conditions=('condition', 'nunique'),
                 n_windows=('window', 'nunique'),
                 n_cells_min=('n_cells', 'min'),
                 n_cells_max=('n_cells', 'max'),
                 n_epochs=('n_epochs', 'max'))[columns])


def save_recovery_summary(
    summary: pd.DataFrame,
    exp_name: str,
    *,
    protocol: str = _RECOVERY_PROTOCOL,
    output_root=None,
    metadata: Optional[Dict] = None,
    figures: Optional[Dict] = None,
    cell_qc: Optional[pd.DataFrame] = None,
    template_match: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> Dict:
    """Print existing dates, then save one date as pickle + JSON + plots.

    One folder per date makes incremental updates safe: adding a new retina
    does not rewrite any previous date, while rerunning a date deliberately
    replaces only that date's derived table, metadata, and named plots.
    """
    from .analysis_results import save_analysis_bundle

    table = summary.copy()
    table['exp_name'] = str(exp_name)
    table = normalize_recovery_summary(table)
    meta = {
        'analysis_name': 'variableMeanDriftingGrating spatial recovery',
        'normalization': {
            'rate': 'within-date, within-condition / late rate',
            'f1': 'within-date, within-condition / late F1',
            'decoding': 'within-date null-to-late index',
            'biological_replicate': 'retina/date',
        },
        **dict(metadata or {}),
    }
    analysis = {'recovery_summary': table}
    if cell_qc is not None:
        qc = cell_qc.copy()
        analysis['cell_qc'] = qc
        if 'excluded_downstream' in qc:
            rejected = qc.loc[qc['excluded_downstream'].astype(bool),
                              'cell_id'].astype(int).tolist()
            meta['cell_qc'] = {
                'n_candidates': int(len(qc)),
                'n_retained': int(len(qc) - len(rejected)),
                'n_excluded': int(len(rejected)),
                'excluded_cell_ids': rejected,
            }
    if template_match is not None:
        analysis['template_match'] = template_match.copy()
    return save_analysis_bundle(
        protocol, str(exp_name), analysis,
        metadata=meta, figures=figures, output_root=output_root,
        verbose=verbose)


def save_recovery_cross_date_summary(
    summary: pd.DataFrame,
    *,
    protocol: str = _RECOVERY_PROTOCOL,
    output_root=None,
    metadata: Optional[Dict] = None,
    figures: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict:
    """Save the combined VMDG dataset and multi-date plots in ``summary/``."""
    from .analysis_results import save_analysis_summary

    table = normalize_recovery_summary(summary)
    dates = sorted(str(v) for v in table['exp_name'].dropna().unique())
    meta = {
        'analysis_name': 'variableMeanDriftingGrating cross-date recovery',
        'dates': dates,
        'n_dates': len(dates),
        'normalization': 'within retina and condition; dates weighted equally',
        **dict(metadata or {}),
    }
    return save_analysis_summary(
        protocol, {'recovery_summary': table}, metadata=meta,
        figures=figures, output_root=output_root, verbose=verbose)


def plot_recovery_across_dates(
    summary: pd.DataFrame,
    *,
    condition: Optional[str] = None,
    figsize: Tuple[float, float] = (13.0, 3.6),
):
    """Plot normalized recovery trajectories with dates weighted equally.

    Thin lines are individual retinas. The heavy line and SEM summarize dates,
    not cells, which keeps preparations with larger cell counts from dominating
    the pooled result.
    """
    import matplotlib.pyplot as plt
    from .style import apply_publication_style

    data = normalize_recovery_summary(summary)
    if condition is not None:
        data = data[data['condition'] == condition]
    if data.empty:
        raise ValueError('No saved recovery rows match the requested condition.')
    if condition is None and data['condition'].nunique() != 1:
        raise ValueError('Multiple conditions are present; pass condition=...')

    apply_publication_style()
    panels = [
        ('rate_late_fraction', 'rate / late rate'),
        ('f1_late_fraction', 'F1 / late F1'),
        ('decode_matched_index', 'matched decoding index'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)
    for ax, (metric, ylabel) in zip(axes, panels):
        if metric not in data or data[metric].notna().sum() == 0:
            ax.set_visible(False)
            continue
        for _, date in data.groupby('exp_name', sort=True):
            date = date.sort_values('t_mid')
            ax.plot(date['t_mid'], date[metric], '-o', ms=2,
                    color='0.65', alpha=0.55, linewidth=0.8)
        by_time = data.groupby('t_mid')[metric]
        mean = by_time.mean()
        sem = by_time.sem()
        ax.plot(mean.index, mean, '-o', color='black', ms=4,
                linewidth=2, label=f'mean of {data.exp_name.nunique()} dates')
        ax.fill_between(mean.index, mean - sem, mean + sem,
                        color='black', alpha=0.15, linewidth=0)
        ax.axhline(1.0, color='0.75', linestyle='--', linewidth=0.8)
        ax.set_xlabel('time since step (s)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
    label = condition or str(data['condition'].iloc[0])
    fig.suptitle(f'cross-date recovery — {label}', y=1.03)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_recovery_summary(modulation: pd.DataFrame,
                          decoding_full: pd.DataFrame,
                          decoding_modpi: Optional[pd.DataFrame] = None,
                          coherence: Optional[pd.DataFrame] = None,
                          reliability: Optional[pd.DataFrame] = None,
                          *,
                          title: str = '',
                          figsize: Tuple[float, float] = (13.5, 8.0)):
    """Four panels: rate and modulation, decoding, coherence, reliability."""
    import matplotlib.pyplot as plt
    from .style import apply_publication_style, colors_for_celltypes

    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # A — mean rate against modulation, the dissociation the analysis is about
    ax = axes[0, 0]
    types = sorted(modulation['cell_type'].dropna().unique())
    colors = colors_for_celltypes(types)
    g = modulation.groupby('t_mid')
    ax.plot(g['f0_hz'].mean().index, g['f0_hz'].mean().to_numpy(),
            'k-o', ms=3, label='mean rate (Hz)')
    ax.set_ylabel('population rate (Hz/cell)')
    ax.set_xlabel('time since step (s)')
    ax2 = ax.twinx()
    for ct in types:
        s = modulation[modulation['cell_type'] == ct].groupby('t_mid')['m1'].mean()
        ax2.plot(s.index, s.to_numpy(), '-o', ms=3, color=colors[ct],
                 label=f'{ct} F1')
    ax2.set_ylabel('bias-corrected F1/F0')
    ax.set_title('A. rate falls while modulation rises')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='center right')

    # B — decoding, the primary endpoint
    ax = axes[0, 1]
    ax.plot(decoding_full['t_mid'], decoding_full['accuracy'], '-o', ms=4,
            color='tab:blue', label='full phase')
    if 'shuffle_accuracy' in decoding_full:
        ax.plot(decoding_full['t_mid'], decoding_full['shuffle_accuracy'],
                '--', color='tab:blue', alpha=0.5, label='shuffle')
        if 'shuffle_accuracy_hi' in decoding_full:
            ax.fill_between(decoding_full['t_mid'],
                            decoding_full['shuffle_accuracy'],
                            decoding_full['shuffle_accuracy_hi'],
                            color='tab:blue', alpha=0.12)
    if decoding_modpi is not None and len(decoding_modpi):
        ax.plot(decoding_modpi['t_mid'], decoding_modpi['accuracy'], '-s', ms=4,
                color='tab:orange', label='phase mod pi')
        if 'shuffle_accuracy' in decoding_modpi:
            ax.plot(decoding_modpi['t_mid'], decoding_modpi['shuffle_accuracy'],
                    '--', color='tab:orange', alpha=0.5)
    ax.set_xlabel('time since step (s)')
    ax.set_ylabel('decoding accuracy')
    ax.set_title('B. population phase decoding')
    ax.legend(fontsize=7)

    # C — position-corrected coherence
    ax = axes[1, 0]
    if coherence is not None and len(coherence):
        for (ct, order), sub in coherence.groupby(['cell_type', 'order']):
            ax.plot(sub['t_mid'], sub['coherence'],
                    '-o' if order == 1 else '--s', ms=3,
                    color=colors.get(ct, '#666666'),
                    label=f'{ct} C{order}')
        ax.legend(fontsize=6, ncol=2)
    ax.set_xlabel('time since step (s)')
    ax.set_ylabel('coherence')
    ax.set_title('C. phases agree with RF position')

    # D — reliability
    ax = axes[1, 1]
    if reliability is not None and len(reliability):
        ax.errorbar(reliability['t_mid'], reliability['reliability'],
                    yerr=reliability['reliability_sd'], fmt='-o', ms=4,
                    color='tab:green')
    ax.set_xlabel('time since step (s)')
    ax.set_ylabel('split-half correlation')
    ax.set_title('D. cycle-to-cycle reliability')

    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96) if title else None)
    return fig
