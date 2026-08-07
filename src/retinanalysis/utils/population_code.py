"""The population representation of a movie that is played twice inside one epoch.

Some protocols repeat a stimulus segment *within* an epoch. That repeat is
worth more than an extra trial, because it holds the stimulus fixed while
letting time since the adaptation step run on: the first presentation is seen
by a retina that has just been stepped, the second by one that has had the
segment's own mean luminance for a whole cycle. Comparing them asks whether
the representation converged, with the stimulus controlled by construction
rather than by a model.

``EyeMovementTrajectoryAlternatingBackground`` is the worked example: 30 s of
an adapting background at 0.1x or 10x the image mean, then 30 s of natural
image swept by an eye-movement trajectory that is **looped twice**
(``xTraj = [xTraj, xTraj]`` in the ``.m``). Nothing here is specific to that
protocol beyond the timing, which is read off the recorded frames.

The pieces, in the order they have to happen:

1. :func:`repeat_timing` — **where the cycles actually start.** The nominal
   answer (``preTime``, then ``stimTime/2``) is wrong on any rig whose display
   does not run at the refresh rate Stage was told about, and the error is a
   fixed fraction of a long interval. See the module note below.
2. :func:`response_timescale` — how fast each cell type's response actually
   varies, measured from cross-trial autocorrelation, so the smoothing and the
   Victor-Purpura cost are set by the data rather than by taste.
3. :func:`repeat_response` — one :class:`RepeatResponse`: a
   ``(cell, epoch, cycle, movie-time)`` array of firing rates on a common
   movie-time axis, with per-cell-type smoothing.
4. :func:`population_similarity` — the headline. Cross-condition population
   correlation per cycle, normalised by each population's own repeat
   reliability, alongside the mean single-cell correlation and both raw and
   shape-only variants.
5. :func:`cycle_interaction` — first-versus-second cycle within each
   condition, and whether that change depends on the adaptation history.
6. :func:`population_spike_distance` — the same question as (4) in spike-train
   distance, with the excess over within-condition repeat variability.
7. :func:`time_resolved_similarity` — the trajectory, in sections short enough
   to resolve recovery and long enough to estimate a correlation from.

Module note — **the cycle is not stimTime/2.**

Stage renders one frame per display refresh and advances its own clock by
``1/declaredRefreshRate`` each time. When the display runs at a different rate
the movie plays at the wrong speed, and over a 15 s cycle a 0.5 % error is
most of a hundred milliseconds — the same order as a parasol cell's entire
response timescale. On the rig this was written against the declared rate is
60 Hz and the measured one 60.31 Hz, so the cycle is **14.922 s**, not 15.000,
and the movie starts at 29.843 s rather than 30.000. Folding cycle 2 onto
cycle 1 at the nominal period therefore misaligns them by 78 ms, which shows
up as a real-looking drop in cycle-to-cycle correlation and would be read as
adaptation.

:func:`repeat_timing` takes the answer from ``frame_times_ms`` — the frame
monitor's record of when frames actually appeared — and only falls back to the
nominal arithmetic when that is unavailable, saying so.
:func:`estimate_cycle_period` recovers the same number from the spikes alone,
which is the check that the frame record means what it is being read to mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


__all__ = [
    'repeat_timing',
    'estimate_cycle_period',
    'plot_cycle_period_scan',
    'ResponseTimescale',
    'response_timescale',
    'plot_response_timescale',
    'RepeatResponse',
    'repeat_response',
    'population_similarity',
    'cycle_interaction',
    'population_spike_distance',
    'time_resolved_similarity',
    'plot_population_matrices',
    'plot_population_similarity',
    'plot_excess_distance',
    'plot_time_resolved_similarity',
    'summarize_eye_movement_cell_type_recovery',
    'fit_eye_movement_cell_type_recovery',
    'compare_eye_movement_cell_type_timescales',
    'plot_eye_movement_cell_type_comparison',
    'plot_eye_movement_cell_type_across_dates',
    'save_eye_movement_results',
    'load_eye_movement_many',
    'summarize_eye_movement_dates',
    'plot_eye_movement_across_dates',
    'save_eye_movement_cross_date_summary',
]

# How each normalisation treats a (cell x time) matrix before it is vectorised
# and correlated. See `_normalize_matrix`.
NORMALIZATIONS = ('raw', 'centered', 'shape')


# ---------------------------------------------------------------------------
# 1. Timing: where the cycles start
# ---------------------------------------------------------------------------

def repeat_timing(stim_block, *, n_cycles: int = 2,
                  epoch: Optional[int] = None,
                  declared_refresh_hz: Optional[float] = None,
                  verbose: bool = True) -> Dict:
    """When the repeated segment starts, and how long one cycle really is.

    Read off ``frame_times_ms`` rather than computed from ``preTime`` and
    ``stimTime``. Stage counts its own time in units of the *declared* refresh
    rate, so on a display that runs at any other rate the stimulus plays fast
    or slow by that ratio, and both the onset and the cycle length are off by
    a fixed fraction. Over a 15 s cycle at 60.31 Hz against a declared 60 that
    fraction is 78 ms.

    Parameters
    ----------
    stim_block : MEAStimBlock
        Supplies ``df_epochs`` (with ``frame_times_ms``) and
        ``d_epoch_block_params`` (``preTime``, ``stimTime``).
    n_cycles : int
        How many times the segment repeats inside ``stimTime``. 2 for
        ``EyeMovementTrajectoryAlternatingBackground``.
    epoch : int, optional
        Which epoch's frame record to read. Default: every epoch is read and
        the spread across them reported, since a single epoch that dropped
        frames would otherwise set the timing for the whole block.
    declared_refresh_hz : float, optional
        The rate Stage was told about. Default: the epoch's own
        ``monitorRefreshRate`` parameter, which is what Stage was configured
        with.

    Returns
    -------
    dict
        ``onset_ms`` (movie onset, ms from epoch start), ``cycle_ms`` (one
        cycle, ms), ``n_cycles``, ``nominal_onset_ms`` / ``nominal_cycle_ms``
        (what the protocol arithmetic would have said), ``refresh_declared_hz``
        / ``refresh_actual_hz``, ``onset_frame`` / ``cycle_frames``,
        ``source`` (``'frame_times'`` or ``'nominal'``), and ``spread_ms``
        (max-minus-min of ``cycle_ms`` across epochs).
    """
    df = stim_block.df_epochs
    block = dict(getattr(stim_block, 'd_epoch_block_params', {}) or {})
    pre_ms = float(block.get('preTime', 0) or 0)
    stim_ms = float(block.get('stimTime', 0) or 0)
    if stim_ms <= 0:
        raise ValueError('repeat_timing: stimTime is 0 or missing from '
                         'd_epoch_block_params; nothing to divide into cycles')

    if declared_refresh_hz is None:
        first = df['epoch_parameters'].iloc[0] if len(df) else {}
        declared_refresh_hz = float(first.get('monitorRefreshRate', 60.0) or 60.0)

    nominal_onset_ms = pre_ms
    nominal_cycle_ms = stim_ms / n_cycles

    onset_frame = int(round(pre_ms / 1000.0 * declared_refresh_hz))
    cycle_frames = int(round(stim_ms / 1000.0 * declared_refresh_hz / n_cycles))

    out = {
        'n_cycles': int(n_cycles),
        'nominal_onset_ms': nominal_onset_ms,
        'nominal_cycle_ms': nominal_cycle_ms,
        'onset_frame': onset_frame,
        'cycle_frames': cycle_frames,
        'refresh_declared_hz': float(declared_refresh_hz),
    }

    rows = df.index.tolist() if epoch is None else [int(epoch)]
    onsets, cycles, actual = [], [], []
    for i in rows:
        ft = df['frame_times_ms'].iloc[i] if 'frame_times_ms' in df.columns else None
        if ft is None:
            continue
        ft = np.asarray(ft, dtype=float).ravel()
        if ft.size <= onset_frame + cycle_frames:
            continue
        onsets.append(ft[onset_frame])
        cycles.append(ft[onset_frame + cycle_frames] - ft[onset_frame])
        actual.append((ft.size - 1) / (ft[-1] - ft[0]) * 1000.0)

    if onsets:
        out.update({
            'source': 'frame_times',
            'onset_ms': float(np.median(onsets)),
            'cycle_ms': float(np.median(cycles)),
            'refresh_actual_hz': float(np.median(actual)),
            'spread_ms': float(np.max(cycles) - np.min(cycles)),
            'n_epochs_read': len(cycles),
        })
    else:
        # No usable frame record. Say so — the nominal numbers are what the
        # protocol intended, not what the retina saw, and every downstream
        # alignment inherits the error.
        out.update({
            'source': 'nominal',
            'onset_ms': nominal_onset_ms,
            'cycle_ms': nominal_cycle_ms,
            'refresh_actual_hz': float(declared_refresh_hz),
            'spread_ms': 0.0,
            'n_epochs_read': 0,
        })
        if verbose:
            print('repeat_timing: no usable frame_times_ms — falling back to '
                  'preTime/stimTime arithmetic. If the display did not run at '
                  f'{declared_refresh_hz:g} Hz, the cycles below are '
                  'misaligned by the same fraction throughout.')

    if verbose and out['source'] == 'frame_times':
        drift = out['cycle_ms'] - nominal_cycle_ms
        print(f"movie onset  {out['onset_ms']:8.1f} ms   "
              f"(nominal {nominal_onset_ms:.0f})")
        print(f"cycle        {out['cycle_ms']:8.1f} ms   "
              f"(nominal {nominal_cycle_ms:.0f}, {drift:+.0f} ms)")
        print(f"display ran at {out['refresh_actual_hz']:.3f} Hz against a "
              f"declared {declared_refresh_hz:g}, so the movie plays "
              f"{(out['refresh_actual_hz'] / declared_refresh_hz - 1) * 100:+.2f} % fast")
        print(f"read from {out['n_epochs_read']} epochs, cycle length spread "
              f"{out['spread_ms']:.0f} ms across them")
    return out


def estimate_cycle_period(pipeline, stim_block, epoch_indices, timing, *,
                          cell_types: Optional[Sequence[str]] = None,
                          cell_ids: Optional[Sequence[int]] = None,
                          bin_ms: float = 5.0,
                          sigma_ms: float = 25.0,
                          search_ms: float = 400.0,
                          verbose: bool = True) -> Dict:
    """Recover the cycle period from the spikes, as a check on the frames.

    The population rate during cycle 2 is the population rate during cycle 1,
    delayed by exactly one cycle — so the lag that maximises the correlation
    between them *is* the cycle period, measured through the retina rather
    than through the frame monitor. It is an independent route to the same
    number, and if the two disagree the frame record is not being read the way
    it is being assumed to be read.

    The scan is over lag, on a single continuous rate trace per cell, so it
    costs one pass over the spikes regardless of how many candidates are
    examined.

    Returns
    -------
    dict
        ``period_ms`` (the peak), ``lags_ms`` and ``correlation`` (the whole
        curve, for plotting), ``nominal_ms``, ``frame_ms`` (what
        ``timing`` said), and ``implied_refresh_hz``.
    """
    from scipy.ndimage import gaussian_filter1d

    epochs = [int(e) for e in epoch_indices]
    onset_ms = float(timing['onset_ms'])
    cycle_ms = float(timing['cycle_ms'])
    n_cycles = int(timing['n_cycles'])

    cells, spikes = _spike_source(pipeline, cell_types=cell_types,
                                  cell_ids=cell_ids)

    # One rate trace per cell per epoch over the whole movie, then averaged
    # across epochs *within condition-blind pooling* — the stimulus is the
    # same movie in every condition here, and pooling only sharpens the peak.
    span_ms = cycle_ms * n_cycles
    n_bins = int(round(span_ms / bin_ms))
    edges = onset_ms + np.arange(n_bins + 1) * bin_ms

    trace = np.zeros((len(cells), n_bins))
    for e in epochs:
        for i, cid in enumerate(cells['cell_id'].astype(int)):
            s = spikes[cid][e]
            if len(s):
                trace[i] += np.histogram(np.asarray(s, dtype=float), bins=edges)[0]
    trace = gaussian_filter1d(trace / (len(epochs) * bin_ms / 1000.0),
                              sigma_ms / bin_ms, axis=1, mode='nearest')

    # Correlate the first cycle against a window of the second, slid over
    # candidate lags. Both halves keep a margin so every lag compares the same
    # number of bins.
    margin = int(round(search_ms / bin_ms))
    per_cycle = n_bins // n_cycles
    a0, a1 = margin, per_cycle - margin
    ref = trace[:, a0:a1]
    ref = ref - ref.mean(axis=1, keepdims=True)

    lags = np.arange(per_cycle - 2 * margin, per_cycle + 2 * margin + 1)
    corr = np.full(lags.size, np.nan)
    for k, lag in enumerate(lags):
        b0, b1 = a0 + lag, a1 + lag
        if b1 > trace.shape[1]:
            continue
        cmp = trace[:, b0:b1]
        cmp = cmp - cmp.mean(axis=1, keepdims=True)
        denom = np.sqrt((ref ** 2).sum() * (cmp ** 2).sum())
        corr[k] = (ref * cmp).sum() / denom if denom > 0 else np.nan

    best = int(np.nanargmax(corr))
    # Parabolic interpolation on the three points around the peak: the bin
    # grid is 5 ms and the quantity of interest is a few tens.
    period_bins = float(lags[best])
    if 0 < best < lags.size - 1 and np.all(np.isfinite(corr[best - 1:best + 2])):
        y0, y1, y2 = corr[best - 1:best + 2]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            period_bins += 0.5 * (y0 - y2) / denom

    period_ms = period_bins * bin_ms
    declared = float(timing['refresh_declared_hz'])
    out = {
        'period_ms': period_ms,
        'lags_ms': lags * bin_ms,
        'correlation': corr,
        'peak_correlation': float(np.nanmax(corr)),
        'nominal_ms': float(timing['nominal_cycle_ms']),
        'frame_ms': cycle_ms,
        'implied_refresh_hz': declared * timing['nominal_cycle_ms'] / period_ms,
        'n_cells': len(cells),
    }
    if verbose:
        print(f"cycle period from spikes  {period_ms:8.1f} ms  "
              f"(r = {out['peak_correlation']:.3f} over {len(cells)} cells)")
        print(f"           from frames    {cycle_ms:8.1f} ms  "
              f"({period_ms - cycle_ms:+.1f} ms)")
        print(f"           nominal        {out['nominal_ms']:8.1f} ms  "
              f"({period_ms - out['nominal_ms']:+.1f} ms)")
        print(f"implied display rate {out['implied_refresh_hz']:.3f} Hz against "
              f"the declared {declared:g}")
    return out


def plot_cycle_period_scan(scan: Dict, *, ax=None, title: Optional[str] = None):
    """The lag scan, with the frame-derived and nominal periods marked."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(scan['lags_ms'] / 1000.0, scan['correlation'], color='0.25', lw=1.4)
    ax.axvline(scan['period_ms'] / 1000.0, color='C3', lw=1.2,
               label=f"spikes {scan['period_ms']/1000:.3f} s")
    ax.axvline(scan['frame_ms'] / 1000.0, color='C0', lw=1.2, ls='--',
               label=f"frames {scan['frame_ms']/1000:.3f} s")
    ax.axvline(scan['nominal_ms'] / 1000.0, color='0.6', lw=1.2, ls=':',
               label=f"nominal {scan['nominal_ms']/1000:.3f} s")
    ax.set_xlabel('assumed cycle period (s)')
    ax.set_ylabel('cycle 1 vs cycle 2\npopulation correlation')
    ax.set_title(title or 'the period that aligns the two presentations')
    ax.legend(frameon=False, fontsize=8)
    ax.figure.tight_layout()
    return ax.figure


# ---------------------------------------------------------------------------
# 2. Response timescale
# ---------------------------------------------------------------------------

@dataclass
class ResponseTimescale:
    """What :func:`response_timescale` measured.

    A small object rather than a DataFrame with things hung off ``.attrs``:
    pandas propagates ``attrs`` through most operations and compares them for
    equality when it concatenates, so an array or a frame parked there turns a
    later ``groupby(...).describe()`` into a ValueError about the truth value
    of a DataFrame. Scalars in ``attrs`` are fine and are used elsewhere here;
    these are not scalars.

    Attributes
    ----------
    per_cell : DataFrame
        One row per cell, for the spread. Not what the kernels come from.
    by_type : DataFrame
        Indexed by cell type: ``f_cutoff_hz``, ``tau_ms``, ``sigma_ms``,
        ``vp_cost_per_s``. Read off the per-type mean spectrum.
    freq_hz, coherence
        The spectra themselves, for :func:`plot_response_timescale`.
    """

    per_cell: pd.DataFrame
    by_type: pd.DataFrame
    freq_hz: np.ndarray
    coherence: Dict[str, np.ndarray]
    kernel_attenuation: float = 0.5

    def sigma_ms(self, cell_types: Optional[Sequence[str]] = None) -> Dict[str, float]:
        """``{cell_type: sigma}`` — hand straight to :func:`repeat_response`."""
        d = self.by_type['sigma_ms'].astype(float).to_dict()
        return d if cell_types is None else {k: d[k] for k in cell_types}

    def vp_cost_per_s(self, cell_types: Optional[Sequence[str]] = None) -> Dict[str, float]:
        """``{cell_type: cost}`` — for :func:`population_spike_distance`."""
        d = self.by_type['vp_cost_per_s'].astype(float).to_dict()
        return d if cell_types is None else {k: d[k] for k in cell_types}

    def __repr__(self) -> str:
        parts = ', '.join(f'{ct} {v:.0f} ms'
                          for ct, v in self.by_type['sigma_ms'].items())
        return f'ResponseTimescale({len(self.per_cell)} cells; sigma: {parts})'


def response_timescale(pipeline, stim_block, epochs_kept, timing, *,
                       condition_keys: Sequence[str],
                       cell_types: Optional[Sequence[str]] = None,
                       cell_ids: Optional[Sequence[int]] = None,
                       bin_ms: float = 2.0,
                       band_hz: float = 2.0,
                       noise_band_hz: Tuple[float, float] = (100.0, 240.0),
                       n_sd: float = 3.0,
                       min_coherence: float = 0.02,
                       kernel_attenuation: float = 0.5,
                       verbose: bool = True) -> ResponseTimescale:
    """The finest temporal structure each cell reproduces, from the spikes.

    Every later choice of temporal resolution — the PSTH kernel, the
    Victor-Purpura cost — has to come from somewhere, and choosing it to
    maximise the effect under test is how a pathway difference in timescale
    becomes whatever the analyst expected. This measures it instead, and the
    measurement is a **bandwidth**: the highest frequency at which repeats of
    the same movie agree.

    The estimator is the cross-trial coherence. Trials of one condition are a
    repeat set, so at each frequency the cross-spectrum between two trials
    estimates the signal power and the auto-spectrum estimates signal plus
    independent spiking noise. Their ratio

    ``C(f) = <Re X_i(f) conj X_j(f)>_(i != j) / <|X_i(f)|^2>_i``

    is the reproducible fraction of power at ``f``: 1 where the trials agree,
    0 where only shot noise is left.

    ``f_cutoff_hz`` is the highest frequency at which ``C`` is still above a
    noise floor measured on the cell's own spectrum in ``noise_band_hz``,
    where nothing is reproducible by construction — so the floor accounts for
    the finite number of repeats without assuming what it should be.
    ``tau_ms = 1000 / (2 f_c)`` is the half-period there, the finest feature
    the cell repeats. ``sigma_ms`` is the Gaussian kernel whose **half-power
    point sits at that cutoff**: ``sigma = sqrt(-ln(a)/2) / (pi f_c)`` with
    ``a = kernel_attenuation``. A kernel narrower than that spends its
    resolution on frequencies where the cell agrees with itself no better
    than chance.

    **A time-domain autocorrelation width does not work here** and was the
    first thing tried. These responses carry a large, highly reproducible slow
    envelope (the movie's own luminance drifts, and the retina is adapting),
    which puts most of the autocorrelation's area at long lags: two thirds of
    the cells never reached ``1/e`` inside 400 ms, and the rest returned the
    bin width. Slow structure lives at low frequency and simply does not
    affect a cutoff, which is why the spectral form is the one that answers
    the question being asked.

    ``epochs_kept`` and ``condition_keys`` are used to find the repeat sets:
    each (condition x cycle) group of epochs is a set of repeats of one movie
    under one history, and their spectra are averaged. Pooling across
    conditions instead would compare responses to different stimuli and read
    the difference as noise.

    Returns
    -------
    ResponseTimescale
        ``.by_type`` is what :func:`repeat_response` and
        :func:`population_spike_distance` consume, via ``.sigma_ms()`` and
        ``.vp_cost_per_s()``.
    """
    keys = list(condition_keys)
    onset_ms = float(timing['onset_ms'])
    cycle_ms = float(timing['cycle_ms'])
    n_cycles = int(timing['n_cycles'])

    cells, spikes = _spike_source(pipeline, cell_types=cell_types,
                                  cell_ids=cell_ids)

    n_bins = int(np.floor(cycle_ms / bin_ms))
    freq = np.fft.rfftfreq(n_bins, d=bin_ms / 1000.0)
    n_smooth = max(int(round(band_hz / (freq[1] - freq[0]))), 1)
    kernel = np.ones(n_smooth) / n_smooth

    # Repeat sets: epochs of one condition, one cycle at a time.
    repeat_sets = []
    for _, rows in _grouped(epochs_kept, keys):
        eps = rows['epoch'].astype(int).tolist()
        if len(eps) >= 2:
            for c in range(n_cycles):
                repeat_sets.append((eps, c))
    if not repeat_sets:
        raise ValueError('response_timescale: no condition has 2+ epochs, so '
                         'there is no repeat set to measure coherence on')

    ids = cells['cell_id'].astype(int).to_numpy()
    cross = np.zeros((len(ids), freq.size))
    auto = np.zeros((len(ids), freq.size))
    for eps, c in repeat_sets:
        start = onset_ms + c * cycle_ms
        edges = start + np.arange(n_bins + 1) * bin_ms
        counts = np.zeros((len(ids), len(eps), n_bins))
        for k, e in enumerate(eps):
            for i, cid in enumerate(ids):
                s = spikes[cid][e]
                if len(s):
                    counts[i, k] = np.histogram(
                        np.asarray(s, dtype=float), bins=edges)[0]
        counts -= counts.mean(axis=2, keepdims=True)
        X = np.fft.rfft(counts, axis=2)
        power = (X * np.conj(X)).real                      # (cells, trials, f)
        total = X.sum(axis=1)
        # sum_{i != j} Re X_i conj X_j = |sum X|^2 - sum |X|^2
        cross += ((total * np.conj(total)).real - power.sum(axis=1))
        auto += power.sum(axis=1) * (len(eps) - 1)

    with np.errstate(invalid='ignore', divide='ignore'):
        coh = np.where(auto > 0, cross / auto, np.nan)
    coh_s = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode='same'),
                                1, np.nan_to_num(coh))

    # sigma such that the Gaussian's transfer function exp(-2 pi^2 f^2 s^2)
    # equals `kernel_attenuation` at the cutoff.
    sigma_at = np.sqrt(-np.log(kernel_attenuation) / 2.0) / np.pi

    nlo, nhi = noise_band_hz
    noise = np.flatnonzero((freq >= nlo) & (freq <= min(nhi, freq[-1])))
    if noise.size < 10:
        raise ValueError(f'noise_band_hz {noise_band_hz} covers only '
                         f'{noise.size} frequency bins of a spectrum reaching '
                         f'{freq[-1]:.0f} Hz; widen it or use finer bin_ms')
    rows = []
    for i, (cid, ctype) in enumerate(zip(ids, cells['cell_type'])):
        floor = float(coh_s[i, noise].mean() + n_sd * coh_s[i, noise].std())
        peak_k = int(np.nanargmax(coh_s[i, :noise[0]]))
        peak = float(coh_s[i, peak_k])
        if not np.isfinite(peak) or peak < max(min_coherence, floor):
            rows.append({'cell_id': int(cid), 'cell_type': ctype,
                         'f_cutoff_hz': np.nan, 'tau_ms': np.nan,
                         'sigma_ms': np.nan, 'coherence': peak,
                         'noise_floor': floor,
                         'n_repeat_sets': len(repeat_sets)})
            continue
        # Walk up from the peak to the first crossing of the floor: the last
        # frequency the cell still reproduces, rather than a stray excursion
        # somewhere out in the noise.
        after = np.flatnonzero(coh_s[i, peak_k:] < floor)
        if after.size == 0:
            f_c = float(freq[-1])
        else:
            k = peak_k + int(after[0])
            y0, y1 = coh_s[i, k - 1], coh_s[i, k]
            frac = (y0 - floor) / (y0 - y1) if y0 > y1 else 0.0
            f_c = float(freq[k - 1] + frac * (freq[k] - freq[k - 1]))
        tau_ms = 1000.0 / (2.0 * f_c)
        rows.append({'cell_id': int(cid), 'cell_type': ctype,
                     'f_cutoff_hz': f_c, 'tau_ms': tau_ms,
                     'sigma_ms': 1000.0 * sigma_at / f_c,
                     'coherence': peak, 'noise_floor': floor,
                     'n_repeat_sets': len(repeat_sets)})

    out = pd.DataFrame(rows)

    # The per-type numbers are read off the per-type mean spectrum, not
    # averaged over the per-cell ones. A kernel width is a property of a
    # pathway and is applied to a pathway, and one cell's coherence over three
    # repeats has a noise floor an order of magnitude above the floor of a
    # mean over ninety cells — so per-cell cutoffs are systematically early,
    # and their median would set the smoothing from the noise level rather
    # than from the kinetics. The per-cell column stays because its spread is
    # worth seeing; it is not what the kernel comes from.
    type_curves = {ct: coh_s[(cells['cell_type'] == ct).to_numpy()].mean(0)
                   for ct in sorted(cells['cell_type'].dropna().unique())}
    per_type = []
    for ct, curve in type_curves.items():
        floor = float(curve[noise].mean() + n_sd * curve[noise].std())
        peak_k = int(np.nanargmax(curve[:noise[0]]))
        after = np.flatnonzero(curve[peak_k:] < floor)
        if curve[peak_k] < max(min_coherence, floor):
            f_c = np.nan
        elif after.size == 0:
            f_c = float(freq[-1])
        else:
            k = peak_k + int(after[0])
            y0, y1 = curve[k - 1], curve[k]
            frac = (y0 - floor) / (y0 - y1) if y0 > y1 else 0.0
            f_c = float(freq[k - 1] + frac * (freq[k] - freq[k - 1]))
        per_type.append({'cell_type': ct, 'f_cutoff_hz': f_c,
                         'tau_ms': 1000.0 / (2.0 * f_c),
                         'sigma_ms': round(1000.0 * sigma_at / f_c),
                         'peak_coherence': float(curve[peak_k]),
                         'noise_floor': floor})
    by_type = pd.DataFrame(per_type).set_index('cell_type')
    by_type['n_cells'] = out.groupby('cell_type').size()
    by_type['cell_median_tau_ms'] = out.groupby('cell_type')['tau_ms'].median()
    # Victor-Purpura shifts a spike by dt at cost q*dt, and deleting plus
    # inserting costs 2, so spikes farther apart than 2/q are never shifted:
    # 2/q is the metric's own resolution, and q = 2/tau sets it to the
    # timescale just measured.
    by_type['vp_cost_per_s'] = 2.0 / (by_type['tau_ms'] / 1000.0)
    by_type = by_type[['n_cells', 'peak_coherence', 'noise_floor',
                       'f_cutoff_hz', 'tau_ms', 'sigma_ms', 'vp_cost_per_s',
                       'cell_median_tau_ms']]

    if verbose:
        n_bad = int(out['tau_ms'].isna().sum())
        print(f'{len(out) - n_bad} of {len(out)} cells reproduce anything '
              f'(peak coherence above {min_coherence:g} and above their own '
              f'{nlo:g}–{nhi:g} Hz noise floor), over {len(repeat_sets)} '
              f'repeat sets of {len(repeat_sets[0][0])} trials\n')
        with pd.option_context('display.float_format', lambda v: f'{v:.1f}'):
            print(by_type.to_string())
    return ResponseTimescale(per_cell=out, by_type=by_type, freq_hz=freq,
                             coherence=type_curves,
                             kernel_attenuation=kernel_attenuation)


def plot_response_timescale(scale: ResponseTimescale, *,
                            cell_types: Optional[Sequence[str]] = None,
                            max_hz: float = 30.0, axes=None):
    """The coherence spectra the timescales were read off, and the timescales.

    **Left**: mean cross-trial coherence against frequency, one curve per
    type, with each type's noise floor and the cutoff it implies. This is the
    panel to read — where the curves separate is where the pathways stop
    carrying the same temporal detail, and the kernel widths come from where
    each meets its own floor. **Right**: the per-cell cutoffs, as a
    half-period, for the spread; the median of these is *not* the number the
    kernel comes from, and the two are printed together so the difference is
    visible rather than surprising.
    """
    import matplotlib.pyplot as plt
    from .style import colors_for_conditions

    per_cell = scale.per_cell
    types = list(cell_types) if cell_types is not None else \
        list(scale.by_type.index)
    types = [t for t in types if t in scale.coherence]
    colors = colors_for_conditions(types)

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9.8, 3.4))
    ax0, ax1 = axes

    freq = scale.freq_hz
    keep = freq <= max_hz
    for ct in types:
        ax0.plot(freq[keep], scale.coherence[ct][keep], color=colors[ct],
                 lw=1.8, label=f'{ct}  σ {scale.by_type.at[ct, "sigma_ms"]:.0f} ms')
        f_c = float(scale.by_type.at[ct, 'f_cutoff_hz'])
        ax0.axvline(f_c, color=colors[ct], lw=0.9, ls='--', alpha=.7)
    ax0.axhline(float(scale.by_type['noise_floor'].max())
                if 'noise_floor' in scale.by_type else 0,
                color='0.7', lw=0.8, ls=':')
    ax0.set_xlabel('frequency (Hz)')
    ax0.set_ylabel('cross-trial coherence\n(reproducible fraction of power)')
    ax0.set_title('what each pathway repeats')
    ax0.legend(frameon=False, fontsize=8)

    for i, ct in enumerate(types):
        v = per_cell.loc[per_cell['cell_type'] == ct, 'tau_ms'].dropna().to_numpy()
        if not v.size:
            continue
        ax1.scatter(v, np.full(v.size, i) + np.random.uniform(-.15, .15, v.size),
                    s=8, alpha=.45, color=colors[ct], lw=0)
        ax1.plot([np.median(v)] * 2, [i - .3, i + .3], color='k', lw=2)
        ax1.text(np.median(v), i + .40, f'{np.median(v):.0f} ms', ha='center',
                 fontsize=8)
    ax1.set_yticks(range(len(types)))
    ax1.set_yticklabels(types)
    ax1.set_xlabel('per-cell half-period at the cutoff (ms)')
    ax1.set_title('spread across cells (noisier floor, so later cutoffs)')
    ax1.figure.tight_layout()
    return ax1.figure


# ---------------------------------------------------------------------------
# 3. The response matrices
# ---------------------------------------------------------------------------

@dataclass
class RepeatResponse:
    """Firing rate as ``(cell, epoch, cycle, movie time)``.

    One object holds everything the population analyses read, so the cell set,
    the smoothing and the cycle alignment are decided once and cannot drift
    apart between sections.

    Attributes
    ----------
    cells : DataFrame
        ``cell_id``, ``cell_type``, the receptive-field columns, and the
        ``sigma_ms`` each row was smoothed with.
    epochs : DataFrame
        One row per epoch in ``rates``, carrying the condition columns. Row
        order matches axis 1 of ``rates``.
    rates : ndarray, (n_cells, n_epochs, n_cycles, n_t), float32
        Firing rate in Hz.
    t_s : ndarray, (n_t,)
        Movie time within a cycle, seconds from cycle onset.
    timing : dict
        What :func:`repeat_timing` returned.
    condition_keys : list[str]
        Columns of ``epochs`` that define a condition.
    """

    cells: pd.DataFrame
    epochs: pd.DataFrame
    rates: np.ndarray
    t_s: np.ndarray
    timing: Dict
    condition_keys: List[str]
    bin_ms: float = 5.0
    sigma_ms: Dict = field(default_factory=dict)

    @property
    def n_cycles(self) -> int:
        return self.rates.shape[2]

    def __repr__(self) -> str:
        mb = self.rates.nbytes / 1e6
        return (f'RepeatResponse({self.rates.shape[0]} cells x '
                f'{self.rates.shape[1]} epochs x {self.n_cycles} cycles x '
                f'{self.rates.shape[3]} bins of {self.bin_ms:g} ms, {mb:.0f} MB)')

    def cell_mask(self, cell_type: Optional[str] = None) -> np.ndarray:
        """Boolean row mask; ``None`` or ``'all'`` selects every cell."""
        if cell_type in (None, 'all'):
            return np.ones(len(self.cells), dtype=bool)
        return (self.cells['cell_type'] == cell_type).to_numpy()

    def epoch_mask(self, **conditions) -> np.ndarray:
        """Boolean epoch mask from ``key=value`` condition constraints."""
        mask = np.ones(len(self.epochs), dtype=bool)
        for key, value in conditions.items():
            mask &= (self.epochs[key] == value).to_numpy()
        return mask


def repeat_response(pipeline, stim_block, epochs_kept, timing, *,
                    condition_keys: Sequence[str],
                    cell_types: Optional[Sequence[str]] = None,
                    cell_ids: Optional[Sequence[int]] = None,
                    sigma_ms: Union[float, Mapping[str, float]] = 25.0,
                    bin_ms: float = 5.0,
                    verbose: bool = True) -> RepeatResponse:
    """Build the ``(cell, epoch, cycle, movie time)`` rate array.

    Every cycle of every epoch is placed on the **same** movie-time axis, so
    a column of ``rates`` is one moment of the movie regardless of which
    presentation or which epoch it came from. That is what makes the later
    correlations comparisons of representation rather than of timing —
    provided ``timing`` is right, which is what :func:`repeat_timing` and
    :func:`estimate_cycle_period` between them establish.

    ``sigma_ms`` may be one number for every cell or a mapping from cell type
    to its own value. The mapping is the point of :func:`response_timescale`:
    smoothing a parasol response with a midget's kernel throws away the
    structure that distinguishes the pathways, and the reverse buries a midget
    response in shot noise. A single number is the common-timescale control
    that says whether any pathway difference survives matched resolution.

    Spikes are binned into ``bin_ms`` bins and smoothed with a Gaussian in
    ``mode='nearest'`` — not ``'wrap'``. The loop is a repeat of the
    trajectory, not a periodic stimulus: cycle 2 begins where cycle 1 began,
    which is one ordinary eye-movement step away from where cycle 1 ended,
    and wrapping would smear the two together across a boundary the retina
    saw as a step.
    """
    from scipy.ndimage import gaussian_filter1d

    keys = list(condition_keys)
    ep_df = epochs_kept.reset_index(drop=True)
    epoch_ids = ep_df['epoch'].astype(int).to_numpy()

    onset_ms = float(timing['onset_ms'])
    cycle_ms = float(timing['cycle_ms'])
    n_cycles = int(timing['n_cycles'])

    cells, spikes = _spike_source(pipeline, cell_types=cell_types,
                                  cell_ids=cell_ids)

    n_t = int(np.floor(cycle_ms / bin_ms))
    t_s = (np.arange(n_t) + 0.5) * bin_ms / 1000.0

    rates = np.zeros((len(cells), len(epoch_ids), n_cycles, n_t), dtype=np.float32)
    for k, e in enumerate(epoch_ids):
        for c in range(n_cycles):
            start = onset_ms + c * cycle_ms
            edges = start + np.arange(n_t + 1) * bin_ms
            for i, cid in enumerate(cells['cell_id'].astype(int)):
                s = spikes[cid][e]
                if len(s):
                    rates[i, k, c] = np.histogram(
                        np.asarray(s, dtype=float), bins=edges)[0]
    rates /= np.float32(bin_ms / 1000.0)

    # Per-cell sigma, applied one group at a time so each call filters a
    # contiguous block rather than a row.
    if isinstance(sigma_ms, Mapping):
        per_cell = cells['cell_type'].map(lambda t: float(sigma_ms.get(t, np.nan)))
        if per_cell.isna().any():
            missing = sorted(cells.loc[per_cell.isna(), 'cell_type'].unique())
            raise KeyError(f'sigma_ms has no entry for cell type(s) {missing}')
    else:
        per_cell = pd.Series(float(sigma_ms), index=cells.index)
    cells = cells.copy()
    cells['sigma_ms'] = per_cell.to_numpy()

    for value in sorted(set(per_cell)):
        rows = np.flatnonzero((per_cell == value).to_numpy())
        if value <= 0:
            continue
        rates[rows] = gaussian_filter1d(rates[rows], value / bin_ms, axis=-1,
                                        mode='nearest')

    rr = RepeatResponse(cells=cells.reset_index(drop=True), epochs=ep_df,
                        rates=rates, t_s=t_s, timing=dict(timing),
                        condition_keys=keys, bin_ms=float(bin_ms),
                        sigma_ms=(dict(sigma_ms) if isinstance(sigma_ms, Mapping)
                                  else {'all': float(sigma_ms)}))
    if verbose:
        print(rr)
        print(f'movie time 0–{t_s[-1]:.2f} s per cycle, {n_cycles} cycles, '
              f'aligned on {onset_ms / 1000:.3f} s + k x {cycle_ms / 1000:.3f} s')
        print('smoothing: ' + ', '.join(
            f'{ct} {v:g} ms' for ct, v in
            sorted(cells.groupby('cell_type')['sigma_ms'].first().items())))
    return rr


# ---------------------------------------------------------------------------
# 4. Population similarity
# ---------------------------------------------------------------------------

def _normalize_matrix(mat: np.ndarray, how: str) -> np.ndarray:
    """Prepare one (cell x time) matrix for vectorising and correlating.

    - ``raw`` — untouched. The vector correlation is then dominated by how
      much cells differ from each other in mean rate, which is identical
      between conditions by construction, so this number is high whatever the
      response does. Report it, but read it against the other two.
    - ``centered`` — each cell's own mean removed. Differences in gain
      survive; the across-cell mean-rate offset that inflates ``raw`` does
      not.
    - ``shape`` — each cell z-scored. Gain and baseline are gone, so what is
      left is whether the temporal pattern matches.
    """
    if how == 'raw':
        return mat
    out = mat - mat.mean(axis=1, keepdims=True)
    if how == 'centered':
        return out
    if how == 'shape':
        sd = out.std(axis=1, keepdims=True)
        return np.divide(out, sd, out=np.zeros_like(out), where=sd > 0)
    raise ValueError(f'unknown normalization {how!r}; '
                     f'expected one of {NORMALIZATIONS}')


def _vec_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(float)
    b = b.ravel().astype(float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else np.nan


def _row_corr_mean(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over cells of the per-cell temporal correlation."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (a * b).sum(1) / denom
    return float(np.nanmean(r))


def _reliability(trials: np.ndarray, how: str, method: str) -> Tuple[float, int]:
    """Reliability of the mean over ``n`` trials, from the trials themselves.

    ``trials`` is ``(n_trials, n_cells, n_t)``.

    ``method='pairs'`` (default) correlates every ordered pair of single
    trials and applies Spearman-Brown to get from one trial to ``n``:
    ``r_n = n r_1 / (1 + (n-1) r_1)``. With three or four repeats that
    enumerates every distinct pair, which is the finest-grained version of
    "average over all splits" and the only one that does not have to choose
    between unequal halves.

    ``method='split_half'`` averages ``corr(mean(A), mean(B))`` over every
    partition into two non-empty groups and applies the same correction. With
    an odd number of trials the halves are unequal, which biases the estimate
    down; it is here because it is the form the comparison is usually written
    in, not because it is better.
    """
    n = trials.shape[0]
    if n < 2:
        return np.nan, 0
    if method == 'pairs':
        vals = [_vec_corr(_normalize_matrix(trials[i], how),
                          _normalize_matrix(trials[j], how))
                for i in range(n) for j in range(i + 1, n)]
        r1 = float(np.nanmean(vals))
        k = n
    elif method == 'split_half':
        from itertools import combinations
        vals, sizes = [], []
        for size in range(1, n // 2 + 1):
            for group in combinations(range(n), size):
                other = [i for i in range(n) if i not in group]
                if size == n - size and group[0] != 0:
                    continue    # each even split once
                vals.append(_vec_corr(
                    _normalize_matrix(trials[list(group)].mean(0), how),
                    _normalize_matrix(trials[other].mean(0), how)))
                sizes.append((size + len(other)) / 2.0)
        r1 = float(np.nanmean(vals))
        k = n / float(np.mean(sizes))
    else:
        raise ValueError(f'unknown reliability method {method!r}')
    if not np.isfinite(r1) or r1 <= -1.0 / (k - 1):
        return np.nan, n
    return float(k * r1 / (1.0 + (k - 1) * r1)), n


def population_similarity(rr: RepeatResponse, *,
                          contrast_key: str,
                          group_keys: Optional[Sequence[str]] = None,
                          cell_types: Optional[Sequence[str]] = None,
                          normalizations: Sequence[str] = NORMALIZATIONS,
                          reliability_method: str = 'pairs',
                          time_slice: Optional[Tuple[float, float]] = None,
                          min_reliability: float = 0.1,
                          match_cells: Optional[Union[int, str]] = None,
                          n_resamples: int = 20,
                          random_state: int = 0) -> pd.DataFrame:
    """Cross-condition population correlation, per cycle, noise-corrected.

    The headline measurement. For each group (a stimulus that both conditions
    saw — here, one natural image), each cycle, each cell type and each
    normalisation:

    - ``rho_cross`` — correlation between the two conditions' trial-averaged
      ``(cell x time)`` matrices, vectorised. This is the population
      quantity: it asks whether the *joint* pattern across cells and time
      matches, not whether each cell's trace does.
    - ``rho_within_a`` / ``rho_within_b`` — the reliability each condition's
      own repeats support, on the same normalisation. A population that
      cannot reproduce itself cannot resemble anything else, and without this
      a noisier population reads as a more adapted one.
    - ``rho_corrected`` — ``rho_cross / sqrt(rho_within_a * rho_within_b)``.
      Near 1 means the two histories are as similar as the repeats allow;
      below 1 means adaptation history changed the representation by more
      than trial-to-trial variability.
    - ``rho_cell`` — the mean over cells of the single-cell temporal
      correlation. Reported beside ``rho_cross`` because they answer different
      questions: whether individual traces converge, and whether the joint
      cell-by-time pattern does. Agreement between them is what makes a
      pathway claim strong.

    ``contrast_key`` must take exactly two levels within a group (the two
    adaptation histories). ``group_keys`` defaults to every other condition
    key — **do not** drop it: a different image is a different movie, and
    pooling them correlates one stimulus against another.

    ``min_reliability`` guards the division: below it the correction is
    dividing by noise and ``rho_corrected`` comes back NaN with
    ``corrected_valid`` False, rather than as a number somewhere near
    infinity.

    ``match_cells`` is the control for the pathway comparison. A vector
    correlation over 94 midget cells and one over 41 parasol cells are not the
    same estimator — the second has fewer numbers in it and so more variance —
    and a claim that one pathway converges further should not turn on which
    type the array happened to sample more of. Pass ``'min'`` to draw every
    type down to the smallest type's count, ``n_resamples`` times without
    replacement, and average; ``n_cells`` then reports the matched count and
    ``n_available`` what each type had. The ``'all'`` row is the whole
    population and is never subsampled.
    """
    keys = list(group_keys) if group_keys is not None else \
        [k for k in rr.condition_keys if k != contrast_key]
    types = ['all'] + (list(cell_types) if cell_types is not None
                       else sorted(rr.cells['cell_type'].dropna().unique()))

    available = {ct: int(rr.cell_mask(None if ct == 'all' else ct).sum())
                 for ct in types}
    if match_cells == 'min':
        match_n = min(v for ct, v in available.items() if ct != 'all') \
            if len(types) > 1 else None
    elif match_cells is None:
        match_n = None
    else:
        match_n = int(match_cells)

    rng = np.random.default_rng(random_state)
    sl = _time_slice(rr, time_slice)
    rows = []
    for group_vals, group_rows in _grouped(rr.epochs, keys):
        levels = sorted(group_rows[contrast_key].unique())
        if len(levels) != 2:
            print(f'skipping {_label(keys, group_vals)}: {contrast_key} takes '
                  f'{len(levels)} level(s) here, needs exactly 2')
            continue
        idx = {lv: group_rows.index[group_rows[contrast_key] == lv].to_numpy()
               for lv in levels}

        for ct in types:
            pool = np.flatnonzero(rr.cell_mask(None if ct == 'all' else ct))
            if pool.size < 2:
                continue
            subsample = (match_n is not None and ct != 'all'
                         and match_n < pool.size)
            draws = ([rng.choice(pool, match_n, replace=False)
                      for _ in range(n_resamples)] if subsample else [pool])
            for cycle in range(rr.n_cycles):
                for how in normalizations:
                    acc = []
                    for rows_sel in draws:
                        trials = {lv: rr.rates[np.ix_(rows_sel, idx[lv])]
                                      [:, :, cycle, sl]
                                      .transpose(1, 0, 2).astype(float)
                                  for lv in levels}
                        a = _normalize_matrix(trials[levels[0]].mean(0), how)
                        b = _normalize_matrix(trials[levels[1]].mean(0), how)
                        rho_cross = _vec_corr(a, b)
                        rel_a, n_a = _reliability(trials[levels[0]], how,
                                                  reliability_method)
                        rel_b, n_b = _reliability(trials[levels[1]], how,
                                                  reliability_method)
                        ok = (np.isfinite(rel_a) and np.isfinite(rel_b)
                              and rel_a > min_reliability
                              and rel_b > min_reliability)
                        acc.append({
                            'n_trials_a': n_a, 'n_trials_b': n_b,
                            'rho_cross': rho_cross,
                            'rho_within_a': rel_a, 'rho_within_b': rel_b,
                            'rho_corrected': (rho_cross / np.sqrt(rel_a * rel_b)
                                              if ok else np.nan),
                            'corrected_valid': float(ok),
                            'rho_cell': _row_corr_mean(
                                trials[levels[0]].mean(0),
                                trials[levels[1]].mean(0)),
                            'rate_a_hz': float(trials[levels[0]].mean()),
                            'rate_b_hz': float(trials[levels[1]].mean()),
                        })
                    mean = pd.DataFrame(acc).mean(numeric_only=True).to_dict()
                    rows.append({
                        **dict(zip(keys, group_vals)),
                        'cell_type': ct,
                        'cycle': cycle + 1,
                        'normalize': how,
                        'level_a': levels[0],
                        'level_b': levels[1],
                        'n_cells': int(draws[0].size),
                        'n_available': available[ct],
                        'n_resamples': len(draws),
                        **mean,
                        'n_trials_a': int(mean['n_trials_a']),
                        'n_trials_b': int(mean['n_trials_b']),
                        'corrected_valid': mean['corrected_valid'] > 0.5,
                    })

    out = pd.DataFrame(rows)
    out.attrs['contrast_key'] = contrast_key
    out.attrs['group_keys'] = keys
    out.attrs['matched_n_cells'] = match_n
    return out


def cycle_interaction(rr: RepeatResponse, *,
                      contrast_key: str,
                      group_keys: Optional[Sequence[str]] = None,
                      cell_types: Optional[Sequence[str]] = None,
                      normalizations: Sequence[str] = NORMALIZATIONS,
                      time_slice: Optional[Tuple[float, float]] = None
                      ) -> pd.DataFrame:
    """First versus second presentation, within and between adaptation histories.

    Two things, because the first alone does not separate adaptation from
    ordinary repetition:

    - ``rho_cycle`` — correlation between cycle 1 and cycle 2 *within* one
      condition. The condition further from the movie's own luminance should
      change more between presentations.
    - the interaction. ``dR_a = R_a2 - R_a1`` and ``dR_b = R_b2 - R_b1`` are
      what each history changed between presentations. If that change were a
      generic repetition effect the two would be the same vector, so
      ``rho_delta`` (their correlation) would be high and ``delta_ratio``
      (the norm of their difference over the mean of their norms) near zero.
      A change that depends on the preceding luminance shows up as the
      opposite.

    Note that ``rho_cycle`` is not comparable across conditions without
    ``rho_within`` from :func:`population_similarity` — a condition whose
    repeats are noisier will have a lower cycle-to-cycle correlation for that
    reason alone. ``rho_cycle_corrected`` divides it by the within-condition
    reliability of each cycle for that reason.
    """
    keys = list(group_keys) if group_keys is not None else \
        [k for k in rr.condition_keys if k != contrast_key]
    types = ['all'] + (list(cell_types) if cell_types is not None
                       else sorted(rr.cells['cell_type'].dropna().unique()))
    if rr.n_cycles < 2:
        raise ValueError('cycle_interaction needs at least 2 cycles')

    sl = _time_slice(rr, time_slice)
    rows = []
    for group_vals, group_rows in _grouped(rr.epochs, keys):
        levels = sorted(group_rows[contrast_key].unique())
        if len(levels) != 2:
            continue
        idx = {lv: group_rows.index[group_rows[contrast_key] == lv].to_numpy()
               for lv in levels}
        for ct in types:
            cmask = np.flatnonzero(rr.cell_mask(None if ct == 'all' else ct))
            if cmask.size < 2:
                continue
            block = {lv: rr.rates[np.ix_(cmask, idx[lv])][..., sl].astype(float)
                     for lv in levels}   # (cells, trials, cycles, t)
            for how in normalizations:
                mean_c = {lv: [_normalize_matrix(block[lv][:, :, c].mean(1), how)
                               for c in range(rr.n_cycles)] for lv in levels}
                rel = {}
                for lv in levels:
                    for c in range(rr.n_cycles):
                        rel[(lv, c)] = _reliability(
                            block[lv][:, :, c].transpose(1, 0, 2), how, 'pairs')[0]
                d = {lv: mean_c[lv][1] - mean_c[lv][0] for lv in levels}
                n0, n1 = np.linalg.norm(d[levels[0]]), np.linalg.norm(d[levels[1]])
                for lv in levels:
                    denom = np.sqrt(rel[(lv, 0)] * rel[(lv, 1)])
                    r = _vec_corr(mean_c[lv][0], mean_c[lv][1])
                    base = np.linalg.norm(mean_c[lv][0])
                    rows.append({
                        **dict(zip(keys, group_vals)),
                        'cell_type': ct, 'normalize': how,
                        contrast_key: lv,
                        'n_cells': int(cmask.size),
                        'rho_cycle': r,
                        'rho_cycle_corrected': (r / denom if np.isfinite(denom)
                                                and denom > 0.1 else np.nan),
                        # How large the between-cycle change is relative to the
                        # first cycle's own response, so the two conditions'
                        # changes are comparable despite different rates.
                        'delta_rel': (float(np.linalg.norm(d[lv]) / base)
                                      if base > 0 else np.nan),
                        'rho_delta': _vec_corr(d[levels[0]], d[levels[1]]),
                        # 0 if both histories changed identically (a pure
                        # repetition effect); sqrt(2) if the two changes are
                        # orthogonal; 2 if opposite.
                        'delta_ratio': (float(np.linalg.norm(
                            d[levels[0]] - d[levels[1]]) / ((n0 + n1) / 2))
                            if (n0 + n1) > 0 else np.nan),
                    })
    out = pd.DataFrame(rows)
    out.attrs['contrast_key'] = contrast_key
    out.attrs['group_keys'] = keys
    return out


# ---------------------------------------------------------------------------
# 5. Spike-train distance
# ---------------------------------------------------------------------------

def population_spike_distance(pipeline, rr: RepeatResponse, *,
                              contrast_key: str,
                              group_keys: Optional[Sequence[str]] = None,
                              cost_per_s: Union[float, Mapping[str, float]],
                              cell_types: Optional[Sequence[str]] = None,
                              normalize_by_count: bool = True,
                              verbose: bool = True) -> pd.DataFrame:
    """Victor-Purpura distance between histories, in excess of repeat variability.

    The single-cell metric, kept as it was, with the one correction that makes
    it mean something across conditions: subtract the distance the *same*
    condition's repeats already show.

    ``d_excess = d_cross - (d_within_a + d_within_b) / 2``

    Two things this does not fix, both reported alongside so they can be read:

    - **VP distance grows with spike count.** ``normalize_by_count`` divides
      by ``n_a + n_b``, which removes most of it, but the two histories fire
      at genuinely different rates and at cost 0 the metric *is* the count
      difference ``|n_a - n_b|``. ``d_excess_count_only`` is the whole
      statistic recomputed that way — what the excess would be if timing
      carried nothing — and ``d_excess`` has to be read against it. On the
      block this was written against ``d_excess`` is roughly half of
      ``d_excess_count_only`` and both fall by the same factor between cycles,
      so the decline is rate convergence seen through a timing metric rather
      than independent evidence about timing.
    - **The cost is a choice.** Pass a mapping from cell type to
      ``cost_per_s`` — ``response_timescale`` computes ``2/tau`` for exactly
      this — so a fast pathway is not scored on a slow pathway's clock. A
      single float is the common-timescale control.
    """
    from .victor_purpura import victor_purpura_batch_pairs

    keys = list(group_keys) if group_keys is not None else \
        [k for k in rr.condition_keys if k != contrast_key]
    types = list(cell_types) if cell_types is not None else \
        sorted(rr.cells['cell_type'].dropna().unique())

    onset_ms = float(rr.timing['onset_ms'])
    cycle_ms = float(rr.timing['cycle_ms'])
    _, spikes = _spike_source(pipeline, cell_ids=rr.cells['cell_id'].tolist())

    # One flat train list; every (cell, epoch, cycle) window appears once and
    # every pair indexes into it, so the whole workload is a single C call.
    trains: List[np.ndarray] = []
    train_index: Dict[Tuple[int, int, int], int] = {}
    epoch_ids = rr.epochs['epoch'].astype(int).to_numpy()
    for i, cid in enumerate(rr.cells['cell_id'].astype(int)):
        for k, e in enumerate(epoch_ids):
            s = np.asarray(spikes[cid][e], dtype=float)
            for c in range(rr.n_cycles):
                lo = onset_ms + c * cycle_ms
                w = s[(s >= lo) & (s < lo + cycle_ms)]
                train_index[(i, k, c)] = len(trains)
                trains.append((w - lo) / 1000.0)

    # Group pairs by cost: the C kernel takes one cost per call.
    def _cost_for(ctype: str) -> float:
        if isinstance(cost_per_s, Mapping):
            if ctype not in cost_per_s:
                raise KeyError(f'cost_per_s has no entry for {ctype!r}')
            return float(cost_per_s[ctype])
        return float(cost_per_s)

    plan = []      # (row_key, kind, [pair indices])
    pairs_by_cost: Dict[float, List[Tuple[int, int]]] = {}
    for group_vals, group_rows in _grouped(rr.epochs, keys):
        levels = sorted(group_rows[contrast_key].unique())
        if len(levels) != 2:
            continue
        pos = {lv: np.flatnonzero(
            (rr.epochs[contrast_key] == lv).to_numpy()
            & np.isin(np.arange(len(rr.epochs)), group_rows.index.to_numpy()))
            for lv in levels}
        for ct in types:
            cost = _cost_for(ct)
            bucket = pairs_by_cost.setdefault(cost, [])
            for i in np.flatnonzero(rr.cell_mask(ct)):
                for c in range(rr.n_cycles):
                    def _slot(kind, ka, kb):
                        start = len(bucket)
                        for x in ka:
                            for y in kb:
                                if x == y:
                                    continue
                                bucket.append((train_index[(i, x, c)],
                                               train_index[(i, y, c)]))
                        plan.append(((*group_vals, ct, c + 1,
                                      int(rr.cells.at[i, 'cell_id'])),
                                     kind, cost, start, len(bucket)))
                    _slot('cross', pos[levels[0]], pos[levels[1]])
                    _slot('within_a', pos[levels[0]], pos[levels[0]])
                    _slot('within_b', pos[levels[1]], pos[levels[1]])

    if verbose:
        total = sum(len(v) for v in pairs_by_cost.values())
        print(f'{total} Victor-Purpura pairs over {len(trains)} trains, '
              f'{len(pairs_by_cost)} cost value(s): '
              + ', '.join(f'{c:.1f}/s' for c in sorted(pairs_by_cost)))

    results: Dict[float, np.ndarray] = {}
    counts = np.array([len(t) for t in trains], dtype=float)
    for cost, bucket in pairs_by_cost.items():
        if not bucket:
            results[cost] = np.zeros(0)
            continue
        results[cost] = victor_purpura_batch_pairs(
            trains, np.asarray(bucket, dtype=np.int32), cost)

    counts_by_cost = {cost: np.asarray(bucket, dtype=int)
                      for cost, bucket in pairs_by_cost.items()}

    def _mean(v):
        """nanmean that returns NaN quietly for an all-NaN slice.

        A cell with no spikes in a window makes every pair it appears in
        undefined once the count normalisation divides by ``n_a + n_b``; that
        is a fact about the cell, not an error worth a RuntimeWarning per
        occurrence.
        """
        v = np.asarray(v, dtype=float)
        return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan

    acc: Dict[Tuple, Dict] = {}
    for row_key, kind, cost, start, end in plan:
        d = results[cost][start:end]
        idx = counts_by_cost[cost][start:end]
        if d.size == 0:
            continue
        tot = counts[idx[:, 0]] + counts[idx[:, 1]]
        floor = np.abs(counts[idx[:, 0]] - counts[idx[:, 1]])
        if normalize_by_count:
            with np.errstate(invalid='ignore', divide='ignore'):
                d = np.where(tot > 0, d / tot, np.nan)
                floor = np.where(tot > 0, floor / tot, np.nan)
        entry = acc.setdefault(row_key, {})
        entry[kind] = _mean(d)
        entry[f'{kind}_floor'] = _mean(floor)
        entry[f'{kind}_n'] = int(d.size)

    rows = []
    for row_key, e in acc.items():
        *group_vals, ct, cycle, cid = row_key
        if 'cross' not in e:
            continue
        within = _mean([e.get('within_a', np.nan), e.get('within_b', np.nan)])
        within_floor = _mean([e.get('within_a_floor', np.nan),
                              e.get('within_b_floor', np.nan)])
        rows.append({
            **dict(zip(keys, group_vals)),
            'cell_type': ct, 'cycle': cycle, 'cell_id': cid,
            'd_cross': e['cross'],
            'd_within_a': e.get('within_a', np.nan),
            'd_within_b': e.get('within_b', np.nan),
            'd_excess': e['cross'] - within,
            'd_cross_count_only': e.get('cross_floor', np.nan),
            # The same statistic computed with a metric that can only see
            # spike counts: at cost 0 the Victor-Purpura distance is exactly
            # |n_a - n_b|, so this is what the excess would be if timing
            # carried nothing at all. It is the number d_excess has to be read
            # against — the two conditions fire at different rates, and a
            # difference in rate alone moves d_excess in the same direction
            # and on the same timescale as a difference in timing would.
            'd_excess_count_only': e.get('cross_floor', np.nan) - within_floor,
            'cost_per_s': _cost_for(ct),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Time-resolved
# ---------------------------------------------------------------------------

def time_resolved_similarity(rr: RepeatResponse, *,
                             contrast_key: str,
                             group_keys: Optional[Sequence[str]] = None,
                             cell_types: Optional[Sequence[str]] = None,
                             n_sections: int = 3,
                             section_s: Optional[float] = None,
                             normalizations: Sequence[str] = ('centered', 'shape'),
                             reliability_method: str = 'pairs',
                             min_reliability: float = 0.1) -> pd.DataFrame:
    """The same similarity, in sections, so recovery has a trajectory.

    Each cycle is cut into sections of ``section_s``, and the cross-condition
    similarity computed inside each. Because the movie repeats, section *k* of
    cycle 1 and section *k* of cycle 2 contain **identical stimulus content**
    — so the difference between them is time since the step and nothing else,
    which is what a within-cycle time course cannot claim.

    ``t_since_movie_s`` is the section's midpoint measured from movie onset
    across both cycles, which is the axis to plot on; ``section`` is its index
    within a cycle, and rows sharing a ``section`` share a stimulus.

    ``n_sections`` divides the cycle evenly, which is the reason to prefer it
    to naming a ``section_s``: the cycle is 14.922 s here, not 15, so asking
    for 5 s sections leaves 4.9 s of every cycle — a third of the movie —
    unanalysed. Pass ``section_s`` explicitly only when the width matters more
    than the coverage; whole sections are taken and the remainder dropped.

    Sections shorter than about 3 s are not worth asking for here: the
    correlation is over ``n_cells x n_bins`` numbers but the effective sample
    size is set by the response's own correlation time, and with three repeats
    a 1 s section gives a reliability estimate too noisy to divide by.
    """
    keys = list(group_keys) if group_keys is not None else \
        [k for k in rr.condition_keys if k != contrast_key]
    types = ['all'] + (list(cell_types) if cell_types is not None
                       else sorted(rr.cells['cell_type'].dropna().unique()))

    cycle_s = float(rr.t_s[-1] + rr.bin_ms / 2000.0)
    if section_s is None:
        section_s = cycle_s / int(n_sections)
    else:
        n_sections = int(np.floor(cycle_s / float(section_s)))
    if n_sections < 1:
        raise ValueError(f'section_s={section_s} is longer than the '
                         f'{cycle_s:.2f} s cycle')

    rows = []
    for group_vals, group_rows in _grouped(rr.epochs, keys):
        levels = sorted(group_rows[contrast_key].unique())
        if len(levels) != 2:
            continue
        idx = {lv: group_rows.index[group_rows[contrast_key] == lv].to_numpy()
               for lv in levels}
        for ct in types:
            cmask = np.flatnonzero(rr.cell_mask(None if ct == 'all' else ct))
            if cmask.size < 2:
                continue
            for cycle in range(rr.n_cycles):
                for sec in range(n_sections):
                    lo, hi = sec * section_s, (sec + 1) * section_s
                    sl = np.flatnonzero((rr.t_s >= lo) & (rr.t_s < hi))
                    trials = {lv: rr.rates[np.ix_(cmask, idx[lv])]
                                  [:, :, cycle][..., sl]
                                  .transpose(1, 0, 2).astype(float)
                              for lv in levels}
                    for how in normalizations:
                        a = _normalize_matrix(trials[levels[0]].mean(0), how)
                        b = _normalize_matrix(trials[levels[1]].mean(0), how)
                        rho = _vec_corr(a, b)
                        rel_a, _ = _reliability(trials[levels[0]], how,
                                                reliability_method)
                        rel_b, _ = _reliability(trials[levels[1]], how,
                                                reliability_method)
                        ok = (np.isfinite(rel_a) and np.isfinite(rel_b)
                              and rel_a > min_reliability
                              and rel_b > min_reliability)
                        rows.append({
                            **dict(zip(keys, group_vals)),
                            'cell_type': ct, 'normalize': how,
                            'cycle': cycle + 1, 'section': sec + 1,
                            't_since_movie_s': cycle * cycle_s + (lo + hi) / 2,
                            'n_cells': int(cmask.size),
                            'rho_cross': rho,
                            'rho_within_a': rel_a, 'rho_within_b': rel_b,
                            'rho_corrected': (rho / np.sqrt(rel_a * rel_b)
                                              if ok else np.nan),
                            'corrected_valid': ok,
                        })
    out = pd.DataFrame(rows)
    out.attrs['contrast_key'] = contrast_key
    out.attrs['group_keys'] = keys
    out.attrs['section_s'] = section_s
    return out


# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------

def plot_population_matrices(rr: RepeatResponse, *, contrast_key: str,
                             group: Optional[Mapping] = None,
                             cell_types: Optional[Sequence[str]] = None,
                             percentile: float = 99.0,
                             scale_by: str = 'all',
                             figsize: Optional[Tuple[float, float]] = None):
    """Panel B: the trial-averaged ``(cell x time)`` matrix, one per history.

    Cells are blocked by type and ordered within type by receptive-field
    position along the canvas x axis, so a band travelling down a block is the
    population following the movie rather than a coincidence of sort order.

    **Every panel shares one colour scale** (``scale_by='all'``), because the
    two histories differ in firing rate by a factor of two in the first
    presentation and that difference is half of what is being shown; rescaling
    each panel to its own range would draw it away. The cost is that a
    low-rate type is dark in every panel, which is why ``scale_by='type'``
    exists — it gives each type block its own scale so the temporal structure
    inside it is visible, and then the panels can no longer be compared for
    rate.
    """
    import matplotlib.pyplot as plt

    types = list(cell_types) if cell_types is not None else \
        sorted(rr.cells['cell_type'].dropna().unique())
    order, boundaries = [], []
    for ct in types:
        rows = rr.cells.index[rr.cells['cell_type'] == ct].to_numpy()
        rows = rows[np.argsort(rr.cells.loc[rows, 'center_x'].to_numpy())]
        order.extend(rows.tolist())
        boundaries.append((ct, len(order)))
    order = np.asarray(order, dtype=int)

    mask = np.ones(len(rr.epochs), dtype=bool)
    if group:
        for k, v in group.items():
            mask &= (rr.epochs[k] == v).to_numpy()
    levels = sorted(rr.epochs.loc[mask, contrast_key].unique())

    panels = {}
    for lv in levels:
        sel = np.flatnonzero(mask & (rr.epochs[contrast_key] == lv).to_numpy())
        for c in range(rr.n_cycles):
            panels[(lv, c)] = rr.rates[np.ix_(order, sel)][:, :, c].mean(1)

    if scale_by == 'type':
        # One scale per type block, applied by rescaling the block in place.
        prev = 0
        for _, edge in boundaries:
            block = np.concatenate([p[prev:edge].ravel() for p in panels.values()])
            top = np.percentile(block, percentile) or 1.0
            for p in panels.values():
                p[prev:edge] = p[prev:edge] / top
            prev = edge
        vmax, label = 1.0, f'rate / this type\'s {percentile:g}th percentile'
    elif scale_by == 'all':
        vmax = np.percentile(
            np.concatenate([p.ravel() for p in panels.values()]), percentile)
        label = 'rate (Hz)'
    else:
        raise ValueError(f"scale_by must be 'all' or 'type'; got {scale_by!r}")

    n_rows, n_cols = len(levels), rr.n_cycles
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, sharex=True,
                             sharey=True,
                             figsize=figsize or (5.0 * n_cols, 3.2 * n_rows))
    centers = [((prev + edge) / 2, ct) for (ct, edge), prev
               in zip(boundaries, [0] + [e for _, e in boundaries[:-1]])]
    for r, lv in enumerate(levels):
        for c in range(rr.n_cycles):
            ax = axes[r][c]
            im = ax.imshow(panels[(lv, c)], aspect='auto', cmap='magma',
                           vmin=0, vmax=vmax, interpolation='nearest',
                           extent=(0, rr.t_s[-1], len(order), 0))
            for _, edge in boundaries[:-1]:
                ax.axhline(edge, color='w', lw=0.6)
            if r == 0:
                ax.set_title(f'presentation {c + 1}')
            if c == 0:
                ax.set_yticks([y for y, _ in centers])
                ax.set_yticklabels([ct for _, ct in centers], fontsize=8)
                ax.set_ylabel(f'{contrast_key} = {lv:g}')
            ax.set_xlabel('movie time (s)')
    fig.colorbar(im, ax=axes, shrink=0.7, label=label)
    return fig


def plot_population_similarity(sim: pd.DataFrame, *,
                               normalize: str = 'centered',
                               cell_types: Optional[Sequence[str]] = None,
                               value: str = 'rho_corrected',
                               ax=None, title: Optional[str] = None):
    """Panels C/D: the similarity per cycle, one line per cell type.

    Groups (images) are individual points and their mean is the line, because
    the group is a different movie and averaging them without showing the
    spread hides whether one image carried the effect.
    """
    import matplotlib.pyplot as plt
    from .style import colors_for_conditions

    df = sim.query('normalize == @normalize')
    types = list(cell_types) if cell_types is not None else \
        [t for t in df['cell_type'].unique()]
    colors = colors_for_conditions(list(types))

    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 3.4))
    for ct in types:
        d = df[df['cell_type'] == ct]
        if d.empty:
            continue
        for _, g in d.groupby([c for c in sim.attrs['group_keys']]):
            ax.plot(g['cycle'], g[value], color=colors[ct], alpha=.3, lw=.9,
                    marker='o', ms=3, mew=0)
        m = d.groupby('cycle')[value].mean()
        ax.plot(m.index, m.to_numpy(), color=colors[ct], lw=2.4, marker='o',
                ms=6, label=f'{ct} (n={int(d["n_cells"].median())})')
    ax.axhline(1.0, color='0.7', lw=0.8, ls=':')
    ax.set_xticks(sorted(df['cycle'].unique()))
    ax.set_xlabel('movie presentation')
    ax.set_ylabel({'rho_corrected': 'noise-corrected cross-condition r',
                   'rho_cross': 'cross-condition population r',
                   'rho_cell': 'mean single-cell r'}.get(value, value))
    ax.set_title(title or f'{value} ({normalize})')
    ax.legend(frameon=False, fontsize=8)
    ax.figure.tight_layout()
    return ax.figure


def plot_excess_distance(dist: pd.DataFrame, *,
                         cell_types: Optional[Sequence[str]] = None,
                         ax=None, title: Optional[str] = None):
    """Panel E: excess spike distance per pathway, first versus second cycle."""
    import matplotlib.pyplot as plt
    from .style import colors_for_conditions

    types = list(cell_types) if cell_types is not None else \
        sorted(dist['cell_type'].unique())
    colors = colors_for_conditions(list(types))
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 3.4))
    for ct in types:
        d = dist[dist['cell_type'] == ct]
        if d.empty:
            continue
        m = d.groupby('cycle')['d_excess'].mean()
        e = d.groupby('cycle')['d_excess'].sem()
        ax.errorbar(m.index, m.to_numpy(), yerr=e.to_numpy(), color=colors[ct],
                    lw=2.2, marker='o', ms=6, capsize=3,
                    label=f'{ct} (n={d["cell_id"].nunique()})')
        f = d.groupby('cycle')['d_excess_count_only'].mean()
        ax.plot(f.index, f.to_numpy(), color=colors[ct], lw=1.0, ls=':')
    ax.axhline(0, color='0.7', lw=0.8)
    ax.set_xticks(sorted(dist['cycle'].unique()))
    ax.set_xlabel('movie presentation')
    ax.set_ylabel('excess VP distance\n(cross − within, per spike)')
    ax.set_title(title or 'spike distance in excess of repeat variability\n'
                          '(dotted: the same statistic, count-only metric)')
    ax.legend(frameon=False, fontsize=8)
    ax.figure.tight_layout()
    return ax.figure


def plot_time_resolved_similarity(tr: pd.DataFrame, *,
                                  normalize: str = 'centered',
                                  cell_types: Optional[Sequence[str]] = None,
                                  value: str = 'rho_corrected',
                                  ax=None, title: Optional[str] = None):
    """The trajectory: similarity against time since movie onset.

    Sections separated by one cycle share their stimulus content, so the
    vertical line at the cycle boundary separates two passes over the *same*
    movie rather than two halves of a longer one.
    """
    import matplotlib.pyplot as plt
    from .style import colors_for_conditions

    df = tr.query('normalize == @normalize')
    types = list(cell_types) if cell_types is not None else \
        list(df['cell_type'].unique())
    colors = colors_for_conditions(list(types))
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.6))
    for ct in types:
        d = df[df['cell_type'] == ct]
        if d.empty:
            continue
        m = d.groupby('t_since_movie_s')[value].agg(['mean', 'sem'])
        ax.errorbar(m.index, m['mean'], yerr=m['sem'], color=colors[ct],
                    lw=2.0, marker='o', ms=5, capsize=2, label=ct)
    edges = sorted(df.loc[df['cycle'] == 2, 't_since_movie_s'].unique())
    if edges:
        ax.axvline(edges[0] - tr.attrs['section_s'] / 2, color='0.6', lw=1.0,
                   ls='--')
    ax.axhline(1.0, color='0.7', lw=0.8, ls=':')
    ax.set_xlabel('time since movie onset (s)')
    ax.set_ylabel({'rho_corrected': 'noise-corrected cross-condition r'}
                  .get(value, value))
    ax.set_title(title or f'{value} in {tr.attrs["section_s"]:.2f} s sections '
                          f'({normalize})')
    ax.legend(frameon=False, fontsize=8)
    ax.figure.tight_layout()
    return ax.figure


# ---------------------------------------------------------------------------
# Eye-movement per-date persistence and cross-date summaries
# ---------------------------------------------------------------------------

_EYE_MOVEMENT_PROC = 'emtraj'


def summarize_eye_movement_cell_type_recovery(
    trajectory: pd.DataFrame,
    *,
    exp_name: Optional[str] = None,
    normalize: str = 'centered',
    metric: str = 'rho_corrected',
    late_points: int = 2,
) -> pd.DataFrame:
    """Reduce images within date and normalize each cell-type trajectory.

    The raw endpoint is already corrected by the population's repeat
    reliability.  For combining dates, the recovery shape is additionally
    scaled within each retina and cell type: the earliest point is 0 and the
    mean of the last ``late_points`` is 1.  This affine scaling leaves an
    exponential time constant unchanged while preventing a retina with a
    larger similarity range from carrying more weight.
    """
    required = {'cell_type', 't_since_movie_s', metric}
    missing = required.difference(trajectory.columns)
    if missing:
        raise KeyError(f'eye-movement trajectory missing: {sorted(missing)}')
    data = trajectory.copy()
    if 'normalize' in data:
        data = data[data['normalize'] == normalize]
    if 'exp_name' not in data:
        if exp_name is None:
            raise ValueError('exp_name is required when trajectory has no exp_name column')
        data['exp_name'] = str(exp_name)
    elif exp_name is not None:
        data['exp_name'] = data['exp_name'].fillna(str(exp_name)).astype(str)
    if data.empty:
        return pd.DataFrame()

    keys = ['exp_name', 'cell_type', 't_since_movie_s']
    grouped = data.groupby(keys, as_index=False, dropna=False)
    summary = grouped[metric].agg(['mean', 'sem', 'count']).reset_index()
    summary = summary.rename(columns={
        'mean': metric, 'sem': f'{metric}_sem', 'count': 'n_observations'})
    if 'n_cells' in data:
        counts = grouped['n_cells'].max().rename(columns={'n_cells': 'n_cells'})
        summary = summary.merge(counts, on=keys, how='left')
    else:
        summary['n_cells'] = np.nan

    pieces = []
    for _, group in summary.groupby(['exp_name', 'cell_type'], sort=False,
                                     dropna=False):
        group = group.sort_values('t_since_movie_s').copy()
        first = float(group[metric].iloc[0])
        late = float(group[metric].tail(max(1, int(late_points))).mean())
        scale = late - first
        if np.isfinite(scale) and abs(scale) > 1e-12:
            group['recovery_fraction'] = (group[metric] - first) / scale
            group['recovery_fraction_sem'] = group[f'{metric}_sem'] / abs(scale)
        else:
            group['recovery_fraction'] = np.nan
            group['recovery_fraction_sem'] = np.nan
        group['recovery_baseline'] = first
        group['recovery_late_mean'] = late
        group['recovery_scale'] = scale
        group['normalization'] = f'first=0; mean(last {max(1, int(late_points))})=1'
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def fit_eye_movement_cell_type_recovery(
    summary: pd.DataFrame,
    *,
    metric: str = 'recovery_fraction',
    skip_first_s: float = 0.0,
    n_boot: int = 250,
    random_seed: Optional[int] = 0,
) -> pd.DataFrame:
    """Fit one descriptive recovery timescale per retina and cell type."""
    from .spatial_recovery import fit_recovery
    import warnings

    required = {'exp_name', 'cell_type', 't_since_movie_s', metric}
    missing = required.difference(summary.columns)
    if missing:
        raise KeyError(f'eye-movement recovery summary missing: {sorted(missing)}')
    rows = []
    for (exp_name, cell_type), group in summary.groupby(
            ['exp_name', 'cell_type'], sort=True, dropna=False):
        base = {
            'exp_name': str(exp_name), 'cell_type': str(cell_type),
            'metric': metric,
            'n_cells': int(np.nanmax(group['n_cells']))
            if 'n_cells' in group and group['n_cells'].notna().any() else 0,
            'n_observations': int(group.get(
                'n_observations', pd.Series(dtype=float)).sum()),
        }
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fit = fit_recovery(
                    group['t_since_movie_s'].to_numpy(),
                    group[metric].to_numpy(), skip_first_s=skip_first_s,
                    n_boot=n_boot, random_seed=random_seed)
            base.update({
                'tau_s': fit['tau_s'], 'tau_ci_low': fit['tau_ci'][0],
                'tau_ci_high': fit['tau_ci'][1],
                'tau_bounded': fit['tau_bounded'], 't50_s': fit['t50_s'],
                't50_ci_low': fit['t50_ci'][0],
                't50_ci_high': fit['t50_ci'][1],
                'r_squared': fit['r_squared'], 'n_points': fit['n_points'],
                'fit_error': '',
            })
        except Exception as exc:
            base.update({
                'tau_s': np.nan, 'tau_ci_low': np.nan, 'tau_ci_high': np.nan,
                'tau_bounded': False, 't50_s': np.nan,
                't50_ci_low': np.nan, 't50_ci_high': np.nan,
                'r_squared': np.nan, 'n_points': 0, 'fit_error': str(exc),
            })
        rows.append(base)
    return pd.DataFrame(rows)


def compare_eye_movement_cell_type_timescales(
    fits: pd.DataFrame,
    *,
    cell_types: Tuple[str, str] = ('OnM', 'OnP'),
) -> pd.DataFrame:
    """Return paired within-retina OnM-versus-OnP timescale differences."""
    a, b = (str(v) for v in cell_types)
    required = {'exp_name', 'cell_type', 'tau_s', 't50_s'}
    missing = required.difference(fits.columns)
    if missing:
        raise KeyError(f'eye-movement recovery fits missing: {sorted(missing)}')
    rows = []
    selected = fits[fits['cell_type'].isin([a, b])]
    for exp_name, group in selected.groupby('exp_name', sort=True):
        by_type = group.drop_duplicates('cell_type').set_index('cell_type')
        if a not in by_type.index or b not in by_type.index:
            continue
        row = {'exp_name': str(exp_name), 'cell_type_a': a, 'cell_type_b': b}
        for metric in ('tau_s', 't50_s'):
            av, bv = float(by_type.at[a, metric]), float(by_type.at[b, metric])
            row[f'{metric}_{a}'] = av
            row[f'{metric}_{b}'] = bv
            row[f'{metric}_diff_{a}_minus_{b}'] = av - bv
            row[f'{metric}_ratio_{a}_over_{b}'] = (
                av / bv if np.isfinite(bv) and bv != 0 else np.nan)
        row[f'n_cells_{a}'] = int(by_type.at[a, 'n_cells'])
        row[f'n_cells_{b}'] = int(by_type.at[b, 'n_cells'])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_eye_movement_cell_type_comparison(
    summary: pd.DataFrame,
    fits: pd.DataFrame,
    *,
    cell_types: Tuple[str, str] = ('OnM', 'OnP'),
    figsize: Tuple[float, float] = (11.0, 4.0),
):
    """Plot one date's normalized trajectory and OnM/OnP fit summaries."""
    import matplotlib.pyplot as plt
    from .style import apply_publication_style, colors_for_celltypes

    types = [str(v) for v in cell_types]
    data = summary[summary['cell_type'].isin(types)].copy()
    fit_data = fits[fits['cell_type'].isin(types)].copy()
    if data.empty:
        raise ValueError('No eye-movement recovery rows match the cell types.')
    apply_publication_style()
    colors = colors_for_celltypes(types)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ct in types:
        d = data[data['cell_type'] == ct].sort_values('t_since_movie_s')
        if d.empty:
            continue
        n_cells = (int(d.n_cells.max())
                   if 'n_cells' in d and d.n_cells.notna().any() else 0)
        axes[0].errorbar(
            d['t_since_movie_s'], d['recovery_fraction'],
            yerr=d.get('recovery_fraction_sem'), fmt='-o', ms=4, capsize=2,
            color=colors[ct], label=f'{ct} (n={n_cells})')
    axes[0].axhline(1, color='0.75', ls='--', lw=0.8)
    axes[0].set(xlabel='time since movie onset (s)',
                ylabel='within-date recovery (first=0, late=1)',
                title='A. normalized recovery trajectory')
    axes[0].legend(fontsize=8)

    x, width = np.arange(len(types), dtype=float), 0.34
    for offset, metric in ((-width / 2, 'tau_s'), (width / 2, 't50_s')):
        vals = [float(fit_data.loc[fit_data.cell_type == ct, metric].iloc[0])
                if (fit_data.cell_type == ct).any() else np.nan for ct in types]
        axes[1].bar(x + offset, vals, width=width, alpha=0.8,
                    color=[colors[ct] for ct in types],
                    hatch='' if metric == 'tau_s' else '//',
                    label=metric.replace('_s', ''))
    axes[1].set_xticks(x, types)
    axes[1].set(ylabel='time (s)', title='B. fitted recovery timescale')
    axes[1].legend(fontsize=8)
    dates = sorted(str(v) for v in data.exp_name.dropna().unique())
    prefix = f'{dates[0]} — ' if len(dates) == 1 else ''
    fig.suptitle(f'{prefix}{types[0]} versus {types[1]}', y=1.03)
    fig.tight_layout()
    return fig, axes


def plot_eye_movement_cell_type_across_dates(
    summary: pd.DataFrame,
    fits: pd.DataFrame,
    *,
    cell_types: Tuple[str, str] = ('OnM', 'OnP'),
    figsize: Tuple[float, float] = (14.0, 4.0),
):
    """Plot equally weighted date trajectories and paired fit timescales."""
    import matplotlib.pyplot as plt
    from .style import apply_publication_style, colors_for_celltypes

    types = [str(v) for v in cell_types]
    data = summary[summary['cell_type'].isin(types)].copy()
    fit_data = fits[fits['cell_type'].isin(types)].copy()
    if data.empty:
        raise ValueError('No saved eye-movement recovery rows match the cell types.')
    apply_publication_style()
    colors = colors_for_celltypes(types)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ct in types:
        d = data[data.cell_type == ct]
        for _, date in d.groupby('exp_name', sort=True):
            date = date.sort_values('t_since_movie_s')
            axes[0].plot(date.t_since_movie_s, date.recovery_fraction,
                         color=colors[ct], alpha=0.22, lw=0.8)
        by_time = d.groupby('t_since_movie_s').recovery_fraction
        mean, sem = by_time.mean(), by_time.sem()
        axes[0].plot(mean.index, mean, '-o', color=colors[ct], lw=2, ms=4,
                     label=f'{ct} ({d.exp_name.nunique()} dates)')
        axes[0].fill_between(mean.index, mean - sem, mean + sem,
                             color=colors[ct], alpha=0.14, linewidth=0)
    axes[0].axhline(1, color='0.75', ls='--', lw=0.8)
    axes[0].set(xlabel='time since movie onset (s)',
                ylabel='within-date recovery (first=0, late=1)',
                title='A. date-normalized trajectories')
    axes[0].legend(fontsize=8)

    for ax, metric, title in zip(
            axes[1:], ('tau_s', 't50_s'),
            ('B. paired tau by date', 'C. paired t50 by date')):
        wide = fit_data.pivot_table(index='exp_name', columns='cell_type',
                                    values=metric, aggfunc='first')
        wide = (wide.dropna(subset=types) if set(types) <= set(wide)
                else wide.iloc[0:0])
        for _, row in wide.iterrows():
            ax.plot([0, 1], [row[types[0]], row[types[1]]], '-o',
                    color='0.65', lw=0.8, ms=3)
        ax.scatter([0, 1], [wide[ct].mean() if ct in wide else np.nan
                            for ct in types], s=55,
                   color=[colors[ct] for ct in types], zorder=4)
        ax.set_xticks([0, 1], types)
        ax.set(ylabel='time (s)', title=f'{title} (n={len(wide)} paired dates)')
    fig.suptitle('EyeMovementTrajectory — cell-type recovery across dates', y=1.03)
    fig.tight_layout()
    return fig, axes


def save_eye_movement_results(
    exp_name: str,
    *,
    similarity: pd.DataFrame,
    cycle_interaction: pd.DataFrame,
    spike_distance: pd.DataFrame,
    trajectory: pd.DataFrame,
    cell_type_recovery: Optional[pd.DataFrame] = None,
    cell_type_fits: Optional[pd.DataFrame] = None,
    cell_type_comparison: Optional[pd.DataFrame] = None,
    response_timescale_by_type: Optional[pd.DataFrame] = None,
    response_timescale_per_cell: Optional[pd.DataFrame] = None,
    extra_analysis: Optional[Mapping[str, object]] = None,
    metadata: Optional[Mapping] = None,
    figures: Optional[Mapping[str, object]] = None,
    output_root=None,
    verbose: bool = True,
) -> Dict:
    """Save one eye-movement date as pickle tables, JSON metadata, and PNGs."""
    from .analysis_results import save_analysis_bundle

    analysis = {
        'similarity': similarity.copy(),
        'cycle_interaction': cycle_interaction.copy(),
        'spike_distance': spike_distance.copy(),
        'trajectory': trajectory.copy(),
        **dict(extra_analysis or {}),
    }
    optional = {
        'cell_type_recovery': cell_type_recovery,
        'cell_type_recovery_fits': cell_type_fits,
        'cell_type_recovery_comparison': cell_type_comparison,
        'response_timescale_by_type': response_timescale_by_type,
        'response_timescale_per_cell': response_timescale_per_cell,
    }
    analysis.update({k: v.copy() if isinstance(v, pd.DataFrame) else v
                     for k, v in optional.items() if v is not None})
    saved_types = sorted({
        str(v) for table in analysis.values() if isinstance(table, pd.DataFrame)
        and 'cell_type' in table for v in table['cell_type'].dropna().unique()
        if str(v) != 'all'
    })
    meta = {
        'analysis_name': 'EyeMovementTrajectoryAlternatingBackground',
        'normalization': {
            'similarity': 'noise-corrected within each date',
            'trajectory': 'noise-corrected within each date',
            'spike_distance': 'excess over within-condition variability, per spike',
            'cross_date_weighting': 'one mean per retina/date',
        },
        'saved_cell_types': saved_types,
        **dict(metadata or {}),
    }
    if cell_type_recovery is not None and not cell_type_recovery.empty:
        typed = cell_type_recovery
        counts = (typed.groupby('cell_type')['n_cells'].max().dropna()
                  .astype(int).to_dict()
                  if {'cell_type', 'n_cells'} <= set(typed) else {})
        meta['cell_type_recovery'] = {
            'cell_types': sorted(str(v) for v in typed.cell_type.dropna().unique()),
            'n_cells_by_type': counts,
            'normalization': 'within retina/type: first=0, late mean=1',
        }
    return save_analysis_bundle(
        _EYE_MOVEMENT_PROC, str(exp_name), analysis,
        metadata=meta, figures=figures, output_root=output_root,
        verbose=verbose)


def load_eye_movement_many(
    exp_names: Optional[Iterable[str]] = None,
    *,
    output_root=None,
) -> Dict[str, pd.DataFrame]:
    """Combine every saved eye-movement table, retaining ``exp_name``."""
    from .analysis_results import load_analysis_many

    bundles = load_analysis_many(
        _EYE_MOVEMENT_PROC, exp_names=exp_names, output_root=output_root)
    names = (
        'similarity', 'cycle_interaction', 'spike_distance', 'trajectory',
        'cell_type_recovery', 'cell_type_recovery_fits',
        'cell_type_recovery_comparison', 'response_timescale_by_type',
        'response_timescale_per_cell', 'similarity_common_timescale')
    out = {}
    for name in names:
        parts = []
        for exp_name, bundle in bundles.items():
            table = bundle['analysis'].get(name)
            if isinstance(table, pd.DataFrame):
                table = table.copy()
                table['exp_name'] = str(exp_name)
                parts.append(table)
        out[name] = (pd.concat(parts, ignore_index=True)
                     if parts else pd.DataFrame())
    return out


def summarize_eye_movement_dates(
    combined: Mapping[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Reduce images/cells within date before comparing biological replicates."""
    out = {}
    sim = combined.get('similarity', pd.DataFrame())
    if not sim.empty:
        selected = sim.query("normalize == 'centered'")
        out['similarity_by_date'] = (selected.groupby(
            ['exp_name', 'cell_type', 'cycle'], as_index=False)
            [['rho_corrected', 'rho_cell', 'rate_a_hz', 'rate_b_hz']].mean())

    cycle = combined.get('cycle_interaction', pd.DataFrame())
    if not cycle.empty:
        selected = cycle.query("normalize == 'centered'")
        out['cycle_by_date'] = (selected.groupby(
            ['exp_name', 'cell_type'], as_index=False)
            [['rho_cycle_corrected', 'rho_delta', 'delta_ratio']].mean())

    dist = combined.get('spike_distance', pd.DataFrame())
    if not dist.empty:
        out['distance_by_date'] = (dist.groupby(
            ['exp_name', 'cell_type', 'cycle'], as_index=False)
            [['d_excess', 'd_excess_count_only']].mean())

    traj = combined.get('trajectory', pd.DataFrame())
    if not traj.empty:
        selected = traj.query("normalize == 'centered'")
        out['trajectory_by_date'] = (selected.groupby(
            ['exp_name', 'cell_type', 't_since_movie_s'], as_index=False)
            [['rho_corrected']].mean())
    return out


def plot_eye_movement_across_dates(
    combined: Mapping[str, pd.DataFrame],
    *,
    cell_types: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (14.0, 3.8),
):
    """Plot date-normalized eye-movement endpoints with dates weighted equally."""
    import matplotlib.pyplot as plt
    from .style import apply_publication_style, colors_for_conditions

    summary = summarize_eye_movement_dates(combined)
    if not summary:
        raise ValueError('No saved eye-movement results are available to plot.')
    available = set()
    for table in summary.values():
        if 'cell_type' in table:
            available.update(str(v) for v in table['cell_type'].dropna().unique())
    types = (list(cell_types) if cell_types is not None
             else sorted(t for t in available if t != 'all'))
    colors = colors_for_conditions(types)
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    def draw(ax, table, x, y, ylabel):
        for ct in types:
            typed = table[table['cell_type'] == ct]
            if typed.empty:
                continue
            for _, date in typed.groupby('exp_name'):
                date = date.sort_values(x)
                ax.plot(date[x], date[y], color=colors[ct], alpha=0.18,
                        linewidth=0.8)
            date_means = typed.groupby(['exp_name', x], as_index=False)[y].mean()
            mean = date_means.groupby(x)[y].mean()
            sem = date_means.groupby(x)[y].sem()
            ax.plot(mean.index, mean, '-o', color=colors[ct], linewidth=2,
                    label=f'{ct} (n={typed.exp_name.nunique()} dates)')
            ax.fill_between(mean.index, mean - sem, mean + sem,
                            color=colors[ct], alpha=0.12, linewidth=0)
        ax.set_xlabel('movie presentation' if x == 'cycle'
                      else 'time since movie onset (s)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)

    sim = summary.get('similarity_by_date', pd.DataFrame())
    dist = summary.get('distance_by_date', pd.DataFrame())
    traj = summary.get('trajectory_by_date', pd.DataFrame())
    if not sim.empty:
        draw(axes[0], sim, 'cycle', 'rho_corrected',
             'noise-corrected cross-history r')
        axes[0].set_title('population similarity')
    else:
        axes[0].set_visible(False)
    if not dist.empty:
        draw(axes[1], dist, 'cycle', 'd_excess',
             'excess distance per spike')
        axes[1].set_title('spike distance')
    else:
        axes[1].set_visible(False)
    if not traj.empty:
        draw(axes[2], traj, 't_since_movie_s', 'rho_corrected',
             'noise-corrected cross-history r')
        axes[2].set_title('time-resolved similarity')
    else:
        axes[2].set_visible(False)
    fig.suptitle('EyeMovementTrajectory — cross-date summary', y=1.03)
    fig.tight_layout()
    return fig, axes


def save_eye_movement_cross_date_summary(
    combined: Mapping[str, pd.DataFrame],
    *,
    metadata: Optional[Mapping] = None,
    figures: Optional[Mapping[str, object]] = None,
    output_root=None,
    verbose: bool = True,
) -> Dict:
    """Save combined tables, per-date reductions, metadata, and pooled plots."""
    from .analysis_results import save_analysis_summary

    date_summary = summarize_eye_movement_dates(combined)
    dates = sorted({
        str(v) for table in combined.values() if isinstance(table, pd.DataFrame)
        and 'exp_name' in table
        for v in table['exp_name'].dropna().unique()
    })
    analysis = {**dict(combined), **date_summary}
    meta = {
        'analysis_name': 'EyeMovementTrajectory cross-date summary',
        'dates': dates,
        'n_dates': len(dates),
        'cell_types': sorted({
            str(v) for table in combined.values()
            if isinstance(table, pd.DataFrame) and 'cell_type' in table
            for v in table['cell_type'].dropna().unique() if str(v) != 'all'}),
        'paired_cell_type_dates': int(combined.get(
            'cell_type_recovery_comparison', pd.DataFrame())
            .get('exp_name', pd.Series(dtype=object)).nunique()),
        'normalization': (
            'similarity/trajectory noise-corrected within date; spike distance '
            'per-spike excess; images/cells reduced within date before date mean'),
        **dict(metadata or {}),
    }
    typed = combined.get('cell_type_recovery', pd.DataFrame())
    if not typed.empty:
        coverage = {}
        for cell_type, group in typed.groupby('cell_type'):
            item = {'n_dates': int(group.exp_name.nunique())}
            if 'n_cells' in group and group.n_cells.notna().any():
                item.update(n_cells_min=int(group.n_cells.min()),
                            n_cells_max=int(group.n_cells.max()))
            coverage[str(cell_type)] = item
        meta['cell_type_coverage'] = coverage
    return save_analysis_summary(
        _EYE_MOVEMENT_PROC, analysis, metadata=meta, figures=figures,
        output_root=output_root, verbose=verbose)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _spike_source(pipeline, *, cell_types=None, cell_ids=None):
    """``(cells DataFrame with RF columns, {cell_id: [per-epoch spike ms]})``.

    The receptive fields come through ``cell_activity_in_window``, which is
    where the protocol-block-to-noise-chunk id match already lives; the spike
    times come straight off the response block, because every window here is
    a different one and re-deriving the match per window is the expensive
    half.
    """
    from .mosaic_overlay import cell_activity_in_window

    ids = list(cell_ids) if cell_ids is not None else None
    types = list(cell_types) if cell_types is not None else None
    cells = cell_activity_in_window(pipeline, 0, (0.0, 1.0),
                                    cell_types=types, cell_ids=ids)
    cells = cells[['cell_id', 'noise_id', 'cell_type', 'center_x', 'center_y',
                   'width', 'height', 'angle']].reset_index(drop=True)
    df = pipeline.resp.df_spike_times.set_index('cell_id')
    spikes = {int(cid): df.at[int(cid), 'spike_times']
              for cid in cells['cell_id'].astype(int)}
    return cells, spikes


def _grouped(epochs: pd.DataFrame, keys: Sequence[str]):
    """Yield ``(values tuple, rows)`` — one entry when ``keys`` is empty."""
    keys = list(keys)
    if not keys:
        yield (), epochs
        return
    for values, rows in epochs.groupby(keys, sort=True):
        yield (values if isinstance(values, tuple) else (values,)), rows


def _label(keys, values) -> str:
    from .protocol_source import condition_label
    return condition_label(list(keys), tuple(values))


def _time_slice(rr: RepeatResponse, window: Optional[Tuple[float, float]]):
    if window is None:
        return slice(None)
    lo, hi = window
    idx = np.flatnonzero((rr.t_s >= lo) & (rr.t_s < hi))
    if idx.size == 0:
        raise ValueError(f'time_slice {window} selects no bins of the '
                         f'0–{rr.t_s[-1]:.2f} s cycle')
    return slice(int(idx[0]), int(idx[-1]) + 1)
