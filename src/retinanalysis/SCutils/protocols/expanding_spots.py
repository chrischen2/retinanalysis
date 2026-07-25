"""ExpandingSpots: spike raster + difference-of-Gaussians area summation.

Python port of ``analyzeExpandingSpots.m`` (coneAdaptation project) for
cell-attached (spiking) recordings. One call does the whole thing: find the
block, load it, detect spikes, count them in the stimulus window, fit the DoG
area-summation model, and draw the figure.

    from retinanalysis.SCutils.protocols import expanding_spots as es

    res = es.analyze_expanding_spots('2019-03-05_G', cell_label='Cell1')
    res.fit['sigma_c'], res.fit['sigma_s']

Model — identical to ``DoG.m`` / ``fitDoG.m``, with the spot *diameter* as the
independent variable and ``r = diameter / 2``::

    R(d) = base + Kc * (1 - exp(-r^2 / (2*sigma_c^2)))
                - Ks * (1 - exp(-r^2 / (2*sigma_s^2)))

Fitting follows the MATLAB in two ways that matter: responses are normalized by
``max|mean response|`` first (which is what makes the MATLAB's amplitude bounds
meaningful), and the fit is over *every epoch* rather than the per-size means,
so sizes with more repeats carry more weight. sigma_c / sigma_s come back in µm
and are unaffected by the normalization.

Spike detection differs from the MATLAB in mechanism but not intent:
``SpikeDetectorNew`` is preceded there by a ``movmedian`` detrend, while
``utils.spike_detector.detector`` high-pass filters at 500 Hz internally
(``spike_detector.py`` line ~80). Both remove slow drift before clustering
peak amplitudes, so no extra detrending is applied here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# MATLAB fitDoG.m: LB = [0, 2, 0, 10, 0]; UB = [3 200 3 1000 1]
DOG_LOWER = (0.0, 2.0, 0.0, 10.0, 0.0)
DOG_UPPER = (3.0, 200.0, 3.0, 1000.0, 1.0)
DOG_P0 = (1.0, 5.0, 3.0, 500.0, 0.0)  # spiking initial guess from the MATLAB


@dataclass
class ExpandingSpotsResult:
    """Everything the analysis computed, so plots can be redrawn without refitting."""
    exp_name: str
    block_id: int
    cell_label: str
    cell_type: str
    spot_sizes: np.ndarray            # unique diameters, ascending (µm)
    epoch_spot_size: np.ndarray       # diameter per epoch (µm)
    epoch_counts: np.ndarray          # spikes in the stim window, per epoch
    mean_counts: np.ndarray           # mean per unique size
    sem_counts: np.ndarray            # SEM per unique size
    scale: float                      # max|mean_counts|, the normalizer
    fit: Dict[str, float]             # Kc, sigma_c, Ks, sigma_s, base, r2 (normalized units)
    spike_times_s: List[np.ndarray]   # per epoch, seconds from epoch start
    pre_s: float
    stim_s: float
    epoch_duration_s: float
    sample_rate: float
    # Gaussian-kernel PSTH, mean over the epochs of each spot size (Hz).
    psth: np.ndarray = None           # (n_spot_sizes, n_bins)
    psth_time_s: np.ndarray = None    # (n_bins,) bin centers
    psth_sigma_ms: float = 10.0
    # Raw traces, kept so plot_example_trace() can draw waveforms without a
    # second (slow) spike-detection pass.
    amp_data: Optional[np.ndarray] = None
    # The figure drawn by analyze_expanding_spots(plot=True), so callers can
    # export or restyle it without redrawing.
    fig: object = None

    @property
    def n_epochs(self) -> int:
        return len(self.epoch_counts)

    def summary(self) -> str:
        f = self.fit
        return (f"{self.exp_name} {self.cell_label} ({self.cell_type}) block {self.block_id}: "
                f"{self.n_epochs} epochs, {len(self.spot_sizes)} spot sizes | "
                f"sigma_c = {f['sigma_c']:.1f} um, sigma_s = {f['sigma_s']:.1f} um, "
                f"Kc/Ks = {f['Kc']:.2f}/{f['Ks']:.2f}, r2 = {f['r2']:.3f}")


def dog_area_summation(spot_diameters, Kc: float, sigma_c: float, Ks: float,
                       sigma_s: float, base: float = 0.0) -> np.ndarray:
    """Difference-of-Gaussians area summation; port of ``DoG.m``.

    ``spot_diameters`` in µm (the protocol's ``currentSpotSize``); sigmas are
    Gaussian SDs of the center / surround in µm.
    """
    r = np.asarray(spot_diameters, dtype=float) / 2.0
    center = Kc * (1.0 - np.exp(-(r ** 2) / (2.0 * sigma_c ** 2)))
    surround = Ks * (1.0 - np.exp(-(r ** 2) / (2.0 * sigma_s ** 2)))
    return base + center - surround


def fit_dog_area_summation(spot_diameters, responses, p0: Sequence[float] = DOG_P0,
                           lower: Sequence[float] = DOG_LOWER,
                           upper: Sequence[float] = DOG_UPPER) -> Dict[str, float]:
    """Least-squares fit of :func:`dog_area_summation`; port of ``fitDoG.m``.

    ``responses`` should already be normalized (see the module docstring) for
    the default amplitude bounds to make sense. Returns Kc, sigma_c, Ks,
    sigma_s, base and the fit r2.
    """
    from scipy.optimize import curve_fit

    x = np.asarray(spot_diameters, dtype=float)
    y = np.asarray(responses, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 5:
        raise ValueError(f'need at least 5 finite points to fit 5 DoG parameters, got {x.size}')

    params, _ = curve_fit(dog_area_summation, x, y, p0=list(p0),
                          bounds=(list(lower), list(upper)), maxfev=20000)
    resid = y - dog_area_summation(x, *params)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else np.nan
    Kc, sigma_c, Ks, sigma_s, base = params
    return {'Kc': float(Kc), 'sigma_c': float(sigma_c), 'Ks': float(Ks),
            'sigma_s': float(sigma_s), 'base': float(base), 'r2': float(r2)}


def find_expanding_spots_block(df_exp: pd.DataFrame, cell_label: Optional[str] = None,
                               technique: str = 'cell-attached') -> Tuple[int, str]:
    """Pick a spiking ExpandingSpots block from an experiment summary frame.

    Returns ``(block_id, cell_label)`` for the longest matching block of
    ``cell_label``, or of the first cell that has one when ``cell_label`` is
    None or ran no such block. Block ids are DB auto-increments that shift on
    repopulate, which is why they are resolved here instead of hardcoded.
    """
    protocol = (df_exp['protocol'] if 'protocol' in df_exp.columns
                else df_exp['protocol_name'])
    match = (protocol.str.contains('ExpandingSpots', na=False)
             & df_exp['recording_technique'].eq(technique))
    if not match.any():
        raise ValueError(f'no {technique} ExpandingSpots block in this experiment')

    if cell_label is not None and not (match & df_exp['cell_label'].eq(cell_label)).any():
        fallback = df_exp.loc[match, 'cell_label'].iloc[0]
        print(f'{cell_label} has no {technique} ExpandingSpots block; using {fallback}.')
        cell_label = fallback
    if cell_label is None:
        cell_label = df_exp.loc[match, 'cell_label'].iloc[0]

    rows = df_exp[match & df_exp['cell_label'].eq(cell_label)]
    # Longest block = most epochs / widest range of spot sizes.
    block_id = int(rows.sort_values('duration_minutes').iloc[-1]['block_id'])
    return block_id, cell_label


def count_spikes_in_window(spike_times_s: Sequence[np.ndarray], pre_s: float,
                           stim_s: float) -> np.ndarray:
    """Spikes per epoch inside the stimulus window, as the MATLAB does:
    ``prePts < t < prePts + stimPts``."""
    return np.array([int(np.sum((np.asarray(st) > pre_s)
                                & (np.asarray(st) < pre_s + stim_s)))
                     for st in spike_times_s])


def mean_psth_by_spot_size(spike_times_s: Sequence[np.ndarray], spot_size: np.ndarray,
                           sizes: np.ndarray, epoch_duration_s: float,
                           psth_sigma_ms: float = 10.0,
                           psth_sample_rate_hz: float = 1000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Mean Gaussian-kernel PSTH per spot size, in Hz.

    Thin wrapper over :func:`retinanalysis.utils.psth.spike_times_to_psth`, which
    is the port of ``spikeTimeToPSTH.m`` / ``gaussFilter1D.m``: a binary spike
    train convolved with a Gaussian of ``psth_sigma_ms`` (kernel spans ±5 sigma
    and integrates to 1), scaled by the sample rate so the result is spikes/s.

    The PSTH is computed at ``psth_sample_rate_hz`` (1 kHz is plenty for a 10 ms
    kernel) rather than the 10-20 kHz amplifier rate, which is what the MATLAB
    passes; the smoothed rate is identical up to the bin width.

    Returns ``(psth, time_s)`` with psth shape ``(len(sizes), n_bins)``.
    """
    from retinanalysis.utils.psth import psth_time_axis, spike_times_to_psth

    dur_ms = epoch_duration_s * 1000.0
    rows = []
    for size in sizes:
        idx = np.flatnonzero(spot_size == size)
        per_epoch = [spike_times_to_psth(np.asarray(spike_times_s[i]) * 1000.0, dur_ms,
                                         psth_sigma_ms, psth_sample_rate_hz)
                     for i in idx]
        rows.append(np.mean(per_epoch, axis=0) if per_epoch else np.zeros(0))
    time_s = psth_time_axis(dur_ms, psth_sample_rate_hz) / 1000.0
    return np.vstack(rows), time_s


def analyze_expanding_spots(exp_name: str, cell_label: Optional[str] = None,
                            block_id: Optional[int] = None,
                            onset_shift_ms: float = 0.0,
                            psth_sigma_ms: float = 10.0,
                            plot: bool = True, verbose: bool = True,
                            detector_kwargs: Optional[dict] = None,
                            fit_kwargs: Optional[dict] = None,
                            df_exp: Optional[pd.DataFrame] = None):
    """Run the whole ExpandingSpots analysis for one cell-attached block.

    Resolves the block (see :func:`find_expanding_spots_block`), loads it,
    detects spikes, counts them per epoch in the stimulus window, fits the DoG
    area-summation model and — with ``plot=True`` — draws the raster and the
    summation curve.

    Parameters
    ----------
    cell_label, block_id
        Give either. ``block_id`` wins; otherwise the longest cell-attached
        ExpandingSpots block for ``cell_label`` (or the first cell that has
        one) is used.
    onset_shift_ms
        Added to the stimulus onset before counting spikes. The frame monitor
        is not yet used for timing, so the true onset is about one frame late;
        pass ``-1000/60`` to reproduce the old one-frame kludge.
    detector_kwargs
        Forwarded to ``utils.spike_detector.detector`` (e.g.
        ``{'min_peak_amplitude': 20}``). Worth setting: with no amplitude floor
        the clustering can call low-amplitude noise spikes and then discard the
        whole epoch, and this function warns when that happens.
    fit_kwargs
        Forwarded to :func:`fit_dog_area_summation` — ``p0``, ``lower``,
        ``upper``. Useful when Kc or Ks saturates at the MATLAB's bound of 3.

    Returns
    -------
    ExpandingSpotsResult
        Carries the counts, the fit, the PSTH, and — when ``plot=True`` — the
        figure it drew in ``result.fig`` (handy for
        ``ra.igor_export.export_figure_to_h5``).
    """
    import retinanalysis as ra

    if df_exp is None:
        df_exp = ra.get_exp_summary(exp_name)
        df_exp['protocol'] = df_exp['protocol_name'].str.split('.protocols.').str[-1]

    if block_id is None:
        block_id, cell_label = find_expanding_spots_block(df_exp, cell_label)
    else:
        rows = df_exp[df_exp['block_id'].eq(block_id)]
        if rows.empty:
            raise ValueError(f'block {block_id} is not in {exp_name}')
        cell_label = rows.iloc[0]['cell_label']

    row = df_exp[df_exp['block_id'].eq(block_id)].iloc[0]
    cell_type = str(row.get('cell_type', '?'))

    sb = ra.StimBlock(exp_name, block_id)
    rb = ra.SCResponseBlock(exp_name, block_id, b_spiking=True,
                            **(detector_kwargs or {}))

    ep = sb.df_epochs
    if 'currentSpotSize' not in ep.columns:
        raise ValueError(f'block {block_id} has no currentSpotSize; is it really ExpandingSpots?')
    spot_size = ep['currentSpotSize'].to_numpy(dtype=float)

    # Protocol timing is in ms and constant within a block.
    pre_s = (float(ep['preTime'].iloc[0]) + onset_shift_ms) / 1000.0
    stim_s = float(ep['stimTime'].iloc[0]) / 1000.0
    sample_rate = float(rb.amp_sample_rate)
    spike_times_s = [np.asarray(st, dtype=float) / sample_rate for st in rb.spike_times]
    epoch_duration_s = rb.amp_data.shape[1] / sample_rate

    counts = count_spikes_in_window(spike_times_s, pre_s, stim_s)

    # A silent epoch is normal for the smallest and most suppressed spots, but a
    # large fraction of them usually means the detector clustered low-amplitude
    # noise as spikes and then threw the whole epoch out via its sigF check. An
    # amplitude floor (the MATLAB's thresholdSpikeFactor) fixes that.
    n_empty = int(np.sum([len(st) == 0 for st in spike_times_s]))
    if verbose and n_empty > len(spike_times_s) / 3:
        print(f'  WARNING: {n_empty}/{len(spike_times_s)} epochs have no detected spikes at all. '
              "If the traces look healthy, retry with e.g. "
              "detector_kwargs={'min_peak_amplitude': 40}.")

    sizes = np.unique(spot_size)
    mean_counts = np.array([counts[spot_size == s].mean() for s in sizes])
    sem_counts = np.array([counts[spot_size == s].std(ddof=0) / max(np.sqrt((spot_size == s).sum()), 1)
                           for s in sizes])

    # Normalize as the MATLAB does, then fit every epoch (not the means).
    scale = float(np.max(np.abs(mean_counts))) or 1.0
    fit = fit_dog_area_summation(spot_size, counts / scale, **(fit_kwargs or {}))

    psth, psth_time_s = mean_psth_by_spot_size(spike_times_s, spot_size, sizes,
                                               epoch_duration_s, psth_sigma_ms)

    result = ExpandingSpotsResult(
        exp_name=exp_name, block_id=block_id, cell_label=str(cell_label),
        cell_type=cell_type, spot_sizes=sizes, epoch_spot_size=spot_size,
        epoch_counts=counts, mean_counts=mean_counts, sem_counts=sem_counts,
        scale=scale, fit=fit, spike_times_s=spike_times_s, pre_s=pre_s,
        stim_s=stim_s, epoch_duration_s=epoch_duration_s, sample_rate=sample_rate,
        psth=psth, psth_time_s=psth_time_s, psth_sigma_ms=psth_sigma_ms,
        amp_data=rb.amp_data)
    if verbose:
        print(result.summary())
    if plot:
        result.fig = plot_expanding_spots(result)
    return result


def plot_expanding_spots(res: ExpandingSpotsResult, figsize: Tuple[float, float] = (7.2, 11.0),
                         raster_color: str = '#222222'):
    """Raster by spot size, PSTH per spot size, and the DoG area-summation fit."""
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    fig, (ax_r, ax_p, ax_s) = plt.subplots(
        3, 1, figsize=figsize,
        gridspec_kw={'height_ratios': [1.6, 1.0, 1.0], 'hspace': 0.34})

    # --- raster: epochs stacked, grouped by spot size ------------------------
    # Alternating neutral bands separate the size groups; a categorical palette
    # here would fight the stimulus-window shading for attention.
    y = 0
    yticks, ylabels, bands = [], [], []
    for band_i, size in enumerate(res.spot_sizes):
        idx = np.flatnonzero(res.epoch_spot_size == size)
        y0 = y
        for i in idx:
            ax_r.eventplot(res.spike_times_s[i], lineoffsets=y, linelengths=0.72,
                           linewidths=0.9, colors=raster_color)
            y += 1
        yticks.append((y0 + y - 1) / 2.0)
        ylabels.append(f'{size:g}')
        if band_i % 2:
            bands.append((y0 - 0.5, y - 0.5))

    for y0, y1 in bands:
        ax_r.axhspan(y0, y1, color='#000000', alpha=0.05, lw=0, zorder=0)
    ax_r.axvspan(res.pre_s, res.pre_s + res.stim_s, color='#F0C000', alpha=0.18, lw=0,
                 zorder=0, label='spot on')
    ax_r.set_ylim(-0.5, max(y - 0.5, 0.5))
    ax_r.invert_yaxis()  # smallest spot at the top
    ax_r.set_yticks(yticks)
    ax_r.set_yticklabels(ylabels)
    ax_r.set_xlim(0, res.epoch_duration_s)
    ax_r.set_xlabel('Time (s)')
    ax_r.set_ylabel('Spot diameter (µm)')
    ax_r.set_title(f'{res.exp_name}  {res.cell_label} ({res.cell_type})  block {res.block_id}')
    ax_r.legend(loc='upper right', frameon=False, fontsize=8)

    # --- PSTH: mean Gaussian-kernel rate per spot size, sharing the time axis --
    if res.psth is not None and res.psth.size:
        colors = style.colors_for_conditions(list(res.spot_sizes))
        for size, row in zip(res.spot_sizes, res.psth):
            ax_p.plot(res.psth_time_s, row, lw=1.3, color=colors[size],
                      label=f'{size:g}')
        ax_p.axvspan(res.pre_s, res.pre_s + res.stim_s, color='#000000', alpha=0.06,
                     lw=0, zorder=0)
        ax_p.set_xlim(ax_r.get_xlim())
        ax_p.set_xlabel('Time (s)')
        ax_p.set_ylabel('Rate (Hz)')
        ax_p.set_title(f'PSTH, {res.psth_sigma_ms:g} ms Gaussian kernel', fontsize=10)
        # Many sizes -> a full legend would swamp the panel; a colorbar-style
        # note plus the raster's y labels already carry the mapping.
        ax_p.legend(frameon=False, fontsize=6.5, ncol=2, title='µm',
                    title_fontsize=7, loc='upper right')

    # --- area summation: per-epoch points, mean +- SEM, DoG fit --------------
    f = res.fit
    ax_s.scatter(res.epoch_spot_size, res.epoch_counts, s=14, color='#9AA0A6',
                 alpha=0.7, lw=0, label='epochs', zorder=2)
    ax_s.errorbar(res.spot_sizes, res.mean_counts, yerr=res.sem_counts, fmt='o',
                  ms=5, color='#0072B2', ecolor='#0072B2', elinewidth=1.2,
                  capsize=3, label='meanSEM', zorder=3)

    grid = np.linspace(0, res.spot_sizes.max() * 1.02, 400)
    curve = dog_area_summation(grid, f['Kc'], f['sigma_c'], f['Ks'], f['sigma_s'],
                               f['base']) * res.scale
    # Artist labels double as Igor wave names via utils.igor_export, so keep them
    # short and stable across cells: the fitted numbers live in the title.
    ax_s.plot(grid, curve, color='#D55E00', lw=2.0, zorder=4, label='DoGfit')
    # Center / surround extents drawn in diameter units (2*sigma) to match the x axis.
    ax_s.axvline(2 * f['sigma_c'], color='#D55E00', ls=':', lw=1.2, alpha=0.8,
                 label='sigma_c')
    ax_s.axvline(2 * f['sigma_s'], color='#666666', ls=':', lw=1.2, alpha=0.8,
                 label='sigma_s')
    ax_s.set_xlabel('Spot diameter (µm)')
    ax_s.set_ylabel('Spikes during spot')
    ax_s.set_title(f"$\\sigma_c$ = {f['sigma_c']:.1f} µm    $\\sigma_s$ = {f['sigma_s']:.1f} µm"
                   f"    (dotted = 2$\\sigma$)    $r^2$ = {f['r2']:.2f}", fontsize=10)
    ax_s.legend(frameon=False, fontsize=8)
    return fig


def plot_example_trace(res: ExpandingSpotsResult, i_epoch: int = 0, amp_data=None,
                       figsize: Tuple[float, float] = (9, 3.2)):
    """One raw trace with detected spikes marked — spike-detection QC.

    Uses ``res.amp_data`` by default; pass ``amp_data`` explicitly to override.
    Without either, only the spike times are drawn.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    sts = res.spike_times_s[i_epoch]
    if amp_data is None:
        amp_data = res.amp_data
    if amp_data is not None:
        raw = np.asarray(amp_data[i_epoch])
        t = np.arange(raw.size) / res.sample_rate
        ax.plot(t, raw, lw=0.6, color='#333333', label='raw')
        idx = np.clip((sts * res.sample_rate).astype(int), 0, raw.size - 1)
        ax.scatter(sts, raw[idx], s=18, color='#D55E00', zorder=3, label='spikes')
        ax.set_ylabel('Amplitude')
    else:
        ax.eventplot(sts, colors='#333333')
    ax.axvspan(res.pre_s, res.pre_s + res.stim_s, color='#F0C000', alpha=0.18, lw=0,
               label='spot on')
    ax.set_xlim(0, res.epoch_duration_s)
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Epoch {i_epoch} — {res.epoch_spot_size[i_epoch]:g} µm spot, '
                 f'{res.epoch_counts[i_epoch]} spikes in window')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    return fig
