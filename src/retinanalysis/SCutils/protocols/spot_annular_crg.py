"""spotWithAnnularContrastReversingGrating: contrast-response at the reversal frequency.

The drifting/reversing sibling of :mod:`spot_annular_grating`, with the same
structure — discover, group, pick the canonical conditions, analyze, store, plot
— so the two notebooks read the same way. Everything that is not specific to the
temporal modulation (light level, R* mapping, Weber prediction, grating site,
canonical conditions, population figures) is imported from that module rather
than copied.

**The stimulus** (``chris-package/.../spotWithAnnularContrastReversingGrating.m``)
is the same annular square-wave grating, but now reversing in time at
``currentTemporalFrequency``. Note the protocol's comment describes a sinusoidal
waveform while ``getGratingFrame`` actually applies ``sign(cos(2*pi*f*t))``, so
the reversal is a **square wave**: the display alternates between two fixed
frames each half cycle.

The two frames are asymmetric. Writing ``Ap = brightBarContrast`` and
``An = |darkBarContrast|``, one frame puts the "bright" bar set at ``-An`` and
the "dark" set at ``+Ap``; the other swaps them. At stimulus onset
``sign(cos(0)) = +1`` selects the *inverted* frame, so the half cycle starting at
onset (called **phase A** here) has the bright-bar set at ``-An``.

**This protocol does not measure the flashed protocol's cancellation null, and
the reason is in the stimulus.** At every instant half the bars sit at ``+Ap``
and half at ``-An``; reversing swaps *which* bars, not the set of intensities on
screen. A spatially symmetric receptive field therefore receives the same drive
in both half cycles, so the bright/dark balance has nothing to null — unlike the
flashed version, where the two bar sets are summed simultaneously and cancel.
What the reversal does modulate is the spatial arrangement, which rectifying
subunits turn into a frequency-doubled (F2) response.

The data bear this out. Across all 17 canonical recordings F2 exceeds F1, and
**F2 grows monotonically with dark-bar contrast in every one** (correlation of
F2 with |darkBarContrast| between +0.92 and +1.00). Neither harmonic dips at an
intermediate contrast, so there is no balance point to extract and no meaningful
comparison against the Weber prediction here.

So the measurement this module reports is a **contrast-response function**: F1
and F2 amplitude versus dark-bar contrast, per cell, temporal frequency and light
level. ``resp_mean`` holds the half-cycle difference (phase A minus phase B),
which is near zero by the symmetry argument above and is kept as a check on it —
a large value means the grating was not centred on the receptive field. The
``crossing_*`` fields exist so records share a schema with the flashed protocol;
they are *not* a cancellation point here and should not be read as one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from retinanalysis.SCutils.protocols import spot_annular_grating as sag
# Shared, protocol-independent pieces — imported so the two analyses cannot drift
# apart on light level, the cone model, or the population figures.
from retinanalysis.SCutils.protocols.spot_annular_grating import (  # noqa: F401
    CANONICAL_CONDITIONS, RSTAR_TABLE, add_condition, apply_rstar_mapping,
    cone_predict_dark_contrast, grating_site, interp_zero_crossing, is_calibrated,
    light_level_rstar, light_setting, read_filter_wheel_ndf, select_canonical,
    weber_curve,
)

PROTOCOL = 'spotWithAnnularContrastReversingGrating'

DEFAULTS = dict(sag.DEFAULTS)
DEFAULTS.update(
    cycles_to_drop=1,   # skip the first reversal cycle (onset transient)
)

# barWidth and temporalFrequency are constant within a block here and both are
# shown in the block table, so they join the config keys.
CONFIG_KEYS = sag.CONFIG_KEYS + ['temporalFrequency', 'barWidth']


# --------------------------------------------------------------------------
# stimulus schematic
# --------------------------------------------------------------------------

def stimulus_frames(aperture_diameter: float, annulus_inner_diameter: float,
                    annulus_outer_diameter: float, bar_width: float,
                    background_intensity: float, spot_intensity: float,
                    bright_bar_contrast: float, dark_bar_contrast: float,
                    extent_um: Optional[float] = None,
                    n_pixels: int = 601) -> Tuple[np.ndarray, np.ndarray, float]:
    """The two half-cycle frames; port of ``precomputeGratingMatrices``.

    Returns ``(phase_a, phase_b, extent_um)``. Phase A is the frame shown at
    stimulus onset, where the bright-bar set sits at ``-|darkBarContrast|``;
    phase B is its reverse. Both are built from the same masks, so they differ
    only by which bar set carries which peak.
    """
    an = abs(dark_bar_contrast)
    # Phase A: bright-bar set at -An, dark-bar set at +Ap.
    phase_a, extent = sag.stimulus_frame(
        aperture_diameter, annulus_inner_diameter, annulus_outer_diameter, bar_width,
        background_intensity, spot_intensity,
        bright_bar_contrast=-an, dark_bar_contrast=bright_bar_contrast,
        grating_polarity=1.0, extent_um=extent_um, n_pixels=n_pixels)
    # Phase B: the reverse.
    phase_b, _ = sag.stimulus_frame(
        aperture_diameter, annulus_inner_diameter, annulus_outer_diameter, bar_width,
        background_intensity, spot_intensity,
        bright_bar_contrast=bright_bar_contrast, dark_bar_contrast=-an,
        grating_polarity=1.0, extent_um=extent_um, n_pixels=n_pixels)
    return phase_a, phase_b, extent


def plot_stimulus_schematic(params: Dict, dark_contrast: Optional[float] = None,
                            figsize: Tuple[float, float] = (10.5, 3.6)):
    """The two half-cycle frames plus the square-wave reversal they alternate on."""
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if dark_contrast is None:
        dc = params.get('darkBarContrast', [-1.0])
        dc = list(dc) if isinstance(dc, (list, tuple, np.ndarray)) else [dc]
        dark_contrast = dc[-1]
    bar = params.get('currentBarWidth', params['barWidth'])
    bright = params.get('currentBrightContrast', params['brightBarContrast'])
    freq = params.get('currentTemporalFrequency', params['temporalFrequency'])
    freq = float(np.atleast_1d(freq)[0])

    phase_a, phase_b, extent = stimulus_frames(
        params['apertureDiameter'], params['annulusInnerDiameter'],
        params['annulusOuterDiameter'], bar, params['backgroundIntensity'],
        params['spotIntensity'], bright, dark_contrast)

    fig, axes = plt.subplots(1, 3, figsize=figsize,
                             gridspec_kw={'width_ratios': [1, 1, 1.45]})
    for ax, frame, name in ((axes[0], phase_a, 'phase A (at onset)'),
                            (axes[1], phase_b, 'phase B')):
        ax.imshow(frame, cmap='gray', vmin=0, vmax=1, origin='lower',
                  extent=[-extent, extent, -extent, extent], interpolation='nearest')
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('µm')
        ax.set_xticks([-extent, 0, extent])
        ax.set_yticks([-extent, 0, extent])
    axes[0].set_ylabel('µm')

    t = np.linspace(0, 2 / freq, 2000)
    ax = axes[2]
    ax.step(t * 1e3, np.where(np.sign(np.cos(2 * np.pi * freq * t)) > 0, 0, 1),
            where='post', color='#0072B2', lw=1.6)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['A', 'B'])
    ax.set_xlabel('Time from stimulus onset (ms)')
    ax.set_title(f'reversal at {freq:g} Hz (square wave)', fontsize=9)

    fig.suptitle(
        f"grating over {grating_site(params['annulusInnerDiameter'])}  |  annulus "
        f"{params['annulusInnerDiameter']:g}-{params['annulusOuterDiameter']:g} µm  |  "
        f"bar {bar:g} µm  |  bg {params['backgroundIntensity']:g}  |  "
        f"bright +{bright:g} / dark {dark_contrast:g}", fontsize=9, y=1.03)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

def find_blocks(exp_names: Optional[Sequence[str]] = None, show: bool = True,
                height: int = 420, verify_fw: bool = False) -> pd.DataFrame:
    """Every single-cell block that ran the reversing-grating protocol.

    Same columns as :func:`spot_annular_grating.find_blocks` plus
    ``temporalFrequency``, which is constant within a block and is a grouping key
    here (it changes the response, unlike bar width which is pooled).
    """
    import retinanalysis as ra
    from retinanalysis.config import schema
    from retinanalysis.SCutils import explore as sc

    blocks = sc.find_blocks(PROTOCOL, show=False)
    if blocks.empty:
        return blocks
    if exp_names is not None:
        blocks = blocks[blocks['exp_name'].isin(exp_names)]

    meta = pd.concat([
        ra.get_exp_summary(exp)[['exp_name', 'block_id', 'cell_label', 'cell_type',
                                 'recording_technique', 'group_label',
                                 'duration_minutes', 'start_time']]
        for exp in sorted(blocks['exp_name'].unique())])

    rows = []
    for bid in blocks['block_id']:
        ep = (schema.Epoch() & f'parent_id={int(bid)}').to_pandas()
        if ep.empty:
            continue
        ids = [int(i) for i in (ep['id'] if 'id' in ep else ep.index)]
        p = (schema.Epoch() & f'id={ids[0]}').fetch1('parameters')
        row = {'block_id': int(bid), 'n_epochs': len(ids)}
        row.update({k: p.get(k, np.nan) for k in CONFIG_KEYS})
        rows.append(row)

    df = pd.DataFrame(rows).merge(blocks[['exp_name', 'block_id']], on='block_id')
    df = df.merge(meta, on=['exp_name', 'block_id'], how='left')

    df['grating_site'] = df['annulusInnerDiameter'].apply(grating_site)
    df['cell_type_short'] = df['cell_type'].astype(str).str.split('\\').str[-1]
    df = df.rename(columns={'NDF': 'filter_wheel_ndf'})
    df['has_filter_wheel'] = df['filter_wheel_ndf'].notna()
    if verify_fw:
        bad = [(r['exp_name'], int(r['block_id']))
               for _, r in df.iterrows()
               if not np.isclose(read_filter_wheel_ndf(r['exp_name'], r['block_id']),
                                 r['filter_wheel_ndf'], equal_nan=True)]
        print(f'filter-wheel verification: {len(df) - len(bad)}/{len(df)} agree with the h5')
        for exp, bid in bad:
            print(f'  MISMATCH {exp} block {bid}')

    rs = [light_level_rstar(n, b)
          for n, b in zip(df['filter_wheel_ndf'], df['backgroundIntensity'])]
    df['rstar'] = [r for r, _ in rs]
    df['light_level'] = [lab for _, lab in rs]
    df['light_setting'] = [light_setting(n, b)
                           for n, b in zip(df['filter_wheel_ndf'], df['backgroundIntensity'])]
    df['rstar_measured'] = [is_calibrated(n, b)
                            for n, b in zip(df['filter_wheel_ndf'], df['backgroundIntensity'])]
    df = df.sort_values(['exp_name', 'cell_label', 'start_time']).reset_index(drop=True)

    if show:
        cols = ['exp_name', 'cell_label', 'cell_type_short', 'onlineAnalysis', 'grating_site',
                'temporalFrequency', 'barWidth', 'filter_wheel_ndf', 'backgroundIntensity',
                'light_level', 'annulusInnerDiameter', 'annulusOuterDiameter',
                'stimTime', 'n_epochs', 'block_id']
        print(f"{len(df)} blocks | {df['exp_name'].nunique()} experiments | "
              f"{df.groupby(['exp_name', 'cell_label']).ngroups} cells")
        missing = df[~df['has_filter_wheel']]
        if len(missing):
            print(f'  WARNING: {len(missing)} block(s) have no background:FilterWheel:NDF '
                  f"-- no light level. Experiments: {', '.join(sorted(missing['exp_name'].unique()))}")
        sc.scroll_table(df[cols], height=height,
                        num_cols=('temporalFrequency', 'barWidth', 'filter_wheel_ndf',
                                  'backgroundIntensity', 'stimTime', 'n_epochs', 'block_id'))
    return df


def group_blocks(df: pd.DataFrame, show: bool = True, height: int = 420,
                 require_filter_wheel: bool = True) -> pd.DataFrame:
    """One row per recording group; like the flashed version plus temporal frequency."""
    from retinanalysis.SCutils import explore as sc

    if require_filter_wheel and 'has_filter_wheel' in df.columns:
        dropped = df[~df['has_filter_wheel']]
        if len(dropped):
            print(f'dropping {len(dropped)} block(s) with no filter-wheel NDF: '
                  f"{', '.join(sorted(dropped['exp_name'].unique()))}")
        df = df[df['has_filter_wheel']]

    keys = ['exp_name', 'cell_label', 'cell_type_short', 'onlineAnalysis', 'grating_site',
            'temporalFrequency', 'filter_wheel_ndf', 'backgroundIntensity']
    g = (df.groupby(keys, dropna=False, sort=False)
           .agg(blocks=('block_id', 'size'), epochs=('n_epochs', 'sum'),
                light_level=('light_level', 'first'), rstar=('rstar', 'first'),
                light_setting=('light_setting', 'first'),
                rstar_measured=('rstar_measured', 'first'),
                bar_widths=('barWidth', lambda s: ', '.join(f'{b:g}' for b in sorted(set(s)))),
                block_ids=('block_id', lambda s: ', '.join(str(int(b)) for b in sorted(s))))
           .reset_index())
    if show:
        print(f'{len(g)} recording groups (experiment x cell x mode x grating site x '
              f'temporal frequency x filter wheel x background)')
        sc.tree_table(g.sort_values(['cell_type_short', 'exp_name', 'cell_label']),
                      levels=['cell_type_short', 'exp_name', 'cell_label'], height=height,
                      num_cols=('blocks', 'epochs', 'temporalFrequency',
                                'filter_wheel_ndf', 'backgroundIntensity', 'rstar'))
    return g


# --------------------------------------------------------------------------
# per-group analysis
# --------------------------------------------------------------------------

@dataclass
class CRGRecord:
    """One (experiment, cell, mode, grating site, temporal frequency, light level).

    Field names match :class:`spot_annular_grating.GratingRecord` wherever the
    quantity is the same, so the population figures are shared.
    """
    exp_name: str
    cell_label: str
    cell_type: str
    online_analysis: str
    grating_site: str
    temporal_frequency: float
    ndf: float
    background_intensity: float
    rstar: float
    light_level: str
    dark_contrasts: np.ndarray
    resp_mean: np.ndarray          # phase A - phase B, the half-cycle difference
    resp_sem: np.ndarray
    resp_n: np.ndarray
    f1_mean: np.ndarray            # amplitude at the reversal frequency
    f2_mean: np.ndarray            # frequency-doubled component
    baseline_mean: float           # 0 by construction: a balanced grating cancels
    baseline_sem: float
    crossing_nearest: float
    crossing_interp: float
    bright_bar_contrast: float
    cone_pred_dark: float
    cone_i0: float
    bar_widths: np.ndarray
    cycles: np.ndarray             # (n_dark_contrasts, n_samples) mean reversal cycle
    cycle_time_ms: np.ndarray
    pre_time_ms: float
    stim_time_ms: float
    n_epochs: int
    block_ids: List[int]
    config: Dict = field(default_factory=dict)
    units: str = ''

    @property
    def key(self) -> str:
        return record_key(self.exp_name, self.cell_label, self.online_analysis,
                          self.grating_site, self.temporal_frequency, self.ndf,
                          self.background_intensity)

    def summary_row(self) -> Dict:
        return {
            'key': self.key, 'exp_name': self.exp_name, 'cell_label': self.cell_label,
            'cell_type': self.cell_type, 'online_analysis': self.online_analysis,
            'grating_site': self.grating_site,
            'temporal_frequency': self.temporal_frequency, 'ndf': self.ndf,
            'background_intensity': self.background_intensity, 'rstar': self.rstar,
            'rstar_measured': is_calibrated(self.ndf, self.background_intensity),
            'light_setting': light_setting(self.ndf, self.background_intensity),
            'light_level': self.light_level, 'baseline_mean': self.baseline_mean,
            'baseline_sem': self.baseline_sem, 'crossing_nearest': self.crossing_nearest,
            'crossing_interp': self.crossing_interp,
            'bright_bar_contrast': self.bright_bar_contrast,
            'cone_pred_dark': self.cone_pred_dark, 'cone_i0': self.cone_i0,
            'n_epochs': self.n_epochs, 'n_contrasts': len(self.dark_contrasts),
            'bar_widths': ','.join(f'{b:g}' for b in self.bar_widths),
            'block_ids': ','.join(str(b) for b in self.block_ids), 'units': self.units,
            'f1_max': float(np.nanmax(self.f1_mean)) if len(self.f1_mean) else np.nan,
            'f2_max': float(np.nanmax(self.f2_mean)) if len(self.f2_mean) else np.nan,
            'aperture_diameter': self.config.get('apertureDiameter', np.nan),
            'annulus_inner': self.config.get('annulusInnerDiameter', np.nan),
            'annulus_outer': self.config.get('annulusOuterDiameter', np.nan),
            'spot_intensity': self.config.get('spotIntensity', np.nan),
        }

    def describe(self) -> str:
        return (f'{self.exp_name} | {self.cell_type} | {self.cell_label} | '
                f'{self.online_analysis} | grating {self.grating_site} | '
                f'{self.temporal_frequency:g} Hz | FW={self.ndf} '
                f'bg={self.background_intensity:.2f} ({self.light_level}) | '
                f'{self.n_epochs} epochs\n'
                f'  dark contrasts   : {np.round(self.dark_contrasts, 3)}\n'
                f'  phase A - B      : {np.round(self.resp_mean, 3)}\n'
                f'  F1 / F2          : {np.round(self.f1_mean, 2)} / {np.round(self.f2_mean, 2)}\n'
                f'  null nearest={self.crossing_nearest:.3f} interp='
                f'{self.crossing_interp:.3f} | cone pred={self.cone_pred_dark:.3f}')


def harmonic_amplitudes(trace: np.ndarray, sample_rate: float,
                        frequency: float) -> Tuple[float, float]:
    """Amplitude at ``frequency`` (F1) and twice it (F2), via a direct projection.

    Uses the whole trace, so with an integer number of cycles this is the same as
    reading the DFT bins, without needing the window length to be a power of two.
    """
    trace = np.asarray(trace, dtype=float)
    if trace.size == 0:
        return np.nan, np.nan
    t = np.arange(trace.size) / sample_rate
    centred = trace - trace.mean()
    out = []
    for f in (frequency, 2 * frequency):
        c = np.mean(centred * np.cos(2 * np.pi * f * t))
        s = np.mean(centred * np.sin(2 * np.pi * f * t))
        out.append(2.0 * np.hypot(c, s))
    return out[0], out[1]


def fold_cycles(trace: np.ndarray, sample_rate: float, frequency: float,
                drop_cycles: int = 0) -> np.ndarray:
    """Average a response over reversal cycles -> one mean cycle."""
    trace = np.asarray(trace, dtype=float)
    period = sample_rate / frequency
    n_cycles = int(trace.size // period)
    if n_cycles <= drop_cycles:
        drop_cycles = 0
    width = int(round(period))
    cycles = [trace[int(round(k * period)):int(round(k * period)) + width]
              for k in range(drop_cycles, n_cycles)]
    cycles = [c for c in cycles if c.size == width]
    if not cycles:
        return np.zeros(width)
    return np.vstack(cycles).mean(axis=0)


def analyze_group(exp_name: str, block_ids: Sequence[int],
                  online_analysis: Optional[str] = None,
                  spike_offset: int = DEFAULTS['spike_offset'],
                  wc_offset: int = DEFAULTS['wc_offset'],
                  smooth_ms: float = DEFAULTS['smooth_ms'],
                  psth_sigma_ms: float = DEFAULTS['psth_sigma_ms'],
                  cone_i0: float = DEFAULTS['cone_i0'],
                  cycles_to_drop: int = DEFAULTS['cycles_to_drop'],
                  detector_kwargs: Optional[dict] = None,
                  drop_epochs: Sequence[int] = (),
                  verbose: bool = True) -> CRGRecord:
    """Reversal-null analysis for one recording group.

    Per epoch: build the response (PSTH for spikes, smoothed current otherwise),
    fold the stimulus window into reversal cycles, and take the difference
    between the two half cycles — phase A (the frame shown at onset, bright-bar
    set at ``-|darkBarContrast|``) minus phase B. That difference is zero when the
    two contrasts balance, so its zero crossing over ``darkBarContrast`` is the
    null, directly comparable to the Weber prediction. F1 and F2 amplitudes are
    computed on the same window.

    ``cycles_to_drop`` skips the first reversal cycle, which carries the onset
    transient rather than a steady-state reversal response.
    """
    import retinanalysis as ra
    from retinanalysis.utils.psth import spike_times_to_psth
    from scipy.ndimage import uniform_filter1d

    dark, bright, bar = [], [], []
    diff, f1, f2, cycles_all = [], [], [], []
    first_params, used_blocks, freq = None, [], None
    trace_rate = None

    for bid in block_ids:
        sb = ra.StimBlock(exp_name, int(bid), verbose=False)
        ep = sb.df_epochs
        p0 = ep['epoch_parameters'].iloc[0]
        if first_params is None:
            first_params = p0
        mode = (online_analysis or p0.get('onlineAnalysis', 'extracellular')).lower()
        spiking = mode == 'extracellular'
        f = float(sag._epoch_param(ep, 'currentTemporalFrequency')[0])
        if not np.isfinite(f):
            f = float(np.atleast_1d(p0['temporalFrequency'])[0])
        freq = f

        rb = ra.SCResponseBlock(exp_name, int(bid), b_spiking=spiking, verbose=False,
                                **(detector_kwargs or {}))
        sr = float(rb.amp_sample_rate)
        pre_pts = int(round(float(p0['preTime']) / 1e3 * sr))
        stim_pts = int(round(float(p0['stimTime']) / 1e3 * sr))
        keep = [i for i in range(len(ep)) if i not in set(drop_epochs)]
        used_blocks.append(int(bid))

        if spiking:
            # PSTH at 1 kHz, then window on the stimulus with the spike latency
            # offset, as in the flashed analysis.
            trace_rate = 1000.0
            lo = int(round((pre_pts + spike_offset) / sr * trace_rate))
            hi = int(round((pre_pts + stim_pts + spike_offset) / sr * trace_rate))
            dur_ms = rb.amp_data.shape[1] / sr * 1e3
            traces = [spike_times_to_psth(np.asarray(rb.spike_times[i], float) / sr * 1e3,
                                          dur_ms, psth_sigma_ms, trace_rate) for i in keep]
            units = 'rate difference (Hz)'
        else:
            trace_rate = sr
            width = max(int(round(smooth_ms / 1e3 * sr)), 1)
            data = uniform_filter1d(np.asarray(rb.amp_data, dtype=float), size=width, axis=1)
            data = data - data[:, :pre_pts].mean(axis=1, keepdims=True)
            sign = -1.0 if mode == 'exc' else 1.0
            lo, hi = pre_pts + wc_offset, pre_pts + stim_pts + wc_offset
            traces = [sign * data[i] for i in keep]
            units = ('excitation difference (pA)' if mode == 'exc'
                     else 'inhibition difference (pA)')

        for tr in traces:
            window = np.asarray(tr)[lo:min(hi, len(tr))]
            cyc = fold_cycles(window, trace_rate, freq, drop_cycles=cycles_to_drop)
            half = cyc.size // 2
            diff.append(float(cyc[:half].mean() - cyc[half:2 * half].mean()))
            a1, a2 = harmonic_amplitudes(window, trace_rate, freq)
            f1.append(a1)
            f2.append(a2)
            cycles_all.append(cyc)

        dark.extend(sag._epoch_param(ep, 'currentDarkContrast')[keep])
        bright.extend(sag._epoch_param(ep, 'currentBrightContrast')[keep])
        bar.extend(sag._epoch_param(ep, 'currentBarWidth')[keep])

    dark = np.asarray(dark); diff = np.asarray(diff)
    f1 = np.asarray(f1); f2 = np.asarray(f2)
    bright = np.asarray(bright); bar = np.asarray(bar)
    width = min(c.size for c in cycles_all)
    cycles_all = np.vstack([c[:width] for c in cycles_all])

    valid = ~np.isnan(dark)
    if not valid.any():
        raise ValueError(f'{exp_name} blocks {list(block_ids)}: no currentDarkContrast values')

    contrasts = np.unique(dark[valid])
    sel = [valid & (dark == c) for c in contrasts]
    resp_mean = np.array([np.nanmean(diff[m]) for m in sel])
    resp_n = np.array([int(m.sum()) for m in sel])
    resp_sem = np.array([np.nanstd(diff[m], ddof=0) / np.sqrt(max(m.sum(), 1)) for m in sel])
    f1_mean = np.array([np.nanmean(f1[m]) for m in sel])
    f2_mean = np.array([np.nanmean(f2[m]) for m in sel])
    cycles = np.vstack([cycles_all[m].mean(axis=0) for m in sel])

    # A balanced grating produces equal half cycles, so the null is at zero.
    crossing_nearest = float(contrasts[int(np.argmin(np.abs(resp_mean)))])
    crossing_interp = interp_zero_crossing(contrasts, resp_mean)

    bright_mode = float(pd.Series(bright[valid]).mode().iloc[0])
    ndf = float(first_params.get('NDF', np.nan))
    bg = float(first_params['backgroundIntensity'])
    rstar, light_label = light_level_rstar(ndf, bg)

    summary = ra.get_exp_summary(exp_name)
    row = summary[summary['block_id'].eq(int(used_blocks[0]))].iloc[0]

    rec = CRGRecord(
        exp_name=exp_name, cell_label=str(row['cell_label']), cell_type=str(row['cell_type']),
        online_analysis=mode, grating_site=grating_site(first_params['annulusInnerDiameter']),
        temporal_frequency=float(freq), ndf=ndf, background_intensity=bg, rstar=rstar,
        light_level=light_label, dark_contrasts=contrasts, resp_mean=resp_mean,
        resp_sem=resp_sem, resp_n=resp_n, f1_mean=f1_mean, f2_mean=f2_mean,
        baseline_mean=0.0, baseline_sem=0.0, crossing_nearest=crossing_nearest,
        crossing_interp=crossing_interp, bright_bar_contrast=bright_mode,
        cone_pred_dark=cone_predict_dark_contrast(rstar, bright_mode, cone_i0),
        cone_i0=cone_i0, bar_widths=np.unique(bar[valid & ~np.isnan(bar)]),
        cycles=cycles, cycle_time_ms=np.arange(width) / trace_rate * 1e3,
        pre_time_ms=float(first_params['preTime']), stim_time_ms=float(first_params['stimTime']),
        n_epochs=int(valid.sum()), block_ids=used_blocks,
        config={k: first_params.get(k) for k in CONFIG_KEYS}, units=units)
    if verbose:
        print(rec.describe())
    return rec


# --------------------------------------------------------------------------
# record store (same layout as the flashed protocol, its own directory)
# --------------------------------------------------------------------------

def store_dir():
    """``<OUTPUT_DIR>/spot_annular_crg``."""
    from pathlib import Path
    from retinanalysis.config.settings import OUTPUT_DIR
    return Path(OUTPUT_DIR) / 'spot_annular_crg'


def record_key(exp_name: str, cell_label: str, online_analysis: str, site: str,
               temporal_frequency: float, ndf: float,
               background_intensity: float) -> str:
    """Recording-group identifier; carries temporal frequency as well."""
    def num(v):
        return ('NaN' if v is None or (isinstance(v, float) and np.isnan(v))
                else f'{v:g}'.replace('.', 'p'))
    return (f'{exp_name}__{cell_label}__{online_analysis}__{site}__'
            f'{num(temporal_frequency)}Hz__FW{num(ndf)}__bg{num(background_intensity)}')


_ARRAY_FIELDS = ('dark_contrasts', 'resp_mean', 'resp_sem', 'resp_n', 'f1_mean',
                 'f2_mean', 'bar_widths', 'cycles', 'cycle_time_ms')


def save_records(records: Sequence[CRGRecord], path=None, verbose: bool = True):
    """Upsert records into ``<store>/records.h5`` and refresh ``summary.csv``."""
    import h5py
    from pathlib import Path

    base = Path(path) if path is not None else store_dir()
    base.mkdir(parents=True, exist_ok=True)
    h5_path = base / 'records.h5'
    with h5py.File(h5_path, 'a') as f:
        for rec in records:
            if rec.key in f:
                del f[rec.key]
                action = 'overwrote'
            else:
                action = 'added'
            g = f.create_group(rec.key)
            for name in _ARRAY_FIELDS:
                g.create_dataset(name, data=np.asarray(getattr(rec, name), dtype=float))
            for k, v in rec.summary_row().items():
                g.attrs[k] = '' if v is None else v
            g.attrs['pre_time_ms'] = rec.pre_time_ms
            g.attrs['stim_time_ms'] = rec.stim_time_ms
            for k, v in (rec.config or {}).items():
                if v is not None and not isinstance(v, (list, tuple, dict)):
                    g.attrs[f'cfg_{k}'] = v
            if verbose:
                print(f'  {action} {rec.key}')
    summary = load_summary(path=base)
    summary.to_csv(base / 'summary.csv', index=False)
    if verbose:
        print(f'{len(records)} record(s) saved -> {h5_path} ({len(summary)} rows total)')
    return h5_path


def load_summary(path=None) -> pd.DataFrame:
    """Scalar fields for every stored record."""
    import h5py
    from pathlib import Path

    base = Path(path) if path is not None else store_dir()
    h5_path = base / 'records.h5'
    if not h5_path.exists():
        return pd.DataFrame()
    rows = []
    with h5py.File(h5_path, 'r') as f:
        for key in f:
            a = dict(f[key].attrs)
            rows.append({k: (v.decode() if isinstance(v, bytes) else v) for k, v in a.items()})
    return (pd.DataFrame(rows).sort_values(['cell_type', 'exp_name', 'cell_label'],
                                           ignore_index=True) if rows else pd.DataFrame())


def load_records(keys: Optional[Sequence[str]] = None, path=None) -> Dict[str, Dict]:
    """Full records (arrays + scalars) as ``{key: dict}``."""
    import h5py
    from pathlib import Path

    base = Path(path) if path is not None else store_dir()
    h5_path = base / 'records.h5'
    out: Dict[str, Dict] = {}
    if not h5_path.exists():
        return out
    with h5py.File(h5_path, 'r') as f:
        for key in (keys if keys is not None else list(f)):
            if key not in f:
                continue
            g = f[key]
            rec = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in g.attrs.items()}
            rec.update({name: g[name][()] for name in g})
            out[key] = rec
    return out


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

def plot_group(rec: CRGRecord, figsize: Tuple[float, float] = (7.2, 9.0)):
    """Mean reversal cycle per dark contrast, the half-cycle difference, and F1/F2."""
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    fig, (ax_c, ax_d, ax_h) = plt.subplots(
        3, 1, figsize=figsize, gridspec_kw={'height_ratios': [1.2, 1.0, 0.9], 'hspace': 0.38})
    colors = style.colors_for_conditions(list(rec.dark_contrasts))

    half_ms = rec.cycle_time_ms[len(rec.cycle_time_ms) // 2] if len(rec.cycle_time_ms) else 0
    for c, cyc in zip(rec.dark_contrasts, rec.cycles):
        ax_c.plot(rec.cycle_time_ms, cyc, lw=1.2, color=colors[c], label=f'{c:g}')
    ax_c.axvspan(0, half_ms, color='#000000', alpha=0.06, lw=0, zorder=0)
    ax_c.set_xlabel('Time in reversal cycle (ms)')
    ax_c.set_ylabel('Rate (Hz)' if 'rate' in rec.units else rec.units.replace(' difference', ''))
    ax_c.set_title(f'{rec.exp_name}  {rec.cell_label} ({rec.cell_type})  '
                   f'{rec.online_analysis}\ngrating over {rec.grating_site}  |  '
                   f'{rec.temporal_frequency:g} Hz  |  FW={rec.ndf:g} '
                   f'bg={rec.background_intensity:g} ({rec.light_level})\n'
                   f'shaded = phase A (bright-bar set at the dark peak)', fontsize=8.5)
    ax_c.legend(frameon=False, fontsize=6.5, ncol=2, title='dark contrast', title_fontsize=7)

    # Contrast-response: this is the measurement for a reversing grating.
    ax_d.plot(rec.dark_contrasts, rec.f2_mean, 's-', ms=4, lw=1.5, color='#0072B2',
              label=f'F2 ({2 * rec.temporal_frequency:g} Hz)')
    ax_d.plot(rec.dark_contrasts, rec.f1_mean, 'o-', ms=4, lw=1.3, color='#D55E00',
              label=f'F1 ({rec.temporal_frequency:g} Hz)')
    ax_d.set_xlabel('Dark bar contrast')
    ax_d.set_ylabel('Amplitude')
    ax_d.set_title('contrast-response at the reversal frequency', fontsize=9)
    ax_d.legend(frameon=False, fontsize=7)

    # Symmetry check: reversing swaps which bars carry +Ap and -An, not the set of
    # intensities, so a centred receptive field should see equal half cycles.
    ax_h.errorbar(rec.dark_contrasts, rec.resp_mean, yerr=rec.resp_sem, fmt='o-', ms=4,
                  lw=1.3, color='#666666', ecolor='#666666', capsize=3)
    ax_h.axhline(0.0, color='#000000', ls='--', lw=1.0)
    ax_h.set_xlabel('Dark bar contrast')
    ax_h.set_ylabel(rec.units)
    ax_h.set_title('half-cycle difference (expected ~0; large = grating off-centre)',
                   fontsize=8.5)
    return fig


def plot_crossing_by_light_setting(summary: pd.DataFrame, **kwargs):
    """Reversal null by light setting; shares the flashed protocol's figure."""
    return sag.plot_crossing_by_light_setting(summary, **kwargs)


def plot_weber_comparison(summary: pd.DataFrame, **kwargs):
    """Reversal null vs the Weber prediction; shares the flashed protocol's figure."""
    return sag.plot_weber_comparison(summary, **kwargs)


def plot_condition_examples(records: Optional[Dict[str, Dict]] = None, **kwargs):
    """Example tuning curves per condition x mode, from the CRG store."""
    if records is None:
        records = load_records()
    return sag.plot_condition_examples(records, **kwargs)


def analyze_all(groups: pd.DataFrame, save: bool = True, plot: bool = False,
                on_error: str = 'log', verbose: bool = False,
                skip_existing: bool = False, **kwargs) -> List[CRGRecord]:
    """Run :func:`analyze_group` over every row of :func:`group_blocks` output."""
    records, failures = [], []
    stored = set(load_summary()['key']) if skip_existing and len(load_summary()) else set()
    skipped = 0
    for _, row in groups.iterrows():
        if skip_existing:
            key = record_key(row['exp_name'], row['cell_label'], row['onlineAnalysis'],
                             row['grating_site'], row['temporalFrequency'],
                             row['filter_wheel_ndf'], row['backgroundIntensity'])
            if key in stored:
                skipped += 1
                continue
        try:
            rec = analyze_group(row['exp_name'],
                                [int(b) for b in str(row['block_ids']).split(',')],
                                online_analysis=row['onlineAnalysis'], verbose=verbose, **kwargs)
            records.append(rec)
            if plot:
                plot_group(rec)
        except Exception as e:
            if on_error != 'log':
                raise
            failures.append((row['exp_name'], row['cell_label'], f'{type(e).__name__}: {e}'))
    print(f'analyzed {len(records)}/{len(groups)} groups'
          + (f' ({skipped} already stored, skipped)' if skipped else ''))
    if failures:
        print(f'{len(failures)} failed:')
        for exp, cell, msg in failures[:20]:
            print(f'  {exp} {cell}: {msg[:110]}')
    if save and records:
        save_records(records, verbose=False)
    return records


def plot_contrast_response(records: Optional[Dict[str, Dict]] = None,
                           conditions: Sequence[str] = ('ON-parasol / surround',
                                                        'OFF-parasol / center'),
                           modes: Sequence[str] = ('extracellular', 'exc'),
                           harmonic: str = 'f2_mean',
                           figsize: Tuple[float, float] = (9.5, 6.6)):
    """Population contrast-response: harmonic amplitude vs dark-bar contrast.

    One panel per condition x recording mode, one line per recording, coloured by
    light setting. This is the population figure for the reversing protocol —
    the crossing-vs-Weber figure of the flashed protocol does not apply here (see
    the module docstring).
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if records is None:
        records = load_records()
    rows = list(records.values())
    settings = sorted({str(r['light_setting']) for r in rows})
    colors = style.colors_for_conditions(settings)

    fig, axes = plt.subplots(len(conditions), len(modes), figsize=figsize, squeeze=False)
    for i, cond in enumerate(conditions):
        for j, mode in enumerate(modes):
            ax = axes[i][j]
            pool = [r for r in rows
                    if CANONICAL_CONDITIONS.get(
                        (str(r['cell_type']).split('\\')[-1], r['grating_site'])) == cond
                    and r['online_analysis'] == mode]
            if not pool:
                ax.text(0.5, 0.5, f'no {cond}\n{mode} recordings', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='#888888')
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f'{cond} — {mode}', fontsize=9)
                continue
            seen = set()
            for r in pool:
                lab = str(r['light_setting'])
                ax.plot(r['dark_contrasts'], r[harmonic], '-', lw=1.2, alpha=0.85,
                        color=colors[lab], label=None if lab in seen else lab)
                seen.add(lab)
            ax.set_title(f'{cond} — {mode} (n={len(pool)})', fontsize=9)
            ax.set_xlabel('dark bar contrast')
            ax.set_ylabel(f"{harmonic.replace('_mean', '').upper()} amplitude")
            ax.legend(frameon=False, fontsize=6.5, title='light setting', title_fontsize=7)
    fig.suptitle(f'Contrast-response at the reversal frequency '
                 f"({harmonic.replace('_mean', '').upper()})", fontsize=11, y=1.0)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# per-cell inspection
# --------------------------------------------------------------------------

def list_cells(groups: pd.DataFrame, show: bool = True, height: int = 400) -> pd.DataFrame:
    """Cells available in a group table, with the conditions each was recorded in.

    Use it to find the ``cell_id`` ('<experiment>/<cell label>') for
    :func:`inspect_cell`.
    """
    from retinanalysis.SCutils import explore as _sc
    return _sc.list_cells(groups, show=show, height=height)


def inspect_cell(cell: str, groups: pd.DataFrame, plot: bool = True,
                 show: bool = True, **kwargs):
    """Analyze and plot every recording of one cell, split by condition.

    ``cell`` is '<experiment>/<cell label>', e.g. ``'2026-04-23_E/Cell5'``; a
    bare label works when it is unambiguous. Returns the records.
    """
    from retinanalysis.SCutils import explore as _sc
    return _sc.inspect_cell(groups, cell, analyze=analyze_group,
                            plot=plot_group if plot else None, show=show, **kwargs)
