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
    ALLOWED_BRIGHT_CONTRAST, ALLOWED_FILTER_WHEEL, CANONICAL_CONDITIONS, MIN_BAR_WIDTH,
    MIN_EPOCHS, RIG_MAX_RSTAR, RSTAR_LEVELS, RSTAR_TABLE, add_condition,
    apply_rstar_mapping, center_spot, cone_predict_dark_contrast, grating_site,
    interp_zero_crossing, is_calibrated, light_level_rstar, light_setting, max_rstar,
    read_filter_wheel_ndf, rig_of, round_rstar, select_canonical, weber_curve,
)
# The recording-mode audit and the raw-trace views are protocol-independent —
# they read amp_data and the epoch timing, both of which this protocol has — so
# they are re-exported rather than reimplemented. crg.check_series_resistance is
# the same function the flashed notebook calls in its section 1b. The raw views
# take a `reversal_hz` that marks each grating reversal: a CRGRecord built with
# keep_raw=True supplies it, and a bare (exp_name, block_ids) call passes it.
from retinanalysis.SCutils.protocols.spot_annular_grating import (  # noqa: F401
    load_raw, plot_raw_blocks, plot_raw_epochs,
)
from retinanalysis.SCutils.recording_mode import (  # noqa: F401
    MAX_SERIES_RESISTANCE, check_series_resistance, mode_family,
    read_series_resistance, resolve_recording_mode, stage_ndf_table,
)

PROTOCOL = 'spotWithAnnularContrastReversingGrating'

DEFAULTS = dict(sag.DEFAULTS)
# This reversing analysis still expresses its legacy offsets in amplifier
# samples. Keep those local rather than inheriting the flashed protocol's new
# visible millisecond controls.
DEFAULTS.update(spike_offset=300, wc_offset=100)
DEFAULTS.update(
    cycles_to_drop=1,   # skip the first reversal cycle (onset transient)
)

# barWidth and temporalFrequency are constant within a block here and both are
# shown in the block table, so they join the config keys. barWidth is already in
# the flashed protocol's list, so only add what is missing — a duplicate would
# survive the dict comprehension but makes the list lie about what it holds.
CONFIG_KEYS = sag.CONFIG_KEYS + [k for k in ('temporalFrequency', 'barWidth')
                                 if k not in sag.CONFIG_KEYS]

# Temporal frequencies the analysis runs on. Both reversal rates in this dataset
# (2 and 4 Hz) are kept by default; the constant exists so a frequency can be
# excluded the same way a filter-wheel setting can.
ALLOWED_TEMPORAL_FREQUENCY: Optional[Tuple[float, ...]] = None


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
        epoch_conditions = pd.DataFrame({
            'filterWheelNDF': pd.to_numeric([
                params.get('NDF', np.nan)
                for params in ep['parameters']], errors='coerce'),
            'temporalFrequency': pd.to_numeric([
                params.get('currentTemporalFrequency', np.nan)
                for params in ep['parameters']], errors='coerce'),
            'brightBarContrast': pd.to_numeric([
                params.get('currentBrightContrast', np.nan)
                for params in ep['parameters']], errors='coerce'),
            'barWidth': pd.to_numeric([
                params.get('currentBarWidth', np.nan)
                for params in ep['parameters']], errors='coerce'),
        })
        for column in epoch_conditions:
            if epoch_conditions[column].notna().any():
                continue
            configured = np.atleast_1d(p.get(column, np.nan))
            if len(configured) == 1:
                epoch_conditions[column] = float(configured[0])
        recorded_conditions = (epoch_conditions.dropna(
            subset=['filterWheelNDF', 'temporalFrequency',
                    'brightBarContrast', 'barWidth'])
            .value_counts(sort=False).rename('n_epochs').reset_index())
        for _, condition in recorded_conditions.iterrows():
            row = {'block_id': int(bid),
                   'n_epochs': int(condition.n_epochs)}
            row.update({k: p.get(k, np.nan) for k in CONFIG_KEYS})
            row['NDF'] = float(condition.filterWheelNDF)
            row['temporalFrequency'] = float(condition.temporalFrequency)
            row['brightBarContrast'] = float(condition.brightBarContrast)
            row['barWidth'] = float(condition.barWidth)
            rows.append(row)

    df = pd.DataFrame(rows).merge(blocks[['exp_name', 'block_id']], on='block_id')
    df = df.merge(meta, on=['exp_name', 'block_id'], how='left')

    df['grating_site'] = df['annulusInnerDiameter'].apply(grating_site)
    df['center_spot'] = df['apertureDiameter'].apply(center_spot)
    df['cell_type_short'] = df['cell_type'].astype(str).str.split('\\').str[-1]
    df['rig'] = df['exp_name'].apply(rig_of)
    df = df.rename(columns={
        'NDF': 'filter_wheel_ndf_recorded', 'barWidth': 'bar_width'})
    if verify_fw:
        bad = []
        for _, row in df.iterrows():
            h5_value = read_filter_wheel_ndf(
                row['exp_name'], int(row['block_id']))
            recorded = row['filter_wheel_ndf_recorded']
            if not (np.isclose(h5_value, recorded)
                    or (np.isnan(h5_value) and pd.isna(recorded))):
                bad.append((row['exp_name'], int(row['block_id'])))
        print(f'filter-wheel verification: {len(df) - len(bad)}/{len(df)} agree with the h5')
        for exp, bid in bad:
            print(f'  MISMATCH {exp} block {bid}')

    # Keep fixed filters and the protected numeric FilterWheel reading separate.
    # Embedded Stage FW labels are descriptive text, not wheel measurements.
    requested_settings = df[[
        'exp_name', 'block_id', 'filter_wheel_ndf_recorded']].rename(
            columns={'filter_wheel_ndf_recorded': 'filter_wheel_ndf'})
    settings = ra.read_block_light_settings(requested_settings, verbose=show)
    drop = [column for column in ('exp_name', 'rig', 'n_epochs')
            if column in settings]
    df = (df.rename(columns={'filter_wheel_ndf_recorded': 'filter_wheel_ndf'})
          .merge(settings.drop(columns=drop),
                 on=['block_id', 'filter_wheel_ndf'], how='left',
                 validate='many_to_one'))
    df['has_filter_wheel'] = df['filter_wheel_ndf'].notna()

    maxima = []
    for rig, fixed, wheel in zip(
            df['rig'], df['fixed_ndfs'], df['filter_wheel_ndf']):
        try:
            maxima.append(ra.visual_stimulus_max(rig, fixed, wheel))
        except (KeyError, TypeError, ValueError):
            maxima.append(np.nan)
    df['max_light_level'] = maxima
    df['rstar'] = df['max_light_level'] * df['backgroundIntensity']
    df['light_level'] = [
        f'{round_rstar(value):g}R*' if np.isfinite(value)
        else f'{combo} (?R*)'
        for value, combo in zip(df['rstar'], df['ndf_combination'])]
    df['rstar_level'] = [round_rstar(value) for value in df['rstar']]
    df['light_setting'] = [light_setting(n, b)
                           for n, b in zip(df['filter_wheel_ndf'], df['backgroundIntensity'])]
    df['rstar_measured'] = df['max_light_level'].notna()
    df['max_rstar'] = df['max_light_level']
    df = df.sort_values(['exp_name', 'cell_label', 'start_time']).reset_index(drop=True)

    if show:
        cols = ['exp_name', 'rig', 'cell_label', 'cell_type_short', 'onlineAnalysis',
                'grating_site', 'center_spot', 'temporalFrequency', 'bar_width',
                'brightBarContrast', 'ndf_combination', 'max_light_level',
                'filter_wheel_ndf', 'stage_ndfs',
                'backgroundIntensity', 'light_level', 'annulusInnerDiameter',
                'annulusOuterDiameter', 'stimTime', 'n_epochs', 'block_id']
        print(f"{len(df)} blocks | {df['exp_name'].nunique()} experiments | "
              f"{df.groupby(['exp_name', 'cell_label']).ngroups} cells")
        missing = df[~df['has_filter_wheel']]
        if len(missing):
            print(f'  WARNING: {len(missing)} block(s) have no background:FilterWheel:NDF '
                  f"-- no light level. Experiments: {', '.join(sorted(missing['exp_name'].unique()))}")
        off = df[df['filter_wheel_ndf'].isin(list(ALLOWED_FILTER_WHEEL))
                 & ((df['rstar'] < min(RSTAR_LEVELS)) | (df['rstar'] > max(RSTAR_LEVELS)))]
        if len(off):
            print(f'  WARNING: {len(off)} block(s) fall outside RSTAR_LEVELS '
                  f'({min(RSTAR_LEVELS):g}-{max(RSTAR_LEVELS):g} R*) and are clamped to the '
                  f'nearest end: '
                  + ', '.join(f'{r:.0f}->{lv:g}R*' for r, lv in
                              sorted({(r, lv) for r, lv in
                                      zip(off['rstar'], off['rstar_level'])})))
        sc.scroll_table(df[cols], height=height,
                        num_cols=('temporalFrequency', 'bar_width', 'brightBarContrast',
                                  'filter_wheel_ndf', 'max_light_level',
                                  'backgroundIntensity', 'stimTime', 'n_epochs',
                                  'block_id'))
    return df


def group_blocks(df: pd.DataFrame, show: bool = True, height: int = 420,
                 require_filter_wheel: bool = True,
                 allowed_filter_wheel: Sequence[float] = ALLOWED_FILTER_WHEEL,
                 allowed_bright_contrast: Optional[Sequence[float]]
                 = ALLOWED_BRIGHT_CONTRAST,
                 allowed_temporal_frequency: Optional[Sequence[float]]
                 = ALLOWED_TEMPORAL_FREQUENCY,
                 min_bar_width: Optional[float] = MIN_BAR_WIDTH,
                 min_epochs: Optional[int] = MIN_EPOCHS,
                 separate_bright_contrast: bool = False,
                 collapse_bar_widths: bool = True) -> pd.DataFrame:
    """One row per recording group; like the flashed version plus temporal frequency.

    A group is (experiment, rig, cell, recording mode, grating site, **temporal
    frequency**, fixed-NDF path, filter wheel, background). Temporal frequency is
    the only grouping dimension beyond the flashed-grating contract.

    The filters are the flashed protocol's, applied the same way and reported the
    same way — ``allowed_filter_wheel`` and ``allowed_bright_contrast`` and
    ``min_bar_width`` on blocks, ``min_epochs`` on the pooled group — plus
    ``allowed_temporal_frequency`` for this protocol's extra axis. Any of them
    takes ``None`` to keep everything. ``separate_bright_contrast`` and
    ``collapse_bar_widths`` have the same meaning as in the flashed protocol;
    the analysis notebooks use ``True`` and ``False`` respectively so distinct
    stimuli are never silently pooled.
    """
    from retinanalysis.SCutils import explore as sc

    needed = ['rig', 'bar_width', 'rstar_level', 'center_spot',
              'light_setting', 'filter_wheel_ndf', 'grating_site',
              'temporalFrequency', 'apertureDiameter',
              'annulusInnerDiameter', 'annulusOuterDiameter']
    absent = [c for c in needed if c not in df.columns]
    if absent:
        raise KeyError(
            f'find_blocks() output is missing {absent}. This usually means the block '
            f'table was built by an older version of this module still held in the '
            f'kernel -- restart the kernel and re-run find_blocks().')

    if require_filter_wheel and 'has_filter_wheel' in df.columns:
        dropped = df[~df['has_filter_wheel']]
        if len(dropped):
            print(f'dropping {len(dropped)} block(s) with no filter-wheel NDF: '
                  f"{', '.join(sorted(dropped['exp_name'].unique()))}")
        df = df[df['has_filter_wheel']]
        off_list = df[~df['filter_wheel_ndf'].isin(list(allowed_filter_wheel))]
        if len(off_list):
            print(f'dropping {len(off_list)} block(s) whose filter wheel is not in '
                  f'{list(allowed_filter_wheel)}: NDF '
                  f"{sorted(off_list['filter_wheel_ndf'].dropna().unique().tolist())}")
        df = df[df['filter_wheel_ndf'].isin(list(allowed_filter_wheel))]

    if allowed_bright_contrast is not None and 'brightBarContrast' in df.columns:
        keep = df['brightBarContrast'].isin(list(allowed_bright_contrast))
        if (~keep).any():
            bad = df[~keep]
            print(f'dropping {len(bad)} block(s) whose bright bar contrast is not in '
                  f'{list(allowed_bright_contrast)}: '
                  + ', '.join(f'{c:g}' for c in sorted(bad['brightBarContrast'].unique())))
        df = df[keep]

    if allowed_temporal_frequency is not None and 'temporalFrequency' in df.columns:
        keep = df['temporalFrequency'].isin(list(allowed_temporal_frequency))
        if (~keep).any():
            bad = df[~keep]
            print(f'dropping {len(bad)} block(s) whose temporal frequency is not in '
                  f'{list(allowed_temporal_frequency)} Hz: '
                  + ', '.join(f'{c:g}' for c in sorted(bad['temporalFrequency'].unique())))
        df = df[keep]

    if min_bar_width is not None and 'bar_width' in df.columns:
        keep = df['bar_width'] >= float(min_bar_width)
        if (~keep).any():
            bad = df[~keep]
            print(f'dropping {len(bad)} block(s) with bar width below '
                  f'{float(min_bar_width):g} µm: '
                  + ', '.join(f'{w:g} µm ({n} block{"s" if n > 1 else ""})'
                              for w, n in sorted(bad['bar_width'].value_counts().items()))
                  + f" -- {', '.join(sorted(bad['exp_name'].unique()))}")
        df = df[keep]

    keys = ['exp_name', 'rig', 'cell_label', 'cell_type_short', 'onlineAnalysis',
            'grating_site', 'temporalFrequency', 'filter_wheel_ndf',
            'backgroundIntensity']
    if 'ndf_combination' in df.columns:
        keys.append('ndf_combination')
    df = df.copy()
    if separate_bright_contrast and 'brightBarContrast' in df.columns:
        df['_condition_bright'] = df['brightBarContrast']
        keys.append('_condition_bright')
    if not collapse_bar_widths and 'bar_width' in df.columns:
        df['_condition_bar_width'] = df['bar_width']
        keys.append('_condition_bar_width')
    agg = dict(blocks=('block_id', 'nunique'), epochs=('n_epochs', 'sum'),
               light_setting=('light_setting', 'first'),
               light_level=('light_level', 'first'), rstar=('rstar', 'first'),
               rstar_level=('rstar_level', 'first'),
               rstar_measured=('rstar_measured', 'first'),
               center_spot=('center_spot',
                            lambda s: ', '.join(sorted(set(str(v) for v in s)))),
               aperture=('apertureDiameter', 'first'),
               annulus_inner=('annulusInnerDiameter', 'first'),
               annulus_outer=('annulusOuterDiameter', 'first'),
               block_ids=('block_id', lambda s: ', '.join(
                   str(int(b)) for b in sorted(set(s)))))
    if 'spotIntensity' in df.columns:
        agg['spot_intensity'] = ('spotIntensity', 'first')
    if collapse_bar_widths:
        agg['bar_width'] = ('bar_width',
                            lambda s: ', '.join(f'{value:g}' for value in sorted(set(s))))
    if not separate_bright_contrast:
        agg['bright'] = ('brightBarContrast',
                         lambda s: ', '.join(
                             f'{value:g}' for value in sorted(set(s), reverse=True)))
    if 'stage_ndfs' in df.columns:
        agg['stage_ndfs'] = ('stage_ndfs', lambda s: ' | '.join(sorted({str(v) for v in s})))
    if 'fixed_ndfs' in df.columns:
        agg['fixed_ndfs'] = ('fixed_ndfs', 'first')
    if 'max_light_level' in df.columns:
        agg['max_light_level'] = ('max_light_level', 'first')
    if 'filter_wheel_status' in df.columns:
        agg['filter_wheel_status'] = (
            'filter_wheel_status',
            lambda values: ' | '.join(sorted(set(str(value) for value in values))))
    if 'series_resistance' in df.columns:
        agg['rs_mohm'] = ('series_resistance', lambda s: np.round(np.nanmedian(s) / 1e6, 2))
        agg['epochs_high_rs'] = ('n_epochs_high_rs', 'sum')
    g = df.groupby(keys, dropna=False, sort=False).agg(**agg).reset_index()
    if separate_bright_contrast and '_condition_bright' in g.columns:
        g['bright'] = g.pop('_condition_bright')
    if not collapse_bar_widths and '_condition_bar_width' in g.columns:
        g['bar_width'] = g.pop('_condition_bar_width')

    # On the pooled group, for the same reason as the flashed protocol: a cell run
    # twice at one condition reaches a usable count between the two blocks.
    if min_epochs is not None:
        keep = g['epochs'] >= int(min_epochs)
        if (~keep).any():
            thin = g[~keep]
            print(f'dropping {len(thin)} recording group(s) with fewer than '
                  f'{int(min_epochs)} epochs ({int(thin["epochs"].sum())} epochs):')
            for _, r in thin.sort_values('epochs').iterrows():
                print(f"    {r['exp_name']} {r['cell_label']} {r['cell_type_short']} "
                      f"{r['onlineAnalysis']} {r['grating_site']} "
                      f"{r['temporalFrequency']:g}Hz: {int(r['epochs'])} epochs")
        g = g[keep].reset_index(drop=True)

    if show:
        print(f'{len(g)} recording groups (experiment x cell x mode x grating site x '
              f'temporal frequency x NDF combination x background)')
        cols = ['cell_type_short', 'rig', 'exp_name', 'cell_label', 'onlineAnalysis',
                'grating_site', 'center_spot', 'temporalFrequency', 'bar_width', 'bright',
                'ndf_combination', 'filter_wheel_ndf', 'backgroundIntensity',
                'rstar_level', 'blocks', 'epochs']
        cols = [column for column in cols if column in g.columns]
        cols += [c for c in ('rs_mohm', 'epochs_high_rs') if c in g.columns]
        sc.tree_table(g.sort_values(['cell_type_short', 'exp_name', 'cell_label'])[cols],
                      levels=['cell_type_short', 'rig', 'exp_name', 'cell_label'],
                      height=height,
                      num_cols=('temporalFrequency', 'filter_wheel_ndf',
                                'backgroundIntensity', 'rstar_level', 'blocks', 'epochs',
                                'rs_mohm', 'epochs_high_rs'))
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
    # Unprocessed amplifier traces, kept only when analyze_group(keep_raw=True),
    # so plot_raw_blocks / plot_raw_epochs can draw the data underneath the
    # summary without loading the blocks again. Never written to the store.
    raw: Optional[Dict] = None

    @property
    def key(self) -> str:
        return record_key(self.exp_name, self.cell_label, self.online_analysis,
                          self.grating_site, self.temporal_frequency, self.ndf,
                          self.background_intensity,
                          self.config.get('ndf_combination'),
                          self.bright_bar_contrast, self.bar_widths)

    def summary_row(self) -> Dict:
        return {
            'key': self.key, 'exp_name': self.exp_name, 'cell_label': self.cell_label,
            'cell_type': self.cell_type, 'online_analysis': self.online_analysis,
            'grating_site': self.grating_site,
            'temporal_frequency': self.temporal_frequency, 'ndf': self.ndf,
            'background_intensity': self.background_intensity, 'rstar': self.rstar,
            'rstar_level': round_rstar(self.rstar), 'rig': rig_of(self.exp_name),
            'rstar_measured': np.isfinite(
                float(self.config.get('max_light_level', np.nan))),
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
            'fixed_ndfs': self.config.get('fixed_ndfs', ''),
            'ndf_combination': self.config.get('ndf_combination', ''),
            'max_light_level': self.config.get('max_light_level', np.nan),
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


@dataclass
class CellConditionAnalysis:
    """Outputs from :func:`analyze_cell_conditions` for notebook reuse."""

    exp_name: str
    condition_rows: pd.DataFrame
    light_conditions: pd.DataFrame
    records: List[CRGRecord]
    condition_figures: List[object] = field(default_factory=list)
    light_tuning_figure: Optional[object] = None
    max_normalized_light_figure: Optional[object] = None


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
                  filter_wheel_ndf: Optional[float] = None,
                  temporal_frequency: Optional[float] = None,
                  bright_bar_contrast: Optional[float] = None,
                  bar_width: Optional[float] = None,
                  spike_offset: int = DEFAULTS['spike_offset'],
                  wc_offset: int = DEFAULTS['wc_offset'],
                  smooth_ms: float = DEFAULTS['smooth_ms'],
                  psth_sigma_ms: float = DEFAULTS['psth_sigma_ms'],
                  cone_i0: float = DEFAULTS['cone_i0'],
                  cycles_to_drop: int = DEFAULTS['cycles_to_drop'],
                  detector_kwargs: Optional[dict] = None,
                  drop_epochs: Sequence[int] = (),
                  keep_raw: bool = False,
                  verbose: bool = True) -> CRGRecord:
    """Reversal-null analysis for one recording group.

    Per epoch: build the response (PSTH for spikes, smoothed current otherwise),
    fold the stimulus window into reversal cycles, and take the difference
    between the two half cycles — phase A (the frame shown at onset, bright-bar
    set at ``-|darkBarContrast|``) minus phase B. That difference is zero when the
    two contrasts balance, so its zero crossing over ``darkBarContrast`` is the
    null, directly comparable to the Weber prediction. F1 and F2 amplitudes are
    computed on the same window.

    ``filter_wheel_ndf``, ``temporal_frequency``, ``bright_bar_contrast``, and
    ``bar_width`` select epochs when a block interleaves conditions. The wheel
    selector uses only protected epoch ``parameters['NDF']`` readings. Frequency must be supplied when
    more than one is present; mixing it would make the cycle fold and F1/F2
    amplitudes invalid. The condition notebooks pass all three selectors.

    ``cycles_to_drop`` skips the first reversal cycle, which carries the onset
    transient rather than a steady-state reversal response.

    ``keep_raw=True`` attaches the unprocessed amplifier traces to the record as
    ``rec.raw``, so :func:`plot_raw_blocks` and :func:`plot_raw_epochs` can draw
    the data underneath the summary without loading the blocks a second time.
    They are never written to the store.
    """
    import retinanalysis as ra
    from retinanalysis.utils.psth import spike_times_to_psth
    from scipy.ndimage import uniform_filter1d

    dark, bright, bar = [], [], []
    diff, f1, f2, cycles_all = [], [], [], []
    first_params, used_blocks, freq, selected_wheel = None, [], None, None
    trace_rate = None
    raw = {'traces': [], 'spike_times_ms': [], 'dark': [], 'block_id': [],
           'series_resistance': [], 'sample_rate': None, 'exp_name': exp_name,
           'units': ''} if keep_raw else None

    for bid in block_ids:
        sb = ra.StimBlock(exp_name, int(bid), verbose=False)
        ep = sb.df_epochs
        p0 = ep['epoch_parameters'].iloc[0]
        if first_params is None:
            first_params = p0
        mode = (online_analysis or p0.get('onlineAnalysis', 'extracellular')).lower()
        spiking = mode == 'extracellular'
        epoch_frequencies = sag._epoch_param(ep, 'currentTemporalFrequency')
        epoch_bright = sag._epoch_param(ep, 'currentBrightContrast')
        epoch_bar_width = sag._epoch_param(ep, 'currentBarWidth')
        epoch_wheel = sag._epoch_param(ep, 'NDF')
        available_wheels = np.unique(epoch_wheel[np.isfinite(epoch_wheel)])
        if filter_wheel_ndf is None:
            if len(available_wheels) > 1:
                raise ValueError(
                    f'{exp_name} block {bid} interleaves protected FilterWheel '
                    f'readings {available_wheels.tolist()}; pass filter_wheel_ndf')
            block_wheel = (float(available_wheels[0]) if len(available_wheels)
                           else float(p0.get('NDF', np.nan)))
        else:
            block_wheel = float(filter_wheel_ndf)
        if (selected_wheel is not None and np.isfinite(selected_wheel)
                and not np.isclose(selected_wheel, block_wheel)):
            raise ValueError(
                f'{exp_name} blocks {list(block_ids)} span FilterWheel readings '
                f'{selected_wheel:g} and {block_wheel:g}')
        selected_wheel = block_wheel
        available_frequencies = np.unique(
            epoch_frequencies[np.isfinite(epoch_frequencies)])
        if temporal_frequency is None:
            if len(available_frequencies) > 1:
                raise ValueError(
                    f'{exp_name} block {bid} interleaves temporal frequencies '
                    f'{available_frequencies.tolist()}; pass temporal_frequency')
            f = (float(available_frequencies[0])
                 if len(available_frequencies) else
                 float(np.atleast_1d(p0['temporalFrequency'])[0]))
        else:
            f = float(temporal_frequency)
        if freq is not None and not np.isclose(freq, f):
            raise ValueError(
                f'{exp_name} blocks {list(block_ids)} span temporal '
                f'frequencies {freq:g} and {f:g} Hz')
        freq = f

        rb = ra.SCResponseBlock(exp_name, int(bid), b_spiking=spiking, verbose=False,
                                **(detector_kwargs or {}))
        sr = float(rb.amp_sample_rate)
        pre_pts = int(round(float(p0['preTime']) / 1e3 * sr))
        stim_pts = int(round(float(p0['stimTime']) / 1e3 * sr))
        keep = [i for i in range(len(ep))
                if i not in set(drop_epochs)
                and (not np.isfinite(epoch_frequencies[i])
                     or np.isclose(epoch_frequencies[i], f))]
        if np.isfinite(block_wheel):
            keep = [i for i in keep if np.isclose(
                epoch_wheel[i], block_wheel)]
        if bright_bar_contrast is not None:
            keep = [i for i in keep if np.isclose(
                epoch_bright[i], float(bright_bar_contrast))]
        if bar_width is not None:
            keep = [i for i in keep if np.isclose(
                epoch_bar_width[i], float(bar_width))]
        if not keep:
            continue
        used_blocks.append(int(bid))

        # The traces exactly as the amplifier wrote them, before the PSTH or the
        # box-car below — kept only so the raw views can be drawn from this load.
        if keep_raw:
            try:
                rs = read_series_resistance(exp_name, int(bid))
            except Exception:
                rs = np.full(len(ep), np.nan)
            raw['sample_rate'] = sr
            raw['pre_time_ms'] = float(p0['preTime'])
            raw['stim_time_ms'] = float(p0['stimTime'])
            raw['units'] = 'rate (Hz)' if spiking else 'pA'
            amp = np.asarray(rb.amp_data, dtype=float)
            for i in keep:
                raw['traces'].append(amp[i])
                raw['spike_times_ms'].append(
                    np.asarray(rb.spike_times[i], float) / sr * 1e3 if spiking else None)
                raw['block_id'].append(int(bid))
                raw['series_resistance'].append(float(rs[i]) if i < rs.size else np.nan)

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

    if not used_blocks or not cycles_all:
        frequency_text = ('requested frequency' if temporal_frequency is None
                          else f'{float(temporal_frequency):g} Hz')
        raise ValueError(
            f'{exp_name} blocks {list(block_ids)} contain no epochs at '
            f'{frequency_text}')

    dark = np.asarray(dark); diff = np.asarray(diff)
    if keep_raw:
        # Same order as the traces: both are appended per block over `keep`.
        raw['dark'] = dark
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
    bg = float(first_params['backgroundIntensity'])
    selected_ndf = float(selected_wheel)
    light_settings = ra.read_block_light_settings(pd.DataFrame({
        'exp_name': exp_name, 'block_id': used_blocks,
        'filter_wheel_ndf': selected_ndf,
    }), verbose=False)
    combinations = light_settings['ndf_combination'].drop_duplicates()
    if len(combinations) != 1:
        raise ValueError(
            f'{exp_name} blocks {used_blocks} span multiple NDF combinations: '
            f'{combinations.tolist()}')
    fixed_ndfs = light_settings.loc[0, 'fixed_ndfs']
    ndf = float(light_settings.loc[0, 'filter_wheel_ndf'])
    ndf_combination = str(combinations.iloc[0])
    try:
        max_light_level = ra.visual_stimulus_max(
            rig_of(exp_name), fixed_ndfs, ndf)
    except (KeyError, TypeError, ValueError):
        max_light_level = np.nan
    rstar = max_light_level * bg
    light_label = (f'{round_rstar(rstar):g}R*' if np.isfinite(rstar)
                   else f'{ndf_combination} (?R*)')

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
        config={**{k: first_params.get(k) for k in CONFIG_KEYS},
                'NDF': ndf,
                'fixed_ndfs': ', '.join(fixed_ndfs),
                'ndf_combination': ndf_combination,
                'max_light_level': max_light_level,
                'temporalFrequency': float(freq)},
        units=units, raw=raw)
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
               background_intensity: float,
               ndf_combination: Optional[str] = None,
               bright_bar_contrast: Optional[float] = None,
               bar_widths=None) -> str:
    """Recording-group identifier; flashed dimensions plus temporal frequency."""
    def num(v):
        return ('NaN' if v is None or (isinstance(v, float) and np.isnan(v))
                else f'{v:g}'.replace('.', 'p'))
    import re
    light_path = ''
    if ndf_combination:
        safe = re.sub(r'[^A-Za-z0-9.-]+', '-', str(ndf_combination)).strip('-')
        light_path = f'__{safe}'
    bright_path = ''
    if bright_bar_contrast is not None and not pd.isna(bright_bar_contrast):
        bright_path = f'__bright{num(float(bright_bar_contrast))}'
    bar_path = ''
    if bar_widths is not None:
        if isinstance(bar_widths, str):
            values = [value.strip() for value in bar_widths.split(',') if value.strip()]
        elif np.isscalar(bar_widths):
            values = [bar_widths]
        else:
            values = list(np.asarray(bar_widths).ravel())
        normalized = []
        for value in values:
            try:
                normalized.append(num(float(value)))
            except (TypeError, ValueError):
                normalized.append(
                    re.sub(r'[^A-Za-z0-9-]+', '-', str(value)).strip('-'))
        if normalized:
            bar_path = f'__bar{"-".join(normalized)}'
    return (f'{exp_name}__{cell_label}__{online_analysis}__{site}__'
            f'{num(temporal_frequency)}Hz__FW{num(ndf)}'
            f'__bg{num(background_intensity)}{light_path}'
            f'{bright_path}{bar_path}')


_ARRAY_FIELDS = ('dark_contrasts', 'resp_mean', 'resp_sem', 'resp_n', 'f1_mean',
                 'f2_mean', 'bar_widths', 'cycles', 'cycle_time_ms')


def group_keys(groups: pd.DataFrame) -> List[str]:
    """The :func:`record_key` each row of a group table would be stored under.

    Reads ``onlineAnalysis``, which :func:`check_series_resistance` **overwrites**
    with the mode the amplifier resolved — run that first, or a relabelled
    recording produces a different key here than its record was saved under.
    """
    return [record_key(r['exp_name'], r['cell_label'], r['onlineAnalysis'],
                       r['grating_site'], r['temporalFrequency'],
                       r['filter_wheel_ndf'], r['backgroundIntensity'],
                       r.get('ndf_combination'), r.get('bright'),
                       r.get('bar_width'))
            for _, r in groups.iterrows()]


def prune_records(keep, path=None, dry_run: bool = False,
                  verbose: bool = True) -> List[str]:
    """Delete stored records no longer in the analysis set.

    The CRG store upserts and never deletes, so a record outlives the reason it
    was made — see :func:`spot_annular_grating.prune_records`, of which this is
    the temporal-frequency-aware twin. ``keep`` is a group table or a list of
    keys; ``dry_run=True`` reports without touching the file; an empty keep set
    raises rather than emptying the store.
    """
    import h5py
    from pathlib import Path

    keep_keys = set(group_keys(keep) if isinstance(keep, pd.DataFrame) else keep)
    if not keep_keys:
        raise ValueError(
            'prune_records() refuses an empty keep set — that would delete every '
            'stored record. Pass the current group table or an explicit key list.')

    base = Path(path) if path is not None else store_dir()
    h5_path, csv_path = base / 'records.h5', base / 'summary.csv'
    if not h5_path.exists():
        if verbose:
            print('no store to prune')
        return []
    stored = load_summary(path=base)
    if stored.empty:
        return []
    orphans = stored[~stored['key'].isin(keep_keys)]
    if orphans.empty:
        if verbose:
            print(f'nothing to prune: all {len(stored)} stored record(s) are in the '
                  f'current set of {len(keep_keys)}')
        return []
    if verbose:
        print(f'{"would remove" if dry_run else "removing"} {len(orphans)} stored '
              f'record(s) no longer in the analysis set '
              f'({len(stored)} stored, {len(keep_keys)} current):')
        for _, r in orphans.iterrows():
            print(f"    {r['exp_name']} {r['cell_label']} "
                  f"{r.get('online_analysis', '')} {r.get('grating_site', '')} "
                  f"{r.get('temporal_frequency', '?')}Hz — "
                  f"{int(r.get('n_epochs', 0))} epochs")
    if dry_run:
        return list(orphans['key'])

    removed = []
    with h5py.File(h5_path, 'a') as f:
        for key in orphans['key']:
            if key in f:
                del f[key]
                removed.append(key)
    load_summary(path=base).to_csv(csv_path, index=False)
    if verbose:
        print(f'{len(removed)} record(s) removed')
    return removed


def refresh_rstar(summary: pd.DataFrame) -> pd.DataFrame:
    """Refresh light calibration with the flashed protocol's exact contract."""
    return sag.refresh_rstar(summary)


def save_records(records: Sequence[CRGRecord], path=None, verbose: bool = True,
                 prune_to=None):
    """Upsert records into ``<store>/records.h5`` and refresh ``summary.csv``.

    ``prune_to`` additionally deletes stored records outside that analysis set
    (:func:`prune_records`). Leave it None when saving incrementally, as
    :func:`analyze_all` does per record.
    """
    import h5py
    from pathlib import Path

    base = Path(path) if path is not None else store_dir()
    base.mkdir(parents=True, exist_ok=True)
    h5_path = base / 'records.h5'
    with h5py.File(h5_path, 'a') as f:
        for rec in records:
            legacy_keys = {
                record_key(rec.exp_name, rec.cell_label, rec.online_analysis,
                           rec.grating_site, rec.temporal_frequency, rec.ndf,
                           rec.background_intensity),
                record_key(rec.exp_name, rec.cell_label, rec.online_analysis,
                           rec.grating_site, rec.temporal_frequency, rec.ndf,
                           rec.background_intensity,
                           rec.config.get('ndf_combination')),
            }
            legacy_keys.discard(rec.key)
            removed_legacy = [key for key in legacy_keys if key in f]
            for key in removed_legacy:
                del f[key]
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
                for key in removed_legacy:
                    print(f'  replaced legacy unsplit record {key}')
                print(f'  {action} {rec.key}')

    if prune_to is not None:
        prune_records(prune_to, path=base, verbose=verbose)

    summary = load_summary(path=base)
    summary.to_csv(base / 'summary.csv', index=False)
    if verbose:
        print(f'{len(records)} record(s) saved -> {h5_path} ({len(summary)} rows total)')
    return h5_path


def load_summary(path=None, rstar: bool = True) -> pd.DataFrame:
    """Scalar fields for every stored record — the population-analysis index.

    The light level is recomputed on read (:func:`refresh_rstar`), so records
    analyzed before the rig calibration existed still come back with an R*.
    Pass ``rstar=False`` to see exactly what is on disk.
    """
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
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(['cell_type', 'exp_name', 'cell_label'],
                                         ignore_index=True)
    # Derived from a field records already store, so backfill rather than
    # requiring re-analysis.
    if 'center_spot' not in out.columns and 'aperture_diameter' in out.columns:
        out['center_spot'] = out['aperture_diameter'].apply(center_spot)
    return refresh_rstar(out) if rstar else out


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

def plot_group(rec: CRGRecord, figsize: Tuple[float, float] = (7.2, 9.0),
               cycle_cmap: str = 'plasma'):
    """Raster, reversal-cycle averages, harmonic amplitudes, and symmetry check.

    Cycle averages use a perceptually ordered ``plasma`` palette by default;
    pass another Matplotlib colormap name through ``cycle_cmap`` to change it.
    Extracellular records built with ``keep_raw=True`` gain a spike raster with
    rows grouped and colored by dark-bar contrast. Whole-cell and stored records
    without raw spikes retain the original three-panel layout.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    raw = rec.raw or {}
    raw_spikes = raw.get('spike_times_ms', [])
    has_raster = (rec.online_analysis == 'extracellular'
                  and len(raw_spikes) > 0
                  and all(spikes is not None for spikes in raw_spikes)
                  and len(raw.get('dark', [])) == len(raw_spikes))
    if has_raster:
        fig = plt.figure(figsize=(figsize[0], figsize[1] + 2.0))
        grid = fig.add_gridspec(
            4, 1, height_ratios=[0.9, 1.2, 1.0, 0.9], hspace=0.42)
        ax_r = fig.add_subplot(grid[0])
        ax_c = fig.add_subplot(grid[1])
        ax_d = fig.add_subplot(grid[2])
        ax_h = fig.add_subplot(grid[3])
    else:
        fig, (ax_c, ax_d, ax_h) = plt.subplots(
            3, 1, figsize=figsize,
            gridspec_kw={
                'height_ratios': [1.2, 1.0, 0.9], 'hspace': 0.38})
        ax_r = None
    colors = style.colors_for_conditions(
        list(rec.dark_contrasts), cmap_name=cycle_cmap, lo=0.12, hi=0.88)

    if ax_r is not None:
        raw_dark = np.asarray(raw['dark'], dtype=float)
        row_position = 0
        tick_positions, tick_labels = [], []
        for contrast in rec.dark_contrasts:
            epoch_indices = np.flatnonzero(np.isclose(raw_dark, contrast))
            if not len(epoch_indices):
                continue
            start = row_position
            for epoch_index in epoch_indices:
                ax_r.eventplot(
                    np.asarray(raw_spikes[epoch_index], dtype=float),
                    lineoffsets=row_position, linelengths=0.75,
                    linewidths=0.7, colors=colors[contrast])
                row_position += 1
            tick_positions.append((start + row_position - 1) / 2.0)
            tick_labels.append(f'{contrast:g}')
            ax_r.axhline(
                row_position - 0.5, color='#DDDDDD', lw=0.6, zorder=0)
        stimulus_start = float(rec.pre_time_ms)
        stimulus_stop = stimulus_start + float(rec.stim_time_ms)
        ax_r.axvspan(
            stimulus_start, stimulus_stop,
            color='#000000', alpha=0.05, lw=0, zorder=0)
        half_cycle_ms = 1000.0 / (2.0 * float(rec.temporal_frequency))
        for boundary in np.arange(
                stimulus_start, stimulus_stop + half_cycle_ms * 0.5,
                half_cycle_ms):
            ax_r.axvline(
                boundary, color='#888888', lw=0.55, alpha=0.7, zorder=1)
        ax_r.set_ylim(max(row_position - 0.5, 0.5), -0.5)
        ax_r.set_yticks(tick_positions, tick_labels, fontsize=7)
        ax_r.set_ylabel('dark contrast', fontsize=8)
        ax_r.set_title(
            'Spike raster — epochs grouped by dark-bar contrast; '
            'vertical lines mark reversals', fontsize=8.5)
        ax_r.tick_params(axis='x', labelbottom=False)
        trace_duration_ms = (
            np.asarray(raw['traces'][0]).size
            / float(raw['sample_rate']) * 1000.0)
        ax_r.set_xlim(0.0, trace_duration_ms)

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


# The harmonic is the measurement for a reversing grating, so the overlay plots
# it rather than the half-cycle difference, and there is nothing to subtract: an
# amplitude is already a modulation depth, zero when the cell does not follow
# the grating. Otherwise this is the flashed protocol's figure.

def tuning_overlay(records: Sequence, harmonic: str = 'f2_mean', **kwargs) -> pd.DataFrame:
    """Long-form contrast-response table for several recordings; see
    :func:`spot_annular_grating.tuning_overlay`.

    ``harmonic`` is 'f2_mean' (the frequency-doubled response this protocol is
    about) or 'f1_mean'. Pass ``value='resp_mean', subtract_baseline=True`` to
    overlay the half-cycle difference and its null instead.
    """
    kwargs.setdefault('subtract_baseline', False)
    return sag.tuning_overlay(records, value=harmonic, **kwargs)


def plot_tuning_overlay(records: Sequence, harmonic: str = 'f2_mean', **kwargs):
    """Several recordings' contrast-response curves overlaid, raw and normalized.

    The reversing twin of :func:`spot_annular_grating.plot_tuning_overlay`: one
    curve per recording, colored by light level and labelled with its temporal
    frequency, in recorded units on the left and normalized to each curve's own
    amplitude at its most negative dark contrast on the right. No crossing
    marker — an F2 curve rises with contrast rather than returning to zero, so
    the null lives in ``resp_mean``, not here.
    """
    kwargs.setdefault('subtract_baseline', False)
    kwargs.setdefault('value_label', f'{harmonic.split("_")[0].upper()} amplitude')
    # 'rate difference (Hz)' is the unit of resp_mean, the half-cycle difference;
    # the harmonic amplitude it is plotted beside is just Hz.
    kwargs.setdefault('units_label', lambda u: u.replace(' difference', ''))
    return sag.plot_tuning_overlay(records, value=harmonic, **kwargs)


def plot_max_normalized_light_overlay(
        records: Sequence, harmonic: str = 'f2_mean',
        labels: Optional[Sequence[str]] = None,
        figsize: Tuple[float, float] = (6.4, 4.8),
        title: Optional[str] = None):
    """Overlay CRG curves after normalizing each to its own maximum response.

    This is the light-level comparison used by Section 3. Unlike
    :func:`plot_tuning_overlay`, whose normalized panel uses the deepest shared
    dark contrast as its reference, this figure divides each complete harmonic
    curve by ``max(abs(curve))``. Colors encode light level; line style keeps
    separate conditions visible when frequency, bar width, or another
    parameter also differs. The function returns ``None`` unless at least two
    light levels are present.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    long = tuning_overlay(records, harmonic=harmonic)
    if long.empty:
        print('no CRG response curves to normalize')
        return None

    level_table = long[['rstar', 'light_level']].drop_duplicates()
    if len(level_table) < 2:
        return None

    finite_levels = sorted(long.loc[np.isfinite(long['rstar']), 'rstar'].unique())
    colors = style.colors_for_conditions(finite_levels)
    missing_levels = sorted(
        long.loc[~np.isfinite(long['rstar']), 'light_level'].unique())
    missing_colors = style.colors_for_conditions(missing_levels)
    line_styles = ('-', '--', '-.', ':')

    fig, ax = plt.subplots(figsize=figsize)
    plotted = 0
    for position, sub in long.groupby('position', sort=True):
        sub = sub.sort_values('dark_contrast').copy()
        maximum = float(np.nanmax(np.abs(sub['rel'])))
        if not np.isfinite(maximum) or maximum <= 0:
            continue
        sub['max_norm'] = sub['rel'] / maximum
        row = sub.iloc[0]
        color = (colors[float(row.rstar)] if np.isfinite(row.rstar)
                 else missing_colors[str(row.light_level)])
        label = (str(labels[int(position)]) if labels is not None
                 and int(position) < len(labels)
                 else (f'{row.light_level} · {row.temporal_frequency:g} Hz'))
        ax.plot(sub['dark_contrast'], sub['max_norm'], marker='o', ms=4,
                lw=1.7, ls=line_styles[int(position) % len(line_styles)],
                color=color, label=label)
        plotted += 1

    if not plotted:
        plt.close(fig)
        print('no CRG response curve had a finite, non-zero maximum')
        return None
    ax.axhline(0.0, color='#666666', ls='--', lw=1.0, zorder=1)
    ax.set_xlabel('dark bar contrast')
    ax.set_ylabel(f'{harmonic.split("_")[0].upper()} response / maximum response')
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=7, title='light level', title_fontsize=7)
    fig.suptitle(title or 'Maximum-normalized CRG response by light level',
                 fontsize=10)
    fig.tight_layout()
    return fig


def analyze_cell_conditions(
        date_index: int, cell_label: str, online_analysis: str,
        site: str = 'center', collapse_bar_widths: bool = False,
        max_series_resistance: Optional[float] = MAX_SERIES_RESISTANCE,
        spike_offset: int = DEFAULTS['spike_offset'],
        wc_offset: int = DEFAULTS['wc_offset'],
        keep_raw: bool = True, plot: bool = True, show: bool = True,
        verbose: bool = True,
        detector_kwargs: Optional[dict] = None) -> CellConditionAnalysis:
    """Analyze every recorded condition for one cell without silent filtering.

    This is the contrast-reversing counterpart of
    :func:`spot_annular_grating.analyze_cell_conditions`. Fixed-NDF path,
    FilterWheel, background, bright contrast, bar width, and temporal frequency
    remain separate. Temporal frequency is the only additional condition field.
    """
    from retinanalysis.SCutils import explore as sc
    if show:
        from IPython.display import display

    protocol_index = sc.find_blocks(PROTOCOL, show=False)
    protocol_dates = sorted(protocol_index.exp_name.dropna().unique())
    if not 1 <= int(date_index) <= len(protocol_dates):
        raise ValueError(
            f'date_index {date_index} is outside 1-{len(protocol_dates)}')
    exp_name = protocol_dates[int(date_index) - 1]

    date_blocks = find_blocks(exp_names=[exp_name], show=False)
    date_blocks = check_series_resistance(
        date_blocks, max_series_resistance=max_series_resistance, show=False)
    date_blocks = date_blocks[date_blocks.grating_site.eq(site)].copy()
    cell_blocks = date_blocks[
        date_blocks.cell_label.eq(cell_label)
        & date_blocks.onlineAnalysis.eq(online_analysis)
    ].copy()
    if cell_blocks.empty:
        available_columns = [
            'cell_label', 'cell_type_short', 'onlineAnalysis']
        available = (date_blocks[available_columns].drop_duplicates()
                     .sort_values(['cell_label', 'onlineAnalysis']))
        if show:
            display(available)
        raise ValueError(
            f'No {site} CRG blocks for {exp_name} {cell_label} '
            f'{online_analysis}')

    alerts = (
        ('bright-bar contrasts', 'brightBarContrast', ''),
        ('bar widths', 'bar_width', ' µm'),
        ('temporal frequencies', 'temporalFrequency', ' Hz'),
    )
    for label, column, unit in alerts:
        values = sorted(cell_blocks[column].dropna().astype(float).unique())
        if show and len(values) > 1:
            action = ''
            if column == 'bar_width':
                action = ('; COLLAPSING them by request' if collapse_bar_widths
                          else '; keeping them separate')
            print(f'ALERT: multiple {label} were recorded: {values}{unit}'
                  f'{action}.')

    condition_rows = group_blocks(
        cell_blocks, show=False, require_filter_wheel=False,
        allowed_bright_contrast=None, allowed_temporal_frequency=None,
        min_bar_width=None, min_epochs=None,
        separate_bright_contrast=True,
        collapse_bar_widths=collapse_bar_widths)
    if condition_rows.empty:
        raise ValueError(f'No conditions found for {exp_name} {cell_label}')
    condition_rows = condition_rows.sort_values(
        ['ndf_combination', 'backgroundIntensity', 'bright', 'bar_width',
         'temporalFrequency'], na_position='last').reset_index(drop=True)
    condition_rows.insert(
        0, 'condition_index', np.arange(1, len(condition_rows) + 1))

    view_columns = [
        'condition_index', 'ndf_combination', 'filter_wheel_ndf',
        'filter_wheel_status',
        'backgroundIntensity', 'bright', 'bar_width', 'temporalFrequency',
        'max_light_level', 'rstar', 'blocks', 'epochs', 'block_ids']
    view_columns = [column for column in view_columns
                    if column in condition_rows]
    if show and len(condition_rows) > 1:
        print(f'ALERT: {len(condition_rows)} unique conditions found; '
              'each will be analyzed separately:')
        display(condition_rows[view_columns])

    light_columns = [
        'ndf_combination', 'filter_wheel_ndf', 'backgroundIntensity']
    light_conditions = condition_rows[light_columns].drop_duplicates()
    records: List[CRGRecord] = []
    condition_figures: List[object] = []
    for _, row in condition_rows.iterrows():
        block_ids = [int(value) for value in str(row.block_ids).split(',')]
        if show:
            metadata = pd.DataFrame([{
                'condition_index': int(row.condition_index),
                'cell_label': row.cell_label,
                'cell_type': row.cell_type_short,
                'onlineAnalysis': row.onlineAnalysis,
                'spotIntensity': row.spot_intensity,
                'brightBarContrast': row.bright,
                'barWidth_um': row.bar_width,
                'temporalFrequency_Hz': row.temporalFrequency,
                'apertureDiameter_um': row.aperture,
                'annulusInnerDiameter_um': row.annulus_inner,
                'annulusOuterDiameter_um': row.annulus_outer,
                'background_Rstar_per_s': row.rstar,
                'block_ids': row.block_ids,
                'epochs': row.epochs,
            }])
            print(f'Condition {int(row.condition_index)}/'
                  f'{len(condition_rows)} metadata:')
            display(metadata)
        record = analyze_group(
            exp_name, block_ids, online_analysis=online_analysis,
            filter_wheel_ndf=float(row.filter_wheel_ndf),
            temporal_frequency=float(row.temporalFrequency),
            bright_bar_contrast=float(row.bright),
            bar_width=(None if collapse_bar_widths
                       else float(row.bar_width)),
            spike_offset=spike_offset, wc_offset=wc_offset,
            detector_kwargs=detector_kwargs, keep_raw=keep_raw,
            verbose=verbose)
        records.append(record)
        if plot:
            condition_figures.append(plot_group(record))

    light_tuning_figure = None
    max_normalized_light_figure = None
    if plot and len(records) > 1:
        labels = [
            (f'C{int(row.condition_index)}: {row.temporalFrequency:g} Hz, '
             f'NDF={row.ndf_combination}, FW={row.filter_wheel_ndf}, '
             f'background={row.backgroundIntensity}, bright={row.bright}, '
             f'width={row.bar_width} µm')
            for _, row in condition_rows.iterrows()]
        light_tuning_figure = plot_tuning_overlay(
            records, labels=labels,
            title=f'{exp_name} {cell_label}: F2 curves by condition')
        if len(light_conditions) > 1:
            max_normalized_light_figure = plot_max_normalized_light_overlay(
                records, labels=labels,
                title=(f'{exp_name} {cell_label}: maximum-normalized F2 '
                       'curves by light level'))

    if show:
        print(f'Analyzed {len(records)} separate condition(s) for '
              f'{exp_name} {cell_label}.')
    return CellConditionAnalysis(
        exp_name=exp_name, condition_rows=condition_rows,
        light_conditions=light_conditions, records=records,
        condition_figures=condition_figures,
        light_tuning_figure=light_tuning_figure,
        max_normalized_light_figure=max_normalized_light_figure)


def analyze_all(groups: pd.DataFrame, save: bool = True, plot: bool = False,
                on_error: str = 'log', verbose: bool = False,
                skip_existing: bool = False, prune: bool = False, path=None,
                **kwargs) -> List[CRGRecord]:
    """Run :func:`analyze_group` over every row of :func:`group_blocks` output.

    ``prune=True`` deletes stored records outside ``groups`` once the batch is
    done, against ``groups`` rather than against what just succeeded — a group
    that failed this run keeps whatever it had.
    """
    records, failures = [], []
    stored_summary = load_summary(path=path)
    stored = (set(stored_summary['key'])
              if skip_existing and len(stored_summary) else set())
    skipped = 0
    for _, row in groups.iterrows():
        if skip_existing:
            key = record_key(row['exp_name'], row['cell_label'], row['onlineAnalysis'],
                             row['grating_site'], row['temporalFrequency'],
                             row['filter_wheel_ndf'], row['backgroundIntensity'],
                             row.get('ndf_combination'), row.get('bright'),
                             row.get('bar_width'))
            if key in stored:
                skipped += 1
                continue
        try:
            bright_selector = row.get('bright')
            bar_selector = row.get('bar_width')
            if isinstance(bright_selector, str) and ',' in bright_selector:
                bright_selector = None
            if isinstance(bar_selector, str) and ',' in bar_selector:
                bar_selector = None
            rec = analyze_group(row['exp_name'],
                                [int(b) for b in str(row['block_ids']).split(',')],
                                online_analysis=row['onlineAnalysis'],
                                filter_wheel_ndf=row['filter_wheel_ndf'],
                                temporal_frequency=row['temporalFrequency'],
                                bright_bar_contrast=bright_selector,
                                bar_width=bar_selector,
                                verbose=verbose, **kwargs)
            records.append(rec)
            if save:
                # Save as we go: a batch this long should survive an
                # interruption, and with skip_existing it can then resume.
                save_records([rec], path=path, verbose=False)
            if plot:
                plot_group(rec)
        except Exception as e:
            if on_error != 'log':
                raise
            failures.append((row['exp_name'], row['cell_label'], f'{type(e).__name__}: {e}'))

    if prune and save and len(groups):
        prune_records(groups, path=path)

    print(f'analyzed {len(records)}/{len(groups)} groups'
          + (f' ({skipped} already stored, skipped)' if skipped else ''))
    if failures:
        print(f'{len(failures)} failed:')
        for exp, cell, msg in failures[:20]:
            print(f'  {exp} {cell}: {msg[:110]}')
    return records


def population_contrast_response(summary: pd.DataFrame,
                                 records: Optional[Dict[str, Dict]] = None,
                                 harmonic: str = 'f2', normalize: bool = True,
                                 min_contrasts: int = 3) -> pd.DataFrame:
    """Mean contrast-response curve per light level, pooled over cells.

    The CRG twin of :func:`spot_annular_grating.population_tuning`. What is
    averaged is the harmonic amplitude against dark-bar contrast — ``f2`` by
    default, since F2 exceeds F1 in every canonical recording here and is the
    frequency-doubled signature of rectifying subunits. There is no baseline to
    subtract: an amplitude at the reversal frequency is already a modulation
    depth, zero when the cell does not follow the grating.

    ``normalize`` divides each cell by its own peak amplitude, for the same
    reason as the flashed protocol — cells differ enormously in absolute
    response, so a raw mean is the loudest cell plus noise. It is a positive
    scalar, so the shape of every curve is untouched.

    Groups on temporal frequency as well as light level, since the two
    frequencies are different stimuli and F1/F2 are measured at them.
    """
    df = add_condition(summary) if 'condition' not in summary.columns else summary.copy()
    if df.empty:
        return pd.DataFrame()
    if records is None:
        records = load_records(list(df['key']))

    field = {'f1': 'f1_mean', 'f2': 'f2_mean', 'diff': 'resp_mean'}.get(harmonic, harmonic)
    rows = []
    for _, r in df.iterrows():
        rec = records.get(r['key'])
        if rec is None or field not in rec:
            continue
        contrasts = np.asarray(rec['dark_contrasts'], dtype=float)
        values = np.asarray(rec[field], dtype=float)
        if contrasts.size < min_contrasts or values.size != contrasts.size:
            continue
        if normalize:
            peak = np.nanmax(np.abs(values))
            if not np.isfinite(peak) or peak == 0:
                continue
            values = values / peak
        for c, v in zip(contrasts, values):
            rows.append({'condition': r['condition'],
                         'online_analysis': r.get('online_analysis', ''),
                         'temporal_frequency': r.get('temporal_frequency', np.nan),
                         'rstar_level': r.get('rstar_level', np.nan),
                         'units': 'normalized' if normalize else r.get('units', ''),
                         'cell': f"{r['exp_name']}/{r['cell_label']}",
                         'dark_contrast': round(float(c), 4), 'value': float(v)})
    long = pd.DataFrame(rows)
    if long.empty:
        return long

    keys = ['condition', 'online_analysis', 'temporal_frequency', 'units',
            'rstar_level', 'dark_contrast']
    per_cell = long.groupby(keys + ['cell'], dropna=False)['value'].mean().reset_index()
    return (per_cell.groupby(keys, dropna=False)['value']
            .agg(mean='mean',
                 sem=lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan,
                 n_cells='size')
            .reset_index().sort_values(keys).reset_index(drop=True))


def plot_population_contrast_response(summary: pd.DataFrame,
                                      records: Optional[Dict[str, Dict]] = None,
                                      harmonic: str = 'f2', normalize: bool = True,
                                      min_cells: int = 1,
                                      figsize: Tuple[float, float] = (10.0, 4.6),
                                      **kwargs):
    """Population contrast-response curves, one line per light level, overlaid.

    One panel per (condition, temporal frequency); within a panel each light
    level is its own curve of mean harmonic amplitude against dark-bar contrast,
    shaded with the SEM across cells. Light level is ordered, so the curves use
    the house sequential ramp (``cividis``, dim to bright) with the mapping built
    once across every level, exactly as in the flashed notebook.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    tuning = population_contrast_response(summary, records=records, harmonic=harmonic,
                                          normalize=normalize, **kwargs)
    if tuning.empty:
        print('no records with a contrast-response curve to plot')
        return None

    levels = sorted(tuning['rstar_level'].dropna().unique())
    colors = style.colors_for_conditions(levels)
    panels = (tuning[['condition', 'temporal_frequency']].drop_duplicates()
              .sort_values(['condition', 'temporal_frequency']).values.tolist())
    fig, axes = plt.subplots(1, max(len(panels), 1), figsize=figsize, squeeze=False)
    for ax, (cond, freq) in zip(axes[0], panels):
        panel = tuning[tuning['condition'].eq(cond)
                       & tuning['temporal_frequency'].eq(freq)]
        for lvl in levels:
            sub = panel[panel['rstar_level'].eq(lvl)].sort_values('dark_contrast')
            if sub.empty or int(sub['n_cells'].max()) < min_cells:
                continue
            n = int(sub['n_cells'].max())
            ax.fill_between(sub['dark_contrast'], sub['mean'] - sub['sem'].fillna(0),
                            sub['mean'] + sub['sem'].fillna(0),
                            color=colors[lvl], alpha=0.16, lw=0, zorder=2)
            ax.plot(sub['dark_contrast'], sub['mean'], 'o-', ms=4, lw=1.8,
                    color=colors[lvl], zorder=3, label=f'{lvl:g} R* (n={n})')
        ax.set_title(f'{cond} — {freq:g} Hz', fontsize=9)
        ax.set_xlabel('dark bar contrast')
        ax.legend(frameon=False, fontsize=7, title='light level', title_fontsize=7)
    units = tuning['units'].iloc[0]
    axes[0][0].set_ylabel(f'{harmonic.upper()} amplitude\n({units})')
    fig.suptitle(f'Population {harmonic.upper()} contrast response by light level',
                 fontsize=11)
    fig.tight_layout()
    return fig


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


def animate_stimulus(params: Dict, dark_contrast: Optional[float] = None,
                     duration_s: float = 1.5, fps: int = 30,
                     figsize: Tuple[float, float] = (4.2, 4.4), embed: bool = True):
    """A short movie of the reversing grating, for the notebook.

    Steps between the two half-cycle frames on the protocol's own schedule —
    ``sign(cos(2*pi*f*t))`` — so what you see is the square-wave reversal the
    cell saw, at ``currentTemporalFrequency``.

    Returns an ``IPython.display.HTML`` video by default (set ``embed=False``
    to get the ``FuncAnimation`` itself, e.g. to save a file with
    ``anim.save('crg.mp4')``).
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from retinanalysis.utils import style

    style.apply_publication_style()
    if dark_contrast is None:
        dc = params.get('darkBarContrast', [-1.0])
        dc = list(dc) if isinstance(dc, (list, tuple, np.ndarray)) else [dc]
        dark_contrast = dc[-1]
    bar = params.get('currentBarWidth', params['barWidth'])
    bright = params.get('currentBrightContrast', params['brightBarContrast'])
    freq = float(np.atleast_1d(params.get('currentTemporalFrequency',
                                          params['temporalFrequency']))[0])

    phase_a, phase_b, extent = stimulus_frames(
        params['apertureDiameter'], params['annulusInnerDiameter'],
        params['annulusOuterDiameter'], bar, params['backgroundIntensity'],
        params['spotIntensity'], bright, dark_contrast)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(phase_a, cmap='gray', vmin=0, vmax=1, origin='lower',
                   extent=[-extent, extent, -extent, extent], interpolation='nearest')
    ax.set_xlabel('µm')
    ax.set_ylabel('µm')
    title = ax.set_title('', fontsize=9)
    fig.suptitle(f'{freq:g} Hz reversal | bar {bar:g} µm | '
                 f'bright +{bright:g} / dark {dark_contrast:g}', fontsize=9)

    n_frames = max(int(round(duration_s * fps)), 2)

    def update(k):
        t = k / fps
        # The protocol's own rule: cos >= 0 shows the inverted frame (phase A).
        phase = 'A' if np.sign(np.cos(2 * np.pi * freq * t)) > 0 else 'B'
        im.set_data(phase_a if phase == 'A' else phase_b)
        title.set_text(f't = {t * 1e3:4.0f} ms   phase {phase}')
        return (im, title)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    if not embed:
        return anim
    from IPython.display import HTML
    html = HTML(anim.to_jshtml(fps=fps))
    plt.close(fig)
    return html


def describe_cell(cell: str, groups: pd.DataFrame, show: bool = True, **kwargs):
    """Basic information about one cell before analyzing any of its recordings.

    Cell type, how many conditions it was recorded in, and one row per
    condition. ``cell`` is '<experiment>/<cell label>'.
    """
    from retinanalysis.SCutils import explore as _sc
    return _sc.describe_cell(groups, cell, show=show, **kwargs)
