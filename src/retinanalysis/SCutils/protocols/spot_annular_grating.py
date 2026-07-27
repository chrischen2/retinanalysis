"""spotWithAnnularGrating: contrast tuning, cancellation point, cone prediction.

Python port of ``analyzeSpotAnnularGrating.m`` / ``spotAnnularGratingMain.m``
(linCone repo), reading from DataJoint + ``SCResponseBlock`` instead of
riekesuite.

The protocol (``chris-package/.../spotWithAnnularGrating.m``) shows a square-wave
grating masked to ``annulusInnerDiameter/2 <= r <= annulusOuterDiameter/2``, with
bright bars fixed at ``brightBarContrast`` and dark bars swept over
``darkBarContrast``; a disc of ``spotIntensity`` and diameter
``apertureDiameter`` sits on top. Bar width and grating polarity (+/-1) are
interleaved and pooled in analysis, so each (cell, recording mode, light level)
yields one response-vs-darkBarContrast tuning curve.

**Where the grating lands is set by annulusInnerDiameter, not apertureDiameter**:
``inner == 0`` makes the mask a disc covering the receptive-field center;
``inner > 0`` makes a true annulus over the surround. ``apertureDiameter`` only
controls the center spot drawn on top. In the recorded data these track cell
polarity as intended -- grating over the center for OFF cells, over the surround
for ON cells -- but the two parameters are independent, so group on
:func:`grating_site`.

Light level is keyed on (filter-wheel NDF, backgroundIntensity) and mapped to R*
by :func:`light_level_rstar`, which needs the rig -- E and G differ by 2.6x at
the same setting. The exact value lives in ``rstar``; ``rstar_level`` snaps it
to the nearest nominal rung in :data:`RSTAR_LEVELS` for grouping and labelling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROTOCOL = 'spotWithAnnularGrating'

# analyzeSpotAnnularGrating.m paras block.
DEFAULTS = dict(
    spike_th=3,            # thresholdSpikeFactor (SpikeDetectorNew)
    psth_sigma_ms=10.0,
    wc_offset=100,         # samples, whole-cell response window offset
    smooth_ms=10.0,        # box-car width for whole-cell traces
    spike_offset=300,      # samples, spiking response window offset
    cone_i0=2000.0,        # Weber I0 (R*) for the cancellation prediction
)

# Photoisomerization rate at display intensity 1.0 with the filter wheel at
# NDF 0, measured per rig. Everything else follows from these two numbers: the
# wheel attenuates by 10**NDF, and the display is linear in intensity, so
#
#     R* = RIG_MAX_RSTAR[rig] / 10**NDF * intensity
#
# The rigs differ by a factor of 2.6, which is why the light level cannot be
# read off the wheel setting alone.
RIG_MAX_RSTAR: Dict[str, float] = {'E': 30000.0, 'G': 77000.0}

# (filter-wheel NDF, backgroundIntensity) -> R*, the table from
# lightLevelRStar() in the MATLAB. Superseded by RIG_MAX_RSTAR and kept only as
# the fallback for a recording whose rig is not in that dict.
#
# Worth knowing what it actually was: every entry matches rig G to within
# 0.91-1.16x (it is rounded), and none describe rig E, which is 2.6x dimmer.
# It keyed on the wheel and background alone, so a rig-E block landing on one
# of these pairs would have been handed rig-G numbers -- as it happens none
# ever did, and the table's real effect was to leave 129 of 180 blocks, and
# every rig-E block, with no light level at all.
RSTAR_TABLE: List[Tuple[float, float, float]] = [
    (0.0, 0.15, 12000.0),
    (1.0, 0.15, 1000.0),
    (1.0, 0.30, 2000.0),
    (0.5, 0.15, 4000.0),
    (0.5, 0.30, 8000.0),
]

# The nominal light levels the experiments were aimed at. The calibration gives
# a continuous R* -- 11550 on rig G at wheel 0 / bg 0.15, 9000 on rig E at wheel
# 0 / bg 0.30 -- and those exact numbers are kept, but for grouping and labelling
# each recording is assigned to the nearest of these rungs.
RSTAR_LEVELS: Tuple[float, ...] = (1000.0, 2000.0, 4000.0, 5000.0, 7000.0,
                                   10000.0, 15000.0, 20000.0)

# Only these filter-wheel settings are calibrated / used for analysis. Anything
# else -- including a missing reading -- is excluded by group_blocks().
ALLOWED_FILTER_WHEEL = (0.0, 0.5, 1.0)

# Bright-bar contrasts the analysis runs on. The protocol was almost always run
# at 0.9 or 1.0; the handful of blocks at 0.25 and 0.5 are a one-cell sweep
# (2026-04-04_E Cell1) too small to say anything on its own, and pooling them
# with the rest would mix stimuli the cone prediction treats differently -- it
# is a function of the (light level, bright bar) pair. group_blocks() drops
# anything else and reports it.
ALLOWED_BRIGHT_CONTRAST = (0.9, 1.0)

# Cell types the analysis is currently restricted to; override per call.
DEFAULT_CELL_TYPES = ('ON-parasol', 'OFF-parasol')

# The recording-mode check lives in SCutils.recording_mode so every single-cell
# protocol resolves it the same way; re-exported here so sag.* keeps working.
from retinanalysis.SCutils.recording_mode import (      # noqa: E402
    MAX_SERIES_RESISTANCE,
    _amp_epoch_groups,
    check_series_resistance,
    mode_family,
    read_series_resistance,
    read_stage_ndfs,
    resolve_recording_mode,
    series_resistance_table,
    stage_ndf_table,
    trace_is_spiking,
)

# Epoch parameters that define a stimulus configuration / light level.
CONFIG_KEYS = ['apertureDiameter', 'annulusInnerDiameter', 'annulusOuterDiameter',
               'backgroundIntensity', 'spotIntensity', 'NDF', 'onlineAnalysis',
               'brightBarContrast', 'barWidth', 'preTime', 'stimTime', 'tailTime',
               'sampleRate', 'micronsPerPixel']

# Narrow bars are carried by the optics as much as by the cell: below roughly a
# cone-spacing-and-blur scale the grating is low-pass filtered before it reaches
# the photoreceptors, so a "cancellation" measured there is partly an optical
# result. group_blocks() keeps blocks at or above this bar width and reports the
# rest.
#
# The protocol interleaves bar width and analyze_group() pools across it, but in
# this dataset every one of the 180 blocks ran a single width, so the filter is
# applied per block on the recorded ``barWidth``.
MIN_BAR_WIDTH = 60.0

# Fewest epochs a recording group may rest on. The tuning curve splits its
# epochs across ~11 dark contrasts and then pools polarity within each, so a
# group of 10 epochs is roughly one epoch per contrast and no polarity averaging
# at all -- the crossing it yields is a single trial, not a measurement.
# group_blocks() drops groups below this and reports them.
MIN_EPOCHS = 16


# --------------------------------------------------------------------------
# light level / model
# --------------------------------------------------------------------------

def max_rstar(rig, ndf: float = 0.0) -> float:
    """R* at display intensity 1.0 for a rig at a given filter-wheel setting.

    The wheel is a neutral density, so it attenuates by ``10**NDF``: rig G's
    77000 R* ceiling becomes 24350 at NDF 0.5 and 7700 at NDF 1. NaN for a rig
    with no measured ceiling or a missing wheel reading.
    """
    base = RIG_MAX_RSTAR.get(str(rig).strip().upper()) if rig is not None else None
    if base is None or ndf is None or np.isnan(float(ndf)):
        return np.nan
    return float(base) / 10.0 ** float(ndf)


def is_calibrated(ndf: float, background_intensity: float, rig=None) -> bool:
    """True when this recording's light level rests on a measured calibration.

    That means a known rig and a filter-wheel reading (:data:`RIG_MAX_RSTAR`).
    Without a rig it falls back to asking whether the combination is an exact
    entry in the older :data:`RSTAR_TABLE`.
    """
    if rig is not None and np.isfinite(max_rstar(rig, ndf if ndf is not None else np.nan)):
        return True
    if ndf is None or np.isnan(ndf):
        return False
    return any(abs(ndf - fw) < 1e-6 and abs(background_intensity - bg) < 1e-6
               for fw, bg, _ in RSTAR_TABLE)


def round_rstar(rstar: float, levels: Sequence[float] = RSTAR_LEVELS) -> float:
    """Snap a measured R* to the nearest nominal level in :data:`RSTAR_LEVELS`.

    Nearest is measured **in log space**, because the levels are rungs on a
    roughly geometric ladder and a light level is only meaningful as a ratio: a
    recording at 1500 R* is 1.5x the 1000 rung and 0.75x the 2000 one, so it
    belongs with 2000, even though the two are equidistant on a linear axis.
    NaN in, NaN out.
    """
    if rstar is None or not np.isfinite(float(rstar)) or float(rstar) <= 0:
        return np.nan
    grid = np.asarray(levels, dtype=float)
    return float(grid[np.argmin(np.abs(np.log(grid) - np.log(float(rstar))))])


def light_setting(ndf: float, background_intensity: float) -> str:
    """The light level as recorded: ``'FW0/bg0.50'``.

    The raw, always-available description — filter-wheel NDF plus background
    intensity — independent of any calibration. :func:`light_level_rstar`
    converts it to R*.
    """
    fw = 'FW?' if ndf is None or np.isnan(ndf) else f'FW{ndf:g}'
    return f'{fw}/bg{background_intensity:g}'


def light_level_rstar(ndf: float, background_intensity: float,
                      rig=None) -> Tuple[float, str]:
    """Mean photoisomerization rate for a recording, and a label for it.

    With a rig, this is the measured calibration::

        R* = RIG_MAX_RSTAR[rig] / 10**NDF * backgroundIntensity

    — the rig's intensity-1 ceiling, attenuated by the wheel, scaled by the
    background the protocol actually ran at. **The rig is not optional in
    practice**: E and G differ by 2.6x, so the same wheel and background mean
    two different light levels on the two rigs.

    Without a rig it falls back to :data:`RSTAR_TABLE`, the older MATLAB
    lookup, which covers five (NDF, background) pairs and is really rig G's
    numbers. Anything it does not cover stays NaN and the label says so, rather
    than being quietly filled in.

    Returns ``(rstar, label)``, where ``rstar`` is the exact calibrated value
    and the label names the nominal rung it rounds to (:func:`round_rstar`), so
    11550 R* and 12000 R* both read ``'10000R*'`` and group together.
    """
    ceiling = max_rstar(rig, ndf)
    if np.isfinite(ceiling) and background_intensity is not None \
            and not np.isnan(float(background_intensity)):
        rstar = ceiling * float(background_intensity)
        return rstar, f'{round_rstar(rstar):g}R*'

    for fw, bg, rstar in RSTAR_TABLE:
        if (ndf is not None and not np.isnan(ndf) and abs(ndf - fw) < 1e-6
                and abs(background_intensity - bg) < 1e-6):
            return rstar, f'{round_rstar(rstar):g}R*'
    return np.nan, f'{light_setting(ndf, background_intensity)} (?R*)'


def apply_rstar_mapping(summary: pd.DataFrame,
                        mapping: Dict[Tuple[float, float], float]) -> pd.DataFrame:
    """Override the light-level conversion on a stored summary.

    :data:`RIG_MAX_RSTAR` now gives every rig-E and rig-G recording an R*, so
    this is no longer needed to fill the Weber comparison in. It remains for
    re-calibrating: measure the rigs again and you can restate stored records
    without re-analyzing anything, since the crossings never depended on the
    conversion::

        summary = sag.apply_rstar_mapping(sag.load_summary(), {
            (0.0, 0.50): 40000, (0.0, 0.30): 24000, (1.0, 0.50): 3333,
        })
        sag.plot_weber_comparison(sag.add_condition(summary))

    Keys are ``(ndf, background_intensity)``, so a mapping applies to **both
    rigs at once** — which is wrong for anything but a single-rig summary,
    since E and G differ by 2.6x at the same setting. Filter the summary by rig
    first, or change :data:`RIG_MAX_RSTAR` and re-run. Returns a copy with
    ``rstar`` and ``light_level`` updated.
    """
    out = summary.copy()
    rstar = list(out['rstar'])
    label = list(out['light_level'])
    for i, (ndf, bg) in enumerate(zip(out['ndf'], out['background_intensity'])):
        for (m_ndf, m_bg), value in mapping.items():
            same_fw = (np.isnan(m_ndf) and pd.isna(ndf)) or np.isclose(m_ndf, ndf)
            if same_fw and np.isclose(m_bg, bg):
                rstar[i] = float(value)
                label[i] = f'{round_rstar(float(value)):g}R*'
                break
    out['rstar'] = rstar
    out['light_level'] = label
    return out


def cone_predict_dark_contrast(rstar: float, bright_contrast: float,
                               i0: float = DEFAULTS['cone_i0']) -> float:
    """Dark-bar contrast that cancels the bright bar under a Weber cone model.

    Port of ``conePredictDarkContrast``. With R(I) = I / (I + I0), balance is
    ``(Ib - Im)/(Ib + I0) = (Im - Id)/(Id + I0)`` where Im = rstar and
    Ib = Im*(1 + bright_contrast). NaN in, NaN out.
    """
    if rstar is None or np.isnan(rstar) or bright_contrast is None or np.isnan(bright_contrast):
        return np.nan
    im = float(rstar)
    ib = im * (1.0 + bright_contrast)
    lam = (ib - im) / (ib + i0)
    idk = (im - lam * i0) / (1.0 + lam)
    return idk / im - 1.0


def interp_zero_crossing(x, y) -> float:
    """First linearly interpolated x where y crosses zero; NaN if it never does."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    for k in range(len(y) - 1):
        if (y[k] <= 0 <= y[k + 1]) or (y[k] >= 0 >= y[k + 1]):
            if y[k + 1] == y[k]:
                return float(x[k])
            return float(x[k] - y[k] * (x[k + 1] - x[k]) / (y[k + 1] - y[k]))
    return np.nan


def read_filter_wheel_ndf(exp_name: str, block_id: int) -> float:
    """Read ``background:FilterWheel:NDF`` straight out of the raw h5.

    Symphony stores it per epoch under
    ``backgrounds/<device>/dataConfigurationSpans/span_0/FilterWheel``, which is
    what ``epoch.protocolSettings('background:FilterWheel:NDF')`` returns in the
    MATLAB. Rigs without a filter wheel have no such group -> NaN.

    The parser already lifts this value into the ``NDF`` epoch parameter, so
    :func:`find_blocks` reads the fast database copy; this function is the
    authority used to verify it (``find_blocks(verify_fw=True)``).
    """
    import h5py
    from retinanalysis.config import schema
    from retinanalysis.utils.datajoint_utils import get_h5_file

    ep = (schema.Epoch() & f'parent_id={int(block_id)}').to_pandas()
    if ep.empty:
        return np.nan
    eid = int((ep['id'] if 'id' in ep else ep.index)[0])
    resp = (schema.Response() & f'parent_id={eid}').to_pandas()
    if resp.empty:
        return np.nan
    epoch_group = str(resp['h5path'].iloc[0]).split('/responses/')[0]

    with h5py.File(get_h5_file(exp_name), 'r') as f:
        if epoch_group not in f:
            raise KeyError(f'block {block_id} is not in {exp_name} '
                           f'(its epoch group is absent from that h5)')
        backgrounds = f[epoch_group].get('backgrounds')
        if backgrounds is None:
            return np.nan
        for dev in backgrounds:
            spans = backgrounds[dev].get('dataConfigurationSpans')
            if spans is None:
                continue
            for span in spans:
                fw = spans[span].get('FilterWheel')
                if fw is not None and 'NDF' in fw.attrs:
                    return float(fw.attrs['NDF'])
    return np.nan




def rig_of(exp_name: str) -> str:
    """Rig letter from an experiment name: ``'2026-06-04_G'`` -> ``'G'``.

    Handles the trailing-index form too (``'2026-01-02_E_2'`` -> ``'E'``).
    """
    import re
    m = re.search(r'_([A-Za-z])(?:_\d+)?$', str(exp_name))
    return m.group(1).upper() if m else '?'


def grating_site(annulus_inner_diameter: float) -> str:
    """Where the grating is: 'center' when its mask starts at r=0, else 'surround'.

    The protocol masks the grating to ``inner/2 <= r <= outer/2``, so the inner
    diameter alone says whether the grating reaches the receptive-field center.

    **``apertureDiameter`` does not decide this, and it is the natural thing to
    assume it does.** The aperture is the center *spot* drawn on top of the
    grating, a separate object — see :func:`center_spot`. The two come apart in
    44 blocks of this dataset, which have ``apertureDiameter == 0`` (no spot)
    and ``annulusInnerDiameter > 0`` (annular grating): at inner 400 / outer
    1200 the grating covers r = 200-600 um and the center r = 0-200 um is plain
    background, stimulated by nothing at all. Those are surround recordings with
    an empty center, not center recordings — 37 of them are the ON-parasol
    surround series. Keying on the aperture would relabel every one.

    So the three configurations that actually occur are::

        inner == 0                 -> grating disc over the center
        inner > 0, aperture == 0   -> grating annulus over the surround, empty center
        inner > 0, aperture > 0    -> grating annulus over the surround, spot in the center

    ``grating_site`` separates the first from the other two; :func:`center_spot`
    separates the second from the third.
    """
    return 'center' if float(annulus_inner_diameter) == 0.0 else 'surround'


def center_spot(aperture_diameter: float) -> str:
    """Whether a center spot was drawn on top of the grating: 'spot' or 'none'.

    ``apertureDiameter`` is the diameter of the ``spotIntensity`` disc the
    protocol paints over the middle of the frame (``stimulus_frame`` fills
    ``r <= apertureDiameter/2``); 0 means no spot was drawn.

    This is the *other* half of the stimulus configuration, independent of
    :func:`grating_site` — a surround grating can be run with the center left
    empty or with a spot in it, and those are different experiments. Splitting
    them out is what makes both groupable without relabelling either.
    """
    if aperture_diameter is None or (isinstance(aperture_diameter, float)
                                     and np.isnan(aperture_diameter)):
        return ''
    return 'spot' if float(aperture_diameter) > 0.0 else 'none'


# --------------------------------------------------------------------------
# stimulus schematic
# --------------------------------------------------------------------------

def stimulus_frame(aperture_diameter: float, annulus_inner_diameter: float,
                   annulus_outer_diameter: float, bar_width: float,
                   background_intensity: float, spot_intensity: float,
                   bright_bar_contrast: float, dark_bar_contrast: float,
                   grating_polarity: float = 1.0,
                   extent_um: Optional[float] = None,
                   n_pixels: int = 601) -> Tuple[np.ndarray, float]:
    """Render the stimulus main frame; port of ``createAnnularGrating`` + spot.

    Computed in microns (the MATLAB works in pixels, but only the ratio
    ``x / barWidth`` matters, so the frames are identical). Returns
    ``(image, extent_um)`` with image in intensity units 0-1 and ``extent_um``
    the half-width of the field, so imshow extent is
    ``[-extent, extent, -extent, extent]``.
    """
    if extent_um is None:
        extent_um = max(annulus_outer_diameter, aperture_diameter) * 0.62
    g = np.linspace(-extent_um, extent_um, n_pixels)
    x, y = np.meshgrid(g, g)
    r = np.hypot(x, y)

    # Square-wave grating: sign(sin(2*pi*x/barWidth)) -> barWidth is the period.
    grating = grating_polarity * np.sign(np.sin(2 * np.pi * x / bar_width))
    img_grating = np.full_like(grating, background_intensity, dtype=float)
    img_grating[grating > 0] = background_intensity * (1.0 + bright_bar_contrast)
    img_grating[grating <= 0] = background_intensity * (1.0 + dark_bar_contrast)

    frame = np.full_like(img_grating, background_intensity, dtype=float)
    annulus = (r >= annulus_inner_diameter / 2.0) & (r <= annulus_outer_diameter / 2.0)
    frame[annulus] = img_grating[annulus]

    # Center spot is drawn on top of the grating.
    if aperture_diameter > 0:
        frame[r <= aperture_diameter / 2.0] = spot_intensity
    return frame, float(extent_um)


def plot_stimulus_schematic(params: Dict, dark_contrasts: Optional[Sequence[float]] = None,
                            polarity: float = 1.0, figsize: Tuple[float, float] = (10.5, 3.4)):
    """Show the stimulus main frame for a few dark-bar contrasts.

    ``params`` is an epoch-parameter dict (or any mapping with the protocol
    keys). Passing several ``dark_contrasts`` shows how the swept variable
    changes the frame; the geometry annotation is shared.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if dark_contrasts is None:
        dc = params.get('darkBarContrast', [-1.0])
        dc = list(dc) if isinstance(dc, (list, tuple, np.ndarray)) else [dc]
        dark_contrasts = [dc[0], dc[len(dc) // 2], dc[-1]] if len(dc) >= 3 else dc

    site = grating_site(params['annulusInnerDiameter'])
    spot = center_spot(params['apertureDiameter'])
    fig, axes = plt.subplots(1, len(dark_contrasts), figsize=figsize, squeeze=False)
    for ax, dark in zip(axes[0], dark_contrasts):
        frame, extent = stimulus_frame(
            params['apertureDiameter'], params['annulusInnerDiameter'],
            params['annulusOuterDiameter'], params.get('currentBarWidth', params['barWidth']),
            params['backgroundIntensity'], params['spotIntensity'],
            params.get('currentBrightContrast', params['brightBarContrast']), dark,
            grating_polarity=polarity)
        ax.imshow(frame, cmap='gray', vmin=0, vmax=1, origin='lower',
                  extent=[-extent, extent, -extent, extent], interpolation='nearest')
        ax.set_title(f'dark contrast {dark:g}', fontsize=9)
        ax.set_xlabel('µm')
        ax.set_xticks([-extent, 0, extent])
        ax.set_yticks([-extent, 0, extent])
    axes[0][0].set_ylabel('µm')

    bar = params.get('currentBarWidth', params['barWidth'])
    spot_txt = ('no center spot' if spot == 'none'
                else f"spot {params['apertureDiameter']:g} µm @ {params['spotIntensity']:g}")
    fig.suptitle(
        f"grating over {site}  |  annulus {params['annulusInnerDiameter']:g}-"
        f"{params['annulusOuterDiameter']:g} µm  |  {spot_txt}"
        f"  |  bar {bar:g} µm  |  bg {params['backgroundIntensity']:g}"
        f"  |  bright {params.get('currentBrightContrast', params['brightBarContrast']):g}",
        fontsize=9, y=1.02)
    return fig


# --------------------------------------------------------------------------
# discovery: which blocks ran this protocol, and how they group
# --------------------------------------------------------------------------

def find_blocks(exp_names: Optional[Sequence[str]] = None, show: bool = True,
                height: int = 420, verify_fw: bool = False) -> pd.DataFrame:
    """Every single-cell block that ran spotWithAnnularGrating, with its config.

    One row per epoch block, carrying the grouping keys the MATLAB epoch tree
    splits on -- cell type, cell label, recording mode (onlineAnalysis),
    filter-wheel NDF and backgroundIntensity -- plus the stimulus geometry and
    the derived ``grating_site`` / ``light_level``.
    """
    import retinanalysis as ra
    from retinanalysis.config import schema
    from retinanalysis.SCutils import explore as sc

    blocks = sc.find_blocks(PROTOCOL, show=False)
    if blocks.empty:
        return blocks
    if exp_names is not None:
        blocks = blocks[blocks['exp_name'].isin(exp_names)]

    # Cell identity comes from the experiment summary; stimulus config from the
    # first epoch of each block (constant within a block).
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
    df['center_spot'] = df['apertureDiameter'].apply(center_spot)
    df['cell_type_short'] = df['cell_type'].astype(str).str.split('\\').str[-1]
    df['rig'] = df['exp_name'].apply(rig_of)

    # NDF here is background:FilterWheel:NDF, lifted from the h5 by the parser
    # (verified equal to read_filter_wheel_ndf on every sampled block). A rig
    # with no filter wheel leaves it missing -- those blocks have no defined
    # light level and should not enter the Weber comparison.
    df = df.rename(columns={'NDF': 'filter_wheel_ndf', 'barWidth': 'bar_width'})
    df['has_filter_wheel'] = df['filter_wheel_ndf'].notna()
    if verify_fw:
        mismatch = []
        for _, r in df.iterrows():
            h5_fw = read_filter_wheel_ndf(r['exp_name'], r['block_id'])
            db_fw = r['filter_wheel_ndf']
            if not (np.isclose(h5_fw, db_fw) or (np.isnan(h5_fw) and pd.isna(db_fw))):
                mismatch.append((r['exp_name'], int(r['block_id']), db_fw, h5_fw))
        print(f'filter-wheel verification against the h5: '
              f'{len(df) - len(mismatch)}/{len(df)} agree')
        for exp, bid, db_fw, h5_fw in mismatch:
            print(f'  MISMATCH {exp} block {bid}: database {db_fw} vs h5 {h5_fw}')

    # The fixed filters in the light path, which the wheel setting does not
    # cover and which change between blocks on some dates.
    df = df.merge(stage_ndf_table(df[['exp_name', 'block_id']], verbose=show),
                  on='block_id', how='left')
    df['stage_ndfs'] = df['stage_ndfs'].fillna('')

    # R* comes from the rig's measured ceiling, so the rig has to come with the
    # wheel setting -- the two rigs differ by 2.6x at the same setting.
    rs = [light_level_rstar(n, b, rig=r)
          for n, b, r in zip(df['filter_wheel_ndf'], df['backgroundIntensity'], df['rig'])]
    df['rstar'] = [r for r, _ in rs]
    df['light_level'] = [lab for _, lab in rs]
    df['rstar_level'] = [round_rstar(r) for r, _ in rs]
    df['light_setting'] = [light_setting(n, b)
                           for n, b in zip(df['filter_wheel_ndf'], df['backgroundIntensity'])]
    df['rstar_measured'] = [is_calibrated(n, b, rig=r)
                            for n, b, r in zip(df['filter_wheel_ndf'],
                                               df['backgroundIntensity'], df['rig'])]
    df['max_rstar'] = [max_rstar(r, n) for r, n in zip(df['rig'], df['filter_wheel_ndf'])]
    df = df.sort_values(['exp_name', 'cell_label', 'start_time']).reset_index(drop=True)

    if show:
        # brightBarContrast is shown because the cone prediction depends on it:
        # the dark bar has to cancel *this* bright bar, so a block's predicted
        # balancing contrast is a function of the pair, not of the light level
        # alone. See cone_predict_dark_contrast().
        cols = ['exp_name', 'cell_label', 'cell_type_short', 'onlineAnalysis', 'grating_site',
                'center_spot', 'filter_wheel_ndf', 'stage_ndfs', 'backgroundIntensity',
                'light_level', 'apertureDiameter', 'annulusInnerDiameter',
                'annulusOuterDiameter', 'brightBarContrast', 'bar_width',
                'n_epochs', 'block_id']
        print(f"{len(df)} blocks | {df['exp_name'].nunique()} experiments | "
              f"{df.groupby(['exp_name', 'cell_label']).ngroups} cells")
        missing = df[~df['has_filter_wheel']]
        if len(missing):
            print(f'  WARNING: {len(missing)} block(s) have no background:FilterWheel:NDF '
                  f'-- no light level, excluded from the Weber comparison. '
                  f"Experiments: {', '.join(sorted(missing['exp_name'].unique()))}")
        # Between two rungs the rounding is by construction the best available
        # description. Past either end it is a clamp, and the rung then
        # misdescribes the recording -- 38500 R* would read as 20000 -- so say so.
        # Restricted to the wheel settings that survive group_blocks; the deep-NDF
        # blocks land at single-digit R* and are dropped there anyway.
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
                        num_cols=('n_epochs', 'block_id', 'filter_wheel_ndf',
                                  'backgroundIntensity', 'brightBarContrast', 'bar_width'))
    return df


def check_polarity_pooling(df: pd.DataFrame, show: bool = True) -> pd.DataFrame:
    """Verify that both grating polarities are pooled, and evenly.

    :func:`analyze_group` groups epochs by ``currentDarkContrast`` alone, so the
    two ``currentGratingPolarity`` values (+1 / -1) are pooled by construction —
    polarity is never a grouping variable. Polarity flips which stripes are
    bright, so pooling it is what removes the dependence on where the bars
    happen to fall on the receptive field.

    Pooling correctly is not the same as pooling *evenly*, though. If a block
    was stopped mid-interleave, one polarity gets an extra epoch and the pooled
    mean is tilted toward it. This walks every epoch and reports, per
    (block, dark contrast, bar width) cell, whether the +1 and -1 counts match.

    Returns one row per cell with ``n_pos`` / ``n_neg`` / ``balanced``. Scanning
    every epoch takes a minute or two on the full dataset.
    """
    from retinanalysis.config import schema

    rows = []
    for bid in df['block_id']:
        for p in (schema.Epoch() & f'parent_id={int(bid)}').to_dicts():
            par = p.get('parameters', {})
            rows.append({'block_id': int(bid),
                         'dark_contrast': par.get('currentDarkContrast'),
                         'bar_width': par.get('currentBarWidth'),
                         'polarity': par.get('currentGratingPolarity')})
    ep = pd.DataFrame(rows)
    if ep.empty:
        if show:
            print('no epochs found')
        return ep

    has_pol = ep[ep['polarity'].notna()]
    counts = (has_pol.assign(pos=has_pol['polarity'] > 0)
              .groupby(['block_id', 'dark_contrast', 'bar_width'], dropna=False)['pos']
              .agg(n_pos='sum', n_total='size').reset_index())
    counts['n_neg'] = counts['n_total'] - counts['n_pos']
    counts['balanced'] = counts['n_pos'] == counts['n_neg']

    if show:
        n_blocks = ep['block_id'].nunique()
        with_pol = has_pol['block_id'].nunique()
        print(f'polarity is pooled by construction: analyze_group groups on '
              f'currentDarkContrast only.')
        print(f'  {len(ep)} epochs over {n_blocks} blocks; '
              f'{with_pol} blocks record currentGratingPolarity '
              f'({len(has_pol)} epochs), {n_blocks - with_pol} do not '
              f'({len(ep) - len(has_pol)} epochs)')
        vals = sorted(has_pol['polarity'].unique())
        print(f'  values recorded: {vals}')
        if len(counts):
            bal = int(counts['balanced'].sum())
            print(f'  (block x contrast x bar width) cells: {len(counts)} — '
                  f'{bal} balanced ({bal / len(counts):.0%}), '
                  f'{len(counts) - bal} off by '
                  f'{int((counts["n_pos"] - counts["n_neg"]).abs().max())} epoch or less')
            only_one = counts[(counts['n_pos'] == 0) | (counts['n_neg'] == 0)]
            if len(only_one):
                print(f'  WARNING: {len(only_one)} cell(s) have only ONE polarity, so '
                      f'nothing is averaged out there:')
                for _, r in only_one.head(8).iterrows():
                    print(f"    block {int(r['block_id'])} dark {r['dark_contrast']:g} "
                          f"bar {r['bar_width']:g}: +1 x{int(r['n_pos'])}, "
                          f"-1 x{int(r['n_neg'])}")
        totals = has_pol['polarity'].value_counts().to_dict()
        print(f'  overall epoch counts: {totals}')
    return counts


def group_blocks(df: pd.DataFrame, show: bool = True, height: int = 420,
                 require_filter_wheel: bool = True,
                 allowed_filter_wheel: Sequence[float] = ALLOWED_FILTER_WHEEL,
                 allowed_bright_contrast: Optional[Sequence[float]]
                 = ALLOWED_BRIGHT_CONTRAST,
                 min_bar_width: Optional[float] = MIN_BAR_WIDTH,
                 min_epochs: Optional[int] = MIN_EPOCHS) -> pd.DataFrame:
    """Collapse the block table to one row per recording group.

    A group is the MATLAB epoch-tree leaf: (experiment, cell, recording mode,
    grating site, filter-wheel NDF, backgroundIntensity). Blocks within a group
    are pooled by :func:`analyze_group`.

    ``require_filter_wheel`` (default) drops blocks recorded on a rig with no
    filter wheel: without ``background:FilterWheel:NDF`` the light level is
    undefined, so those recordings cannot enter the Weber comparison. What was
    dropped is always reported.

    ``allowed_bright_contrast`` keeps only blocks whose ``brightBarContrast`` is
    one of :data:`ALLOWED_BRIGHT_CONTRAST` (0.9, 1.0) — the two the protocol was
    effectively always run at. Pass ``None`` to keep every contrast, which is
    what you want if you are analyzing the bright-contrast sweep itself.

    ``min_bar_width`` drops blocks run at a bar width below
    :data:`MIN_BAR_WIDTH` (60 µm), where the optics low-pass the grating enough
    that the cancellation is partly an optical result. ``None`` keeps every
    width. This is a per-block test: bar width is interleaved in principle and
    :func:`analyze_group` pools across it, but every block in this dataset ran a
    single width.

    ``min_epochs`` drops whole recording groups with fewer than
    :data:`MIN_EPOCHS` (16) epochs. Unlike the others this is a test on the
    *group*, applied after the blocks are pooled, since a cell can reach a
    usable count by having been run twice. ``None`` keeps every group.
    """
    from retinanalysis.SCutils import explore as sc

    needed = ['rig', 'light_setting', 'filter_wheel_ndf', 'grating_site', 'rstar_level',
              'apertureDiameter', 'annulusInnerDiameter', 'annulusOuterDiameter']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(
            f'find_blocks() output is missing {missing}. This usually means the block '
            f'table was built by an older version of this module still held in the '
            f'kernel -- restart the kernel and re-run find_blocks().')

    if require_filter_wheel and 'has_filter_wheel' in df.columns:
        dropped = df[~df['has_filter_wheel']]
        if len(dropped):
            print(f'dropping {len(dropped)} block(s) with no filter-wheel NDF: '
                  f"{', '.join(sorted(dropped['exp_name'].unique()))}")
        df = df[df['has_filter_wheel']]
        # Only calibrated wheel settings are analyzable.
        off_list = df[~df['filter_wheel_ndf'].isin(list(allowed_filter_wheel))]
        if len(off_list):
            print(f'dropping {len(off_list)} block(s) whose filter wheel is not in '
                  f'{list(allowed_filter_wheel)}: NDF '
                  f"{sorted(off_list['filter_wheel_ndf'].dropna().unique().tolist())}")
        df = df[df['filter_wheel_ndf'].isin(list(allowed_filter_wheel))]

    # Bright bar is the other half of the cone prediction, so a block shown at a
    # contrast the rest of the dataset never used cannot be pooled with them.
    if allowed_bright_contrast is not None and 'brightBarContrast' in df.columns:
        keep = df['brightBarContrast'].isin(list(allowed_bright_contrast))
        if (~keep).any():
            dropped = df[~keep]
            print(f'dropping {len(dropped)} block(s) whose bright bar contrast is not in '
                  f'{list(allowed_bright_contrast)}: '
                  + ', '.join(f'{c:g} ({n} block{"s" if n > 1 else ""})'
                              for c, n in sorted(
                                  dropped['brightBarContrast'].value_counts().items()))
                  + f" -- {', '.join(sorted(dropped['exp_name'].unique()))}")
        df = df[keep]

    if min_bar_width is not None and 'bar_width' in df.columns:
        keep = df['bar_width'] >= float(min_bar_width)
        if (~keep).any():
            dropped = df[~keep]
            print(f'dropping {len(dropped)} block(s) with bar width below '
                  f'{float(min_bar_width):g} µm: '
                  + ', '.join(f'{w:g} µm ({n} block{"s" if n > 1 else ""})'
                              for w, n in sorted(dropped['bar_width'].value_counts().items()))
                  + f" -- {', '.join(sorted(dropped['exp_name'].unique()))}")
        df = df[keep]

    # Purely derived from apertureDiameter, so fill it in rather than demanding
    # the caller's block table already carry it.
    if 'center_spot' not in df.columns:
        df = df.assign(center_spot=df['apertureDiameter'].apply(center_spot))

    keys = ['exp_name', 'rig', 'cell_label', 'cell_type_short', 'onlineAnalysis',
            'grating_site', 'filter_wheel_ndf', 'backgroundIntensity']
    agg = dict(blocks=('block_id', 'size'), epochs=('n_epochs', 'sum'),
               light_setting=('light_setting', 'first'),
               light_level=('light_level', 'first'),
               rstar=('rstar', 'first'),
               rstar_level=('rstar_level', 'first'),
               aperture=('apertureDiameter', 'first'),
               annulus_inner=('annulusInnerDiameter', 'first'),
               annulus_outer=('annulusOuterDiameter', 'first'),
               spot_intensity=('spotIntensity', 'first'),
               # Joined like bright: bar width is not a grouping key either, and
               # analyze_group pools across it, so a group spanning two widths
               # has to say so rather than report whichever came first.
               bar_width=('bar_width',
                          lambda s: ', '.join(f'{v:g}' for v in sorted(set(s)))),
               # Joined for the same reason as bright: the aperture is not a
               # grouping key either, so a group that ever mixed spot with no
               # spot has to show it rather than report whichever came first.
               center_spot=('center_spot',
                            lambda s: ', '.join(sorted(set(str(v) for v in s)))),
               # Joined, not 'first': brightBarContrast is a block-level setting
               # (constant within a block, varied between them) but it is *not*
               # a grouping key, so a cell that was swept over bright contrast
               # lands in one group. Showing only the first would hide that the
               # tuning curve underneath averages different stimuli.
               bright=('brightBarContrast',
                       lambda s: ', '.join(f'{v:g}' for v in sorted(set(s), reverse=True))),
               block_ids=('block_id', lambda s: ', '.join(str(int(b)) for b in sorted(s))))
    if 'stage_ndfs' in df.columns:
        # Joined rather than 'first': a group can span blocks run behind
        # different fixed filters, and silently showing one would hide that.
        agg['stage_ndfs'] = ('stage_ndfs',
                             lambda s: ' | '.join(sorted({str(v) for v in s})))
    # Carry the amplifier reading through when check_series_resistance() has run,
    # so the recording-group table shows how each cell was actually held. In
    # MOhm here because the table is for reading; the ohms stay canonical.
    has_rs = 'series_resistance' in df.columns
    if has_rs:
        agg['rs_mohm'] = ('series_resistance', lambda s: np.round(np.nanmedian(s) / 1e6, 2))
        agg['epochs_high_rs'] = ('n_epochs_high_rs', 'sum')
    g = df.groupby(keys, dropna=False, sort=False).agg(**agg).reset_index()

    # After pooling, not before: a cell run twice at the same condition reaches a
    # usable count between them, and dropping its blocks individually would lose
    # a group that is actually well sampled.
    if min_epochs is not None:
        keep = g['epochs'] >= int(min_epochs)
        if (~keep).any():
            thin = g[~keep]
            print(f'dropping {len(thin)} recording group(s) with fewer than '
                  f'{int(min_epochs)} epochs '
                  f'({int(thin["epochs"].sum())} epochs, '
                  f'{thin["epochs"].min():g}-{thin["epochs"].max():g} each):')
            for _, r in thin.sort_values('epochs').iterrows():
                print(f"    {r['exp_name']} {r['cell_label']} {r['cell_type_short']} "
                      f"{r['onlineAnalysis']} {r['grating_site']} "
                      f"{r['rstar_level']:g}R*: {int(r['epochs'])} epochs")
        g = g[keep].reset_index(drop=True)

    if show:
        print(f'{len(g)} recording groups '
              f'(experiment x cell x mode x grating site x filter wheel x background)')
        # The cone prediction is a function of the bright bar the dark bar has to
        # cancel, so pooling several bright contrasts into one tuning curve makes
        # the measured crossing uninterpretable against it.
        mixed = g[g['bright'].str.contains(',')]
        if len(mixed):
            print(f'  WARNING: {len(mixed)} group(s) pool more than one '
                  f'brightBarContrast -- their tuning curve averages different '
                  f'stimuli, and the crossing cannot be compared to a single '
                  f'cone prediction. brightBarContrast is not a grouping key:')
            for _, r in mixed.iterrows():
                print(f"    {r['exp_name']} {r['cell_label']} {r['onlineAnalysis']} "
                      f"{r['grating_site']} FW{r['filter_wheel_ndf']:g}/"
                      f"bg{r['backgroundIntensity']:g}: bright {r['bright']} "
                      f"({r['blocks']} blocks, {r['epochs']} epochs)")
        cols = ['cell_type_short', 'rig', 'exp_name', 'cell_label', 'onlineAnalysis',
                'grating_site', 'center_spot', 'aperture', 'annulus_inner', 'annulus_outer',
                'spot_intensity', 'bright', 'bar_width', 'filter_wheel_ndf',
                'backgroundIntensity', 'rstar_level', 'blocks', 'epochs']
        cols += [c for c in ('rs_mohm', 'epochs_high_rs') if c in g.columns]
        sc.tree_table(g.sort_values(['cell_type_short', 'exp_name', 'cell_label'])[cols],
                      levels=['cell_type_short', 'rig', 'exp_name', 'cell_label'],
                      height=height, num_cols=('aperture', 'annulus_inner', 'annulus_outer',
                                               'spot_intensity', 'filter_wheel_ndf',
                                               'backgroundIntensity',
                                               'rstar_level', 'blocks', 'epochs', 'rs_mohm',
                                               'epochs_high_rs'))
    return g


def select_cell_types(groups: pd.DataFrame,
                      cell_types: Sequence[str] = DEFAULT_CELL_TYPES,
                      show: bool = True) -> pd.DataFrame:
    """Keep only the given cell types.

    Set ``cell_types`` to whatever you want to analyze; the default restricts to
    ON and OFF parasols. Unlike :func:`select_canonical` this does not also pin
    the grating site, so a cell recorded with the grating on either side is kept
    and ``grating_site`` stays a condition you can group on.
    """
    wanted = [str(c) for c in cell_types]
    out = groups[groups['cell_type_short'].isin(wanted)].reset_index(drop=True)
    if show:
        missing = [c for c in wanted if c not in set(groups['cell_type_short'])]
        print(f'{len(out)} of {len(groups)} groups are {", ".join(wanted)}')
        if missing:
            print(f'  no groups found for: {", ".join(missing)}')
        import pandas as _pd
        print(_pd.crosstab([out['cell_type_short'], out['grating_site']],
                           [out['rig'], out['onlineAnalysis']]).to_string())
    return out


# The two configurations the experiment was designed around: the grating covers
# the receptive-field center for OFF parasols and the surround for ON parasols.
CANONICAL_CONDITIONS = {
    ('ON-parasol', 'surround'): 'ON-parasol / surround',
    ('OFF-parasol', 'center'): 'OFF-parasol / center',
}


def select_canonical(groups: pd.DataFrame, show: bool = True) -> pd.DataFrame:
    """Keep only ON-parasol/surround and OFF-parasol/center groups.

    Adds a ``condition`` column naming the pairing. Everything else (midgets,
    horizontals, bipolars, and parasols recorded with the grating on the other
    side) is dropped.
    """
    keys = list(zip(groups['cell_type_short'], groups['grating_site']))
    out = groups[[k in CANONICAL_CONDITIONS for k in keys]].copy()
    out['condition'] = [CANONICAL_CONDITIONS[k] for k in keys if k in CANONICAL_CONDITIONS]
    if show:
        print(f'{len(out)} of {len(groups)} groups are ON-parasol/surround or '
              f'OFF-parasol/center')
        import pandas as _pd
        print(_pd.crosstab(out['condition'], out['onlineAnalysis']).to_string())
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# per-group analysis
# --------------------------------------------------------------------------

@dataclass
class GratingRecord:
    """One (experiment, cell, recording mode, grating site, NDF, background).

    This is the unit the MATLAB upserts into spotAnnularGratingSummary.mat, and
    the unit population analysis iterates over.
    """
    exp_name: str
    cell_label: str
    cell_type: str
    online_analysis: str
    grating_site: str
    ndf: float
    background_intensity: float
    rstar: float
    light_level: str
    dark_contrasts: np.ndarray
    resp_mean: np.ndarray
    resp_sem: np.ndarray
    resp_n: np.ndarray
    baseline_mean: float
    baseline_sem: float
    crossing_nearest: float
    crossing_interp: float
    bright_bar_contrast: float
    cone_pred_dark: float
    cone_i0: float
    bar_widths: np.ndarray
    traces: np.ndarray            # (n_dark_contrasts, n_samples) PSTH or mean current
    trace_time_ms: np.ndarray
    pre_time_ms: float
    stim_time_ms: float
    n_epochs: int
    block_ids: List[int]
    # Amplifier access resistance (ohms), median over the epochs that survived
    # the cutoff; 0 for a cell-attached recording, NaN when the h5 had no
    # reading. See check_series_resistance().
    series_resistance: float = np.nan
    n_epochs_high_rs: int = 0
    # True when the amplifier overruled onlineAnalysis. online_analysis is then
    # the mode the data was actually analyzed as, and online_analysis_recorded
    # the label the experimenter set.
    mode_mismatch: bool = False
    online_analysis_recorded: str = ''
    config: Dict = field(default_factory=dict)
    units: str = ''
    # Unprocessed amplifier traces, kept only when analyze_group(keep_raw=True),
    # so the raw views can be drawn without reloading the blocks. Never stored
    # on disk.
    raw: Optional[Dict] = None

    @property
    def key(self) -> str:
        return record_key(self.exp_name, self.cell_label, self.online_analysis,
                          self.grating_site, self.ndf, self.background_intensity)

    def summary_row(self) -> Dict:
        """Scalar fields only -- what goes in the population index."""
        return {
            'key': self.key, 'exp_name': self.exp_name, 'cell_label': self.cell_label,
            'cell_type': self.cell_type, 'online_analysis': self.online_analysis,
            'grating_site': self.grating_site, 'ndf': self.ndf,
            'background_intensity': self.background_intensity, 'rstar': self.rstar,
            'rstar_level': round_rstar(self.rstar), 'rig': rig_of(self.exp_name),
            'rstar_measured': is_calibrated(self.ndf, self.background_intensity,
                                            rig=rig_of(self.exp_name)),
            'light_setting': light_setting(self.ndf, self.background_intensity),
            'light_level': self.light_level, 'baseline_mean': self.baseline_mean,
            'baseline_sem': self.baseline_sem, 'crossing_nearest': self.crossing_nearest,
            'crossing_interp': self.crossing_interp,
            'bright_bar_contrast': self.bright_bar_contrast,
            'cone_pred_dark': self.cone_pred_dark, 'cone_i0': self.cone_i0,
            'n_epochs': self.n_epochs, 'n_contrasts': len(self.dark_contrasts),
            'bar_widths': ','.join(f'{b:g}' for b in self.bar_widths),
            'block_ids': ','.join(str(b) for b in self.block_ids),
            'units': self.units, 'series_resistance': self.series_resistance,
            'n_epochs_high_rs': self.n_epochs_high_rs,
            'mode_mismatch': self.mode_mismatch,
            'online_analysis_recorded': self.online_analysis_recorded or self.online_analysis,
            'aperture_diameter': self.config.get('apertureDiameter', np.nan),
            'center_spot': center_spot(self.config.get('apertureDiameter', np.nan)),
            'annulus_inner': self.config.get('annulusInnerDiameter', np.nan),
            'annulus_outer': self.config.get('annulusOuterDiameter', np.nan),
            'spot_intensity': self.config.get('spotIntensity', np.nan),
        }

    def describe(self) -> str:
        if not np.isfinite(self.series_resistance):
            rs = ''
        elif self.series_resistance > 0:
            rs = f'  Rs={self.series_resistance / 1e6:.1f} MOhm'
        elif self.online_analysis == 'extracellular':
            rs = '  Rs=0 (cell-attached)'
        else:
            # Whole-cell with a 0 reading: the field was never filled in, which
            # is not the same claim as "no access resistance".
            rs = '  Rs=0 (never set)'
        if self.n_epochs_high_rs:
            rs += f'  [{self.n_epochs_high_rs} epoch(s) dropped over the Rs cutoff]'
        mode = self.online_analysis
        if self.mode_mismatch:
            mode = f"{mode} (recorded as '{self.online_analysis_recorded}', " \
                   f'relabelled from the amplifier)'
        return (f'{self.exp_name} | {self.cell_type} | {self.cell_label} | '
                f'{mode} | grating {self.grating_site} | '
                f'FW={self.ndf} bg={self.background_intensity:.2f} ({self.light_level}) | '
                f'{self.n_epochs} epochs{rs}\n'
                f'  dark contrasts : {np.round(self.dark_contrasts, 3)}\n'
                f'  response       : {np.round(self.resp_mean, 3)}\n'
                f'  baseline={self.baseline_mean:.3f} | crossing nearest='
                f'{self.crossing_nearest:.3f} interp={self.crossing_interp:.3f} | '
                f'cone pred={self.cone_pred_dark:.3f}')


def _epoch_param(df_epochs: pd.DataFrame, name: str) -> np.ndarray:
    """Per-epoch parameter, whether StimBlock promoted it to a column or not.

    StimBlock only makes columns for parameters that vary within the block, so
    constants (often currentBarWidth / currentBrightContrast) live in the
    epoch_parameters dict.
    """
    if name in df_epochs.columns:
        return df_epochs[name].to_numpy(dtype=float)
    return np.array([float(p.get(name, np.nan)) for p in df_epochs['epoch_parameters']])


def analyze_group(exp_name: str, block_ids: Sequence[int], online_analysis: Optional[str] = None,
                  spike_th: float = DEFAULTS['spike_th'],
                  spike_offset: int = DEFAULTS['spike_offset'],
                  wc_offset: int = DEFAULTS['wc_offset'],
                  smooth_ms: float = DEFAULTS['smooth_ms'],
                  psth_sigma_ms: float = DEFAULTS['psth_sigma_ms'],
                  cone_i0: float = DEFAULTS['cone_i0'],
                  detector_kwargs: Optional[dict] = None,
                  drop_epochs: Sequence[int] = (),
                  max_series_resistance: Optional[float] = MAX_SERIES_RESISTANCE,
                  keep_raw: bool = False,
                  verbose: bool = True) -> GratingRecord:
    """Port of the per-node body of analyzeSpotAnnularGrating.m.

    Pools the given blocks (same cell, mode and light level), pools across bar
    width and grating polarity, and returns the response-vs-darkBarContrast
    tuning curve with the measured cancellation point and the cone prediction.

    Extracellular responses are firing rates in the stimulus window (Hz);
    whole-cell responses are the mean smoothed current in pA (box-car of
    ``smooth_ms``), with the sign flipped for 'exc' so larger always means a
    larger response.

    Every block is checked against the amplifier's ``seriesResistance`` before
    it is used, by the same :func:`resolve_recording_mode` rule
    :func:`check_series_resistance` applies — so a block handed straight to this
    function is corrected too, not only one that came through the block table.
    If the reading overrules ``onlineAnalysis``, the block is analyzed as it was
    actually recorded (spike sorted rather than averaged, or the reverse) and
    the record is marked ``mode_mismatch`` with the recorded label kept in
    ``online_analysis_recorded``.

    An epoch recorded through more than ``max_series_resistance`` ohms is
    dropped, per epoch, since the current is too filtered to trust; set it to
    ``None`` to keep them.

    ``keep_raw=True`` attaches the unprocessed amplifier traces to the record as
    ``rec.raw``, so :func:`plot_raw_blocks` and :func:`plot_raw_epochs` can draw
    the data underneath the summary without loading the blocks a second time.
    They are never written to the store.
    """
    import retinanalysis as ra
    from retinanalysis.utils.psth import psth_time_axis, spike_times_to_psth
    from scipy.ndimage import uniform_filter1d

    dark, bright, bar, pol, resp_stim, resp_base = [], [], [], [], [], []
    traces_all, first_params, used_blocks = [], None, []
    n_samples = None
    rs_kept, n_high_rs, mode_mismatch = [], 0, False
    raw = {'traces': [], 'spike_times_ms': [], 'block_id': [], 'sample_rate': None,
           'series_resistance': []} if keep_raw else None

    for bid in block_ids:
        sb = ra.StimBlock(exp_name, int(bid), verbose=False)
        ep = sb.df_epochs
        p0 = ep['epoch_parameters'].iloc[0]
        if first_params is None:
            first_params = p0
        recorded_mode = (online_analysis or p0.get('onlineAnalysis', 'extracellular')).lower()

        # spike_th is analyzeSpotAnnularGrating.m's paras.spikeTh, passed to
        # SpikeDetectorNew as thresholdSpikeFactor. detector_kwargs still wins.
        det = {'threshold_spike_factor': spike_th, **(detector_kwargs or {})}
        # Load the trace before deciding how to treat it: the recorded label may
        # be wrong, and resolving that needs the data as well as the reading.
        rb = ra.SCResponseBlock(exp_name, int(bid), b_spiking=False, verbose=False)
        sr = float(rb.amp_sample_rate)
        pre_pts = int(round(float(p0['preTime']) / 1e3 * sr))
        stim_pts = int(round(float(p0['stimTime']) / 1e3 * sr))

        keep = [i for i in range(len(ep)) if i not in set(drop_epochs)]

        # Access resistance, per epoch, straight from the h5. A missing reading
        # (no h5, old rig) leaves every epoch in: absent evidence is not
        # evidence of a bad recording.
        try:
            rs = read_series_resistance(exp_name, int(bid))
        except Exception as e:
            if verbose:
                print(f'  block {bid}: could not read series resistance '
                      f'({type(e).__name__}); no epochs excluded on that basis')
            rs = np.full(len(ep), np.nan)
        if rs.size < len(ep):
            rs = np.concatenate([rs, np.full(len(ep) - rs.size, np.nan)])

        # Applied whatever the label says: a non-zero reading means the cell was
        # held whole-cell, so the cutoff is about the recording, not the label.
        if max_series_resistance is not None:
            too_high = [i for i in keep
                        if np.isfinite(rs[i]) and rs[i] > max_series_resistance]
            if too_high:
                n_high_rs += len(too_high)
                keep = [i for i in keep if i not in set(too_high)]
                if verbose:
                    print(f'  block {bid}: dropped {len(too_high)}/{len(ep)} epoch(s) with '
                          f'series resistance above {max_series_resistance / 1e6:g} MOhm '
                          f'(up to {np.nanmax(rs[too_high]) / 1e6:.1f} MOhm)')
        if not keep:
            if verbose:
                print(f'  block {bid}: no epochs left after the series-resistance '
                      f'cutoff — block skipped')
            continue

        # The amplifier overrules the label, so the block is analyzed as it was
        # actually recorded rather than as it was tagged. Same rule as
        # check_series_resistance, applied here too so a block analyzed directly
        # is corrected as well.
        rs_median = float(np.nanmedian(rs[keep])) if np.isfinite(rs[keep]).any() else np.nan
        mode, note = resolve_recording_mode(recorded_mode, rs_median,
                                            amp_data=np.asarray(rb.amp_data)[keep],
                                            sample_rate=sr, detector_kwargs=det)
        spiking = mode == 'extracellular'
        if note:
            mode_mismatch = mode_mismatch or (mode != recorded_mode)
            if verbose:
                print(f'  block {bid}: {note}')
        if spiking:
            rb.get_spike_times(**det)
        rs_kept.extend(rs[i] for i in keep)
        used_blocks.append(int(bid))
        if keep_raw:
            raw['sample_rate'] = sr
            for i in keep:
                raw['traces'].append(np.asarray(rb.amp_data[i], dtype=float))
                raw['spike_times_ms'].append(
                    np.asarray(rb.spike_times[i], dtype=float) / sr * 1e3 if spiking else None)
                raw['block_id'].append(int(bid))
                raw['series_resistance'].append(float(rs[i]))

        if spiking:
            # Firing rate in the stimulus window, with a pre-stim baseline rate
            # for comparison. The MATLAB reports raw spike counts; dividing by
            # the window duration puts the tuning curve in Hz, on the same scale
            # as the PSTH traces above it. It is a constant factor within a
            # recording, so the crossing is unchanged.
            stim_s = stim_pts / sr
            pre_s = pre_pts / sr
            for i in keep:
                st = np.asarray(rb.spike_times[i], dtype=float)
                stim_n = np.sum((st > pre_pts + spike_offset)
                                & (st < pre_pts + stim_pts + spike_offset))
                base_n = np.sum(st < pre_pts)
                resp_stim.append(float(stim_n) / stim_s)
                resp_base.append(float(base_n) / pre_s if pre_pts else np.nan)
                traces_all.append(spike_times_to_psth(st / sr * 1000.0,
                                                      rb.amp_data.shape[1] / sr * 1000.0,
                                                      psth_sigma_ms, 1000.0))
            units = 'rate (Hz)'
        else:
            # Whole-cell: smooth with a smooth_ms box, subtract the pre-stim
            # mean, and take the mean current over the stimulus window. This
            # keeps the response in the recorded units (pA) rather than the
            # MATLAB's integrated charge (it multiplied by the stimulus
            # duration to get pA*s); the two differ only by that constant
            # factor, so the tuning-curve shape and the crossing are the same.
            sign = -1.0 if mode == 'exc' else 1.0
            width = max(int(round(smooth_ms / 1e3 * sr)), 1)
            data = uniform_filter1d(np.asarray(rb.amp_data, dtype=float), size=width, axis=1)
            data = data - data[:, :pre_pts].mean(axis=1, keepdims=True)
            lo, hi = pre_pts + wc_offset, min(pre_pts + stim_pts + wc_offset, data.shape[1])
            for i in keep:
                resp_stim.append(sign * float(data[i, lo:hi].mean()))
                resp_base.append(sign * float(data[i, :pre_pts].mean()))
                traces_all.append(sign * data[i])
            units = 'excitation (pA)' if mode == 'exc' else 'inhibition (pA)'

        dark.extend(_epoch_param(ep, 'currentDarkContrast')[keep])
        bright.extend(_epoch_param(ep, 'currentBrightContrast')[keep])
        bar.extend(_epoch_param(ep, 'currentBarWidth')[keep])
        pol.extend(_epoch_param(ep, 'currentGratingPolarity')[keep])
        n_samples = traces_all[-1].size

    if not used_blocks:
        raise ValueError(
            f'{exp_name} blocks {list(block_ids)}: every epoch was excluded by the '
            f'{max_series_resistance / 1e6:g} MOhm series-resistance cutoff')

    dark = np.asarray(dark); resp_stim = np.asarray(resp_stim); resp_base = np.asarray(resp_base)
    bright = np.asarray(bright); bar = np.asarray(bar)
    traces_all = np.vstack([t[:n_samples] for t in traces_all])

    valid = ~np.isnan(dark)
    if not valid.any():
        raise ValueError(f'{exp_name} blocks {list(block_ids)}: no currentDarkContrast values')

    # Pool across bar width and grating polarity; group by dark-bar contrast.
    contrasts = np.unique(dark[valid])
    resp_mean = np.array([np.nanmean(resp_stim[valid & (dark == c)]) for c in contrasts])
    resp_n = np.array([int(np.sum(valid & (dark == c))) for c in contrasts])
    resp_sem = np.array([
        (np.nanstd(resp_stim[valid & (dark == c)], ddof=0) / np.sqrt(max(n, 1)))
        for c, n in zip(contrasts, resp_n)])
    traces = np.vstack([traces_all[valid & (dark == c)].mean(axis=0) for c in contrasts])

    baseline_mean = float(np.nanmean(resp_base[valid]))
    baseline_sem = float(np.nanstd(resp_base[valid], ddof=0) / np.sqrt(max(valid.sum(), 1)))

    rel = resp_mean - baseline_mean
    crossing_nearest = float(contrasts[int(np.argmin(np.abs(rel)))])
    crossing_interp = interp_zero_crossing(contrasts, rel)

    bright_mode = float(pd.Series(bright[valid]).mode().iloc[0]) if valid.any() else np.nan
    ndf = float(first_params.get('NDF', np.nan))
    bg = float(first_params['backgroundIntensity'])
    rstar, light_label = light_level_rstar(ndf, bg, rig=rig_of(exp_name))

    sr = float(first_params['sampleRate'])
    trace_ms = (psth_time_axis(traces.shape[1], 1000.0) if 'Hz' in units
                else np.arange(traces.shape[1]) / sr * 1000.0)

    summary = ra.get_exp_summary(exp_name)
    row = summary[summary['block_id'].eq(int(used_blocks[0]))].iloc[0]

    if keep_raw:
        raw['dark'] = dark
        raw['pre_time_ms'] = float(first_params['preTime'])
        raw['stim_time_ms'] = float(first_params['stimTime'])
        raw['units'] = units
        raw['exp_name'] = exp_name

    rs_kept = np.asarray(rs_kept, dtype=float)
    rs_median = float(np.nanmedian(rs_kept)) if np.isfinite(rs_kept).any() else np.nan

    rec = GratingRecord(
        exp_name=exp_name, cell_label=str(row['cell_label']), cell_type=str(row['cell_type']),
        online_analysis=mode, grating_site=grating_site(first_params['annulusInnerDiameter']),
        ndf=ndf, background_intensity=bg, rstar=rstar, light_level=light_label,
        dark_contrasts=contrasts, resp_mean=resp_mean, resp_sem=resp_sem, resp_n=resp_n,
        baseline_mean=baseline_mean, baseline_sem=baseline_sem,
        crossing_nearest=crossing_nearest, crossing_interp=crossing_interp,
        bright_bar_contrast=bright_mode,
        cone_pred_dark=cone_predict_dark_contrast(rstar, bright_mode, cone_i0),
        cone_i0=cone_i0, bar_widths=np.unique(bar[valid & ~np.isnan(bar)]),
        traces=traces, trace_time_ms=trace_ms,
        pre_time_ms=float(first_params['preTime']), stim_time_ms=float(first_params['stimTime']),
        n_epochs=int(valid.sum()), block_ids=used_blocks,
        series_resistance=rs_median, n_epochs_high_rs=n_high_rs,
        mode_mismatch=mode_mismatch, online_analysis_recorded=recorded_mode,
        config={k: first_params.get(k) for k in CONFIG_KEYS}, units=units, raw=raw)
    if verbose:
        print(rec.describe())
    return rec


# --------------------------------------------------------------------------
# per-record store (the Python stand-in for spotAnnularGratingSummary.mat)
# --------------------------------------------------------------------------

def store_dir():
    """Where records live: ``<OUTPUT_DIR>/spot_annular_grating``."""
    from pathlib import Path
    from retinanalysis.config.settings import OUTPUT_DIR
    return Path(OUTPUT_DIR) / 'spot_annular_grating'


def describe_group_row(row, index: Optional[int] = None, total: Optional[int] = None) -> str:
    """State in words what one recording group is, before it is analyzed.

    Says which cell, what the stimulus actually was, and — the two things a
    row of the table does not make obvious — *why* it counts as center or
    surround, and *why* it is analyzed as spikes or as current. Both are
    derived rather than recorded, so both are spelled out with the number they
    came from.
    """
    def num(v, fmt='g'):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return '?'
        return '?' if np.isnan(v) else format(v, fmt)

    def get(name, default=None):
        try:
            value = row[name]
        except Exception:
            return default
        return default if value is None else value

    head = f"{get('exp_name')} {get('cell_label')} ({get('cell_type_short')})"
    if index is not None and total is not None:
        head = f'[{index:>3}/{total}] {head}'
    head += f"  —  {get('blocks', '?')} block(s), {get('epochs', '?')} epochs"

    inner, outer = get('annulus_inner'), get('annulus_outer')
    site = get('grating_site')
    why_site = ('inner diameter 0, so the grating covers the receptive-field centre'
                if str(site) == 'center' else
                f'inner diameter {num(inner)} > 0, so the grating is an annulus over the surround')
    site_line = (f"  {'grating over ' + str(site):<22s} annulus {num(inner)}-{num(outer)} µm; "
                 f'{why_site}')

    mode = str(get('onlineAnalysis', ''))
    rs = get('rs_mohm')
    rs_txt = '' if rs is None or (isinstance(rs, float) and np.isnan(rs)) else \
        (f', Rs {num(rs, ".2f")} MOhm' if float(rs) > 0 else ', Rs 0')
    recorded = str(get('onlineAnalysis_recorded', mode))
    corrected = f" (recorded as '{recorded}', relabelled from the amplifier)" \
        if recorded and recorded != mode else ''
    how = {'extracellular': 'spikes, so the response is a firing rate in Hz',
           'exc': 'whole-cell at the excitatory reversal, so the response is a current in pA',
           'inh': 'whole-cell at the inhibitory reversal, so the response is a current in pA',
           }.get(mode, 'unknown recording mode')
    mode_line = f"  {mode:<22s} {how}{rs_txt}{corrected}"

    stim = (f"background {num(get('backgroundIntensity'))}, "
            f"spot {num(get('spot_intensity'))} over a {num(get('aperture'))} µm aperture, "
            f"bright bars {num(get('bright'))}")
    stim_line = f"  {'stimulus':<22s} {stim}"

    ndfs = str(get('stage_ndfs', '') or 'none')
    rstar = get('rstar')
    calibrated = rstar is not None and not (isinstance(rstar, float) and np.isnan(rstar))
    rstar_txt = f" = {num(rstar)}R*" if calibrated else ' (no R* calibration yet)'
    light = (f"wheel NDF {num(get('filter_wheel_ndf'))} + fixed filters {ndfs}, "
             f"{get('light_setting', '?')}{rstar_txt}")
    light_line = f"  {'light':<22s} {light}"

    return '\n'.join([head, site_line, mode_line, stim_line, light_line])


def record_key(exp_name: str, cell_label: str, online_analysis: str, site: str,
               ndf: float, background_intensity: float) -> str:
    """Stable identifier for one recording group, safe as an HDF5 group name.

    Same key the MATLAB upserts on (date, cell, mode, FW, background), plus the
    grating site, since a cell can be recorded with the grating over the center
    and over the surround.
    """
    def num(v):
        return 'NaN' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:g}'.replace('.', 'p')
    return f'{exp_name}__{cell_label}__{online_analysis}__{site}__FW{num(ndf)}__bg{num(background_intensity)}'


_ARRAY_FIELDS = ('dark_contrasts', 'resp_mean', 'resp_sem', 'resp_n', 'bar_widths',
                 'traces', 'trace_time_ms')


def group_keys(groups: pd.DataFrame) -> List[str]:
    """The :func:`record_key` each row of a group table would be stored under.

    Note this reads ``onlineAnalysis``, which :func:`check_series_resistance`
    **overwrites** with the mode the amplifier resolved — so run that first, or
    a relabelled recording produces a different key here than the one its record
    was actually saved under.
    """
    return [record_key(r['exp_name'], r['cell_label'], r['onlineAnalysis'],
                       r['grating_site'], r['filter_wheel_ndf'], r['backgroundIntensity'])
            for _, r in groups.iterrows()]


def prune_records(keep, path=None, dry_run: bool = False,
                  verbose: bool = True) -> List[str]:
    """Delete stored records that are no longer in the analysis set.

    The store upserts and never deletes, so a record outlives the reason it was
    made: tighten a filter, change ``CELL_TYPES``, or let the amplifier relabel
    a block's recording mode, and the old record stays in ``records.h5`` and
    keeps turning up in :func:`load_summary` and every population figure. This
    is the verb that removes them.

    ``keep`` is the current analysis set — a group table (``selected``) or an
    iterable of keys. Anything stored under a key not in it is deleted.

    **Run :func:`check_series_resistance` before building ``keep`` from a group
    table.** It rewrites ``onlineAnalysis`` to the mode the amplifier resolved,
    and :func:`record_key` includes that mode, so skipping it makes every
    relabelled recording look like an orphan and deletes a live record.

    ``dry_run=True`` reports what would go without touching the file — worth
    doing first, since this is the one operation here that loses data. An empty
    ``keep`` raises rather than emptying the store.

    Returns the keys removed. HDF5 does not reclaim the freed space in place, so
    the file will not shrink until it is rewritten.
    """
    import h5py
    from pathlib import Path

    keep_keys = set(group_keys(keep) if isinstance(keep, pd.DataFrame) else keep)
    if not keep_keys:
        raise ValueError(
            'prune_records() refuses an empty keep set — that would delete every '
            'stored record. Pass the current group table (e.g. `selected`) or an '
            'explicit list of keys.')

    base = Path(path) if path is not None else store_dir()
    h5_path, csv_path = base / 'records.h5', base / 'summary.csv'
    if not h5_path.exists():
        if verbose:
            print('no store to prune')
        return []

    stored = load_summary(path=base, rstar=False)
    if stored.empty:
        return []
    orphans = stored[~stored['key'].isin(keep_keys)]
    if orphans.empty:
        if verbose:
            print(f'nothing to prune: all {len(stored)} stored record(s) are in the '
                  f'current set of {len(keep_keys)}')
        return []

    if verbose:
        verb = 'would remove' if dry_run else 'removing'
        print(f'{verb} {len(orphans)} stored record(s) no longer in the analysis set '
              f'({len(stored)} stored, {len(keep_keys)} current):')
        for _, r in orphans.sort_values(['cell_type', 'exp_name', 'cell_label']).iterrows():
            print(f"    {r['exp_name']} {r['cell_label']} "
                  f"{str(r.get('cell_type', '')).split(chr(92))[-1]} "
                  f"{r.get('online_analysis', '')} {r.get('grating_site', '')} "
                  f"— {int(r.get('n_epochs', 0))} epochs")
    if dry_run:
        return list(orphans['key'])

    removed = []
    with h5py.File(h5_path, 'a') as f:
        for key in orphans['key']:
            if key in f:
                del f[key]
                removed.append(key)
    summary = load_summary(path=base)
    summary.to_csv(csv_path, index=False)
    if verbose:
        print(f'{len(removed)} record(s) removed -> {len(summary)} rows remain')
    return removed


def save_records(records: Sequence[GratingRecord], path=None, verbose: bool = True,
                 prune_to=None):
    """Upsert records into ``<store>/records.h5`` and refresh ``summary.csv``.

    One HDF5 group per :func:`record_key`, overwritten if it already exists —
    the same upsert semantics as ``upsertSummary`` in the MATLAB, so re-running
    a cell replaces its row instead of duplicating it. ``summary.csv`` holds the
    scalar fields so population analysis can filter without opening the HDF5.

    ``prune_to`` additionally *deletes* any stored record outside that analysis
    set, via :func:`prune_records` — the store otherwise only ever grows. Leave
    it None (the default) when saving incrementally, as :func:`analyze_all` does
    per record: pruning against one record's worth of keys would delete the rest
    of the store.
    """
    import h5py
    from pathlib import Path

    base = Path(path) if path is not None else store_dir()
    base.mkdir(parents=True, exist_ok=True)
    h5_path, csv_path = base / 'records.h5', base / 'summary.csv'

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

    if prune_to is not None:
        prune_records(prune_to, path=base, verbose=verbose)

    summary = load_summary(path=base)
    summary.to_csv(csv_path, index=False)
    if verbose:
        print(f'{len(records)} record(s) saved -> {h5_path} ({len(summary)} rows total)')
    return h5_path


def refresh_rstar(summary: pd.DataFrame) -> pd.DataFrame:
    """Recompute ``rstar`` / ``light_level`` from the stored setting and the rig.

    What a record stores is the *setting* — filter-wheel NDF and background
    intensity — which is a fact about the experiment. R* is a fact about the
    rig, and it can be restated whenever the calibration is. Since the analysis
    never uses R* (the crossings come from the tuning curves), doing the
    conversion on read means a change to :data:`RIG_MAX_RSTAR` reaches records
    that were analyzed before it, with no re-analysis.

    Returns a copy with ``rstar``, ``rstar_level``, ``light_level`` and
    ``rstar_measured`` brought up to date.
    """
    if summary.empty or 'ndf' not in summary.columns:
        return summary
    out = summary.copy()
    # The rig is the letter at the end of the experiment name, so derive it and
    # let a stored value override only where it actually has one. Records saved
    # before summary_row() carried 'rig' have the column but leave it empty, and
    # trusting that blank dropped them to the RSTAR_TABLE fallback -- which is
    # rig G's numbers and covers five settings -- leaving most of them with no
    # light level at all.
    rigs = out['exp_name'].apply(rig_of)
    if 'rig' in out.columns:
        stored = out['rig']
        rigs = stored.where(stored.notna() & (stored.astype(str).str.strip() != ''), rigs)
    values = [light_level_rstar(n, b, rig=r)
              for n, b, r in zip(out['ndf'], out['background_intensity'], rigs)]
    out['rig'] = list(rigs)
    out['rstar'] = [v for v, _ in values]
    out['light_level'] = [lab for _, lab in values]
    out['rstar_level'] = [round_rstar(v) for v, _ in values]
    out['rstar_measured'] = [is_calibrated(n, b, rig=r)
                             for n, b, r in zip(out['ndf'], out['background_intensity'], rigs)]
    return out


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
    # Derived from a field records have always stored, so backfill it on read
    # rather than making every pre-existing record need re-analysis for it.
    if 'center_spot' not in out.columns and 'aperture_diameter' in out.columns:
        out['center_spot'] = out['aperture_diameter'].apply(center_spot)
    return refresh_rstar(out) if rstar else out


def load_records(keys: Optional[Sequence[str]] = None, path=None) -> Dict[str, Dict]:
    """Load full records (arrays + scalars) as ``{key: dict}``."""
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

def plot_group(rec: GratingRecord, figsize: Tuple[float, float] = (7.2, 7.6)):
    """Mean traces per dark-bar contrast (top) + contrast tuning curve (bottom).

    Mirrors plotMeanTraces / plotTuningCurve in the MATLAB.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    fig, (ax_t, ax_c) = plt.subplots(2, 1, figsize=figsize,
                                     gridspec_kw={'height_ratios': [1.15, 1.0], 'hspace': 0.34})
    colors = style.colors_for_conditions(list(rec.dark_contrasts))

    for c, tr in zip(rec.dark_contrasts, rec.traces):
        ax_t.plot(rec.trace_time_ms, tr, lw=1.2, color=colors[c], label=f'{c:g}')
    ax_t.axvspan(rec.pre_time_ms, rec.pre_time_ms + rec.stim_time_ms,
                 color='#000000', alpha=0.06, lw=0, zorder=0)
    ax_t.set_xlabel('Time (ms)')
    ax_t.set_ylabel('Rate (Hz)' if 'Hz' in rec.units else rec.units)
    ax_t.set_title(f'{rec.exp_name}  {rec.cell_label} ({rec.cell_type})  {rec.online_analysis}\n'
                   f'grating over {rec.grating_site}  |  FW={rec.ndf:g} '
                   f'bg={rec.background_intensity:g} ({rec.light_level})', fontsize=9)
    ax_t.legend(frameon=False, fontsize=6.5, ncol=2, title='dark contrast', title_fontsize=7)

    ax_c.errorbar(rec.dark_contrasts, rec.resp_mean, yerr=rec.resp_sem, fmt='o-',
                  ms=4, lw=1.4, color='#0072B2', ecolor='#0072B2', capsize=3,
                  label='response', zorder=3)
    ax_c.axhline(rec.baseline_mean, color='#666666', ls='--', lw=1.1, label='baseline')
    ax_c.axvline(rec.crossing_nearest, color='#D55E00', ls=':', lw=1.3, label='crossing')
    if np.isfinite(rec.cone_pred_dark):
        ax_c.axvline(rec.cone_pred_dark, color='#009E73', ls='-.', lw=1.3, label='conepred')
    ax_c.set_xlabel('Dark bar contrast')
    ax_c.set_ylabel(rec.units)
    ax_c.set_title(f'crossing {rec.crossing_nearest:.2f} (interp {rec.crossing_interp:.2f})'
                   f'   cone prediction {rec.cone_pred_dark:.2f}', fontsize=9)
    ax_c.legend(frameon=False, fontsize=8)
    return fig


# Marker per grating site and dash per recording mode, so two recordings that
# share a light level (hence a color) are still told apart.
_SITE_MARKERS = {'center': 'o', 'surround': 's'}
_MODE_STYLES = {'extracellular': '-', 'exc': '--', 'inh': ':'}


def _curve_fields(rec) -> Dict:
    """Pull the tuning curve off a GratingRecord or a stored record dict."""
    get = rec.get if isinstance(rec, dict) else lambda k, d=None: getattr(rec, k, d)
    return {
        'exp_name': str(get('exp_name')), 'cell_label': str(get('cell_label')),
        'online_analysis': str(get('online_analysis')),
        'grating_site': str(get('grating_site')),
        'light_level': str(get('light_level')), 'units': str(get('units', '')),
        'rstar': float(get('rstar', np.nan)),
        'crossing_interp': float(get('crossing_interp', np.nan)),
        'n_epochs': int(get('n_epochs', 0)),
        'dark_contrasts': np.asarray(get('dark_contrasts'), dtype=float),
        'resp_mean': np.asarray(get('resp_mean'), dtype=float),
        'resp_sem': np.asarray(get('resp_sem'), dtype=float),
        'baseline_mean': float(get('baseline_mean', np.nan)),
    }


def tuning_overlay(records: Sequence, ref_contrast: Optional[float] = None) -> pd.DataFrame:
    """Long-form table behind :func:`plot_tuning_overlay`, one row per point.

    Each record contributes ``resp_mean - baseline_mean`` — the response
    relative to its own pre-stimulus baseline, the curve whose zero crossing is
    the balancing contrast — in ``rel``, and that curve divided by
    ``|rel|`` at its most negative dark contrast in ``norm``.

    That reference point is the deepest dark bar the recording was run at, where
    the grating is furthest from cancelling and the response is largest, so it
    is the one contrast every recording of this protocol has in common. Dividing
    by it puts every curve at ±1 there and asks the only question worth asking
    across conditions: *where along the contrast axis does the response come
    back to baseline*, not how many Hz or pA it started from. The divisor is a
    magnitude, hence positive, so the crossing does not move.

    ``ref_contrast`` overrides the reference with the contrast nearest a given
    value (e.g. ``-1.0``); the default uses each record's own most negative,
    which matters when two recordings sampled different contrast ranges.

    ``ref_amplitude`` is the divisor, in recorded units — the amplitude the
    normalized curve is expressed as a fraction of.
    """
    rows = []
    for i, rec in enumerate(records):
        f = _curve_fields(rec)
        contrasts, rel = f['dark_contrasts'], f['resp_mean'] - f['baseline_mean']
        if contrasts.size == 0:
            continue
        ref_idx = (int(np.argmin(contrasts)) if ref_contrast is None
                   else int(np.argmin(np.abs(contrasts - float(ref_contrast)))))
        amp = float(abs(rel[ref_idx]))
        for c, v, s in zip(contrasts, rel, f['resp_sem']):
            rows.append({
                'position': i, 'cell': f"{f['exp_name']}/{f['cell_label']}",
                'online_analysis': f['online_analysis'], 'grating_site': f['grating_site'],
                'light_level': f['light_level'], 'rstar': f['rstar'], 'units': f['units'],
                'n_epochs': f['n_epochs'], 'crossing_interp': f['crossing_interp'],
                'dark_contrast': float(c), 'rel': float(v), 'sem': float(s),
                'ref_contrast': float(contrasts[ref_idx]), 'ref_amplitude': amp,
                'norm': float(v / amp) if np.isfinite(amp) and amp > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def plot_tuning_overlay(records: Sequence, labels: Optional[Sequence[str]] = None,
                        ref_contrast: Optional[float] = None,
                        figsize: Tuple[float, float] = (11.0, 4.6),
                        title: Optional[str] = None):
    """Several recordings' tuning curves on one pair of axes, raw and normalized.

    Built for comparing one cell across conditions — the rows of ``selected``
    for a single cell — where each curve is the same cell at a different light
    level, grating site or recording mode, and the question is how the
    cancellation point moves between them.

    Left: ``response - baseline`` in recorded units, one panel per unit. Two
    panels appear when the picks span both recording modes, because Hz and pA
    share no axis. Right: every curve normalized to its own amplitude at the
    most negative dark contrast (see :func:`tuning_overlay`), which is what
    makes modes comparable — it is one axes however many units are on the left.

    Color is the light level on the house sequential ramp (dim to bright), dash
    is the recording mode, marker is the grating site, so two recordings that
    coincide on one of those are still distinguishable. A triangle on the zero
    line marks each recording's interpolated crossing.

    The ramp is stretched across the levels *in this figure*, not across
    :data:`RSTAR_LEVELS`, so a cell recorded at two neighboring rungs still gets
    two clearly different colors — at the cost of a given R* not being the same
    color in another figure. The legend names each recording's setting, so read
    color within a figure only.

    ``labels`` overrides the legend text, one per record — pass the section-2
    row indices to tie a curve back to the table it came from.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    long = tuning_overlay(records, ref_contrast=ref_contrast)
    if long.empty:
        print('no tuning curves to overlay')
        return None

    # Color follows the light level, which is ordered, so it gets the sequential
    # ramp rather than categorical hues -- same convention as the population
    # figures. Recordings with no R* fall back to gray rather than dropping out.
    # Wider than the default 0.15-0.85 slice of the ramp: an overlay is a
    # handful of curves that often share mode and site, leaving color as the
    # only thing separating them, so the extra spread is worth the darker and
    # paler ends.
    levels = sorted(long.loc[long['rstar'].notna(), 'rstar'].map(round_rstar).unique())
    ramp = style.colors_for_conditions(levels, lo=0.05, hi=0.95)

    # Constrained rather than tight layout: the normalized axes spans every row
    # of the grid, which tight_layout cannot size.
    units_present = list(dict.fromkeys(long['units']))
    fig = plt.figure(figsize=figsize, layout='constrained')
    gs = fig.add_gridspec(len(units_present), 2)
    raw_axes = {u: fig.add_subplot(gs[i, 0]) for i, u in enumerate(units_present)}
    ax_norm = fig.add_subplot(gs[:, 1])

    for position, sub in long.groupby('position', sort=True):
        sub = sub.sort_values('dark_contrast')
        r = sub.iloc[0]
        color = (ramp.get(round_rstar(r['rstar']), '#888888')
                 if np.isfinite(r['rstar']) else '#888888')
        ls = _MODE_STYLES.get(r['online_analysis'], '-')
        marker = _SITE_MARKERS.get(r['grating_site'], 'D')
        label = (str(labels[int(position)]) if labels is not None
                 and int(position) < len(labels)
                 else f"{r['online_analysis']} · {r['grating_site']} · {r['light_level']}")

        ax_raw = raw_axes[r['units']]
        ax_raw.errorbar(sub['dark_contrast'], sub['rel'], yerr=sub['sem'], fmt=marker,
                        ls=ls, ms=4, lw=1.5, color=color, ecolor=color, capsize=2.5,
                        label=label, zorder=3)
        ax_norm.plot(sub['dark_contrast'], sub['norm'], marker=marker, ls=ls, ms=4,
                     lw=1.5, color=color, label=label, zorder=3)
        # Where this recording's response returns to baseline -- the number the
        # whole protocol is after, on the axis it is measured on.
        if np.isfinite(r['crossing_interp']):
            for ax in (ax_raw, ax_norm):
                ax.plot([r['crossing_interp']], [0.0], marker='v', ms=6, color=color,
                        mec='#333333', mew=0.5, zorder=4, clip_on=False)

    for units, ax in raw_axes.items():
        ax.axhline(0.0, color='#666666', ls='--', lw=1.0, zorder=1)
        ax.set_ylabel(f'response − baseline\n({units})')
        ax.legend(frameon=False, fontsize=7, loc='best')
    list(raw_axes.values())[-1].set_xlabel('dark bar contrast')
    list(raw_axes.values())[0].set_title('recorded units', fontsize=9)

    ax_norm.axhline(0.0, color='#666666', ls='--', lw=1.0, zorder=1)
    ref = long['ref_contrast'].unique()
    ref_txt = (f'{ref[0]:g}' if len(ref) == 1 else 'each curve’s own deepest')
    ax_norm.set_xlabel('dark bar contrast')
    ax_norm.set_ylabel(f'response − baseline,\nnormalized at contrast {ref_txt}')
    ax_norm.set_title('normalized' + (' — comparable across modes'
                                      if len(units_present) > 1 else ''), fontsize=9)
    ax_norm.legend(frameon=False, fontsize=7, loc='best')

    cells = list(dict.fromkeys(long['cell']))
    fig.suptitle(title if title is not None else
                 f"{', '.join(cells)} — {long['position'].nunique()} recordings"
                 + ('  (▾ = interpolated crossing)'), fontsize=10)
    return fig


def analyze_all(groups: pd.DataFrame, save: bool = True, plot: bool = False,
                on_error: str = 'log', verbose: bool = False,
                skip_existing: bool = False, status: bool = True,
                prune: bool = False, **kwargs) -> List[GratingRecord]:
    """Run :func:`analyze_group` over every row of :func:`group_blocks` output.

    ``status=True`` (the default) announces each recording before it is
    analyzed — which cell, which condition, and the stimulus parameters behind
    it, via :func:`describe_group_row` — then reports the crossing it produced.
    A batch of this length is otherwise a long silence, and the announcement is
    what makes it reviewable: it says why each recording counts as center or
    surround and why it is treated as spikes or current, both of which are
    derived rather than recorded. Set ``status=False`` for a quiet run.

    ``on_error='log'`` keeps the batch going past individual failures (a cell
    with unreadable data should not abort 100 others). ``skip_existing=True``
    leaves groups already in the store untouched, so re-running a notebook does
    not redo hours of spike detection; the stored records are still there for
    :func:`load_records`.

    ``prune=True`` deletes stored records outside ``groups`` once the batch is
    done (:func:`prune_records`), so the store ends up matching the analysis set
    instead of accumulating every record ever made. It prunes against ``groups``
    rather than against the records that just succeeded, so a group that failed
    this run keeps whatever it had — a failure should not delete data.
    """
    records, failures = [], []
    stored = set(load_summary()['key']) if skip_existing else set()
    skipped = 0
    total = len(groups)
    for position, (_, row) in enumerate(groups.iterrows(), start=1):
        if skip_existing:
            key = record_key(row['exp_name'], row['cell_label'], row['onlineAnalysis'],
                             row['grating_site'], row['filter_wheel_ndf'],
                             row['backgroundIntensity'])
            if key in stored:
                skipped += 1
                if status:
                    print(f"[{position:>3}/{total}] {row['exp_name']} {row['cell_label']} "
                          f'— already stored, skipped')
                continue
        block_ids = [int(b) for b in str(row['block_ids']).split(',')]
        if status:
            print(describe_group_row(row, index=position, total=total))
        try:
            rec = analyze_group(row['exp_name'], block_ids,
                                online_analysis=row['onlineAnalysis'], verbose=verbose, **kwargs)
            records.append(rec)
            if save:
                # Save as we go: a batch this long should survive an
                # interruption, and with skip_existing it can then resume.
                save_records([rec], verbose=False)
            if plot:
                plot_group(rec)
            if status:
                dropped = (f", {rec.n_epochs_high_rs} epoch(s) dropped over the Rs cutoff"
                           if rec.n_epochs_high_rs else '')
                print(f'  -> crossing {rec.crossing_nearest:.2f} '
                      f'(interp {rec.crossing_interp:.2f}) over '
                      f'{len(rec.dark_contrasts)} contrasts, {rec.n_epochs} epochs '
                      f'in {rec.units}{dropped}\n')
        except Exception as e:
            if on_error != 'log':
                raise
            failures.append((row['exp_name'], row['cell_label'], f'{type(e).__name__}: {e}'))
            if status:
                print(f'  -> FAILED: {type(e).__name__}: {str(e)[:110]}\n')
    # Against groups, not against `records`: a group that failed this run is
    # still part of the analysis set, and deleting its stored record because the
    # rerun crashed would turn a transient failure into data loss.
    if prune and save and len(groups):
        prune_records(groups, verbose=status)

    print(f'analyzed {len(records)}/{len(groups)} groups'
          + (f' ({skipped} already stored, skipped)' if skipped else ''))
    if failures:
        print(f'{len(failures)} failed:')
        for exp, cell, msg in failures[:20]:
            print(f'  {exp} {cell}: {msg[:110]}')
    return records


# --------------------------------------------------------------------------
# Weber comparison (port of populationSpotAnnularGrating.m, crossing figure)
# --------------------------------------------------------------------------

def weber_curve(rstar_grid, bright_contrast: float = 0.9,
                i0: float = DEFAULTS['cone_i0']) -> np.ndarray:
    """Predicted balancing dark contrast across light levels — the Weber curve."""
    return np.array([cone_predict_dark_contrast(r, bright_contrast, i0) for r in rstar_grid])


def plot_weber_comparison(summary: pd.DataFrame, i0: float = DEFAULTS['cone_i0'],
                          bright_contrast: Optional[float] = None,
                          figsize: Tuple[float, float] = (9.5, 6.6),
                          conditions: Sequence[str] = ('ON-parasol / surround',
                                                       'OFF-parasol / center'),
                          modes: Sequence[str] = ('extracellular', 'exc')):
    """Measured cancellation contrast vs light level, over the Weber prediction.

    One panel per (condition, recording mode): the Weber/Naka-Rushton curve
    ``R(I) = I/(I+I0)`` drawn continuously across R*, with each recording's
    measured crossing on top. Open markers mark light levels whose R* was
    extrapolated rather than measured (see :func:`light_level_rstar`).

    ``summary`` is :func:`load_summary` output with a ``condition`` column (add
    it with :func:`add_condition`). Records with no R* cannot be placed on the
    light-level axis and are reported rather than dropped silently — supply a
    calibration with :func:`apply_rstar_mapping` to include them.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    df = summary[summary['rstar'].notna()].copy()
    missing = len(summary) - len(df)
    if missing:
        settings = sorted(summary[summary['rstar'].isna()]['light_setting'].unique())
        print(f'{missing}/{len(summary)} records have no R* and are not plotted; '
              f"they need a calibration for: {', '.join(settings)}")
    if bright_contrast is None:
        bright_contrast = float(pd.Series(df['bright_bar_contrast']).mode().iloc[0]) \
            if len(df) else 0.9

    fig, axes = plt.subplots(len(conditions), len(modes), figsize=figsize,
                             squeeze=False, sharex=True, sharey=True)
    grid = np.logspace(np.log10(200), np.log10(60000), 200)
    curve = weber_curve(grid, bright_contrast, i0)

    for i, cond in enumerate(conditions):
        for j, mode in enumerate(modes):
            ax = axes[i][j]
            ax.plot(grid, curve, '-', color='#0072B2', lw=2, zorder=2,
                    label=f'Weber (I0={i0:g})')
            sub = df[df['condition'].eq(cond) & df['online_analysis'].eq(mode)]
            meas = sub[sub.get('rstar_measured', pd.Series(True, index=sub.index)).astype(bool)]
            est = sub.drop(meas.index)
            ax.scatter(meas['rstar'], meas['crossing_nearest'], s=34, color='#D55E00',
                       zorder=4, label=f'measured R* (n={len(meas)})')
            ax.scatter(est['rstar'], est['crossing_nearest'], s=34, facecolors='none',
                       edgecolors='#D55E00', zorder=3, label=f'estimated R* (n={len(est)})')
            ax.set_xscale('log')
            ax.set_title(f'{cond} — {mode}', fontsize=9)
            ax.legend(frameon=False, fontsize=7, loc='lower right')
            if i == len(conditions) - 1:
                ax.set_xlabel('light level (R*)')
            if j == 0:
                ax.set_ylabel('balancing dark contrast')
    fig.suptitle('Measured cancellation vs Weber cone prediction', fontsize=11, y=1.0)
    return fig


def add_condition(summary: pd.DataFrame) -> pd.DataFrame:
    """Attach the canonical ``condition`` label to a stored summary table."""
    short = summary['cell_type'].astype(str).str.split('\\').str[-1]
    keys = list(zip(short, summary['grating_site']))
    out = summary.copy()
    out['cell_type_short'] = short
    out['condition'] = [CANONICAL_CONDITIONS.get(k, 'other') for k in keys]
    return out


def population_tuning(summary: pd.DataFrame, records: Optional[Dict[str, Dict]] = None,
                      normalize: bool = True, min_contrasts: int = 3,
                      allowed_bright_contrast: Optional[Sequence[float]]
                      = ALLOWED_BRIGHT_CONTRAST) -> pd.DataFrame:
    """Mean response-vs-dark-contrast curve per light level, pooled over cells.

    The per-recording tuning curve is ``resp_mean - baseline_mean``: the
    response relative to that cell's own pre-stimulus baseline, which is the
    quantity whose zero crossing is the balancing contrast. Subtracting the
    baseline per cell is what makes cells poolable at all — the raw rate says as
    much about the cell's spontaneous activity as about the stimulus.

    ``normalize`` (default) then divides each cell's curve by its own peak
    ``|response - baseline|``. Two reasons it is the default:

    - Cells differ enormously in absolute response — in this dataset the
      excitatory-current records span 1.2 to 538 pA, so a raw mean is just the
      loudest cell with a little noise added. Firing rates are milder (16 to 92
      Hz) but still 6-fold.
    - It is a *positive* scalar, so it leaves every zero crossing exactly where
      it was. The balancing contrast — the thing being measured — is unchanged.

    Set ``normalize=False`` to keep the recorded units, which is only meaningful
    within one recording mode.

    A cell recorded more than once in the same (condition, mode, light level) is
    averaged to one curve before entering the population mean, so a cell with
    several blocks does not count several times. Recordings sampling fewer than
    ``min_contrasts`` dark contrasts are not tuning curves and are dropped.

    Returns one row per (condition, mode, light level, dark contrast) with the
    mean, its SEM across cells, and the cell count.
    """
    df = add_condition(summary) if 'condition' not in summary.columns else summary.copy()
    if allowed_bright_contrast is not None and 'bright_bar_contrast' in df.columns:
        df = df[df['bright_bar_contrast'].isin(list(allowed_bright_contrast))]
    if df.empty:
        return pd.DataFrame(columns=['condition', 'online_analysis', 'units', 'rstar_level',
                                     'dark_contrast', 'mean', 'sem', 'n_cells'])
    if records is None:
        records = load_records(list(df['key']))

    rows = []
    for _, r in df.iterrows():
        rec = records.get(r['key'])
        if rec is None:
            continue
        contrasts = np.asarray(rec['dark_contrasts'], dtype=float)
        if contrasts.size < min_contrasts:
            continue
        rel = np.asarray(rec['resp_mean'], dtype=float) - float(rec['baseline_mean'])
        if normalize:
            peak = np.nanmax(np.abs(rel))
            if not np.isfinite(peak) or peak == 0:
                continue
            rel = rel / peak
        for c, v in zip(contrasts, rel):
            rows.append({'condition': r['condition'], 'online_analysis': r['online_analysis'],
                         'units': 'normalized' if normalize else r.get('units', ''),
                         'rstar_level': r['rstar_level'],
                         'cell': f"{r['exp_name']}/{r['cell_label']}",
                         'dark_contrast': round(float(c), 4), 'value': float(v)})
    long = pd.DataFrame(rows)
    if long.empty:
        return long

    keys = ['condition', 'online_analysis', 'units', 'rstar_level', 'dark_contrast']
    per_cell = long.groupby(keys + ['cell'], dropna=False)['value'].mean().reset_index()
    out = (per_cell.groupby(keys, dropna=False)['value']
           .agg(mean='mean',
                sem=lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan,
                n_cells='size')
           .reset_index()
           .sort_values(keys))
    return out.reset_index(drop=True)


def plot_population_tuning(summary: pd.DataFrame, records: Optional[Dict[str, Dict]] = None,
                           normalize: bool = True, min_cells: int = 2,
                           conditions: Sequence[str] = ('ON-parasol / surround',
                                                        'OFF-parasol / center'),
                           modes: Sequence[str] = ('extracellular', 'exc'),
                           figsize: Tuple[float, float] = (10.0, 7.0),
                           **kwargs):
    """Population tuning curves, one line per light level, overlaid.

    One panel per (condition, recording mode); within a panel each light level
    is its own curve of mean response against dark-bar contrast, with a shaded
    SEM across cells. Under a Weber cone model the curve should shift as the
    mean light level changes, so overlaying the levels puts that shift on one
    pair of axes.

    Light level is an *ordered* quantity, so the curves are colored on the
    house sequential ramp (``cividis``, dim to bright) rather than categorical
    hues, and the mapping is built once across every level in ``summary`` — a
    given R* is the same color in every panel.

    ``min_cells`` hides a level whose mean rests on fewer than that many cells
    at every contrast; those points are still in :func:`population_tuning`.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    tuning = population_tuning(summary, records=records, normalize=normalize, **kwargs)
    if tuning.empty:
        print('no records with a tuning curve to plot')
        return None

    # Color follows the light level, not its rank within a panel, so build the
    # mapping once over every level present.
    levels = sorted(tuning['rstar_level'].dropna().unique())
    colors = style.colors_for_conditions(levels)

    fig, axes = plt.subplots(len(conditions), len(modes), figsize=figsize,
                             squeeze=False, sharex=True)
    for i, cond in enumerate(conditions):
        for j, mode in enumerate(modes):
            ax = axes[i][j]
            panel = tuning[tuning['condition'].eq(cond) & tuning['online_analysis'].eq(mode)]
            ax.axhline(0.0, color='#666666', ls='--', lw=1.0, zorder=1)
            if panel.empty:
                ax.text(0.5, 0.5, 'no recordings', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='#888888')
                ax.set_title(f'{cond} — {mode}', fontsize=9)
                continue
            drawn = 0
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
                drawn += 1
            units = panel['units'].iloc[0]
            ax.set_title(f'{cond} — {mode}', fontsize=9)
            if drawn:
                ax.legend(frameon=False, fontsize=7, title='light level',
                          title_fontsize=7, loc='best')
            if i == len(conditions) - 1:
                ax.set_xlabel('dark bar contrast')
            if j == 0:
                ax.set_ylabel(f'response − baseline\n({units})')
    fig.suptitle('Population tuning curves by light level'
                 + ('' if normalize else '  (recorded units — never pooled across modes)'),
                 fontsize=11)
    fig.tight_layout()
    return fig


def plot_condition_examples(records: Optional[Dict[str, Dict]] = None,
                            conditions: Sequence[str] = ('ON-parasol / surround',
                                                         'OFF-parasol / center'),
                            modes: Sequence[str] = ('extracellular', 'exc'),
                            i0: float = DEFAULTS['cone_i0'],
                            prefer_calibrated: bool = True,
                            figsize: Tuple[float, float] = (9.5, 6.6)):
    """One example tuning curve per (condition, recording mode).

    Reads the stored records by default (:func:`load_records`), so it works in a
    cold kernel with no DataJoint. Draws each example's tuning curve, baseline,
    measured crossing and — where the light level has an R* — the Weber
    prediction for that recording.

    ``prefer_calibrated`` picks the recording with the most epochs *among those
    with an R\\**, so the Weber line is shown wherever a calibration exists,
    falling back to the most epochs overall when none is calibrated.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if records is None or (hasattr(records, '__len__') and len(records) == 0):
        records = load_records()          # fall back to the store
    if isinstance(records, dict):
        rows = list(records.values())
    else:                                  # a list of GratingRecord from analyze_all
        rows = [{**r.summary_row(), 'dark_contrasts': r.dark_contrasts,
                 'resp_mean': r.resp_mean, 'resp_sem': r.resp_sem} for r in records]
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
            calibrated = [r for r in pool if np.isfinite(float(r['rstar']))]
            choose_from = calibrated if (prefer_calibrated and calibrated) else pool
            rec = max(choose_from, key=lambda r: float(r['n_epochs']))
            ax.errorbar(rec['dark_contrasts'], rec['resp_mean'], yerr=rec['resp_sem'],
                        fmt='o-', ms=4, lw=1.4, color='#0072B2', ecolor='#0072B2',
                        capsize=3, label='response')
            ax.axhline(float(rec['baseline_mean']), color='#666666', ls='--', lw=1.1,
                       label='baseline')
            ax.axvline(float(rec['crossing_nearest']), color='#D55E00', ls=':', lw=1.4,
                       label=f"nearest {float(rec['crossing_nearest']):.2f}")
            # The interpolated first crossing is steadier than "nearest to
            # baseline" once the curve saturates at baseline, where many tested
            # contrasts are equally close.
            if np.isfinite(float(rec['crossing_interp'])):
                ax.axvline(float(rec['crossing_interp']), color='#CC79A7', ls='--', lw=1.2,
                           label=f"interp {float(rec['crossing_interp']):.2f}")
            pred = cone_predict_dark_contrast(float(rec['rstar']),
                                              float(rec['bright_bar_contrast']), i0)
            if np.isfinite(pred):
                ax.axvline(pred, color='#009E73', ls='-.', lw=1.4, label=f'Weber {pred:.2f}')
            ax.set_title(f"{cond} — {mode}\n{rec['exp_name']} {rec['cell_label']} "
                         f"({rec['light_level']}, {int(rec['n_epochs'])} epochs)", fontsize=8)
            ax.set_xlabel('dark bar contrast')
            ax.set_ylabel(str(rec['units']))
            ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def plot_crossing_by_light_setting(summary: pd.DataFrame,
                                   conditions: Sequence[str] = ('ON-parasol / surround',
                                                                'OFF-parasol / center'),
                                   modes: Sequence[str] = ('extracellular', 'exc'),
                                   figsize: Tuple[float, float] = (9.5, 6.6)):
    """Measured cancellation contrast against the light level *as recorded*.

    Needs no R* calibration: the x axis is the (filter wheel, background) setting
    itself, so every record is plotted. Use this to see the data now, and
    :func:`plot_weber_comparison` once R* values exist
    (:func:`apply_rstar_mapping`).
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    settings = sorted(summary['light_setting'].unique())
    pos = {s: i for i, s in enumerate(settings)}

    fig, axes = plt.subplots(len(conditions), len(modes), figsize=figsize,
                             squeeze=False, sharex=True, sharey=True)
    rng = np.random.RandomState(0)
    for i, cond in enumerate(conditions):
        for j, mode in enumerate(modes):
            ax = axes[i][j]
            sub = summary[summary['condition'].eq(cond) & summary['online_analysis'].eq(mode)]
            for setting, grp in sub.groupby('light_setting'):
                x = pos[setting] + rng.uniform(-0.12, 0.12, len(grp))
                calibrated = bool(grp['rstar_measured'].astype(bool).iloc[0])
                ax.scatter(x, grp['crossing_nearest'], s=30, color='#D55E00',
                           facecolors='#D55E00' if calibrated else 'none',
                           edgecolors='#D55E00', zorder=3)
                ax.scatter([pos[setting]], [grp['crossing_nearest'].mean()], marker='_',
                           s=320, color='#0072B2', zorder=4)
            ax.set_title(f'{cond} — {mode} (n={len(sub)})', fontsize=9)
            ax.set_xticks(range(len(settings)))
            ax.set_xticklabels(settings, rotation=45, ha='right', fontsize=7)
            if j == 0:
                ax.set_ylabel('balancing dark contrast')
    fig.suptitle('Measured cancellation by light setting  '
                 '(filled = R* calibrated, open = not yet)', fontsize=10, y=1.0)
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


def describe_cell(cell: str, groups: pd.DataFrame, show: bool = True, **kwargs):
    """Basic information about one cell before analyzing any of its recordings.

    Cell type, how many conditions it was recorded in, and one row per
    condition. ``cell`` is '<experiment>/<cell label>'.
    """
    from retinanalysis.SCutils import explore as _sc
    return _sc.describe_cell(groups, cell, show=show, **kwargs)


def load_raw(exp_name, block_ids: Optional[Sequence[int]] = None,
             online_analysis: Optional[str] = None,
             detector_kwargs: Optional[dict] = None) -> Dict:
    """Every epoch's unprocessed amplifier trace for a recording, as loaded from the h5.

    ``SCResponseBlock`` stores ``amp_data`` exactly as the amplifier wrote it —
    the high-pass filter that precedes spike detection is applied inside the
    detector, not to this array — so these are the raw traces, with no
    filtering, smoothing or baseline subtraction.

    Pass a :class:`GratingRecord` that was built with ``analyze_group(keep_raw=True)``
    as ``exp_name`` and the traces it already holds are returned instead, so the
    blocks are not loaded a second time. Otherwise give ``exp_name`` and
    ``block_ids`` and they are loaded now.

    Returns a dict with ``traces`` (list of 1-D arrays, one per epoch),
    ``spike_times_ms`` (per epoch, ``None`` for whole-cell), ``dark`` (the
    epoch's ``currentDarkContrast``), ``block_id``, ``series_resistance``,
    ``sample_rate``, ``pre_time_ms``, ``stim_time_ms`` and ``units``.
    """
    import retinanalysis as ra

    if isinstance(exp_name, GratingRecord):
        if exp_name.raw is None:
            raise ValueError('this record has no raw traces; build it with '
                             'analyze_group(..., keep_raw=True) or pass '
                             'exp_name and block_ids instead')
        return exp_name.raw
    if block_ids is None:
        raise ValueError('block_ids is required unless a GratingRecord is passed')

    out = {'traces': [], 'spike_times_ms': [], 'dark': [], 'block_id': [],
           'series_resistance': [], 'sample_rate': None, 'exp_name': exp_name,
           'units': ''}
    for bid in block_ids:
        sb = ra.StimBlock(exp_name, int(bid), verbose=False)
        ep = sb.df_epochs
        p0 = ep['epoch_parameters'].iloc[0]
        recorded = (online_analysis or p0.get('onlineAnalysis', 'extracellular')).lower()
        rb = ra.SCResponseBlock(exp_name, int(bid), b_spiking=False, verbose=False)
        sr = float(rb.amp_sample_rate)
        try:
            rs = read_series_resistance(exp_name, int(bid))
        except Exception:
            rs = np.full(len(ep), np.nan)
        # Same relabelling rule as the analysis, so the raster appears for a
        # block that was recorded cell-attached whatever its label says.
        rs_median = float(np.nanmedian(rs)) if np.isfinite(rs).any() else np.nan
        mode, _ = resolve_recording_mode(recorded, rs_median, amp_data=rb.amp_data,
                                         sample_rate=sr, detector_kwargs=detector_kwargs)
        spiking = mode == 'extracellular'
        if spiking:
            rb.get_spike_times(**(detector_kwargs or {}))
        data = np.asarray(rb.amp_data, dtype=float)
        for i in range(data.shape[0]):
            out['traces'].append(data[i])
            out['spike_times_ms'].append(
                np.asarray(rb.spike_times[i], float) / sr * 1e3 if spiking else None)
            out['block_id'].append(int(bid))
            out['series_resistance'].append(float(rs[i]) if i < rs.size else np.nan)
        out['dark'].extend(_epoch_param(ep, 'currentDarkContrast'))
        out['sample_rate'] = sr
        out['pre_time_ms'] = float(p0['preTime'])
        out['stim_time_ms'] = float(p0['stimTime'])
        out['units'] = 'rate (Hz)' if spiking else 'pA'
    out['dark'] = np.asarray(out['dark'], dtype=float)
    return out


def plot_raw_epochs(exp_name, block_ids: Optional[Sequence[int]] = None,
                    online_analysis: Optional[str] = None,
                    max_contrasts: int = 6, max_epochs_per_contrast: int = 8,
                    detector_kwargs: Optional[dict] = None,
                    smooth: bool = True,
                    figsize: Tuple[float, float] = (10.0, 7.0)):
    """Raw data behind one recording: traces overlaid by contrast, and a spike raster.

    For a spiking recording the left column overlays the amplifier traces per
    dark-bar contrast and the right column is the raster across epochs, one row
    per epoch, with detected spikes marked against the same time base. For
    whole-cell there is no raster, so the traces are overlaid alone and
    ``smooth=True`` box-cars them (by ``DEFAULTS['smooth_ms']``) to make the
    current legible; ``smooth=False`` leaves them exactly as recorded.

    Accepts either a :class:`GratingRecord` built with ``keep_raw=True`` (no
    reload) or ``exp_name`` + ``block_ids``. Epochs are grouped by
    ``currentDarkContrast`` and at most ``max_contrasts`` of them are shown,
    spread across the tested range.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style
    from scipy.ndimage import uniform_filter1d

    style.apply_publication_style()
    raw = load_raw(exp_name, block_ids, online_analysis, detector_kwargs)
    exp_label = raw.get('exp_name', '')
    sr = float(raw['sample_rate'])
    pre_ms, stim_ms = float(raw['pre_time_ms']), float(raw['stim_time_ms'])
    spikes = raw['spike_times_ms']
    spiking = len(spikes) > 0 and spikes[0] is not None
    mode = 'extracellular' if spiking else 'whole cell'

    traces = [np.asarray(t, dtype=float) for t in raw['traces']]
    if not spiking and smooth:
        width = max(int(round(DEFAULTS['smooth_ms'] / 1e3 * sr)), 1)
        traces = [uniform_filter1d(t, size=width) for t in traces]

    darks = np.asarray(raw['dark'], dtype=float)
    contrasts = np.unique(darks[~np.isnan(darks)])
    if len(contrasts) > max_contrasts:                 # spread across the range
        contrasts = contrasts[np.linspace(0, len(contrasts) - 1, max_contrasts).astype(int)]

    n_col = 2 if spiking else 1
    fig, axes = plt.subplots(len(contrasts), n_col, figsize=figsize, squeeze=False,
                             sharex=True)
    # Blocks can differ in length, so the axis spans the longest of them and
    # each trace is drawn against as much of it as it has.
    t_ms = np.arange(max(len(t) for t in traces)) / sr * 1e3

    for r, c in enumerate(contrasts):
        idx = [i for i in np.flatnonzero(darks == c)][:max_epochs_per_contrast]
        ax = axes[r][0]
        for i in idx:
            ax.plot(t_ms[:len(traces[i])], traces[i], lw=0.5, alpha=0.7, color='#333333')
        ax.axvspan(pre_ms, pre_ms + stim_ms, color='#F0C000', alpha=0.15, lw=0, zorder=0)
        ax.set_ylabel(f'{c:g}\n({len(idx)} ep)', fontsize=7)
        if r == 0:
            smoothed = '' if spiking or not smooth else f" (box-car {DEFAULTS['smooth_ms']:g} ms)"
            ax.set_title(f'traces from the h5 — {exp_label} {mode}{smoothed}', fontsize=9)
        if spiking:
            ax_r = axes[r][1]
            for k, i in enumerate(idx):
                ax_r.eventplot(spikes[i], lineoffsets=k, linelengths=0.7, linewidths=0.8,
                               colors='#222222')
            ax_r.axvspan(pre_ms, pre_ms + stim_ms, color='#F0C000', alpha=0.15, lw=0, zorder=0)
            ax_r.set_ylim(-0.6, max(len(idx) - 0.4, 0.6))
            ax_r.invert_yaxis()
            ax_r.set_yticks([])
            if r == 0:
                ax_r.set_title('spike raster across epochs', fontsize=9)
    axes[-1][0].set_xlabel('Time (ms)')
    if spiking:
        axes[-1][1].set_xlabel('Time (ms)')
    fig.supylabel('dark bar contrast', fontsize=9)
    fig.tight_layout()
    return fig


def plot_raw_blocks(exp_name, block_ids: Optional[Sequence[int]] = None,
                    max_epochs: Optional[int] = None, show_mean: bool = True,
                    online_analysis: Optional[str] = None,
                    figsize: Optional[Tuple[float, float]] = None):
    """Raw amplifier traces, one panel per block, every epoch overlaid.

    This is the data before any processing: ``SCResponseBlock`` stores ``amp_data``
    exactly as recorded, and the high-pass filter that precedes spike detection
    is applied inside the detector, not to this array. So what is plotted here is
    what the amplifier wrote, with no filtering, smoothing or spike sorting.

    One panel per block, because a recording group pools several and they are
    where things go wrong independently — a cell lost partway, a seal drifting,
    a gain change between blocks. All epochs of a block are drawn over each
    other, with their mean on top by default. Each panel's title carries the
    block's series resistance, so a whole-cell block recorded through a bad
    electrode is visible next to its trace.

    Accepts either a :class:`GratingRecord` built with ``keep_raw=True`` (drawn
    from the traces it already holds, no reload) or ``exp_name`` + ``block_ids``.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    raw = load_raw(exp_name, block_ids, online_analysis)
    exp_label = raw.get('exp_name', '')
    sr = float(raw['sample_rate'])
    pre_ms, stim_ms = float(raw['pre_time_ms']), float(raw['stim_time_ms'])
    ep_block = np.asarray(raw['block_id'], dtype=int)
    ep_dark = np.asarray(raw['dark'], dtype=float)
    ep_rs = np.asarray(raw['series_resistance'], dtype=float)

    blocks = list(dict.fromkeys(ep_block.tolist()))
    if figsize is None:
        figsize = (8.6, 1.9 * len(blocks) + 0.6)
    fig, axes = plt.subplots(len(blocks), 1, figsize=figsize, squeeze=False, sharex=True)

    for ax, bid in zip(axes[:, 0], blocks):
        idx = np.flatnonzero(ep_block == bid)
        n = len(idx) if max_epochs is None else min(max_epochs, len(idx))
        shown = idx[:n]
        length = min(len(raw['traces'][i]) for i in shown)
        data = np.vstack([np.asarray(raw['traces'][i], dtype=float)[:length] for i in shown])
        t_ms = np.arange(length) / sr * 1e3

        for row in data:
            ax.plot(t_ms, row, lw=0.4, alpha=0.45, color='#666666')
        if show_mean:
            ax.plot(t_ms, data.mean(axis=0), lw=1.1, color='#D55E00', label='mean')
        ax.axvspan(pre_ms, pre_ms + stim_ms, color='#F0C000', alpha=0.15, lw=0, zorder=0)

        darks = np.unique(ep_dark[idx][~np.isnan(ep_dark[idx])])
        rs = np.nanmedian(ep_rs[idx]) if np.isfinite(ep_rs[idx]).any() else np.nan
        rs_txt = ('' if not np.isfinite(rs)
                  else ', Rs 0 (cell-attached)' if rs == 0
                  else f', Rs {rs / 1e6:.1f} MOhm')
        contrast_txt = (f', {len(darks)} dark contrasts '
                        f'({darks.min():g} to {darks.max():g})' if len(darks) else '')
        ax.set_ylabel('Amplitude', fontsize=8)
        ax.set_title(f'block {bid} — {n} of {len(idx)} epochs overlaid'
                     f'{contrast_txt}{rs_txt}', fontsize=8.5)
        if show_mean:
            ax.legend(frameon=False, fontsize=7, loc='upper right')
    axes[-1, 0].set_xlabel('Time (ms)')
    fig.suptitle(f'{exp_label} — raw traces per block, before filtering or spike detection',
                 fontsize=9.5, y=1.0)
    fig.tight_layout()
    return fig
