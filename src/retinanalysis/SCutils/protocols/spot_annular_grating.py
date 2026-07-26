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
by :func:`light_level_rstar`.
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

# (filter-wheel NDF, backgroundIntensity) -> mean photoisomerization rate (R*).
# Calibration table from lightLevelRStar() in the MATLAB.
RSTAR_TABLE: List[Tuple[float, float, float]] = [
    (0.0, 0.15, 12000.0),
    (1.0, 0.15, 1000.0),
    (1.0, 0.30, 2000.0),
    (0.5, 0.15, 4000.0),
    (0.5, 0.30, 8000.0),
]

# Epoch parameters that define a stimulus configuration / light level.
CONFIG_KEYS = ['apertureDiameter', 'annulusInnerDiameter', 'annulusOuterDiameter',
               'backgroundIntensity', 'spotIntensity', 'NDF', 'onlineAnalysis',
               'brightBarContrast', 'preTime', 'stimTime', 'tailTime', 'sampleRate',
               'micronsPerPixel']


# --------------------------------------------------------------------------
# light level / model
# --------------------------------------------------------------------------

def is_calibrated(ndf: float, background_intensity: float) -> bool:
    """True when (NDF, background) is an exact entry in :data:`RSTAR_TABLE`."""
    if ndf is None or np.isnan(ndf):
        return False
    return any(abs(ndf - fw) < 1e-6 and abs(background_intensity - bg) < 1e-6
               for fw, bg, _ in RSTAR_TABLE)


def light_setting(ndf: float, background_intensity: float) -> str:
    """The light level as recorded: ``'FW0/bg0.50'``.

    This is the raw, always-available description — filter-wheel NDF plus
    background intensity. Converting it to R* needs a calibration that belongs
    to the experimenter; see :func:`apply_rstar_mapping`.
    """
    fw = 'FW?' if ndf is None or np.isnan(ndf) else f'FW{ndf:g}'
    return f'{fw}/bg{background_intensity:g}'


def light_level_rstar(ndf: float, background_intensity: float) -> Tuple[float, str]:
    """Map (filter-wheel NDF, backgroundIntensity) to R*; port of lightLevelRStar.m.

    Returns ``(rstar, label)``, NaN for anything not in :data:`RSTAR_TABLE`.
    Nothing is estimated or interpolated here: an uncalibrated combination stays
    NaN and the label falls back to the raw setting, so a missing calibration is
    always visible rather than silently filled in. Supply your own conversion
    with :func:`apply_rstar_mapping`.
    """
    for fw, bg, rstar in RSTAR_TABLE:
        if (ndf is not None and not np.isnan(ndf) and abs(ndf - fw) < 1e-6
                and abs(background_intensity - bg) < 1e-6):
            return rstar, f'{rstar:g}R*'
    return np.nan, f'{light_setting(ndf, background_intensity)} (?R*)'


def apply_rstar_mapping(summary: pd.DataFrame,
                        mapping: Dict[Tuple[float, float], float]) -> pd.DataFrame:
    """Attach your own (filter wheel, background) -> R* calibration to a summary.

    The analysis never needs R* — crossings come from the tuning curves — so the
    conversion can be done at any time, on stored records, without re-running
    anything::

        summary = sag.apply_rstar_mapping(sag.load_summary(), {
            (0.0, 0.50): 40000, (0.0, 0.30): 24000, (1.0, 0.50): 3333,
        })
        sag.plot_weber_comparison(sag.add_condition(summary))

    Keys are ``(ndf, background_intensity)``; entries already covered by
    :data:`RSTAR_TABLE` keep their measured value unless the mapping overrides
    them. Returns a copy with ``rstar`` and ``light_level`` updated.
    """
    out = summary.copy()
    rstar = list(out['rstar'])
    label = list(out['light_level'])
    for i, (ndf, bg) in enumerate(zip(out['ndf'], out['background_intensity'])):
        for (m_ndf, m_bg), value in mapping.items():
            same_fw = (np.isnan(m_ndf) and pd.isna(ndf)) or np.isclose(m_ndf, ndf)
            if same_fw and np.isclose(m_bg, bg):
                rstar[i] = float(value)
                label[i] = f'{float(value):g}R*'
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


def grating_site(annulus_inner_diameter: float) -> str:
    """'center' when the grating mask starts at r=0, else 'surround'.

    The protocol masks the grating to inner/2 <= r <= outer/2, so an inner
    diameter of 0 puts the grating over the receptive-field center.
    """
    return 'center' if float(annulus_inner_diameter) == 0.0 else 'surround'


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
    fig.suptitle(
        f"grating over {site}  |  annulus {params['annulusInnerDiameter']:g}-"
        f"{params['annulusOuterDiameter']:g} µm  |  spot {params['apertureDiameter']:g} µm "
        f"@ {params['spotIntensity']:g}  |  bar {bar:g} µm  |  bg {params['backgroundIntensity']:g}"
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
    df['cell_type_short'] = df['cell_type'].astype(str).str.split('\\').str[-1]

    # NDF here is background:FilterWheel:NDF, lifted from the h5 by the parser
    # (verified equal to read_filter_wheel_ndf on every sampled block). A rig
    # with no filter wheel leaves it missing -- those blocks have no defined
    # light level and should not enter the Weber comparison.
    df = df.rename(columns={'NDF': 'filter_wheel_ndf'})
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
                'filter_wheel_ndf', 'backgroundIntensity', 'light_level', 'apertureDiameter',
                'annulusInnerDiameter', 'annulusOuterDiameter', 'n_epochs', 'block_id']
        print(f"{len(df)} blocks | {df['exp_name'].nunique()} experiments | "
              f"{df.groupby(['exp_name', 'cell_label']).ngroups} cells")
        missing = df[~df['has_filter_wheel']]
        if len(missing):
            print(f'  WARNING: {len(missing)} block(s) have no background:FilterWheel:NDF '
                  f'-- no light level, excluded from the Weber comparison. '
                  f"Experiments: {', '.join(sorted(missing['exp_name'].unique()))}")
        sc.scroll_table(df[cols], height=height,
                        num_cols=('n_epochs', 'block_id', 'filter_wheel_ndf',
                                  'backgroundIntensity'))
    return df


def group_blocks(df: pd.DataFrame, show: bool = True, height: int = 420,
                 require_filter_wheel: bool = True) -> pd.DataFrame:
    """Collapse the block table to one row per recording group.

    A group is the MATLAB epoch-tree leaf: (experiment, cell, recording mode,
    grating site, filter-wheel NDF, backgroundIntensity). Blocks within a group
    are pooled by :func:`analyze_group`.

    ``require_filter_wheel`` (default) drops blocks recorded on a rig with no
    filter wheel: without ``background:FilterWheel:NDF`` the light level is
    undefined, so those recordings cannot enter the Weber comparison. What was
    dropped is always reported.
    """
    from retinanalysis.SCutils import explore as sc

    if require_filter_wheel and 'has_filter_wheel' in df.columns:
        dropped = df[~df['has_filter_wheel']]
        if len(dropped):
            print(f'dropping {len(dropped)} block(s) with no filter-wheel NDF: '
                  f"{', '.join(sorted(dropped['exp_name'].unique()))}")
        df = df[df['has_filter_wheel']]

    keys = ['exp_name', 'cell_label', 'cell_type_short', 'onlineAnalysis',
            'grating_site', 'filter_wheel_ndf', 'backgroundIntensity']
    g = (df.groupby(keys, dropna=False, sort=False)
           .agg(blocks=('block_id', 'size'), epochs=('n_epochs', 'sum'),
                light_level=('light_level', 'first'), rstar=('rstar', 'first'),
                light_setting=('light_setting', 'first'),
                rstar_measured=('rstar_measured', 'first'),
                block_ids=('block_id', lambda s: ', '.join(str(int(b)) for b in sorted(s))))
           .reset_index())
    if show:
        print(f'{len(g)} recording groups '
              f'(experiment x cell x mode x grating site x filter wheel x background)')
        sc.tree_table(g.sort_values(['cell_type_short', 'exp_name', 'cell_label']),
                      levels=['cell_type_short', 'exp_name', 'cell_label'],
                      height=height, num_cols=('blocks', 'epochs', 'filter_wheel_ndf',
                                               'backgroundIntensity', 'rstar'))
    return g


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
        n_meas = int(out['rstar_measured'].sum()) if 'rstar_measured' in out else 0
        print(f"  with a measured light level: {n_meas}; "
              f"with any light level: {int(out['rstar'].notna().sum())}")
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
    config: Dict = field(default_factory=dict)
    units: str = ''

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
            'rstar_measured': is_calibrated(self.ndf, self.background_intensity),
            'light_setting': light_setting(self.ndf, self.background_intensity),
            'light_level': self.light_level, 'baseline_mean': self.baseline_mean,
            'baseline_sem': self.baseline_sem, 'crossing_nearest': self.crossing_nearest,
            'crossing_interp': self.crossing_interp,
            'bright_bar_contrast': self.bright_bar_contrast,
            'cone_pred_dark': self.cone_pred_dark, 'cone_i0': self.cone_i0,
            'n_epochs': self.n_epochs, 'n_contrasts': len(self.dark_contrasts),
            'bar_widths': ','.join(f'{b:g}' for b in self.bar_widths),
            'block_ids': ','.join(str(b) for b in self.block_ids),
            'units': self.units,
            'aperture_diameter': self.config.get('apertureDiameter', np.nan),
            'annulus_inner': self.config.get('annulusInnerDiameter', np.nan),
            'annulus_outer': self.config.get('annulusOuterDiameter', np.nan),
            'spot_intensity': self.config.get('spotIntensity', np.nan),
        }

    def describe(self) -> str:
        return (f'{self.exp_name} | {self.cell_type} | {self.cell_label} | '
                f'{self.online_analysis} | grating {self.grating_site} | '
                f'FW={self.ndf} bg={self.background_intensity:.2f} ({self.light_level}) | '
                f'{self.n_epochs} epochs\n'
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
                  verbose: bool = True) -> GratingRecord:
    """Port of the per-node body of analyzeSpotAnnularGrating.m.

    Pools the given blocks (same cell, mode and light level), pools across bar
    width and grating polarity, and returns the response-vs-darkBarContrast
    tuning curve with the measured cancellation point and the cone prediction.

    Extracellular responses are firing rates in the stimulus window (Hz);
    whole-cell responses are the mean smoothed current in pA (box-car of
    ``smooth_ms``), with the sign flipped for 'exc' so larger always means a
    larger response.
    """
    import retinanalysis as ra
    from retinanalysis.utils.psth import psth_time_axis, spike_times_to_psth
    from scipy.ndimage import uniform_filter1d

    dark, bright, bar, pol, resp_stim, resp_base = [], [], [], [], [], []
    traces_all, first_params, used_blocks = [], None, []
    n_samples = None

    for bid in block_ids:
        sb = ra.StimBlock(exp_name, int(bid), verbose=False)
        ep = sb.df_epochs
        p0 = ep['epoch_parameters'].iloc[0]
        if first_params is None:
            first_params = p0
        mode = (online_analysis or p0.get('onlineAnalysis', 'extracellular')).lower()
        spiking = mode == 'extracellular'

        rb = ra.SCResponseBlock(exp_name, int(bid), b_spiking=spiking, verbose=False,
                                **(detector_kwargs or {}))
        sr = float(rb.amp_sample_rate)
        pre_pts = int(round(float(p0['preTime']) / 1e3 * sr))
        stim_pts = int(round(float(p0['stimTime']) / 1e3 * sr))

        keep = [i for i in range(len(ep)) if i not in set(drop_epochs)]
        used_blocks.append(int(bid))

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
    rstar, light_label = light_level_rstar(ndf, bg)

    sr = float(first_params['sampleRate'])
    trace_ms = (psth_time_axis(traces.shape[1], 1000.0) if 'Hz' in units
                else np.arange(traces.shape[1]) / sr * 1000.0)

    summary = ra.get_exp_summary(exp_name)
    row = summary[summary['block_id'].eq(int(used_blocks[0]))].iloc[0]

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
        config={k: first_params.get(k) for k in CONFIG_KEYS}, units=units)
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


def save_records(records: Sequence[GratingRecord], path=None, verbose: bool = True):
    """Upsert records into ``<store>/records.h5`` and refresh ``summary.csv``.

    One HDF5 group per :func:`record_key`, overwritten if it already exists —
    the same upsert semantics as ``upsertSummary`` in the MATLAB, so re-running
    a cell replaces its row instead of duplicating it. ``summary.csv`` holds the
    scalar fields so population analysis can filter without opening the HDF5.
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

    summary = load_summary(path=base)
    summary.to_csv(csv_path, index=False)
    if verbose:
        print(f'{len(records)} record(s) saved -> {h5_path} ({len(summary)} rows total)')
    return h5_path


def load_summary(path=None) -> pd.DataFrame:
    """Scalar fields for every stored record — the population-analysis index."""
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
    return pd.DataFrame(rows).sort_values(['cell_type', 'exp_name', 'cell_label'],
                                          ignore_index=True) if rows else pd.DataFrame()


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


def analyze_all(groups: pd.DataFrame, save: bool = True, plot: bool = False,
                on_error: str = 'log', verbose: bool = False,
                skip_existing: bool = False, **kwargs) -> List[GratingRecord]:
    """Run :func:`analyze_group` over every row of :func:`group_blocks` output.

    ``on_error='log'`` keeps the batch going past individual failures (a cell
    with unreadable data should not abort 100 others). ``skip_existing=True``
    leaves groups already in the store untouched, so re-running a notebook does
    not redo hours of spike detection; the stored records are still there for
    :func:`load_records`.
    """
    records, failures = [], []
    stored = set(load_summary()['key']) if skip_existing else set()
    skipped = 0
    for _, row in groups.iterrows():
        if skip_existing:
            key = record_key(row['exp_name'], row['cell_label'], row['onlineAnalysis'],
                             row['grating_site'], row['filter_wheel_ndf'],
                             row['backgroundIntensity'])
            if key in stored:
                skipped += 1
                continue
        block_ids = [int(b) for b in str(row['block_ids']).split(',')]
        try:
            rec = analyze_group(row['exp_name'], block_ids,
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
