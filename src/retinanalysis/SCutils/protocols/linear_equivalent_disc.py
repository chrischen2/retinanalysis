"""Linear-equivalent disc / annulus with cone linearization: NLI analysis.

Python port of ``analyzeLinearDiscCone.m`` and ``populationLinConeDisc.m``
(linCone repo), reading from DataJoint + ``SCResponseBlock`` instead of
riekesuite. Shares the light-level and grouping helpers with
:mod:`spot_annular_grating` so the single-cell analyses stay consistent.

**The experiment.** Each natural-image patch is shown three ways, interleaved:

``image``
    the image patch itself;
``intensity``
    a uniform disc at the linear-equivalent intensity — the patch averaged over
    the receptive field with a Gaussian weight;
``linConeIntensity`` / ``lin cone intensity``
    a uniform disc at the *cone-linearized* equivalent intensity, which averages
    the patch after passing it through a Weber cone nonlinearity
    ``I / (I + WeberConstant)``.

Comparing image against each disc asks how much of the cell's preference for the
real image survives when the averaging is done in cone-response space rather
than in intensity space. The measure is the nonlinearity index, per patch::

    NLI = (image - disc) / (|image| + |disc|)

zeroed when neither response clears a recording-mode threshold (3 spikes for
extracellular, 10 for exc, 5 for inh), exactly as ``computeNLI`` does.

**Three protocols feed this analysis**, and one of them needs filtering:

===========================  ===================================================
``LinearEquivalentDiscConeLin``  disc over the centre; always has ``linearizeCones``
``LinearEquivalentAnnulus``      disc over the surround (ON parasols); always has it
``LinearEquivalentDisc``         **only** blocks that carry a ``linearizeCones``
                                 parameter — the same protocol name was reused for
                                 an older experiment without cone linearization,
                                 and those blocks are not this experiment
===========================  ===================================================

In the current database that filter keeps 58 of 421 ``LinearEquivalentDisc``
blocks; :func:`find_blocks` applies it and reports what it dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from retinanalysis.SCutils.protocols import spot_annular_grating as sag
from retinanalysis.SCutils.protocols.spot_annular_grating import (  # noqa: F401
    apply_rstar_mapping, is_calibrated, light_level_rstar, light_setting,
    read_filter_wheel_ndf,
)

# Protocol leaf names that feed this analysis. LinearEquivalentDisc is only
# included per-block, when the block carries a linearizeCones parameter.
PROTOCOLS = ('LinearEquivalentDiscConeLin', 'LinearEquivalentAnnulus',
             'LinearEquivalentDisc')
NEEDS_LINEARIZE_FILTER = ('LinearEquivalentDisc',)

# stimulusTag spellings: DiscConeLin writes camelCase, the other two use spaces.
TAG_IMAGE = 'image'
TAG_DISC = 'intensity'
TAG_CONE_DISC = ('linConeIntensity', 'lin cone intensity')

# nliThreshold() in the MATLAB: below this the responses are noise and the index
# is meaningless, so it is set to zero rather than dividing tiny by tiny.
NLI_THRESHOLD = {'extracellular': 3.0, 'exc': 10.0, 'inh': 5.0}

DEFAULTS = dict(
    psth_sigma_ms=10.0,
    wc_offset=100,        # samples, whole-cell response window offset
    spike_offset=300,     # samples, spiking response window offset
    smooth_ms=10.0,
)

CONFIG_KEYS = ['apertureDiameter', 'annulusInnerDiameter', 'annulusOuterDiameter',
               'backgroundIntensity', 'NDF', 'onlineAnalysis', 'linearizeCones',
               'WeberConstant', 'maxIntensity', 'rfSigmaCenter', 'rfSigmaSurround',
               'linearIntegrationFunction', 'currentImageSet', 'noPatches',
               'preTime', 'stimTime', 'tailTime', 'sampleRate', 'micronsPerPixel']


def _technique_for(exp_name: str, block_id: int) -> str:
    """How the cell was recorded, for blocks that left onlineAnalysis as 'none'."""
    import retinanalysis as ra
    summary = ra.get_exp_summary(exp_name)
    row = summary[summary['block_id'].eq(int(block_id))]
    return str(row['recording_technique'].iloc[0]) if len(row) else 'unknown'


def stimulus_site(protocol: str) -> str:
    """'surround' for the annulus protocol, 'centre' for the disc protocols."""
    return 'surround' if 'Annulus' in protocol else 'center'


def category_of(tag) -> str:
    """Map a stimulusTag to 'image' / 'disc' / 'cone_disc', or '' if unknown."""
    t = str(tag).strip()
    if t == TAG_IMAGE:
        return 'image'
    if t == TAG_DISC:
        return 'disc'
    if t in TAG_CONE_DISC or t.replace(' ', '').lower() == 'linconeintensity':
        return 'cone_disc'
    return ''


def compute_nli(image_mean, disc_mean, threshold: float) -> np.ndarray:
    """Port of ``computeNLI``: (image - disc) / (|image| + |disc|).

    Patches where neither response reaches ``threshold`` are set to zero (the
    index is meaningless for noise), and non-finite values are dropped.
    """
    img = np.asarray(image_mean, dtype=float)
    disc = np.asarray(disc_mean, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        nli = (img - disc) / (np.abs(img) + np.abs(disc))
    nli = np.where(np.maximum(np.abs(img), np.abs(disc)) < threshold, 0.0, nli)
    return nli[np.isfinite(nli)]


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

def find_blocks(exp_names: Optional[Sequence[str]] = None, show: bool = True,
                height: int = 420) -> pd.DataFrame:
    """Blocks from all three protocols, with the ``linearizeCones`` filter applied.

    ``LinearEquivalentDisc`` blocks without a ``linearizeCones`` parameter are the
    older, unrelated experiment and are dropped (reported in the output).
    """
    import retinanalysis as ra
    from retinanalysis.config import schema
    from retinanalysis.SCutils import explore as sc

    found = []
    for protocol in PROTOCOLS:
        b = sc.find_blocks(protocol, show=False)
        if b.empty:
            continue
        found.append(b[b['protocol'].eq(protocol)])      # exact leaf name
    if not found:
        return pd.DataFrame()
    blocks = pd.concat(found, ignore_index=True)
    if exp_names is not None:
        blocks = blocks[blocks['exp_name'].isin(exp_names)]

    meta = pd.concat([
        ra.get_exp_summary(exp)[['exp_name', 'block_id', 'cell_label', 'cell_type',
                                 'recording_technique', 'duration_minutes', 'start_time']]
        for exp in sorted(blocks['exp_name'].unique())])

    rows, dropped = [], []
    for _, blk in blocks.iterrows():
        bid = int(blk['block_id'])
        ep = (schema.Epoch() & f'parent_id={bid}').to_pandas()
        if ep.empty:
            continue
        ids = [int(i) for i in (ep['id'] if 'id' in ep else ep.index)]
        p = (schema.Epoch() & f'id={ids[0]}').fetch1('parameters')
        if blk['protocol'] in NEEDS_LINEARIZE_FILTER and 'linearizeCones' not in p:
            dropped.append(bid)
            continue
        row = {'block_id': bid, 'protocol': blk['protocol'], 'n_epochs': len(ids)}
        row.update({k: p.get(k, np.nan) for k in CONFIG_KEYS})
        rows.append(row)

    df = pd.DataFrame(rows).merge(blocks[['exp_name', 'block_id']], on='block_id')
    df = df.merge(meta, on=['exp_name', 'block_id'], how='left')
    df['site'] = df['protocol'].apply(stimulus_site)
    df['cell_type_short'] = df['cell_type'].astype(str).str.split('\\').str[-1]
    df = df.rename(columns={'NDF': 'filter_wheel_ndf'})
    df['has_filter_wheel'] = df['filter_wheel_ndf'].notna()
    df['light_setting'] = [light_setting(n, b) for n, b in
                           zip(df['filter_wheel_ndf'], df['backgroundIntensity'])]
    rs = [light_level_rstar(n, b) for n, b in
          zip(df['filter_wheel_ndf'], df['backgroundIntensity'])]
    df['rstar'] = [r for r, _ in rs]
    df['light_level'] = [lab for _, lab in rs]
    df = df.sort_values(['exp_name', 'cell_label', 'start_time']).reset_index(drop=True)

    if show:
        print(f"{len(df)} blocks | {df['exp_name'].nunique()} experiments | "
              f"{df.groupby(['exp_name', 'cell_label']).ngroups} cells")
        if dropped:
            print(f'  dropped {len(dropped)} LinearEquivalentDisc block(s) with no '
                  f'linearizeCones parameter (the older, unrelated protocol)')
        print('  by protocol: ' + ', '.join(f'{k} {v}' for k, v in
                                            df['protocol'].value_counts().items()))
        cols = ['exp_name', 'cell_label', 'cell_type_short', 'protocol', 'site',
                'onlineAnalysis', 'filter_wheel_ndf', 'backgroundIntensity',
                'light_setting', 'WeberConstant', 'n_epochs', 'block_id']
        sc.scroll_table(df[cols], height=height,
                        num_cols=('filter_wheel_ndf', 'backgroundIntensity',
                                  'WeberConstant', 'n_epochs', 'block_id'))
    return df


def group_blocks(df: pd.DataFrame, show: bool = True, height: int = 420) -> pd.DataFrame:
    """One row per recording group: experiment x cell x mode x site x light level."""
    from retinanalysis.SCutils import explore as sc

    keys = ['exp_name', 'cell_label', 'cell_type_short', 'onlineAnalysis', 'site',
            'filter_wheel_ndf', 'backgroundIntensity']
    g = (df.groupby(keys, dropna=False, sort=False)
           .agg(blocks=('block_id', 'size'), epochs=('n_epochs', 'sum'),
                protocols=('protocol', lambda s: ', '.join(sorted(set(s)))),
                light_setting=('light_setting', 'first'), rstar=('rstar', 'first'),
                light_level=('light_level', 'first'),
                weber=('WeberConstant', 'first'),
                block_ids=('block_id', lambda s: ', '.join(str(int(b)) for b in sorted(s))))
           .reset_index())
    if show:
        print(f'{len(g)} recording groups (experiment x cell x mode x site x light level)')
        sc.tree_table(g.sort_values(['cell_type_short', 'exp_name', 'cell_label']),
                      levels=['cell_type_short', 'exp_name', 'cell_label'], height=height,
                      num_cols=('blocks', 'epochs', 'filter_wheel_ndf',
                                'backgroundIntensity', 'weber', 'rstar'))
    return g


# --------------------------------------------------------------------------
# per-group analysis
# --------------------------------------------------------------------------

@dataclass
class DiscRecord:
    """One (experiment, cell, recording mode, site, light level)."""
    exp_name: str
    cell_label: str
    cell_type: str
    online_analysis: str
    site: str
    ndf: float
    background_intensity: float
    rstar: float
    light_setting: str
    weber_constant: float
    image_names: List[str]
    patch_ids: np.ndarray
    # Per-patch mean response, one row per patch, for each stimulus category.
    image_onset: np.ndarray
    image_offset: np.ndarray
    disc_onset: np.ndarray
    disc_offset: np.ndarray
    cone_onset: np.ndarray
    cone_offset: np.ndarray
    # Nonlinearity indices, image vs each disc.
    nli_disc_onset: np.ndarray
    nli_disc_offset: np.ndarray
    nli_cone_onset: np.ndarray
    nli_cone_offset: np.ndarray
    n_epochs: int
    n_patches: int
    threshold: float
    block_ids: List[int]
    config: Dict = field(default_factory=dict)
    units: str = ''

    @property
    def key(self) -> str:
        return record_key(self.exp_name, self.cell_label, self.online_analysis,
                          self.site, self.ndf, self.background_intensity)

    def summary_row(self) -> Dict:
        def m(x):
            x = np.asarray(x, dtype=float)
            return float(np.nanmean(x)) if x.size else np.nan
        return {
            'key': self.key, 'exp_name': self.exp_name, 'cell_label': self.cell_label,
            'cell_type': self.cell_type, 'online_analysis': self.online_analysis,
            'site': self.site, 'ndf': self.ndf,
            'background_intensity': self.background_intensity, 'rstar': self.rstar,
            'rstar_measured': is_calibrated(self.ndf, self.background_intensity),
            'light_setting': self.light_setting, 'weber_constant': self.weber_constant,
            'image_names': ','.join(self.image_names),
            'n_epochs': self.n_epochs, 'n_patches': self.n_patches,
            'threshold': self.threshold,
            'nli_disc_onset': m(self.nli_disc_onset),
            'nli_disc_offset': m(self.nli_disc_offset),
            'nli_cone_onset': m(self.nli_cone_onset),
            'nli_cone_offset': m(self.nli_cone_offset),
            'block_ids': ','.join(str(b) for b in self.block_ids), 'units': self.units,
        }

    def describe(self) -> str:
        def m(x):
            x = np.asarray(x, float)
            return np.nanmean(x) if x.size else np.nan
        return (f'{self.exp_name} | {self.cell_type} | {self.cell_label} | '
                f'{self.online_analysis} | disc over {self.site} | '
                f'{self.light_setting} | {self.n_patches} patches, {self.n_epochs} epochs\n'
                f'  NLI standard disc : onset {m(self.nli_disc_onset):+.3f}  '
                f'offset {m(self.nli_disc_offset):+.3f}\n'
                f'  NLI cone-lin disc : onset {m(self.nli_cone_onset):+.3f}  '
                f'offset {m(self.nli_cone_offset):+.3f}')


def analyze_group(exp_name: str, block_ids: Sequence[int],
                  online_analysis: Optional[str] = None,
                  spike_offset: int = DEFAULTS['spike_offset'],
                  wc_offset: int = DEFAULTS['wc_offset'],
                  smooth_ms: float = DEFAULTS['smooth_ms'],
                  detector_kwargs: Optional[dict] = None,
                  verbose: bool = True) -> DiscRecord:
    """Port of the per-node body of ``analyzeLinearDiscCone.m``.

    Measures each epoch's onset and offset response, averages within
    (image, patch, stimulus category), and forms the two nonlinearity indices.
    A patch is kept only if it has an image trial and at least one disc trial,
    matching the MATLAB.
    """
    import retinanalysis as ra
    from scipy.ndimage import uniform_filter1d

    per_epoch = []          # (image, patch, category, onset, offset)
    first_params, used_blocks = None, []

    for bid in block_ids:
        sb = ra.StimBlock(exp_name, int(bid), verbose=False)
        ep = sb.df_epochs
        params = list(ep['epoch_parameters'])
        p0 = params[0]
        if first_params is None:
            first_params = p0
        mode = (online_analysis or p0.get('onlineAnalysis', 'extracellular')).lower()
        if mode in ('', 'none', 'nan'):
            # Many blocks were recorded with onlineAnalysis left at 'none'; fall
            # back to how the cell was actually recorded. Whole-cell polarity is
            # then read off the data below, as the MATLAB does.
            technique = str(_technique_for(exp_name, int(bid)))
            mode = 'extracellular' if technique == 'cell-attached' else 'whole_cell'
        spiking = mode == 'extracellular'

        rb = ra.SCResponseBlock(exp_name, int(bid), b_spiking=spiking, verbose=False,
                                **(detector_kwargs or {}))
        sr = float(rb.amp_sample_rate)
        pre_pts = int(round(float(p0['preTime']) / 1e3 * sr))
        stim_pts = int(round(float(p0['stimTime']) / 1e3 * sr))
        n_samples = rb.amp_data.shape[1]
        used_blocks.append(int(bid))

        if spiking:
            onsets, offsets = [], []
            for st in rb.spike_times:
                st = np.asarray(st, dtype=float)
                on_lo, on_hi = pre_pts + spike_offset, pre_pts + stim_pts + spike_offset
                onsets.append(float(np.sum((st > on_lo) & (st < on_hi))))
                offsets.append(float(np.sum(st > on_hi)))
            units = 'spike count'
        else:
            if mode == 'whole_cell':
                # Excitatory currents are inward (negative); flip so a larger
                # number always means a larger response.
                mode = 'exc' if float(np.mean(rb.amp_data)) < 0 else 'inh'
            sign = -1.0 if mode == 'exc' else 1.0
            width = max(int(round(smooth_ms / 1e3 * sr)), 1)
            data = uniform_filter1d(np.asarray(rb.amp_data, dtype=float), size=width, axis=1)
            data = data - data[:, :pre_pts].mean(axis=1, keepdims=True)
            lo = pre_pts + wc_offset
            hi = min(pre_pts + stim_pts + wc_offset, n_samples)
            onsets = [sign * float(data[i, lo:hi].mean()) for i in range(data.shape[0])]
            offsets = [sign * float(data[i, hi:].mean()) if hi < n_samples else np.nan
                       for i in range(data.shape[0])]
            units = 'excitation (pA)' if mode == 'exc' else 'current (pA)'

        for i, p in enumerate(params):
            cat = category_of(p.get('stimulusTag'))
            if not cat or i >= len(onsets):
                continue
            per_epoch.append({'image': str(p.get('imageName')),
                              'patch': float(p.get('imagePatchIndex', np.nan)),
                              'category': cat, 'onset': onsets[i], 'offset': offsets[i]})

    df = pd.DataFrame(per_epoch)
    if df.empty:
        raise ValueError(f'{exp_name} blocks {list(block_ids)}: no usable epochs')

    # Mean response per (image, patch, category).
    means = df.groupby(['image', 'patch', 'category'])[['onset', 'offset']].mean()
    wide = means.unstack('category')

    def col(field, cat):
        return (wide[(field, cat)].to_numpy(dtype=float)
                if (field, cat) in wide.columns else np.full(len(wide), np.nan))

    image_on, image_off = col('onset', 'image'), col('offset', 'image')
    disc_on, disc_off = col('onset', 'disc'), col('offset', 'disc')
    cone_on, cone_off = col('onset', 'cone_disc'), col('offset', 'cone_disc')

    # Keep patches with an image trial and at least one disc trial.
    keep = np.isfinite(image_on) & (np.isfinite(disc_on) | np.isfinite(cone_on))
    idx = wide.index[keep]
    image_on, image_off = image_on[keep], image_off[keep]
    disc_on, disc_off = disc_on[keep], disc_off[keep]
    cone_on, cone_off = cone_on[keep], cone_off[keep]

    thresh = NLI_THRESHOLD.get(mode, 3.0)
    ndf = float(first_params.get('NDF', np.nan))
    bg = float(first_params['backgroundIntensity'])
    rstar, _ = light_level_rstar(ndf, bg)

    import retinanalysis as _ra
    summary = _ra.get_exp_summary(exp_name)
    row = summary[summary['block_id'].eq(int(used_blocks[0]))].iloc[0]

    rec = DiscRecord(
        exp_name=exp_name, cell_label=str(row['cell_label']), cell_type=str(row['cell_type']),
        online_analysis=mode, site=stimulus_site(str(row['protocol_name']).split('.')[-1]),
        ndf=ndf, background_intensity=bg, rstar=rstar,
        light_setting=light_setting(ndf, bg),
        weber_constant=float(first_params.get('WeberConstant', np.nan)),
        image_names=sorted({str(i) for i, _ in idx}),
        patch_ids=np.array([p for _, p in idx], dtype=float),
        image_onset=image_on, image_offset=image_off,
        disc_onset=disc_on, disc_offset=disc_off,
        cone_onset=cone_on, cone_offset=cone_off,
        nli_disc_onset=compute_nli(image_on, disc_on, thresh),
        nli_disc_offset=compute_nli(image_off, disc_off, thresh),
        nli_cone_onset=compute_nli(image_on, cone_on, thresh),
        nli_cone_offset=compute_nli(image_off, cone_off, thresh),
        n_epochs=int(len(df)), n_patches=int(keep.sum()), threshold=thresh,
        block_ids=used_blocks,
        config={k: first_params.get(k) for k in CONFIG_KEYS}, units=units)
    if verbose:
        print(rec.describe())
    return rec


# --------------------------------------------------------------------------
# record store
# --------------------------------------------------------------------------

def store_dir():
    """``<OUTPUT_DIR>/linear_equivalent_disc``."""
    from pathlib import Path
    from retinanalysis.config.settings import OUTPUT_DIR
    return Path(OUTPUT_DIR) / 'linear_equivalent_disc'


def record_key(exp_name: str, cell_label: str, online_analysis: str, site: str,
               ndf: float, background_intensity: float) -> str:
    def num(v):
        return ('NaN' if v is None or (isinstance(v, float) and np.isnan(v))
                else f'{v:g}'.replace('.', 'p'))
    return (f'{exp_name}__{cell_label}__{online_analysis}__{site}__'
            f'FW{num(ndf)}__bg{num(background_intensity)}')


_ARRAY_FIELDS = ('patch_ids', 'image_onset', 'image_offset', 'disc_onset', 'disc_offset',
                 'cone_onset', 'cone_offset', 'nli_disc_onset', 'nli_disc_offset',
                 'nli_cone_onset', 'nli_cone_offset')


def save_records(records: Sequence[DiscRecord], path=None, verbose: bool = True):
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
            g = f.create_group(rec.key)
            for name in _ARRAY_FIELDS:
                g.create_dataset(name, data=np.asarray(getattr(rec, name), dtype=float))
            for k, v in rec.summary_row().items():
                g.attrs[k] = '' if v is None else v
            for k, v in (rec.config or {}).items():
                if v is not None and not isinstance(v, (list, tuple, dict)):
                    g.attrs[f'cfg_{k}'] = v
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

def plot_group(rec: DiscRecord, figsize: Tuple[float, float] = (9.0, 4.2)):
    """Image vs disc response per patch, and the resulting NLI distributions.

    Left: the scatter the online figure shows — each patch's image response
    against its standard disc (grey) and cone-linearized disc (orange). Points
    above the unity line are patches where the real image drove the cell more
    than the equivalent uniform disc. Right: the NLI values those produce.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    fig, (ax_s, ax_n) = plt.subplots(1, 2, figsize=figsize)

    # Orientation follows analyzeLinearDiscCone.m: image on x, disc on y, so
    # points *below* unity are patches the image drove harder than the disc.
    ax_s.scatter(rec.image_onset, rec.disc_onset, s=26, color='#666666',
                 label='standard disc', zorder=3)
    ax_s.scatter(rec.image_onset, rec.cone_onset, s=26, color='#D55E00',
                 label='cone-linearized disc', zorder=3)
    lims = np.array([np.nanmin([rec.disc_onset.min(initial=0), rec.image_onset.min(initial=0)]),
                     np.nanmax([rec.disc_onset.max(initial=1), rec.image_onset.max(initial=1)])])
    ax_s.plot(lims, lims, '--', color='#000000', lw=1, zorder=2, label='unity')
    ax_s.set_xlabel(f'image response ({rec.units})')
    ax_s.set_ylabel(f'disc response ({rec.units})')
    ax_s.set_title(f'{rec.exp_name} {rec.cell_label} ({rec.cell_type})\n'
                   f'{rec.online_analysis} | disc over {rec.site} | {rec.light_setting} | '
                   f'{rec.n_patches} patches', fontsize=8.5)
    ax_s.legend(frameon=False, fontsize=7)

    data = [rec.nli_disc_onset, rec.nli_cone_onset,
            rec.nli_disc_offset, rec.nli_cone_offset]
    labels = ['disc\nonset', 'cone\nonset', 'disc\noffset', 'cone\noffset']
    colors = ['#666666', '#D55E00', '#666666', '#D55E00']
    for i, (d, c) in enumerate(zip(data, colors)):
        d = np.asarray(d, dtype=float)
        if d.size:
            ax_n.scatter(np.full(d.size, i) + np.random.uniform(-.09, .09, d.size), d,
                         s=14, color=c, alpha=0.6, lw=0)
            ax_n.scatter([i], [np.nanmean(d)], marker='_', s=340, color='#0072B2', zorder=4)
    ax_n.axhline(0, color='#000000', ls='--', lw=1)
    ax_n.set_xticks(range(4))
    ax_n.set_xticklabels(labels, fontsize=8)
    ax_n.set_ylabel('NLI  (image - disc) / (|image| + |disc|)')
    ax_n.set_title(f'nonlinearity index (threshold {rec.threshold:g})', fontsize=9)
    fig.tight_layout()
    return fig


def plot_population_nli(summary: Optional[pd.DataFrame] = None,
                        window: str = 'onset',
                        figsize: Tuple[float, float] = (9.0, 4.4)):
    """Per-cell paired NLI, standard disc vs cone-linearized disc.

    Section 1 of ``populationLinConeDisc.m``: one line per recording joining its
    mean standard-disc NLI to its mean cone-linearized NLI, one panel per
    recording mode, with a paired Wilcoxon signed-rank test.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if summary is None:
        summary = load_summary()
    a_col, b_col = f'nli_disc_{window}', f'nli_cone_{window}'
    modes = [m for m in ('extracellular', 'exc', 'inh') if m in set(summary['online_analysis'])]
    fig, axes = plt.subplots(1, max(len(modes), 1), figsize=figsize, squeeze=False)

    for ax, mode in zip(axes[0], modes):
        sub = summary[summary['online_analysis'].eq(mode)].dropna(subset=[a_col, b_col])
        for _, r in sub.iterrows():
            ax.plot([0, 1], [r[a_col], r[b_col]], '-', color='#999999', lw=0.9, alpha=0.8)
        ax.scatter(np.zeros(len(sub)), sub[a_col], s=26, color='#666666', zorder=3)
        ax.scatter(np.ones(len(sub)), sub[b_col], s=26, color='#D55E00', zorder=3)
        for x, col in ((0, a_col), (1, b_col)):
            ax.scatter([x], [sub[col].mean()], marker='_', s=420, color='#0072B2', zorder=4)
        stat = ''
        if len(sub) >= 3:
            from scipy.stats import wilcoxon
            try:
                stat = f'\nsigned-rank p = {wilcoxon(sub[a_col], sub[b_col]).pvalue:.3g}'
            except ValueError:
                stat = ''
        ax.axhline(0, color='#000000', ls='--', lw=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['standard\ndisc', 'cone-lin\ndisc'])
        ax.set_ylabel(f'mean NLI ({window})')
        ax.set_title(f'{mode} (n={len(sub)}){stat}', fontsize=9)
    fig.suptitle(f'Per-cell NLI, standard vs cone-linearized disc ({window})',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


def plot_nli_distributions(records: Optional[Dict[str, Dict]] = None,
                           window: str = 'onset',
                           figsize: Tuple[float, float] = (9.0, 4.2)):
    """Pooled per-patch NLI: histogram and CDF, standard vs cone-linearized.

    Section 2 of ``populationLinConeDisc.m`` — every patch from every recording,
    without per-cell averaging.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if records is None:
        records = load_records()
    disc = np.concatenate([np.asarray(r[f'nli_disc_{window}'], float).ravel()
                           for r in records.values()]) if records else np.array([])
    cone = np.concatenate([np.asarray(r[f'nli_cone_{window}'], float).ravel()
                           for r in records.values()]) if records else np.array([])
    disc, cone = disc[np.isfinite(disc)], cone[np.isfinite(cone)]

    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=figsize)
    bins = np.linspace(-1, 1, 41)
    for d, c, lab in ((disc, '#666666', f'standard disc (n={disc.size})'),
                      (cone, '#D55E00', f'cone-linearized (n={cone.size})')):
        if d.size:
            ax_h.hist(d, bins=bins, histtype='step', lw=1.6, color=c, label=lab)
            ax_c.plot(np.sort(d), np.linspace(0, 1, d.size), '-', lw=1.6, color=c, label=lab)
    for ax in (ax_h, ax_c):
        ax.axvline(0, color='#000000', ls='--', lw=1)
        ax.set_xlabel(f'NLI ({window})')
        ax.legend(frameon=False, fontsize=7)
    ax_h.set_ylabel('patches')
    ax_c.set_ylabel('cumulative fraction')
    if disc.size and cone.size:
        from scipy.stats import ks_2samp
        ax_c.set_title(f'KS p = {ks_2samp(disc, cone).pvalue:.3g}', fontsize=9)
    fig.suptitle(f'Pooled per-patch NLI ({window})', fontsize=11, y=1.0)
    fig.tight_layout()
    return fig


def analyze_all(groups: pd.DataFrame, save: bool = True, plot: bool = False,
                on_error: str = 'log', verbose: bool = False,
                skip_existing: bool = False, **kwargs) -> List[DiscRecord]:
    """Run :func:`analyze_group` over every row of :func:`group_blocks` output."""
    records, failures = [], []
    existing = load_summary()
    stored = set(existing['key']) if skip_existing and len(existing) else set()
    skipped = 0
    for _, row in groups.iterrows():
        if skip_existing:
            key = record_key(row['exp_name'], row['cell_label'], row['onlineAnalysis'],
                             row['site'], row['filter_wheel_ndf'], row['backgroundIntensity'])
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


# --------------------------------------------------------------------------
# stimulus rendering (natural image patch + the two discs)
# --------------------------------------------------------------------------

# NaturalImageFlashProtocol.m works in van Hateren pixels at this scale.
VH_MICRONS_PER_PIXEL = 6.6
VH_SHAPE = (1536, 1024)          # as MATLAB reads it: fread(fid, [1536 1024])


def vh_image_path(image_name: str, image_set: str = 'VHsubsample_20160105'):
    """Locate ``imk<name>.iml`` in the turner package resources, or None."""
    from pathlib import Path
    from retinanalysis.config.settings import PROTOCOL_REPOS_ROOT

    if not PROTOCOL_REPOS_ROOT:
        return None
    root = Path(PROTOCOL_REPOS_ROOT)
    name = f'imk{str(image_name).strip()}.iml'
    for cand in root.glob(f'**/+resources/{image_set}/{name}'):
        return cand
    hits = list(root.glob(f'**/{image_set}/{name}')) or list(root.glob(f'**/{name}'))
    return hits[0] if hits else None


def load_vh_contrast_image(image_name: str, image_set: str = 'VHsubsample_20160105'):
    """Load a van Hateren image the way the protocol does.

    Port of the loader in ``NaturalImageFlashProtocol.m``: big-endian uint16 read
    as a 1536x1024 matrix, rescaled so the brightest pixel is 1, then converted
    to contrast about the image mean. Returns ``(contrast_image, background)``
    with the image in MATLAB's (x, y) orientation so patch locations index it
    directly, or ``(None, nan)`` when the file is not on this machine.
    """
    path = vh_image_path(image_name, image_set)
    if path is None:
        return None, np.nan
    raw = np.fromfile(str(path), dtype='>u2')
    if raw.size < VH_SHAPE[0] * VH_SHAPE[1]:
        return None, np.nan
    # MATLAB fills [1536,1024] column-wise; the row-major read is its transpose.
    img = raw[:VH_SHAPE[0] * VH_SHAPE[1]].reshape(VH_SHAPE[1], VH_SHAPE[0]).T.astype(float)
    img = img / img.max()
    background = float(img.mean())
    return (img - background) / background, background


def image_patch(image_name: str, patch_location, aperture_diameter: float,
                image_set: str = 'VHsubsample_20160105', pad: float = 1.35,
                inner_diameter: float = 0.0):
    """The image patch a cell saw, in intensity units, plus its aperture mask.

    ``patch_location`` is ``currentPatchLocation`` (van Hateren pixels).
    ``inner_diameter`` > 0 makes the mask an annulus, which is what the
    LinearEquivalentAnnulus protocol shows. Returns
    ``(patch, mask, extent_um, background)``; ``patch`` is background *
    (1 + contrast) so it is directly comparable with the discs' equivalent
    intensities, and ``mask`` is True where the stimulus was visible.
    """
    contrast, background = load_vh_contrast_image(image_name, image_set)
    if contrast is None:
        return None, None, np.nan, np.nan
    x, y = (int(round(v)) for v in np.asarray(patch_location, dtype=float)[:2])
    half = int(round(aperture_diameter * pad / 2 / VH_MICRONS_PER_PIXEL))
    x0, x1 = max(x - half, 0), min(x + half, contrast.shape[0])
    y0, y1 = max(y - half, 0), min(y + half, contrast.shape[1])
    patch_contrast = contrast[x0:x1, y0:y1].T      # to (row, col) for imshow
    extent_um = half * VH_MICRONS_PER_PIXEL
    g = np.linspace(-extent_um, extent_um, patch_contrast.shape[1])
    h = np.linspace(-extent_um, extent_um, patch_contrast.shape[0])
    r = np.hypot(*np.meshgrid(g, h))
    mask = (r <= aperture_diameter / 2) & (r >= inner_diameter / 2)
    return background * (1 + patch_contrast), mask, extent_um, background


def plot_stimulus_example(params: Dict, patch_location=None,
                          equivalent_intensity: Optional[float] = None,
                          equivalent_intensity_cone: Optional[float] = None,
                          figsize: Tuple[float, float] = (10.0, 3.4)):
    """The three stimuli for one patch: image, standard disc, cone-linearized disc.

    ``params`` is an epoch-parameter dict from an ``image`` trial — it carries
    the patch location and both equivalent intensities. Everything is drawn on
    one grey scale so the discs can be compared with the patch they replace.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils import style

    style.apply_publication_style()
    if patch_location is None:
        patch_location = params.get('currentPatchLocation')
    if equivalent_intensity is None:
        equivalent_intensity = float(params.get('equivalentIntensity', np.nan))
    if equivalent_intensity_cone is None:
        equivalent_intensity_cone = float(params.get('equivalentIntensityConeLin', np.nan))
    # The annulus protocol has no apertureDiameter; its stimulus runs between
    # the annulus diameters instead.
    outer = params.get('apertureDiameter') or params.get('annulusOuterDiameter') or 200.0
    inner = float(params.get('annulusInnerDiameter') or 0.0)
    aperture = float(outer)

    patch, mask, extent, background = image_patch(
        str(params.get('imageName')), patch_location, aperture,
        str(params.get('currentImageSet', 'VHsubsample_20160105')),
        inner_diameter=inner)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    if patch is None:
        axes[0].text(0.5, 0.5, f"image imk{params.get('imageName')} not found\n"
                                'under PROTOCOL_REPOS_ROOT', ha='center', va='center',
                     transform=axes[0].transAxes, fontsize=8, color='#888888')
        axes[0].set_xticks([]); axes[0].set_yticks([])
        vmax = max(v for v in (equivalent_intensity, equivalent_intensity_cone, 1e-6)
                   if np.isfinite(v))
    else:
        shown = np.where(mask, patch, np.nan)
        vmax = float(np.nanmax([np.nanmax(shown), equivalent_intensity,
                                equivalent_intensity_cone]))
        axes[0].imshow(shown, cmap='gray', vmin=0, vmax=vmax, origin='lower',
                       extent=[-extent, extent, -extent, extent], interpolation='nearest')
        axes[0].set_xlabel('µm')
        axes[0].set_ylabel('µm')
    axes[0].set_title(f"image patch {params.get('imagePatchIndex')}\n"
                      f"imk{params.get('imageName')}", fontsize=9)

    disc_extent = extent if np.isfinite(extent) else aperture * 0.7
    for ax, value, name in ((axes[1], equivalent_intensity, 'linear-equivalent disc'),
                            (axes[2], equivalent_intensity_cone, 'cone-linearized disc')):
        g = np.linspace(-disc_extent, disc_extent, 301)
        r = np.hypot(*np.meshgrid(g, g))
        frame = np.full_like(r, np.nan)
        frame[(r <= aperture / 2) & (r >= inner / 2)] = value
        ax.imshow(frame, cmap='gray', vmin=0, vmax=vmax, origin='lower',
                  extent=[-disc_extent, disc_extent, -disc_extent, disc_extent],
                  interpolation='nearest')
        ax.set_title(f'{name}\nintensity {value:.4f}', fontsize=9)
        ax.set_xlabel('µm')
    geom = (f'annulus {inner:g}-{aperture:g} µm' if inner > 0
            else f'aperture {aperture:g} µm')
    fig.suptitle(f'{geom} | image mean {background:.4f}'
                 if np.isfinite(background) else geom, fontsize=9, y=1.02)
    fig.tight_layout()
    return fig


def example_patch_params(exp_name: str, block_id: int, patch_index: Optional[float] = None):
    """Epoch parameters of one ``image`` trial, for :func:`plot_stimulus_example`."""
    import retinanalysis as ra

    ep = ra.StimBlock(exp_name, int(block_id), verbose=False).df_epochs
    for p in ep['epoch_parameters']:
        if category_of(p.get('stimulusTag')) != 'image':
            continue
        if patch_index is None or float(p.get('imagePatchIndex', -1)) == float(patch_index):
            return p
    raise ValueError(f'no image trial for patch {patch_index} in block {block_id}')
