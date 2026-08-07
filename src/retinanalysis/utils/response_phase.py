"""Response phase to a drifting grating, and what it says about registration.

A drifting grating gives every point on the display its own temporal phase:
luminance at a point on the axis across the bars is ``mean · (1 + contrast ·
sin(2π f a + 2π F t))``, so two cells half a spatial period apart are driven in
antiphase. A cell that follows the stimulus therefore has a firing phase that
is *predicted by where its receptive field sits* — one full cycle of response
phase per spatial period of grating, along the drift axis and nowhere else.

That prediction has no free parameter, which makes it useful twice over:

**As a check on the overlay.** The receptive fields and the reconstructed
stimulus co-register by construction (see
:mod:`retinanalysis.utils.stimulus_frame`), but "by construction" is an
argument, not a measurement. The phase gradient measures it: scanning candidate
periods and orientations and asking which one makes the residual phases agree
recovers the grating's spatial period and its orientation from spike times
alone. If what comes back is the period the epoch actually ran, the mosaic and
the stimulus are in the same frame, and a spatial offset between them would
have to be smaller than the phase scatter allows.

**As a measurement in its own right.** Whether the response phase tracks
position at all is the question of whether the population resolves the grating.
Bars finer than a receptive field cancel within it, and then there is no F1 to
have a phase — the concentration collapses and no candidate period fits better
than any other. Comparing bar widths in the same block is comparing resolvable
against unresolvable, which is what a block that alternates them is for.

The phase convention here is the one the reconstruction uses: a spike at
stimulus time ``t`` contributes ``exp(i·2πF·t)``, so a hypothetical
zero-latency ON cell fires at the luminance peak and lands on
``stim_phase(a) = π/2 − 2π f a``. Everything downstream is stated as the
**residual** ``resp_phase − stim_phase``, which is constant across cells when
the prediction holds — its value is the cell's response latency (as a phase; a
2 Hz drift wraps every 500 ms) and OFF cells sit half a cycle from ON ones.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


__all__ = [
    'with_drift_freq',
    'drift_phase_response',
    'phase_period_scan',
    'phase_alignment_by_condition',
    'residual_latency_table',
    'describe_phase_alignment',
    'plot_phase_alignment',
    'browse_phase_alignment',
]


def _wrap_pi(angle):
    """Wrap to (-π, π]."""
    return (np.asarray(angle, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def _axis_position(center_x, center_y, geometry: Dict,
                   orientation_deg: Optional[float] = None) -> np.ndarray:
    """Signed distance of each RF center from the canvas center, across the bars.

    Same projection :func:`retinanalysis.utils.stimulus_frame.project_on_axis`
    applies to the canvas, evaluated at the cells instead of at every pixel.
    """
    theta = np.deg2rad(float(geometry['orientation_deg']
                             if orientation_deg is None else orientation_deg))
    dx = np.asarray(center_x, dtype=float) - float(geometry['center_x'])
    dy = np.asarray(center_y, dtype=float) - float(geometry['center_y'])
    return dx * np.cos(theta) + dy * np.sin(theta)


def with_drift_freq(geometry: Dict, drift_freq_hz: Optional[float]) -> Dict:
    """A copy of ``geometry`` folding at ``drift_freq_hz`` instead of the nominal.

    The temporal frequency reaches every phase calculation through the
    geometry dict, so replacing it there is enough to correct all of them at
    once — the F1 sums in :func:`drift_phase_response` and the cycle length
    :func:`residual_latency_table` divides by.

    Worth correcting because the declared frequency is usually not the
    delivered one: Stage advances the grating one increment per rendered frame
    sized from the *declared* refresh rate, and the display runs at its own.
    :func:`~retinanalysis.utils.spatial_recovery.estimate_drift_frequency`
    recovers the real value from the spikes. ``None`` returns the geometry
    unchanged.
    """
    if drift_freq_hz is None:
        return geometry
    out = dict(geometry)
    out['nominal_temporal_freq_hz'] = float(geometry['temporal_freq_hz'])
    out['temporal_freq_hz'] = float(drift_freq_hz)
    return out


def drift_phase_response(pipeline, epoch_indices, geometry: Dict,
                         window_s: Optional[Tuple[float, float]] = None,
                         cell_types: Optional[Iterable[str]] = None,
                         cell_ids: Optional[Iterable[int]] = None,
                         min_spikes: int = 30,
                         std_scaling: float = 1.6,
                         drift_freq_hz: Optional[float] = None):
    """Each cell's F1 phase, and the phase the grating predicts at its position.

    Parameters
    ----------
    pipeline : MEAPipeline
        Supplies the spike trains and the receptive fields, joined by the
        cluster match the way :func:`retinanalysis.utils.mosaic_overlay
        .cell_activity_in_window` joins them.
    epoch_indices : int or sequence[int]
        Block positions of the epochs to pool. **Pool only epochs that ran the
        same geometry** — the drift phase is measured from stimulus onset, so
        epochs of one condition are all in register with each other and their
        F1 vectors add, but epochs of a different bar width are not.
    geometry : dict
        Output of :func:`retinanalysis.regen.variable_mean_drifting_grating
        .grating_geometry` for one of those epochs. Read for the drift
        frequency, the spatial frequency, the orientation, the aperture and
        ``pre_time_ms``.
    window_s : (start, end), optional
        Seconds from the epoch start. Default is the whole stimulus, from
        ``pre_time_ms`` to the end of ``stim_time_ms`` — the phase estimate is
        a vector average over cycles, so more of them is better and there is
        no reason to shorten it unless the block drifts.
    cell_types, cell_ids : optional
        The usual restrictions; pass the QC survivors as ``cell_ids``.
    min_spikes : int
        A cell needs this many spikes pooled across the epochs for its phase
        to mean anything. Below it, phase and strength are NaN — kept as rows
        rather than dropped, so the caller can say how many were too quiet.
    std_scaling : float
        Receptive-field ellipse size in σ units, as elsewhere.
    drift_freq_hz : float, optional
        Fold at this frequency instead of the geometry's nominal one — see
        :func:`with_drift_freq`. On a 60 s epoch a fraction of a percent of
        frequency error costs most of the measured F1, so this is worth
        supplying rather than trusting the declared value.

    Returns
    -------
    pandas.DataFrame
        One row per cell: ``cell_id``, ``noise_id``, ``cell_type``,
        ``center_x``/``center_y``/``width``/``height``/``angle`` (canvas
        pixels), ``n_spikes``, ``rate_hz``, ``axis_px`` (position across the
        bars, from the canvas center), ``radius_px`` and ``inside_aperture``,
        ``f1_strength`` (vector strength, 0–1), ``rayleigh_p``,
        ``resp_phase_rad``, ``stim_phase_rad`` and ``residual_rad``.
    """
    import pandas as pd

    from retinanalysis.utils.mosaic_overlay import cell_activity_in_window

    epochs = ([int(epoch_indices)] if np.isscalar(epoch_indices)
              else [int(e) for e in epoch_indices])
    if not epochs:
        raise ValueError('epoch_indices is empty')

    geometry = with_drift_freq(geometry, drift_freq_hz)
    pre_s = float(geometry.get('pre_time_ms', 0.0)) / 1000.0
    if window_s is None:
        window_s = (pre_s, pre_s + float(geometry['stim_time_ms']) / 1000.0)
    t0, t1 = (float(w) for w in window_s)
    if not t1 > t0:
        raise ValueError(f'window_s must be (start, end) with end > start; '
                         f'got {window_s}')

    freq = float(geometry['temporal_freq_hz'])

    # Pool by summing each epoch's F1 vector. Phase is measured from stimulus
    # onset, which every epoch shares, so vectors from different epochs of the
    # same condition are directly additive.
    pooled: Dict[int, Dict] = {}
    for epoch in epochs:
        table = cell_activity_in_window(pipeline, epoch, (t0, t1),
                                        cell_types=cell_types,
                                        cell_ids=cell_ids,
                                        std_scaling=std_scaling)
        for row in table.itertuples():
            spikes_s = np.asarray(row.spike_times_s, dtype=float)
            vector = np.exp(1j * 2 * np.pi * freq * (spikes_s - pre_s)).sum()
            entry = pooled.setdefault(int(row.cell_id), {
                'cell_id': int(row.cell_id),
                'noise_id': int(row.noise_id),
                'cell_type': row.cell_type,
                'center_x': row.center_x, 'center_y': row.center_y,
                'width': row.width, 'height': row.height, 'angle': row.angle,
                'n_spikes': 0, 'vector': 0j,
            })
            entry['n_spikes'] += int(row.n_spikes)
            entry['vector'] += vector

    df = pd.DataFrame(list(pooled.values()))
    if df.empty:
        return df

    duration_s = (t1 - t0) * len(epochs)
    n = df['n_spikes'].to_numpy(dtype=float)
    vector = df.pop('vector').to_numpy()

    with np.errstate(invalid='ignore', divide='ignore'):
        strength = np.abs(vector) / n
    enough = n >= int(min_spikes)

    df['rate_hz'] = n / duration_s
    df['axis_px'] = _axis_position(df['center_x'], df['center_y'], geometry)
    df['radius_px'] = np.hypot(df['center_x'] - float(geometry['center_x']),
                               df['center_y'] - float(geometry['center_y']))
    df['inside_aperture'] = (df['radius_px']
                             <= float(geometry['aperture_diameter_px']) / 2.0)

    df['f1_strength'] = np.where(enough, strength, np.nan)
    # Rayleigh's test for a non-uniform phase distribution, in the large-n
    # form. A cell with many spikes needs only a small vector strength to be
    # significantly modulated, which is the regime these epochs are in.
    df['rayleigh_p'] = np.where(enough, np.exp(-n * strength ** 2), np.nan)

    resp = np.where(enough, np.angle(vector), np.nan)
    stim = np.pi / 2 - 2 * np.pi * float(geometry['spatial_freq_cyc_per_px']) \
        * df['axis_px'].to_numpy()
    df['resp_phase_rad'] = resp
    df['stim_phase_rad'] = _wrap_pi(stim)
    df['residual_rad'] = _wrap_pi(resp - stim)

    return df.sort_values(['cell_type', 'axis_px']).reset_index(drop=True)


def phase_period_scan(phase_df, geometry: Dict,
                      period_range_px: Optional[Tuple[float, float]] = None,
                      n_periods: int = 240,
                      orientations_deg: Optional[Sequence[float]] = None,
                      min_strength: float = 0.0,
                      inside_only: bool = True,
                      min_cells_per_type: int = 3,
                      n_shuffles: int = 0,
                      random_seed: Optional[int] = 0) -> Dict:
    """Recover the grating's period and orientation from response phases alone.

    For a candidate period ``P`` and orientation ``θ``, every cell gets a
    predicted stimulus phase from its own position and the residuals are
    averaged as unit vectors — **per cell type**, because ON and OFF cells sit
    half a cycle apart and pooling them across that offset would cancel a
    real alignment. The concentration reported is the type-count-weighted mean
    of the per-type vector lengths, so it is 1 when every cell of every type
    agrees and ~``1/√n`` when phases are unrelated to position.

    The peak of that surface is a measurement of the stimulus made through the
    retina. Comparing it to ``geometry`` compares two independent routes to the
    same number: one from the recorded parameters and the MATLAB conversions,
    one from spike times and STA centers.

    Parameters
    ----------
    phase_df : pandas.DataFrame
        From :func:`drift_phase_response`.
    geometry : dict
        The epoch's geometry, for the true period/orientation and the aperture.
    period_range_px : (low, high), optional
        Candidate periods in canvas pixels. Default spans a factor of four
        around the true one, which is wide enough to show that the peak is a
        peak rather than the edge of the search.
    n_periods : int
        Grid resolution over periods, log-spaced.
    orientations_deg : sequence, optional
        Candidate orientations. Default is 24 steps over 180°, which is enough
        to see the ridge; pass a finer grid around the true value to sharpen
        the estimate. A single-element sequence scans period only.
    min_strength : float
        Drop cells whose vector strength is below this. The default keeps all
        of them: weak cells lower the concentration everywhere without moving
        its peak, and cutting on strength risks selecting cells for having a
        phase at all.
    inside_only : bool
        Use only cells whose RF center is inside the aperture. The cells
        outside it were looking at black.
    min_cells_per_type : int
        Cell types with fewer cells than this are left out of the
        concentration. A type of one cell has a vector length of exactly 1 at
        every candidate period — it cannot disagree with itself — so it adds a
        constant that looks like alignment and is not.
    n_shuffles : int
        Permute the receptive-field positions among the cells and rescan, this
        many times, to get the null the peak has to beat. The statistic is the
        same one reported — the maximum over the whole grid — so the
        comparison is like for like and includes whatever the search itself
        buys. 0 (default) skips it.
    random_seed : int, optional
        Seed for that permutation, so a reported null is reproducible.

    Returns
    -------
    dict
        ``periods_px`` / ``periods_um``, ``orientations_deg``,
        ``concentration`` (an ``[orientation, period]`` array),
        ``best_period_px`` / ``best_period_um``,
        ``best_orientation_deg``, ``best_concentration``, ``n_cells``,
        ``period_curve`` (concentration at the best orientation),
        ``true_period_px`` / ``true_period_um``, ``true_orientation_deg``,
        ``by_type`` —
        ``{cell_type: {'n', 'concentration', 'mean_residual_rad'}}`` evaluated
        at the true geometry — and, when ``n_shuffles`` is set,
        ``null_concentration`` (the shuffled maxima), ``null_p`` and
        ``null_95``.
    """
    df = phase_df
    keep = df['resp_phase_rad'].notna() & df['center_x'].notna()
    if inside_only:
        keep &= df['inside_aperture']
    if min_strength > 0:
        keep &= df['f1_strength'] >= float(min_strength)
    df = df[keep]
    if df.empty:
        raise ValueError('no cells left to scan — every one is NaN, outside '
                         'the aperture, or below min_strength')

    true_period = 1.0 / float(geometry['spatial_freq_cyc_per_px'])
    if period_range_px is None:
        period_range_px = (true_period / 4.0, true_period * 4.0)
    periods = np.geomspace(*period_range_px, int(n_periods))
    orientations = (np.linspace(0.0, 180.0, 24, endpoint=False)
                    if orientations_deg is None
                    else np.asarray(orientations_deg, dtype=float))

    resp = df['resp_phase_rad'].to_numpy()
    types = df['cell_type'].to_numpy()
    unique_types = [t for t in dict.fromkeys(types)
                    if (types == t).sum() >= int(min_cells_per_type)]
    if not unique_types:
        raise ValueError(f'no cell type has {min_cells_per_type} cells with a '
                         f'phase inside the aperture')
    masks = [types == t for t in unique_types]
    counts = np.array([m.sum() for m in masks], dtype=float)

    # Positions enter only through the axis projection, so precompute one per
    # orientation and reuse them for the scan and for every shuffle.
    axes = [_axis_position(df['center_x'], df['center_y'], geometry, ori)
            for ori in orientations]

    def _surface(order) -> np.ndarray:
        """Concentration over [orientation, period] for one cell ordering."""
        out = np.empty((len(orientations), len(periods)))
        for i, axis in enumerate(axes):
            # residual = resp - (π/2 - 2π·a/P); the constant drops out of a
            # vector length, so it is left off here.
            angles = resp[None, :] + 2 * np.pi * axis[order][None, :] / periods[:, None]
            per_type = np.array([np.abs(np.exp(1j * angles[:, m]).mean(axis=1))
                                 for m in masks])        # [type, period]
            out[i] = (per_type * counts[:, None]).sum(axis=0) / counts.sum()
        return out

    identity = np.arange(len(df))
    concentration = _surface(identity)
    best = np.unravel_index(np.argmax(concentration), concentration.shape)

    # Per-type summary at the true geometry, which is what the notebook prints.
    by_type = {}
    for t in unique_types:
        residual = df.loc[types == t, 'residual_rad'].to_numpy()
        z = np.exp(1j * residual).mean()
        by_type[t] = {'n': int((types == t).sum()),
                      'concentration': float(np.abs(z)),
                      'mean_residual_rad': float(np.angle(z))}

    null = {}
    if n_shuffles:
        rng = np.random.default_rng(random_seed)
        maxima = np.array([_surface(rng.permutation(identity)).max()
                           for _ in range(int(n_shuffles))])
        observed = float(concentration[best])
        null = {
            'null_concentration': maxima,
            'null_95': float(np.percentile(maxima, 95)),
            # Add-one so a p from N shuffles is never reported as exactly 0.
            'null_p': float((np.sum(maxima >= observed) + 1) / (len(maxima) + 1)),
        }

    return {
        **null,
        'periods_px': periods,
        'periods_um': periods * float(geometry['microns_per_pixel']),
        'orientations_deg': orientations,
        'concentration': concentration,
        'best_period_px': float(periods[best[1]]),
        'best_period_um': float(
            periods[best[1]] * float(geometry['microns_per_pixel'])),
        'best_orientation_deg': float(orientations[best[0]]),
        'best_concentration': float(concentration[best]),
        'period_curve': concentration[best[0]],
        'n_cells': int(len(df)),
        'true_period_px': float(true_period),
        'true_period_um': float(
            true_period * float(geometry['microns_per_pixel'])),
        'true_orientation_deg': float(geometry['orientation_deg']),
        'by_type': by_type,
    }


def phase_alignment_by_condition(pipeline, stim_block, epochs,
                                 condition_keys: Sequence[str],
                                 *,
                                 geometry_fn=None,
                                 cell_types: Optional[Iterable[str]] = None,
                                 cell_ids: Optional[Iterable[int]] = None,
                                 window_s: Optional[Tuple[float, float]] = None,
                                 min_spikes: int = 30,
                                 n_shuffles: int = 0,
                                 epoch_col: str = 'epoch',
                                 drift_freq_hz: Optional[float] = None,
                                 **scan_kwargs) -> Tuple[Dict, 'pd.DataFrame']:
    """Measure and scan the phase alignment separately for each condition.

    **A phase estimate may only pool epochs that ran the same geometry.** Drift
    phase is measured from stimulus onset, so epochs of one condition are in
    register with each other and their F1 vectors add, while a different bar
    width is a different stimulus with a different prediction. That is the
    whole reason this is a loop over conditions rather than one call — every
    protocol that alternates grating parameters needs the same loop.

    Parameters
    ----------
    pipeline, stim_block
        The pipeline supplies spikes and receptive fields; the stimulus block
        supplies each epoch's recorded parameters, so the geometry comes from
        the epoch that ran rather than from the protocol's declared defaults.
    epochs : pandas.DataFrame
        The analyzed epochs, one row each, with ``epoch_col`` and every name in
        ``condition_keys`` as columns — an ``epoch_condition_table`` restricted
        to the epochs being kept.
    condition_keys : sequence[str]
        The condition axes to group on, from
        :func:`~retinanalysis.utils.protocol_source.condition_keys`.
    geometry_fn : callable, optional
        ``(stim_block, epoch_index) -> geometry dict``. Defaults to
        :func:`~retinanalysis.regen.variable_mean_drifting_grating
        .grating_geometry`; pass another protocol's regenerator to reuse this.
    n_shuffles : int
        Shuffled-position null for each condition's scan. On a handful of
        driven cells a grid search finds a peak in noise, so without this there
        is nothing to compare a concentration against.
    drift_freq_hz : float, optional
        Fold every condition at this frequency rather than each geometry's
        nominal one. All conditions ran on the same display, so one estimate
        from :func:`~retinanalysis.utils.spatial_recovery
        .estimate_drift_frequency` applies to all of them. Leaving it ``None``
        keeps the declared value and, over a 60 s epoch, most of the F1 with
        it — the *period and orientation* the scan picks are unaffected, since
        that is a spatial fit at fixed temporal frequency, but every strength
        and every residual latency is.
    **scan_kwargs
        Passed to :func:`phase_period_scan` (``orientations_deg``,
        ``period_range_px``, ``min_cells_per_type``, …).

    Returns
    -------
    (phase_by_condition, summary)
        ``phase_by_condition`` maps a condition — always a tuple of levels, in
        ``condition_keys`` order — to ``(phases, scan, geometry)``. ``summary``
        is one row per condition: the condition levels, ``n_epochs``, the
        stimulus's own ``period_um`` / ``orient_deg`` (with ``period_px``
        retained for computation and backward compatibility), how many cells were
        driven and how modulated they were, what the phases ``picks_um`` /
        ``picks_deg`` (also retaining ``picks_px``), the concentration ``R``,
        and ``shuffled_95``.
    """
    import pandas as pd

    if geometry_fn is None:
        from retinanalysis.regen.variable_mean_drifting_grating import (
            grating_geometry)
        geometry_fn = grating_geometry

    keys = [condition_keys] if isinstance(condition_keys, str) else list(condition_keys)

    phase_by_condition: Dict[Tuple, Tuple] = {}
    rows = []
    for levels, group in epochs.groupby(keys):
        levels = levels if isinstance(levels, tuple) else (levels,)
        epoch_indices = group[epoch_col].astype(int).tolist()
        # Correct the geometry once, here, so the frequency every consumer
        # reads is the corrected one: the F1 sums below, the cycle length
        # `residual_latency_table` turns a residual into milliseconds with,
        # and the copy stored in `phase_by_condition` for the figures.
        geometry = with_drift_freq(geometry_fn(stim_block, epoch_indices[0]),
                                   drift_freq_hz)

        phases = drift_phase_response(pipeline, epoch_indices, geometry,
                                      window_s=window_s,
                                      cell_types=cell_types,
                                      cell_ids=cell_ids,
                                      min_spikes=min_spikes)
        scan = phase_period_scan(phases, geometry, n_shuffles=n_shuffles,
                                 **scan_kwargs)
        phase_by_condition[levels] = (phases, scan, geometry)

        # Cells outside the aperture were looking at black, so they are not
        # part of any claim about the grating.
        driven = phases[phases['inside_aperture']
                        & phases['resp_phase_rad'].notna()]
        rows.append({
            **dict(zip(keys, levels)),
            'n_epochs': len(epoch_indices),
            'period_px': scan['true_period_px'],
            'period_um': scan['true_period_um'],
            'orient_deg': scan['true_orientation_deg'],
            'n_cells': len(driven),
            'median_f1': driven['f1_strength'].median(),
            'n_modulated': int((driven['rayleigh_p'] < 0.01).sum()),
            'picks_px': scan['best_period_px'],
            'picks_um': scan['best_period_um'],
            'picks_deg': scan['best_orientation_deg'],
            'R': scan['best_concentration'],
            'shuffled_95': scan.get('null_95', np.nan),
        })

    summary = pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
    return phase_by_condition, summary


def residual_latency_table(scan: Dict, geometry: Dict) -> 'pd.DataFrame':
    """Per cell type, the mean residual phase read as a response latency.

    The residual holds constant across cells when the prediction holds, and
    what it holds at is the cell's latency expressed as a phase — so it is a
    latency **mod one drift cycle** (500 ms at 2 Hz), not an absolute one. ON
    and OFF cells come out half a cycle apart, which is why this is per type.

    **Sign.** ``stim_phase`` is the drift phase at which luminance peaks over
    that cell, and a cell firing ``tau`` later has its mean spike phase that
    much further on, so ``residual = +2*pi*f*tau`` — a positive residual is a
    positive lag. Reporting ``-residual`` instead returns the complement,
    ``cycle - tau``, which for a real retinal latency lands near a full cycle
    and reads as though the response preceded the stimulus.

    This is only legible once the fold uses the *delivered* drift frequency:
    at the nominal one the residual is averaged over a phase that drifts
    across the epoch, so it is not an estimate of anything and its sign is
    arbitrary. See :func:`with_drift_freq`.

    Returns a frame with ``cell_type``, ``n``, ``concentration``,
    ``mean_residual_deg``, ``lag_ms`` (in ``[0, cycle)``) and
    ``lag_ms_signed`` (wrapped to +/- half a cycle, so a lag just under a
    full cycle reads as the small negative it is).
    """
    import pandas as pd

    cycle_ms = 1000.0 / float(geometry['temporal_freq_hz'])
    rows = [{
        'cell_type': cell_type,
        'n': int(d['n']),
        'concentration': float(d['concentration']),
        'mean_residual_deg': float(np.degrees(d['mean_residual_rad'])),
        'lag_ms': float((d['mean_residual_rad'] % (2 * np.pi))
                        / (2 * np.pi) * cycle_ms),
        'lag_ms_signed': float(_wrap_pi(d['mean_residual_rad'])
                               / (2 * np.pi) * cycle_ms),
        'cycle_ms': cycle_ms,
    } for cell_type, d in scan['by_type'].items()]
    return pd.DataFrame(rows)


def describe_phase_alignment(summary, phase_by_condition,
                             condition_keys: Sequence[str],
                             verbose: bool = True) -> Dict:
    """Which conditions align above their own null, and what latency that implies.

    Two questions, in the order they have to be asked. **Did anything align?**
    — only a peak that beats its shuffled null is evidence, because the grid
    search finds a peak in noise when few cells are driven. **If so, at what
    lag?** — the mean residual per cell type in the best-aligned condition,
    from :func:`residual_latency_table`.

    Returns ``{'aligned': DataFrame, 'best_condition': tuple or None,
    'latency': DataFrame or None}``, and prints the same when ``verbose``.
    """
    from .protocol_source import condition_label

    keys = [condition_keys] if isinstance(condition_keys, str) else list(condition_keys)
    aligned = summary[summary['R'] > summary['shuffled_95']]

    if verbose:
        print(f'{len(aligned)} of {len(summary)} conditions align above the '
              f'shuffle null:')
        for _, row in aligned.iterrows():
            print(f"  {condition_label(keys, row)} — {int(row['n_cells'])} "
                  f"cells, {int(row['n_modulated'])} of them modulated at "
                  f"p<0.01; the "
                  f"phases pick {row['picks_um']:.1f} µm at "
                  f"{row['picks_deg']:.0f}° against a stimulus of "
                  f"{row['period_um']:.0f} µm at {row['orient_deg']:.0f}°  "
                  f"(R = {row['R']:.2f}, shuffled {row['shuffled_95']:.2f})")

    if summary.empty or not summary['R'].notna().any():
        return {'aligned': aligned, 'best_condition': None, 'latency': None}

    best_key = tuple(summary.loc[summary['R'].idxmax(), keys].tolist())
    _, best_scan, best_geometry = phase_by_condition[best_key]
    latency = residual_latency_table(best_scan, best_geometry)

    if verbose:
        cycle_ms = latency['cycle_ms'].iloc[0] if len(latency) else float('nan')
        print(f'\nresidual phase by type at {condition_label(keys, best_key)} '
              f'(a latency mod {cycle_ms:.0f} ms):')
        for row in latency.itertuples():
            print(f'  {row.cell_type:5s} n = {row.n:3d}   '
                  f'R = {row.concentration:.2f}   mean residual '
                  f'{row.mean_residual_deg:7.1f}°  ->  {row.lag_ms:.0f} ms'
                  + (f' (= {row.lag_ms_signed:.0f} ms)'
                     if row.lag_ms > cycle_ms / 2 else ''))

    return {'aligned': aligned, 'best_condition': best_key, 'latency': latency}


def plot_phase_alignment(phase_df, geometry: Dict, scan: Optional[Dict] = None,
                         cell_types: Optional[Iterable[str]] = None,
                         inside_only: bool = True,
                         title: Optional[str] = None,
                         figsize: Tuple[float, float] = (14.0, 4.2)):
    """The prediction, the residual in space, and the period the phases pick.

    Three panels, left to right:

    1. **Response phase against position across the bars.** The prediction is
       the sawtooth: one cycle of phase per spatial period, drawn per cell type
       through that type's own mean residual, so what is being judged is the
       *slope*, not an offset that latency and polarity are free to set.
    2. **The residual on the mosaic**, on a cyclic colormap. One flat color
       inside the aperture is the alignment holding; a gradient left to right
       means the period is wrong, and a patch of unrelated color means those
       cells are not following the grating.
    3. **Concentration against candidate period**, if ``scan`` is given, with
       the geometry's period marked. A peak on the mark is the measurement
       agreeing with the reconstruction.

    Returns the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    df = phase_df[phase_df['resp_phase_rad'].notna() & phase_df['center_x'].notna()]
    if inside_only:
        df = df[df['inside_aperture']]
    if cell_types is not None:
        df = df[df['cell_type'].isin(list(cell_types))]
    if df.empty:
        raise ValueError('nothing to plot: no cells with a phase')

    n_panels = 3 if scan is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    period_px = 1.0 / float(geometry['spatial_freq_cyc_per_px'])
    um_per_px = float(geometry['microns_per_pixel'])
    period_um = period_px * um_per_px
    types = list(dict.fromkeys(df['cell_type']))
    colors = dict(zip(types, plt.rcParams['axes.prop_cycle'].by_key()['color']))

    # --- 1. phase vs position, against the predicted slope -----------------
    ax = axes[0]
    span_px = np.linspace(df['axis_px'].min(), df['axis_px'].max(), 1200)
    span_um = span_px * um_per_px
    for t in types:
        rows = df[df['cell_type'] == t]
        offset = np.angle(np.exp(1j * rows['residual_rad'].to_numpy()).mean())
        ax.scatter(rows['axis_px'] * um_per_px,
                   np.degrees(_wrap_pi(rows['resp_phase_rad'])),
                   s=18, alpha=0.8, color=colors[t],
                   label=f'{t} (n={len(rows)})')
        predicted = _wrap_pi(
            np.pi / 2 - 2 * np.pi * span_px / period_px + offset)
        # Break the line where the wrap jumps, so the sawtooth has no risers.
        predicted = np.where(np.abs(np.diff(predicted, prepend=predicted[0])) > np.pi,
                             np.nan, predicted)
        ax.plot(span_um, np.degrees(predicted), color=colors[t], lw=1.0,
                alpha=0.6)
    ax.set_xlabel('position across the bars (µm from center)')
    ax.set_ylabel('response phase (deg)')
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_title(f'one cycle per {period_um:.0f} µm period')
    ax.legend(fontsize=7, framealpha=0.85, loc='lower left')

    # --- 2. the residual on the mosaic -------------------------------------
    # Shown relative to each type's own mean, so the ON/OFF half-cycle offset
    # does not take up the color scale that spatial structure needs.
    ax = axes[1]
    centered = np.concatenate([
        _wrap_pi(df.loc[df['cell_type'] == t, 'residual_rad'].to_numpy()
                 - np.angle(np.exp(1j * df.loc[df['cell_type'] == t,
                                               'residual_rad'].to_numpy()).mean()))
        for t in types])
    order = np.concatenate([np.flatnonzero((df['cell_type'] == t).to_numpy())
                            for t in types])
    x_um = ((df['center_x'].to_numpy() - float(geometry['center_x']))
            * um_per_px)
    y_um = ((df['center_y'].to_numpy() - float(geometry['center_y']))
            * um_per_px)
    scat = ax.scatter(x_um[order], y_um[order],
                      c=np.degrees(centered),
                      s=np.clip(40 * df['f1_strength'].to_numpy()[order]
                                / max(df['f1_strength'].max(), 1e-9), 6, 60),
                      cmap='twilight_shifted', vmin=-180, vmax=180)
    ax.add_patch(Circle((0.0, 0.0), geometry['aperture_diameter_um'] / 2.0,
                        fill=False, color='0.4', ls='--', lw=1.0))
    ax.set_xlim(-geometry['canvas_w'] * um_per_px / 2.0,
                geometry['canvas_w'] * um_per_px / 2.0)
    ax.set_ylim(geometry['canvas_h'] * um_per_px / 2.0,
                -geometry['canvas_h'] * um_per_px / 2.0)
    ax.set_aspect('equal')
    ax.set_xlabel('retinal x (µm from center)')
    ax.set_ylabel('retinal y (µm from center)')
    ax.set_title('residual phase, sized by F1 strength')
    fig.colorbar(scat, ax=ax, label='residual re. type mean (deg)',
                 fraction=0.046)

    # --- 3. which period the phases prefer ---------------------------------
    if scan is not None:
        ax = axes[2]
        periods_um = scan.get(
            'periods_um', scan['periods_px'] * um_per_px)
        true_period_um = scan.get(
            'true_period_um', scan['true_period_px'] * um_per_px)
        best_period_um = scan.get(
            'best_period_um', scan['best_period_px'] * um_per_px)
        ax.semilogx(periods_um, scan['period_curve'], color='0.2')
        ax.axvline(true_period_um, color='crimson', ls='--', lw=1.0,
                   label=f"stimulus {true_period_um:.0f} µm")
        ax.axvline(best_period_um, color='steelblue', ls=':', lw=1.2,
                   label=f"phases pick {best_period_um:.0f} µm")
        if 'null_95' in scan:
            ax.axhline(scan['null_95'], color='0.6', lw=0.9,
                       label=f"shuffled positions, 95th pct")
        ax.set_xlabel('candidate spatial period (µm)')
        ax.set_ylabel('phase concentration')
        ax.set_ylim(0, 1)
        ticks = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
        ticks = ticks[(ticks >= periods_um[0]) & (ticks <= periods_um[-1])]
        ax.set_xticks(ticks, [str(t) for t in ticks], minor=False)
        ax.set_xticks([], minor=True)
        ax.set_title(f"at {scan['best_orientation_deg']:.0f}°, "
                     f"{scan['n_cells']} cells")
        ax.legend(fontsize=7, frameon=False)

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def browse_phase_alignment(phase_by_condition, condition_keys: Sequence[str],
                           description: str = 'Condition:', **plot_kwargs):
    """Dropdown over conditions, each drawn by :func:`plot_phase_alignment`.

    The label carries the condition's concentration against its shuffled null,
    so the one that aligned is findable without rendering the rest — and the
    ones that did not are worth a look too, since a failed alignment and a
    resolved-but-offset one look nothing alike.

    Takes the ``phase_by_condition`` mapping from
    :func:`phase_alignment_by_condition`. Returns the widget, or ``None`` when
    there is nothing to show.
    """
    from .browse import figure_to_png, png_browser
    from .protocol_source import condition_label

    keys = [condition_keys] if isinstance(condition_keys, str) else list(condition_keys)

    def _render(key):
        phases, scan, geometry = phase_by_condition[key]
        fig = plot_phase_alignment(phases, geometry, scan,
                                   title=condition_label(keys, key),
                                   **plot_kwargs)
        return None, figure_to_png(fig)

    options = [
        (f"{condition_label(keys, key)}   (R {scan['best_concentration']:.2f} "
         f"vs shuffled {scan.get('null_95', float('nan')):.2f})", key)
        for key, (_, scan, _) in phase_by_condition.items()
    ]
    return png_browser(options, _render, description=description,
                       empty_message='No conditions scanned.')
