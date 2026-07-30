"""Per-cell-type summaries of a loaded ``AnalysisChunk``.

Answers "what is actually in this mosaic?" — how many cells of each type were
classified, what their temporal filters and spike-timing statistics look like,
and how hard they were firing during the noise run. Use it next to ``plot_rfs``
when judging which date to analyze: a spatial mosaic can look tidy while
resting on four cells, and a type whose firing rates sit near zero is usually a
sorting artifact rather than a population.

The four views a chunk supports, and what each one catches:

- **Spatial RF** (``AnalysisChunk.plot_rfs``) — mosaic regularity and coverage.
- **Temporal RF** (:func:`plot_chunk_panels`) — whether a type's filter has the
  polarity and biphasic shape its name claims. An "OnP" group whose mean filter
  is flat was never a population.
- **Autocorrelation / ISI** (:func:`plot_chunk_panels`) — spike-timing
  structure. Power at very short lags means the refractory period was violated,
  i.e. the cluster merges more than one unit.
- **Spike count** (:func:`plot_spike_count_distribution`) — how much data each
  cell actually contributes over the noise run.

Firing rate here is the mean rate over the whole noise chunk (total spikes
divided by the recording duration), not a stimulus-locked rate. It is a
data-quality number, not a response measure. Spike count is the same number
before dividing by duration, which is what matters when asking whether an STA
had enough spikes behind it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


__all__ = [
    'firing_rates_by_type',
    'spike_counts_by_type',
    'cell_type_summary',
    'plot_firing_rate_distribution',
    'plot_spike_count_distribution',
    'plot_chunk_panels',
]

# Fallback STA frame interval, in ms, for chunks whose Vision params don't
# carry a refresh period. 8.33 ms is what ``AnalysisChunk.plot_timecourses``
# hard-codes; prefer the recorded ``refreshPeriod`` whenever it is there,
# since the STA depth in frames varies between sorts (30 and 61 both occur)
# and a fixed window would silently mislabel the axis.
_STA_FRAME_MS_FALLBACK = 8.33


def _typing_column(chunk, typing_file: Optional[str]) -> Optional[str]:
    """Name of the ``df_cell_params`` column holding ``typing_file``'s labels.

    Mirrors how ``AnalysisChunk.plot_rfs`` resolves the typing file, so a
    summary always describes the same classification the mosaic was drawn
    from. Returns None when the chunk has no typing files.
    """
    if not getattr(chunk, 'typing_files', None):
        return None
    if typing_file is None:
        typing_file = chunk.typing_files[0]
    if typing_file not in chunk.typing_files:
        raise ValueError(
            f'{typing_file} is not one of the typing files for '
            f'{chunk.exp_name} {chunk.chunk_name}: {chunk.typing_files}')
    return f'typing_file_{chunk.typing_files.index(typing_file)}'


def _ids_by_type(chunk, cell_types: Optional[Sequence[str]] = None,
                 typing_file: Optional[str] = None,
                 minimum_n: int = 1) -> Dict[str, List[int]]:
    """``{cell_type: [cell_id, ...]}`` for the types that clear ``minimum_n``.

    When ``cell_types`` is given the returned dict preserves that order, so a
    figure built from it lines up column-for-column with the mosaic drawn from
    the same list. Types with too few cells are dropped, not emptied.
    """
    col = _typing_column(chunk, typing_file)
    if col is None:
        return {}

    df = chunk.df_cell_params
    grouped = {str(t): rows['cell_id'].astype(int).tolist()
               for t, rows in df.groupby(col)}

    order = list(cell_types) if cell_types is not None else sorted(grouped)
    return {t: grouped[t] for t in order
            if t in grouped and len(grouped[t]) >= minimum_n}


def _counts_and_duration(chunk, cell_types: Optional[Sequence[str]],
                         typing_file: Optional[str],
                         minimum_n: int) -> Tuple[Dict[str, np.ndarray], float]:
    """``({cell_type: per-cell total spike counts}, chunk duration in sec)``.

    The one place that touches spike times; rates and counts are both derived
    from it so they can never disagree about which cells were countable.
    """
    from retinanalysis.classes.response import SAMPLE_RATE

    ids = _ids_by_type(chunk, cell_types=cell_types, typing_file=typing_file,
                       minimum_n=minimum_n)
    if not ids:
        return {}, 0.0

    n_samples = getattr(chunk.vcd, 'n_samples', None)
    if not n_samples:
        raise ValueError(
            f'{chunk.exp_name} {chunk.chunk_name}: no sample count on the VCD, '
            'so firing rate cannot be computed. Load the chunk with '
            'include_neurons=True.')
    duration_sec = float(n_samples) / SAMPLE_RATE

    out: Dict[str, np.ndarray] = {}
    for cell_type, cell_ids in ids.items():
        counts = []
        for cell_id in cell_ids:
            try:
                spikes = chunk.vcd.get_spike_times_for_cell(int(cell_id))
            except Exception:
                continue
            counts.append(len(spikes))
        # minimum_n applies again: cells can drop out for want of spike times.
        if len(counts) >= minimum_n:
            out[cell_type] = np.asarray(counts, dtype=float)
    return out, duration_sec


def firing_rates_by_type(chunk, cell_types: Optional[Sequence[str]] = None,
                         typing_file: Optional[str] = None,
                         minimum_n: int = 1) -> Dict[str, np.ndarray]:
    """``{cell_type: array of per-cell mean firing rates in Hz}``.

    The chunk must have been loaded with ``include_neurons=True``; without the
    .neurons file there are no spike times to count and this raises.

    ``minimum_n`` drops types with fewer than that many classified cells, the
    same knob ``plot_rfs`` uses, so the summary and the mosaic agree on which
    types exist.
    """
    counts, duration_sec = _counts_and_duration(chunk, cell_types, typing_file,
                                                minimum_n)
    return {t: c / duration_sec for t, c in counts.items()}


def spike_counts_by_type(chunk, cell_types: Optional[Sequence[str]] = None,
                         typing_file: Optional[str] = None,
                         minimum_n: int = 1) -> Dict[str, np.ndarray]:
    """``{cell_type: array of per-cell total spike counts over the chunk}``.

    Same numbers as :func:`firing_rates_by_type` before dividing by the
    recording duration. Counts are the right unit when the question is whether
    a cell contributed enough spikes for its STA to mean anything; rate is the
    right unit when comparing across chunks of different length.
    """
    counts, _ = _counts_and_duration(chunk, cell_types, typing_file, minimum_n)
    return counts


def cell_type_summary(chunk, cell_types: Optional[Sequence[str]] = None,
                      typing_file: Optional[str] = None,
                      minimum_n: int = 1,
                      include_rates: bool = True) -> pd.DataFrame:
    """One row per cell type: how many cells, and how hard they fire.

    Columns: ``n_cells``, plus ``mean_rate_hz`` / ``median_rate_hz`` /
    ``min_rate_hz`` / ``max_rate_hz`` / ``mean_spikes`` when ``include_rates``
    and the chunk carries spike times. Sorted by descending cell count.

    Set ``include_rates=False`` for a counts-only table when the chunk was
    loaded without the .neurons file.
    """
    col = _typing_column(chunk, typing_file)
    if col is None:
        return pd.DataFrame(columns=['n_cells'])

    df = chunk.df_cell_params
    if cell_types is not None:
        df = df[df[col].isin(list(cell_types))]

    counts = df.groupby(col).size().rename('n_cells')
    counts = counts[counts >= minimum_n]
    summary = counts.to_frame()
    summary.index.name = 'cell_type'

    if include_rates:
        try:
            rates = firing_rates_by_type(chunk, cell_types=cell_types,
                                         typing_file=typing_file,
                                         minimum_n=minimum_n)
        except ValueError:
            rates = {}
        if rates:
            from retinanalysis.classes.response import SAMPLE_RATE
            duration_sec = float(chunk.vcd.n_samples) / SAMPLE_RATE
            for stat, fn in (('mean_rate_hz', np.mean),
                             ('median_rate_hz', np.median),
                             ('min_rate_hz', np.min),
                             ('max_rate_hz', np.max)):
                summary[stat] = [round(float(fn(rates[t])), 2)
                                 if t in rates else np.nan
                                 for t in summary.index]
            summary['mean_spikes'] = [
                int(round(float(np.mean(rates[t])) * duration_sec))
                if t in rates else -1
                for t in summary.index]

    return summary.sort_values('n_cells', ascending=False)


def _draw_ecdf(ax, values_by_type: Dict[str, np.ndarray], xlabel: str,
               title: Optional[str], log_x: bool, legend: bool = True):
    """Overlay one ECDF step curve per cell type onto ``ax``.

    An ECDF rather than a histogram or KDE on purpose: a well-populated type
    here has a few dozen cells and a marginal one has three, and at those
    counts a histogram's shape is mostly bin placement. The ECDF is exact at
    any n — every cell is one step — so overlaying a sparse type against a
    dense one stays honest, and the curves separate vertically instead of
    occluding each other the way filled histograms do.
    """
    from retinanalysis.utils.style import colors_for_celltypes

    ordered = sorted(values_by_type, key=lambda t: -len(values_by_type[t]))
    colors = colors_for_celltypes(ordered)
    for cell_type in ordered:
        values = np.sort(values_by_type[cell_type])
        # Step from 0 to 1; one step per cell.
        y = np.arange(1, len(values) + 1) / len(values)
        ax.step(np.concatenate([values, values[-1:]]),
                np.concatenate([y, [1.0]]),
                where='post', color=colors[cell_type], linewidth=1.6,
                label=f'{cell_type} (n={len(values)})')

    if log_x:
        ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative fraction of cells')
    ax.set_ylim(0, 1.02)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(bbox_to_anchor=[1.02, 1], loc='upper left')
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    return ax


def plot_firing_rate_distribution(chunk, cell_types: Optional[Sequence[str]] = None,
                                  typing_file: Optional[str] = None,
                                  minimum_n: int = 3, ax=None,
                                  title: Optional[str] = None,
                                  log_x: bool = False):
    """Overlay each cell type's firing-rate distribution as an ECDF.

    Read it as: how far right a curve sits is how fast that type fires, and
    how steep it is is how tightly the type clusters. See :func:`_draw_ecdf`
    for why this is an ECDF rather than a histogram.

    Colors come from the package cell-type map, so a type is the same color
    here as in every other figure. Returns the Axes.
    """
    import matplotlib.pyplot as plt

    rates = firing_rates_by_type(chunk, cell_types=cell_types,
                                 typing_file=typing_file, minimum_n=minimum_n)
    if not rates:
        print(f'{chunk.exp_name} {chunk.chunk_name}: no cell types with '
              f'>= {minimum_n} cells to plot.')
        return None

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.4))

    return _draw_ecdf(ax, rates, 'Mean firing rate over the chunk (Hz)',
                      title or f'{chunk.exp_name} {chunk.chunk_name}', log_x)


def plot_spike_count_distribution(chunk, cell_types: Optional[Sequence[str]] = None,
                                  typing_file: Optional[str] = None,
                                  minimum_n: int = 3, ax=None,
                                  title: Optional[str] = None,
                                  log_x: bool = True):
    """Overlay each cell type's total-spike-count distribution as an ECDF.

    The count is over the whole noise run, so this is the "how much data is
    behind each STA" view: a type whose curve sits left of a few thousand
    spikes has receptive fields fit from very little, however clean the mosaic
    looks. Counts span orders of magnitude across types, so the x axis is
    logarithmic by default — pass ``log_x=False`` for a linear one.

    Returns the Axes.
    """
    import matplotlib.pyplot as plt

    counts = spike_counts_by_type(chunk, cell_types=cell_types,
                                  typing_file=typing_file, minimum_n=minimum_n)
    if not counts:
        print(f'{chunk.exp_name} {chunk.chunk_name}: no cell types with '
              f'>= {minimum_n} cells to plot.')
        return None

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.4))

    return _draw_ecdf(ax, counts, 'Total spikes over the noise run',
                      title or f'{chunk.exp_name} {chunk.chunk_name}', log_x)


def _isi_bin_centers(analysis_chunk) -> np.ndarray:
    edges = np.asarray(analysis_chunk.isi_bin_edges)
    return 0.5 * (edges[:-1] + edges[1:])


def _sta_time_axis_ms(analysis_chunk, n_frames: int) -> np.ndarray:
    """Milliseconds relative to the spike for each STA frame, latest last.

    The frame interval comes from the noise run's recorded ``refreshPeriod``
    rather than an assumed 60 Hz, because the STA depth differs between sorts
    — 30 and 61 frames both occur — and the elapsed time each frame covers is
    the only thing that makes those two comparable.
    """
    params = getattr(getattr(analysis_chunk, 'vcd', None),
                     'runtimemovie_params', None)
    dt = getattr(params, 'refreshPeriod', None) or _STA_FRAME_MS_FALLBACK
    # The last frame is the one immediately preceding the spike, so it sits
    # at 0 and everything before it is negative.
    return (np.arange(n_frames) - (n_frames - 1)) * float(dt)


def _draw_tc_panel(ax, analysis_chunk, ids, color, time_ms: bool = False):
    """Mean temporal filter (green channel) + per-cell traces dim.

    With ``time_ms`` the x axis is milliseconds before the spike, from the
    chunk's own frame interval; otherwise it is the bare STA frame index.
    """
    tcs = []
    for cid in ids:
        tc = analysis_chunk.d_timecourses.get(cid)
        if tc is None:
            continue
        tcs.append(tc['green'])
    if not tcs:
        ax.text(0.5, 0.5, '(no timecourses)', transform=ax.transAxes,
                ha='center', va='center', fontsize=8)
        return
    L = min(len(t) for t in tcs)
    mat = np.stack([t[:L] for t in tcs])
    x = _sta_time_axis_ms(analysis_chunk, L) if time_ms else np.arange(L)
    for row in mat:
        ax.plot(x, row, color=color, alpha=0.12, linewidth=0.6)
    mean = mat.mean(axis=0)
    sem = mat.std(axis=0) / np.sqrt(max(mat.shape[0], 1))
    ax.plot(x, mean, color=color, linewidth=1.6)
    ax.fill_between(x, mean - sem, mean + sem,
                    color=color, alpha=0.25, linewidth=0)
    ax.axhline(0, color='gray', lw=0.4, alpha=0.5)
    if time_ms:
        ax.set_xlim(x[0], x[-1])


def _draw_isi_panel(ax, analysis_chunk, ids, color, xlim_ms=200.0):
    """Mean ISI density + per-cell traces dim."""
    centers = _isi_bin_centers(analysis_chunk)
    rows = []
    for cid in ids:
        h = analysis_chunk.d_ISIs.get(cid)
        if h is None:
            continue
        h = np.asarray(h, dtype=float)
        s = h.sum()
        rows.append(h / s if s > 0 else h)
    if not rows:
        ax.text(0.5, 0.5, '(no ISI data)', transform=ax.transAxes,
                ha='center', va='center', fontsize=8)
        return
    mat = np.stack(rows)
    for row in mat:
        ax.plot(centers, row, color=color, alpha=0.12, linewidth=0.6)
    mean = mat.mean(axis=0)
    sem = mat.std(axis=0) / np.sqrt(max(mat.shape[0], 1))
    ax.plot(centers, mean, color=color, linewidth=1.6)
    ax.fill_between(centers, mean - sem, mean + sem,
                    color=color, alpha=0.25, linewidth=0)
    ax.set_xlim(0, xlim_ms)


def plot_chunk_panels(chunk, cell_types: Optional[Sequence[str]] = None,
                      typing_file: Optional[str] = None,
                      minimum_n: int = 3,
                      isi_xlim_ms: float = 200.0,
                      title: Optional[str] = None,
                      log_x: bool = True,
                      panel_width: float = 2.6):
    """Temporal RF, autocorrelation and spike count for one loaded chunk.

    The companion to the spatial mosaic: ``plot_rfs`` shows where a type's
    receptive fields sit, this shows whether the cells behind them behave like
    a population. One column per cell type, in the order ``cell_types`` names
    them so the columns line up with the mosaic, and three rows:

    1. **Temporal RF** — the green-channel STA time course, per cell in dim
       lines with the mean ± SEM over the top. Time is milliseconds relative
       to the spike.
    2. **Autocorrelation (ISI)** — Vision's autocorrelation histogram,
       sum-normalized per cell so a fast-firing cell doesn't dominate the
       mean. Density at lags under ~2 ms means the refractory period was
       violated and the cluster is probably two units.
    3. **Spike count** — one ECDF across all types (spanning the full width),
       total spikes per cell over the noise run.

    Requires a chunk loaded with ``include_neurons=True`` for row 3; rows 1
    and 2 come from the Vision params file and are always available. Returns
    the Figure, or None when no type clears ``minimum_n``.
    """
    import matplotlib.pyplot as plt
    from retinanalysis.utils.style import (NEUTRAL_GRAY, apply_publication_style,
                                           colors_for_celltypes)

    apply_publication_style()

    ids = _ids_by_type(chunk, cell_types=cell_types, typing_file=typing_file,
                       minimum_n=minimum_n)
    if not ids:
        print(f'{chunk.exp_name} {chunk.chunk_name}: no cell types with '
              f'>= {minimum_n} cells to plot.')
        return None

    try:
        counts = spike_counts_by_type(chunk, cell_types=cell_types,
                                      typing_file=typing_file,
                                      minimum_n=minimum_n)
    except ValueError as err:
        # No .neurons file: keep the two params-derived rows rather than
        # failing the whole figure.
        print(f'{chunk.exp_name} {chunk.chunk_name}: skipping spike counts — {err}')
        counts = {}

    types = list(ids)
    colors = colors_for_celltypes(types)

    n_cols = len(types)
    fig = plt.figure(figsize=(max(panel_width * n_cols, 5.5), 8.2))
    # Row 3 is one shared axis, so it spans every column.
    gs = fig.add_gridspec(3, n_cols, height_ratios=[1.0, 1.0, 1.25],
                          hspace=0.55, wspace=0.35)

    for col, cell_type in enumerate(types):
        color = colors.get(cell_type, NEUTRAL_GRAY)
        cell_ids = ids[cell_type]

        ax_tc = fig.add_subplot(gs[0, col])
        _draw_tc_panel(ax_tc, chunk, cell_ids, color, time_ms=True)
        ax_tc.set_title(f'{cell_type} (n={len(cell_ids)})', color=color, pad=3)
        ax_tc.set_xlabel('Time from spike (ms)')
        if col == 0:
            ax_tc.set_ylabel('STA (arb. units)')

        ax_isi = fig.add_subplot(gs[1, col])
        _draw_isi_panel(ax_isi, chunk, cell_ids, color, xlim_ms=isi_xlim_ms)
        ax_isi.set_xlabel('Lag (ms)')
        if col == 0:
            ax_isi.set_ylabel('Autocorrelation (norm.)')

    ax_counts = fig.add_subplot(gs[2, :])
    if counts:
        _draw_ecdf(ax_counts, counts, 'Total spikes over the noise run',
                   None, log_x)
    else:
        ax_counts.text(0.5, 0.5, '(no spike times — load with include_neurons=True)',
                       transform=ax_counts.transAxes, ha='center', va='center',
                       fontsize=9)
        ax_counts.set_axis_off()

    fig.suptitle(title or f'{chunk.exp_name} {chunk.chunk_name}')
    return fig
