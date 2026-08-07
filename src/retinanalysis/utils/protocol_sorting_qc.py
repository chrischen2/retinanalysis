"""Quantitative multichannel spike-sorting QC for protocol recordings.

The ordinary chunk/noise QC asks whether a cluster has a stable EI and RF.
This module asks a protocol-specific question instead: in the raw segment used
by an experiment, were waveform-like events for a sampled RGC detected and
assigned to the correct cluster?

The detector is deliberately liberal and uses one peak channel only to propose
candidate times. Every decision after that uses the waveform across the
target's strongest channels and empirical templates for spatially overlapping
clusters. Thus sharing a peak electrode is never treated as sharing a cell.

The main entry points are:

``load_kilosort_output``
    Read standard Kilosort arrays without loading the raw recording.
``analyze_unit_sorting_qc``
    Analyze one cluster in an already-loaded raw segment.
``analyze_protocol_sorting_qc``
    Load one epoch/window from a ``MEAResponseBlock`` and analyze several
    sampled protocol cell IDs with one shared raw read.
``plot_unit_sorting_qc``
    The four-panel diagnostic page described in the project notebook.
``plot_sampled_detected_spikes``
    Raw peak-channel waveform overlays for every sampled protocol cell.

This is sampled QC, not a replacement sorter. Candidate events that look like
collisions or are nearly tied between templates are reported separately and
are not counted as missed or misassigned spikes.

Typical protocol-notebook use::

    sampled = ra.sample_cells_by_type(
        qc.query('passes'), cell_types=('OnM', 'OnP'),
        n_cells_per_type=5, random_seed=0)
    SORT_QC, SORT_QC_RESULTS, SORT_QC_RAW = ra.analyze_protocol_sorting_qc(
        pipeline.resp, sampled.cell_id, epoch_index=3,
        window_s=(0, 60), min_empirical_spikes=20)
    display(SORT_QC)
    ra.plot_sorting_qc_summary(SORT_QC)
    ra.plot_unit_sorting_qc(SORT_QC_RESULTS[sampled.cell_id.iloc[0]])

For an ordinary sample-major ``int16`` binary, use
``analyze_binary_sorting_qc`` instead. Both high-level paths return the same
table and per-unit result objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


__all__ = [
    'KilosortOutput',
    'UnitSortingQC',
    'load_kilosort_output',
    'load_binary_segment',
    'extract_multichannel_waveforms',
    'empirical_template',
    'template_similarity',
    'strong_template_channels',
    'nearby_template_clusters',
    'detect_candidate_events',
    'refractory_violation_fraction',
    'refractory_contamination_estimate',
    'analyze_unit_sorting_qc',
    'analyze_binary_sorting_qc',
    'sorting_qc_table',
    'analyze_protocol_sorting_qc',
    'plot_unit_sorting_qc',
    'browse_unit_sorting_qc',
    'plot_sorting_qc_summary',
    'plot_sampled_detected_spikes',
    'browse_sampled_detected_spikes',
]


@dataclass
class KilosortOutput:
    """The Kilosort arrays needed for local waveform QC.

    ``templates`` is normalized to ``(template, time, channel-position)``.
    ``channel_map`` converts template channel positions to raw channel IDs.
    Cluster IDs are kept exactly as Kilosort wrote them; protocol/Vision cell
    IDs are converted only by :func:`analyze_protocol_sorting_qc`.
    """

    spike_times: np.ndarray
    spike_clusters: np.ndarray
    templates: np.ndarray
    channel_map: np.ndarray
    spike_templates: Optional[np.ndarray] = None
    channel_positions: Optional[np.ndarray] = None
    sample_rate_hz: float = 20_000.0
    source: Optional[Path] = None
    _cluster_template_cache: Dict[int, np.ndarray] = field(
        default_factory=dict, init=False, repr=False)
    _cluster_template_weights: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.spike_times = np.asarray(self.spike_times).reshape(-1).astype(np.int64)
        self.spike_clusters = np.asarray(self.spike_clusters).reshape(-1).astype(np.int64)
        self.channel_map = np.asarray(self.channel_map).reshape(-1).astype(np.int64)
        if self.channel_positions is not None:
            positions = np.asarray(self.channel_positions, dtype=float)
            if positions.ndim != 2 or positions.shape[0] != len(self.channel_map) \
                    or positions.shape[1] < 2:
                raise ValueError(
                    'channel_positions must be n_channels × 2 (or wider)')
            self.channel_positions = positions[:, :2]
        templates = np.asarray(self.templates)
        if templates.ndim != 3:
            raise ValueError(f'templates must be 3-D; got {templates.shape}')
        if templates.shape[-1] == len(self.channel_map):
            self.templates = templates
        elif templates.shape[1] == len(self.channel_map):
            self.templates = templates.transpose(0, 2, 1)
        else:
            raise ValueError(
                f'no templates axis matches channel_map ({templates.shape} vs '
                f'{len(self.channel_map)} channels)')
        if len(self.spike_times) != len(self.spike_clusters):
            raise ValueError('spike_times and spike_clusters have different lengths')
        if self.spike_templates is not None:
            self.spike_templates = np.asarray(self.spike_templates).reshape(-1).astype(np.int64)
            if len(self.spike_templates) != len(self.spike_times):
                raise ValueError('spike_templates and spike_times have different lengths')
            valid = ((self.spike_templates >= 0)
                     & (self.spike_templates < len(self.templates))
                     & (self.spike_clusters >= 0))
            # One O(n_spikes) grouping here avoids scanning every spike once
            # per cluster during the spatial-neighbor search.
            n_templates = len(self.templates)
            codes = (self.spike_clusters[valid] * n_templates
                     + self.spike_templates[valid])
            unique, counts = np.unique(codes, return_counts=True)
            clusters = unique // n_templates
            template_ids = unique % n_templates
            for cluster_id in np.unique(clusters):
                keep = clusters == cluster_id
                self._cluster_template_weights[int(cluster_id)] = (
                    template_ids[keep].astype(int), counts[keep].astype(float))
        self.sample_rate_hz = float(self.sample_rate_hz)

    @property
    def cluster_ids(self) -> np.ndarray:
        return np.unique(self.spike_clusters)

    def cluster_template(self, cluster_id: int) -> np.ndarray:
        """Weighted Kilosort template for a cluster, including template drift."""
        cluster_id = int(cluster_id)
        if cluster_id in self._cluster_template_cache:
            return self._cluster_template_cache[cluster_id]
        if self.spike_templates is None:
            if cluster_id not in set(self.cluster_ids):
                raise KeyError(f'cluster {cluster_id} has no Kilosort spikes')
            if not 0 <= cluster_id < len(self.templates):
                raise KeyError(
                    f'cluster {cluster_id} is not a template index and '
                    'spike_templates.npy was not loaded')
            value = np.asarray(self.templates[cluster_id], dtype=np.float32)
            self._cluster_template_cache[cluster_id] = value
            return value
        ids, counts = self._cluster_template_weights.get(
            cluster_id, (np.array([], dtype=int), np.array([], dtype=float)))
        if not len(ids):
            raise KeyError(f'cluster {cluster_id} has no valid template IDs')
        value = np.average(np.asarray(self.templates[ids], dtype=np.float32),
                           axis=0, weights=counts)
        self._cluster_template_cache[cluster_id] = value
        return value


@dataclass
class UnitSortingQC:
    """Quantitative and inspectable result for one target Kilosort cluster."""

    target_cluster: int
    local_channel_positions: np.ndarray
    local_channel_xy: Optional[np.ndarray]
    electrode_map_xy: Optional[np.ndarray]
    raw_channel_ids: np.ndarray
    nearby_clusters: np.ndarray
    empirical_templates: Dict[int, np.ndarray]
    assigned_waveforms: np.ndarray
    candidate_waveforms: np.ndarray
    assigned_scores: np.ndarray
    score_matrix: pd.DataFrame
    events: pd.DataFrame
    summary: Dict[str, object]
    sample_rate_hz: float
    segment_start_sample: int


def load_kilosort_output(
    folder,
    *,
    sample_rate_hz: float = 20_000.0,
    mmap_mode: Optional[str] = 'r',
) -> KilosortOutput:
    """Load standard Kilosort output, preserving cluster/template identity."""
    folder = Path(folder)

    def required(name):
        path = folder / name
        if not path.is_file():
            raise FileNotFoundError(path)
        return np.load(path, mmap_mode=mmap_mode)

    spike_templates_path = folder / 'spike_templates.npy'
    channel_positions_path = folder / 'channel_positions.npy'
    return KilosortOutput(
        spike_times=required('spike_times.npy'),
        spike_clusters=required('spike_clusters.npy'),
        templates=required('templates.npy'),
        channel_map=required('channel_map.npy'),
        spike_templates=(np.load(spike_templates_path, mmap_mode=mmap_mode)
                         if spike_templates_path.is_file() else None),
        channel_positions=(np.load(channel_positions_path, mmap_mode=mmap_mode)
                           if channel_positions_path.is_file() else None),
        sample_rate_hz=sample_rate_hz,
        source=folder,
    )


def load_binary_segment(
    raw_path,
    *,
    n_channels: int,
    start_sample: int,
    n_samples: int,
    dtype=np.int16,
    channel_ids: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Memory-map a sample-major interleaved binary and copy one small segment."""
    raw = np.memmap(raw_path, dtype=dtype, mode='r')
    if raw.size % int(n_channels):
        raise ValueError(
            f'{raw.size} values cannot be reshaped into {n_channels} channels')
    raw = raw.reshape(-1, int(n_channels))
    start, stop = int(start_sample), int(start_sample) + int(n_samples)
    if start < 0 or stop > len(raw) or stop <= start:
        raise ValueError(f'invalid raw segment [{start}, {stop}) of {len(raw)} samples')
    ids = (slice(None) if channel_ids is None
           else np.asarray(channel_ids, dtype=int))
    return np.asarray(raw[start:stop, ids], dtype=np.float32)


def extract_multichannel_waveforms(
    data: np.ndarray,
    times: Sequence[int],
    pre_samples: int,
    post_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ``event × time × channel`` waveforms and return valid times."""
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f'data must be samples × channels; got {data.shape}')
    pre, post = int(pre_samples), int(post_samples)
    times = np.asarray(times, dtype=np.int64).reshape(-1)
    valid = times[(times - pre >= 0) & (times + post < len(data))]
    if not len(valid):
        return (np.empty((0, pre + post + 1, data.shape[1]), dtype=np.float32),
                valid)
    waveforms = np.stack([data[t - pre:t + post + 1] for t in valid])
    return np.asarray(waveforms, dtype=np.float32), valid


def empirical_template(waveforms: np.ndarray) -> np.ndarray:
    """Robust median raw waveform for a unit."""
    waveforms = np.asarray(waveforms, dtype=np.float32)
    if waveforms.ndim != 3 or not len(waveforms):
        raise ValueError('waveforms must contain at least one event')
    return np.median(waveforms, axis=0)


def _center_flat(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean(axis=-2, keepdims=True)).reshape(*x.shape[:-2], -1)


def template_similarity(waveforms: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Normalized multichannel dot product after per-channel DC removal."""
    waveforms = np.asarray(waveforms, dtype=np.float32)
    template = np.asarray(template, dtype=np.float32)
    if waveforms.ndim != 3 or template.shape != waveforms.shape[1:]:
        raise ValueError(
            f'expected waveforms N × {template.shape}; got {waveforms.shape}')
    w = _center_flat(waveforms)
    t = _center_flat(template[None])[0]
    t_norm = np.linalg.norm(t)
    denom = np.linalg.norm(w, axis=1) * t_norm
    return np.divide(w @ t, denom, out=np.zeros(len(w), dtype=float),
                     where=denom > 0)


def strong_template_channels(
    ks: KilosortOutput,
    cluster_id: int,
    *,
    n_channels: int = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Strong template channel positions and corresponding raw channel IDs."""
    template = ks.cluster_template(cluster_id)
    ptp = np.ptp(template, axis=0)
    n = min(max(1, int(n_channels)), len(ptp))
    positions = np.argsort(ptp)[-n:][::-1]
    return positions.astype(int), ks.channel_map[positions].astype(int)


def nearby_template_clusters(
    ks: KilosortOutput,
    target_cluster: int,
    *,
    min_spatial_overlap: float = 0.20,
    max_clusters: int = 12,
) -> pd.DataFrame:
    """Rank competitors by cosine overlap of multichannel PTP footprints."""
    target = np.ptp(ks.cluster_template(target_cluster), axis=0).astype(float)
    target_norm = np.linalg.norm(target)
    rows = []
    for cluster_id in ks.cluster_ids:
        footprint = np.ptp(ks.cluster_template(int(cluster_id)), axis=0).astype(float)
        denom = target_norm * np.linalg.norm(footprint)
        overlap = float(target @ footprint / denom) if denom else 0.0
        if int(cluster_id) == int(target_cluster) or overlap >= min_spatial_overlap:
            rows.append({'cluster_id': int(cluster_id),
                         'spatial_overlap': overlap,
                         'peak_channel_position': int(np.argmax(footprint)),
                         'peak_raw_channel': int(ks.channel_map[np.argmax(footprint)])})
    out = pd.DataFrame(rows).sort_values(
        ['spatial_overlap', 'cluster_id'], ascending=[False, True])
    target = out[out.cluster_id == int(target_cluster)]
    others = out[out.cluster_id != int(target_cluster)].head(
        max(0, int(max_clusters) - 1))
    return pd.concat([target, others], ignore_index=True)


def detect_candidate_events(
    signal: np.ndarray,
    *,
    threshold_sigma: float = 4.0,
    polarity: str = 'negative',
    dead_samples: int = 14,
    snap_samples: int = 10,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Liberal threshold detector used only to propose candidate times."""
    signal = np.asarray(signal, dtype=float).reshape(-1)
    centered = signal - np.median(signal)
    mad = float(np.median(np.abs(centered)))
    sigma = mad / 0.67448975 if mad > 0 else float(np.std(centered))
    if not np.isfinite(sigma) or sigma <= 0:
        return np.array([], dtype=int), {'sigma': sigma, 'threshold': np.nan}
    if polarity not in ('negative', 'positive'):
        raise ValueError("polarity must be 'negative' or 'positive'")
    threshold = (-1 if polarity == 'negative' else 1) * float(threshold_sigma) * sigma
    if polarity == 'negative':
        crossings = np.flatnonzero(
            (centered[1:] < threshold) & (centered[:-1] >= threshold)) + 1
    else:
        crossings = np.flatnonzero(
            (centered[1:] > threshold) & (centered[:-1] <= threshold)) + 1
    snapped = []
    for crossing in crossings:
        lo, hi = max(0, crossing - snap_samples), min(len(signal), crossing + snap_samples + 1)
        local = centered[lo:hi]
        peak = lo + int(np.argmin(local) if polarity == 'negative'
                        else np.argmax(local))
        if not snapped or peak - snapped[-1] > int(dead_samples):
            snapped.append(peak)
        elif ((polarity == 'negative' and centered[peak] < centered[snapped[-1]])
              or (polarity == 'positive' and centered[peak] > centered[snapped[-1]])):
            snapped[-1] = peak
    return np.asarray(snapped, dtype=int), {
        'sigma': sigma, 'threshold': threshold, 'threshold_sigma': float(threshold_sigma)}


def refractory_violation_fraction(
    spike_times: Sequence[int],
    *,
    sample_rate_hz: float = 20_000.0,
    refractory_ms: float = 1.5,
) -> float:
    """Fraction of consecutive ISIs shorter than ``refractory_ms``."""
    times = np.sort(np.asarray(spike_times, dtype=np.int64).reshape(-1))
    if len(times) < 2:
        return np.nan
    limit = float(refractory_ms) / 1000.0 * float(sample_rate_hz)
    return float(np.mean(np.diff(times) < limit))


def refractory_contamination_estimate(
    spike_times: Sequence[int],
    *,
    duration_s: float,
    sample_rate_hz: float = 20_000.0,
    refractory_ms: float = 1.5,
    censored_ms: float = 0.3,
) -> float:
    """Hill et al. refractory-violation estimate of false-positive fraction.

    This solves

    ``r = 2 (tau_R - tau_C) N^2 (1-f) f / T``

    for the smaller root ``f``. It assumes an uncorrelated contaminating point
    process and that the target supplies the majority of the cluster. Retina
    refractory periods vary, so both time constants are explicit. ``NaN``
    means the observed violation count is too high for those assumptions to
    have a real-valued solution; it must not be interpreted as zero.
    """
    times = np.sort(np.asarray(spike_times, dtype=np.int64).reshape(-1))
    n = len(times)
    tau = (float(refractory_ms) - float(censored_ms)) / 1000.0
    if n < 2 or duration_s <= 0 or tau <= 0:
        return np.nan
    isi_ms = np.diff(times) / float(sample_rate_hz) * 1000.0
    violations = int(((isi_ms > float(censored_ms))
                      & (isi_ms < float(refractory_ms))).sum())
    a = violations * float(duration_s) / (2.0 * tau * n * n)
    discriminant = 1.0 - 4.0 * a
    return (float((1.0 - np.sqrt(discriminant)) / 2.0)
            if discriminant >= 0 else np.nan)


def _nearest_assignments(query, times, clusters, tolerance):
    order = np.argsort(times)
    times = np.asarray(times, dtype=np.int64)[order]
    clusters = np.asarray(clusters, dtype=np.int64)[order]
    assigned = np.full(len(query), -1, dtype=np.int64)
    delta = np.full(len(query), np.nan)
    counts = np.zeros(len(query), dtype=int)
    for i, value in enumerate(np.asarray(query, dtype=np.int64)):
        lo = np.searchsorted(times, value - tolerance, side='left')
        hi = np.searchsorted(times, value + tolerance, side='right')
        counts[i] = hi - lo
        if hi > lo:
            j = lo + int(np.argmin(np.abs(times[lo:hi] - value)))
            assigned[i], delta[i] = int(clusters[j]), float(times[j] - value)
    return assigned, delta, counts


def analyze_unit_sorting_qc(
    raw_segment: np.ndarray,
    ks: KilosortOutput,
    target_cluster: int,
    *,
    segment_start_sample: int = 0,
    n_local_channels: int = 9,
    pre_ms: float = 1.0,
    post_ms: float = 2.0,
    threshold_sigma: float = 4.0,
    similarity_percentile: float = 2.0,
    min_empirical_spikes: int = 20,
    assignment_tolerance_ms: float = 0.3,
    collision_window_ms: float = 0.7,
    winner_margin: float = 0.05,
    refractory_ms: float = 1.5,
    refractory_censored_ms: float = 0.3,
    min_spatial_overlap: float = 0.20,
    max_competitors: int = 12,
    electrode_map_xy: Optional[np.ndarray] = None,
) -> UnitSortingQC:
    """Quantify missed detection, misassignment, collision, and RP violations.

    ``raw_segment`` must be ``samples × raw channels`` and must use the same
    raw channel numbering as ``channel_map.npy``. It is median-centered per
    selected channel internally. Candidate thresholds are calibrated from the
    segment; similarity thresholds are calibrated from the target's assigned
    spikes, not hard-coded.
    """
    fs = float(ks.sample_rate_hz)
    target_cluster = int(target_cluster)
    raw_segment = np.asarray(raw_segment)
    if raw_segment.ndim != 2:
        raise ValueError('raw_segment must be samples × raw channels')
    positions, raw_ids = strong_template_channels(
        ks, target_cluster, n_channels=n_local_channels)
    if raw_ids.min() < 0 or raw_ids.max() >= raw_segment.shape[1]:
        raise IndexError(
            f'channel_map requests raw channel {raw_ids.min()}–{raw_ids.max()}, '
            f'but raw_segment has {raw_segment.shape[1]} channels')
    local = np.asarray(raw_segment[:, raw_ids], dtype=np.float32)
    local -= np.median(local, axis=0, keepdims=True)
    pre = int(round(float(pre_ms) / 1000.0 * fs))
    post = int(round(float(post_ms) / 1000.0 * fs))
    start, stop = int(segment_start_sample), int(segment_start_sample) + len(local)
    in_segment = (ks.spike_times >= start) & (ks.spike_times < stop)
    segment_times = ks.spike_times[in_segment] - start
    segment_clusters = ks.spike_clusters[in_segment]
    target_times = segment_times[segment_clusters == target_cluster]
    assigned_wfs, _ = extract_multichannel_waveforms(
        local, target_times, pre, post)
    if len(assigned_wfs) < max(3, int(min_empirical_spikes)):
        raise ValueError(
            f'cluster {target_cluster} has {len(assigned_wfs)} valid spikes in '
            f'this segment; need at least {max(3, int(min_empirical_spikes))}')
    target_template = empirical_template(assigned_wfs)
    assigned_scores = template_similarity(assigned_wfs, target_template)
    similarity_threshold = float(np.percentile(
        assigned_scores, float(similarity_percentile)))

    nearby = nearby_template_clusters(
        ks, target_cluster, min_spatial_overlap=min_spatial_overlap,
        max_clusters=max_competitors)
    templates_empirical: Dict[int, np.ndarray] = {target_cluster: target_template}
    for cluster_id in nearby.cluster_id.astype(int):
        if cluster_id == target_cluster:
            continue
        times = segment_times[segment_clusters == cluster_id]
        wfs, _ = extract_multichannel_waveforms(local, times, pre, post)
        if len(wfs) >= int(min_empirical_spikes):
            templates_empirical[cluster_id] = empirical_template(wfs)

    # The Kilosort footprint identifies the peak proposal channel; the raw
    # empirical target decides its polarity.
    peak_local = int(np.argmax(np.ptp(target_template, axis=0)))
    peak_wave = target_template[:, peak_local]
    polarity = ('negative' if abs(float(peak_wave.min())) >= abs(float(peak_wave.max()))
                else 'positive')
    candidate_times, detector = detect_candidate_events(
        local[:, peak_local], threshold_sigma=threshold_sigma,
        polarity=polarity, dead_samples=max(1, int(round(0.0007 * fs))),
        snap_samples=max(1, int(round(0.0005 * fs))))
    candidate_wfs, candidate_times = extract_multichannel_waveforms(
        local, candidate_times, pre, post)
    cluster_order = np.array(sorted(templates_empirical), dtype=int)
    scores = np.column_stack([
        template_similarity(candidate_wfs, templates_empirical[c])
        for c in cluster_order
    ]) if len(candidate_wfs) else np.empty((0, len(cluster_order)))
    target_col = int(np.flatnonzero(cluster_order == target_cluster)[0])
    target_scores = scores[:, target_col] if len(scores) else np.array([])
    rank = np.argsort(scores, axis=1)[:, ::-1] if len(scores) else np.empty((0, 0), int)
    winner = cluster_order[rank[:, 0]] if len(scores) else np.array([], int)
    best = scores[np.arange(len(scores)), rank[:, 0]] if len(scores) else np.array([])
    second = (scores[np.arange(len(scores)), rank[:, 1]]
              if scores.shape[1] > 1 else np.full(len(scores), -np.inf))
    margin = best - second

    tol = int(round(float(assignment_tolerance_ms) / 1000.0 * fs))
    assigned_cluster, assignment_delta, _ = _nearest_assignments(
        candidate_times, segment_times, segment_clusters, tol)
    collision_tol = int(round(float(collision_window_ms) / 1000.0 * fs))
    _, _, collision_counts = _nearest_assignments(
        candidate_times, segment_times, segment_clusters, collision_tol)
    target_like = target_scores >= similarity_threshold
    target_wins = (winner == target_cluster) & (margin >= float(winner_margin))
    collision = target_like & ((collision_counts > 1) | (margin < float(winner_margin)))
    status = np.full(len(candidate_times), 'background/competitor', dtype=object)
    status[target_like & (assigned_cluster == target_cluster)] = 'assigned_target'
    status[target_like & target_wins & (assigned_cluster < 0) & ~collision] = 'missed_detection'
    status[target_like & target_wins & (assigned_cluster >= 0)
           & (assigned_cluster != target_cluster) & ~collision] = 'misassigned'
    status[target_like & collision] = 'possible_collision'
    events = pd.DataFrame({
        'candidate_sample': candidate_times,
        'absolute_sample': candidate_times + start,
        'time_in_segment_s': candidate_times / fs,
        'target_score': target_scores,
        'similarity_threshold': similarity_threshold,
        'target_like': target_like,
        'winner_cluster': winner,
        'winner_score': best,
        'winner_margin': margin,
        'nearest_ks_cluster': np.where(assigned_cluster < 0, np.nan, assigned_cluster),
        'assignment_delta_samples': assignment_delta,
        'ks_events_within_collision_window': collision_counts,
        'status': status,
    })
    score_frame = pd.DataFrame(scores, columns=cluster_order)
    n_assigned = int(len(target_times))
    n_missed = int((status == 'missed_detection').sum())
    n_misassigned = int((status == 'misassigned').sum())
    n_collision = int((status == 'possible_collision').sum())
    summary = {
        'target_cluster': target_cluster,
        'n_ks_assigned': n_assigned,
        'n_candidates': int(len(candidate_times)),
        'n_template_like': int(target_like.sum()),
        'n_missed': n_missed,
        'n_misassigned': n_misassigned,
        'n_possible_collisions': n_collision,
        'detection_miss_fraction': (
            n_missed / (n_assigned + n_missed) if n_assigned + n_missed else np.nan),
        'misassignment_fraction': (
            n_misassigned / (n_assigned + n_misassigned)
            if n_assigned + n_misassigned else np.nan),
        'refractory_violation_fraction': refractory_violation_fraction(
            target_times, sample_rate_hz=fs, refractory_ms=refractory_ms),
        'refractory_contamination_estimate': refractory_contamination_estimate(
            target_times, duration_s=len(local) / fs, sample_rate_hz=fs,
            refractory_ms=refractory_ms,
            censored_ms=refractory_censored_ms),
        'refractory_ms': float(refractory_ms),
        'refractory_censored_ms': float(refractory_censored_ms),
        'similarity_threshold': similarity_threshold,
        'similarity_percentile': float(similarity_percentile),
        'n_local_channels': int(len(raw_ids)),
        'raw_channel_ids': raw_ids.tolist(),
        'n_competing_templates': int(len(templates_empirical) - 1),
        'nearby_clusters': nearby.cluster_id.astype(int).tolist(),
        'detector_sigma': detector['sigma'],
        'detector_threshold': detector['threshold'],
        'candidate_polarity': polarity,
        'segment_start_sample': start,
        'segment_n_samples': int(len(local)),
        'segment_duration_s': float(len(local) / fs),
    }
    full_xy = None
    local_xy = None
    if electrode_map_xy is not None:
        candidate_xy = np.asarray(electrode_map_xy, dtype=float)
        if candidate_xy.ndim != 2 or candidate_xy.shape[1] < 2:
            raise ValueError('electrode_map_xy must be n_electrodes × 2')
        if raw_ids.max() >= len(candidate_xy):
            raise IndexError(
                f'electrode map has {len(candidate_xy)} rows but raw channel '
                f'{raw_ids.max()} was selected')
        full_xy = candidate_xy[:, :2].copy()
        local_xy = full_xy[raw_ids]
    elif ks.channel_positions is not None:
        full_xy = ks.channel_positions.copy()
        local_xy = ks.channel_positions[positions].copy()

    return UnitSortingQC(
        target_cluster=target_cluster,
        local_channel_positions=positions,
        local_channel_xy=local_xy,
        electrode_map_xy=full_xy,
        raw_channel_ids=raw_ids,
        nearby_clusters=nearby.cluster_id.to_numpy(dtype=int),
        empirical_templates=templates_empirical,
        assigned_waveforms=assigned_wfs,
        candidate_waveforms=candidate_wfs,
        assigned_scores=assigned_scores,
        score_matrix=score_frame,
        events=events,
        summary=summary,
        sample_rate_hz=fs,
        segment_start_sample=start,
    )


def sorting_qc_table(results: Mapping[int, UnitSortingQC]) -> pd.DataFrame:
    """One compact row per sampled unit."""
    return pd.DataFrame([dict(result.summary) for result in results.values()])


def analyze_binary_sorting_qc(
    raw_path,
    kilosort_dir,
    target_clusters: Iterable[int],
    *,
    n_channels: int,
    start_sample: int,
    n_samples: int,
    dtype=np.int16,
    sample_rate_hz: float = 20_000.0,
    verbose: bool = True,
    **analysis_kwargs,
):
    """Convenience entry point for a standard interleaved binary + KS folder."""
    ks = load_kilosort_output(kilosort_dir, sample_rate_hz=sample_rate_hz)
    raw = load_binary_segment(
        raw_path, n_channels=n_channels, start_sample=start_sample,
        n_samples=n_samples, dtype=dtype)
    results = {}
    for cluster_id in target_clusters:
        cluster_id = int(cluster_id)
        if verbose:
            print(f'analyzing Kilosort cluster {cluster_id}')
        results[cluster_id] = analyze_unit_sorting_qc(
            raw, ks, cluster_id, segment_start_sample=int(start_sample),
            **analysis_kwargs)
    table = sorting_qc_table(results)
    if verbose and not table.empty:
        print(table.to_string(index=False))
    return table, results, ks


def analyze_protocol_sorting_qc(
    response_block,
    cell_ids: Iterable[int],
    epoch_index: int,
    *,
    window_s: Tuple[float, float] = (0.0, 60.0),
    kilosort_dir=None,
    electrode_map_xy=None,
    vision_id_offset: int = 1,
    verbose: bool = True,
    **analysis_kwargs,
):
    """Protocol adapter: one shared raw read, then multichannel QC per cell.

    The repository's Kilosort-to-Vision conversion writes Vision cell ID as
    ``cluster_id + 1``. ``vision_id_offset=1`` therefore maps protocol IDs back
    to Kilosort clusters. Set it to 0 when passing native Kilosort IDs.

    Run this before deduplicated clusters have had their spike trains unioned;
    a merged protocol cell no longer corresponds to one Kilosort cluster.
    """
    from ..classes.raw import RawTraces
    from ..config.settings import DATA_DIR

    if kilosort_dir is None:
        kilosort_dir = (Path(DATA_DIR) / response_block.exp_name
                        / response_block.datafile_name / response_block.ss_version)
    ks = load_kilosort_output(kilosort_dir)
    if electrode_map_xy is None:
        electrode_map_xy = np.asarray(
            response_block.vcd.get_electrode_map(), dtype=float)
    raw = RawTraces(response_block)
    start_s, end_s = (float(v) for v in window_s)
    raw.load_window(int(epoch_index), start_s, end_s, verbose=False)
    segment_start = (int(response_block.d_timing['epochStarts'][int(epoch_index)])
                     + int(round(start_s * raw.sample_rate)))
    raw_segment = np.asarray(raw.data.T, dtype=np.float32)
    results = {}
    type_map = (response_block.df_spike_times.set_index('cell_id')['cell_type'].to_dict()
                if 'cell_type' in response_block.df_spike_times else {})
    for cell_id in cell_ids:
        cell_id = int(cell_id)
        cluster_id = cell_id - int(vision_id_offset)
        if verbose:
            print(f'cell {cell_id} ({type_map.get(cell_id, "untyped")}) -> '
                  f'Kilosort cluster {cluster_id}')
        result = analyze_unit_sorting_qc(
            raw_segment, ks, cluster_id,
            segment_start_sample=segment_start,
            electrode_map_xy=electrode_map_xy, **analysis_kwargs)
        result.summary['cell_id'] = cell_id
        result.summary['cell_type'] = type_map.get(cell_id, '')
        result.summary['epoch_index'] = int(epoch_index)
        result.summary['window_start_s'] = start_s
        result.summary['window_end_s'] = end_s
        results[cell_id] = result
    table = sorting_qc_table(results)
    if verbose and not table.empty:
        columns = ['cell_id', 'cell_type', 'n_ks_assigned', 'n_missed',
                   'n_misassigned', 'n_possible_collisions',
                   'detection_miss_fraction', 'misassignment_fraction',
                   'refractory_contamination_estimate',
                   'refractory_violation_fraction']
        print(table[[c for c in columns if c in table]].to_string(index=False))
    return table, results, raw


def plot_unit_sorting_qc(
    result: UnitSortingQC,
    *,
    max_suspicious_waveforms: int = 20,
    competitor_cluster: Optional[int] = None,
    figsize: Tuple[float, float] = (12.0, 8.0),
):
    """Four panels: score calibration, candidates, waveforms, competition."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    threshold = float(result.summary['similarity_threshold'])
    axes[0, 0].hist(result.assigned_scores, bins=40, color='0.35', alpha=0.85)
    axes[0, 0].axvline(threshold, color='crimson', ls='--',
                       label=f'threshold {threshold:.3f}')
    axes[0, 0].set(xlabel='similarity to empirical target template',
                   ylabel='assigned spikes', title='A. assigned-score calibration')
    axes[0, 0].legend(fontsize=8)

    colors = {
        'assigned_target': '#009E73', 'missed_detection': '#D55E00',
        'misassigned': '#CC79A7', 'possible_collision': '#E69F00',
        'background/competitor': '0.65',
    }
    for status, group in result.events.groupby('status', sort=False):
        axes[0, 1].hist(group.target_score, bins=35, alpha=0.55,
                        color=colors.get(status, '0.5'), label=f'{status} ({len(group)})')
    axes[0, 1].axvline(threshold, color='crimson', ls='--')
    axes[0, 1].set(xlabel='similarity to target', ylabel='candidate events',
                   title='B. independent detector candidates')
    axes[0, 1].legend(fontsize=7)

    suspicious = result.events.status.isin(
        ['missed_detection', 'misassigned', 'possible_collision']).to_numpy()
    indices = np.flatnonzero(suspicious)[:int(max_suspicious_waveforms)]
    ax = axes[1, 0]
    if len(indices):
        waveforms = result.candidate_waveforms[indices]
        scale = float(np.nanpercentile(np.abs(waveforms), 95)) or 1.0
        target_peak = result.empirical_templates[result.target_cluster]
        center = int(np.argmax(np.max(np.abs(target_peak), axis=1)))
        time_ms = ((np.arange(waveforms.shape[1]) - center)
                   / result.sample_rate_hz * 1000.0)
        for waveform in waveforms:
            for channel in range(waveform.shape[1]):
                ax.plot(time_ms, waveform[:, channel] / scale + 2 * channel,
                        color='0.25', lw=0.5, alpha=0.35)
        ax.set(xlabel='time around candidate (ms)', ylabel='local channel (offset)',
               title=f'C. suspicious events (first {len(indices)})')
    else:
        ax.text(0.5, 0.5, 'no missed/misassigned/collision candidates',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('C. suspicious events')

    competitors = [c for c in result.score_matrix.columns
                   if int(c) != result.target_cluster]
    if competitor_cluster is None and competitors:
        competitor_cluster = int(max(
            competitors,
            key=lambda c: float(result.score_matrix[c].mean())))
    ax = axes[1, 1]
    if competitor_cluster is not None and competitor_cluster in result.score_matrix:
        target = result.score_matrix[result.target_cluster]
        other = result.score_matrix[competitor_cluster]
        for status, group in result.events.groupby('status', sort=False):
            idx = group.index
            ax.scatter(target.iloc[idx], other.iloc[idx], s=12, alpha=0.65,
                       color=colors.get(status, '0.5'), label=status)
        lo = min(float(target.min()), float(other.min()), -0.1)
        hi = max(float(target.max()), float(other.max()), 1.0)
        ax.plot([lo, hi], [lo, hi], '--', color='0.4', lw=0.8)
        ax.set(xlabel=f'similarity to target {result.target_cluster}',
               ylabel=f'similarity to competitor {competitor_cluster}',
               title='D. multichannel template competition')
    else:
        ax.text(0.5, 0.5, 'no competitor with enough spikes',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('D. multichannel template competition')
    summary = result.summary
    fig.suptitle(
        f'cluster {result.target_cluster}: {summary["n_ks_assigned"]} KS spikes; '
        f'{summary["n_missed"]} missed, {summary["n_misassigned"]} misassigned, '
        f'{summary["n_possible_collisions"]} collision/ambiguous', y=1.01)
    fig.tight_layout()
    return fig, axes


def plot_sorting_qc_summary(
    table: pd.DataFrame,
    *,
    label_column: str = 'cell_id',
    figsize: Tuple[float, float] = (10.0, 4.0),
):
    """Compact sampled-unit comparison of miss, misassignment, and RP rates."""
    import matplotlib.pyplot as plt

    refractory_column = ('refractory_contamination_estimate'
                          if 'refractory_contamination_estimate' in table
                          else 'refractory_violation_fraction')
    required = {'detection_miss_fraction', 'misassignment_fraction',
                refractory_column}
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f'sorting QC table missing: {sorted(missing)}')
    labels = table[label_column].astype(str).tolist()
    x = np.arange(len(table), dtype=float)
    width = 0.25
    fig, ax = plt.subplots(figsize=figsize)
    for offset, column, label in (
            (-width, 'detection_miss_fraction', 'missed detection'),
            (0.0, 'misassignment_fraction', 'misassigned'),
            (width, refractory_column,
             'RP contamination' if refractory_column.endswith('estimate')
             else 'RP violations')):
        ax.bar(x + offset, 100 * table[column].to_numpy(dtype=float),
               width=width, label=label)
    ax.set_xticks(x, labels, rotation=45, ha='right')
    ax.set_ylabel('events (%)')
    ax.set_title('protocol-specific multichannel sorting QC')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_sampled_detected_spikes(
    results: Mapping[int, UnitSortingQC],
    *,
    max_waveforms_per_class: int = 30,
    n_neighbor_electrodes: int = 48,
    figsize_per_cell: Tuple[float, float] = (19.0, 3.2),
):
    """Array-map target/suspicious spikes, assignments, and similarity.

    The first two panels use the canonical Vision electrode map supplied by
    ``response_block.vcd.get_electrode_map()``. They place the assigned target
    median and suspicious events on the actual array, with only the nearest
    ``n_neighbor_electrodes`` shown for readable local context. The remaining
    panels show assigned spikes and multichannel target similarity.

    "Raw threshold candidate" has a precise meaning here: a liberal threshold
    crossing proposed from the strongest channel without consulting Kilosort
    labels. Only after proposal is its multichannel waveform compared with the
    target and nearby-cell templates. This is not a second spike assignment.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from .style import colors_for_celltypes

    if not results:
        raise ValueError('results is empty')
    if len(results) != 1:
        raise ValueError(
            'plot_sampled_detected_spikes shows one cluster; use '
            'browse_sampled_detected_spikes for multiple sampled clusters')
    items = list(results.items())
    fig, axes = plt.subplots(
        len(items), 4, squeeze=False,
        figsize=(float(figsize_per_cell[0]),
                 float(figsize_per_cell[1]) * len(items)))
    colors = {
        'assigned_target': '#009E73',
        'missed_detection': '#D55E00',
        'misassigned': '#CC79A7',
        'possible_collision': '#E69F00',
        'background/competitor': '0.65',
    }
    type_colors = colors_for_celltypes(sorted({
        str(result.summary.get('cell_type', '')) for _, result in items}))
    rng = np.random.default_rng(0)

    def _limited(indices):
        indices = np.asarray(indices, dtype=int)
        limit = max(1, int(max_waveforms_per_class))
        if len(indices) <= limit:
            return indices
        return np.sort(rng.choice(indices, size=limit, replace=False))

    def _spatial_waveform(ax, waveform, result, title, *,
                          event_waveforms=None, event_colors=None):
        xy = result.local_channel_xy
        layout_note = ''
        if xy is None:
            n_cols = int(np.ceil(np.sqrt(len(result.raw_channel_ids))))
            xy = np.column_stack((np.arange(len(result.raw_channel_ids)) % n_cols,
                                  np.arange(len(result.raw_channel_ids)) // n_cols))
            layout_note = ' (index-grid fallback)'
        xy = np.asarray(xy, dtype=float)
        full_xy = result.electrode_map_xy
        if full_xy is not None:
            full_xy = np.asarray(full_xy, dtype=float)
            center = np.mean(xy, axis=0)
            distance = np.linalg.norm(full_xy - center[None, :], axis=1)
            n_context = min(max(len(xy), int(n_neighbor_electrodes)),
                            len(full_xy))
            context_idx = np.argsort(distance)[:n_context]
            context_xy = full_xy[context_idx]
            ax.scatter(context_xy[:, 0], context_xy[:, 1],
                       s=5, color='0.82',
                       zorder=0, rasterized=True)
        else:
            context_xy = xy
        if waveform is None:
            ax.scatter(xy[:, 0], xy[:, 1], s=12, facecolors='none',
                       edgecolors='0.35')
            ax.text(0.5, 0.5, 'no missed, misassigned,\nor collision events',
                    transform=ax.transAxes, ha='center', va='center', fontsize=8)
        else:
            unique_x = np.unique(xy[:, 0])
            unique_y = np.unique(xy[:, 1])
            spacings = np.r_[np.diff(unique_x), np.diff(unique_y)]
            spacings = np.abs(spacings[np.abs(spacings) > 0])
            pitch = float(np.min(spacings)) if len(spacings) else 1.0
            global_amp = float(np.max(np.abs(waveform))) or 1.0
            if event_waveforms is not None and len(event_waveforms):
                global_amp = max(
                    global_amp, float(np.max(np.abs(event_waveforms))) or 1.0)
            latencies = np.argmax(np.abs(waveform), axis=0)
            latency_norm = Normalize(
                float(latencies.min()),
                float(max(latencies.max(), latencies.min() + 1)))
            cmap = plt.get_cmap('viridis')
            trace_x = np.linspace(-0.38 * pitch, 0.38 * pitch,
                                  waveform.shape[0])
            for channel, (x_pos, y_pos) in enumerate(xy):
                if event_waveforms is not None:
                    for event_index, event in enumerate(event_waveforms):
                        event_color = (event_colors[event_index]
                                       if event_colors is not None else '0.45')
                        event_y = (event[:, channel] / global_amp
                                   * 0.38 * pitch)
                        ax.plot(x_pos + trace_x, y_pos + event_y,
                                color=event_color, lw=0.45, alpha=0.18)
                trace_y = waveform[:, channel] / global_amp * 0.38 * pitch
                ax.plot(x_pos + trace_x, y_pos + trace_y,
                        color=cmap(latency_norm(latencies[channel])), lw=1.15)
                ax.text(x_pos, y_pos - 0.46 * pitch,
                        str(int(result.raw_channel_ids[channel])),
                        ha='center', va='top', fontsize=5.5, color='0.3')
            ax.scatter(xy[:, 0], xy[:, 1], s=7, color='black', zorder=3)
        shown_xy = np.vstack([context_xy, xy])
        x_span = float(np.ptp(shown_xy[:, 0]))
        y_span = float(np.ptp(shown_xy[:, 1]))
        pad = 0.08 * max(x_span, y_span, 1.0)
        ax.set_xlim(float(shown_xy[:, 0].min() - pad),
                    float(shown_xy[:, 0].max() + pad))
        ax.set_ylim(float(shown_xy[:, 1].max() + pad),
                    float(shown_xy[:, 1].min() - pad))
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title + layout_note)
        ax.set_xlabel('electrode x (µm); labels = raw channel')
        ax.set_ylabel('electrode y (µm)')

    for row, (cell_id, result) in enumerate(items):
        target = result.empirical_templates[result.target_cluster]
        peak_channel = int(np.argmax(np.ptp(target, axis=0)))
        peak_sample = int(np.argmax(np.abs(target[:, peak_channel])))
        time_ms = ((np.arange(target.shape[0]) - peak_sample)
                   / result.sample_rate_hz * 1000.0)
        cell_type = str(result.summary.get('cell_type', ''))
        cell_color = type_colors.get(cell_type, '#0072B2')

        # A/B. Target and suspicious spikes on the repository electrode map.
        _spatial_waveform(
            axes[row, 0], target, result,
            f'cell {cell_id} ({cell_type}) — assigned target median')
        suspicious_statuses = [
            'missed_detection', 'misassigned', 'possible_collision']
        suspicious_mask = result.events['status'].isin(
            suspicious_statuses).to_numpy()
        suspicious_waveforms = result.candidate_waveforms[suspicious_mask]
        suspicious_median = (np.median(suspicious_waveforms, axis=0)
                             if len(suspicious_waveforms) else None)
        status_counts = result.events.loc[suspicious_mask, 'status'].value_counts()
        count_label = ', '.join(
            f'{name.replace("_", " ")}={int(count)}'
            for name, count in status_counts.items())
        shown_suspicious = _limited(np.flatnonzero(suspicious_mask))
        shown_waveforms = result.candidate_waveforms[shown_suspicious]
        shown_colors = [colors[str(result.events.iloc[index]['status'])]
                        for index in shown_suspicious]
        _spatial_waveform(
            axes[row, 1], suspicious_median, result,
            f'suspicious events + median (n={len(suspicious_waveforms)}'
            + (f'; {count_label}' if count_label else '') + ')',
            event_waveforms=shown_waveforms, event_colors=shown_colors)

        # C. All assigned spikes overlaid on the strongest electrode.
        assigned = result.assigned_waveforms[:, :, peak_channel]
        chosen = _limited(np.arange(len(assigned)))
        ax = axes[row, 2]
        if len(chosen):
            ax.plot(time_ms, assigned[chosen].T, color=cell_color, lw=0.5,
                    alpha=0.22)
            ax.plot(time_ms, np.median(assigned, axis=0), color='black', lw=1.6)
        ax.axvline(0, color='0.75', lw=0.7, ls=':')
        raw_channel = int(result.raw_channel_ids[peak_channel])
        ax.set_title(f'Kilosort-assigned spikes — raw channel {raw_channel}')
        ax.set_ylabel('raw amplitude')
        ax.legend([Line2D([0], [0], color=cell_color, lw=1.5)],
                  [f'cell {cell_id} — {cell_type} (n={len(assigned)})'],
                  fontsize=7, loc='best')

        # D. Similarity uses every local channel, not only the peak channel.
        ax = axes[row, 3]
        status_order = ['assigned_target', 'missed_detection', 'misassigned',
                        'possible_collision', 'background/competitor']
        y_labels = ['KS assigned target', 'raw candidate: assigned target',
                    'raw candidate: no KS event',
                    'raw candidate: KS assigned another cell',
                    'raw candidate: collision/ambiguous',
                    'raw candidate: other waveform']
        assigned_idx = _limited(np.arange(len(result.assigned_scores)))
        ax.scatter(result.assigned_scores[assigned_idx],
                   np.zeros(len(assigned_idx)), s=10, color=cell_color,
                   alpha=0.45)
        for level, status in enumerate(status_order, start=1):
            indices = _limited(np.flatnonzero(
                result.events['status'].eq(status).to_numpy()))
            if len(indices):
                ax.scatter(result.events.iloc[indices]['target_score'],
                           np.full(len(indices), level), s=11,
                           color=colors[status], alpha=0.55)
        ax.axvline(float(result.summary['similarity_threshold']),
                   color='crimson', lw=1.0, ls='--', label='target threshold')
        ax.set_xlim(-1.02, 1.02)
        ax.set_yticks(np.arange(len(y_labels)), y_labels, fontsize=6)
        ax.set_xlabel('multichannel cosine similarity to target template')
        ax.set_title('Similarity to target (raw threshold candidates)')
        ax.legend(fontsize=7, loc='lower right')
        for panel in axes[row, 2:]:
            panel.grid(alpha=0.15, linewidth=0.5)
        axes[row, 2].set_xlabel('time from waveform peak (ms)')

    fig.suptitle('Sampled cell — target and suspicious spikes on the MEA map',
                 y=1.001)
    fig.tight_layout()
    return fig, axes


def browse_unit_sorting_qc(
    results: Mapping[int, UnitSortingQC],
    *,
    figure_sink: Optional[Dict[int, object]] = None,
    description: str = 'Full sorting QC:',
):
    """Dropdown over the full four-panel diagnostic for sampled clusters."""
    import matplotlib.pyplot as plt
    from .browse import figure_to_png, png_browser

    if not results:
        print('No sampled sorting-QC results to browse.')
        return None
    ordered = [(int(cell_id), result) for cell_id, result in results.items()]

    def _figure(cell_id):
        fig, _ = plot_unit_sorting_qc(results[int(cell_id)])
        return fig

    if figure_sink is not None:
        for cell_id, _ in ordered:
            saved_fig = _figure(cell_id)
            figure_sink[cell_id] = saved_fig
            plt.close(saved_fig)

    def _render(cell_id):
        result = results[int(cell_id)]
        cell_type = str(result.summary.get('cell_type', ''))
        html = (f'<b>cell {int(cell_id)} — {cell_type}; '
                f'Kilosort cluster {result.target_cluster}</b>')
        return html, figure_to_png(_figure(cell_id))

    options = [
        (f'cell {cell_id} — {result.summary.get("cell_type", "")}; '
         f'KS cluster {result.target_cluster}', cell_id)
        for cell_id, result in ordered
    ]
    return png_browser(options, _render, description=description)


def browse_sampled_detected_spikes(
    results: Mapping[int, UnitSortingQC],
    *,
    max_waveforms_per_class: int = 30,
    n_neighbor_electrodes: int = 48,
    figure_sink: Optional[Dict[int, object]] = None,
    description: str = 'Sampled cluster:',
):
    """Dropdown showing one sampled cell's three-panel sorting view at a time.

    Labels retain all identities: protocol/Vision cell ID, cell type, and the
    Kilosort cluster ID used for raw-waveform analysis. ``figure_sink`` can be
    supplied by a notebook that must also archive one PNG per selection.
    """
    import matplotlib.pyplot as plt
    from .browse import figure_to_png, png_browser

    if not results:
        print('No sampled sorting-QC results to browse.')
        return None
    ordered = [(int(cell_id), result) for cell_id, result in results.items()]

    def _figure(cell_id):
        fig, _ = plot_sampled_detected_spikes(
            {int(cell_id): results[int(cell_id)]},
            max_waveforms_per_class=max_waveforms_per_class,
            n_neighbor_electrodes=n_neighbor_electrodes)
        return fig

    if figure_sink is not None:
        for cell_id, _ in ordered:
            saved_fig = _figure(cell_id)
            figure_sink[cell_id] = saved_fig
            plt.close(saved_fig)

    def _render(cell_id):
        result = results[int(cell_id)]
        cell_type = str(result.summary.get('cell_type', ''))
        html = (f'<b>cell {int(cell_id)} — {cell_type}; '
                f'Kilosort cluster {result.target_cluster}</b><br>'
                'Raw threshold candidates are proposed without Kilosort labels; '
                'the similarity panel classifies them afterward.')
        return html, figure_to_png(_figure(cell_id))

    options = [
        (f'cell {cell_id} — {result.summary.get("cell_type", "")}; '
         f'KS cluster {result.target_cluster}', cell_id)
        for cell_id, result in ordered
    ]
    return png_browser(options, _render, description=description)
