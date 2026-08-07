import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from retinanalysis.utils.protocol_sorting_qc import (
    KilosortOutput,
    analyze_unit_sorting_qc,
    browse_sampled_detected_spikes,
    browse_unit_sorting_qc,
    extract_multichannel_waveforms,
    load_binary_segment,
    load_kilosort_output,
    nearby_template_clusters,
    plot_sorting_qc_summary,
    plot_sampled_detected_spikes,
    plot_unit_sorting_qc,
    refractory_contamination_estimate,
    sorting_qc_table,
    template_similarity,
)


def _waveform(scale):
    t = np.arange(31) - 10
    base = -np.exp(-(t / 2.2) ** 2) + 0.25 * np.exp(-((t - 5) / 3.0) ** 2)
    return base[:, None] * np.asarray(scale)[None, :]


def _synthetic():
    rng = np.random.default_rng(4)
    n_samples, n_channels = 60_000, 4
    raw = rng.normal(0, 1.5, size=(n_samples, n_channels)).astype(np.float32)
    target_template = _waveform([70, 50, 8, 2])
    competitor_template = _waveform([35, 8, 75, 55])
    target_assigned = np.arange(2_000, 22_000, 1_000)
    competitor_assigned = np.arange(25_000, 45_000, 1_000)
    missed, misassigned = 47_000, 49_000

    def add(time, waveform):
        raw[time - 10:time + 21] += waveform

    for time in target_assigned:
        add(time, target_template)
    for time in competitor_assigned:
        add(time, competitor_template)
    add(missed, target_template)
    add(misassigned, target_template)

    spike_times = np.r_[target_assigned, competitor_assigned, misassigned]
    spike_clusters = np.r_[np.zeros(len(target_assigned), int),
                           np.ones(len(competitor_assigned) + 1, int)]
    spike_templates = spike_clusters.copy()
    order = np.argsort(spike_times)
    ks = KilosortOutput(
        spike_times=spike_times[order], spike_clusters=spike_clusters[order],
        spike_templates=spike_templates[order],
        templates=np.stack([target_template, competitor_template]),
        channel_map=np.arange(n_channels),
        channel_positions=np.array([[0, 0], [30, 0], [0, 30], [30, 30]]),
        sample_rate_hz=10_000)
    return raw, ks, missed, misassigned


def test_waveform_similarity_and_competing_cluster_overlap():
    raw, ks, _, _ = _synthetic()
    wfs, valid = extract_multichannel_waveforms(raw, [2_000, 3_000], 10, 20)
    assert valid.tolist() == [2_000, 3_000]
    score = template_similarity(wfs, np.median(wfs, axis=0))
    assert np.all(score > 0.99)
    nearby = nearby_template_clusters(ks, 0, min_spatial_overlap=0.1)
    assert nearby.cluster_id.tolist() == [0, 1]


def test_quantitative_qc_separates_missed_from_misassigned():
    raw, ks, missed, misassigned = _synthetic()
    result = analyze_unit_sorting_qc(
        raw, ks, 0, n_local_channels=4, min_empirical_spikes=10,
        threshold_sigma=4.0, similarity_percentile=2,
        min_spatial_overlap=0.1)
    by_sample = result.events.set_index('candidate_sample').status
    assert by_sample.loc[missed] == 'missed_detection'
    assert by_sample.loc[misassigned] == 'misassigned'
    assert result.summary['n_missed'] == 1
    assert result.summary['n_misassigned'] == 1
    assert result.summary['n_ks_assigned'] == 20
    assert np.isclose(result.summary['detection_miss_fraction'], 1 / 21)
    table = sorting_qc_table({0: result})
    assert table.loc[0, 'refractory_violation_fraction'] == 0
    assert table.loc[0, 'refractory_contamination_estimate'] == 0

    fig, axes = plot_unit_sorting_qc(result)
    assert axes.shape == (2, 2)
    plt.close(fig)
    fig, ax = plot_sorting_qc_summary(table, label_column='target_cluster')
    assert len(ax.patches) == 3
    plt.close(fig)
    fig, axes = plot_sampled_detected_spikes({123: result})
    assert axes.shape == (1, 3)
    assert 'cell 123' in axes[0, 0].get_title()
    assert len(axes[0, 1].lines) > 1
    assert 'similarity' in axes[0, 2].get_title().lower()
    plt.close(fig)


def test_standard_kilosort_and_binary_loaders(tmp_path):
    raw, ks, _, _ = _synthetic()
    folder = tmp_path / 'ks'
    folder.mkdir()
    np.save(folder / 'spike_times.npy', ks.spike_times)
    np.save(folder / 'spike_clusters.npy', ks.spike_clusters)
    np.save(folder / 'spike_templates.npy', ks.spike_templates)
    np.save(folder / 'templates.npy', ks.templates)
    np.save(folder / 'channel_map.npy', ks.channel_map)
    np.save(folder / 'channel_positions.npy', ks.channel_positions)
    loaded = load_kilosort_output(folder, sample_rate_hz=10_000)
    assert loaded.templates.shape == ks.templates.shape
    np.testing.assert_array_equal(loaded.channel_positions, ks.channel_positions)

    binary = tmp_path / 'raw.bin'
    raw_i16 = np.round(raw).astype(np.int16)
    raw_i16.tofile(binary)
    segment = load_binary_segment(
        binary, n_channels=4, start_sample=100, n_samples=200,
        channel_ids=[0, 2])
    np.testing.assert_array_equal(segment, raw_i16[100:300][:, [0, 2]])


def test_sampled_spike_browser_shows_one_labeled_cluster_and_collects_figure(
        monkeypatch):
    import retinanalysis.utils.browse as browse

    raw, ks, _, _ = _synthetic()
    result = analyze_unit_sorting_qc(
        raw, ks, 0, n_local_channels=4, min_empirical_spikes=10,
        threshold_sigma=4.0, min_spatial_overlap=0.1)
    result.summary['cell_type'] = 'OnM'

    def render_first(options, render, **kwargs):
        assert options[0][0] == 'cell 123 — OnM; KS cluster 0'
        return render(options[0][1])

    def close_figure(fig):
        plt.close(fig)
        return b'png'

    monkeypatch.setattr(browse, 'png_browser', render_first)
    monkeypatch.setattr(browse, 'figure_to_png', close_figure)
    saved = {}
    html, png = browse_sampled_detected_spikes(
        {123: result}, figure_sink=saved)

    assert 'cell 123 — OnM' in html
    assert 'Kilosort cluster 0' in html
    assert png == b'png'
    assert list(saved) == [123]

    full_saved = {}
    html, png = browse_unit_sorting_qc(
        {123: result}, figure_sink=full_saved)
    assert 'cell 123 — OnM' in html
    assert png == b'png'
    assert list(full_saved) == [123]


def test_refractory_contamination_estimate_is_explicit_about_assumptions():
    # One 2 ms violation among 100 spikes in a 10 s segment.
    times = np.arange(100) * 10
    times[50] = times[49] + 2
    estimate = refractory_contamination_estimate(
        times, duration_s=1.0, sample_rate_hz=1_000,
        refractory_ms=3.0, censored_ms=1.0)
    assert 0 < estimate < 0.5
