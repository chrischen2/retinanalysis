import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import os

def fit_kmeans(data, n_clusters, verbose=False):
    from sklearn.cluster import KMeans   # heavy import, deferred to use site
    # KMeans
    # Fix non-spike cluster index to k-1, and spike cluster indices to k-2
    spike_cluster_indices = np.arange(n_clusters - 1)  # k-1 clusters for spikes
    non_spike_cluster_index = n_clusters - 1  # k-1 cluster for non-spikes
    # Seeds follow SpikeDetectorNew.m: the median of each feature for the noise
    # cluster, then the column-wise max for the spike cluster. That max is not
    # generally one of the observations -- it is the corner of the bounding box
    # -- which is what the MATLAB seeds with, so it is reproduced here rather
    # than substituting the largest-amplitude point.
    start_matrix = np.zeros((n_clusters, 3))
    start_matrix[0, :] = np.median(data, axis=0)
    if n_clusters > 1:
        start_matrix[1, :] = data.max(axis=0)
    # Only k=2 has a MATLAB counterpart; extra clusters are seeded from the
    # next-largest peaks, as this port did before k was configurable.
    sorted_peak_indices = np.argsort(data[:, 0])[::-1]
    for i in range(2, n_clusters):
        start_matrix[i, :] = data[sorted_peak_indices[i - 2]]

    def _fit(init, n_init):
        km = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=10000)
        km.fit(data)
        # Sort clusters by peak amplitude, so label 0 is the largest-amplitude
        # cluster and label k-1 the noise one.
        sorted_indices = np.argsort(km.cluster_centers_[:, 0])[::-1]
        lab = np.zeros_like(km.labels_)
        for i, label in enumerate(sorted_indices):
            lab[km.labels_ == label] = i
        return lab, np.isin(lab, spike_cluster_indices)

    try:
        labels, spike_index_logical = _fit(start_matrix, 1)
    except Exception:
        # A trace with no spikes can collapse the seeded fit into an empty
        # cluster. SpikeDetectorNew.m retries from a random sample rather than
        # abandoning the trial, so the same retry happens here; only if that
        # also fails does everything fall to the noise cluster.
        try:
            labels, spike_index_logical = _fit('random', 10)
        except Exception as e:
            if verbose:
                print(f'KMeans failed even from a random start ({type(e).__name__}: {e}); '
                      f'treating every peak as noise')
            labels = np.full(len(data), non_spike_cluster_index, dtype=int)
            spike_index_logical = np.zeros(len(data), dtype=bool)

    return labels, spike_index_logical

def apply_min_peak_amp(min_peak_amplitude, peak_amplitudes, peak_times, spike_index_logical,
                       cluster_index, non_spike_cluster_index, verbose=False):
    if min_peak_amplitude > 0:
        n_old_spikes = np.sum(spike_index_logical)
        temp_amp = peak_amplitudes[spike_index_logical]
        temp_times = peak_times[spike_index_logical]
        b_keep_sps = temp_amp > min_peak_amplitude
        indices = np.where(b_keep_sps)[0]

        # Update spike_index_logical to only include indices that pass the min_peak_amplitude threshold
        new_spike_index_logical = np.zeros(len(peak_amplitudes), dtype=bool)
        true_indices = np.where(spike_index_logical)[0]  # Get indices where spike_index_logical is True
        new_spike_index_logical[true_indices[b_keep_sps]] = True

        spike_index_logical = new_spike_index_logical
        n_new_spikes = np.sum(spike_index_logical)
        n_rejected_spikes = n_old_spikes - n_new_spikes
        if verbose and n_rejected_spikes:
            print(f'Rejected {n_rejected_spikes}/{n_old_spikes} spikes with amplitude < {min_peak_amplitude:.2f} peak amp.')

        non_spike_amplitudes = peak_amplitudes[~spike_index_logical]
        spike_times = temp_times[indices]
        spike_amplitudes = temp_amp[indices]

        # Update cluster_index so that failed spikes are set to non_spike cluster
        cluster_index[~spike_index_logical] = non_spike_cluster_index
    else:
        spike_times = peak_times[spike_index_logical]
        spike_amplitudes = peak_amplitudes[spike_index_logical]
        non_spike_amplitudes = peak_amplitudes[~spike_index_logical]

    return spike_times, spike_amplitudes, non_spike_amplitudes, spike_index_logical, cluster_index

def detector(data_matrix, check_detection=False, sample_rate=1e4, refractory_period=1.5e-3,
             search_window=1.2e-3, cutoff_frequency=300, global_polarity=False,
             min_peak_amplitude=0, n_clusters=2, threshold_spike_factor=3,
             remove_refractory_violations=True, max_trial_length_s=1, str_save_dir=None,
             verbose=False, cluster_across_trials=False):
    """Detect spikes in extracellular / cell-attached traces; port of SpikeDetectorNew.m.

    Each row of ``data_matrix`` is one trial. The trace is high-pass filtered,
    every local extremum is taken as a *candidate*, and k-means on (peak
    amplitude, left rebound, right rebound) splits those candidates into a spike
    and a noise cluster. A trial only counts as spiking when its spike cluster
    stands ``threshold_spike_factor`` noise standard deviations clear of the
    noise cluster; otherwise it is returned with zero spikes.

    The candidate pool is large by construction — a local minimum occurs at
    roughly every third sample of band-passed noise, so a 1 s trial at 10 kHz
    yields a few thousand — and it is an intermediate, not a result. Only the
    final spike counts are reported.

    Defaults follow ``SpikeDetectorNew.m``: 300 Hz high pass, polarity decided
    from the tails of the sorted trace, and refractory violations removed from
    the returned spike times.

    ``threshold_spike_factor`` defaults to 3, not to the MATLAB signature's 1.5:
    every linCone analysis script passes ``paras.spikeTh = 3``
    (``spotAnnularGratingMain.m``, ``linConeMain.m``), so 3 is the value the
    MATLAB pipeline actually runs at and 1.5 is never reached. The distinction
    matters — on a non-spiking trace, where the clustering has only noise to
    work with, 1.5 admits hundreds of "spikes" per trial while 3 admits almost
    none. Pass ``threshold_spike_factor=1.5`` for the bare function default.

    With ``cluster_across_trials=True``, candidate features from every trial are
    fit by one shared k-means model and pass one shared spike-factor gate. This
    prevents a trial containing only small contamination events from promoting
    those events into its local "spike" cluster. Spike times and refractory
    filtering remain trial-specific. ``global_polarity=True`` is recommended
    with pooled clustering so trials without real spikes cannot flip polarity
    independently.

    Two documented departures remain: ``n_clusters`` may exceed the MATLAB's
    fixed 2, and in per-trial mode traces longer than ``max_trial_length_s`` are
    clustered in sections (the MATLAB always clusters the whole trace) so that
    a slow drift in noise amplitude does not swamp the spike cluster. Pooled
    mode intentionally uses one fit across all trials and therefore does not
    section that shared fit.

    Returns ``(spike_times, spike_amplitudes, refractory_violations)``, each a
    list with one entry per trial. Spike times are in samples. ``verbose=True``
    prints per-trial diagnostics; by default only the caller sees the counts.
    """
    refractory_period_dp = refractory_period * sample_rate  # datapoints
    search_window_dp = search_window * sample_rate  # datapoints
    max_trial_length_dp = int(max_trial_length_s * sample_rate)  # Convert seconds to datapoints

    data_matrix = high_pass_filter(data_matrix, cutoff_frequency, 1/sample_rate)

    n_traces = data_matrix.shape[0]
    spike_times = [[] for _ in range(n_traces)]
    spike_amplitudes = [[] for _ in range(n_traces)]
    refractory_violations = [[] for _ in range(n_traces)]

    # Fix non-spike cluster index to k-1, and spike cluster indices to k-2
    spike_cluster_indices = np.arange(n_clusters - 1)  # k-1 clusters for spikes
    non_spike_cluster_index = n_clusters - 1  # k-1 cluster for non-spikes

    def _empty(tt):
        # Sample indices, so keep them integer even when empty -- a float64
        # empty array cannot index a trace (raw[spike_times[i]] raises
        # IndexError on no-spike epochs).
        spike_times[tt] = np.array([], dtype=int)
        spike_amplitudes[tt] = np.array([])
        refractory_violations[tt] = np.array([], dtype=int)

    if cluster_across_trials:
        # Extract candidates separately so epoch boundaries remain intact, but
        # fit one cluster boundary using all epochs. A single global quality
        # gate replaces the old all-or-none gate fitted independently within
        # each epoch.
        oriented_traces = []
        candidate_times = []
        candidate_amplitudes = []
        candidate_rebounds = []
        candidate_features = []
        polarity_source_global = data_matrix.ravel() if global_polarity else None

        for tt in range(n_traces):
            current_trace = np.asarray(data_matrix[tt, :]).copy()
            polarity_source = (polarity_source_global if global_polarity
                               else current_trace)
            tail = min(500, max(polarity_source.size // 2, 1))
            sorted_trace = np.sort(polarity_source)
            if abs(np.mean(sorted_trace[-tail:])) > abs(np.mean(sorted_trace[:tail])):
                current_trace = -current_trace
            oriented_traces.append(current_trace)

            amplitudes, times = get_peaks(current_trace, -1)
            negative = amplitudes < 0
            times = np.asarray(times[negative], dtype=int)
            amplitudes = np.abs(np.asarray(amplitudes[negative], dtype=float))
            rebounds = get_rebounds(times, current_trace, search_window_dp)
            features = np.column_stack(
                (amplitudes, rebounds['Left'], rebounds['Right']))

            candidate_times.append(times)
            candidate_amplitudes.append(amplitudes)
            candidate_rebounds.append(rebounds)
            candidate_features.append(features)

        total_candidates = sum(len(values) for values in candidate_times)
        if total_candidates < n_clusters:
            if verbose:
                print(f'Pooled detector: only {total_candidates} candidate peak(s); '
                      'returning zero spikes')
            for tt in range(n_traces):
                _empty(tt)
            return spike_times, spike_amplitudes, refractory_violations

        pooled_features = np.vstack(candidate_features)
        pooled_amplitudes = np.concatenate(candidate_amplitudes)
        pooled_cluster_index, pooled_spike_mask = fit_kmeans(
            pooled_features, n_clusters, verbose=verbose)
        if min_peak_amplitude > 0:
            pooled_spike_mask &= pooled_amplitudes > float(min_peak_amplitude)
            pooled_cluster_index[~pooled_spike_mask] = non_spike_cluster_index

        spike_pool = pooled_amplitudes[pooled_spike_mask]
        noise_pool = pooled_amplitudes[~pooled_spike_mask]
        if not len(spike_pool) or not len(noise_pool):
            sigF = -np.inf
        else:
            noise_sd = float(np.std(noise_pool))
            separation = float(np.mean(spike_pool) - np.mean(noise_pool))
            sigF = separation / noise_sd if noise_sd > 0 else (
                np.inf if separation > 0 else -np.inf)
        if sigF < abs(threshold_spike_factor):
            pooled_spike_mask[:] = False
            if verbose:
                print(f'Pooled detector: 0 spikes (spike factor {sigF:.2f} < '
                      f'{abs(threshold_spike_factor):g})')
        elif verbose:
            print(f'Pooled detector: {int(pooled_spike_mask.sum())} spikes from '
                  f'{n_traces} trials (spike factor {sigF:.2f})')

        start = 0
        for tt, times in enumerate(candidate_times):
            stop = start + len(times)
            local_mask = pooled_spike_mask[start:stop]
            local_clusters = pooled_cluster_index[start:stop]
            spike_times[tt] = np.asarray(times[local_mask], dtype=int)
            spike_amplitudes[tt] = np.asarray(
                candidate_amplitudes[tt][local_mask], dtype=float)
            refractory_violations[tt] = (
                np.where(np.diff(spike_times[tt]) < refractory_period_dp)[0] + 1)

            if check_detection:
                save_path = (os.path.join(str_save_dir, f'trial_{tt + 1}_clustering.png')
                             if str_save_dir else None)
                plot_clustering_data(
                    candidate_amplitudes[tt], candidate_rebounds[tt],
                    local_clusters, spike_cluster_indices, non_spike_cluster_index,
                    oriented_traces[tt], spike_times[tt], refractory_violations[tt],
                    sigF, str_save_plot=save_path)
            start = stop

        if remove_refractory_violations:
            for tt in range(n_traces):
                if len(refractory_violations[tt]) == 0:
                    continue
                keep = np.ones(len(spike_times[tt]), dtype=bool)
                keep[np.asarray(refractory_violations[tt], dtype=int)] = False
                spike_times[tt] = np.asarray(spike_times[tt])[keep]
                spike_amplitudes[tt] = np.asarray(spike_amplitudes[tt])[keep]
        return spike_times, spike_amplitudes, refractory_violations

    for tt in range(n_traces):
        current_trace = data_matrix[tt, :]
        # Polarity from the tails of the sorted trace rather than the single
        # most extreme sample, as SpikeDetectorNew.m does: one perfusion
        # transient can outweigh every spike in the trace and flip it wrongly.
        # Oriented big-peaks-down here, so the candidates below are minima.
        polarity_source = data_matrix.ravel() if global_polarity else current_trace
        tail = min(500, max(polarity_source.size // 2, 1))
        sorted_trace = np.sort(polarity_source)
        if abs(np.mean(sorted_trace[-tail:])) > abs(np.mean(sorted_trace[:tail])):
            current_trace = -current_trace

        # Candidate peaks: every local minimum. This is the pool k-means sorts
        # into spikes and noise, not a spike count.
        peak_amplitudes, peak_times = get_peaks(current_trace, -1)  # -1 for negative peaks
        if len(peak_times) < 2:
            _empty(tt)
            if verbose:
                print(f'Trial {tt + 1}: 0 spikes (trace is flat)')
            continue
        peak_times = peak_times[peak_amplitudes < 0]  # only negative deflections
        peak_amplitudes = np.abs(peak_amplitudes[peak_amplitudes < 0])  # only negative deflections
        if len(peak_times) < 2:
            _empty(tt)
            if verbose:
                print(f'Trial {tt + 1}: 0 spikes (no negative deflections)')
            continue

        # get rebounds on either side of each peak
        rebound = get_rebounds(peak_times, current_trace, search_window_dp)

        # cluster spikes
        clustering_data = np.column_stack((peak_amplitudes, rebound['Left'], rebound['Right']))

        if len(current_trace) > max_trial_length_dp:
            num_sections = int(np.ceil(len(current_trace) / max_trial_length_dp))
            section_indices = np.array_split(np.arange(len(current_trace)), num_sections)

            cluster_index = np.zeros(len(peak_amplitudes), dtype=int)
            spike_index_logical = np.zeros(len(peak_amplitudes), dtype=bool)
            for section_idx, section in enumerate(section_indices):
                section_mask = np.isin(peak_times, section)  # Select peaks within the current section
                section_data = clustering_data[section_mask]

                if len(section_data) == 0:
                    continue
                cluster_index[section_mask], spike_index_logical[section_mask] = fit_kmeans(
                    section_data, n_clusters, verbose=verbose)

        else:
            # Standard KMeans clustering for shorter traces
            cluster_index, spike_index_logical = fit_kmeans(clustering_data, n_clusters, verbose=verbose)

        spike_times[tt], spike_amplitudes[tt], non_spike_amplitudes, spike_index_logical, cluster_index = apply_min_peak_amp(
            min_peak_amplitude, peak_amplitudes, peak_times, spike_index_logical,
            cluster_index, non_spike_cluster_index, verbose=verbose
        )

        # check for no spikes trace. With no surviving spike or non-spike peaks
        # (e.g. min_peak_amplitude rejected them all) sigF is undefined, and
        # NaN < threshold would be False -- i.e. the trace would wrongly pass as
        # spiking.
        if len(spike_amplitudes[tt]) == 0 or len(non_spike_amplitudes) == 0:
            sigF = -np.inf
        else:
            sigF = (np.mean(spike_amplitudes[tt]) - np.mean(non_spike_amplitudes)) / np.std(non_spike_amplitudes)

        if sigF < abs(threshold_spike_factor):  # no spikes
            _empty(tt)
            if verbose:
                print(f'Trial {tt + 1}: 0 spikes (spike factor {sigF:.2f} < '
                      f'{abs(threshold_spike_factor):g})')
            if check_detection:
                plot_clustering_data(peak_amplitudes, rebound, cluster_index, spike_cluster_indices,
                                      non_spike_cluster_index, current_trace, spike_times[tt],
                                      refractory_violations[tt], sigF)
            continue

        # check for refractory violations
        refractory_violations[tt] = np.where(np.diff(spike_times[tt]) < refractory_period_dp)[0] + 1
        ref_violations = len(refractory_violations[tt])
        if ref_violations > 0 and verbose:
            print(f'Trial {tt + 1}: {ref_violations} refractory violations '
                  + ('removed' if remove_refractory_violations else 'remain'))

        if check_detection:
            # Plot clustering data for each section
            if len(current_trace) > max_trial_length_dp:
                for section_idx, section in enumerate(section_indices):
                    section_mask = np.isin(peak_times, section)
                    s_peak_amps = peak_amplitudes[section_mask]
                    s_rebound = {'Left': rebound['Left'][section_mask], 'Right': rebound['Right'][section_mask]}                 
                    s_cluster_index = cluster_index[section_mask]
                    s_trace = current_trace[section]
                    s_spike_times = spike_times[tt][np.isin(spike_times[tt], section)]
                    # Subtract time of section start
                    s_spike_times = s_spike_times - section[0]
                    s_refractory_violations = np.where(np.diff(s_spike_times) < refractory_period_dp)[0] + 1
                    if str_save_dir:
                        str_save_plot = os.path.join(str_save_dir, f'trial_{tt + 1}_section_{section_idx + 1}_clustering.png')
                    else:
                        str_save_plot=None
                    plot_clustering_data(s_peak_amps, s_rebound, s_cluster_index, spike_cluster_indices,
                                        non_spike_cluster_index, s_trace, s_spike_times,
                                        s_refractory_violations, sigF, str_save_plot=str_save_plot)
            else:
                if str_save_dir:
                    str_save_plot = os.path.join(str_save_dir, f'trial_{tt + 1}_clustering.png')
                else:
                    str_save_plot=None
                plot_clustering_data(peak_amplitudes, rebound, cluster_index, spike_cluster_indices,
                                    non_spike_cluster_index, current_trace, spike_times[tt],
                                    refractory_violations[tt], sigF, str_save_plot=str_save_plot)

    if remove_refractory_violations:
        # SpikeDetectorNew.m drops the violating spikes rather than only
        # reporting them: a peak within the refractory period of its
        # predecessor is a second detection of one spike, not a second spike.
        # The indices stay in refractory_violations so the count is still
        # visible, but they now index into the pre-removal spike train.
        for tt in range(n_traces):
            if len(refractory_violations[tt]) == 0:
                continue
            keep = np.ones(len(spike_times[tt]), dtype=bool)
            keep[np.asarray(refractory_violations[tt], dtype=int)] = False
            spike_times[tt] = np.asarray(spike_times[tt])[keep]
            spike_amplitudes[tt] = np.asarray(spike_amplitudes[tt])[keep]

    return spike_times, spike_amplitudes, refractory_violations


def plot_clustering_data(peak_amplitudes, rebound, cluster_index, spike_cluster_indices, non_spike_cluster_index, 
                         current_trace, spike_times, refractory_violations, sigF, str_save_plot=None):
    """
    Plot clustering data in 3D and the trace with spikes and refractory violations.

    Parameters:
    - peak_amplitudes: Array of peak amplitudes.
    - rebound: Dictionary with 'Left' and 'Right' rebound values.
    - cluster_indices: Array of cluster indices.
    - spike_cluster_index: Index of the spike cluster.
    - non_spike_cluster_index: Index of the non-spike cluster.
    - current_trace: The signal trace being analyzed.
    - spike_times: Indices of detected spikes.
    - refractory_violations: Indices of refractory violations.
    - sigF: Spike factor for the current trace.
    - str_save_plot: Path to save the plots.
    """
    fig = plt.figure(figsize=(12, 6))

    # 3D scatter plot for clustering data
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # ax1.scatter(
    #     peak_amplitudes[cluster_index == spike_cluster_indices],
    #     rebound['Left'][cluster_index == spike_cluster_indices],
    #     rebound['Right'][cluster_index == spike_cluster_indices],
    #     c='r', label='Spikes'
    # )
    # ax1.scatter(
    #     peak_amplitudes[cluster_index == non_spike_cluster_indices],
    #     rebound['Left'][cluster_index == non_spike_cluster_indices],
    #     rebound['Right'][cluster_index == non_spike_cluster_indices],
    #     c='k', label='Non-Spikes'
    # )
    ls_colors = [f'C{i}' for i in range(len(spike_cluster_indices))]
    for i in range(len(spike_cluster_indices)):
        ax1.scatter(
            peak_amplitudes[cluster_index == spike_cluster_indices[i]],
            rebound['Left'][cluster_index == spike_cluster_indices[i]],
            rebound['Right'][cluster_index == spike_cluster_indices[i]],
            c=ls_colors[i], label=f'Spikes {i+1}'
        )
    ax1.scatter(
        peak_amplitudes[cluster_index == non_spike_cluster_index],
        rebound['Left'][cluster_index == non_spike_cluster_index],
        rebound['Right'][cluster_index == non_spike_cluster_index],
        c='k', label='Non-Spikes'
    )
    ax1.set_xlabel('Peak Amplitude')
    ax1.set_ylabel('L Rebound')
    ax1.set_zlabel('R Rebound')
    ax1.view_init(elev=8, azim=36)  # Match MATLAB's view
    ax1.legend()
    ax1.set_title('Clustering Data')

    # 2D plot for the trace
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(current_trace, 'k', label='Trace')
    if len(spike_times) > 0:
        ax2.scatter(spike_times, current_trace[spike_times], c='r', label='Spikes')
    # Check if refractory violations exist before plotting
    if len(refractory_violations) > 0:
        # Print dtype of refractory_violations
        # print(f'Refractory violations dtype: {type(refractory_violations[0])}, shape: {np.array(refractory_violations).shape}')
        # return refractory_violations, spike_times, current_trace
        refractory_violations = np.array(refractory_violations).astype(int)
        spike_times = np.array(spike_times).astype(int)
        # print(refractory_violations)
        # print(spike_times)
        ref_sts = spike_times[refractory_violations]
        ref_sts = np.array(ref_sts).astype(int)
        ax2.scatter(spike_times[refractory_violations], current_trace[ref_sts], c='g', label='Refractory Violations')
    ax2.set_title(f'SpikeFactor = {sigF:.2f}')
    ax2.legend()

    plt.tight_layout()
    if str_save_plot:
        plt.savefig(str_save_plot, bbox_inches='tight')
        print(f'Saved clustering plot to {str_save_plot}')
        plt.close()
    else:
        plt.show()

def get_peaks(X, direction):
    """
    Identify local peaks in the input data based on the specified direction.

    Parameters:
    X : array-like
        Input data (1D array).
    direction : int
        Direction for peak detection; 1 for local maxima, -1 for local minima.

    Returns:
    peaks : array-like
        Values of the detected peaks.
    Ind : array-like
        Indices of the detected peaks.
    """

    if direction > 0:  # local max
        Ind = np.where(np.diff(np.sign(np.diff(X))) < 0)[0] + 1
    else:  # local min
        Ind = np.where(np.diff(np.sign(np.diff(X))) > 0)[0] + 1

    peaks = X[Ind]
    return peaks, Ind

def get_rebounds(peaks_ind, trace, search_interval):
    """Return the first left/right rebound around every candidate peak.

    ``SpikeDetectorNew.m`` uses the first local extremum in each half-window.
    The previous port reproduced that result by slicing the trace and calling
    :func:`get_peaks` twice for every candidate. Extracellular noise produces
    thousands of candidates per epoch, making those tiny Python calls the
    dominant cost of condition analysis. Precomputing all extrema once and
    locating the first one in each window with ``searchsorted`` preserves the
    same boundary and ordering rules without the per-candidate loop.
    """
    trace = np.asarray(trace)
    peaks_ind = np.asarray(peaks_ind, dtype=int)
    peaks = trace[peaks_ind]
    r = {'Left': np.zeros_like(peaks), 'Right': np.zeros_like(peaks)}

    if not len(peaks_ind):
        return r

    half_window = round(search_interval / 2)
    _, maxima = get_peaks(trace, 1)
    _, minima = get_peaks(trace, -1)

    def assign_first(mask, extrema_ind):
        selected = np.flatnonzero(mask)
        if not len(selected) or not len(extrema_ind):
            return
        candidate_times = peaks_ind[selected]
        start_points = np.maximum(0, candidate_times - half_window)
        end_points = np.minimum(candidate_times + half_window, len(trace) - 1)

        # get_peaks(trace[start:peak]) can only select [start + 1, peak - 2].
        left_positions = np.searchsorted(extrema_ind, start_points + 1)
        left_safe = np.minimum(left_positions, len(extrema_ind) - 1)
        left_valid = ((left_positions < len(extrema_ind))
                      & (extrema_ind[left_safe] <= candidate_times - 2))
        left_rows = selected[left_valid]
        r['Left'][left_rows] = trace[extrema_ind[left_safe[left_valid]]]

        # get_peaks(trace[peak:end]) can only select [peak + 1, end - 2].
        right_positions = np.searchsorted(extrema_ind, candidate_times + 1)
        right_safe = np.minimum(right_positions, len(extrema_ind) - 1)
        right_valid = ((right_positions < len(extrema_ind))
                       & (extrema_ind[right_safe] <= end_points - 2))
        right_rows = selected[right_valid]
        r['Right'][right_rows] = trace[extrema_ind[right_safe[right_valid]]]

    assign_first(peaks < 0, maxima)
    assign_first(peaks > 0, minima)

    return r

def high_pass_filter(X, F, SampleInterval):
    L = X.shape[1] if X.ndim > 1 else len(X)
    if L == 1:  # flip if given a column vector
        X = X.T
        L = X.shape[1]

    FreqStepSize = 1 / (SampleInterval * L)
    FreqKeepPts = round(F / FreqStepSize)

    FFTData = np.fft.fft(X, axis=1)
    FFTData[:, :FreqKeepPts] = 0
    FFTData[:, -FreqKeepPts:] = 0

    Xfilt = np.real(np.fft.ifft(FFTData, axis=1))
    return Xfilt
