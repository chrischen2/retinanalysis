import bin2py
import numpy as np
import os
import matplotlib.pyplot as plt
from retinanalysis.classes.response import MEAResponseBlock
from retinanalysis.config.settings import RAW_DIR
from retinanalysis import ei_utils as eiu
from typing import Iterable, List, Optional, Tuple

# Constants
RW_BLOCKSIZE = 100000  # Block size for reading data
TTL_THRESHOLD = 1000
SAMPLE_RATE = 20000 # Hz
class RawTraces:
    def __init__(self, rb: MEAResponseBlock):
        self.binpath = os.path.join(RAW_DIR, rb.exp_name, rb.datafile_name)
        self.d_timing = rb.d_timing
        self.sorted_electrodes = eiu.sort_electrode_map(rb.vcd.get_electrode_map())
        self.data = None
        self.ttl_times = None
        self.ttl_samples = None
        self.sample_rate = SAMPLE_RATE  # Hz
        self.epoch_idx = None

    def load_bin_data(self, start_sample=0, end_sample=None, verbose=False):
        """
        Load raw .bin data into a NumPy array.

        Parameters:
            binpath (str): Path to the .bin file.
            start_sample (int): Starting sample index (default: 0).
            end_sample (int): Ending sample index (default: None, reads till the end).

        Returns:
            np.ndarray: Loaded data as a NumPy array of shape [electrodes, samples].
        """
        with bin2py.PyBinFileReader(self.binpath, chunk_samples=RW_BLOCKSIZE, is_row_major=True) as pbfr:
            # Determine the number of electrodes and total samples
            n_channels = pbfr.num_electrodes
            total_samples = pbfr.length
            
            if verbose:
                print(f"Number of electrodes: {n_channels}, Total samples: {total_samples}.")
                print(f"Total time: {total_samples / SAMPLE_RATE} seconds")
                print(f"Sample rate: {SAMPLE_RATE} Hz")

            # Set end_sample to the total length if not specified
            if end_sample is None:
                end_sample = total_samples

            # Validate sample range
            if start_sample < 0 or end_sample > total_samples or start_sample >= end_sample:
                raise ValueError("Invalid start_sample or end_sample range.")

            query_samples = end_sample - start_sample
            if verbose:
                print(f"Querying {query_samples} samples from {start_sample} to {end_sample}.")
                print(f'Queried time: {query_samples / SAMPLE_RATE} seconds')
                print(f'From {start_sample / SAMPLE_RATE} to {end_sample / SAMPLE_RATE} seconds')
                
            # Preallocate array for the data
            data = np.zeros((n_channels, query_samples), dtype=np.float32)

            ttl_times_buffer = []
            ttl_samples = np.zeros((query_samples,), dtype=np.float32)
            # Read data in chunks
            for start_idx in range(start_sample, end_sample, RW_BLOCKSIZE):
                n_samples_to_get = min(RW_BLOCKSIZE, end_sample - start_idx)
                chunk = pbfr.get_data(start_idx, n_samples_to_get)

                # Extract TTL data (channel 0) and compute TTL times
                ttl_samples = chunk[0, :]
                below_threshold = (ttl_samples < -TTL_THRESHOLD)
                above_threshold = np.logical_not(below_threshold)
                below_to_above = np.logical_and.reduce([
                    below_threshold[:-1],
                    above_threshold[1:]
                ])
                trigger_indices = np.argwhere(below_to_above) + start_idx
                ttl_times_buffer.append(trigger_indices[:, 0])

                # Populate the data matrix (exclude channel 0)
                data[:, start_idx - start_sample:start_idx - start_sample + n_samples_to_get] = chunk[1:, :]
                # ttl_samples[start_idx - start_sample:start_idx - start_sample + n_samples_to_get] = chunk[0, :]

            # Concatenate TTL times
            ttl_times = np.concatenate(ttl_times_buffer, axis=0)
        
        if verbose:
            print(f'Data shape: {data.shape}')
        # print(f'TTL times shape: {ttl_times.shape}')
        self.data = data
        self.ttl_times = ttl_times
        self.ttl_samples = ttl_samples
        
    def load_epoch_index(self, epoch_idx, verbose=True):
        epoch_start = self.d_timing['epochStarts'][epoch_idx]
        epoch_end = self.d_timing['epochEnds'][epoch_idx]
        self.load_bin_data(start_sample=epoch_start, end_sample=epoch_end, verbose=verbose)
        self.epoch_idx = epoch_idx

def plot_sts_over_trace(rt: RawTraces, rb: MEAResponseBlock, 
                        cell_id, epoch_idx, start_time=0, end_time=None,
                        n_highlight_width=18, channel_idx: Optional[int]=None, ax=None):
    # Load epoch_idx if needed
    if rt.epoch_idx != epoch_idx:
        rt.load_epoch_index(epoch_idx, verbose=True)
    
    # Get max amplitude channel for this cell_id
    if channel_idx is None:
        top_idx = eiu.get_top_electrodes(cell_id, rb.vcd, n_markers=1, b_sort=False)[0]
        channel_idx = rt.sorted_electrodes[top_idx]

    raw_ts = rt.data[channel_idx, :]
    time = np.arange(len(raw_ts)) / rt.sample_rate # in seconds
    
    if end_time is None:
        end_time = time[-1]
    
    mask = np.where((time >= start_time) & (time <= end_time))[0]
    time = time[mask]
    raw_ts = raw_ts[mask]
    # print(f'Raw trace shape: {raw_ts.shape}, Time shape: {time.shape}')
    # print(f'Channel index: {channel_idx}, Cell ID: {cell_id}, Epoch index: {epoch_idx}')
    # print(f'Time range: {start_time} to {end_time} seconds, Mask shape: {mask.shape}')
    # return
    if ax is None:
        f, ax = plt.subplots(figsize=(12,6))
    ax.plot(time, raw_ts)
    ax.set_title(f'Cell {cell_id}, Channel {channel_idx}, Epoch {epoch_idx}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Raw Signal')
    
    df_st = rb.df_spike_times
    cell_idx = np.where(df_st['cell_id'] == cell_id)[0]
    if len(cell_idx) == 0:
        raise ValueError(f'Cell ID {cell_id} not found in response block {rb.exp_name}.')
    cell_idx = cell_idx[0]
    sts = df_st.at[cell_idx, 'spike_times'][epoch_idx]
    if len(sts) == 0:
        print(f'No spikes found for cell {cell_id} in epoch {epoch_idx}.')
        return
    # Convert from ms back to samples
    sts = np.round(sts * rt.sample_rate / 1000).astype(int)
    # Keep only spike times in time range given by sample index
    sts = sts[(sts>=mask[0]) & (sts<=mask[-1])]
    sts -= mask[0]  # Adjust spike times to the new time range
    ax.scatter(time[sts], raw_ts[sts], color='red', zorder=10)
    for st in sts:
        ax.plot(time[st:st+n_highlight_width],
                raw_ts[st:st+n_highlight_width], color='red', lw=2, alpha=0.5)


# ---------------------------------------------------------------------------
# Spike-sorting visual QC: raw trace + asterisks-at-top for every cell on the
# same primary electrode, so the user can eyeball whether spikes really belong
# to the target cell or are leaking in from a neighbor.
# ---------------------------------------------------------------------------

def primary_electrode_of_cell(rb: MEAResponseBlock, cell_id: int) -> int:
    """Return the raw channel index where this cell's |EI| peaks."""
    ei = rb.vcd.get_ei_for_cell(int(cell_id)).ei  # (n_electrodes, n_samples)
    return int(np.argmax(np.max(np.abs(ei), axis=1)))


def cells_sharing_electrode(
    rb: MEAResponseBlock,
    electrode_idx: int,
    exclude_cell: Optional[int] = None,
    cell_ids: Optional[Iterable[int]] = None,
) -> List[Tuple[int, str, float]]:
    """List cells whose primary electrode is ``electrode_idx``.

    Returns a list of ``(cell_id, cell_type, ei_peak_amp)`` sorted by
    descending peak amplitude. Use ``exclude_cell`` to drop the target
    cell itself; use ``cell_ids`` to restrict to a subset (e.g. only
    QC-passing cells, to avoid drowning the plot in noise units).
    """
    candidates = (cell_ids if cell_ids is not None
                  else rb.df_spike_times['cell_id'].astype(int).tolist())
    type_map = (rb.df_spike_times.set_index('cell_id')['cell_type'].to_dict()
                if 'cell_type' in rb.df_spike_times.columns else {})
    out: List[Tuple[int, str, float]] = []
    for cid in candidates:
        cid = int(cid)
        if exclude_cell is not None and cid == int(exclude_cell):
            continue
        try:
            ei = rb.vcd.get_ei_for_cell(cid).ei
        except Exception:
            continue
        elec_peak = np.max(np.abs(ei), axis=1)
        peak_elec = int(np.argmax(elec_peak))
        if peak_elec == int(electrode_idx):
            out.append((cid, str(type_map.get(cid, '')), float(elec_peak[peak_elec])))
    out.sort(key=lambda t: -t[2])
    return out


def plot_sorting_qc(
    rt: 'RawTraces',
    rb: MEAResponseBlock,
    cell_id: int,
    epoch_idx: int,
    *,
    start_time: float = 0.0,
    end_time: Optional[float] = 2.0,
    candidate_cell_ids: Optional[Iterable[int]] = None,
    max_other_cells: int = 6,
    target_color: str = 'tab:red',
    other_cmap: str = 'tab10',
    ax: Optional[plt.Axes] = None,
    show_legend: bool = True,
) -> plt.Axes:
    """Plot the raw trace of a cell's primary electrode with multi-cell ticks.

    The target cell's spikes get **down-arrows + asterisks at the top of the
    panel** in ``target_color``; every other cell whose primary electrode
    is the same gets its own color and a row of asterisks slightly above
    the trace. Lets you visually judge whether the spikes assigned to the
    target cell really belong to it or are stolen from neighbors.

    Parameters
    ----------
    rt : RawTraces
    rb : MEAResponseBlock
    cell_id : int
        Target cell. Its primary electrode is what gets plotted.
    epoch_idx : int
        Which epoch's raw data to load (cached across calls).
    start_time, end_time : float
        Window in seconds within the epoch. Default first 2 s.
    candidate_cell_ids : iterable[int], optional
        Restrict the "other cells" search to these IDs. Pass the QC-pass
        or visual-QC `good` set to avoid cluttering with noise units.
    max_other_cells : int
        Cap on how many other cells get drawn (sorted by EI peak amp).
        Useful when a hot electrode has 10+ matched cells.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib Axes
    """
    # Load epoch_idx if needed (caches inside rt)
    if rt.epoch_idx != epoch_idx:
        rt.load_epoch_index(epoch_idx, verbose=False)

    # Resolve target cell's primary electrode (raw channel index)
    electrode_idx = primary_electrode_of_cell(rb, cell_id)

    raw_ts = rt.data[electrode_idx, :]
    time = np.arange(len(raw_ts)) / rt.sample_rate
    if end_time is None:
        end_time = time[-1]
    mask = np.where((time >= start_time) & (time <= end_time))[0]
    if mask.size == 0:
        raise ValueError(f'Empty window [{start_time}, {end_time}] for epoch {epoch_idx}.')
    win_time = time[mask]
    win_trace = raw_ts[mask]
    sample_offset = mask[0]

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.5))

    ax.plot(win_time, win_trace, color='0.25', lw=0.6, zorder=1)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('raw signal')

    # Pull spike times (ms) for the target cell within the window
    def _spike_samples_in_window(cid: int) -> np.ndarray:
        df_st = rb.df_spike_times
        idx = df_st.index[df_st['cell_id'] == int(cid)]
        if len(idx) == 0:
            return np.array([], dtype=int)
        sts_ms = df_st.at[idx[0], 'spike_times'][epoch_idx]
        if len(sts_ms) == 0:
            return np.array([], dtype=int)
        s = np.round(np.asarray(sts_ms) * rt.sample_rate / 1000).astype(int)
        s = s[(s >= mask[0]) & (s <= mask[-1])] - sample_offset
        return s

    # Plot the target cell's spikes prominently
    type_map = (rb.df_spike_times.set_index('cell_id')['cell_type'].to_dict()
                if 'cell_type' in rb.df_spike_times.columns else {})
    target_type = type_map.get(int(cell_id), '')
    target_samples = _spike_samples_in_window(cell_id)

    # Reserve a "marker band" above the trace for asterisks
    y_lo, y_hi = ax.get_ylim() if ax.lines else (float(win_trace.min()), float(win_trace.max()))
    span = max(y_hi - y_lo, 1e-9)
    y_top = y_hi + 0.15 * span
    y_extra = y_hi + 0.04 * span  # baseline for "other" cells' rows
    y_target = y_hi + 0.12 * span

    # Target cell: red asterisks at the top, plus red overlay on the trace
    if target_samples.size > 0:
        ax.scatter(win_time[target_samples],
                   [y_target] * target_samples.size,
                   marker='*', s=85, color=target_color, zorder=5,
                   label=f'cell {cell_id} ({target_type})  TARGET')
        ax.scatter(win_time[target_samples], win_trace[target_samples],
                   color=target_color, s=18, zorder=4)

    # Find and plot other cells on the same electrode
    others = cells_sharing_electrode(
        rb, electrode_idx, exclude_cell=cell_id,
        cell_ids=candidate_cell_ids,
    )[:max_other_cells]
    cmap = plt.get_cmap(other_cmap)
    for k, (cid, ctype, peak_amp) in enumerate(others):
        color = cmap(k % cmap.N)
        samples = _spike_samples_in_window(cid)
        # Each other cell gets its own row of asterisks slightly below the
        # target band so colors don't pile on top of each other.
        y_row = y_extra + 0.012 * span * k
        if samples.size > 0:
            ax.scatter(win_time[samples], [y_row] * samples.size,
                       marker='*', s=55, color=color, zorder=4,
                       label=f'cell {cid} ({ctype})  peak={peak_amp:.0f}')

    ax.set_ylim(y_lo - 0.05 * span, y_top)
    ax.set_title(
        f'cell {cell_id} ({target_type}) | electrode {electrode_idx} | '
        f'epoch {epoch_idx} | {target_samples.size} target spikes, '
        f'{len(others)} neighbor cells on same electrode'
    )
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='lower right', fontsize=7,
                      framealpha=0.85,
                      ncol=2 if len(others) >= 3 else 1)
    return ax


def plot_sorting_qc_grid(
    rt: 'RawTraces',
    rb: MEAResponseBlock,
    cell_ids: Iterable[int],
    epoch_idxs: Iterable[int],
    *,
    window_s: float = 2.0,
    start_time: float = 0.0,
    candidate_cell_ids: Optional[Iterable[int]] = None,
    max_other_cells: int = 6,
):
    """Grid: rows = cells, cols = epochs. Loads each epoch's raw data once."""
    cell_ids = list(cell_ids)
    epoch_idxs = list(epoch_idxs)
    fig, axes = plt.subplots(
        len(cell_ids), len(epoch_idxs),
        figsize=(6.5 * len(epoch_idxs), 2.8 * len(cell_ids)),
        squeeze=False, sharex=False, sharey=False,
    )
    # Outer loop on epochs so we load the raw block once per epoch.
    for j, ep in enumerate(epoch_idxs):
        rt.load_epoch_index(ep, verbose=False)
        for i, cid in enumerate(cell_ids):
            ax = axes[i, j]
            try:
                plot_sorting_qc(
                    rt, rb, cid, ep,
                    start_time=start_time, end_time=start_time + window_s,
                    candidate_cell_ids=candidate_cell_ids,
                    max_other_cells=max_other_cells,
                    ax=ax, show_legend=(j == 0),
                )
            except Exception as exc:
                ax.set_title(f'cell {cid}, epoch {ep}: {exc!r}', fontsize=8)
                ax.axis('off')
    fig.tight_layout()
    return fig, axes



