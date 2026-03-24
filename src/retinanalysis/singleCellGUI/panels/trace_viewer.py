"""Trace viewer: matplotlib-based raw trace display with stimulus markers."""
 
import numpy as np
import panel as pn
import param
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
 
from retinanalysis.singleCellGUI.state import AppState
 
 
class TraceViewer(pn.viewable.Viewer):
    """Displays selected epoch traces with stimulus timing and optional analysis overlays."""
 
    state = param.ClassSelector(class_=AppState)
 
    def __init__(self, state, **params):
        super().__init__(state=state, **params)
 
        self._plot_pane = pn.pane.Matplotlib(
            self._make_empty_fig(),
            dpi=100, tight=True,
            sizing_mode='stretch_width',
        )
        self._metadata_pane = pn.pane.Markdown("", sizing_mode='stretch_width')
 
        # Display mode
        self._mode_select = pn.widgets.RadioButtonGroup(
            options=['Overlay', 'Grid'],
            value='Overlay',
            button_type='default',
            button_style='outline',
        )
        self._spike_toggle = pn.widgets.Checkbox(
            name="Show Spikes", value=True,
        )
 
        # Watch for selection and analysis changes
        state.param.watch(self._on_selection_change, 'selected_epochs')
        state.param.watch(self._on_selection_change, 'active_analyses')
        self._mode_select.param.watch(self._on_selection_change, 'value')
        self._spike_toggle.param.watch(self._on_selection_change, 'value')
 
    @staticmethod
    def _make_empty_fig():
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.text(0.5, 0.5, 'Select epochs to plot', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='grey')
        plt.close(fig)
        return fig
 
    def _on_selection_change(self, event=None):
        """Redraw traces for the current selection."""
        epochs = self.state.selected_epochs
        if not epochs:
            self._plot_pane.object = self._make_empty_fig()
            self._metadata_pane.object = ""
            return
 
        mode = self._mode_select.value
        show_spikes = self._spike_toggle.value
        analyses = self.state.active_analyses
 
        if mode == 'Grid' and len(epochs) > 1:
            fig = self._plot_grid(epochs, show_spikes, analyses)
        else:
            fig = self._plot_overlay(epochs, show_spikes, analyses)
 
        self._plot_pane.object = fig
        self._update_metadata(epochs)
 
    def _load_epoch(self, exp_name, block_id, epoch_idx):
        """Load and return (trace, spike_times, sample_rate, pre_time_ms, stim_time_ms, label)."""
        sb, rb = self.state.get_or_load_block(exp_name, block_id, b_spiking=True)
        trace = rb.amp_data[epoch_idx]
        sample_rate = rb.amp_sample_rate
 
        spike_times = None
        if hasattr(rb, 'spike_times') and rb.spike_times is not None:
            spike_times = rb.spike_times[epoch_idx]
 
        pre_time_ms = None
        stim_time_ms = None
        if hasattr(sb, 'df_epochs') and epoch_idx < len(sb.df_epochs):
            pre_time_ms = sb.df_epochs.at[epoch_idx, 'preTime'] if 'preTime' in sb.df_epochs.columns else None
            stim_time_ms = sb.df_epochs.at[epoch_idx, 'stimTime'] if 'stimTime' in sb.df_epochs.columns else None
 
        # Build label
        short_proto = sb.protocol_name.rsplit('.', 1)[-1] if hasattr(sb, 'protocol_name') else ''
        label = f"{exp_name} B{block_id} E{epoch_idx}"
        if short_proto:
            label += f" ({short_proto})"
 
        return trace, spike_times, sample_rate, pre_time_ms, stim_time_ms, label
 
    def _plot_overlay(self, epochs, show_spikes, analyses):
        """Plot all selected traces on a single axes."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        colors = plt.cm.tab10.colors
 
        for i, (exp_name, block_id, epoch_idx) in enumerate(epochs):
            try:
                trace, spike_times, sr, pre_ms, stim_ms, label = self._load_epoch(
                    exp_name, block_id, epoch_idx
                )
            except Exception as e:
                ax.text(0.5, 0.5 - i * 0.05, f"Error: {e}",
                        transform=ax.transAxes, color='red', fontsize=10)
                continue
 
            color = colors[i % len(colors)]
            time = np.arange(len(trace)) / sr
            ax.plot(time, trace, color=color, alpha=0.8, label=label)
 
            # Spike overlay
            if show_spikes and spike_times is not None and len(spike_times) > 0:
                valid = spike_times[spike_times < len(trace)]
                ax.scatter(valid / sr, trace[valid], color=color, s=15, zorder=5, marker='|')
 
            # Stimulus markers (only draw once)
            if i == 0 and pre_ms is not None:
                onset = pre_ms / 1000.0
                ax.axvline(onset, color='k', linestyle='--', alpha=0.5, label='Stim onset')
                if stim_ms is not None:
                    offset = (pre_ms + stim_ms) / 1000.0
                    ax.axvline(offset, color='grey', linestyle='--', alpha=0.5, label='Stim offset')
 
            # Analysis overlays
            for plugin in analyses:
                try:
                    processed = plugin.process(
                        trace, sr, pre_time_ms=pre_ms, stim_time_ms=stim_ms
                    )
                    ax.plot(time, processed, color=color, alpha=0.5,
                            linestyle=':', label=f"{label} [{plugin.get_label()}]")
                except Exception:
                    pass
 
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        if len(epochs) <= 8:
            ax.legend(fontsize=8, loc='upper right')
        ax.set_title(f"{len(epochs)} epoch(s) selected")
        plt.close(fig)
        return fig
 
    def _plot_grid(self, epochs, show_spikes, analyses):
        """Plot each selected epoch in its own subplot."""
        n = len(epochs)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
 
        for i, (exp_name, block_id, epoch_idx) in enumerate(epochs):
            ax = axes[i // ncols][i % ncols]
            try:
                trace, spike_times, sr, pre_ms, stim_ms, label = self._load_epoch(
                    exp_name, block_id, epoch_idx
                )
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                        color='red', fontsize=9, ha='center')
                continue
 
            time = np.arange(len(trace)) / sr
            ax.plot(time, trace, 'k', alpha=0.8, linewidth=0.8)
 
            if show_spikes and spike_times is not None and len(spike_times) > 0:
                valid = spike_times[spike_times < len(trace)]
                ax.scatter(valid / sr, trace[valid], color='r', s=10, zorder=5, marker='|')
 
            if pre_ms is not None:
                ax.axvline(pre_ms / 1000.0, color='k', linestyle='--', alpha=0.4)
                if stim_ms is not None:
                    ax.axvline((pre_ms + stim_ms) / 1000.0, color='grey', linestyle='--', alpha=0.4)
 
            for plugin in analyses:
                try:
                    processed = plugin.process(trace, sr, pre_time_ms=pre_ms, stim_time_ms=stim_ms)
                    ax.plot(time, processed, color='blue', alpha=0.5, linestyle=':')
                except Exception:
                    pass
 
            ax.set_title(label, fontsize=9)
            ax.set_xlabel('Time (s)', fontsize=8)
 
        # Hide unused axes
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)
 
        fig.tight_layout()
        plt.close(fig)
        return fig
 
    def _update_metadata(self, epochs):
        """Update the metadata pane with info about the first selected epoch."""
        if not epochs:
            self._metadata_pane.object = ""
            return
        exp_name, block_id, epoch_idx = epochs[0]
        try:
            sb, rb = self.state.get_or_load_block(exp_name, block_id, b_spiking=False)
            lines = [f"**{exp_name}** Block {block_id}, Epoch {epoch_idx}"]
            if hasattr(sb, 'df_epochs') and epoch_idx < len(sb.df_epochs):
                row = sb.df_epochs.iloc[epoch_idx]
                for col in sb.df_epochs.columns:
                    if col in ('epoch_parameters', 'frame_times_ms'):
                        continue
                    lines.append(f"- {col}: `{row[col]}`")
            lines.append(f"- sample_rate: `{rb.amp_sample_rate}` Hz")
            lines.append(f"- n_samples: `{len(rb.amp_data[epoch_idx])}`")
            self._metadata_pane.object = '\n'.join(lines)
        except Exception as e:
            self._metadata_pane.object = f"Error loading metadata: {e}"
 
    def __panel__(self):
        controls = pn.Row(self._mode_select, self._spike_toggle)
        return pn.Column(
            controls,
            self._plot_pane,
            pn.layout.Divider(),
            self._metadata_pane,
            sizing_mode='stretch_width',
        )