"""Export panel: dropdown format selector + export button."""
 
import os
import numpy as np
import pandas as pd
import panel as pn
import param
 
from retinanalysis.gui.state import AppState
 
 
class ExportPanel(pn.viewable.Viewer):
    """Bottom sidebar widget for exporting selected epoch data."""
 
    state = param.ClassSelector(class_=AppState)
 
    def __init__(self, state, **params):
        super().__init__(state=state, **params)
 
        self._format_select = pn.widgets.Select(
            name="Format",
            options=['CSV', 'HDF5', 'Pickle', 'NumPy (.npz)'],
            value='CSV',
            width=150,
        )
        self._path_input = pn.widgets.TextInput(
            name="Output Dir",
            value='./exports',
            placeholder='./exports',
            width=200,
        )
        self._export_btn = pn.widgets.Button(
            name="Export Selected",
            button_type="success",
        )
        self._export_btn.on_click(self._on_export)
        self._status = pn.pane.Markdown("", sizing_mode='stretch_width')
 
    def _on_export(self, event):
        epochs = self.state.selected_epochs
        if not epochs:
            self._status.object = "_No epochs selected._"
            return
 
        out_dir = self._path_input.value.strip() or './exports'
        os.makedirs(out_dir, exist_ok=True)
        fmt = self._format_select.value
 
        try:
            data = self._collect_data(epochs)
            if fmt == 'CSV':
                self._export_csv(data, out_dir)
            elif fmt == 'HDF5':
                self._export_hdf5(data, out_dir)
            elif fmt == 'Pickle':
                self._export_pickle(data, out_dir)
            elif fmt == 'NumPy (.npz)':
                self._export_npz(data, out_dir)
            self._status.object = f"Exported **{len(epochs)}** epoch(s) to `{out_dir}/` as {fmt}."
        except Exception as e:
            self._status.object = f"**Export error:** {e}"
 
    def _collect_data(self, epochs):
        """Gather data dicts for each selected epoch."""
        records = []
        for exp_name, block_id, epoch_idx in epochs:
            sb, rb = self.state.get_or_load_block(exp_name, block_id, b_spiking=True)
            trace = rb.amp_data[epoch_idx]
            sr = rb.amp_sample_rate
 
            spike_times = None
            if hasattr(rb, 'spike_times') and rb.spike_times is not None:
                spike_times = rb.spike_times[epoch_idx]
 
            epoch_params = {}
            if hasattr(sb, 'df_epochs') and epoch_idx < len(sb.df_epochs):
                row = sb.df_epochs.iloc[epoch_idx]
                for col in sb.df_epochs.columns:
                    if col in ('epoch_parameters', 'frame_times_ms'):
                        continue
                    epoch_params[col] = row[col]
 
            records.append({
                'exp_name': exp_name,
                'block_id': block_id,
                'epoch_idx': epoch_idx,
                'trace': trace,
                'sample_rate': sr,
                'spike_times': spike_times,
                'epoch_params': epoch_params,
            })
        return records
 
    def _export_csv(self, data, out_dir):
        """Export each epoch trace as a CSV file + a summary CSV."""
        rows = []
        for rec in data:
            fname = f"{rec['exp_name']}_B{rec['block_id']}_E{rec['epoch_idx']}.csv"
            df_trace = pd.DataFrame({
                'time_s': np.arange(len(rec['trace'])) / rec['sample_rate'],
                'amplitude': rec['trace'],
            })
            df_trace.to_csv(os.path.join(out_dir, fname), index=False)
            row = {
                'exp_name': rec['exp_name'],
                'block_id': rec['block_id'],
                'epoch_idx': rec['epoch_idx'],
                'sample_rate': rec['sample_rate'],
                'n_samples': len(rec['trace']),
                'trace_file': fname,
            }
            row.update(rec['epoch_params'])
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
 
    def _export_hdf5(self, data, out_dir):
        import h5py
        path = os.path.join(out_dir, 'exported_epochs.h5')
        with h5py.File(path, 'w') as f:
            for rec in data:
                grp_name = f"{rec['exp_name']}/B{rec['block_id']}/E{rec['epoch_idx']}"
                grp = f.create_group(grp_name)
                grp.create_dataset('trace', data=rec['trace'])
                grp.attrs['sample_rate'] = rec['sample_rate']
                if rec['spike_times'] is not None:
                    grp.create_dataset('spike_times', data=rec['spike_times'])
                for k, v in rec['epoch_params'].items():
                    try:
                        grp.attrs[k] = v
                    except TypeError:
                        grp.attrs[k] = str(v)
 
    def _export_pickle(self, data, out_dir):
        import pickle
        path = os.path.join(out_dir, 'exported_epochs.pkl')
        with open(path, 'wb') as f:
            pickle.dump(data, f)
 
    def _export_npz(self, data, out_dir):
        arrays = {}
        for rec in data:
            key = f"{rec['exp_name']}_B{rec['block_id']}_E{rec['epoch_idx']}"
            arrays[key] = rec['trace']
        np.savez(os.path.join(out_dir, 'exported_traces.npz'), **arrays)
 
    def export_selected(self, format='dict'):
        """Programmatic export for notebook use. Returns collected data."""
        epochs = self.state.selected_epochs
        if not epochs:
            return []
        return self._collect_data(epochs)
 
    def __panel__(self):
        return pn.Row(
            self._format_select,
            self._path_input,
            self._export_btn,
            self._status,
            sizing_mode='stretch_width',
        )