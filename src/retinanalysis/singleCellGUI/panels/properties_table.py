"""Properties table: key-value display of selected epoch metadata (like Clarinet)."""

import panel as pn
import param

from retinanalysis.singleCellGUI.state import AppState


class PropertiesTable(pn.viewable.Viewer):
    """Displays metadata key-value pairs for the currently selected epoch."""

    state = param.ClassSelector(class_=AppState)

    def __init__(self, state, **params):
        super().__init__(state=state, **params)
        self._table_pane = pn.pane.HTML(
            self._empty_html(),
            sizing_mode='stretch_width',
        )
        state.param.watch(self._update, 'selected_epochs')

    @staticmethod
    def _empty_html():
        return '<div style="color: #999; font-size: 12px; padding: 4px;">No epoch selected</div>'

    def _update(self, event=None):
        epochs = self.state.selected_epochs
        if not epochs:
            self._table_pane.object = self._empty_html()
            return

        exp_name, block_id, epoch_idx = epochs[0]
        try:
            sb, rb = self.state.get_or_load_block(exp_name, block_id, b_spiking=False)
        except Exception as e:
            self._table_pane.object = f'<div style="color:red;font-size:12px;">{e}</div>'
            return

        rows = []
        # Epoch params from stimulus block
        if hasattr(sb, 'df_epochs') and epoch_idx < len(sb.df_epochs):
            row = sb.df_epochs.iloc[epoch_idx]
            for col in sb.df_epochs.columns:
                if col in ('epoch_parameters', 'frame_times_ms'):
                    continue
                val = row[col]
                rows.append((col, str(val)))

        # Add response info
        rows.append(('sample_rate', f'{rb.amp_sample_rate} Hz'))
        rows.append(('n_samples', str(len(rb.amp_data[epoch_idx]))))
        rows.append(('epoch_idx', str(epoch_idx)))
        rows.append(('block_id', str(block_id)))
        rows.append(('exp_name', exp_name))

        # Build HTML table
        html = [
            '<table style="width:100%; font-size:11px; border-collapse:collapse;">',
            '<tr style="background:#f0f0f0; font-weight:600;">',
            '<td style="padding:3px 6px; border:1px solid #ddd;">Property</td>',
            '<td style="padding:3px 6px; border:1px solid #ddd;">Value</td></tr>',
        ]
        for prop, val in rows:
            html.append(
                f'<tr><td style="padding:2px 6px; border:1px solid #eee; color:#555;">{prop}</td>'
                f'<td style="padding:2px 6px; border:1px solid #eee;">{val}</td></tr>'
            )
        html.append('</table>')
        self._table_pane.object = '\n'.join(html)

    def __panel__(self):
        return pn.Column(
            pn.pane.Markdown("**Properties**", margin=(0, 5)),
            self._table_pane,
            sizing_mode='stretch_width',
        )
