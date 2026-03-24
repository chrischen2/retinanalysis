"""Filter panel: protocol, cell type, recording technique, and custom filters."""
 
import panel as pn
import param
 
from retinanalysis.singleCellGUI.state import AppState
 
 
class FilterPanel(pn.viewable.Viewer):
    """Criteria panel that filters which nodes are visible in the data tree."""
 
    state = param.ClassSelector(class_=AppState)
 
    def __init__(self, state, **params):
        super().__init__(state=state, **params)
 
        # Protocol filter
        self._protocol_input = pn.widgets.TextInput(
            name="Protocol", placeholder="e.g. ExpandingSpots"
        )
        self._protocol_mode = pn.widgets.RadioButtonGroup(
            options=['contains', 'equals'], value='contains',
            button_type='default', button_style='outline',
        )
 
        # Cell type multi-select
        self._celltype_select = pn.widgets.MultiChoice(
            name="Cell Type", options=[], solid=False,
        )
 
        # Recording technique multi-select
        self._rec_tech_select = pn.widgets.MultiChoice(
            name="Recording Technique", options=[], solid=False,
        )
 
        # Custom filter row
        self._custom_col_input = pn.widgets.TextInput(
            name="Parameter", placeholder="column name", width=120,
        )
        self._custom_op = pn.widgets.Select(
            name="Op", options=['==', 'contains', '>', '<', '>=', '<='],
            value='==', width=80,
        )
        self._custom_val_input = pn.widgets.TextInput(
            name="Value", placeholder="value", width=100,
        )
        self._custom_add_btn = pn.widgets.Button(
            name="+", button_type="default", width=40,
        )
        self._custom_add_btn.on_click(self._on_add_custom)
        self._custom_tags = pn.Column()
 
        # Apply button
        self._apply_btn = pn.widgets.Button(
            name="Apply Filters", button_type="primary",
        )
        self._apply_btn.on_click(self._on_apply)
 
        # Clear button
        self._clear_btn = pn.widgets.Button(
            name="Clear", button_type="warning",
        )
        self._clear_btn.on_click(self._on_clear)
 
        # Populate options when experiments change
        state.param.watch(self._refresh_options, 'loaded_exp_names')
 
    def _refresh_options(self, event=None):
        """Rebuild cell-type and recording-technique options from loaded experiments."""
        cell_types = set()
        rec_techs = set()
        for exp_name, df in self.state.exp_summaries.items():
            if 'cell_label' in df.columns:
                pass  # cell_label is not cell_type
            # Cell types from the all_experiments_df
            row = self.state.all_experiments_df[
                self.state.all_experiments_df['exp_name'] == exp_name
            ]
            if len(row) > 0 and 'cell_types' in row.columns:
                types_str = row['cell_types'].values[0]
                if types_str:
                    cell_types.update(t.strip() for t in str(types_str).split(',') if t.strip())
 
            # Recording technique from exp summary
            if 'recording_technique' in df.columns:
                rec_techs.update(
                    v for v in df['recording_technique'].dropna().unique() if v
                )
 
        self._celltype_select.options = sorted(cell_types)
        self._rec_tech_select.options = sorted(rec_techs)
 
    def _on_add_custom(self, event):
        col = self._custom_col_input.value.strip()
        val = self._custom_val_input.value.strip()
        op = self._custom_op.value
        if not col or not val:
            return
        tag_text = f"`{col} {op} {val}`"
        remove_btn = pn.widgets.Button(name="✕", width=25, height=25, button_type="danger")
        tag_row = pn.Row(pn.pane.Markdown(tag_text), remove_btn)
        remove_btn.on_click(lambda e, r=tag_row: self._custom_tags.remove(r))
        self._custom_tags.append(tag_row)
        self._custom_col_input.value = ""
        self._custom_val_input.value = ""
 
    def _collect_custom_filters(self):
        """Parse custom filter tags into (column, op, value) tuples."""
        filters = []
        for row in self._custom_tags:
            md = row[0].object  # the markdown string like `col op val`
            # Parse back from markdown
            inner = md.strip('`').strip()
            parts = inner.split(' ', 2)
            if len(parts) == 3:
                filters.append(tuple(parts))
        return filters
 
    def _on_apply(self, event):
        self.state.protocol_filter = self._protocol_input.value.strip()
        self.state.protocol_match_mode = self._protocol_mode.value
        self.state.celltype_filter = list(self._celltype_select.value)
        self.state.recording_technique_filter = list(self._rec_tech_select.value)
        self.state.custom_filters = self._collect_custom_filters()
 
    def _on_clear(self, event):
        self._protocol_input.value = ''
        self._protocol_mode.value = 'contains'
        self._celltype_select.value = []
        self._rec_tech_select.value = []
        self._custom_tags.clear()
        self._on_apply(event)
 
    def __panel__(self):
        return pn.Column(
            pn.pane.Markdown("### Filters", margin=(0, 5)),
            self._protocol_input,
            self._protocol_mode,
            self._celltype_select,
            self._rec_tech_select,
            pn.layout.Divider(),
            pn.pane.Markdown("**Custom Filter:**", margin=(5, 5)),
            pn.Row(
                self._custom_col_input, self._custom_op,
                self._custom_val_input, self._custom_add_btn,
            ),
            self._custom_tags,
            pn.Row(self._apply_btn, self._clear_btn),
            sizing_mode='stretch_width',
        )