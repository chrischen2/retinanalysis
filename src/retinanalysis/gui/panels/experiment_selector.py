"""Experiment selector panel: species pre-filter, experiment picker, loaded list."""
 
import panel as pn
import param
 
from retinanalysis.gui.state import AppState
 
 
class ExperimentSelector(pn.viewable.Viewer):
    """Three-section sidebar panel for experiment management.
 
    Sections:
      a) Species pre-filter
      b) Experiment dropdown + Add button
      c) Loaded experiments list with Remove buttons
    """
 
    state = param.ClassSelector(class_=AppState)
 
    def __init__(self, state, **params):
        super().__init__(state=state, **params)
 
        # --- a) Species pre-filter ---
        self._species_select = pn.widgets.MultiChoice(
            name="Filter by Species",
            options=list(state.all_species),
            value=list(state.all_species),
            solid=False,
        )
        self._species_select.param.watch(self._on_species_change, 'value')
 
        # --- b) Experiment picker ---
        self._exp_dropdown = pn.widgets.Select(
            name="Experiment",
            options=self._build_exp_options(),
        )
        self._add_btn = pn.widgets.Button(name="Add Experiment", button_type="primary")
        self._add_btn.on_click(self._on_add)
 
        # --- c) Loaded experiments list ---
        self._loaded_pane = pn.Column(name="Loaded Experiments")
        self._rebuild_loaded_list()
 
        # Watch for external changes to loaded_exp_names
        state.param.watch(self._on_loaded_change, 'loaded_exp_names')
 
    def _build_exp_options(self):
        """Build dropdown options filtered by selected species."""
        df = self.state.available_experiments
        if df.empty:
            return {}
        options = {}
        for _, row in df.iterrows():
            label = f"{row['exp_name']} ({row.get('species', '?')})"
            options[label] = row['exp_name']
        return options
 
    def _on_species_change(self, event):
        self.state.selected_species = list(event.new)
        self._exp_dropdown.options = self._build_exp_options()
 
    def _on_add(self, event):
        exp_name = self._exp_dropdown.value
        if exp_name:
            self.state.add_experiment(exp_name)
 
    def _on_loaded_change(self, event):
        self._rebuild_loaded_list()
 
    def _rebuild_loaded_list(self):
        """Rebuild the list of loaded experiments with remove buttons."""
        items = []
        for exp_name in self.state.loaded_exp_names:
            df_all = self.state.all_experiments_df
            row = df_all[df_all['exp_name'] == exp_name]
            species = row['species'].values[0] if len(row) > 0 else '?'
            n_cells = row['n_cells'].values[0] if len(row) > 0 else '?'
            label = f"**{exp_name}** — {species}, {n_cells} cells"
 
            remove_btn = pn.widgets.Button(
                name="✕", button_type="danger", width=30, height=30,
                margin=(0, 5)
            )
            # Capture exp_name in closure
            remove_btn.on_click(lambda event, en=exp_name: self.state.remove_experiment(en))
 
            row_widget = pn.Row(
                pn.pane.Markdown(label, margin=(5, 5)),
                remove_btn,
                sizing_mode='stretch_width',
            )
            items.append(row_widget)
 
        self._loaded_pane.clear()
        if items:
            self._loaded_pane.extend(items)
        else:
            self._loaded_pane.append(
                pn.pane.Markdown("_No experiments loaded._", margin=(5, 5))
            )
 
    def __panel__(self):
        return pn.Column(
            pn.pane.Markdown("### Experiments", margin=(0, 5)),
            self._species_select,
            pn.Row(self._exp_dropdown, self._add_btn, sizing_mode='stretch_width'),
            pn.layout.Divider(),
            pn.pane.Markdown("**Loaded:**", margin=(5, 5)),
            self._loaded_pane,
            sizing_mode='stretch_width',
        )