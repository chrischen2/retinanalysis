"""SCExplorer: main application with Clarinet-style layout.

Layout:
  ┌──────────────────────────────────────────────────────────────┐
  │  Toolbar: [Add Experiment dropdown] [Processors] [Pipeline]  │
  ├──────────────────┬───────────────────────────────────────────┤
  │  Left Panel      │  Center: Trace Viewer                     │
  │  ┌────────────┐  │  ┌─────────────────────────────────────┐  │
  │  │ Data Tree   │  │  │  [Overlay | Grid]  [✓ Show Spikes] │  │
  │  │ (browse     │  │  │                                     │  │
  │  │  hierarchy) │  │  │  Matplotlib plot                    │  │
  │  │             │  │  │                                     │  │
  │  ├────────────┤  │  │                                     │  │
  │  │ Properties  │  │  │                                     │  │
  │  │ (key-value) │  │  │  [2D/Time(s)]                      │  │
  │  └────────────┘  │  └─────────────────────────────────────┘  │
  ├──────────────────┴───────────────────────────────────────────┤
  │  Status bar                                                   │
  └──────────────────────────────────────────────────────────────┘
"""

import panel as pn

from retinanalysis.singleCellGUI.state import AppState
from retinanalysis.singleCellGUI.panels.experiment_selector import ExperimentSelector
from retinanalysis.singleCellGUI.panels.filter_panel import FilterPanel
from retinanalysis.singleCellGUI.panels.data_tree import DataTree
from retinanalysis.singleCellGUI.panels.trace_viewer import TraceViewer
from retinanalysis.singleCellGUI.panels.analysis_panel import AnalysisPanel
from retinanalysis.singleCellGUI.panels.export_panel import ExportPanel
from retinanalysis.singleCellGUI.panels.properties_table import PropertiesTable


class SCExplorer:
    """Single-Cell Explorer -- interactive GUI for browsing retinal ephys data.

    Usage in a Jupyter notebook::

        import retinanalysis as ra
        ra.populate_database()
        explorer = ra.singleCellGUI.launch_explorer()

    After interacting with the GUI, access loaded data programmatically::

        sb, rb = explorer.state.loaded_blocks[(exp_name, block_id)]
        data = explorer.export_panel.export_selected()
    """

    def __init__(self, extra_plugin_dirs=None):
        pn.extension(sizing_mode='stretch_width')

        # Central state
        self.state = AppState()
        self.state.initialize()

        # Panels
        self.experiment_selector = ExperimentSelector(self.state)
        self.filter_panel = FilterPanel(self.state)
        self.data_tree = DataTree(self.state)
        self.trace_viewer = TraceViewer(self.state)
        self.analysis_panel = AnalysisPanel(self.state, extra_plugin_dirs=extra_plugin_dirs)
        self.export_panel = ExportPanel(self.state)
        self.properties_table = PropertiesTable(self.state)

        # Status bar
        self._status = pn.pane.Markdown(
            "_Ready_",
            style={'font-size': '11px', 'color': '#666', 'padding': '2px 8px',
                   'background': '#f5f5f5', 'border-top': '1px solid #ddd'},
            sizing_mode='stretch_width',
        )

    def _build_layout(self):
        """Compose the Clarinet-style layout."""

        # -- Toolbar (top bar) --
        toolbar = pn.Row(
            self.experiment_selector,
            pn.layout.HSpacer(),
            pn.pane.Markdown("**Epoch Processors**", margin=(10, 5)),
            self.analysis_panel,
            background='#f8f8f8',
            sizing_mode='stretch_width',
            height=None,
        )

        # -- Left panel: tree + filter + properties --
        left_panel = pn.Column(
            self.filter_panel,
            pn.layout.Divider(),
            pn.pane.Markdown("### Data Browser", margin=(0, 5)),
            self.data_tree,
            pn.layout.Divider(),
            self.properties_table,
            width=320,
            scroll=True,
            sizing_mode='stretch_height',
        )

        # -- Center: trace viewer --
        center = pn.Column(
            self.trace_viewer,
            sizing_mode='stretch_both',
        )

        # -- Main body (left + center) --
        body = pn.Row(
            left_panel,
            center,
            sizing_mode='stretch_both',
            min_height=500,
        )

        # -- Full layout --
        layout = pn.Column(
            toolbar,
            pn.layout.Divider(margin=(0, 0)),
            body,
            self._status,
            sizing_mode='stretch_both',
            min_height=600,
        )
        return layout

    def show(self):
        """Display the explorer inline in a Jupyter notebook."""
        layout = self._build_layout()
        return layout

    def launch(self, port=0, threaded=True):
        """Open the explorer in a standalone browser window.

        Parameters
        ----------
        port : int
            Port for the local Bokeh server. 0 picks a random free port.
        threaded : bool
            If True (default), run the server in a background thread so
            the Python session stays interactive.
        """
        layout = self._build_layout()
        layout.show(title="SC Explorer - RetinAnalysis", port=port,
                    threaded=threaded)

    def servable(self):
        """Make the explorer servable via ``panel serve``."""
        layout = self._build_layout()
        layout.servable(title="SC Explorer - RetinAnalysis")
        return layout
