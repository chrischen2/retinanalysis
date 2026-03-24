"""SCExplorer: main application composing all panels into a 3-column layout."""

import panel as pn

from retinanalysis.singleCellGUI.state import AppState
from retinanalysis.singleCellGUI.panels.experiment_selector import ExperimentSelector
from retinanalysis.singleCellGUI.panels.filter_panel import FilterPanel
from retinanalysis.singleCellGUI.panels.data_tree import DataTree
from retinanalysis.singleCellGUI.panels.trace_viewer import TraceViewer
from retinanalysis.singleCellGUI.panels.analysis_panel import AnalysisPanel
from retinanalysis.singleCellGUI.panels.export_panel import ExportPanel


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

    def _build_layout(self):
        """Compose the 3-column layout."""
        sidebar = pn.Column(
            self.experiment_selector,
            pn.layout.Divider(),
            self.filter_panel,
            pn.layout.Divider(),
            self.data_tree,
            pn.layout.Divider(),
            self.export_panel,
            width=380,
            scroll=True,
        )

        main_area = pn.Column(
            self.trace_viewer,
            sizing_mode='stretch_both',
        )

        analysis_sidebar = pn.Column(
            self.analysis_panel,
            width=280,
            scroll=True,
        )

        layout = pn.Row(
            sidebar,
            main_area,
            analysis_sidebar,
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
