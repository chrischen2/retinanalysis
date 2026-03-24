"""Analysis panel: plugin discovery, activation, and parameter controls."""
 
import panel as pn
import param
 
from retinanalysis.singleCellGUI.state import AppState
from retinanalysis.singleCellGUI.analysis import discover_plugins
from retinanalysis.singleCellGUI.analysis._base import AnalysisPlugin
 
 
class AnalysisPanel(pn.viewable.Viewer):
    """Right sidebar showing available analysis plugins with parameter controls."""
 
    state = param.ClassSelector(class_=AppState)
 
    def __init__(self, state, extra_plugin_dirs=None, **params):
        super().__init__(state=state, **params)
 
        # Discover plugins
        self._plugin_classes = discover_plugins(extra_dirs=extra_plugin_dirs)
        self._plugin_instances = {}  # name -> instance
        self._plugin_widgets = {}    # name -> Column of param widgets
 
        # Build UI
        self._checkboxes = {}
        self._controls_column = pn.Column()
        self._checkbox_column = pn.Column()
 
        for plugin_name, plugin_cls in self._plugin_classes.items():
            cb = pn.widgets.Checkbox(name=plugin_name, value=False)
            cb.param.watch(lambda event, pn=plugin_name: self._on_toggle(pn, event), 'value')
            self._checkboxes[plugin_name] = cb
            self._checkbox_column.append(cb)
 
    def _on_toggle(self, plugin_name, event):
        """Enable/disable a plugin and show/hide its controls."""
        if event.new:
            # Create instance if needed
            if plugin_name not in self._plugin_instances:
                instance = self._plugin_classes[plugin_name]()
                self._plugin_instances[plugin_name] = instance
 
            instance = self._plugin_instances[plugin_name]
 
            # Build param controls
            user_params = instance.user_params()
            if user_params:
                widgets = pn.Param(
                    instance,
                    parameters=user_params,
                    show_name=False,
                    widgets={},
                )
                # Watch all user params to re-trigger plot
                for p_name in user_params:
                    instance.param.watch(self._on_param_change, p_name)
 
                controls = pn.Column(
                    pn.pane.Markdown(f"**{plugin_name}**", margin=(5, 0)),
                    widgets,
                    pn.layout.Divider(),
                )
            else:
                controls = pn.Column(
                    pn.pane.Markdown(f"**{plugin_name}** _(no parameters)_", margin=(5, 0)),
                    pn.layout.Divider(),
                )
            self._plugin_widgets[plugin_name] = controls
        else:
            # Remove from active
            self._plugin_widgets.pop(plugin_name, None)
            self._plugin_instances.pop(plugin_name, None)
 
        # Rebuild controls column
        self._controls_column.clear()
        for name, widget_col in self._plugin_widgets.items():
            self._controls_column.append(widget_col)
 
        # Update state
        self.state.active_analyses = list(self._plugin_instances.values())
 
    def _on_param_change(self, event):
        """Re-trigger plot when a plugin parameter changes."""
        # Touch active_analyses to trigger the watcher in TraceViewer
        self.state.active_analyses = list(self._plugin_instances.values())
 
    def __panel__(self):
        return pn.Column(
            pn.pane.Markdown("### Analysis Plugins", margin=(0, 5)),
            self._checkbox_column,
            pn.layout.Divider(),
            self._controls_column,
            sizing_mode='stretch_width',
        )