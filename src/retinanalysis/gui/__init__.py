"""Single-Cell Explorer GUI for retinanalysis.
 
Quick start in a Jupyter notebook::
 
    import retinanalysis as ra
    ra.populate_database()
    explorer = ra.gui.launch_explorer()
"""
 
from retinanalysis.gui.app import SCExplorer
 
 
def launch_explorer(extra_plugin_dirs=None):
    """Create and display the Single-Cell Explorer GUI.
 
    Parameters
    ----------
    extra_plugin_dirs : list[str] | None
        Additional directories to scan for analysis plugin .py files.
 
    Returns
    -------
    SCExplorer
        The explorer instance. Access ``explorer.state`` for loaded data
        and ``explorer.export_panel.export_selected()`` for programmatic export.
    """
    explorer = SCExplorer(extra_plugin_dirs=extra_plugin_dirs)
    display_obj = explorer.show()
    try:
        from IPython.display import display
        display(display_obj)
    except ImportError:
        pass
    return explorer