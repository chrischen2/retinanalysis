"""Single-Cell Explorer GUI for retinanalysis.

Quick start in a Jupyter notebook::

    import retinanalysis as ra
    ra.populate_database()
    explorer = ra.gui.launch_explorer()
"""

from retinanalysis.gui.app import SCExplorer


def launch_explorer(extra_plugin_dirs=None, inline=False, port=0):
    """Create and display the Single-Cell Explorer GUI.

    Parameters
    ----------
    extra_plugin_dirs : list[str] | None
        Additional directories to scan for analysis plugin .py files.
    inline : bool
        If False (default), open the GUI in a standalone browser window.
        If True, display inline in a Jupyter notebook.
    port : int
        Port for the standalone server (0 picks a random free port).
        Only used when ``inline=False``.

    Returns
    -------
    SCExplorer
        The explorer instance. Access ``explorer.state`` for loaded data
        and ``explorer.export_panel.export_selected()`` for programmatic export.
    """
    explorer = SCExplorer(extra_plugin_dirs=extra_plugin_dirs)

    if inline:
        display_obj = explorer.show()
        try:
            from IPython.display import display
            display(display_obj)
        except ImportError:
            pass
    else:
        explorer.launch(port=port)

    return explorer
