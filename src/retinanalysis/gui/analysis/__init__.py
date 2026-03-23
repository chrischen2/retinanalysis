"""Analysis plugin discovery.
 
Scans this directory (and optional extra directories) for AnalysisPlugin subclasses.
"""
 
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
 
from retinanalysis.gui.analysis._base import AnalysisPlugin
 
 
def discover_plugins(extra_dirs=None):
    """Find all AnalysisPlugin subclasses in the built-in and extra directories.
 
    Parameters
    ----------
    extra_dirs : list[str | Path] | None
        Additional directories to scan for plugin .py files.
 
    Returns
    -------
    dict[str, type]
        {plugin_name: PluginClass} for every discovered plugin.
    """
    dirs = [Path(__file__).parent]
    if extra_dirs:
        dirs.extend(Path(d) for d in extra_dirs)
 
    plugins = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for py_file in sorted(d.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    f"ra_plugin_{py_file.stem}", py_file
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for _, obj in inspect.getmembers(mod, inspect.isclass):
                    if issubclass(obj, AnalysisPlugin) and obj is not AnalysisPlugin:
                        instance = obj()
                        plugins[instance.name] = obj
            except Exception:
                pass
    return plugins