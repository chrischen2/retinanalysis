"""Base class for analysis plugins.
 
To create a new plugin, subclass AnalysisPlugin and implement process().
Drop the .py file into the gui/analysis/ directory (or a custom directory)
and it will be auto-discovered.
"""
 
import param
import numpy as np
 
 
class AnalysisPlugin(param.Parameterized):
    """Base class for trace analysis plugins.
 
    Subclasses must implement process() and set a unique ``name``.
    Tunable parameters should be declared as param attributes so the GUI
    can auto-generate controls for them.
    """
 
    name = param.String(default="Unnamed Plugin", doc="Display name in the GUI")
    description = param.String(default="", doc="Short description / tooltip")
 
    def process(self, trace: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        """Transform a raw trace and return the processed result.
 
        Parameters
        ----------
        trace : np.ndarray
            1-D amplitude array for a single epoch.
        sample_rate : float
            Sampling rate in Hz.
        **kwargs
            Extra context (e.g. pre_time_ms, stim_time_ms) passed by the
            TraceViewer so plugins can use stimulus timing.
 
        Returns
        -------
        np.ndarray
            Processed trace (same length as input).
        """
        raise NotImplementedError
 
    def get_label(self) -> str:
        """Legend label for the processed trace."""
        return self.name
 
    @classmethod
    def user_params(cls):
        """Return the list of param names that users can tune (excludes name/description)."""
        skip = {'name', 'description'}
        return [p for p in cls.param if p not in skip]