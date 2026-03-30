"""DJ-GUI: DataJoint web interface for retinanalysis.

Launches the dj-server Flask backend and the Next.js frontend to provide
a browser-based GUI for querying, visualizing, and tagging experiment data.

Quick start::

    import retinanalysis as ra
    ra.populate_database()
    ra.DJ_GUI.launch()
"""

from retinanalysis.DJ_GUI.launcher import launch

__all__ = ["launch"]
