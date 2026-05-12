"""DJ-GUI: DataJoint web interface for retinanalysis.

Launches the dj-server Flask backend and the Next.js frontend to provide
a browser-based GUI for querying, visualizing, and tagging experiment data.

Quick start::

    import retinanalysis as ra
    ra.populate_database()
    ra.DJ_GUI.launch()
"""

__all__ = ["launch"]


def __getattr__(name):
    # Lazy import: pulling in the Flask/Next.js launcher is slow and only needed
    # when the user actually calls ra.DJ_GUI.launch(). PEP 562 module __getattr__.
    if name == "launch":
        from retinanalysis.DJ_GUI.launcher import launch
        return launch
    raise AttributeError(f"module 'retinanalysis.DJ_GUI' has no attribute {name!r}")
