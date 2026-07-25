"""Per-protocol single-cell analyses.

One module per Symphony protocol, loaded lazily so importing the package stays
cheap:

    from retinanalysis.SCutils.protocols import expanding_spots as es
"""

__all__ = ["expanding_spots"]


def __getattr__(name):
    if name in __all__:
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(
        f"module 'retinanalysis.SCutils.protocols' has no attribute {name!r}")
