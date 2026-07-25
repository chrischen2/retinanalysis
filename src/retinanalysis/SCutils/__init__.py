"""SCutils - Standalone single-cell analysis utilities.

Provides pure-function implementations of Clarinet's builtinProcessors
and builtinExtractors, independent of any GUI framework, plus `explore`:
read-only DataJoint queries and notebook table rendering for patch data.

Submodules (`dataprocessor`, `protocols`, `explore`) are loaded lazily on
first access so that `import retinanalysis` stays fast for users who never
touch the single-cell path.
"""

__all__ = ["dataprocessor", "protocols", "explore"]


def __getattr__(name):
    if name in __all__:
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'retinanalysis.SCutils' has no attribute {name!r}")
