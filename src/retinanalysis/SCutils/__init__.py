"""SCutils - Standalone single-cell analysis utilities.

Provides pure-function implementations of Clarinet's builtinProcessors
and builtinExtractors, independent of any GUI framework, plus `explore`:
read-only DataJoint queries and notebook table rendering for patch data.

Submodules and file-conversion helpers are loaded lazily on first access so
that `import retinanalysis` stays fast for users who never touch the
single-cell path.
"""

_MODULES = {
    "auisql_json", "dataprocessor", "protocols", "explore", "h5_json",
    "recording_classifier",
}
_ATTRIBUTES = {
    "AuisqlReader": ("auisql_json", "AuisqlReader"),
    "SingleCellJsonUpdate": ("h5_json", "SingleCellJsonUpdate"),
    "convert_auisql_to_json": ("auisql_json", "convert_auisql_to_json"),
    "update_single_cell_json": ("h5_json", "update_single_cell_json"),
    "parse_single_cell_recording_modes": (
        "recording_mode", "parse_single_cell_recording_modes"),
    "extract_recording_block_features": (
        "recording_classifier", "extract_recording_block_features"),
    "load_recording_technique_classifier": (
        "recording_classifier", "load_recording_technique_classifier"),
    "predict_recording_techniques": (
        "recording_classifier", "predict_recording_techniques"),
    "retrain_recording_technique_classifier": (
        "recording_classifier", "retrain_recording_technique_classifier"),
}

__all__ = sorted(_MODULES | set(_ATTRIBUTES))


def __getattr__(name):
    import importlib

    if name in _MODULES:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    if name in _ATTRIBUTES:
        module_name, attribute_name = _ATTRIBUTES[name]
        mod = importlib.import_module(f".{module_name}", __name__)
        value = getattr(mod, attribute_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'retinanalysis.SCutils' has no attribute {name!r}")
