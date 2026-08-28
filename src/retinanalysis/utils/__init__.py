"""retinanalysis.utils — lazy subpackage.

The path constants (``DATA_DIR``, ...) load eagerly: they come from
``config.settings``, which is cheap (stdlib only) and is imported by many
modules via ``from retinanalysis.utils import DATA_DIR``. Keeping them eager
means those imports never pull DataJoint.

The DataJoint-backed names (``schema``, ``database_pop``, ``datajoint_utils``,
``cell_type_utils`` and their re-exports) load on first access via PEP 562
``__getattr__``. This is what lets a light utility — ``psth``, ``raster``,
``offline_store``, ... — be imported without dragging in DataJoint, since
importing any ``retinanalysis.utils`` submodule first runs this ``__init__``.
"""
from __future__ import annotations

import importlib

from retinanalysis.config.settings import (ANALYSIS_DIR,
                                           DATA_DIR,
                                           RAW_DIR,
                                           H5_DIR,
                                           META_DIR,
                                           TAGS_DIR,
                                           QUERY_DIR,
                                           OUTPUT_DIR,
                                           USER)

_SUBMODULES = {
    'schema': 'retinanalysis.config.schema',
    'database_pop': 'retinanalysis.utils.database_pop',
    'datajoint_utils': 'retinanalysis.utils.datajoint_utils',
    'cell_type_utils': 'retinanalysis.utils.cell_type_utils',
    'isomerization': 'retinanalysis.utils.isomerization',
    'light_levels': 'retinanalysis.utils.light_levels',
}

_ATTR_TO_MODULE = {
    'apply_publication_style': 'retinanalysis.utils.style',
    'format_figure': 'retinanalysis.utils.style',
    'igor_output': 'retinanalysis.utils.igor_export',
    'igor_axis_struct': 'retinanalysis.utils.igor_export',
    'igor_dir': 'retinanalysis.utils.igor_export',
    'export_axis_to_h5': 'retinanalysis.utils.igor_export',
    'export_figure_to_h5': 'retinanalysis.utils.igor_export',
    'get_exp_summary': 'retinanalysis.utils.datajoint_utils',
    'load_canonical_cell_types': 'retinanalysis.utils.cell_type_utils',
    'map_cell_type': 'retinanalysis.utils.cell_type_utils',
    'filter_available_types': 'retinanalysis.utils.cell_type_utils',
    'collecting_area': 'retinanalysis.utils.isomerization',
    'convert_isomerizations': 'retinanalysis.utils.isomerization',
    'convert_experiment_receptors': 'retinanalysis.utils.isomerization',
    'epoch_group_ndf_table': 'retinanalysis.utils.isomerization',
    'infer_rig_name': 'retinanalysis.utils.isomerization',
    'isomerizations_converter': 'retinanalysis.utils.isomerization',
    'isomerization_converter_widget': 'retinanalysis.utils.isomerization',
    'isomerization_converter_browser': 'retinanalysis.utils.isomerization',
    'isomerizations_per_watt': 'retinanalysis.utils.isomerization',
    'filter_wheel_ndf_from_epoch_parameters': 'retinanalysis.utils.light_levels',
    'read_block_light_settings': 'retinanalysis.utils.light_levels',
    'visual_stimulus_max': 'retinanalysis.utils.light_levels',
    'select_flux_factor': 'retinanalysis.utils.isomerization',
}


def __getattr__(name):
    target = _SUBMODULES.get(name)
    if target is not None:
        module = importlib.import_module(target)
        globals()[name] = module
        return module

    target = _ATTR_TO_MODULE.get(name)
    if target is not None:
        value = getattr(importlib.import_module(target), name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBMODULES) | set(_ATTR_TO_MODULE))
