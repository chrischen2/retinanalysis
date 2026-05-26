"""Guards the lazy top-level import (PEP 562 ``__getattr__``).

These tests lock in the two properties the lazy ``__init__`` exists to
provide, so a future eager re-import can't silently regress kernel-start
time or drop a public name:

1. ``import retinanalysis`` is cheap — it must NOT pull DataJoint,
   matplotlib, scipy or xarray. Those load only on first access to
   something that needs them.
2. Every name the package advertises (both registries) resolves, and the
   incidental names that the old ``from .x import *`` statements leaked are
   still recoverable via the ``_STAR_MODULES`` fallback.

The import-footprint check runs in a fresh subprocess so it sees a clean
``sys.modules`` regardless of what the rest of the suite has imported.
"""
from __future__ import annotations

import subprocess
import sys

# Heavy libraries that the eager __init__ used to import unconditionally.
# A bare `import retinanalysis` must leave all of these unloaded.
_HEAVY = ['datajoint', 'matplotlib.pyplot', 'scipy.ndimage', 'xarray']


def test_bare_import_does_not_load_heavy_libraries():
    code = (
        "import sys, retinanalysis\n"
        f"loaded = [m for m in {_HEAVY!r} if m in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    out = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, check=True
    )
    loaded = [m for m in out.stdout.strip().split(',') if m]
    assert loaded == [], f"import retinanalysis eagerly loaded heavy libs: {loaded}"


def test_every_registered_name_resolves():
    import retinanalysis as ra

    names = set(ra._SUBMODULES) | set(ra._ATTR_TO_MODULE)
    for name in sorted(names):
        getattr(ra, name)  # raises AttributeError if the mapping is wrong


def test_leaked_star_import_names_still_resolve():
    """Names that only existed because an old ``import *`` leaked them must
    still be reachable through the fallback (workflow-preservation)."""
    import retinanalysis as ra

    leaked = ['Counter', 'DataFrame', 'Ellipse', 'TYPE_CHECKING', 'annotations',
              'display', 'gaussian_filter', 'load_vision_data', 'loadmat',
              'List', 'Optional', 'Union', 'Tuple', 'Dict']
    for name in leaked:
        getattr(ra, name)


def test_common_public_api_is_callable():
    import retinanalysis as ra

    assert ra.np is __import__('numpy')
    assert ra.DATA_DIR == ra.config.settings.DATA_DIR
    assert callable(ra.analyze_experiment)
    assert callable(ra.load_offline_data)
    assert ra.PURGE_CONFIRM_TOKEN == 'YES_DELETE_ALL'


if __name__ == '__main__':
    test_bare_import_does_not_load_heavy_libraries()
    test_every_registered_name_resolves()
    test_leaked_star_import_names_still_resolve()
    test_common_public_api_is_callable()
    print('lazy-import tests OK')
