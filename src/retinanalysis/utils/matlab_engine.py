"""Lazy MATLAB engine singleton.

Why this module exists
----------------------
The MATLAB Engine API for Python lets you call MATLAB functions from a
Python process. We use it (when available) to get byte-exact MATLAB RNG
parity in :mod:`retinanalysis.regen` — NumPy's MT19937 matches MATLAB's
``.rand()`` byte-for-byte, but not ``.randn()`` (different normal-from-
uniform algorithms).

Design choices
--------------
- **Lazy import.** `matlab.engine` takes a second or two to import and is
  not always installed; we never import at module load. A `try` inside
  :func:`get_matlab_engine` keeps the failure mode obvious and importable
  on any machine.
- **Singleton.** Starting MATLAB takes ~5-10 s and consumes hundreds of MB.
  We cache a single engine for the lifetime of the Python process; callers
  share it. Use :func:`shutdown_matlab_engine` to release.
- **Cheap availability check.** :func:`is_matlab_engine_available` only
  asks whether `matlab.engine` can be imported — it does NOT spin up the
  engine. Use it to decide between MATLAB-backed and pure-Python code
  paths before paying the startup cost.

How to install
--------------
With MATLAB installed at ``/Applications/MATLAB_R<rel>.app``::

    pip install /Applications/MATLAB_R<rel>.app/extern/engines/python

Or, equivalently, from inside MATLAB::

    >> cd(fullfile(matlabroot,'extern','engines','python'))
    >> system('python -m pip install .')
"""

from __future__ import annotations

import atexit
from typing import Optional

_engine = None  # cached MATLAB engine handle
_engine_failed = False  # remember failed starts so we don't retry every call


def is_matlab_engine_available() -> bool:
    """Return True if `matlab.engine` is importable (doesn't start MATLAB)."""
    try:
        import matlab.engine  # noqa: F401
        return True
    except Exception:
        return False


def get_matlab_engine(start_options: str = '-nodesktop -nosplash'):
    """Return the cached MATLAB engine, starting one on first call.

    Returns ``None`` if `matlab.engine` isn't installed or MATLAB can't be
    started; in that case the failure is sticky for the rest of the process
    so callers don't pay repeated startup attempts.

    Pass ``start_options`` to control MATLAB launch flags (default is
    headless: no desktop, no splash).
    """
    global _engine, _engine_failed
    if _engine is not None:
        return _engine
    if _engine_failed:
        return None

    try:
        import matlab.engine
    except Exception as e:
        print(f'[matlab_engine] matlab.engine not installed: {e}')
        print('[matlab_engine] install via: '
              'pip install /Applications/MATLAB_R<release>.app/extern/engines/python')
        _engine_failed = True
        return None

    print('[matlab_engine] starting MATLAB engine (~5-10s)...')
    try:
        _engine = matlab.engine.start_matlab(start_options)
    except Exception as e:
        print(f'[matlab_engine] failed to start MATLAB: {e}')
        _engine_failed = True
        return None

    print('[matlab_engine] MATLAB engine ready.')
    atexit.register(shutdown_matlab_engine)
    return _engine


def shutdown_matlab_engine() -> None:
    """Stop the cached MATLAB engine if one is running."""
    global _engine
    if _engine is None:
        return
    try:
        _engine.quit()
    except Exception:
        pass
    _engine = None
