"""Stimulus regeneration.

Each Symphony protocol in the database has a matching regen module in this
package that, given a ``StimBlock``, returns an ``xarray.Dataset`` (or
``DataArray``) with whatever can be reconstructed: trajectories, frame
indexes, base images, etc.

Two design choices worth knowing:

1. **Resource discovery is best-effort.** Regen modules call
   :func:`retinanalysis.config.settings.find_protocol_repo` to locate the
   cloned source (turner-package, manookin-package, ...). If the repo is
   absent, the regen function still returns what it can derive purely from
   per-epoch parameters stored in the H5 — many protocols save their full
   trajectory / seeds per epoch, so the source code is only needed to
   resolve external resource files (.iml images, .mat libraries).

2. **Dispatch is by full protocol class name.** The registry key is the
   ``protocol_name`` string as stored in the database (e.g.
   ``"edu.washington.riekelab.turner.protocols.EyeMovementTrajectoryAlternatingBackground"``).
   Use :func:`register` to add new protocols.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Any

_REGISTRY: Dict[str, Callable] = {}

# Optional per-protocol "render the displayed canvas" implementations.
# Populated via :func:`register_renderer`; consumed by
# :func:`render_displayed_canvas`.
_RENDERERS: Dict[str, Callable] = {}


def register(protocol_name: str, fn: Callable) -> None:
    """Add a regen function for ``protocol_name`` (full Symphony class string)."""
    _REGISTRY[protocol_name] = fn


def register_renderer(protocol_name: str, fn: Callable) -> None:
    """Add a ``render_displayed_canvas`` implementation for ``protocol_name``."""
    _RENDERERS[protocol_name] = fn


def available_protocols():
    """Return the list of protocol class names that have a registered regen function."""
    return sorted(_REGISTRY.keys())


def regen_stimulus(stim_block: Any, verbose: bool = True, **kwargs):
    """Regenerate stimulus frames/metadata for a StimBlock.

    Dispatches on ``stim_block.protocol_name``. If no regen function is
    registered for that protocol, or the protocol's required repo isn't
    cloned locally, prints a clear message and returns ``None``.

    Pass-through kwargs are forwarded to the underlying regen function (e.g.
    ``downsample=4`` for protocols that render to canvas pixels).
    """
    protocol_name = getattr(stim_block, 'protocol_name', None)
    if protocol_name is None:
        if verbose:
            print('regen_stimulus: stim_block has no protocol_name attribute; nothing to do.')
        return None

    fn = _REGISTRY.get(protocol_name)
    if fn is None:
        if verbose:
            print(f'regen_stimulus: no regen function registered for "{protocol_name}".')
            print(f'  Registered protocols: {available_protocols()}')
        return None

    if verbose:
        print(f'regen_stimulus: using {fn.__module__}.{fn.__name__} for {protocol_name}')
    return fn(stim_block, verbose=verbose, **kwargs)


def render_displayed_canvas(stim_ds, *args, verbose: bool = True, **kwargs):
    """Render the canvas-pixel frame as displayed on the monitor.

    Dispatches on ``stim_ds.attrs['protocol_name']`` to a per-protocol
    renderer. Each renderer composites the protocol's stim onto a
    canvas-sized buffer, fills any background, applies per-frame
    trajectory / jitter as appropriate, and clips to the rig canvas. The
    result is exactly what the retina saw.

    All extra positional/keyword args are forwarded to the renderer (e.g.
    ``epoch=0, frame=900``). If no renderer is registered for this
    protocol, returns ``None`` and prints a notice.
    """
    protocol_name = stim_ds.attrs.get('protocol_name') if hasattr(stim_ds, 'attrs') else None
    fn = _RENDERERS.get(protocol_name)
    if fn is None:
        if verbose:
            print(f'render_displayed_canvas: no renderer registered for "{protocol_name}".')
            print(f'  Renderers available for: {sorted(_RENDERERS)}')
        return None
    return fn(stim_ds, *args, **kwargs)


# Side-effect imports register their protocols in _REGISTRY (and _RENDERERS,
# where applicable).
from . import eye_movement_alt_bg as _emab  # noqa: F401,E402
from . import variable_mean_spatial_noise as _vmsn  # noqa: F401,E402
from . import mea_variable_mean_noise_cone  # noqa: F401,E402

register_renderer(_emab.PROTOCOL_NAME, _emab.render_displayed_canvas)
register_renderer(_vmsn.PROTOCOL_NAME, _vmsn.render_displayed_canvas)

__all__ = [
    'regen_stimulus', 'register', 'available_protocols',
    'render_displayed_canvas', 'register_renderer',
]
