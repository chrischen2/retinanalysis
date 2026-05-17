"""Session-wide network-bandwidth gauge.

The counter itself (cumulative bytes / resolution count) lives in
``retinanalysis.config.settings`` because that's where ``find_path``
charges resolutions. This module owns the *user-facing* presentation:

- :func:`print_network_gauge`  — one-liner for cells / scripts
- :func:`network_bandwidth_gauge_widget` — ipywidgets HTML chip that
  matches the §8 sorting-QC GUI style, with a Refresh + Reset button.

Cache resolutions (via ``ra.mirror_to_local_cache`` + the local_cache
tier) never bump the counter. So the gauge naturally goes flat once
your reads are served locally.
"""
from __future__ import annotations

from typing import Optional

from ..config.settings import (
    network_bytes_resolved, network_resolutions_count,
    reset_network_gauge, _TIER_KIND, mea_config,
)


__all__ = ['print_network_gauge', 'network_bandwidth_gauge_widget',
           'bandwidth_scope']


def _human_mb(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f'{n_bytes} B'
    if n_bytes < 1024 ** 2:
        return f'{n_bytes / 1024:.1f} KB'
    if n_bytes < 1024 ** 3:
        return f'{n_bytes / (1024 ** 2):.1f} MB'
    return f'{n_bytes / (1024 ** 3):.2f} GB'


def _gauge_summary() -> str:
    bytes_total = network_bytes_resolved()
    n_resol = network_resolutions_count()
    # List which configured tiers are network-mount, for context.
    net_tiers = sorted(t for t, k in _TIER_KIND.items() if k == 'network')
    paths = ', '.join(
        f'{t}({mea_config[t].get("data","?")})' for t in net_tiers
    ) or '(none)'
    return (f'network reads resolved this session: '
            f'{_human_mb(bytes_total)} across {n_resol} resolution(s). '
            f'Network-mount tiers: {paths}')


def print_network_gauge() -> None:
    """Print a one-line summary of the network-bandwidth gauge."""
    print(_gauge_summary())


class bandwidth_scope:
    """Context manager that prints a clear "from network: X MB" summary.

    Wraps any block of code that may trigger ``find_path`` resolutions
    (e.g. pipeline build, mirror, analysis run) so the user sees an
    obvious one-line summary after the block exits, distinguishing:

    - **💾 LOCAL / CACHE** when the block resolved 0 bytes from the
      network — every file came from a local SSD tier or the local
      cache.
    - **📡 NETWORK** when ≥1 file was resolved from a network-mount
      tier, showing total MB and the count of resolutions.

    Usage::

        with ra.bandwidth_scope('Pipeline build') as bw:
            pipeline = ra.create_mea_pipeline_cached(...)
        # Auto-prints: "📡 Pipeline build → ~31 MB from NETWORK (3 lookups)"
        # … or: "💾 Pipeline build → 0 MB from network — served local/cache"

        # bw.bytes_resolved / bw.resolutions are also available after exit.

    Multiple scopes can nest; each tracks its own delta.
    """

    def __init__(self, label: str = '',
                 also_print_total: bool = False,
                 enabled: bool = True):
        self.label = label
        self.also_print_total = also_print_total
        self.enabled = enabled
        self.bytes_resolved = 0
        self.resolutions = 0
        self._b0 = 0
        self._r0 = 0

    def __enter__(self):
        self._b0 = network_bytes_resolved()
        self._r0 = network_resolutions_count()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.bytes_resolved = network_bytes_resolved() - self._b0
        self.resolutions = network_resolutions_count() - self._r0
        if not self.enabled:
            return
        label = self.label or 'block'
        if self.bytes_resolved > 0:
            print(f'📡 {label} → ~{_human_mb(self.bytes_resolved)} '
                  f'from NETWORK ({self.resolutions} lookup'
                  f'{"s" if self.resolutions != 1 else ""})')
        else:
            print(f'💾 {label} → 0 MB from network — '
                  f'served local/cache')
        if self.also_print_total:
            print(f'   session total: '
                  f'{_human_mb(network_bytes_resolved())} '
                  f'across {network_resolutions_count()} lookup(s)')
        return False  # never suppress exceptions


def network_bandwidth_gauge_widget(initial_visible: bool = True):
    """Notebook chip + Refresh / Reset buttons for the network gauge.

    Returns
    -------
    ipywidgets.VBox
        Display with ``IPython.display.display(...)``. The chip is
        amber when bytes > 0, green when zero. Refresh updates the
        chip with the current counter; Reset zeroes the counter
        (useful when timing a specific analysis call).
    """
    import ipywidgets as widgets

    def _chip_html() -> str:
        bytes_total = network_bytes_resolved()
        n_resol = network_resolutions_count()
        if bytes_total == 0:
            bg, fg, icon = '#d1e7dd', '#0a3622', '💾'
            label = 'No network reads yet'
        else:
            bg, fg, icon = '#fff3cd', '#664d03', '📡'
            label = (f'~{_human_mb(bytes_total)} resolved from network · '
                     f'{n_resol} call(s)')
        return (f'<span style="background:{bg};color:{fg};'
                f'padding:3px 10px;border-radius:8px;font-family:monospace;'
                f'font-size:12px;">{icon} {label}</span> '
                f'<span style="color:#888;font-size:11px"> · '
                f'upper bound; OS may cache</span>')

    w_chip = widgets.HTML(value=_chip_html())
    w_refresh = widgets.Button(description='refresh',
                                layout=widgets.Layout(width='90px'))
    w_reset = widgets.Button(description='reset',
                              layout=widgets.Layout(width='90px'))

    def _on_refresh(_):
        w_chip.value = _chip_html()

    def _on_reset(_):
        reset_network_gauge()
        w_chip.value = _chip_html()

    w_refresh.on_click(_on_refresh)
    w_reset.on_click(_on_reset)

    box = widgets.HBox([w_chip, w_refresh, w_reset])
    box.layout.display = 'flex' if initial_visible else 'none'
    # Expose a programmatic refresh so embedders (e.g. the §8 sorting-QC
    # GUI) can update the chip after each Load click without forcing the
    # user to press the inline Refresh button.
    box.refresh = _on_refresh  # type: ignore[attr-defined]
    return box
