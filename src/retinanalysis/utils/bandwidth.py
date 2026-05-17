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


__all__ = ['print_network_gauge', 'network_bandwidth_gauge_widget']


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
    return box
