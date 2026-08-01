"""Dropdown-and-image browsers for notebook figures.

Several notebook sections have the same shape: a handful of things you could
render, each expensive enough that rendering all of them makes the cell
unreadable and slow. The answer each time is one dropdown, one view, rendered
on first selection and cached after — see ``chunk_summary.browse_chunk_summaries``
and ``raster.browse_epoch_rasters``.

This module holds the wiring those share so the pattern exists once.

Two deliberate choices, both learned the hard way elsewhere in this package:

- **HTML and Image widgets, never an Output.** ``Output`` with
  ``clear_output(wait=True)`` duplicates renders in JupyterLab;
  ``db_summary`` and ``sorting_qc`` both avoid it for the same reason.
- **The figure is closed after capture.** Otherwise it lands in the cell's
  inline output as well, and selecting one item stops meaning anything.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple


def figure_to_png(fig, dpi: int = 110) -> bytes:
    """Rasterize ``fig`` and close it. Empty bytes for a None figure."""
    import io

    import matplotlib.pyplot as plt

    if fig is None:
        return b''
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()


def png_browser(options: Sequence[Tuple[str, object]],
                render: Callable[[object], Tuple[Optional[str], bytes]],
                description: str = 'Show:',
                empty_message: str = 'Nothing to browse.'):
    """Dropdown over ``options``, showing one rendered item at a time.

    ``options`` is a sequence of ``(label, key)`` pairs — the label is what the
    dropdown shows, so put the identifying facts in it. ``render(key)`` returns
    ``(html_or_None, png_bytes)``; it is called at most once per key and the
    result cached, so an item nobody selects is never rendered.

    Returns the displayed widget, or None when there is nothing to show or
    ipywidgets is unavailable (the caller is expected to have a non-widget
    fallback for that case).
    """
    from IPython.display import display

    if not options:
        print(empty_message)
        return None

    try:
        import ipywidgets as widgets
    except ImportError:
        print('ipywidgets not available — cannot build the selector.')
        return None

    cache: Dict[object, Tuple[Optional[str], bytes]] = {}

    def _cached(key):
        if key not in cache:
            cache[key] = render(key)
        return cache[key]

    dropdown = widgets.Dropdown(
        options=list(options), description=description,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='max-content'))
    html = widgets.HTML()
    note = widgets.HTML()
    image = widgets.Image(format='png',
                          layout=widgets.Layout(max_width='100%'))

    def _show(key):
        body, png = _cached(key)
        html.value = body or ''
        image.value = png
        note.value = '' if png else '<em>Nothing to plot for this selection.</em>'

    dropdown.observe(lambda change: _show(change['new']), names='value')
    _show(dropdown.value)

    box = widgets.VBox([dropdown, html, note, image])
    display(box)
    return box


def lazy_tabs(titles, render, description: str = '', widths=None):
    """Tabbed views, each built on first look and kept after.

    ``render(index)`` returns the widget for that tab. It is called the first
    time a tab is selected, never before — so a tab holding a slow figure
    costs nothing until someone opens it, and opening it twice costs once.

    ``widths`` optionally gives a CSS width per tab (``None`` to leave one
    alone), applied to the container rather than to individual images. That
    catches everything a tab renders, including the per-cell-type browsers,
    which display themselves and so are never handed back to be resized.

    Returns the displayed ``Tab`` widget, or None without ipywidgets.
    """
    from IPython.display import display

    try:
        import ipywidgets as widgets
    except ImportError:
        print('ipywidgets not available — cannot build the tabbed view.')
        return None

    placeholders = [widgets.VBox([widgets.HTML('<em>loading…</em>')])
                    for _ in titles]
    tab = widgets.Tab(children=placeholders)
    for i, title in enumerate(titles):
        tab.set_title(i, title)

    built = set()

    def _build(index):
        if index in built:
            return
        built.add(index)
        children = list(tab.children)
        width = widths[index] if widths and index < len(widths) else None
        out = widgets.Output(layout=widgets.Layout(width=width) if width
                             else widgets.Layout())
        # The view is captured into an Output rather than returned as a
        # widget: the per-cell-type browsers display themselves, and this is
        # the one place that needs to catch that rather than re-plumb them.
        with out:
            widget = render(index)
            if widget is not None:
                display(widget)
        children[index] = out
        tab.children = tuple(children)

    tab.observe(lambda change: _build(change['new']), names='selected_index')
    _build(0)

    if description:
        display(widgets.HTML(description))
    display(tab)
    return tab
