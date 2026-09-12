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
                empty_message: str = 'Nothing to browse.',
                display_widget: bool = True):
    """Dropdown over ``options``, showing one rendered item at a time.

    ``options`` is a sequence of ``(label, key)`` pairs — the label is what the
    dropdown shows, so put the identifying facts in it. ``render(key)`` returns
    ``(html_or_None, png_bytes)``; it is called at most once per key and the
    result cached, so an item nobody selects is never rendered.

    Returns the widget, or None when there is nothing to show or ipywidgets is
    unavailable (the caller is expected to have a non-widget fallback for
    that case). Set ``display_widget=False`` when embedding it in another
    widget such as :func:`lazy_tabs`.
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
    if display_widget:
        display(box)
    return box


def lazy_tabs(titles, render, description: str = '', widths=None):
    """Tabbed views, each built on first look and kept after.

    ``render(index)`` returns the widget for that tab. It is called the first
    time a tab is selected, never before — so a tab holding a slow figure
    costs nothing until someone opens it, and opening it twice costs once.

    ``widths`` optionally gives a CSS width per tab (``None`` to leave one
    alone), applied to the widget returned by ``render``.

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
        widget = render(index)
        if widget is None:
            widget = widgets.HTML('<em>Nothing to plot.</em>')
        if width:
            widget.layout.width = width
        children[index] = widget
        tab.children = tuple(children)

    tab.observe(lambda change: _build(change['new']), names='selected_index')
    _build(0)

    if description:
        display(widgets.HTML(description))
    display(tab)
    return tab


def saved_figure_review_browser(
        options, *, load_item, sections, panels, figure_options, describe,
        review_flags, set_keep, set_example, item_description='Cell:',
        section_description='Recording:', toggle_panels=()):
    """Browse saved PNGs and review an item using caller-supplied adapters.

    ``load_item(key)`` returns any caller-owned record. ``sections(item)`` gives
    its selectable conditions; ``figure_options(item, panel, section)`` gives
    ``(label, path)`` pairs. ``describe(item, section)`` returns plain text.
    ``review_flags(item, section)`` returns ``(kept, is_example)``.
    ``set_keep(item, section, bool)`` and ``set_example(item, bool)`` persist
    decisions; example scope is chosen by the caller. No database or protocol
    dependency, automatic display, or file mutation occurs on opening.
    Panels listed in ``toggle_panels`` show all figure choices as buttons.

    The returned widget's ``review_state`` exposes selectors and buttons for
    notebook integration and testing. Missing images show an empty panel;
    loading/persistence failures appear in the status line.
    """
    import html
    from pathlib import Path
    import ipywidgets as widgets

    if not options:
        raise ValueError('No saved items were found for visual review.')
    selector = widgets.Dropdown(options=list(options), description=item_description,
                                layout=widgets.Layout(width='520px'))
    section_selector = widgets.Dropdown(options=(), description=section_description,
                                        layout=widgets.Layout(width='320px'))
    info, status = widgets.HTML(), widgets.HTML()
    keep = widgets.Button(description='Keep', button_style='success', icon='check')
    remove = widgets.Button(description='Remove', button_style='danger', icon='trash')
    example = widgets.Button(description='Set example', icon='star-o')
    selectors = {
        name: (widgets.ToggleButtons(description=name + ':',
                                     style={'button_width': 'auto'},
                                     layout=widgets.Layout(width='100%'))
               if name in toggle_panels else widgets.Dropdown(description=name + ':'))
        for name in panels}
    images = {name: widgets.Image(format='png', layout=widgets.Layout(
        width='100%', height='auto')) for name in panels}
    state = dict(item=None, selector=selector, section_selector=section_selector,
                 figure_selectors=selectors, figure_images=images, keep_button=keep,
                 remove_button=remove, example_button=example, status=status,
                 is_example=False, loading=False)

    def show_image(panel):
        path = selectors[panel].value
        images[panel].value = Path(path).read_bytes() if path else b''

    def refresh_status():
        kept, is_example = review_flags(state['item'], section_selector.value)
        state['is_example'] = bool(is_example)
        example.description = 'Unset example' if is_example else 'Set example'
        example.icon = 'star' if is_example else 'star-o'
        example.button_style = 'warning' if is_example else ''
        decision = '<b style="color:#188038">KEPT</b>' if kept else 'not kept'
        flag = '<b>★ EXAMPLE</b>' if is_example else 'Example: false'
        status.value = f'Visual inspection: {decision} | {flag}'

    def show_section(_change=None):
        if state['loading'] or state['item'] is None:
            return
        section = section_selector.value
        enabled = section is not None
        keep.disabled = remove.disabled = example.disabled = not enabled
        if not enabled:
            info.value = ''
            status.value = 'No review sections available.'
        else:
            info.value = '<b>' + html.escape(describe(state['item'], section)) + '</b>'
        for panel, dropdown in selectors.items():
            choices = (figure_options(state['item'], panel, section) if enabled else [])
            choices = [(label, str(path)) for label, path in choices if Path(path).is_file()]
            dropdown.options = choices or [('not available', '')]
            dropdown.value = choices[0][1] if choices else ''
            dropdown.disabled = not bool(choices)
            show_image(panel)
        if enabled:
            refresh_status()

    def show_item(_change=None):
        state['loading'] = True
        state['item'] = None
        state['is_example'] = False
        example.description, example.icon, example.button_style = 'Set example', 'star-o', ''
        section_selector.disabled = True
        for dropdown in selectors.values():
            dropdown.disabled = True
        keep.disabled = remove.disabled = example.disabled = True
        info.value = ''
        for image in images.values():
            image.value = b''
        try:
            state['item'] = load_item(selector.value)
            section_selector.options = list(sections(state['item']))
            section_selector.value = (section_selector.options[0]
                                      if section_selector.options else None)
        finally:
            state['loading'] = False
        section_selector.disabled = not bool(section_selector.options)
        show_section()

    def guarded(action):
        def run(*args):
            try:
                action(*args)
            except Exception as exc:
                status.value = '<b>Review error:</b> ' + html.escape(str(exc))
        return run

    def decide(value):
        set_keep(state['item'], section_selector.value, value)
        refresh_status()

    def toggle_example(_button):
        # Read current persisted state so a second browser cannot stale-toggle it.
        _, current = review_flags(state['item'], section_selector.value)
        set_example(state['item'], not current)
        refresh_status()

    selector.observe(guarded(show_item), names='value')
    section_selector.observe(guarded(show_section), names='value')
    for panel, dropdown in selectors.items():
        dropdown.observe(guarded(lambda _change, name=panel: show_image(name)), names='value')
    keep.on_click(guarded(lambda _button: decide(True)))
    remove.on_click(guarded(lambda _button: decide(False)))
    example.on_click(guarded(toggle_example))
    show_item()
    box = widgets.VBox([
        widgets.HBox([selector, section_selector]), info,
        widgets.HBox([keep, remove, example]), status,
        widgets.GridBox([widgets.VBox([widgets.HTML('<b>' + html.escape(name) + '</b>'),
                                      selectors[name], images[name]]) for name in panels],
                        layout=widgets.Layout(grid_template_columns='repeat(2, minmax(0, 1fr))',
                                              grid_gap='12px'))])
    box.review_state = state
    return box
