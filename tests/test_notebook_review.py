"""Shared review utilities work without VariableMeanNoise or a database."""
import base64
from pathlib import Path

import pandas as pd
import pytest

from retinanalysis.utils.browse import saved_figure_review_browser
from retinanalysis.utils.review_store import ReviewStore


def test_browser_group_filter_preserves_selection_and_loads_only_new_item():
    loads, decisions = [], []
    browser = saved_figure_review_browser(
        [('one', 1), ('two', 2), ('three', 3)],
        item_groups={1: 'ON-midget', 2: 'ON-parasol', 3: 'ON-midget'},
        group_description='Cell type:', load_item=lambda key: loads.append(key) or key,
        sections=lambda item: ['spike', 'exc'], panels=('Trace',),
        figure_options=lambda *args: [], describe=lambda item, section: str(item),
        review_flags=lambda *args: (False, False),
        set_keep=lambda *args: decisions.append(args),
        set_example=lambda *args: decisions.append(args))
    state = browser.review_state
    state['section_selector'].value = 'exc'
    state['group_selector'].value = 'ON-midget'
    assert state['selector'].options == (('one', 1), ('three', 3))
    assert loads == [1]  # same cell/recording: filtering needs no image/data reload
    assert state['section_selector'].value == 'exc'
    state['selector'].value = 3
    state['group_selector'].value = None
    assert state['selector'].value == 3
    assert loads == [1, 3]
    state['group_selector'].value = 'ON-parasol'
    assert state['selector'].value == 2
    assert loads == [1, 3, 2]  # one load, not an intermediate reset to cell 1
    assert decisions == []


def test_recording_filter_intersects_cell_type_and_handles_empty_selection():
    loads, decisions = [], []
    recordings = {1: ('spike', 'exc'), 2: ('spike',), 3: ('exc',)}
    browser = saved_figure_review_browser(
        [('one', 1), ('two', 2), ('three', 3)],
        item_groups={1: 'midget', 2: 'parasol', 3: 'midget'},
        item_sections=recordings, load_item=lambda key: loads.append(key) or key,
        sections=lambda item: recordings[item], panels=('Trace',),
        figure_options=lambda *args: [], describe=lambda *args: '',
        review_flags=lambda *args: (False, False),
        set_keep=lambda *args: decisions.append(args),
        set_example=lambda *args: decisions.append(args))
    state = browser.review_state
    state['section_filter'].value = 'exc'
    assert state['selector'].options == (('one', 1), ('three', 3))
    assert state['section_selector'].options == ('exc',)
    assert loads == [1]  # changing recording within the cell needs no data reload
    state['selector'].value = 3
    assert state['section_selector'].value == 'exc'
    state['group_selector'].value = 'parasol'
    assert state['selector'].options == ()
    assert state['item'] is None
    assert state['keep_button'].disabled and state['example_button'].disabled
    assert state['figure_images']['Trace'].value == b''
    assert 'No cells match' in state['status'].value
    assert loads == [1, 3]  # empty intersections must not load None
    state['section_filter'].value = None
    assert state['selector'].value == 2
    assert state['section_selector'].value == 'spike'
    state['group_selector'].value = None
    assert state['selector'].value == 2
    assert loads == [1, 3, 2]
    assert decisions == []


def test_review_store_identity_defaults_and_independent_updates(tmp_path):
    store = ReviewStore(tmp_path / 'examples.csv', keys=('experiment', 'unit'),
                        columns=('experiment', 'unit', 'index', 'example'),
                        boolean_columns=('example',))
    identity = {'experiment': '001', 'unit': '007'}
    assert not store.flag(identity, 'example')
    assert not store.path.exists()
    store.update(identity, {'index': 1, 'example': True})
    second = ReviewStore(store.path, keys=store.keys, columns=store.columns,
                         boolean_columns=store.boolean_columns)
    second.update({'experiment': '002', 'unit': '007'}, {'index': 2})
    assert store.flag(identity, 'example')
    assert not store.flag({'experiment': '002', 'unit': '007'}, 'example')
    store.update(identity, {'index': 99, 'example': False})
    assert len(store.read()) == 2  # index changes do not create a new identity
    store.update(identity, None)
    assert len(second.read()) == 1
    assert not store.flag(identity, 'example')
    with pytest.raises(ValueError, match='exactly'):
        store.update({'unit': '007'}, {})
    assert not list(tmp_path.glob('*.tmp'))


def test_review_store_legacy_boolean_and_invalid_value(tmp_path):
    path = tmp_path / 'review.csv'
    pd.DataFrame([{'unit': '007'}]).to_csv(path, index=False)
    store = ReviewStore(path, keys=('unit',), columns=('unit', 'example'),
                        boolean_columns=('example',))
    assert not store.flag({'unit': '007'}, 'example')
    pd.DataFrame([{'unit': '007', 'example': 'maybe'}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='invalid boolean'):
        store.read()


def test_generic_browser_navigation_persistence_and_errors(tmp_path, monkeypatch):
    # A different protocol's units and conditions: no VMN-shaped records.
    png = base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII=')
    image = tmp_path / 'response.png'
    image.write_bytes(png)
    reads = []
    original_read = type(image).read_bytes
    def read_once(path):
        reads.append(path)
        return original_read(path)
    monkeypatch.setattr(type(image), 'read_bytes', read_once)
    kept = ReviewStore(tmp_path / 'keep.csv', keys=('unit', 'condition'),
                       columns=('unit', 'condition'))
    examples = ReviewStore(tmp_path / 'example.csv', keys=('unit',),
                           columns=('unit', 'example'), boolean_columns=('example',))
    records = {1: {'unit': '007', 'conditions': ['center', 'annulus']},
               2: {'unit': '008', 'conditions': []}}
    loads = []

    def flags(item, condition):
        identity = dict(unit=item['unit'], condition=condition)
        frame = kept.read()
        return (bool(kept.matches(frame, identity).any()),
                examples.flag({'unit': item['unit']}, 'example'))

    def make():
        return saved_figure_review_browser(
            [('first', 1), ('second', 2)],
            load_item=lambda key: loads.append(key) or records[key],
            sections=lambda item: item['conditions'], panels=('Response',),
            figure_options=lambda item, panel, condition: [
                ('saved response', image), ('missing', tmp_path / 'missing.png')],
            describe=lambda item, section: '<unit> ' + item['unit'] + ' ' + section,
            review_flags=flags,
            set_keep=lambda item, condition, value: kept.update(
                dict(unit=item['unit'], condition=condition), {} if value else None),
            set_example=lambda item, value: examples.update(
                {'unit': item['unit']}, {'example': value}),
            item_description='Unit:', section_description='Condition:')

    browser = make()
    state = browser.review_state
    assert loads == [1]  # unselected records are not loaded
    assert reads == [image]  # options/value observers must not duplicate PNG reads
    assert state['figure_images']['Response'].value == png
    assert len(state['figure_selectors']['Response'].options) == 1
    assert not state['is_example']
    state['keep_button'].click()
    state['example_button'].click()
    assert flags(records[1], 'center') == (True, True)
    assert '★ EXAMPLE' in state['status'].value
    state['section_selector'].value = 'annulus'
    assert flags(records[1], 'annulus') == (False, True)
    state['keep_button'].click()
    state['remove_button'].click()
    assert flags(records[1], 'center') == (True, True)
    state['selector'].value = 2
    assert state['keep_button'].disabled
    assert state['figure_images']['Response'].value == b''
    state['selector'].value = 1
    reopened = make().review_state
    assert reopened['is_example']
    reopened['example_button'].click()
    assert not examples.flag({'unit': '007'}, 'example')
    # The older browser reads the current flag instead of toggling stale state.
    state['example_button'].click()
    assert examples.flag({'unit': '007'}, 'example')
    original = kept.path.read_bytes()
    def fail(*args, **kwargs):
        raise OSError('disk unavailable')
    kept.update = fail
    state['remove_button'].click()
    assert 'disk unavailable' in state['status'].value
    assert kept.path.read_bytes() == original
