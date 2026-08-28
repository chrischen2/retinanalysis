"""Tests for retinanalysis.SCutils.explore.

Only the pure display helpers are covered here: they need no database, and
importing the module must not pull DataJoint (the query functions import it
lazily). The DataJoint-backed functions are exercised by using the notebook.
"""
import re
import sys

import pandas as pd
import pytest

from retinanalysis.SCutils import explore as sc


def test_import_does_not_pull_datajoint():
    """explore is importable without a DB: schema is imported per-call."""
    for mod in ('retinanalysis.SCutils.explore', 'datajoint'):
        sys.modules.pop(mod, None)
    import retinanalysis.SCutils.explore  # noqa: F401
    assert 'datajoint' not in sys.modules


@pytest.mark.parametrize('ids, expected', [
    ([], ''),
    ([7], '7'),
    ([5, 6, 7], '5-7'),
    ([5911, 5912, 5913, 5936, 5937], '5911-5913, 5936-5937'),
    ([34286, 34281, 34285, 34284], '34281, 34284-34286'),  # unsorted input
    ([1, 3, 5], '1, 3, 5'),
])
def test_compact_ids(ids, expected):
    assert sc.compact_ids(ids) == expected


def _tbody_rows(html):
    body = html.split('<tbody>')[1]
    return re.findall(r'<tr([^>]*)>(.*?)</tr>', body, re.S)


def _cells(inner):
    return [re.sub('<[^>]+>', '', c).strip()
            for c in re.findall(r'<td[^>]*>(.*?)</td>', inner, re.S)]


@pytest.fixture
def tree_df():
    return pd.DataFrame({
        'cell': ['Cell1', 'Cell1', 'Cell2', 'Cell2'],
        'recording': ['cell-attached', 'cell-attached', 'cell-attached', 'whole-cell'],
        'protocol': ['ExpandingSpots', 'SingleSpot', 'ExpandingSpots', 'SingleSpot'],
        'blocks': [2, 1, 3, 4],
    })


def test_tree_table_blanks_repeated_parents(tree_df):
    html = sc.tree_table(tree_df, levels=['cell', 'recording', 'protocol'],
                         show=False, num_cols=('blocks',))
    rows = [_cells(inner) for _, inner in _tbody_rows(html)]
    assert rows[0][:3] == ['Cell1', 'cell-attached', 'ExpandingSpots']
    # second row repeats cell + recording, so both are blanked
    assert rows[1][:3] == ['', '', 'SingleSpot']
    # new cell re-prints both levels
    assert rows[2][:3] == ['Cell2', 'cell-attached', 'ExpandingSpots']
    # same cell, different recording: only the cell is blanked
    assert rows[3][:3] == ['', 'whole-cell', 'SingleSpot']


def test_tree_table_marks_group_starts(tree_df):
    html = sc.tree_table(tree_df, levels=['cell', 'recording', 'protocol'], show=False)
    attrs = [a for a, _ in _tbody_rows(html)]
    # one separator per new top-level value after the first
    assert [i for i, a in enumerate(attrs) if 'grp' in a] == [2]


def test_tree_table_does_not_mutate_input(tree_df):
    before = tree_df.copy()
    sc.tree_table(tree_df, levels=['cell', 'recording'], show=False)
    pd.testing.assert_frame_equal(tree_df, before)


def test_scroll_table_summary_collapses(tree_df):
    plain = sc.scroll_table(tree_df, show=False)
    assert '<details>' not in plain
    collapsed = sc.scroll_table(tree_df, summary='all 4 blocks', show=False)
    assert '<details>' in collapsed and 'all 4 blocks' in collapsed


def test_scroll_table_escapes_html():
    df = pd.DataFrame({'protocol': ['<script>alert(1)</script>']})
    html = sc.scroll_table(df, show=False)
    assert '<script>' not in html
    assert '&lt;script&gt;' in html


def test_scroll_table_renders_consolidated_list_values():
    df = pd.DataFrame({'filters': [[0.0, 0.5, 1.0]],
                       'images': [['00152', '01192']]})
    html = sc.scroll_table(df, show=False)
    assert '[0.0, 0.5, 1.0]' in html
    assert '[&#x27;00152&#x27;, &#x27;01192&#x27;]' in html


@pytest.mark.parametrize('ndfs, wheel, recorded, expected', [
    ('["EL3"]', 0.0, True, 'EL3 + FW0'),
    ('["EL06", "EL2", "FW1"]', 3.0, True, 'EL06 + EL2 + FW3'),
    ('[]', 2.0, True, 'FW2'),
    ('[]', float('nan'), True, 'none'),
    ('', float('nan'), False, 'not recorded'),
])
def test_format_ndf_fw_uses_numeric_wheel_not_embedded_token(
        ndfs, wheel, recorded, expected):
    assert sc._format_ndf_fw(ndfs, wheel, recorded) == expected


def test_block_light_filters_delegates_to_shared_loader(monkeypatch):
    import retinanalysis as ra

    calls = []

    def trusted(blocks, verbose=False, **kwargs):
        calls.append(blocks.copy())
        return pd.DataFrame({
            'block_id': [10], 'fixed_ndfs': [('EL3',)],
            'filter_wheel_ndf': [0.5], 'ndf_combination': ['EL3 + FW0.5'],
            'filter_wheel_status': ['recorded'], 'fixed_ndf_source': ['stage'],
        })

    monkeypatch.setattr(ra, 'read_block_light_settings', trusted)
    result = sc._block_light_filters(pd.DataFrame({
        'exp_name': ['2026-08-01_E'], 'block_id': [10],
    }))

    assert len(calls) == 1
    assert result.loc[0, 'ndfs'] == 'EL3'
    assert result.loc[0, 'filter_wheel_ndf'] == pytest.approx(0.5)
    assert result.loc[0, 'ndf_fw'] == 'EL3 + FW0.5'


def test_num_cols_right_aligned(tree_df):
    html = sc.tree_table(tree_df, levels=['cell'], show=False, num_cols=('blocks',))
    assert html.count('class="num"') == len(tree_df)


def test_protocol_tree_rejects_mea_summary():
    """An MEA summary has no cell_label/recording_technique -> clear error."""
    mea = pd.DataFrame({'datafile_name': ['data001'], 'protocol_name': ['a.protocols.B'],
                        'block_id': [1], 'duration_minutes': [1.0], 'group_label': ['x']})
    with pytest.raises(ValueError, match='single-cell'):
        sc.protocol_tree(mea, show=False)


def test_protocol_tree_groups_cell_epoch_group_protocol_without_block_ids():
    blocks = pd.DataFrame({
        'cell_label': ['Cell1', 'Cell1', 'Cell1'],
        'cell_type': ['RGC', 'RGC', 'RGC'],
        'recording_technique': ['cell-attached'] * 3,
        'group_label': ['Control', 'Control', 'Drug'],
        'protocol_name': ['x.protocols.Spot', 'x.protocols.Spot', 'x.protocols.Spot'],
        'block_id': [11, 12, 13],
        'epochs': [5, 7, 3],
    })
    tree = sc.protocol_tree(blocks, show=False)
    assert list(tree.columns) == ['cell', 'epoch_group', 'protocol', 'blocks', 'epochs']
    assert tree.iloc[0].to_dict() == {
        'cell': 'Cell1  (RGC)', 'epoch_group': 'Control', 'protocol': 'Spot',
        'blocks': 2, 'epochs': 12,
    }
    assert 'block_ids' not in tree.columns


@pytest.mark.parametrize('path, expected', [
    ('/Volumes/SSD/single_cell/chris_data/h5/2026-01-01_G.h5', 'chris_data'),
    ('/Volumes/SSD/single_cell/fred_data/json/2026-01-01_E.json', 'fred_data'),
    ('/tmp/unclassified.h5', 'other_data'),
])
def test_data_owner_from_source_path(path, expected):
    assert sc._data_owner(path) == expected


def test_project_label_prefers_normalized_then_json_metadata():
    assert sc._project_label({'project': 'Retina atlas', 'properties': {}}) == 'Retina atlas'
    assert sc._project_label({'project': None,
                              'properties': {'projectLabel': 'Night vision'}}) == 'Night vision'


@pytest.mark.parametrize('label, expected', [
    (r'RGC\ON-parasol', 'ON-parasol'),
    ('RGC/OFF-parasol/', 'OFF-parasol'),
    ('horizontal', 'horizontal'),
])
def test_cell_type_short_uses_final_path_component(label, expected):
    assert sc._cell_type_short(label) == expected


@pytest.mark.parametrize('name, expected', [
    ('edu.washington.riekelab.turner.protocols.ExpandingSpots', 'ExpandingSpots'),
    ('manookinlab.protocols.JitteredNoise', 'JitteredNoise'),
])
def test_protocol_short_removes_package(name, expected):
    assert sc._protocol_short(name) == expected


def test_list_experiments_keeps_one_row_per_protocol(monkeypatch):
    catalog = pd.DataFrame({
        'data_owner': ['chris_data', 'chris_data'],
        'species': ['Primate', 'Primate'],
        'exp_name': ['2026-08-10_G', '2026-08-10_G'],
        'project': ['?', '?'],
        'cell_types': ['ON-parasol', 'ON-parasol'],
        'protocols': ['LedPulse', 'VariableMeanNoise'],
    })
    monkeypatch.setattr(sc, '_experiment_catalog', lambda: catalog)
    result = sc.list_experiments(show=False)
    assert list(result.columns) == ['exp_name', 'project', 'cell_types', 'protocol']
    assert result['protocol'].tolist() == ['LedPulse', 'VariableMeanNoise']


def test_protocol_inventory_counts_unique_dates_by_species(monkeypatch):
    catalog = pd.DataFrame({
        'data_owner': ['chris_data'] * 5,
        'species': ['Primate', 'Primate', 'Mouse', 'Mouse', 'Mouse'],
        'exp_name': ['p1', 'p1', 'm1', 'm2', 'm2'],
        'project': ['?'] * 5,
        'cell_types': ['RGC'] * 5,
        'protocols': ['Spot', 'Spot', 'Spot', 'Spot', 'Noise'],
    })
    monkeypatch.setattr(sc, '_experiment_catalog', lambda: catalog)
    result = sc.protocol_inventory(show=False)
    spot = result.set_index('protocol').loc['Spot']
    assert spot.to_dict() == {
        'primate_dates': 1, 'mouse_dates': 2, 'total_dates': 3,
    }


def test_experiment_browser_lists_each_date_once(monkeypatch):
    pytest.importorskip('ipywidgets')
    catalog = pd.DataFrame({
        'data_owner': ['chris_data'] * 3,
        'species': ['Primate'] * 3,
        'exp_name': ['2026-08-04_G', '2026-08-04_G', '2026-08-10_G'],
        'project': ['?'] * 3,
        'cell_types': ['RGC'] * 3,
        'protocols': ['Spot', 'Noise', 'Pulse'],
    })
    summary = pd.DataFrame({
        'cell_label': ['Cell1'], 'cell_type': ['RGC'],
        'group_label': ['Control'], 'protocol': ['Spot'],
        'block_id': [1], 'epochs': [2],
    })
    monkeypatch.setattr(sc, '_experiment_catalog', lambda: catalog)
    monkeypatch.setattr(sc, 'summarize_experiment',
                        lambda *args, **kwargs: summary.copy())
    monkeypatch.setattr('IPython.display.display', lambda *args, **kwargs: None)

    browser = sc.summarize_experiments(
        catalog[['exp_name']].drop_duplicates())
    values = [value for _, value in browser.selectors['experiment'].options]
    assert values == ['2026-08-04_G', '2026-08-10_G']
