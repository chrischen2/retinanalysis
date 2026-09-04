"""Regression tests for mounted H5/JSON experiment discovery."""

from pathlib import Path

import pytest

from retinanalysis.utils import database_pop


def test_complete_auisql_bundle_is_ingestible_fallback(tmp_path):
    h5_dir = tmp_path / 'h5'
    meta_dir = tmp_path / 'meta'
    tags_dir = tmp_path / 'tags'
    for directory in (h5_dir, meta_dir, tags_dir):
        directory.mkdir()
    stem = '2022-09-09_B'
    database = h5_dir / f'{stem}.auisql'
    response_h5 = h5_dir / f'{stem}.auisql.h5'
    database.touch()
    response_h5.touch()
    metadata = meta_dir / f'{stem}.json'
    metadata.write_text('{}')

    rows = database_pop.gen_meta_list(
        str(h5_dir), str(meta_dir), str(tags_dir))

    assert rows == [[str(metadata), str(response_h5),
                     str(tags_dir / f'{stem}.json')]]


def test_primary_h5_wins_over_complete_auisql_bundle(tmp_path):
    h5_dir = tmp_path / 'h5'
    meta_dir = tmp_path / 'meta'
    tags_dir = tmp_path / 'tags'
    for directory in (h5_dir, meta_dir, tags_dir):
        directory.mkdir()
    stem = '2022-09-09_B'
    primary_h5 = h5_dir / f'{stem}.h5'
    primary_h5.touch()
    (h5_dir / f'{stem}.auisql').touch()
    (h5_dir / f'{stem}.auisql.h5').touch()
    metadata = meta_dir / f'{stem}.json'
    metadata.write_text('{}')

    rows = database_pop.gen_meta_list(
        str(h5_dir), str(meta_dir), str(tags_dir))

    assert rows == [[str(metadata), str(primary_h5),
                     str(tags_dir / f'{stem}.json')]]


def test_mea_json_is_loaded_when_same_stem_h5_exists(tmp_path, monkeypatch):
    """A compact MEA H5 is auxiliary and must not hide its metadata JSON."""
    h5_dir = tmp_path / 'h5'
    meta_dir = tmp_path / 'meta'
    tags_dir = tmp_path / 'tags'
    sorted_dir = tmp_path / 'sorted'
    for directory in (h5_dir, meta_dir, tags_dir, sorted_dir):
        directory.mkdir()

    exp_name = '20260901C'
    (h5_dir / f'{exp_name}.h5').touch()
    meta_file = meta_dir / f'{exp_name}.json'
    meta_file.write_text('{}')
    data_file = sorted_dir / exp_name
    data_file.mkdir()

    monkeypatch.setattr(
        database_pop, 'find_path',
        lambda kind, name: str(sorted_dir / name) if kind == 'data' else None)

    rows = database_pop.gen_meta_list(
        str(h5_dir), str(meta_dir), str(tags_dir))

    assert rows == [[str(meta_file), exp_name,
                     str(tags_dir / f'{exp_name}.json')]]
    assert (tags_dir / f'{exp_name}.json').read_text() == '{}'


def test_mea_json_still_requires_sorted_data(tmp_path, monkeypatch, capsys):
    """The presence of an auxiliary H5 alone does not make MEA data usable."""
    h5_dir = tmp_path / 'h5'
    meta_dir = tmp_path / 'meta'
    tags_dir = tmp_path / 'tags'
    for directory in (h5_dir, meta_dir, tags_dir):
        directory.mkdir()

    exp_name = '20260901C'
    (h5_dir / f'{exp_name}.h5').touch()
    (meta_dir / f'{exp_name}.json').write_text('{}')
    monkeypatch.setattr(
        database_pop, 'find_path',
        lambda kind, name: str(tmp_path / 'missing' / name))

    rows = database_pop.gen_meta_list(
        str(h5_dir), str(meta_dir), str(tags_dir))

    assert rows == []
    assert exp_name in capsys.readouterr().out


@pytest.mark.parametrize('exp_name', ['20260804C', '20260728H'])
def test_missing_rig_type_defaults_to_mea_for_c_and_h(exp_name):
    assert database_pop.resolve_rig_type({}, exp_name) == 'MEA'


def test_explicit_rig_type_is_preserved():
    assert database_pop.resolve_rig_type(
        {'rig_type': 'PATCH'}, '20260804C') == 'PATCH'


def test_missing_rig_type_is_not_guessed_for_other_rigs():
    with pytest.raises(KeyError, match='limited.*rig C or H'):
        database_pop.resolve_rig_type({}, '20260804A')


def _source(label, uuid, properties=None, sources=None):
    return {
        'attributes': {'label': label, 'uuid': uuid},
        'label': label,
        'properties': properties or {},
        'notes': [],
        'sources': sources or [],
    }


def test_analysis_mea_json_is_expanded_for_database_ingest():
    cell = _source('20260804Cm1', 'cell-uuid', {'type': 'RGC'})
    preparation = _source(
        'Mount1', 'prep-uuid', {'preparation': 'wholemount'}, [cell])
    animal = _source(
        '20260804C', 'animal-uuid', {'species': 'mouse'}, [preparation])
    epoch = {
        'attributes': {'uuid': 'epoch-uuid'},
        'properties': {},
        'parameters': {},
        'responses': {},
        'stimuli': {},
    }
    block = {
        'uuid': 'block-uuid',
        'protocolID': 'example.Protocol',
        'attributes': {'uuid': 'block-uuid', 'protocolID': 'example.Protocol'},
        'properties': {},
        'parameters': {},
        'arrayPitch': '60um',
        'dataFile': '/data000/',
        'epoch': [epoch],
    }
    group = {
        'attributes': {'label': 'group', 'uuid': 'group-uuid'},
        'label': 'group',
        'properties': {},
        'source': {'uuid': 'cell-uuid'},
        'block': [block],
    }
    second_block = {
        **block,
        'uuid': 'second-block-uuid',
        'attributes': {
            'uuid': 'second-block-uuid',
            'protocolID': 'second.Protocol',
        },
        'protocolID': 'second.Protocol',
        'epoch': [],
    }
    second_group_piece = {**group, 'block': [second_block]}
    metadata = {
        'protocol': [
            {'label': 'example.Protocol', 'group': [group]},
            {'label': 'second.Protocol', 'group': [second_group_piece]},
        ],
        'sources': [animal],
    }

    normalized = database_pop.prepare_experiment_for_ingest(
        metadata, '20260804C')

    assert normalized['rig_type'] == 'MEA'
    assert normalized['animals'][0]['species'] == 'mouse'
    normalized_preparation = normalized['animals'][0]['preparations'][0]
    assert normalized_preparation['arrayPitch'] == '60um'
    normalized_cell = normalized_preparation['cells'][0]
    assert normalized_cell['type'] == 'RGC'
    assert len(normalized_cell['epoch_groups']) == 1
    normalized_blocks = normalized_cell['epoch_groups'][0]['epoch_blocks']
    assert len(normalized_blocks) == 2
    assert normalized_blocks[0]['epochs'] == [epoch]
    assert 'epoch' in metadata['protocol'][0]['group'][0]['block'][0]
