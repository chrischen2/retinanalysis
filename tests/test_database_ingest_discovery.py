"""Regression tests for mounted H5/JSON experiment discovery."""

from pathlib import Path

from retinanalysis.utils import database_pop


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
