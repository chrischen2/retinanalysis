import os

import datajoint as dj
import pandas as pd
from retinanalysis.utils import (USER,
                                 H5_DIR,
                                 META_DIR,
                                 TAGS_DIR,
                                 database_pop)
import retinanalysis.config.schema as schema
from typing import List, Union


def populate_database(username = USER, h5_dir = H5_DIR,
                        meta_dir = META_DIR, tags_dir = TAGS_DIR,
                        update_if_modified: bool = True,
                        watch_data_file: bool = False):
    """Ingest every experiment under ``h5_dir`` / ``meta_dir`` into DataJoint.

    Experiments already in the database are skipped, *except* when their
    source files are newer than the stored ``date_added``. With
    ``update_if_modified=True`` (the default), a meta .json or tags .json that
    has been touched since ingest — a fresher copy synced down from the shared
    drive, say — causes the existing experiment to be dropped (cascading to
    all downstream tables) and re-ingested from the new file.

    Only the .json files are watched: nothing in the database is read from the
    .h5, so re-copying a raw file bumps its mtime without changing anything
    ingestible. Pass ``watch_data_file=True`` to trigger on the .h5 too, or
    ``update_if_modified=False`` for strictly append-only behaviour.

    Returns a dict with ``n_ingested``, ``added``, ``updated`` and ``skipped``.
    """
    db = dj.VirtualModule('schema.py', 'schema', create_schema=True)

    return database_pop.append_data(h5_dir, meta_dir, tags_dir, username, db,
                                    update_if_modified=update_if_modified,
                                    watch_data_file=watch_data_file)

def reload_experiment_data(exp_name, username = USER, h5_dir = H5_DIR,
                    meta_dir = META_DIR, tags_dir = TAGS_DIR):

    (schema.Experiment() & {'exp_name' : exp_name}).delete(prompt=False)

    populate_database(username, h5_dir, meta_dir, tags_dir)

def delete_experiments(exp_names: Union[str, List[str]]):
    """Drop the named experiments, cascading to every downstream table.

    Accepts a single ``exp_name`` or a list. Names not present in the database
    are reported and ignored. Returns the number of experiments dropped.
    """
    if isinstance(exp_names, str):
        exp_names = [exp_names]

    n_dropped = 0
    for exp in exp_names:
        q = schema.Experiment() & {'exp_name': exp}
        if len(q) == 0:
            print(f"Not in database, nothing to delete: {exp}")
            continue
        q.delete(prompt=False)
        print(f"Deleted: {exp}")
        n_dropped += 1
    print(f"Dropped {n_dropped} experiment(s); "
          f"{len(schema.Experiment())} remain in the database.")
    return n_dropped


def list_database_experiments(is_mea: bool = None,
                              watch_data_file: bool = False) -> pd.DataFrame:
    """One row per experiment in the database, with source-file freshness.

    Columns: ``exp_name``, ``is_mea``, ``date_added``, ``meta_file``,
    ``source_mtime`` (latest mtime across the watched source files),
    ``source_file`` (which of them is newest) and ``is_stale`` (True when the
    source is newer than ``date_added``, i.e. the next ``populate_database()``
    would re-ingest it).

    Watches the same files ``populate_database`` does: the meta .json and tags
    .json, plus the .h5 only when ``watch_data_file=True``. Freshness is
    resolved against the paths stored at ingest time, so a row whose files
    have since moved shows ``source_mtime = None``.

    Pass ``is_mea=False`` for just the single-cell (patch) experiments.
    """
    q = schema.Experiment()
    if is_mea is not None:
        q = q & {'is_mea': 1 if is_mea else 0}
    if len(q) == 0:
        return pd.DataFrame(columns=['exp_name', 'is_mea', 'date_added',
                                     'meta_file', 'source_mtime',
                                     'source_file', 'is_stale'])

    df = q.fetch(format='frame').reset_index()
    df = df[['exp_name', 'is_mea', 'date_added', 'meta_file', 'tags_file',
             'data_file']].copy()

    rows = []
    for _, r in df.iterrows():
        watched = [r['meta_file'], r['tags_file']]
        if watch_data_file:
            watched.append(r['data_file'])
        newest, newest_path = database_pop.newest_source_mtime(*watched)
        date_added = database_pop._as_datetime(r['date_added'])
        rows.append({
            'source_mtime': newest,
            'source_file': os.path.basename(newest_path) if newest_path else None,
            'is_stale': bool(newest is not None and date_added is not None
                             and newest > date_added),
        })
    df = pd.concat([df.drop(columns=['tags_file', 'data_file']),
                    pd.DataFrame(rows, index=df.index)], axis=1)
    return df.sort_values('exp_name').reset_index(drop=True)


def purge_experiments(exp_names: Union[str, List[str]]):
    """Alias for :func:`delete_experiments` — drop specific experiments.

    Named for symmetry with :func:`purge_database`, which wipes everything.
    Deleting one date is cheap to undo (re-run ``populate_database()``), so
    unlike ``purge_database`` this needs no confirmation sentinel.
    """
    return delete_experiments(exp_names)

PURGE_CONFIRM_TOKEN = 'YES_DELETE_ALL'


def purge_database(confirm: str = ''):
    """Delete EVERY experiment (cascades to all downstream tables).

    This is irreversible. To run it you must pass the literal sentinel
    ``confirm='YES_DELETE_ALL'`` — calling ``purge_database()`` with no
    argument raises instead of wiping the DB. The sentinel exists so an
    accidental call (auto-complete, copy/paste from a notebook,
    refactoring that touches this module) cannot silently delete the
    full table.
    """
    if confirm != PURGE_CONFIRM_TOKEN:
        raise PermissionError(
            "purge_database() refuses to run without explicit confirmation. "
            f"Pass confirm='{PURGE_CONFIRM_TOKEN}' to actually delete every "
            "experiment. This action is irreversible."
        )

    all_experiments = schema.Experiment()
    all_exp_names = all_experiments.fetch('exp_name')

    print(f'Purging {len(all_exp_names)} experiments from the database...')
    for exp in all_exp_names:
        (schema.Experiment() & {'exp_name' : exp}).delete(prompt=False)
    print(f'Purge complete: {len(all_exp_names)} experiments dropped.')

