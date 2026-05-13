import datajoint as dj
from retinanalysis.utils import (USER,
                                 H5_DIR,
                                 META_DIR,
                                 TAGS_DIR,
                                 database_pop)
import retinanalysis.config.schema as schema
from typing import List

    
def populate_database(username = USER, h5_dir = H5_DIR, 
                        meta_dir = META_DIR, tags_dir = TAGS_DIR):
    
    db = dj.VirtualModule('schema.py', 'schema', create_schema=True)

    database_pop.append_data(h5_dir, meta_dir, tags_dir, username, db)

def reload_experiment_data(exp_name, username = USER, h5_dir = H5_DIR, 
                    meta_dir = META_DIR, tags_dir = TAGS_DIR):
    
    (schema.Experiment() & {'exp_name' : exp_name}).delete(prompt=False)

    populate_database(username, h5_dir, meta_dir, tags_dir)

def delete_experiments(exp_names: List[str]):

    for exp in exp_names:
        (schema.Experiment() & {'exp_name' : exp}).delete(prompt=False)

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

