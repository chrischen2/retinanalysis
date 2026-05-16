from configparser import ConfigParser
import platform
import os
# import importlib.resources as ir
try:
    import importlib.resources as ir
except:
    import importlib_resources as ir # type: ignore

import retinanalysis


def load_config(config_path):
    if os.path.exists(config_path):
        config_path = os.path.abspath(config_path)
        configfile = ConfigParser()
        configfile.read(config_path)
    else:
        template = os.path.join(os.path.dirname(str(config_path)),
                                'config.ini.example')
        raise FileNotFoundError(
            f"No config file found at {config_path}.\n"
            f"Copy the template at {template} to that path "
            f"and edit the absolute paths in each tier for your machine."
        )

    if platform.system() == 'Darwin':
        DEFAULT_config = configfile['DEFAULT']
        SECONDARY_config = configfile['SECONDARY']
        TERTIARY_config = configfile['TERTIARY']
    elif platform.system() == 'Linux':
        DEFAULT_config = configfile['LINUX_DEFAULT']
        SECONDARY_config = configfile['LINUX_SECONDARY']
    else:
        DEFAULT_config = configfile['WINDOWS_DEFAULT']
        SECONDARY_config = configfile['WINDOWS_SECONDARY']
    
    def _row(cfg):
        # protocol_repos_root / output / local_cache are optional — fall
        # back to '' if absent. Keep the existing 8 read-keys required
        # for backward compatibility with older config.ini files.
        return {'data' : cfg['data'], 'raw': cfg['raw'], 'analysis': cfg['analysis'],
                'h5': cfg['h5'], 'meta': cfg['meta'], 'tags': cfg['tags'],
                'query': cfg['query'], 'user': cfg['user'],
                'protocol_repos_root': cfg.get('protocol_repos_root', ''),
                'output': cfg.get('output', ''),
                'local_cache': cfg.get('local_cache', '')}

    mea_config = dict()
    if os.path.exists(os.path.abspath(DEFAULT_config['data'])):
        mea_config['primary'] = _row(DEFAULT_config)
    if os.path.exists(os.path.abspath(SECONDARY_config['data'])):
        mea_config['secondary'] = _row(SECONDARY_config)

    if platform.system() == 'Darwin' and os.path.exists(os.path.abspath(TERTIARY_config['data'])):
        mea_config['tertiary'] = _row(TERTIARY_config)

    if not mea_config:
        mea_config['primary'] = {'data': '', 'raw': '', 'analysis': '', 'h5': '', 'meta': '',
                                 'tags': '', 'query': '', 'user': '', 'protocol_repos_root': ''}
        print("No NAS or SSD paths found, check that one of them is connected")

    return mea_config


def create_config(config_path, config_name,
                      data_dir, analysis_dir, 
                      h5_dir, meta_dir, tags_dir, username = 'drezeanu'):

    config = ConfigParser()
    # TODO update
    config[config_name] = {'Analysis': analysis_dir,
                         'Data': data_dir,
                         'H5': h5_dir,
                         'Meta': meta_dir,
                         'Tags': tags_dir,
                         'User': username}

    if os.path.exists(config_path):
        with open(config_path, "a") as configfile:
            config.write(configfile)

    else:
        with open(config_path, "w") as configfile:
            config.write(configfile)

def reset_config(config_path):
    os.remove(config_path)
    print("Config file successfully deleted")


config_path = ir.files(retinanalysis) / os.path.join("config", "config.ini")
mea_config = load_config(config_path)

# ---------------------------------------------------------------------------
# Local file cache (highest-priority tier)
#
# When the user is working off a remote NAS mount, the per-experiment Vision
# files (.ei, .neurons, .params, .classification.txt, …) get pulled over the
# wire every time the pipeline is rebuilt — hundreds of MB per kernel restart.
# `ra.mirror_to_local_cache(...)` copies those files into this directory once,
# and `find_path()` then transparently returns the local copy.
#
# Resolution order for the cache root:
#   1. RA_LOCAL_CACHE_ROOT environment variable (per-shell override).
#   2. First non-empty `local_cache` field across configured tiers.
#   3. `~/.cache/retinanalysis` — portable default for anyone using the
#      package out-of-the-box with no config edits.
# ---------------------------------------------------------------------------
def _first_nonempty_tier_value(key: str, _config: dict) -> str:
    """Return the first non-empty value of `key` across tiers, or ''."""
    for tier in ('primary', 'secondary', 'tertiary'):
        if tier in _config and _config[tier].get(key):
            return _config[tier][key]
    return ''


LOCAL_CACHE_ROOT = (
    os.environ.get('RA_LOCAL_CACHE_ROOT', '')
    or _first_nonempty_tier_value('local_cache', mea_config)
    or os.path.join(os.path.expanduser('~'), '.cache', 'retinanalysis')
)
mea_config['local_cache'] = {
    'data':     os.path.join(LOCAL_CACHE_ROOT, 'data'),
    'raw':      os.path.join(LOCAL_CACHE_ROOT, 'raw'),
    'analysis': os.path.join(LOCAL_CACHE_ROOT, 'analysis'),
    # Other kinds are not cached locally — leave blank so find_path()
    # falls through to the real tiers for h5/meta/tags/query.
    'h5': '', 'meta': '', 'tags': '', 'query': '', 'user': '',
    'protocol_repos_root': '', 'output': '', 'local_cache': '',
}

# Read priority: local file cache first (free, when populated), then the
# lab server tier ("tertiary"), then progressively more local SSD tiers.
# config.ini owns the actual paths; this list controls their priority.
_TIER_PRIORITY = ['local_cache', 'tertiary', 'secondary', 'primary']

# Module-level constants point to the highest-priority *source* tier (not
# the local cache) whose root exists. Callers that do `os.listdir(DATA_DIR)`
# or similar need the canonical source — not the cache, which only holds
# per-file copies of what `mirror_to_local_cache()` was asked to populate.
# `find_path()` still walks the cache tier first for per-file lookups.
def _pick_top_tier():
    for tier in _TIER_PRIORITY:
        if tier == 'local_cache':
            continue
        if tier in mea_config:
            return mea_config[tier]
    raise RuntimeError("No valid config paths found.")

_top = _pick_top_tier()
DATA_DIR = _top['data']
RAW_DIR = _top['raw']
ANALYSIS_DIR = _top['analysis']
H5_DIR = _top['h5']
META_DIR = _top['meta']
TAGS_DIR = _top['tags']
QUERY_DIR = _top['query']
USER = _top['user']

# Root directory holding locally-cloned protocol packages (turner-package, manookin-package,
# riekelab-package-master, ...). Used by retinanalysis.regen to find resource files (.iml,
# .mat libraries) and protocol source code. Empty string if not configured / not present.
PROTOCOL_REPOS_ROOT = next(
    (mea_config[tier]['protocol_repos_root']
     for tier in _TIER_PRIORITY
     if tier in mea_config
     and mea_config[tier].get('protocol_repos_root')
     and os.path.isdir(mea_config[tier]['protocol_repos_root'])),
    '',
)


def find_protocol_repo(repo_name):
    """Return absolute path to a cloned protocol repo, or None if not found.

    Looks under PROTOCOL_REPOS_ROOT for a directory matching ``repo_name``. Common
    Riekelab/Manookin packages live under conventional names: ``turner-package``,
    ``manookin-package``, ``riekelab-package-master``, ``chris-package``, etc.
    """
    if not PROTOCOL_REPOS_ROOT:
        return None
    candidate = os.path.join(PROTOCOL_REPOS_ROOT, repo_name)
    return candidate if os.path.isdir(candidate) else None


def find_path(kind, *parts):
    """Resolve a path by searching configured tiers in priority order.

    Useful when a specific experiment may only exist on one volume (e.g. the
    server has the analysis chunk but the spike-sort output for a datafile is
    only on a local SSD). Returns the first existing path; if none exist,
    returns the top-tier path so the caller still has a sensible default to
    show in an error message.

    Example: find_path('data', '20221006C', 'data029', 'kilosort2')
    """
    if kind not in ('data', 'raw', 'analysis', 'h5', 'meta', 'tags', 'query'):
        raise ValueError(f"Unknown path kind: {kind}")
    fallback = None
    for tier in _TIER_PRIORITY:
        if tier not in mea_config:
            continue
        root = mea_config[tier].get(kind, '')
        # Skip tiers that don't own this kind (e.g. the local-cache tier
        # leaves h5/meta/tags blank). Without this guard the cache tier
        # would generate spurious relative-path fallbacks on first miss.
        if not root:
            continue
        candidate = os.path.join(root, *parts)
        if fallback is None:
            fallback = candidate
        if os.path.exists(candidate):
            return candidate
    return fallback


# Write target for ad-hoc outputs (figures, processed dataframes, derived
# classification files, etc.). Resolution order:
#   1. RA_OUTPUT_DIR environment variable (per-shell override).
#   2. First non-empty `output` field across configured tiers — the
#      tier walk prefers the most-local writeable tier (primary →
#      secondary → tertiary), which avoids writing onto the read-only
#      lab server unless the user explicitly set it there.
#   3. `~/retinanalysis_output` — portable default.
OUTPUT_DIR = (
    os.environ.get('RA_OUTPUT_DIR', '')
    or _first_nonempty_tier_value('output', mea_config)
    or os.path.join(os.path.expanduser('~'), 'retinanalysis_output')
)
os.makedirs(OUTPUT_DIR, exist_ok=True)