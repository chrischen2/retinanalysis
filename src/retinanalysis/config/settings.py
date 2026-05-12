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
        print()
        raise FileNotFoundError(f"No config file found at {config_path}.\nUse reset_config() and create_config() to make one.")

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
    
    mea_config = dict()
    if os.path.exists(os.path.abspath(DEFAULT_config['data'])):
        use_config = DEFAULT_config
        mea_config['primary'] = {'data' : use_config['data'], 'raw': use_config['raw'], 'analysis': use_config['analysis'],
                    'h5': use_config['h5'], 'meta': use_config['meta'], 'tags': use_config['tags'],
                    'query': use_config['query'], 'user': use_config['user']}
    if os.path.exists(os.path.abspath(SECONDARY_config['data'])):
        use_config = SECONDARY_config
        mea_config['secondary'] = {'data' : use_config['data'], 'raw': use_config['raw'], 'analysis': use_config['analysis'],
                    'h5': use_config['h5'], 'meta': use_config['meta'], 'tags': use_config['tags'],
                    'query': use_config['query'], 'user': use_config['user']}

    if os.path.exists(os.path.abspath(TERTIARY_config['data'])):
        use_config = TERTIARY_config
        mea_config['tertiary'] = {'data' : use_config['data'], 'raw': use_config['raw'], 'analysis': use_config['analysis'],
                    'h5': use_config['h5'], 'meta': use_config['meta'], 'tags': use_config['tags'],
                    'query': use_config['query'], 'user': use_config['user']}
        
    if not mea_config:
        use_config = {'data': '', 'raw': '', 'analysis': '', 'h5': '', 'meta': '', 'tags': '', 'query': '', 'user': ''}
        mea_config['primary'] = {'data' : use_config['data'], 'raw': use_config['raw'], 'analysis': use_config['analysis'],
                    'h5': use_config['h5'], 'meta': use_config['meta'], 'tags': use_config['tags'],
                    'query': use_config['query'], 'user': use_config['user']}
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

# Read priority: prefer the lab server (/Volumes/data, "tertiary") when connected,
# fall back to the local SSDs ("secondary" = ChrisProSSD, "primary" = ChrisNewSSD).
# The server is the canonical source of truth for shared data; local SSDs act as
# caches when the server is offline (or when an experiment hasn't been synced yet).
_TIER_PRIORITY = ['tertiary', 'secondary', 'primary']

# Module-level constants point to the highest-priority tier whose root path exists.
# This preserves backward compatibility for code that does `os.path.join(DATA_DIR, ...)`.
# For per-experiment resolution that automatically falls back across volumes when an
# experiment isn't on the server, use `find_path()` below.
def _pick_top_tier():
    for tier in _TIER_PRIORITY:
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
        candidate = os.path.join(mea_config[tier][kind], *parts)
        if fallback is None:
            fallback = candidate
        if os.path.exists(candidate):
            return candidate
    return fallback


# Write target for ad-hoc outputs (figures, processed dataframes, derived classification
# files, etc.). Prefer the fast local SSD when mounted; otherwise drop into ~/Downloads
# so writes never silently fail or land on a slow/shared volume.
_CHRIS_SSD = '/Volumes/ChrisProSSD'
if os.path.exists(_CHRIS_SSD):
    OUTPUT_DIR = os.path.join(_CHRIS_SSD, 'retinanalysis_output')
else:
    OUTPUT_DIR = os.path.join(os.path.expanduser('~'), 'Downloads', 'retinanalysis_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)