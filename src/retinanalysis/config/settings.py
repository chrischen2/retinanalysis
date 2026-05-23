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

# ---------------------------------------------------------------------------
# Tier kind classification (local vs network) — auto-detected from the
# mount fstype of each tier's `data` path. The user shouldn't have to
# encode "this tier is the slow NAS" in config.ini labels; we just look
# at what kind of filesystem the data root is on.
#
# Reads ALWAYS prefer local tiers over network tiers, regardless of how
# the user labeled them ([DEFAULT] / [SECONDARY] / [TERTIARY]). Bandwidth
# conservation: when both a local SSD and the NAS hold the same file,
# always pay the local read.
# ---------------------------------------------------------------------------
_NETWORK_FSTYPES = frozenset({
    'smbfs', 'cifs', 'nfs', 'nfs4', 'afpfs', 'webdav',
    'fuse.sshfs', 'sshfs', 'ftpfs',
})


def _classify_tier(path: str) -> str:
    """Return 'local' / 'network' / 'unknown' for the mount holding ``path``."""
    if not path:
        return 'unknown'
    try:
        import psutil
        ap = os.path.abspath(path)
        best = None
        for part in psutil.disk_partitions(all=True):
            mp = part.mountpoint
            if ap == mp or ap.startswith(mp.rstrip('/') + '/'):
                if best is None or len(mp) > len(best.mountpoint):
                    best = part
        if best is not None:
            return ('network'
                    if (best.fstype or '').lower() in _NETWORK_FSTYPES
                    else 'local')
    except Exception:
        pass
    return 'unknown'


_TIER_KIND = {}
for _tier in ('primary', 'secondary', 'tertiary'):
    if _tier in mea_config:
        _TIER_KIND[_tier] = _classify_tier(
            mea_config[_tier].get('data', ''))

# Compose priority: local cache → local source tiers → unknown → network.
# Within each group, config order (primary → secondary → tertiary) is
# preserved so users still have control over relative ordering of
# same-kind tiers. The labels (DEFAULT/SECONDARY/TERTIARY) become a
# *secondary* sort key behind local-vs-network.
_local_first = [t for t in ('primary', 'secondary', 'tertiary')
                if _TIER_KIND.get(t) == 'local']
_unknown_tiers = [t for t in ('primary', 'secondary', 'tertiary')
                   if _TIER_KIND.get(t) == 'unknown']
_network_tiers = [t for t in ('primary', 'secondary', 'tertiary')
                   if _TIER_KIND.get(t) == 'network']
_TIER_PRIORITY = (['local_cache'] + _local_first
                   + _unknown_tiers + _network_tiers)


# ---------------------------------------------------------------------------
# Network bandwidth gauge: tally bytes whose canonical resolution is on a
# network mount. We instrument `find_path` (below): every time it returns
# a file/dir on a network tier, we bump a counter by the size of the
# resolved path. This is an *upper bound* on bytes actually transferred —
# the kernel may cache reads, and the caller may only read part of a
# resolved file — but it's a useful conservative estimate of "how much
# would have flowed over the wire without any caching."
#
# Cache hits at the local_cache tier (i.e. after mirror_to_local_cache)
# never touch this counter. So the gauge naturally drops to zero once
# all your reads are served locally.
# ---------------------------------------------------------------------------
_NETWORK_BYTES_RESOLVED = 0
_NETWORK_RESOLUTIONS = 0


def _record_network_resolution(path: str) -> None:
    """Increment the network gauge by the size of ``path`` (file or dir)."""
    global _NETWORK_BYTES_RESOLVED, _NETWORK_RESOLUTIONS
    try:
        if os.path.isfile(path):
            _NETWORK_BYTES_RESOLVED += os.path.getsize(path)
        elif os.path.isdir(path):
            # Walk one level deep first (typical vision-data dir layout
            # is flat). Avoid scanning huge nested trees on slow mounts.
            total = 0
            for entry in os.scandir(path):
                if entry.is_file(follow_symlinks=False):
                    try:
                        total += entry.stat().st_size
                    except OSError:
                        pass
            _NETWORK_BYTES_RESOLVED += total
        _NETWORK_RESOLUTIONS += 1
    except OSError:
        pass


def network_bytes_resolved() -> int:
    """Cumulative bytes of network-tier resolutions this session.

    Upper bound on actual wire traffic — see module comment above.
    """
    return int(_NETWORK_BYTES_RESOLVED)


def network_resolutions_count() -> int:
    """How many times ``find_path`` resolved to a network tier this session."""
    return int(_NETWORK_RESOLUTIONS)


def reset_network_gauge() -> None:
    """Zero the network-bytes gauge (useful when starting a fresh analysis)."""
    global _NETWORK_BYTES_RESOLVED, _NETWORK_RESOLUTIONS
    _NETWORK_BYTES_RESOLVED = 0
    _NETWORK_RESOLUTIONS = 0

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


# Raw .bin files are the exception to "prefer local". They are ~30 GB per
# experiment and are kept canonical on the NAS — local SSDs only mirror a
# few recent dates. Resolving raw against the network tier first means a
# connected NAS always wins, even if a partial SSD copy happens to be
# present. Falls back to local tiers when NAS is not mounted so the few
# dates that *are* on SSD still work.
_RAW_TIER_PRIORITY = _network_tiers + _unknown_tiers + _local_first


def _pick_raw_tier():
    for tier in _RAW_TIER_PRIORITY:
        if tier in mea_config and mea_config[tier].get('raw'):
            return mea_config[tier]
    return None


_top = _pick_top_tier()
DATA_DIR = _top['data']
ANALYSIS_DIR = _top['analysis']
H5_DIR = _top['h5']
META_DIR = _top['meta']
TAGS_DIR = _top['tags']
QUERY_DIR = _top['query']
USER = _top['user']

_raw_top = _pick_raw_tier()
RAW_DIR = _raw_top['raw'] if _raw_top is not None else ''

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


def find_raw_path(exp_name, datafile_name):
    """Resolve a raw .bin folder by walking tiers network-first.

    Raw .bin files live canonical on the NAS (~30 GB per experiment); local
    SSDs only mirror a few recent dates. So unlike :func:`find_path` — which
    prefers local tiers to conserve NAS bandwidth — this resolver tries the
    network tier first and only falls back to local copies when the NAS
    isn't mounted.

    Returns the absolute path to ``<raw_root>/<exp>/<datafile>`` if it
    exists on any configured tier, else ``None`` so the caller can raise
    a clear error rather than fail deep inside ``bin2py``.
    """
    for tier in _RAW_TIER_PRIORITY:
        if tier not in mea_config:
            continue
        root = mea_config[tier].get('raw', '')
        if not root:
            continue
        candidate = os.path.join(root, exp_name, datafile_name)
        if os.path.exists(candidate):
            if _TIER_KIND.get(tier) == 'network':
                _record_network_resolution(candidate)
            return candidate
    return None


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
            # Charge the network gauge when this lookup resolves to a
            # network-mount tier. Local cache + local source tiers are
            # free. See module-level _record_network_resolution doc.
            if _TIER_KIND.get(tier) == 'network':
                _record_network_resolution(candidate)
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