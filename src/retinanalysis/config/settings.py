from configparser import ConfigParser
import platform
import os
# import importlib.resources as ir
try:
    import importlib.resources as ir
except:
    import importlib_resources as ir # type: ignore

import retinanalysis


# Positional names for the tiers, in config-file order. Beyond this list
# tiers are keyed 'tier7', 'tier8', ... — there is no fixed slot count, so a
# new volume is added by dropping another section into config.ini.
_TIER_KEY_NAMES = ('primary', 'secondary', 'tertiary', 'quaternary',
                   'quinary', 'senary')


def _tier_key(idx):
    if idx < len(_TIER_KEY_NAMES):
        return _TIER_KEY_NAMES[idx]
    return f'tier{idx + 1}'


def _platform_sections(configfile):
    """Config section names for this platform, highest read-priority first.

    Order is the order the sections appear in config.ini. On Darwin every
    section that is not explicitly Linux/Windows-prefixed counts as a tier;
    on the other platforms only the matching prefix does.
    """
    system = platform.system()
    if system == 'Darwin':
        return ['DEFAULT'] + [s for s in configfile.sections()
                              if not s.startswith(('LINUX_', 'WINDOWS_'))]
    head = 'LINUX_DEFAULT' if system == 'Linux' else 'WINDOWS_DEFAULT'
    prefix = 'LINUX_' if system == 'Linux' else 'WINDOWS_'
    return [head] + [s for s in configfile.sections()
                     if s.startswith(prefix) and s != head]


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

    section_names = _platform_sections(configfile)

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

    # Tiers keep their positional identity: the Nth section in the file is
    # always 'primary'/'secondary'/... even if an earlier volume is
    # unmounted, so a missing drive shifts nothing under it.
    mea_config = dict()
    for idx, name in enumerate(section_names):
        cfg = configfile[name]
        if not os.path.exists(os.path.abspath(cfg['data'])):
            continue
        row = _row(cfg)
        row['_section'] = name
        mea_config[_tier_key(idx)] = row

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

# Source tiers that are actually mounted, in config-file order. Everything
# below iterates this instead of a hard-coded primary/secondary/tertiary
# triple, so adding a section to config.ini is enough to add a volume.
_SOURCE_TIERS = list(mea_config)

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
    for tier in _SOURCE_TIERS:
        if tier in _config and _config[tier].get(key):
            return _config[tier][key]
    return ''


def _add_local_cache_tier():
    """Resolve the cache root and install it as a tier ahead of every source."""
    global LOCAL_CACHE_ROOT
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


_add_local_cache_tier()

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


def _classify_all_tiers():
    """Rebuild the tier-kind map and the read-priority orders from what is mounted.

    Split out of module scope so :func:`reload_config` can redo it. A tier
    that was unmounted when the package was imported is absent from
    ``mea_config`` entirely, so none of this can be derived once and reused.
    """
    global _TIER_KIND, _local_first, _unknown_tiers, _network_tiers
    global _TIER_PRIORITY, _RAW_TIER_PRIORITY

    _TIER_KIND = {}
    for tier in _SOURCE_TIERS:
        if tier in mea_config:
            _TIER_KIND[tier] = _classify_tier(mea_config[tier].get('data', ''))

    # Compose priority: local cache → local source tiers → unknown → network.
    # Within each group, config-file order is preserved so users still have
    # control over relative ordering of same-kind tiers. The section labels
    # become a *secondary* sort key behind local-vs-network.
    _local_first = [t for t in _SOURCE_TIERS if _TIER_KIND.get(t) == 'local']
    _unknown_tiers = [t for t in _SOURCE_TIERS if _TIER_KIND.get(t) == 'unknown']
    _network_tiers = [t for t in _SOURCE_TIERS if _TIER_KIND.get(t) == 'network']
    _TIER_PRIORITY = (['local_cache'] + _local_first
                      + _unknown_tiers + _network_tiers)
    # Raw .bin files are the exception to "prefer local". They are ~30 GB per
    # experiment and are kept canonical on the NAS — local SSDs only mirror a
    # few recent dates. Resolving raw against the network tier first means a
    # connected NAS always wins, even if a partial SSD copy happens to be
    # present. Falls back to local tiers when NAS is not mounted so the few
    # dates that *are* on SSD still work.
    _RAW_TIER_PRIORITY = _network_tiers + _unknown_tiers + _local_first


_classify_all_tiers()


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


def _pick_raw_tier():
    for tier in _RAW_TIER_PRIORITY:
        if tier in mea_config and mea_config[tier].get('raw'):
            return mea_config[tier]
    return None


def _bind_tier_paths():
    """Point the module-level path constants at the top mounted tier of each kind."""
    global _top, _raw_top, PROTOCOL_REPOS_ROOT
    global DATA_DIR, ANALYSIS_DIR, H5_DIR, META_DIR, TAGS_DIR, QUERY_DIR
    global USER, RAW_DIR

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

    # Root directory holding locally-cloned protocol packages (turner-package,
    # manookin-package, riekelab-package-master, ...). Used by
    # retinanalysis.regen to find resource files (.iml, .mat libraries) and
    # protocol source code. Empty string if not configured / not present.
    PROTOCOL_REPOS_ROOT = next(
        (mea_config[tier]['protocol_repos_root']
         for tier in _TIER_PRIORITY
         if tier in mea_config
         and mea_config[tier].get('protocol_repos_root')
         and os.path.isdir(mea_config[tier]['protocol_repos_root'])),
        '',
    )


_bind_tier_paths()


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


def raw_tier_report(exp_name, datafile_name) -> str:
    """Explain, tier by tier, why :func:`find_raw_path` came up empty.

    The failure worth distinguishing is a tier configured but *not mounted at
    import*: it is missing from ``mea_config`` entirely, so a NAS mounted since
    then is not merely unsearched, it is unknown. That reads identically to a
    genuinely absent file unless the report says so.
    """
    configfile = ConfigParser()
    configfile.read(str(config_path))
    mounted_roots = {mea_config[t].get('raw', '') for t in _RAW_TIER_PRIORITY
                     if t in mea_config}

    lines, unmounted = ['Where it looked:'], []
    for name in _platform_sections(configfile):
        root = configfile[name].get('raw', '')
        if not root:
            continue
        if root in mounted_roots:
            lines.append(f'  {name}: no {exp_name}/{datafile_name} under {root}')
        else:
            unmounted.append(name)
            lines.append(f'  {name}: not searched — {root} was not mounted '
                         f'when retinanalysis was imported')
    if unmounted:
        lines.append('Mounted one of those since? Tiers are discovered at '
                     'import, so run ra.reload_config() and retry — no kernel '
                     'restart needed.')
    return '\n'.join(lines)


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


def tier_dirs(kind):
    """Every existing root for ``kind``, in read-priority order.

    ``find_path`` answers "where is this one file"; this answers "which
    volumes should I sweep". Duplicate roots (two tiers pointing at the same
    tree) collapse to one entry.
    """
    if kind not in ('data', 'raw', 'analysis', 'h5', 'meta', 'tags', 'query'):
        raise ValueError(f"Unknown path kind: {kind}")
    roots = []
    for tier in _TIER_PRIORITY:
        if tier == 'local_cache' or tier not in mea_config:
            continue
        root = mea_config[tier].get(kind, '')
        if root and os.path.isdir(root) and root not in roots:
            roots.append(root)
    return roots


def ingest_source_dirs():
    """``(h5, meta, tags)`` root triples for every mounted tier.

    ``populate_database`` walks these in order so a date that only lives on a
    secondary drive still gets ingested. A tier missing any of the three
    roots is dropped, and tiers resolving to the same triple collapse.
    """
    triples = []
    for tier in _TIER_PRIORITY:
        if tier == 'local_cache' or tier not in mea_config:
            continue
        row = mea_config[tier]
        triple = tuple(row.get(k, '') for k in ('h5', 'meta', 'tags'))
        if not all(os.path.isdir(p) for p in triple if p):
            continue
        if not all(triple) or triple in triples:
            continue
        triples.append(triple)
    return triples


# Write target for ad-hoc outputs (figures, processed dataframes, derived
# classification files, etc.). Resolution order:
#   1. RA_OUTPUT_DIR environment variable (per-shell override).
#   2. First non-empty `output` field across configured tiers — the
#      tier walk prefers the most-local writeable tier (primary →
#      secondary → tertiary), which avoids writing onto the read-only
#      lab server unless the user explicitly set it there.
#   3. `~/retinanalysis_output` — portable default.
def _bind_output_dir():
    global OUTPUT_DIR
    OUTPUT_DIR = (
        os.environ.get('RA_OUTPUT_DIR', '')
        or _first_nonempty_tier_value('output', mea_config)
        or os.path.join(os.path.expanduser('~'), 'retinanalysis_output')
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)


_bind_output_dir()


def _invalidate_lazy_path_constants() -> None:
    """Drop ``ra``'s memoized copies of this module's path constants.

    ``retinanalysis.__getattr__`` writes each resolved name into the package
    globals, so ``ra.DATA_DIR`` is read from settings exactly once per process.
    Deleting the cached entry sends the next access back through the registry.
    """
    names = ('DATA_DIR', 'ANALYSIS_DIR', 'RAW_DIR', 'H5_DIR', 'META_DIR',
             'TAGS_DIR', 'QUERY_DIR', 'OUTPUT_DIR', 'USER',
             'PROTOCOL_REPOS_ROOT')
    for name in names:
        retinanalysis.__dict__.pop(name, None)


def reload_config(verbose: bool = True) -> dict:
    """Re-read config.ini and rediscover which volumes are mounted.

    Everything about tiers is decided when ``retinanalysis`` is imported: a
    section whose root does not exist is dropped from ``mea_config`` outright,
    so a volume mounted afterwards is invisible for the life of the process —
    including to :func:`find_raw_path`, which is how a mounted NAS still
    reports "not found on any configured tier". Mounting a drive mid-session is
    ordinary, and a kernel holding a built pipeline is expensive to restart, so
    this rebuilds the tier state in place instead.

    It rebinds module state, not what other modules already imported. Code that
    did ``from ...settings import OUTPUT_DIR`` keeps its old value; the
    resolver functions — ``find_path``, ``find_raw_path``, ``tier_dirs`` — read
    the globals on each call and so pick the new tiers up immediately. The
    ``ra.DATA_DIR``-style constants are re-resolved too, since ``ra``'s lazy
    ``__getattr__`` caches whatever it hands out and would otherwise keep
    serving the pre-mount value. Call it as ``ra.reload_config()``.

    Returns
    -------
    dict
        ``{'mounted': [...], 'added': [...], 'removed': [...]}``, tiers named
        by their config.ini section.
    """
    global mea_config, _SOURCE_TIERS

    def _sections(cfg):
        return {t: cfg[t].get('_section', t) for t in cfg
                if t != 'local_cache'}

    before = _sections(mea_config)
    mea_config = load_config(config_path)
    _SOURCE_TIERS = list(mea_config)
    _add_local_cache_tier()
    _classify_all_tiers()
    _bind_tier_paths()
    _bind_output_dir()
    _invalidate_lazy_path_constants()
    after = _sections(mea_config)

    added = [after[t] for t in after if t not in before]
    removed = [before[t] for t in before if t not in after]
    result = {'mounted': list(after.values()),
              'added': added, 'removed': removed}
    if verbose:
        order = ' -> '.join(f'{after.get(t, t)} [{_TIER_KIND.get(t, "?")}]'
                            for t in _SOURCE_TIERS)
        print(f'{len(after)} volume(s) mounted, read in priority order: {order}')
        if added:
            print(f'  newly visible: {", ".join(added)}')
        if removed:
            print(f'  no longer mounted: {", ".join(removed)}')
        if not added and not removed:
            print('  unchanged — nothing was mounted or unmounted since import')
    return result