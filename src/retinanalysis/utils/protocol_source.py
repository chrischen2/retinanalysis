"""Read a protocol's MATLAB source to learn what its parameters are.

A Symphony protocol is a MATLAB class in one of the Rieke-Lab packages.

**The data is the authority on what ran; the source is the authority on what
it means.** The ``properties`` block declares defaults, and a default is only
what a parameter would have been had nobody touched it — on a real rig most
of them are touched. ``variableMeanDriftingGrating`` declares a temporal
frequency of 4 Hz and an aperture of 800; the block analyzed in
``demos/variableMeanDriftingGrating.ipynb`` ran 2 Hz and 2000. Reading
defaults back as if they described the experiment is how you report a
stimulus the retina never saw.

So :func:`block_parameters` takes every value from the recorded epochs and
uses the source for one thing the data cannot supply: the comment beside each
property, which says what the parameter *is*. Which parameters vary across
epochs — the condition axes — is likewise determined by looking at the
epochs, not by trusting ``epoch.addParameter``.

The source still earns its place for names. This protocol writes
``epoch.addParameter('currentBarWdith', ...)``, misspelled, so that is the
column in the database and an analysis reaching for ``currentBarWidth``
silently finds nothing. :func:`condition_keys` reports any name the source
declares per-epoch that does not actually vary in the block, which is how a
mismatch like that surfaces.

This is a regex reader, not a MATLAB parser. It handles the conventional
layout these protocols are written in and reports what it could not read
rather than guessing.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


__all__ = [
    'protocol_source_path',
    'parse_protocol_source',
    'block_parameters',
    'condition_keys',
    'ProtocolSource',
]


# Package component of the dotted protocol name -> cloned repo directory.
# `edu.washington.riekelab.chris.protocols.X` lives in `chris-package`; the
# shared ones don't follow that pattern, hence the explicit entries.
_REPO_FOR_PACKAGE = {
    'riekelab': 'riekelab-package-master',
    'manookin': 'manookin-package',
}

_GITHUB_ORG = 'https://github.com/Rieke-Lab'


@dataclass
class ProtocolSource:
    """What a protocol's ``.m`` file declares."""

    protocol_name: str
    class_name: str
    superclass: str
    path: str
    github_url: str
    parameters: pd.DataFrame = field(default_factory=pd.DataFrame)
    epoch_parameters: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        n_vis = int((~self.parameters['hidden']).sum()) if len(self.parameters) else 0
        return (f'ProtocolSource({self.class_name}: {n_vis} parameters, '
                f'{len(self.epoch_parameters)} epoch-specific)')


def _repo_for(protocol_name: str) -> str:
    """Cloned repo directory name implied by a dotted protocol name."""
    parts = protocol_name.split('.')
    package = parts[-3] if len(parts) >= 3 else ''
    return _REPO_FOR_PACKAGE.get(package, f'{package}-package')


def _github_repo_for(repo_dir: str) -> str:
    return f'{_GITHUB_ORG}/{repo_dir}'


def protocol_source_path(protocol_name: str,
                         repo_name: Optional[str] = None) -> Optional[str]:
    """Absolute path to a protocol's ``.m``, or None if it isn't cloned.

    A dotted name maps onto MATLAB's package layout directly: every component
    but the last becomes a ``+``-prefixed directory, and the last is the file.
    ``edu.washington.riekelab.chris.protocols.variableMeanDriftingGrating``
    becomes ``+edu/+washington/+riekelab/+chris/+protocols/….m``.

    The class name is matched case-insensitively, because the database and the
    filesystem do not always agree on it.
    """
    from retinanalysis.config.settings import find_protocol_repo

    repo_dir = repo_name or _repo_for(protocol_name)
    repo = find_protocol_repo(repo_dir)
    if repo is None:
        return None

    parts = protocol_name.split('.')
    class_name, package_parts = parts[-1], parts[:-1]
    folder = os.path.join(repo, *[f'+{p}' for p in package_parts])
    if not os.path.isdir(folder):
        return None

    exact = os.path.join(folder, f'{class_name}.m')
    if os.path.isfile(exact):
        return exact
    for entry in os.listdir(folder):
        if entry.lower() == f'{class_name.lower()}.m':
            return os.path.join(folder, entry)
    return None


def _parse_properties(text: str) -> pd.DataFrame:
    """Every ``properties`` block, one row per declared parameter.

    Hidden blocks are kept but flagged: they hold the rig plumbing and the
    working variables (``currentBarWidth`` and friends), which is useful
    context even though you never set them.
    """
    rows = []
    # Non-greedy to the matching `end` at the same indent — these classes put
    # `end` on its own line, which is what makes the simple form safe here.
    for match in re.finditer(r'^\s*properties(\s*\([^)]*\))?\s*$(.*?)^\s*end\s*$',
                             text, re.MULTILINE | re.DOTALL):
        attrs = (match.group(1) or '').strip('() \t')
        hidden = 'hidden' in attrs.lower()
        for line in match.group(2).splitlines():
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            # `name = value   % comment`, or a bare `name   % comment`.
            body, _, comment = line.partition('%')
            body = body.strip().rstrip(';').strip()
            if not body:
                continue
            name, eq, default = body.partition('=')
            name = name.strip()
            if not re.fullmatch(r'[A-Za-z_]\w*', name):
                continue
            rows.append({
                'parameter': name,
                'default': default.strip() if eq else '',
                'comment': comment.strip(),
                'hidden': hidden,
            })
    return pd.DataFrame(rows, columns=['parameter', 'default', 'comment', 'hidden'])


def _parse_epoch_parameters(text: str) -> List[str]:
    """Names passed to ``epoch.addParameter`` — the per-epoch condition axes."""
    found = re.findall(r"""addParameter\(\s*['"]([^'"]+)['"]""", text)
    seen, out = set(), []
    for name in found:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def parse_protocol_source(protocol_name: str, repo_name: Optional[str] = None,
                          branch: str = 'master') -> Optional[ProtocolSource]:
    """Read a protocol's ``.m`` and return what it declares.

    ``protocol_name`` is the dotted name as the database stores it. Returns
    None, with a printed reason, when the package isn't cloned locally —
    analysis shouldn't stop because a source repo is missing.
    """
    path = protocol_source_path(protocol_name, repo_name=repo_name)
    if path is None:
        repo_dir = repo_name or _repo_for(protocol_name)
        print(f'No local source for {protocol_name}. Expected it under '
              f'{repo_dir}; clone {_github_repo_for(repo_dir)} into '
              f'PROTOCOL_REPOS_ROOT to read it.')
        return None

    from retinanalysis.config.settings import PROTOCOL_REPOS_ROOT

    text = open(path, 'r', errors='replace').read()

    classdef = re.search(r'classdef\s+(\w+)\s*(?:<\s*([\w.]+))?', text)
    class_name = classdef.group(1) if classdef else protocol_name.split('.')[-1]
    superclass = (classdef.group(2) or '') if classdef else ''

    repo_dir = repo_name or _repo_for(protocol_name)
    repo_root = os.path.join(PROTOCOL_REPOS_ROOT, repo_dir)
    rel = os.path.relpath(path, repo_root).replace(os.sep, '/')
    github_url = f'{_github_repo_for(repo_dir)}/blob/{branch}/{rel}'

    return ProtocolSource(
        protocol_name=protocol_name,
        class_name=class_name,
        superclass=superclass,
        path=path,
        github_url=github_url,
        parameters=_parse_properties(text),
        epoch_parameters=_parse_epoch_parameters(text),
    )


def block_parameters(stim_block, source: Optional[ProtocolSource] = None,
                     max_levels: int = 8) -> pd.DataFrame:
    """What this block actually ran, one row per recorded parameter.

    Values come from the data, never from the ``.m``. A default in the source
    is only what the parameter would have been had nobody touched it, and on
    a real rig most of them are touched — reading them back as if they
    described the experiment is how you end up reporting a temporal frequency
    the retina never saw.

    Columns:

    - ``value`` — the recorded value. For a parameter that changes across
      epochs this is the sorted list of levels it took, which is the useful
      form: those levels are the experiment's design.
    - ``epoch_specific`` — whether the value changes from epoch to epoch,
      determined by looking at the epochs. These are the condition axes.
    - ``n_levels`` — how many distinct values it took.
    - ``comment`` — the description beside the property in the ``.m``, when
      the source is available. The one thing here the data cannot supply,
      and the reason to pass ``source`` at all.

    ``source`` is optional; without it you get the same table minus the
    descriptions.
    """
    block_params = dict(getattr(stim_block, 'd_epoch_block_params', {}) or {})
    df_epochs = getattr(stim_block, 'df_epochs', None)

    # Per-epoch values come from the protocol's own parameter dict, which is
    # the record of what the protocol set. Scanning df_epochs columns instead
    # sweeps in database bookkeeping — epoch_id and epoch_index take a
    # different value every epoch and would read as the strongest condition
    # axes in the block. The promoted columns are derived from this dict
    # anyway, so nothing is lost.
    per_epoch: Dict[str, list] = {}
    if df_epochs is not None and len(df_epochs):
        if 'epoch_parameters' in df_epochs.columns:
            for record in df_epochs['epoch_parameters']:
                if isinstance(record, dict):
                    for key, value in record.items():
                        per_epoch.setdefault(key, []).append(value)
        else:
            # No raw dict on this frame: fall back to columns, minus the
            # identifiers and payloads that are about the recording rather
            # than the stimulus.
            skip = {'epoch_parameters', 'frame_times_ms', 'data_dir',
                    'epoch_index', 'epoch_id', 'block_id', 'group_id',
                    'protocol_id', 'experiment_id', 'chunk_id',
                    'exp_name', 'datafile_name', 'group_label',
                    'protocol_name'}
            for column in df_epochs.columns:
                if column not in skip:
                    per_epoch[column] = df_epochs[column].tolist()

    comments = {}
    if source is not None and len(source.parameters):
        comments = dict(zip(source.parameters['parameter'],
                            source.parameters['comment']))

    def _levels(values):
        """Distinct values, sorted where they are comparable.

        Deduplicated on ``repr`` rather than equality: several of these
        parameters are lists (``barWidths``, ``canvasSize``) or numpy arrays,
        where ``in`` either raises or compares elementwise. Skipping those
        types instead — the first version did — silently blanked exactly the
        settings that say what the stimulus swept over.
        """
        unique, seen = [], set()
        for v in values:
            key = repr(v)
            if key in seen:
                continue
            seen.add(key)
            unique.append(v)
        try:
            return sorted(unique)
        except TypeError:
            return unique

    rows = []
    for name, values in per_epoch.items():
        levels = _levels(values)
        if len(levels) <= 1:
            # Recorded per epoch but the same every time — a setting, not an
            # axis. It still belongs in the table: most of a protocol's
            # parameters live here, and they are what was on the screen.
            rows.append({
                'parameter': name,
                'value': levels[0] if levels else '',
                'epoch_specific': False,
                'n_levels': 1,
                'comment': comments.get(name, ''),
            })
            continue
        rows.append({
            'parameter': name,
            'value': levels if len(levels) <= max_levels else f'{len(levels)} values',
            'epoch_specific': True,
            'n_levels': len(levels),
            'comment': comments.get(name, ''),
        })

    # Block-level record for anything the epochs did not carry.
    epoch_names = set(per_epoch)
    for name, value in block_params.items():
        if name in epoch_names:
            continue
        rows.append({
            'parameter': name,
            'value': value,
            'epoch_specific': False,
            'n_levels': 1,
            'comment': comments.get(name, ''),
        })

    out = pd.DataFrame(rows, columns=['parameter', 'value', 'epoch_specific',
                                      'n_levels', 'comment'])
    if out.empty:
        return out
    # Condition axes first — they are what an analysis is built around.
    return out.sort_values(['epoch_specific', 'parameter'],
                           ascending=[False, True]).reset_index(drop=True)


def condition_keys(stim_block, source: Optional[ProtocolSource] = None) -> List[str]:
    """Names of the parameters that change from epoch to epoch.

    Taken from the epochs, so it reflects what the block did rather than what
    the protocol intended. When ``source`` is given, any name the protocol
    passes to ``epoch.addParameter`` but that does *not* vary in the data is
    reported — that is either a condition the block only sampled at one level,
    or a parameter recorded under a different name than the source suggests.
    """
    table = block_parameters(stim_block, source=source)
    keys = table.query('epoch_specific')['parameter'].tolist() if len(table) else []

    if source is not None:
        missing = [n for n in source.epoch_parameters if n not in keys]
        if missing:
            print(f'Declared per-epoch but not varying in this block: {missing} '
                  f'— one level only, or recorded under another name.')
    return keys
