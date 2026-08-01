"""Read a protocol's MATLAB source to learn what its parameters are.

A Symphony protocol is a MATLAB class in one of the Rieke-Lab packages, and
it is the authority on what the experiment actually did. Two things in it
matter for analysis:

- the ``properties`` block, which declares the protocol's parameters and
  their defaults — these are fixed for a whole epoch block;
- the ``epoch.addParameter(...)`` calls in ``prepareEpoch``, which are
  exactly the parameters that **vary from epoch to epoch**. Those are the
  condition axes of the experiment, and nothing else is.

Reading them beats guessing from the database. The recorded parameters tell
you what values occurred; the source tells you what they *mean*, what the
defaults were, and — importantly — what a parameter is actually called.
``variableMeanDriftingGrating`` writes ``currentBarWdith``, misspelled, so
that is the column name in the data and any analysis that reasonably guesses
``currentBarWidth`` silently finds nothing.

:func:`compare_with_block` puts the declared parameters beside what a
recorded block actually carries, which is the fastest way to see a name that
didn't survive, a default that was overridden on the rig, or a condition axis
you didn't know existed.

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
    'compare_with_block',
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


def compare_with_block(source: ProtocolSource, stim_block) -> pd.DataFrame:
    """Declared parameters beside what the recorded block actually carries.

    One row per parameter the source declares (hidden ones excluded — they
    are rig plumbing, never recorded) plus any epoch parameter the source
    adds. Columns:

    - ``declared`` — the default in the ``.m``.
    - ``recorded`` — the value in the block's ``d_epoch_block_params``, or
      ``varies`` for a parameter that changes per epoch.
    - ``epoch_specific`` — whether ``prepareEpoch`` calls ``addParameter``
      for it, i.e. whether it is a condition axis.
    - ``in_data`` — whether it reached the database at all.

    A declared parameter with ``in_data`` False either wasn't recorded by
    that protocol version or is spelled differently in the data, which is
    the thing worth catching before an analysis quietly reads nothing.
    """
    block_params = dict(getattr(stim_block, 'd_epoch_block_params', {}) or {})
    df_epochs = getattr(stim_block, 'df_epochs', None)

    # Per-epoch names, from the promoted columns and the raw dicts alike:
    # which parameters get their own column depends on the protocol version.
    epoch_names = set()
    if df_epochs is not None and len(df_epochs):
        epoch_names.update(df_epochs.columns)
        if 'epoch_parameters' in df_epochs.columns:
            first = df_epochs['epoch_parameters'].iloc[0]
            if isinstance(first, dict):
                epoch_names.update(first.keys())

    declared = source.parameters.query('not hidden') if len(source.parameters) \
        else pd.DataFrame(columns=['parameter', 'default', 'comment'])

    rows = []
    for _, row in declared.iterrows():
        name = row['parameter']
        rows.append({
            'parameter': name,
            'declared': row['default'],
            'recorded': block_params.get(name, ''),
            'epoch_specific': name in source.epoch_parameters,
            'in_data': name in block_params or name in epoch_names,
            'comment': row['comment'],
        })

    # Epoch parameters are usually derived (currentBarWdith from barWidths),
    # so they are not in the properties block and need adding by hand.
    for name in source.epoch_parameters:
        if any(r['parameter'] == name for r in rows):
            continue
        rows.append({
            'parameter': name,
            'declared': '(per epoch)',
            'recorded': 'varies',
            'epoch_specific': True,
            'in_data': name in epoch_names,
            'comment': '',
        })

    out = pd.DataFrame(rows, columns=['parameter', 'declared', 'recorded',
                                      'epoch_specific', 'in_data', 'comment'])
    # Condition axes first — they are what an analysis is built around.
    return out.sort_values(['epoch_specific', 'in_data'],
                           ascending=[False, False]).reset_index(drop=True)
