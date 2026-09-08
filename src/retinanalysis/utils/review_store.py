"""Protocol-independent CSV review decisions keyed by stable recording identity.

Callers choose the file, identity columns, and decision columns. Use separate
stores when decisions have different scopes (e.g. cell versus cell/condition).
Reading never creates files; absent boolean flags mean False.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd


class ReviewStore:
    """Small persisted review table; indices can be provenance, not identity.

    ``update(identity, values)`` replaces one row; ``values=None`` removes it.
    Removal only changes this review table, never the underlying recording.
    """

    def __init__(self, path, *, keys, columns, boolean_columns=()):
        self.path = Path(path)
        self.keys = tuple(keys)
        self.columns = tuple(columns)
        self.boolean_columns = tuple(boolean_columns)
        if not self.keys or not set(self.keys).issubset(self.columns):
            raise ValueError('Review keys must be nonempty and present in columns')
        if not set(self.boolean_columns).issubset(self.columns):
            raise ValueError('Boolean columns must be present in columns')

    def read(self):
        if not self.path.exists():
            frame = pd.DataFrame(columns=self.columns)
        else:
            frame = pd.read_csv(self.path, dtype={key: str for key in self.keys})
            missing = set(self.keys) - set(frame.columns)
            if missing:
                raise ValueError(f'{self.path} lacks review keys {sorted(missing)}')
        for column in self.boolean_columns:
            values = frame.get(column, pd.Series(False, index=frame.index))
            text = values.fillna(False).astype(str).str.lower()
            if not text.isin(['true', 'false', '1', '0']).all():
                raise ValueError(f'{self.path}: invalid boolean in {column}')
            frame[column] = text.isin(['true', '1'])
        return frame.reindex(columns=self.columns)

    def matches(self, frame, identity):
        if set(identity) != set(self.keys):
            raise ValueError(f'Identity must supply exactly {self.keys}')
        mask = pd.Series(True, index=frame.index)
        for key in self.keys:
            if pd.isna(identity[key]):
                raise ValueError(f'Missing review identity: {key}')
            mask &= frame[key].astype(str).eq(str(identity[key]))
        return mask

    def flag(self, identity, column):
        """Return a boolean flag, defaulting to False for unreviewed entries."""
        if column not in self.boolean_columns:
            raise ValueError(f'{column} is not a boolean review column')
        frame = self.read()
        return bool(frame.loc[self.matches(frame, identity), column].any())

    def update(self, identity, values, *, frame=None):
        """Update one identity atomically; ``frame`` supports legacy migrations.

        Without ``frame``, reread at every action so a browser does not overwrite
        decisions another browser saved after it was opened.
        """
        frame = self.read() if frame is None else frame.copy()
        frame = frame.loc[~self.matches(frame, identity)].copy()
        if values is not None:
            if set(values) - set(self.columns) or set(values) & set(self.keys):
                raise ValueError('Values must be non-identity review columns')
            row = {**identity, **values}
            for column in self.boolean_columns:
                row.setdefault(column, False)
                if not isinstance(row[column], bool):
                    raise ValueError(f'{column} must be a bool')
            frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        frame = frame.reindex(columns=self.columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                dir=self.path.parent, prefix=self.path.name + '.',
                suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            frame.to_csv(temporary, index=False)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
        return frame
