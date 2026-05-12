"""Utilities for normalizing path-style cell-type labels from classification files.

Classification text files often store labels as paths like ``All/onP/`` or
``All/OffP``. The casing isn't standardized across labs and time, so a robust
mapper to the canonical names in ``assets/cell_types.csv`` (``OnP``, ``OffP``,
``LOnM``, ...) needs to be case-insensitive.
"""

import os
import importlib.resources as ir
from typing import Iterable, List, Optional, Union

import pandas as pd

import retinanalysis


def load_canonical_cell_types() -> List[str]:
    """Return the canonical cell type names from ``assets/cell_types.csv``."""
    csv_path = str(ir.files(retinanalysis) / "assets/cell_types.csv")
    return list(pd.read_csv(csv_path)['cell_types'].values)


def map_cell_type(label: Union[str, Iterable[str]],
                  canonical_types: Optional[Iterable[str]] = None,
                  case_insensitive: bool = True) -> Optional[str]:
    """Map a path-style label to its canonical cell-type name.

    ``label`` can be the raw string (``"All/onP/"``) or the already-split path
    parts (``["All", "onP"]``). Returns the canonical name with its original
    casing from ``cell_types.csv``, or ``None`` if no part of the label matches
    a canonical type.

    Parameters
    ----------
    label : str | iterable[str]
        Either the raw classification-file value or its slash-split parts.
    canonical_types : iterable[str], optional
        Override the canonical list. Defaults to the contents of
        ``assets/cell_types.csv``.
    case_insensitive : bool, default True
        If True, match parts to canonical types regardless of casing.
    """
    if canonical_types is None:
        canonical_types = load_canonical_cell_types()

    if isinstance(label, str):
        parts = [p for p in label.split('/') if p]
    else:
        parts = [p for p in label if p]

    if case_insensitive:
        lookup = {c.lower(): c for c in canonical_types}
        for part in parts:
            hit = lookup.get(part.lower())
            if hit is not None:
                return hit
    else:
        canon_set = set(canonical_types)
        for part in parts:
            if part in canon_set:
                return part
    return None


def filter_available_types(requested: Iterable[str],
                           available: Iterable[str],
                           case_insensitive: bool = True) -> List[str]:
    """Return only the requested types that actually appear in ``available``.

    Use this before passing a hard-coded list like ``['OnP', 'OffP', 'OnM']``
    into a plotting / xarray helper — types that aren't present are silently
    skipped instead of raising.
    """
    available = list(available)
    if case_insensitive:
        avail_lookup = {a.lower(): a for a in available}
        return [avail_lookup[r.lower()] for r in requested if r.lower() in avail_lookup]
    avail_set = set(available)
    return [r for r in requested if r in avail_set]
