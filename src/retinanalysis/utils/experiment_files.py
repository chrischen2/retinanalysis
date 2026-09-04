"""Filename rules for experiment files consumed by the analysis pipeline."""

from __future__ import annotations

import datetime
from pathlib import Path
import re
from typing import Optional


# Single-cell Symphony dates use an ISO date, a one-letter recording code,
# and occasionally a run/cell suffix: 2023-08-29_G, 2025-04-17_E_2,
# 2021-04-27_G-2, or 2023-08-29_G_c1-3. Dots are deliberately forbidden,
# which excludes auxiliary files such as 2020-10-08_B.auisql.h5.
_SINGLE_CELL_STEM = re.compile(
    r'^(?P<date>\d{4}-\d{2}-\d{2})_[A-Za-z](?:[_-][A-Za-z0-9-]+)*$')

# MEA metadata has a separate established name and no H5 counterpart.
_MEA_STEM = re.compile(
    r'^(?P<date>\d{8})[A-Za-z](?:[_-][A-Za-z0-9-]+)*$')


def _has_suffix(name: str, suffix: Optional[str]) -> bool:
    return suffix is None or Path(name).suffix.lower() == suffix.lower()


def is_single_cell_experiment_file(
        path_or_name, suffix: Optional[str] = None) -> bool:
    """Whether a filename is a canonical single-cell experiment file.

    ``suffix`` can restrict the check to ``'.h5'`` or ``'.json'``. Hidden
    files, AppleDouble files, invalid calendar dates, old undated names and
    auxiliary multi-suffix files are rejected silently by discovery callers.
    """
    name = Path(path_or_name).name
    if name.startswith('.') or not _has_suffix(name, suffix):
        return False
    match = _SINGLE_CELL_STEM.fullmatch(Path(name).stem)
    if match is None:
        return False
    try:
        datetime.date.fromisoformat(match.group('date'))
    except ValueError:
        return False
    return True


def single_cell_json_stem(path_or_name) -> Optional[str]:
    """Return the canonical JSON stem for a single-cell H5 source.

    Ordinary files map directly (``2020-10-08_B.h5`` ->
    ``2020-10-08_B``). An AUISQL fallback maps to that same canonical stem
    (``2020-10-08_B.auisql.h5`` -> ``2020-10-08_B``). Other filenames return
    ``None``.
    """
    name = Path(path_or_name).name
    if name.startswith('.') or Path(name).suffix.lower() != '.h5':
        return None

    stem = Path(name).stem
    if stem.lower().endswith('.auisql'):
        stem = stem[:-len('.auisql')]

    match = _SINGLE_CELL_STEM.fullmatch(stem)
    if match is None:
        return None
    try:
        datetime.date.fromisoformat(match.group('date'))
    except ValueError:
        return None
    return stem


def is_mea_experiment_file(path_or_name, suffix: Optional[str] = None) -> bool:
    """Whether a filename follows the established MEA date-name format."""
    name = Path(path_or_name).name
    if name.startswith('.') or not _has_suffix(name, suffix):
        return False
    match = _MEA_STEM.fullmatch(Path(name).stem)
    if match is None:
        return False
    try:
        datetime.datetime.strptime(match.group('date'), '%Y%m%d')
    except ValueError:
        return False
    return True


__all__ = [
    'is_mea_experiment_file',
    'is_single_cell_experiment_file',
    'single_cell_json_stem',
]
