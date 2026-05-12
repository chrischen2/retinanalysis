"""Shared I/O helpers for stimulus regeneration."""

from __future__ import annotations

import os
import glob
import numpy as np
from typing import Optional


def load_iml_image(path: str) -> np.ndarray:
    """Load a Van Hateren ``.iml`` natural-image file.

    The file format is big-endian uint16 stored column-major as ``[1536, 1024]``
    in MATLAB. We read in F-order to match, then transpose so the returned
    array has the conventional ``(rows, cols)`` orientation (1024, 1536).

    Values are rescaled to ``[0, 255]`` uint8 (so the brightest pixel becomes
    255), matching the in-protocol normalization in
    ``EyeMovementTrajectoryAlternatingBackground.m``.
    """
    raw = np.fromfile(path, dtype='>u2', count=1536 * 1024)
    if raw.size != 1536 * 1024:
        raise ValueError(
            f'{path}: expected {1536*1024} uint16 values, got {raw.size}'
        )
    matlab_shape = raw.reshape((1536, 1024), order='F')  # MATLAB [1536,1024]
    img = matlab_shape.T  # transpose to (rows=1024, cols=1536)
    img = img.astype(np.float64)
    img *= 255.0 / img.max()
    return img.astype(np.uint8)


def find_image_in_library(repo_root: str, image_name: str,
                          image_set: str = 'VHsubsample_20160105') -> Optional[str]:
    """Locate ``imk{image_name}.iml`` inside a cloned protocol repo.

    Searches a few conventional locations. Returns ``None`` if not found —
    callers should treat that as "image file missing, return metadata only".
    """
    if not repo_root or not os.path.isdir(repo_root):
        return None

    # Try the canonical layout first, then walk a couple alternatives.
    fname = f'imk{image_name}.iml'
    candidates = [
        os.path.join(repo_root, 'resources', image_set, fname),
        os.path.join(repo_root, 'resources', fname),
        os.path.join(repo_root, image_set, fname),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # Fallback: glob for it anywhere under the repo (cheap on small repos).
    matches = glob.glob(os.path.join(repo_root, '**', fname), recursive=True)
    return matches[0] if matches else None
