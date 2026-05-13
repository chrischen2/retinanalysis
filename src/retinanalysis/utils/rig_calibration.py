"""Per-experiment MEA→canvas alignment learned from EI/STA correspondences.

The Litke chip is physically mounted at a particular orientation for each
recording session, so electrode coordinates (µm, chip-centered) only map
to the display canvas after a rigid alignment. We learn this alignment
empirically: for each cell, the electrode where the EI peaks earliest (the
AP-initiation / soma electrode) is paired with that same cell's STA
center on the canvas. A 2D similarity transform (rotation + uniform scale
+ translation) is fit to the pairs by Procrustes / SVD.

**The chip is remounted between recording sessions**, so the calibration
is per-experiment (i.e. per recording day), not per-rig — empirically a
calibration fit on one day differs by 60-80° and 30-60% scale from one
fit three months later on the same rig. Within a recording day, multiple
chunks can be combined (chip doesn't move during a day).

JSON layout (one file per experiment at
``<repo>/rig_calibrations/<exp_name>.json``)::

    {
      "exp_name": "20220823C",
      "rig_id": "C",
      "scale_px_per_um": 0.262...,
      "rotation_deg": -88.8...,
      "tx_px": 340.1, "ty_px": 363.5,
      "flip_y": true,
      "residual_um_rms": 98.0,
      "n_cells": 211,
      "n_recordings": 1,
      "contributing": [{"exp_name": "20220823C", "chunk_name": "chunk5", "n_cells": 234}],
      "updated_utc": "2026-05-12T18:00:00Z"
    }

A row of ``contributing`` records which chunk supplied which pairs, so
calibrations can be inspected, re-derived, or pruned.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, asdict, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


__all__ = [
    'RigCalibration',
    'rig_id_from_exp_name',
    'compute_soma_electrode',
    'extract_alignment_pairs',
    'fit_similarity_transform',
    'fit_similarity_transform_robust',
    'apply_similarity_transform',
    'rig_calibrations_dir',
    'rig_calibration_path',
    'save_rig_calibration',
    'load_rig_calibration',
    'accumulate_and_save',
]


# ---------------------------------------------------------------------------
# Default cell-quality screens for calibration
# ---------------------------------------------------------------------------

# Cells with these labels in a typing file are excluded from calibration by
# default: their STA centers are either undefined or unreliable as a soma-
# position proxy. Override via the ``cell_types`` kwarg.
_EXCLUDE_TYPES_DEFAULT = frozenset({'Unknown', 'Amacrine', 'Crap', 'Junk'})


# ---------------------------------------------------------------------------
# Storage / filesystem layout
# ---------------------------------------------------------------------------

def _repo_root() -> str:
    """Return the repo root (parent of ``src/``)."""
    here = os.path.abspath(__file__)
    # .../src/retinanalysis/utils/rig_calibration.py → up 4 levels
    return os.path.abspath(os.path.join(here, '..', '..', '..', '..'))


def rig_calibrations_dir() -> str:
    """Return ``<repo>/rig_calibrations/`` (created if missing)."""
    p = os.path.join(_repo_root(), 'rig_calibrations')
    os.makedirs(p, exist_ok=True)
    return p


def rig_calibration_path(exp_name: str) -> str:
    """Per-experiment calibration JSON path (one file per recording day)."""
    return os.path.join(rig_calibrations_dir(), f'{exp_name}.json')


def rig_id_from_exp_name(exp_name: str) -> str:
    """Extract the rig letter from an experiment name.

    Experiment names look like ``YYYYMMDD<RIG>[suffix]`` (e.g. ``20220823C``),
    so the 9th character is the rig identifier. Raises ``ValueError`` if the
    name is too short to contain a rig letter.
    """
    if len(exp_name) < 9:
        raise ValueError(f'exp_name {exp_name!r} too short to contain a rig id')
    return exp_name[8]


# ---------------------------------------------------------------------------
# EI soma electrode detection
# ---------------------------------------------------------------------------

def compute_soma_electrode(
    ei: np.ndarray,
    initiation_frac: float = 1.0 / 3.0,
    min_amp: float = 0.0,
) -> Tuple[int, float]:
    """Identify the AP-initiation electrode for a single cell's EI.

    The extracellular action potential begins near the soma, so the electrode
    whose ``|EI|`` is largest within the first ``initiation_frac`` of the EI
    time window is the best single-electrode proxy for the cell's soma
    location. Time samples after that window pick up downstream axonal
    propagation and would bias the estimate away from the soma.

    Parameters
    ----------
    ei : np.ndarray
        Shape ``(n_electrodes, n_frames)`` (Vision's ``get_ei_for_cell(id).ei``).
    initiation_frac : float
        Fraction of the EI window treated as the initiation phase
        (default 1/3 — empirically captures the soma peak without leaking
        into the axonal trail).
    min_amp : float
        Reject cells whose peak amplitude in the initiation window is below
        this threshold. ``0`` accepts everything.

    Returns
    -------
    (electrode_idx, peak_amp) : (int, float)
        Electrode index (into ``vcd.get_electrode_map()``) and the peak
        ``|EI|`` value in the initiation window.

    Raises
    ------
    ValueError if ``ei`` is not 2-D or has zero frames, or if the peak
    amplitude is below ``min_amp``.
    """
    if ei.ndim != 2:
        raise ValueError(f'EI must be 2-D, got shape {ei.shape}')
    n_frames = ei.shape[1]
    if n_frames == 0:
        raise ValueError('EI has zero frames')

    n_init = max(1, int(round(initiation_frac * n_frames)))
    window = np.abs(ei[:, :n_init])
    per_electrode_peak = window.max(axis=1)
    elec = int(np.argmax(per_electrode_peak))
    amp = float(per_electrode_peak[elec])
    if amp < min_amp:
        raise ValueError(f'Initiation-window peak {amp:.3g} below min_amp={min_amp}')
    return elec, amp


def _sta_center_canvas_px(analysis_chunk, cell_id: int) -> Tuple[float, float]:
    """Return the cell's STA center in canvas-pixel coords.

    ``rf_params`` is in *full-noise-grid stixels* with the y-flip already
    applied (see ``analysis_chunk.get_rf_params``). Multiply by
    ``pixels_per_stixel`` to land on the same canvas-pixel grid the stim
    and electrode overlay use.
    """
    p = analysis_chunk.rf_params[cell_id]
    pps = analysis_chunk.pixels_per_stixel
    return float(p['center_x']) * pps, float(p['center_y']) * pps


def _resolve_keep_cell_ids(
    analysis_chunk,
    cell_ids: Optional[Iterable[int]],
    cell_types: Optional[Iterable[str]],
    exclude_types: Iterable[str],
    typing_file: Optional[str],
) -> List[int]:
    """Apply cell-type include/exclude filters → ordered list of cell_ids."""
    if cell_ids is not None:
        return [int(c) for c in cell_ids]

    if typing_file is None:
        typing_file = analysis_chunk.typing_files[0] if analysis_chunk.typing_files else None
    if typing_file is None:
        # No typing → fall back to every cell that has an EI.
        return [int(c) for c in analysis_chunk.d_EIs.keys()]

    idx = analysis_chunk.typing_files.index(typing_file)
    col = f'typing_file_{idx}'
    df = analysis_chunk.df_cell_params

    if cell_types is not None:
        wanted = set(cell_types)
        keep = df.loc[df[col].isin(wanted), 'cell_id']
    else:
        excl = set(exclude_types)
        keep = df.loc[~df[col].isin(excl), 'cell_id']
    return [int(c) for c in keep.tolist()]


def extract_alignment_pairs(
    analysis_chunk,
    cell_ids: Optional[Iterable[int]] = None,
    cell_types: Optional[Iterable[str]] = None,
    exclude_types: Iterable[str] = _EXCLUDE_TYPES_DEFAULT,
    typing_file: Optional[str] = None,
    initiation_frac: float = 1.0 / 3.0,
    min_init_amp: float = 0.0,
    sta_sigma_range_stixels: Tuple[float, float] = (0.5, 50.0),
    require_finite_rf: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Pair (soma electrode µm, STA center canvas px) for cells in a chunk.

    Default quality screens (override per kwarg):
      - Drop cells in ``exclude_types`` (``Unknown``, ``Amacrine``, …) so
        only well-classified RGCs contribute. Pass ``cell_types`` to use
        an allow-list instead.
      - Require STA σ in ``sta_sigma_range_stixels`` along both axes — too
        small means an artifact, too large means a bad fit.
      - Drop cells whose initiation-window peak ``|EI|`` is below
        ``min_init_amp``.
      - Drop cells with missing EI / missing RF / NaN-Inf RF centers.

    Returns
    -------
    em_um : (n, 2) array
        Chip-coord soma electrode positions in µm.
    sta_px : (n, 2) array
        STA-center positions in canvas pixels.
    used_ids : list[int]
        Cell ids actually used (same length / order as the arrays above).
    """
    if not hasattr(analysis_chunk, 'd_EIs') or not analysis_chunk.d_EIs:
        raise RuntimeError(
            f'AnalysisChunk for {analysis_chunk.exp_name}/{analysis_chunk.chunk_name} '
            'has no EI dictionary — re-init with include_ei=True.'
        )
    em = analysis_chunk.vcd.get_electrode_map()  # (n_electrodes, 2) in µm

    keep_ids = _resolve_keep_cell_ids(
        analysis_chunk, cell_ids, cell_types, exclude_types, typing_file,
    )
    sigma_lo, sigma_hi = sta_sigma_range_stixels

    used_ids: List[int] = []
    em_pts: List[Tuple[float, float]] = []
    sta_pts: List[Tuple[float, float]] = []
    for cid in keep_ids:
        ei = analysis_chunk.d_EIs.get(int(cid))
        if ei is None or cid not in analysis_chunk.rf_params:
            continue
        p = analysis_chunk.rf_params[int(cid)]
        if not (sigma_lo <= float(p['std_x']) <= sigma_hi
                and sigma_lo <= float(p['std_y']) <= sigma_hi):
            continue
        try:
            elec, _ = compute_soma_electrode(
                ei, initiation_frac=initiation_frac, min_amp=min_init_amp,
            )
        except ValueError:
            continue
        sx, sy = _sta_center_canvas_px(analysis_chunk, int(cid))
        if require_finite_rf and not (np.isfinite(sx) and np.isfinite(sy)):
            continue
        em_pts.append((float(em[elec, 0]), float(em[elec, 1])))
        sta_pts.append((sx, sy))
        used_ids.append(int(cid))

    return np.asarray(em_pts), np.asarray(sta_pts), used_ids


# ---------------------------------------------------------------------------
# Similarity transform
# ---------------------------------------------------------------------------

@dataclass
class RigCalibration:
    """Fitted similarity transform µm → canvas pixels for one experiment."""

    exp_name: str
    rig_id: str
    scale_px_per_um: float
    rotation_deg: float
    tx_px: float
    ty_px: float
    flip_y: bool = True  # canvas y points down; convenience-only, included in fit
    residual_um_rms: float = float('nan')
    n_cells: int = 0
    n_recordings: int = 0
    contributing: List[dict] = field(default_factory=list)
    updated_utc: str = ''

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'RigCalibration':
        # Back-compat: older files may not have exp_name.
        if 'exp_name' not in d:
            # Pull from the first contributing chunk if available.
            contrib = d.get('contributing', [])
            d['exp_name'] = contrib[0]['exp_name'] if contrib else ''
        return cls(**d)


def fit_similarity_transform(
    em_um: np.ndarray,
    sta_px: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """Solve ``sta_px ≈ s * R(θ) @ em_um + (tx, ty)`` (closed-form Procrustes).

    Returns ``(scale, theta_rad, tx, ty, residual_rms_px)``.

    Notes
    -----
    - ``scale`` has units px/µm — multiply chip-µm by it to get canvas pixels.
    - ``theta`` is in radians, CCW. Note this is the rotation in the
      destination frame; if you want to compare against ``mea_rotation_deg``
      from rig config, account for the y-flip (canvas y points down).
    - At least 3 well-conditioned pairs are required (anything less is
      degenerate). With 3 collinear points scale + rotation are not jointly
      identifiable — callers should accumulate ≥ ~20 spatially spread
      cells before trusting the fit.
    """
    em_um = np.asarray(em_um, dtype=float)
    sta_px = np.asarray(sta_px, dtype=float)
    if em_um.shape != sta_px.shape or em_um.ndim != 2 or em_um.shape[1] != 2:
        raise ValueError(f'em_um and sta_px must be (n, 2); got {em_um.shape}, {sta_px.shape}')
    n = em_um.shape[0]
    if n < 3:
        raise ValueError(f'Need at least 3 pairs to fit a similarity transform; got {n}')

    em_mean = em_um.mean(axis=0)
    sta_mean = sta_px.mean(axis=0)
    em_c = em_um - em_mean
    sta_c = sta_px - sta_mean

    # Cross-covariance and SVD (Kabsch–Umeyama)
    H = em_c.T @ sta_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    if d == 0:
        d = 1.0
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T  # 2x2 rotation in destination frame
    # Scale: trace(S' * D) / sum(em_c**2)
    var_em = (em_c ** 2).sum()
    scale = float((S * np.array([1.0, d])).sum() / var_em) if var_em > 0 else float('nan')
    t = sta_mean - scale * (R @ em_mean)
    theta = float(np.arctan2(R[1, 0], R[0, 0]))

    # Residual
    pred = scale * (em_um @ R.T) + t
    residual_rms_px = float(np.sqrt(((pred - sta_px) ** 2).sum(axis=1).mean()))

    return scale, theta, float(t[0]), float(t[1]), residual_rms_px


def fit_similarity_transform_robust(
    em_um: np.ndarray,
    sta_px: np.ndarray,
    max_iter: int = 5,
    inlier_sigma: float = 3.0,
    min_inliers: int = 8,
    verbose: bool = False,
) -> Tuple[float, float, float, float, float, np.ndarray]:
    """Iteratively reweighted similarity fit that trims residual-outlier pairs.

    At each iteration: fit, compute per-pair residuals, keep pairs whose
    residual is within ``inlier_sigma * median_abs_dev`` of the median,
    refit. Stops when the inlier set stops shrinking or after
    ``max_iter`` iterations. The final return is the fit on the final
    inlier set plus the boolean inlier mask (length matches the input).

    Falls back to a single non-robust fit if the inlier set ever drops
    below ``min_inliers`` (raises if even the first fit has fewer pairs).
    """
    em_um = np.asarray(em_um, dtype=float)
    sta_px = np.asarray(sta_px, dtype=float)
    n_total = em_um.shape[0]
    if n_total < max(3, min_inliers):
        raise ValueError(
            f'Need at least {max(3, min_inliers)} pairs for robust fit; got {n_total}'
        )

    inliers = np.ones(n_total, dtype=bool)
    last_count = n_total
    scale = theta = tx = ty = residual = float('nan')
    for it in range(max_iter):
        scale, theta, tx, ty, residual = fit_similarity_transform(
            em_um[inliers], sta_px[inliers],
        )
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        pred = scale * (em_um @ R.T) + np.array([tx, ty])
        per_pair = np.sqrt(((pred - sta_px) ** 2).sum(axis=1))
        med = float(np.median(per_pair[inliers]))
        mad = float(np.median(np.abs(per_pair[inliers] - med)))
        # 1.4826 * MAD ≈ σ for Gaussian residuals
        sigma_est = max(1.4826 * mad, 1e-6)
        new_inliers = per_pair <= (med + inlier_sigma * sigma_est)
        new_count = int(new_inliers.sum())
        if verbose:
            print(f'[robust fit] iter={it}  inliers={new_count}/{n_total}  '
                  f'residual={residual:.2f} px  med={med:.2f}  σ̂={sigma_est:.2f}')
        if new_count < min_inliers:
            # Don't shrink further — keep the previous inlier set.
            break
        if new_count == last_count:
            inliers = new_inliers
            break
        inliers = new_inliers
        last_count = new_count

    return scale, theta, tx, ty, residual, inliers


def apply_similarity_transform(
    em_um: np.ndarray, calib: RigCalibration,
) -> np.ndarray:
    """Map MEA µm → canvas pixels using a saved RigCalibration."""
    em_um = np.asarray(em_um, dtype=float).reshape(-1, 2)
    theta = np.deg2rad(calib.rotation_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    out = calib.scale_px_per_um * (em_um @ R.T)
    out[:, 0] += calib.tx_px
    out[:, 1] += calib.ty_px
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_rig_calibration(calib: RigCalibration) -> str:
    """Write ``calib`` to ``rig_calibrations/<exp_name>.json``. Returns the path."""
    if not calib.updated_utc:
        calib.updated_utc = _dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    path = rig_calibration_path(calib.exp_name)
    with open(path, 'w') as f:
        json.dump(calib.to_dict(), f, indent=2)
    return path


def load_rig_calibration(exp_name: str) -> Optional[RigCalibration]:
    """Return the saved calibration for ``exp_name``, or ``None`` if absent."""
    path = rig_calibration_path(exp_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return RigCalibration.from_dict(d)


# ---------------------------------------------------------------------------
# High-level: accumulate across chunks → fit → save
# ---------------------------------------------------------------------------

def _calibration_from_pairs(
    exp_name: str,
    rig_id: str,
    em_um: np.ndarray,
    sta_px: np.ndarray,
    contributing: List[dict],
    robust: bool = True,
    inlier_sigma: float = 3.0,
    max_iter: int = 5,
) -> Tuple[RigCalibration, np.ndarray]:
    """Fit transform from pairs; returns (calibration, inlier_mask).

    The mask has length == ``em_um.shape[0]``. When ``robust=False`` every
    pair is treated as an inlier.
    """
    if robust:
        scale, theta, tx, ty, residual_px, inliers = fit_similarity_transform_robust(
            em_um, sta_px, max_iter=max_iter, inlier_sigma=inlier_sigma,
        )
    else:
        scale, theta, tx, ty, residual_px = fit_similarity_transform(em_um, sta_px)
        inliers = np.ones(em_um.shape[0], dtype=bool)
    # Residual back to µm so it's interpretable against electrode pitch (~60 µm).
    residual_um = residual_px / scale if scale > 0 else float('nan')
    n_recs = len({(c['exp_name'], c['chunk_name']) for c in contributing})
    calib = RigCalibration(
        exp_name=exp_name,
        rig_id=rig_id,
        scale_px_per_um=float(scale),
        rotation_deg=float(np.rad2deg(theta)),
        tx_px=float(tx), ty_px=float(ty),
        flip_y=True,
        residual_um_rms=float(residual_um),
        n_cells=int(inliers.sum()),
        n_recordings=int(n_recs),
        contributing=contributing,
    )
    return calib, inliers


def fit_calibration_for_chunk(
    analysis_chunk,
    initiation_frac: float = 1.0 / 3.0,
    min_init_amp: float = 0.0,
    cell_types: Optional[Iterable[str]] = None,
    exclude_types: Iterable[str] = _EXCLUDE_TYPES_DEFAULT,
    typing_file: Optional[str] = None,
    sta_sigma_range_stixels: Tuple[float, float] = (0.5, 50.0),
    robust: bool = True,
    inlier_sigma: float = 3.0,
    verbose: bool = False,
) -> RigCalibration:
    """Fit a fresh similarity transform from a single chunk (no persistence).

    Returns a :class:`RigCalibration` whose ``n_cells`` reflects the
    *inlier* count after robust trimming.
    """
    em_um, sta_px, used_ids = extract_alignment_pairs(
        analysis_chunk,
        cell_types=cell_types, exclude_types=exclude_types,
        typing_file=typing_file,
        initiation_frac=initiation_frac, min_init_amp=min_init_amp,
        sta_sigma_range_stixels=sta_sigma_range_stixels,
    )
    if verbose:
        print(f'[rig_calibration] {analysis_chunk.exp_name}/{analysis_chunk.chunk_name}: '
              f'{len(used_ids)} cells after quality filter')
    rig_id = rig_id_from_exp_name(analysis_chunk.exp_name)
    contributing = [{
        'exp_name': analysis_chunk.exp_name,
        'chunk_name': analysis_chunk.chunk_name,
        'n_cells': len(used_ids),
    }]
    calib, _inliers = _calibration_from_pairs(
        analysis_chunk.exp_name, rig_id, em_um, sta_px, contributing,
        robust=robust, inlier_sigma=inlier_sigma,
    )
    return calib


def accumulate_and_save(
    analysis_chunks: Sequence,
    initiation_frac: float = 1.0 / 3.0,
    min_init_amp: float = 0.0,
    cell_types: Optional[Iterable[str]] = None,
    exclude_types: Iterable[str] = _EXCLUDE_TYPES_DEFAULT,
    typing_file: Optional[str] = None,
    sta_sigma_range_stixels: Tuple[float, float] = (0.5, 50.0),
    robust: bool = True,
    inlier_sigma: float = 3.0,
    merge_with_existing: bool = False,
    verbose: bool = True,
) -> RigCalibration:
    """Fit a single calibration from many chunks of the **same experiment**.

    All ``analysis_chunks`` must share the same ``exp_name`` (the chip is
    physically remounted between recording sessions, so calibrations are
    not transferable across days). To pool multiple chunks from the same
    day pass them all here — the chip doesn't move within a recording day.

    With ``merge_with_existing=True``, contributing-record history from a
    previously saved calibration is merged in (but the fit itself only sees
    the pairs from ``analysis_chunks`` — there is no per-pair persistence
    yet, so the "merge" is informational, not numerical).
    """
    if not analysis_chunks:
        raise ValueError('analysis_chunks is empty')

    exp_names = {ac.exp_name for ac in analysis_chunks}
    if len(exp_names) != 1:
        raise ValueError(
            f'Mixed experiments in input: {exp_names}. Calibrations are '
            'per-experiment (chip is remounted between recording days).'
        )
    exp_name = exp_names.pop()
    rig_id = rig_id_from_exp_name(exp_name)

    all_em: List[np.ndarray] = []
    all_sta: List[np.ndarray] = []
    contributing: List[dict] = []
    for ac in analysis_chunks:
        em_um, sta_px, used_ids = extract_alignment_pairs(
            ac,
            cell_types=cell_types, exclude_types=exclude_types,
            typing_file=typing_file,
            initiation_frac=initiation_frac, min_init_amp=min_init_amp,
            sta_sigma_range_stixels=sta_sigma_range_stixels,
        )
        if verbose:
            print(f'[rig_calibration] {ac.exp_name}/{ac.chunk_name}: '
                  f'{len(used_ids)} usable cells')
        if len(used_ids) == 0:
            continue
        all_em.append(em_um)
        all_sta.append(sta_px)
        contributing.append({
            'exp_name': ac.exp_name,
            'chunk_name': ac.chunk_name,
            'n_cells': len(used_ids),
        })
    if not all_em:
        raise RuntimeError('No usable cells across the supplied chunks.')

    em_all = np.concatenate(all_em, axis=0)
    sta_all = np.concatenate(all_sta, axis=0)

    if merge_with_existing:
        existing = load_rig_calibration(exp_name)
        if existing is not None:
            new_keys = {(c['exp_name'], c['chunk_name']) for c in contributing}
            for c in existing.contributing:
                if (c['exp_name'], c['chunk_name']) not in new_keys:
                    contributing.append(c)

    calib, _inliers = _calibration_from_pairs(
        exp_name, rig_id, em_all, sta_all, contributing,
        robust=robust, inlier_sigma=inlier_sigma,
    )
    path = save_rig_calibration(calib)
    if verbose:
        print(f'[rig_calibration] saved → {path}')
        print(f'  rotation={calib.rotation_deg:.2f}°  '
              f'scale={calib.scale_px_per_um:.4f} px/µm  '
              f't=({calib.tx_px:.1f}, {calib.ty_px:.1f}) px  '
              f'residual={calib.residual_um_rms:.1f} µm  '
              f'n_cells={calib.n_cells}  n_rec={calib.n_recordings}')
    return calib
