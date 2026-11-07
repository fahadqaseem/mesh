"""Bring a reconstruction (VGGT's own frame and scale) into the ground-truth frame.

Two steps:
    1. camera alignment: a similarity transform (scale, rotation, translation) fitted between
       VGGT's camera centres and the GT render camera centres (Umeyama). Needs no geometry.
    2. ICP refinement against GT surface samples (optional), to remove the small residual.

A similarity is stored as (s, R, t) and applied as  x_gt = s * R @ x + t.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (s, R, t) with dst ≈ s R src + t (Umeyama 1991). src, dst: (N, 3), N >= 3."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:   # keep a proper rotation (no reflection)
        sign[2, 2] = -1
    rot = u @ sign @ vt
    scale = float(np.trace(np.diag(d) @ sign) / (xs ** 2).sum(1).mean()) if with_scale else 1.0
    trans = mu_d - scale * rot @ mu_s
    return scale, rot, trans


def apply_sim3(points: np.ndarray, s: float, rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    """x -> s R x + t for (N, 3) points."""
    return (s * np.asarray(points, np.float64) @ np.asarray(rot).T + t).astype(np.float32)


def compose_sim3(a: tuple, b: tuple) -> tuple[float, np.ndarray, np.ndarray]:
    """Return a ∘ b (apply b first, then a)."""
    sa, ra, ta = a
    sb, rb, tb = b
    return sa * sb, ra @ rb, sa * ra @ tb + ta


def camera_centres_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    """(S, 3, 4) camera-from-world [R|t] -> (S, 3) world camera centres C = -R^T t."""
    rot, trans = extrinsic[:, :, :3], extrinsic[:, :, 3]
    return -np.einsum("sji,sj->si", rot, trans)


def align_by_cameras(pred_centres: np.ndarray, gt_c2w: np.ndarray) -> tuple[tuple, float]:
    """Similarity mapping predicted camera centres onto GT camera centres (camera order must match).

    Returns ((s, R, t), rms residual in GT units).
    """
    gt_centres = np.asarray(gt_c2w)[:, :3, 3]
    sim = umeyama(pred_centres, gt_centres)
    resid = apply_sim3(pred_centres, *sim) - gt_centres
    return sim, float(np.sqrt((resid ** 2).sum(1).mean()))


def icp_sim3(src: np.ndarray, dst: np.ndarray, iters: int = 30, trim: float = 0.9,
             with_scale: bool = True, tol: float = 1e-7) -> tuple[tuple, float]:
    """Trimmed point-to-point ICP: similarity moving ``src`` onto ``dst`` (both (N, 3)).

    Each iteration keeps the ``trim`` fraction of closest pairs, so outliers and parts of the GT
    that were never seen do not drag the fit. Start from a rough alignment (e.g. align_by_cameras).
    Returns ((s, R, t), final mean distance of kept pairs).
    """
    tree = cKDTree(dst)
    total = (1.0, np.eye(3), np.zeros(3))
    cur = np.asarray(src, np.float64)
    prev = np.inf
    err = np.inf
    for _ in range(iters):
        d, idx = tree.query(cur)
        keep = d <= np.quantile(d, trim)
        step = umeyama(cur[keep], dst[idx[keep]], with_scale=with_scale)
        cur = apply_sim3(cur, *step).astype(np.float64)
        total = compose_sim3(step, total)
        err = float(d[keep].mean())
        if abs(prev - err) < tol:
            break
        prev = err
    return total, err
