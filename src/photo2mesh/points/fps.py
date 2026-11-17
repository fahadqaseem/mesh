"""Farthest-point sampling (FPS): pick k points that spread evenly over a cloud."""
from __future__ import annotations

import numpy as np


def farthest_point_sample(points: np.ndarray, k: int, start: int | None = None,
                          weights: np.ndarray | None = None) -> np.ndarray:
    """Indices of k points chosen greedily, each the farthest from those already chosen.

    With weights (N,) > 0, each step takes the point maximising weight x distance (feature-aware FPS: points with a
    larger weight, e.g. higher curvature, are picked with less spacing).

    Deterministic: starts at `start`, or by default at the point farthest from the cloud's centroid.
    Invariant to rotation, translation and uniform scale of the cloud. O(N k) time, O(N) memory.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if k >= n:
        return np.arange(n)
    if start is None:
        start = int(np.argmax(((pts - pts.mean(0)) ** 2).sum(1)))
    w = None if weights is None else np.asarray(weights, np.float64) ** 2   # d is squared
    idx = np.empty(k, dtype=np.int64)
    idx[0] = start
    d = ((pts - pts[start]) ** 2).sum(1)
    for i in range(1, k):
        j = int(np.argmax(d if weights is None else d * w))
        idx[i] = j
        np.minimum(d, ((pts - pts[j]) ** 2).sum(1), out=d)
    return idx


def surface_variation(points: np.ndarray, k: int = 16, chunk: int = 200_000) -> np.ndarray:
    """PCA surface variation per point (Pauly et al. 2002): l_min / (l_1 + l_2 + l_3) of its k-neighbourhood covariance.

    0 on a plane, up to 1/3 for isotropic scatter; a standard curvature proxy for point clouds.
    """
    from scipy.spatial import cKDTree
    pts = np.asarray(points, np.float64)
    tree = cKDTree(pts)
    out = np.empty(len(pts))
    for s in range(0, len(pts), chunk):
        _, nb = tree.query(pts[s:s + chunk], k=k)
        q = pts[nb] - pts[nb].mean(1, keepdims=True)
        ev = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", q, q))
        out[s:s + chunk] = ev[:, 0] / np.maximum(ev.sum(1), 1e-18)
    return out
