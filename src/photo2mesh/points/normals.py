"""Normal estimation for point clouds, oriented with known camera centres.

VGGT gives points and cameras but no normals. We estimate normals with local PCA and
flip each one to face the camera that saw the point, which gives consistent outward
normals without a global orientation step.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def pca_normals(points: np.ndarray, k: int = 20) -> np.ndarray:
    """Unoriented unit normals from the smallest principal axis of each k-neighbourhood."""
    pts = np.asarray(points, dtype=np.float64)
    k = min(k, len(pts))
    _, idx = cKDTree(pts).query(pts, k=k)
    nbrs = pts[idx]                                   # (N, k, 3)
    centred = nbrs - nbrs.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centred, centred) / k
    _, vecs = np.linalg.eigh(cov)                     # ascending eigenvalues
    normals = vecs[:, :, 0]
    return (normals / np.linalg.norm(normals, axis=1, keepdims=True)).astype(np.float32)


def orient_towards(normals: np.ndarray, points: np.ndarray, viewpoints: np.ndarray) -> np.ndarray:
    """Flip normals so each faces its viewpoint (the camera centre that observed the point).

    Args:
        viewpoints: (N, 3) per-point camera centre, or (3,) one shared viewpoint.
    """
    view = np.broadcast_to(np.asarray(viewpoints, dtype=np.float32), points.shape)
    to_view = view - points
    flip = np.einsum("ij,ij->i", normals, to_view) < 0
    out = normals.copy()
    out[flip] *= -1
    return out


def estimate_normals(points: np.ndarray, viewpoints: np.ndarray | None = None, k: int = 20) -> np.ndarray:
    """PCA normals, oriented to the viewpoints if given, otherwise away from the centroid."""
    normals = pca_normals(points, k=k)
    if viewpoints is None:
        centroid = points.mean(axis=0)
        # outward from the centroid: flip those pointing inward
        inward = np.einsum("ij,ij->i", normals, points - centroid) < 0
        normals[inward] *= -1
        return normals
    return orient_towards(normals, points, viewpoints)
