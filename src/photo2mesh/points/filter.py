"""Cleaning steps between raw reconstructed points and the mesher."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def confidence_mask(conf: np.ndarray, percentile: float = 50.0, min_conf: float = 1.0) -> np.ndarray:
    """Keep points above a confidence percentile (VGGT confidences are unbounded, >= 1)."""
    conf = np.asarray(conf).reshape(-1)
    thresh = max(np.percentile(conf, percentile), min_conf)
    return conf >= thresh


def remove_statistical_outliers(points: np.ndarray, k: int = 16, std_ratio: float = 2.0) -> np.ndarray:
    """Boolean mask of inliers: mean distance to k neighbours within mean + std_ratio * std."""
    k = min(k + 1, len(points))
    d, _ = cKDTree(points).query(points, k=k)
    mean_d = d[:, 1:].mean(axis=1)
    return mean_d <= mean_d.mean() + std_ratio * mean_d.std()


def voxel_downsample(points: np.ndarray, voxel: float, *extras: np.ndarray) -> tuple[np.ndarray, ...]:
    """Average points (and per-point extras like colours/normals) inside each voxel."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)

    def pool(a: np.ndarray) -> np.ndarray:
        out = np.zeros((counts.size, a.shape[1]), dtype=np.float64)
        np.add.at(out, inv, a)
        return (out / counts[:, None]).astype(np.float32)

    return (pool(points), *(pool(e) for e in extras))


def normalize_to_unit_sphere(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre on the bounding-box centre and scale into the unit sphere. Returns (points, centre, scale)."""
    centre = (points.max(axis=0) + points.min(axis=0)) / 2.0
    scale = float(np.linalg.norm(points - centre, axis=1).max()) or 1.0
    return ((points - centre) / scale).astype(np.float32), centre, scale
