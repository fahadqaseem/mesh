"""Cleaning steps between raw reconstructed points and the mesher."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def confidence_mask(conf: np.ndarray, percentile: float = 50.0, min_conf: float = 1.0) -> np.ndarray:
    """Keep points at or above a confidence percentile and strictly above ``min_conf``.

    VGGT confidences are 1 + exp(x) >= 1, and background pixels sit exactly on the floor of 1.0.
    When over half the image is background, the percentile itself is 1.0, so the floor must be a
    strict bound or every background pixel survives.
    """
    conf = np.asarray(conf).reshape(-1)
    return (conf >= np.percentile(conf, percentile)) & (conf > min_conf)


def depth_edge_mask(depth: np.ndarray, rel_thresh: float = 0.02, dilate: int = 1,
                    foreground: np.ndarray | None = None, erode: int = 0) -> np.ndarray:
    """Keep-mask (S, H, W) that drops flying pixels at depth discontinuities.

    A pixel is an edge if its depth differs from any 8-neighbour by more than ``rel_thresh`` times its own depth.
    Edges are widened by ``dilate`` pixels. If ``foreground`` (S, H, W bool) is given, it is also eroded by ``erode``
    pixels, which removes the silhouette rim where depth blends object and background.
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    d = np.asarray(depth, np.float32)
    pad = np.pad(d, ((0, 0), (1, 1), (1, 1)), mode="edge")
    h, w = d.shape[1:]
    jump = np.zeros_like(d)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                jump = np.maximum(jump, np.abs(d - pad[:, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]))
    edge = jump > rel_thresh * np.maximum(np.abs(d), 1e-6)
    struct = np.zeros((1, 3, 3), bool)
    struct[0] = True
    if dilate > 0:
        edge = binary_dilation(edge, structure=struct, iterations=dilate)
    keep = ~edge
    if foreground is not None and erode > 0:
        keep &= binary_erosion(foreground, structure=struct, iterations=erode)
    return keep


def multiview_consistency(points: np.ndarray, view_of: np.ndarray, depth: np.ndarray, extrinsic: np.ndarray,
                          intrinsic: np.ndarray, valid: np.ndarray, rel_thresh: float = 0.01) -> np.ndarray:
    """Number of *other* views that agree with each point (depth-map fusion check, as in MVS).

    points (N, 3) world, view_of (N,) index of the view each point came from, depth (S, H, W), extrinsic (S, 3, 4)
    camera-from-world (OpenCV), intrinsic (S, 3, 3), valid (S, H, W) pixels that count (e.g. foreground).
    A view j agrees if the point projects inside it onto a valid pixel whose depth is within rel_thresh of the
    point's depth in view j. Uses only the reconstruction itself (no ground truth).
    """
    s, h, w = depth.shape
    count = np.zeros(len(points), np.int32)
    for j in range(s):
        cam = points @ extrinsic[j, :, :3].T + extrinsic[j, :, 3]
        z = cam[:, 2]
        front = z > 1e-6
        u = np.full(len(points), -1, np.int64)
        v = np.full(len(points), -1, np.int64)
        u[front] = np.round(intrinsic[j, 0, 0] * cam[front, 0] / z[front] + intrinsic[j, 0, 2]).astype(np.int64)
        v[front] = np.round(intrinsic[j, 1, 1] * cam[front, 1] / z[front] + intrinsic[j, 1, 2]).astype(np.int64)
        inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h) & (view_of != j)
        idx = np.flatnonzero(inside)
        dj = depth[j, v[idx], u[idx]]
        ok = valid[j, v[idx], u[idx]] & (np.abs(z[idx] - dj) < rel_thresh * dj)
        count[idx[ok]] += 1
    return count


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
