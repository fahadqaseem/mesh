"""Experiment A tools: controlled corruptions of a clean point cloud, and statistics that describe a cloud's corruption.

Sizes are fractions of the object size L, the largest side of the GT bounding box. The protocol scales the GT so that
L = 1 (docs/eval_protocol.md), so noise of 0.01 means 1% of the object and compares directly with F1@0.01.

The same statistics are computed for every cloud (synthetic corruptions and real VGGT clouds), so the real clouds can be
placed on the synthetic degradation curves:
    noise_sigma   robust spread of the signed point-to-surface offsets (1.4826 * MAD). For isotropic Gaussian noise
                  of standard deviation s per axis this is s, because only the normal component moves a point off
                  the surface.
    bias          median signed offset (positive = outside the GT surface): a systematic dilation or shrinkage.
    outlier_frac  fraction of points farther than OUTLIER_DIST from the GT surface.
    coverage      fraction of the GT surface (area-uniform samples) within COVER_DIST of the cloud's footprint: its
                  inlier points projected onto the GT surface. Projecting first makes coverage measure where the
                  surface has points, independent of how noisy or offset they are (noise and bias are measured above).
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .metrics import sample_surface
from .protocol import unit_cube_transform

OUTLIER_DIST = 0.05   # farther than 5% of the object from the GT surface = outlier (VGGT noise is 0.5-2%)
COVER_DIST = 0.01     # GT sample within 1% of the object from the cloud's footprint = covered (~2x point spacing)
MAD_TO_SIGMA = 1.4826


def object_size(mesh: trimesh.Trimesh) -> float:
    """L: the largest side of the bounding box."""
    return float(mesh.extents.max())


def add_noise(points: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Isotropic Gaussian noise with standard deviation ``sigma`` per axis (same units as the points)."""
    return np.asarray(points, np.float64) + rng.normal(scale=sigma, size=np.shape(points))


def offset_along_normals(points: np.ndarray, normals: np.ndarray, offset: float) -> np.ndarray:
    """Move every point by ``offset`` along its unit normal (positive = outwards): a systematic bias."""
    return np.asarray(points, np.float64) + offset * np.asarray(normals, np.float64)


def replace_with_outliers(points: np.ndarray, normals: np.ndarray, frac: float, lo: np.ndarray, hi: np.ndarray,
                          rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace round(frac * N) random points by points uniform in the box [lo, hi], with random unit normals.

    The count stays the same, so MeshFlow's random 4096-point subset sees the same outlier fraction.
    Returns (points, normals, is_outlier).
    """
    pts, nrm = np.array(points, np.float64), np.array(normals, np.float64)
    n_out = int(round(frac * len(pts)))
    idx = rng.choice(len(pts), n_out, replace=False)
    pts[idx] = rng.uniform(lo, hi, size=(n_out, 3))
    r = rng.normal(size=(n_out, 3))
    nrm[idx] = r / np.linalg.norm(r, axis=1, keepdims=True)
    is_out = np.zeros(len(pts), bool)
    is_out[idx] = True
    return pts, nrm, is_out


def farthest_point_centres(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of k well-spread points (farthest point sampling from a random start)."""
    pts = np.asarray(points, np.float64)
    idx = [int(rng.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[idx[0]], axis=1)
    for _ in range(k - 1):
        idx.append(int(d.argmax()))
        d = np.minimum(d, np.linalg.norm(pts - pts[idx[-1]], axis=1))
    return np.array(idx)


def patch_mask(points: np.ndarray, frac: float, centres: np.ndarray) -> np.ndarray:
    """Keep-mask that removes round(frac * N) points in patches: the points closest to any of the centre points.

    Every patch is a ball of the same radius around its centre, cut out of the surface. With area-uniform samples the
    dropped fraction of points is the dropped fraction of surface area. For one set of centres the masks are nested:
    a larger frac removes everything a smaller one does.
    """
    pts = np.asarray(points, np.float64)
    d, _ = cKDTree(pts[centres]).query(pts)
    order = np.argsort(d, kind="stable")
    keep = np.ones(len(pts), bool)
    keep[order[:int(round(frac * len(pts)))]] = False
    return keep


def _raycasting_scene(mesh: trimesh.Trimesh):
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                        o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
    return scene


def visible_mask(points: np.ndarray, mesh: trimesh.Trimesh, cam_to_world: np.ndarray, yfov_deg: float,
                 eps: float | None = None) -> np.ndarray:
    """True for surface points that at least one camera sees: inside its square frustum and not occluded by the mesh.

    cam_to_world (V, 4, 4) OpenGL poses (camera looks down -z), as written by scripts/render_views.py.
    A point counts as seen when the ray from the camera centre towards it first hits the mesh at the point itself
    (within eps, default 5e-4 * L).
    """
    import open3d as o3d

    scene = _raycasting_scene(mesh)
    pts = np.asarray(points, np.float64)
    eps = 5e-4 * object_size(mesh) if eps is None else eps
    tan = np.tan(np.deg2rad(yfov_deg) / 2.0)
    seen = np.zeros(len(pts), bool)
    for pose in np.asarray(cam_to_world, np.float64):
        d = pts - pose[:3, 3]
        dist = np.linalg.norm(d, axis=1)
        local = d @ pose[:3, :3]                         # camera coordinates (x right, y up, -z forward)
        z = -local[:, 2]
        inside = (z > 0) & (np.abs(local[:, 0]) <= tan * z) & (np.abs(local[:, 1]) <= tan * z)
        rays = np.c_[np.broadcast_to(pose[:3, 3], pts.shape), d / dist[:, None]].astype(np.float32)
        t_hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        seen |= inside & (t_hit >= dist - eps)
    return seen


def closest_on_surface(points: np.ndarray, mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """(signed offsets, closest surface points). The offset is positive on the side the closest face's normal points to.

    For an outward-oriented mesh (all GT meshes here: positive volume, consistent winding) positive means outside.
    """
    import open3d as o3d

    pts = np.asarray(points, np.float32)
    q = _raycasting_scene(mesh).compute_closest_points(o3d.core.Tensor(pts))
    proj = q["points"].numpy()
    return ((pts - proj) * q["primitive_normals"].numpy()).sum(axis=1).astype(np.float64), proj.astype(np.float64)


def footprint(points: np.ndarray, mesh: trimesh.Trimesh, max_offset: float) -> np.ndarray:
    """Where on the surface a cloud has points: its points within max_offset of the mesh, projected onto it."""
    sd, proj = closest_on_surface(points, mesh)
    return proj[np.abs(sd) <= max_offset]


def covered(samples: np.ndarray, foot: np.ndarray, dist: float) -> np.ndarray:
    """True for surface samples within dist of a footprint point."""
    if len(foot) == 0:
        return np.zeros(len(samples), bool)
    d, _ = cKDTree(foot).query(samples)
    return d < dist


def cloud_stats(points: np.ndarray, gt: trimesh.Trimesh, n_gt: int = 100_000) -> dict[str, float]:
    """Noise, bias, outlier and coverage statistics of a cloud in the GT frame, in units of L (see module docstring)."""
    centre, scale = unit_cube_transform(gt)
    gt_n = trimesh.Trimesh((gt.vertices - centre) * scale, gt.faces, process=False)
    p = (np.asarray(points, np.float64) - centre) * scale
    sd, proj = closest_on_surface(p, gt_n)
    med = float(np.median(sd))
    d = np.abs(sd)
    inl = d <= OUTLIER_DIST
    g, _ = sample_surface(gt_n, n_gt, seed=1)
    dg, _ = cKDTree(p).query(g)
    return {"n_points": int(len(p)),
            "noise_sigma": float(MAD_TO_SIGMA * np.median(np.abs(sd - med))),
            "bias": med,
            "rms_inlier": float(np.sqrt((sd[inl] ** 2).mean())) if inl.any() else float("nan"),
            "outlier_frac": float((~inl).mean()),
            "coverage": float(covered(g, proj[inl], COVER_DIST).mean()),
            "coverage_raw@0.02": float((dg < 0.02).mean()),      # unprojected: mixes noise into coverage
            "coverage_raw@0.05": float((dg < 0.05).mean()),
            "d_p50": float(np.percentile(d, 50)), "d_p90": float(np.percentile(d, 90)),
            "d_p99": float(np.percentile(d, 99))}
