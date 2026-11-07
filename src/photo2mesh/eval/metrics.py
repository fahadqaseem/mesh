"""Mesh quality metrics, matching the ones HD-PEA reports so numbers compare directly.

All metrics sample points on both surfaces and use nearest neighbours:
    chamfer      mean squared distance both ways (lower is better)
    normal_cons  mean |cos| between matched normals (higher is better)
    f1           F1 of points within `tau` of the other surface (higher is better)
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def sample_surface(mesh: trimesh.Trimesh, n: int = 100_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample n points and their face normals from a mesh surface."""
    rng = np.random.default_rng(seed)
    areas = mesh.area_faces
    face_idx = rng.choice(len(areas), size=n, p=areas / areas.sum())
    tri = mesh.triangles[face_idx]
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])
    return pts.astype(np.float32), mesh.face_normals[face_idx].astype(np.float32)


def compare_point_sets(pred_pts: np.ndarray, pred_n: np.ndarray, gt_pts: np.ndarray, gt_n: np.ndarray,
                       tau: float = 0.01) -> dict[str, float]:
    d_pg, i_pg = cKDTree(gt_pts).query(pred_pts)
    d_gp, i_gp = cKDTree(pred_pts).query(gt_pts)
    chamfer = float((d_pg ** 2).mean() + (d_gp ** 2).mean())
    nc = float(0.5 * (np.abs((pred_n * gt_n[i_pg]).sum(1)).mean() + np.abs((gt_n * pred_n[i_gp]).sum(1)).mean()))
    precision, recall = float((d_pg < tau).mean()), float((d_gp < tau).mean())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"chamfer": chamfer, "normal_cons": nc, "f1": f1, "precision": precision, "recall": recall}


def evaluate_mesh(pred: trimesh.Trimesh, gt: trimesh.Trimesh, n: int = 100_000, tau: float = 0.01,
                  normalize: bool = True) -> dict[str, float]:
    """Compare a predicted mesh with the ground-truth mesh.

    If `normalize`, both are scaled so the GT fits the unit sphere (pred uses the GT transform),
    which makes `tau` and Chamfer comparable across shapes. Assumes the meshes are already aligned.
    """
    pred, gt = pred.copy(), gt.copy()
    if normalize:
        centre = gt.bounds.mean(axis=0)
        scale = float(np.linalg.norm(gt.vertices - centre, axis=1).max())
        for m in (pred, gt):
            m.apply_translation(-centre)
            m.apply_scale(1.0 / scale)
    p_pts, p_n = sample_surface(pred, n)
    g_pts, g_n = sample_surface(gt, n)
    out = compare_point_sets(p_pts, p_n, g_pts, g_n, tau=tau)
    out.update({"n_vertices": int(len(pred.vertices)), "n_faces": int(len(pred.faces))})
    return out


def evaluate_points(pred_pts: np.ndarray, pred_normals: np.ndarray, gt: trimesh.Trimesh, n: int = 100_000,
                    tau: float = 0.01, seed: int = 0) -> dict[str, float]:
    """Compare an (already aligned) oriented point cloud with the GT mesh surface.

    Same metrics as ``evaluate_mesh`` on n points per side (the cloud is randomly subsampled),
    using the same GT-unit-sphere normalisation. Also reports ``normal_outward``: the fraction of
    predicted normals that agree in sign with the nearest GT normal (1.0 = all oriented outward).
    """
    gt = gt.copy()
    centre = gt.bounds.mean(axis=0)
    scale = float(np.linalg.norm(gt.vertices - centre, axis=1).max())
    gt.apply_translation(-centre)
    gt.apply_scale(1.0 / scale)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pred_pts), size=min(n, len(pred_pts)), replace=False)
    p_pts = ((np.asarray(pred_pts, np.float64)[idx] - centre) / scale).astype(np.float32)
    p_n = np.asarray(pred_normals, np.float32)[idx]
    g_pts, g_n = sample_surface(gt, n, seed=seed)
    out = compare_point_sets(p_pts, p_n, g_pts, g_n, tau=tau)
    _, i_pg = cKDTree(g_pts).query(p_pts)
    out["normal_outward"] = float(((p_n * g_n[i_pg]).sum(1) > 0).mean())
    out["n_points"] = int(len(pred_pts))
    return out
