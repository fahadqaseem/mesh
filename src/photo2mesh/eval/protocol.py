"""Evaluation protocol v1 (docs/eval_protocol.md): one function per kind of prediction.

Both functions assume the prediction is already in the GT frame (camera alignment or MeshFlow's exact inverse
normalisation; never ICP). They normalise GT and prediction together (GT bbox centre, GT max extent = 1, as in the
MeshFlow paper) and return a flat dict of metrics.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .metrics import sample_surface

N_CD = 5_000          # samples per surface for CD / HD (MeshFlow evaluate.py default)
N_DENSE = 100_000     # samples per surface for NC / F1
TAUS = (0.01, 0.02)


def unit_cube_transform(gt: trimesh.Trimesh) -> tuple[np.ndarray, float]:
    """(centre, scale) mapping the GT into [-0.5, 0.5] by its bbox: x -> (x - centre) * scale."""
    lo, hi = gt.bounds
    return (lo + hi) / 2.0, 1.0 / float((hi - lo).max())


def chamfer_hausdorff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """MeshFlow-style Chamfer (mean L2 a->b + mean L2 b->a, not squared) and symmetric Hausdorff."""
    d_ab, _ = cKDTree(b).query(a)
    d_ba, _ = cKDTree(a).query(b)
    return float(d_ab.mean() + d_ba.mean()), float(max(d_ab.max(), d_ba.max()))


def dense_metrics(p: np.ndarray, pn: np.ndarray, g: np.ndarray, gn: np.ndarray) -> dict[str, float]:
    """NC, precision / recall / F1 at each tau, squared Chamfer (kept for HD-PEA-style comparison), normal sign agreement."""
    d_pg, i_pg = cKDTree(g).query(p)
    d_gp, i_gp = cKDTree(p).query(g)
    out = {"NC": float(0.5 * (np.abs((pn * gn[i_pg]).sum(1)).mean() + np.abs((gn * pn[i_gp]).sum(1)).mean())),
           "cd_sq": float((d_pg ** 2).mean() + (d_gp ** 2).mean()),
           "normal_outward": float(((pn * gn[i_pg]).sum(1) > 0).mean())}
    for tau in TAUS:
        prec, rec = float((d_pg < tau).mean()), float((d_gp < tau).mean())
        out[f"P@{tau}"], out[f"R@{tau}"] = prec, rec
        out[f"F1@{tau}"] = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return out


def triangle_quality(mesh: trimesh.Trimesh) -> float:
    """Mean G = 2*sqrt(3)*S / (p*h) over non-degenerate faces (HD-PEA's isotropic criterion; 1 = equilateral)."""
    tri = mesh.triangles
    e = np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2)
    area = mesh.area_faces
    p, h = e.sum(1) / 2.0, e.max(1)
    ok = (p > 0) & (h > 0)
    return float((2 * np.sqrt(3) * area[ok] / (p[ok] * h[ok])).mean()) if ok.any() else 0.0


def score_mesh(pred: trimesh.Trimesh, gt: trimesh.Trimesh) -> dict[str, float]:
    """Protocol v1 for a mesh already in the GT frame."""
    pred = trimesh.Trimesh(pred.vertices, pred.faces)          # merge duplicate vertices for honest counts
    centre, scale = unit_cube_transform(gt)
    gt_n = trimesh.Trimesh((gt.vertices - centre) * scale, gt.faces, process=False)
    pr_n = trimesh.Trimesh((pred.vertices - centre) * scale, pred.faces, process=False)
    cd, hd = chamfer_hausdorff(sample_surface(pr_n, N_CD, seed=0)[0], sample_surface(gt_n, N_CD, seed=1)[0])
    p, pn = sample_surface(pr_n, N_DENSE, seed=0)
    g, gn = sample_surface(gt_n, N_DENSE, seed=1)
    out = {"CD": 100 * cd, "HD": 100 * hd, **dense_metrics(p, pn, g, gn),
           "V": int(len(pred.vertices)), "F": int(len(pred.faces)), "G": triangle_quality(pred),
           "watertight": bool(pred.is_watertight), "size_ratio": float(pr_n.extents.max())}   # GT max extent is 1
    return out


def score_points(points: np.ndarray, normals: np.ndarray, gt: trimesh.Trimesh, seed: int = 0) -> dict[str, float]:
    """Protocol v1 for an oriented point cloud already in the GT frame (random subsamples stand in for surface samples)."""
    centre, scale = unit_cube_transform(gt)
    gt_n = trimesh.Trimesh((gt.vertices - centre) * scale, gt.faces, process=False)
    pts = (np.asarray(points, np.float64) - centre) * scale
    rng = np.random.default_rng(seed)
    i_cd = rng.choice(len(pts), min(N_CD, len(pts)), replace=False)
    i_dn = rng.choice(len(pts), min(N_DENSE, len(pts)), replace=False)
    cd, hd = chamfer_hausdorff(pts[i_cd], sample_surface(gt_n, N_CD, seed=1)[0])
    g, gn = sample_surface(gt_n, N_DENSE, seed=1)
    out = {"CD": 100 * cd, "HD": 100 * hd, **dense_metrics(pts[i_dn].astype(np.float32), np.asarray(normals, np.float32)[i_dn], g, gn),
           "V": int(len(pts))}
    return out


def perfect_scores(gt: trimesh.Trimesh) -> dict[str, float]:
    """What a perfect reconstruction scores under this protocol (two independent GT samplings): CD floor, F1 ceilings."""
    centre, scale = unit_cube_transform(gt)
    gt_n = trimesh.Trimesh((gt.vertices - centre) * scale, gt.faces, process=False)
    cd, _ = chamfer_hausdorff(sample_surface(gt_n, N_CD, seed=2)[0], sample_surface(gt_n, N_CD, seed=1)[0])
    a, an = sample_surface(gt_n, N_DENSE, seed=2)
    b, bn = sample_surface(gt_n, N_DENSE, seed=1)
    d = dense_metrics(a, an, b, bn)
    return {"CD_floor": 100 * cd, **{f"F1ceil@{t}": d[f"F1@{t}"] for t in TAUS}}
