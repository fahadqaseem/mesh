"""What MeshFlow actually sees of a point cloud: its 4,096 points, voxelised at 32^3, compared with the GT.

Usage: python scripts/meshflow_input_stats.py --run <run> --scene data/scenes/<s> [--seed 42]
Reads outputs/<run>/align/points_aligned.ply (GT frame), writes outputs/<run>/eval/meshflow_input.json.
Mimics third_party/meshflow (read only): bsphere normalisation on the full cloud, then 4,096 points
(torch.randperm after torch.manual_seed(seed): all of them when the cloud has exactly 4,096; for larger clouds
this is drawn on the CPU, while MeshFlow draws on MPS, so it is a pick like MeshFlow's, not necessarily the same one),
then voxel indices at resolution 32 as in voxelize_pc. Reports:
  cells          occupied 32^3 cells (what conditions the network)
  cell_recall    share of the cells the GT surface occupies that the input occupies too
  cell_precision share of the input's cells that the GT surface occupies
  off_surface    share of the 4,096 points more than 2% of the GT's largest extent from the GT surface
  gap_p95        95th percentile distance from GT surface samples to the nearest input point (% of extent)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from photo2mesh.io import load_points_ply

N_VERTS, RES = 4096, 32


def cells(p: np.ndarray) -> set:
    idx = np.clip(((p + 1.0) * 0.5 * RES).astype(np.int64), 0, RES - 1)
    return set(map(tuple, idx))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pts = load_points_ply(Path("outputs") / args.run / "align" / "points_aligned.ply")["points"].astype(np.float64)
    centre = (pts.min(0) + pts.max(0)) / 2
    scale = 2.0 / (np.linalg.norm(pts - centre, axis=1).max() * 2)
    torch.manual_seed(args.seed)
    pick = torch.randperm(len(pts))[:N_VERTS].numpy()
    p = pts[pick]

    gt = trimesh.load(Path(args.scene) / "gt_normalized.ply", force="mesh")
    g, _ = trimesh.sample.sample_surface(gt, 100_000, seed=0)
    ext = float(np.ptp(gt.bounds, axis=0).max())
    c_in, c_gt = cells((p - centre) * scale), cells((g - centre) * scale)
    d_pg = cKDTree(g).query(p)[0] / ext
    d_gp = cKDTree(p).query(g)[0] / ext
    out = {"points_in": int(len(pts)), "points_used": int(len(p)), "cells": len(c_in), "gt_cells": len(c_gt),
           "cell_recall": len(c_in & c_gt) / len(c_gt), "cell_precision": len(c_in & c_gt) / len(c_in),
           "off_surface": float((d_pg > 0.02).mean()), "gap_p95": float(100 * np.percentile(d_gp, 95))}
    dst = Path("outputs") / args.run / "eval"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meshflow_input.json").write_text(json.dumps(out, indent=2))
    print(f"[mf-input] {args.run}: {out['cells']} cells, recall {out['cell_recall']:.3f}, "
          f"precision {out['cell_precision']:.3f}, off-surface {out['off_surface']:.3f}, gap p95 {out['gap_p95']:.2f}%")


if __name__ == "__main__":
    main()
