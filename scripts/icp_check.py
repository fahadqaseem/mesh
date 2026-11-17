"""Alignment check (diagnostic only, NOT protocol v1): how much of the TSDF cloud's error is camera alignment?

Usage: python scripts/icp_check.py [--runs _dense] [--out outputs/icp_check/icp_check.json]
For each set B shape, takes the protocol v1 cloud outputs/<s><suffix>/align/points_aligned.ply (Umeyama on camera
centres, no ICP) and refines it with trimmed point-to-point ICP against 100k GT surface samples (photo2mesh.eval.align
icp_sim3, trim 0.9, fitted on a 50k-point subsample, seed 0), once rigid and once with scale. The refined transform is
applied to all points, which are rescored: protocol v1 numbers plus outliers (exact distance to the GT triangles > 0.01 /
0.02 in the normalised frame, scripts/equal_count.py surface_distance). ICP uses the GT, so this is a diagnostic split of
the error, never a pipeline step: what ICP removes is alignment error; what stays is depth / fusion error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from photo2mesh.eval.align import apply_sim3, icp_sim3
from photo2mesh.eval.metrics import sample_surface
from photo2mesh.eval.protocol import score_points, unit_cube_transform
from photo2mesh.io import load_mesh, load_points_ply

sys.path.insert(0, str(Path(__file__).parent))
from equal_count import surface_distance  # noqa: E402
from make_points_queue import SET_A, SET_B  # noqa: E402


def outliers(p: np.ndarray, gt) -> dict:
    centre, scale = unit_cube_transform(gt)
    gt_n = gt.copy()
    gt_n.vertices = (gt.vertices - centre) * scale
    d = surface_distance((p - centre) * scale, gt_n)
    from conf_report import signed_offset
    sub = np.random.default_rng(0).choice(len(p), min(100_000, len(p)), replace=False)
    return {"out@0.01": float(100 * (d > 0.01).mean()), "out@0.02": float(100 * (d > 0.02).mean()),
            "dist_median": float(np.median(d)), "signed_offset": signed_offset(((p[sub] - centre) * scale), gt_n)}


def check(shape: str, suffix: str) -> dict:
    run = Path("outputs") / f"{shape}{suffix}"
    pts = load_points_ply(run / "align" / "points_aligned.ply")
    p, nrm = pts["points"].astype(np.float64), pts["normals"]
    gt = load_mesh(Path("data/scenes") / shape / "gt_normalized.ply")
    _, scale = unit_cube_transform(gt)
    rng = np.random.default_rng(0)
    sub = p[rng.choice(len(p), min(50_000, len(p)), replace=False)]
    gt_pts, _ = sample_surface(gt, 100_000)
    res = {"shape": shape, "run": run.name, "camera": {**score_points(p, nrm, gt), **outliers(p, gt)}}
    for mode in ["rigid", "sim3"]:
        (s, rot, t), err = icp_sim3(sub, gt_pts, with_scale=mode == "sim3")
        q = apply_sim3(p, s, rot, t)
        res[mode] = {**score_points(q, (nrm @ rot.T).astype(np.float32), gt), **outliers(q, gt),
                     "step_scale": float(s),
                     "step_rot_deg": float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))),
                     "step_trans": float(np.linalg.norm(t) * scale)}   # in the normalised frame (GT extent = 1)
    print(f"[icp] {shape}: out@0.02 camera {res['camera']['out@0.02']:.1f}%  rigid {res['rigid']['out@0.02']:.1f}%  "
          f"sim3 {res['sim3']['out@0.02']:.1f}%  (rot {res['rigid']['step_rot_deg']:.2f} deg, "
          f"trans {res['rigid']['step_trans']:.4f}, scale {res['sim3']['step_scale']:.4f})", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_dense")
    ap.add_argument("--sets", default="B", help="B, or AB for set A too")
    ap.add_argument("--out", default="outputs/icp_check/icp_check.json")
    args = ap.parse_args()
    shapes = SET_B + (SET_A if "A" in args.sets else [])
    res = [check(s, args.suffix) for s in shapes]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[icp] -> {args.out}")


if __name__ == "__main__":
    main()
