"""Stage 3: compare a predicted mesh with ground truth (Chamfer, normal consistency, F1, counts).

Usage:
    python scripts/evaluate.py --pred outputs/<run>/mesh/meshflow_cvpr.obj --gt <scene>/gt_normalized.ply --run <run>
    python scripts/evaluate.py --pred-points outputs/<run>/align/points_aligned.ply --gt <scene>/gt_normalized.ply --run <run>
Writes outputs/<run>/eval/metrics.json (mesh) or outputs/<run>/eval/points_metrics.json (points)

Prediction and GT must share a frame: run scripts/align.py first and use the scene's
gt_normalized.ply (the frame the render cameras live in), not the raw mesh.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from photo2mesh.eval.metrics import evaluate_mesh, evaluate_points
from photo2mesh.io import load_mesh, load_points_ply


def main() -> None:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--pred", help="predicted mesh")
    group.add_argument("--pred-points", help="predicted oriented point cloud (.ply with normals)")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--tau", type=float, default=0.01)
    args = ap.parse_args()

    if args.pred:
        metrics = evaluate_mesh(load_mesh(args.pred), load_mesh(args.gt), n=args.n, tau=args.tau)
        out = Path("outputs") / args.run / "eval" / "metrics.json"
    else:
        pts = load_points_ply(args.pred_points)
        metrics = evaluate_points(pts["points"], pts["normals"], load_mesh(args.gt), n=args.n, tau=args.tau)
        out = Path("outputs") / args.run / "eval" / "points_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    for k, v in metrics.items():
        print(f"{k:>12}: {v:.5f}" if isinstance(v, float) else f"{k:>12}: {v}")


if __name__ == "__main__":
    main()
