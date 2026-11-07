"""Stage 2+3: mesh one point cloud with CVPR MeshFlow, then score it with protocol v1 (docs/eval_protocol.md).

Usage:
    python scripts/mesh_eval.py --points outputs/<run>/align/points_aligned.ply --scene data/scenes/<scene> \
        --out outputs/<run>/mesh --seed 42 [--device hybrid] [--tag meshflow]
Writes <out>/<tag>_s<seed>.ply and <out>/<tag>_s<seed>.json (timing, mesh scores, perfect-reconstruction scores).
The cloud must already be in the GT frame (scripts/align.py, camera alignment only). No ICP anywhere.
One process per run: the model is loaded, used once and freed when the process exits (Mac memory rules).
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import yaml

from photo2mesh.eval.protocol import perfect_scores, score_mesh
from photo2mesh.io import load_mesh
from photo2mesh.mesh.meshflow_cvpr import MeshFlowCVPR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, help="hybrid (Mac default) | mps | cpu | cuda")
    ap.add_argument("--tag", default="meshflow")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = dict(yaml.safe_load(open(args.config))["mesh"]["meshflow_cvpr"])
    cfg.pop("seed", None)
    if args.device:
        cfg["device"] = args.device

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.tag}_s{args.seed}"
    t0 = time.time()
    mesher = MeshFlowCVPR(**cfg)
    mesher._load()
    load_s = time.time() - t0
    mesh = mesher.run(args.points, out / f"{stem}.ply", seed=args.seed)

    gt = load_mesh(Path(args.scene) / "gt_normalized.ply")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {"points": args.points, "scene": args.scene, "seed": args.seed, "device": mesher.device,
              "load_s": load_s, **mesher.last_timing,
              "peak_rss_gb": peak / 1e9 if sys.platform == "darwin" else peak / 1e6,
              **score_mesh(mesh, gt), **perfect_scores(gt)}
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2))
    keys = ["total_s", "CD", "HD", "NC", "F1@0.01", "F1@0.02", "V", "F", "G", "size_ratio", "peak_rss_gb"]
    print(f"[mesh_eval] {stem}: " + "  ".join(f"{k}={report[k]:.4g}" if isinstance(report[k], float) else f"{k}={report[k]}" for k in keys))


if __name__ == "__main__":
    main()
