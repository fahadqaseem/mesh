"""Score an aligned point cloud with protocol v1 (docs/eval_protocol.md).

Usage:
    python scripts/points_eval.py --run <run> --scene data/scenes/<scene>
Reads outputs/<run>/align/points_aligned.ply; writes outputs/<run>/eval/points_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from photo2mesh.eval.protocol import score_points
from photo2mesh.io import load_mesh, load_points_ply


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()
    run = Path("outputs") / args.run
    pts = load_points_ply(run / "align" / "points_aligned.ply")
    sim = json.loads((run / "align" / "sim3.json").read_text())
    report = {"run": args.run, "scene": args.scene, "camera_rms": sim.get("camera_rms"),
              **score_points(pts["points"], pts["normals"], load_mesh(Path(args.scene) / "gt_normalized.ply"))}
    (run / "eval").mkdir(parents=True, exist_ok=True)
    (run / "eval" / "points_v1.json").write_text(json.dumps(report, indent=2))
    print("[points_eval] " + "  ".join(f"{k}={report[k]:.4g}" for k in ["CD", "HD", "NC", "F1@0.01", "F1@0.02", "normal_outward"]))


if __name__ == "__main__":
    main()
