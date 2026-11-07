"""Noise, bias, outlier and coverage statistics of an aligned point cloud (Experiment A; see eval/robustness.py).

Usage:
    python scripts/cloud_stats.py --run-dir outputs/<run> --scene data/scenes/<s>
Reads <run-dir>/align/points_aligned.ply (GT frame); writes <run-dir>/eval/cloud_stats.json.
Uses the GT only to measure the cloud. Nothing here feeds back into the pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from photo2mesh.eval.robustness import cloud_stats
from photo2mesh.io import load_mesh, load_points_ply


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    pts = load_points_ply(run / "align" / "points_aligned.ply")["points"]
    st = cloud_stats(pts, load_mesh(Path(args.scene) / "gt_normalized.ply"))
    (run / "eval").mkdir(parents=True, exist_ok=True)
    (run / "eval" / "cloud_stats.json").write_text(json.dumps({"run_dir": str(run), "scene": args.scene, **st}, indent=2))
    print(f"[cloud_stats] {run}: " + "  ".join(f"{k}={v:.4g}" for k, v in st.items()))


if __name__ == "__main__":
    main()
