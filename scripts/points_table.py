"""Align + score every points run of a set of scenes and print one markdown table.

Usage:
    python scripts/points_table.py --runs pencilpot_299291_grey_v1 pencilpot_299291_tex_v1 ...
    python scripts/points_table.py --suffix tex_v1            (every scene in data/scenes)
Run names must be <scene>_<variant>_v<k>, where <scene> is a folder in data/scenes.
Writes outputs/points_table.md and outputs/points_table.json.

f1_ceiling is the F1 that a *perfect* reconstruction gets at the same n and tau (two independent
surface samplings of the GT). It shows how much of the F1 gap is sampling, not reconstruction.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from photo2mesh.eval.metrics import compare_point_sets, evaluate_points, sample_surface
from photo2mesh.io import load_mesh, load_points_ply


def scene_of(run: str, scenes: list[str]) -> str:
    matches = [s for s in scenes if run.startswith(s + "_")]
    if not matches:
        raise SystemExit(f"no scene in data/scenes matches run {run}")
    return max(matches, key=len)


def f1_ceiling(gt, n: int, tau: float) -> float:
    a, an = sample_surface(gt, n, seed=1)
    b, bn = sample_surface(gt, n, seed=2)
    return compare_point_sets(a, an, b, bn, tau=tau)["f1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--suffix", help="score <scene>_<suffix> for every scene in data/scenes")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--tau", type=float, default=0.01)
    args = ap.parse_args()

    scenes = sorted(p.name for p in Path("data/scenes").iterdir() if p.is_dir())
    runs = list(args.runs) + ([f"{s}_{args.suffix}" for s in scenes] if args.suffix else [])
    rows = []
    for run in runs:
        scene = scene_of(run, scenes)
        if not (Path("outputs") / run / "points" / "points.ply").exists():
            print(f"[table] skip {run}: no points.ply")
            continue
        subprocess.run([sys.executable, "scripts/align.py", "--run", run, "--scene", f"data/scenes/{scene}"], check=True)
        sim = json.loads((Path("outputs") / run / "align" / "sim3.json").read_text())
        pts = load_points_ply(Path("outputs") / run / "align" / "points_aligned.ply")
        gt = load_mesh(Path("data/scenes") / scene / "gt_normalized.ply")
        m = evaluate_points(pts["points"], pts["normals"], gt, n=args.n, tau=args.tau)
        m.update(run=run, f1_ceiling=f1_ceiling(gt, args.n, args.tau), camera_rms=sim["camera_rms"])
        (Path("outputs") / run / "eval").mkdir(parents=True, exist_ok=True)
        (Path("outputs") / run / "eval" / "points_metrics.json").write_text(json.dumps(m, indent=2))
        rows.append(m)

    cols = ["chamfer", "normal_cons", "normal_outward", "precision", "recall", "f1", "f1_ceiling", "camera_rms"]
    lines = [f"Points stage vs GT (n={args.n}, tau={args.tau}, GT in unit sphere)", "",
             "| run | #points | " + " | ".join(cols) + " |", "|---|---:|" + "---:|" * len(cols)]
    for m in rows:
        vals = [f"{m[c]:.2e}" if c == "chamfer" else f"{m[c]:.3f}" for c in cols]
        lines.append(f"| {m['run']} | {m['n_points']:,} | " + " | ".join(vals) + " |")
    table = "\n".join(lines)
    Path("outputs/points_table.md").write_text(table + "\n")
    Path("outputs/points_table.json").write_text(json.dumps(rows, indent=2))
    print(table)


if __name__ == "__main__":
    main()
