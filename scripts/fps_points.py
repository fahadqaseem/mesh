"""Hand MeshFlow exactly 4,096 points chosen by farthest-point sampling, so its random pick keeps all of them.

Usage:
    python scripts/fps_points.py --src <run> [--src <run2> ...] --run <new_run> [--n 4096] [--no-fps]
Reads outputs/<src>/align/points_aligned.ply of each source (already in the GT frame), concatenates them, and
writes outputs/<new_run>/align/points_aligned.ply (+ fps.json): FPS down to n points, or with --no-fps the full
concatenation (the control: MeshFlow then picks 4,096 at random itself). FPS is deterministic (starts at the point
farthest from the centroid) and invariant to the similarity alignment, so running it on the aligned cloud picks
the same points as on the raw one.

Why exactly 4,096: MeshFlow's loader normalises the cloud (bsphere) and then keeps
torch.randperm(N)[:4096] (third_party/meshflow/meshflow/utils/mesh.py, _resample_point_cloud); with N = 4,096
that is a permutation, so every FPS point reaches the network.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from photo2mesh.io import load_points_ply, save_points_ply
from photo2mesh.points.fps import farthest_point_sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--no-fps", action="store_true")
    args = ap.parse_args()

    clouds = [load_points_ply(Path("outputs") / s / "align" / "points_aligned.ply") for s in args.src]
    pts = np.concatenate([c["points"] for c in clouds])
    nrm = np.concatenate([c["normals"] for c in clouds]) if all("normals" in c for c in clouds) else None
    t0 = time.time()
    idx = np.arange(len(pts)) if args.no_fps else farthest_point_sample(pts, args.n)
    secs = time.time() - t0
    out = Path("outputs") / args.run / "align"
    out.mkdir(parents=True, exist_ok=True)
    save_points_ply(out / "points_aligned.ply", pts[idx], None if nrm is None else nrm[idx])
    info = {"src": args.src, "src_points": [len(c["points"]) for c in clouds], "fps": not args.no_fps,
            "n": int(len(idx)), "fps_s": round(secs, 2)}
    (out / "fps.json").write_text(json.dumps(info, indent=2))
    print(f"[fps] {' + '.join(args.src)}: {len(pts):,} -> {len(idx):,} points ({secs:.1f} s) -> {out}")


if __name__ == "__main__":
    main()
