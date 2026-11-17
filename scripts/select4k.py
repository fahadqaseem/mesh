"""Task D: choose exactly 4,096 points for MeshFlow ourselves (so its own random pick keeps all of them).

Usage: python scripts/select4k.py --from <s>_dense --run <s>_k4<method> --method random|fps|wfps [--lam 2.0] [--seed 0]
Reads outputs/<from>/align/points_aligned.ply (the dense TSDF cloud, GT frame); writes outputs/<run>/align/
{points_aligned.ply, select.json}. Methods:
  random  uniform without replacement, numpy default_rng(seed) (seed 0 for every shape and method: a fixed rule, not
          MeshFlow's torch RNG)
  fps     farthest-point sampling (src/photo2mesh/points/fps.py; deterministic start: farthest from the centroid)
  wfps    feature-aware FPS: each step maximises distance x (1 + lam x c / mean(c)), c = PCA surface variation over 16
          neighbours (curvature proxy, fps.py surface_variation). lam = 2 fixed in advance, not tuned.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from photo2mesh.io import load_points_ply, save_points_ply
from photo2mesh.points.fps import farthest_point_sample, surface_variation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--method", choices=["random", "fps", "wfps"], required=True)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    c = load_points_ply(Path("outputs") / args.src / "align" / "points_aligned.ply")
    pts = c["points"]
    t0 = time.time()
    if args.method == "random":
        idx = np.random.default_rng(args.seed).choice(len(pts), args.n, replace=False)
    elif args.method == "fps":
        idx = farthest_point_sample(pts, args.n)
    else:
        cv = surface_variation(pts)
        idx = farthest_point_sample(pts, args.n, weights=1.0 + args.lam * cv / cv.mean())
    out = Path("outputs") / args.run / "align"
    out.mkdir(parents=True, exist_ok=True)
    save_points_ply(out / "points_aligned.ply", pts[idx], c.get("normals")[idx] if c.get("normals") is not None else None)
    (out / "select.json").write_text(json.dumps({"src": args.src, "method": args.method, "n": int(len(idx)),
                                                 "lam": args.lam, "seed": args.seed, "secs": round(time.time() - t0, 1)}, indent=2))
    print(f"[select4k] {args.src} -> {args.method} {len(idx):,} points ({time.time() - t0:.0f} s) -> {out}")


if __name__ == "__main__":
    main()
