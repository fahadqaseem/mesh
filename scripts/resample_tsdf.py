"""Resample an existing TSDF run's fused surface at another density (no VGGT, no fusion).

Usage: python scripts/resample_tsdf.py --from <s>_tex_v3 --run <s>_dense [--n 1000000]
Reads outputs/<from>/points/{tsdf_mesh.ply, cameras.npz, tsdf.json} and samples n points with the same sampler and
seed as the points stage (src/photo2mesh/points/fusion.py sample_with_colours, seed 0), so the result equals what
run_points.py writes with points.tsdf.n_points = n. Writes outputs/<run>/points/{points.ply, cameras.npz, tsdf.json};
align.py, points_eval.py and cloud_stats.py work on it as usual.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import trimesh

from photo2mesh.io import save_points_ply
from photo2mesh.points.fusion import sample_with_colours


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=1_000_000)
    args = ap.parse_args()
    src, out = Path("outputs") / args.src / "points", Path("outputs") / args.run / "points"
    mesh = trimesh.load(src / "tsdf_mesh.ply", process=False)
    pts, nrm, cols = sample_with_colours(mesh, args.n, seed=0)
    out.mkdir(parents=True, exist_ok=True)
    save_points_ply(out / "points.ply", pts, normals=nrm, colors=cols)
    shutil.copy(src / "cameras.npz", out / "cameras.npz")
    info = {**json.loads((src / "tsdf.json").read_text()), "n": args.n, "resampled_from": args.src}
    (out / "tsdf.json").write_text(json.dumps(info, indent=2))
    print(f"[resample] {args.src}: {len(mesh.faces):,} faces -> {args.n:,} points -> {out}")


if __name__ == "__main__":
    main()
