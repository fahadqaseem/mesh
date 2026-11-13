"""Gaussian kernel sampling: points drawn from each trained Gaussian's covariance, not just its centre.

Usage:
    python scripts/gaussian_kernel_points.py --from <s>_gs --run <s>_gsk [--weight opacity|opacity_area] [--n 200000]
Reads outputs/<from>/gaussians/splat.ply (trained by scripts/gaussian_points.py; the same splats as the centres run,
so the two are directly comparable) and outputs/<from>/points/{cameras.npz, gaussians.json}.
Same Gaussian filter as the centres (opacity >= 0.5, largest axis <= 5% of the object). Each point picks a Gaussian
with probability proportional to its opacity (or opacity x footprint area), then is drawn from N(mu, R S^2 R^T)
truncated at 2 sigma per axis. Normals: the PCA normal of the source Gaussian's centre (as in the centres run), oriented towards the nearest camera.
Writes outputs/<run>/points/{points.ply, cameras.npz, kernels.json}. CPU only, seconds per object.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from photo2mesh.io import save_points_ply
from photo2mesh.points.gaussians import read_ply_vertices, sample_kernels
from photo2mesh.points.normals import estimate_normals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--weight", choices=["opacity", "opacity_area"], default="opacity")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--trunc", type=float, default=2.0)
    ap.add_argument("--normals", choices=["centre", "pca", "axis"], default="centre",
                    help="centre (default): the PCA normal of the source Gaussian's centre, as in the centres run; "
                         "pca: PCA on the samples; axis: each Gaussian's shortest axis "
                         "(pencilpot NC: axis 0.69, pca 0.66; chosen there)")
    args = ap.parse_args()

    src = Path("outputs") / args.src
    g = json.loads((src / "points" / "gaussians.json").read_text())
    cams = np.load(src / "points" / "cameras.npz")
    t0 = time.time()
    v = read_ply_vertices(src / "gaussians" / "splat.ply")
    pts, nrm, pick = sample_kernels(v, args.n, g["min_opacity"], g["max_scale"] * g["object_size"], args.weight,
                                    args.trunc, seed=0)
    _, nearest = cKDTree(cams["cam_centres"]).query(pts)
    if args.normals == "pca":
        nrm = estimate_normals(pts.astype(np.float32), viewpoints=cams["cam_centres"][nearest]).astype(np.float64)
    elif args.normals == "centre":
        ids = np.unique(pick)
        c = np.stack([v["x"][ids], v["y"][ids], v["z"][ids]], 1).astype(np.float32)
        _, near_c = cKDTree(cams["cam_centres"]).query(c)
        cn = estimate_normals(c, viewpoints=cams["cam_centres"][near_c]).astype(np.float64)
        nrm = cn[np.searchsorted(ids, pick)]
    flip = ((cams["cam_centres"][nearest] - pts) * nrm).sum(1) < 0
    nrm[flip] *= -1
    dt = time.time() - t0
    out = Path("outputs") / args.run / "points"
    out.mkdir(parents=True, exist_ok=True)
    save_points_ply(out / "points.ply", pts.astype(np.float32), normals=nrm.astype(np.float32))
    shutil.copy(src / "points" / "cameras.npz", out / "cameras.npz")
    info = {"from": args.src, "weight": args.weight, "normals": args.normals, "n": int(len(pts)), "trunc": args.trunc,
            "gaussians_used": int(len(np.unique(pick))), "sample_s": dt, "train_s": g["train_s"],
            "min_opacity": g["min_opacity"], "max_scale": g["max_scale"]}
    (out / "kernels.json").write_text(json.dumps(info, indent=2))
    print(f"[kernels] {len(pts):,} points from {info['gaussians_used']:,} Gaussians ({args.weight}) in {dt:.1f} s -> {out}")


if __name__ == "__main__":
    main()
