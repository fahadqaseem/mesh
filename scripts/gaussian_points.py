"""Gaussian centres as the point input (decision 5): VGGT scene -> COLMAP -> 3DGS with msplat (Metal) -> centres.

Usage:
    python scripts/gaussian_points.py --images data/scenes/<s>/images_tex --from <s>_tex_v3 --run <s>_gs \
        [--iters 7000] [--init-n 100000] [--min-opacity 0.5] [--max-scale 0.05]
Reads outputs/<from>/points/{cameras.npz, points.ply}: VGGT's cameras (known focal) and the v3 (TSDF) cloud, which is
subsampled to the initial cloud. Trains in VGGT's frame (keep_crs, white background like the renders).
Writes outputs/<run>/colmap/ (images + sparse/0/*.bin), outputs/<run>/gaussians/splat.ply and
outputs/<run>/points/{points.ply, cameras.npz, gaussians.json}; align.py, points_eval.py and mesh_eval.py run unchanged.
One heavy job: one object per process (Mac memory rules).
"""
from __future__ import annotations

import argparse
import json
import resource
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from photo2mesh.io import load_points_ply, save_points_ply
from photo2mesh.points.gaussians import gaussian_centres, read_ply_vertices, write_colmap
from photo2mesh.points.normals import estimate_normals
from photo2mesh.points.vggt_points import list_images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--init-n", type=int, default=100_000, help="initial points from the v3 cloud (chosen on pencilpot)")
    ap.add_argument("--min-opacity", type=float, default=0.5)
    ap.add_argument("--max-scale", type=float, default=0.05, help="drop Gaussians larger than this fraction of the object")
    ap.add_argument("--densify-grad", type=float, default=0.0002, help="msplat densification gradient threshold")
    ap.add_argument("--n", type=int, default=200_000, help="cap on output points")
    args = ap.parse_args()

    import msplat

    src, run = Path("outputs") / args.src / "points", Path("outputs") / args.run
    cams = np.load(src / "cameras.npz")
    init = load_points_ply(src / "points.ply")
    images = list_images(args.images)
    names = [Path(images[i]).name for i in cams["view_ids"]]

    # 1. COLMAP scene (VGGT frame): the 8 images, one PINHOLE camera each, a sparse initial cloud
    col = run / "colmap"
    if col.exists():
        shutil.rmtree(col)
    (col / "images").mkdir(parents=True)
    for i, name in zip(cams["view_ids"], names):
        shutil.copy(images[i], col / "images" / name)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(init["points"]), min(args.init_n, len(init["points"])), replace=False)
    from PIL import Image
    w, h = Image.open(images[0]).size
    write_colmap(col / "sparse" / "0", names, cams["extrinsic"], cams["intrinsic"], (w, h), init["points"][idx],
                 init["colors"][idx] if "colors" in init else np.full((len(idx), 3), 0.5))

    # 2. train 3DGS on the Metal GPU
    (run / "gaussians").mkdir(parents=True, exist_ok=True)
    splat = run / "gaussians" / "splat.ply"
    cfg = msplat.TrainingConfig(iterations=args.iters, keep_crs=True, bg_color=[1.0, 1.0, 1.0], output=str(splat),
                                 densify_grad_thresh=args.densify_grad)
    t0 = time.time()
    trainer = msplat.GaussianTrainer(msplat.load_dataset(str(col)), cfg)
    trainer.train(lambda *a, **k: None, 1000)
    train_s = time.time() - t0
    trainer.export_ply(str(splat))
    n_gauss = int(trainer.splat_count)
    del trainer

    # 3. centres of the opaque, compact Gaussians -> points
    size = float((np.percentile(init["points"], 99, axis=0) - np.percentile(init["points"], 1, axis=0)).max())
    v = read_ply_vertices(splat)
    pts, opacity, scale = gaussian_centres(v, args.min_opacity, args.max_scale * size)
    if len(pts) > args.n:
        keep = np.sort(np.random.default_rng(0).choice(len(pts), args.n, replace=False))
        pts = pts[keep]
    # normals: local PCA, oriented towards the nearest camera (MeshFlow ignores them; NC uses them)
    _, nearest = cKDTree(cams["cam_centres"]).query(pts)
    nrm = estimate_normals(pts.astype(np.float32), viewpoints=cams["cam_centres"][nearest])
    out = run / "points"
    out.mkdir(parents=True, exist_ok=True)
    save_points_ply(out / "points.ply", pts, normals=nrm)
    shutil.copy(src / "cameras.npz", out / "cameras.npz")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    info = {"from": args.src, "iters": args.iters, "densify_grad": args.densify_grad, "init_n": int(len(idx)), "train_s": train_s, "gaussians": n_gauss,
            "kept": int(len(pts)), "min_opacity": args.min_opacity, "max_scale": args.max_scale, "object_size": size,
            "peak_rss_gb": peak}
    (out / "gaussians.json").write_text(json.dumps(info, indent=2))
    print(f"[gaussians] {n_gauss:,} Gaussians in {train_s:.0f} s ({args.iters} iters), {len(pts):,} centres kept "
          f"-> {out / 'points.ply'}  peak {peak:.1f} GB")


if __name__ == "__main__":
    main()
