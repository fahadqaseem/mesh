"""Stage 1: photos -> filtered, oriented point cloud (.ply with xyz, normals, rgb).

Usage:
    python scripts/run_points.py --images data/scenes/bunny/images --run bunny_v1
Writes outputs/<run>/points/points.ply and cameras.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from photo2mesh.io import save_points_ply
from photo2mesh.points.filter import confidence_mask, remove_statistical_outliers, voxel_downsample
from photo2mesh.points.normals import estimate_normals
from photo2mesh.points.vggt_points import flatten_views, list_images, run_vggt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))["points"]

    out_dir = Path("outputs") / args.run / "points"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = run_vggt(list_images(args.images), use_depth_unprojection=cfg["use_depth_unprojection"])
    np.savez(out_dir / "cameras.npz", extrinsic=pred["extrinsic"], intrinsic=pred["intrinsic"],
             cam_centres=pred["cam_centres"])
    flat = flatten_views(pred)
    pts, conf, cols, views = flat["points"], flat["conf"], flat["colors"], flat["viewpoints"]

    keep = confidence_mask(conf, percentile=cfg["conf_percentile"], min_conf=cfg["min_conf"])
    pts, cols, views = pts[keep], cols[keep], views[keep]
    keep = remove_statistical_outliers(pts, k=cfg["outlier_k"], std_ratio=cfg["outlier_std"])
    pts, cols, views = pts[keep], cols[keep], views[keep]
    if cfg["voxel"]:
        pts, cols, views = voxel_downsample(pts, cfg["voxel"], cols, views)

    normals = estimate_normals(pts, viewpoints=views, k=cfg["normal_k"])
    save_points_ply(out_dir / "points.ply", pts, normals=normals, colors=cols)
    print(f"[points] {len(pts):,} points -> {out_dir / 'points.ply'}")


if __name__ == "__main__":
    main()
