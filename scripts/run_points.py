"""Stage 1: photos -> filtered, oriented point cloud (.ply with xyz, normals, rgb).

Usage:
    python scripts/run_points.py --images data/scenes/bunny/images --run bunny_v1
Writes outputs/<run>/points/points.ply and cameras.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from photo2mesh.io import save_points_ply
from photo2mesh.points.filter import (confidence_mask, depth_edge_mask, multiview_consistency,
                                     remove_statistical_outliers, voxel_downsample)
from photo2mesh.points.normals import estimate_normals
from photo2mesh.points.vggt_points import flatten_views, list_images, run_vggt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override points.* config values, e.g. --set mv_rel=0.01 intrinsics=known")
    ap.add_argument("--raw-from", default=None,
                    help="with --reuse-raw: take raw.npz from this run instead (one VGGT pass, several filter variants)")
    ap.add_argument("--view-ids", default=None,
                    help="comma-separated image indices to use (default: config points.view_ids; 'all' = every image)")
    ap.add_argument("--keep-raw", action="store_true",
                    help="also save the unfiltered VGGT output as raw.npz (~115 MB per 16-view scene)")
    ap.add_argument("--reuse-raw", action="store_true",
                    help="skip VGGT and reload outputs/<run>/points/raw.npz (to re-tune filtering)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))["points"]
    for kv in args.set:
        key, value = kv.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    print("[points] config: " + json.dumps(cfg))

    out_dir = Path("outputs") / args.run / "points"
    raw_dir = Path("outputs") / (args.raw_from or args.run) / "points"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(args.images)
    ids = args.view_ids if args.view_ids is not None else cfg.get("view_ids", "all")
    view_ids = list(range(len(images))) if ids in ("all", None) else \
        [int(i) for i in (ids.split(",") if isinstance(ids, str) else ids)]
    raw_path = raw_dir / "raw.npz"
    if args.reuse_raw:
        if not raw_path.exists():
            raise SystemExit(f"--reuse-raw: {raw_path} not found (refusing to rerun VGGT silently)")
        pred = dict(np.load(raw_path))
        view_ids = [int(i) for i in pred.pop("view_ids", view_ids)]
    else:
        pred = run_vggt([images[i] for i in view_ids], use_depth_unprojection=cfg["use_depth_unprojection"])
        pred["view_ids"] = np.array(view_ids)
        if args.keep_raw:
            raw_dir.mkdir(parents=True, exist_ok=True)
            np.savez(raw_path, **pred)   # unfiltered VGGT output, so filtering can be rerun alone
        pred.pop("view_ids")
    intrinsic_pred = pred["intrinsic"].copy()          # VGGT's own estimate, kept for diagnostics
    if cfg.get("intrinsics", "predicted") == "known":
        # calibrated setting: the render focal is known (as EXIF would give for real photos); re-unproject VGGT depth
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        meta = json.loads((Path(args.images).parent / "gt_transform.json").read_text())
        k = pred["intrinsic"].copy()
        f = (meta["size"] / 2.0) / np.tan(np.deg2rad(meta["yfov_deg"]) / 2.0)
        k[:, 0, 0] = k[:, 1, 1] = f
        k[:, 0, 2] = k[:, 1, 2] = meta["size"] / 2.0
        pred["intrinsic"] = k
        pred["points"] = unproject_depth_map_to_point_map(pred["depth"][..., None], pred["extrinsic"], k).astype(np.float32)
    np.savez(out_dir / "cameras.npz", extrinsic=pred["extrinsic"], intrinsic=pred["intrinsic"],
             cam_centres=pred["cam_centres"], view_ids=np.array(view_ids), intrinsic_pred=intrinsic_pred)
    flat = flatten_views(pred)
    pts, conf, cols, views = flat["points"], flat["conf"], flat["colors"], flat["viewpoints"]

    keep = confidence_mask(conf, percentile=cfg["conf_percentile"], min_conf=cfg["min_conf"])
    if cfg.get("edge_rel", 0) or cfg.get("fg_erode", 0):
        fg = (pred["conf"] > cfg["min_conf"])
        keep &= depth_edge_mask(pred["depth"], rel_thresh=cfg.get("edge_rel", 0) or np.inf, dilate=cfg.get("edge_dilate", 1),
                                foreground=fg, erode=cfg.get("fg_erode", 0)).reshape(-1)
    if cfg.get("mv_rel", 0):
        s_, h_, w_ = pred["depth"].shape
        view_of = np.repeat(np.arange(s_), h_ * w_)
        idx = np.flatnonzero(keep)
        agree = multiview_consistency(pts[idx], view_of[idx], pred["depth"], pred["extrinsic"], pred["intrinsic"],
                                      pred["conf"] > cfg["min_conf"], rel_thresh=cfg["mv_rel"])
        keep[idx[agree < cfg.get("mv_min_views", 1)]] = False
    pts, cols, views = pts[keep], cols[keep], views[keep]
    keep = remove_statistical_outliers(pts, k=cfg["outlier_k"], std_ratio=cfg["outlier_std"])
    pts, cols, views = pts[keep], cols[keep], views[keep]
    if cfg["voxel"]:
        pts, cols, views = voxel_downsample(pts, cfg["voxel"], cols, views)

    normals = estimate_normals(pts, viewpoints=views, k=cfg["normal_k"])   # on the dense cloud: better normals
    cap = cfg.get("max_points") or 0
    if cap and len(pts) > cap:
        idx = np.sort(np.random.default_rng(0).choice(len(pts), cap, replace=False))
        pts, normals, cols = pts[idx], normals[idx], cols[idx]
    save_points_ply(out_dir / "points.ply", pts, normals=normals, colors=cols)
    print(f"[points] {len(pts):,} points -> {out_dir / 'points.ply'}")


if __name__ == "__main__":
    main()
