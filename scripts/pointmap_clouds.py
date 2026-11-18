"""VGGT point map vs unprojected depth vs TSDF (Prof. Zhong, 17 Nov, item 1): build the two unfused clouds.

Usage: python scripts/pointmap_clouds.py --run <s>_pm --scene data/scenes/<s>
Reads outputs/<run>/points/raw.npz (run_points.py --keep-raw, 8 default views) and writes two runs next to it, each with
points/{points.ply, cameras.npz} for scripts/align.py:
  <s>_pmap: VGGT point head world_points (its own world frame = the frame of VGGT's cameras)
  <s>_dunp: VGGT depth maps unprojected with the known focal (exactly the points the TSDF fuses, before fusion)
Both keep the same pixels: the default TSDF pixel filter (filter.py filter_pixels with points.tsdf.mv_rel = 0, i.e.
conf > 1 and the 2 px foreground erosion), computed once from the depth head with the known focal. Normals: PCA on the
kept points (k = points.normal_k), flipped towards the camera that saw each point (normals.py), as in the v2 points
stage. Every kept pixel is written; scoring draws exactly 1M points from them (scripts/pm_report.py).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from photo2mesh.io import save_points_ply
from photo2mesh.points.filter import filter_pixels
from photo2mesh.points.fusion import TSDF_DEFAULTS
from photo2mesh.points.normals import estimate_normals
from photo2mesh.points.vggt_points import set_known_intrinsics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/default.yaml"))["points"]
    src = Path("outputs") / args.run / "points"
    pred = dict(np.load(src / "raw.npz"))
    pred.pop("view_ids", None)
    yfov = json.loads((Path(args.scene) / "gt_transform.json").read_text())["yfov_deg"]
    set_known_intrinsics(pred, pred["depth"].shape[2], yfov)
    t = {**TSDF_DEFAULTS, **(cfg.get("tsdf") or {})}
    keep = filter_pixels(pred, {**cfg, "mv_rel": t["mv_rel"]})
    s, h, w = pred["depth"].shape
    view = np.repeat(np.arange(s), h * w)[keep]
    cols = (pred["colors"].reshape(-1, 3)[keep] * 255).clip(0, 255).astype(np.uint8)
    base = args.run.rsplit("_", 1)[0]
    info = {"pixels_kept": int(keep.sum())}
    for suf, key in [("_pmap", "world_points"), ("_dunp", "points")]:
        p = pred[key].reshape(-1, 3)[keep].astype(np.float32)
        nrm = estimate_normals(p, viewpoints=pred["cam_centres"][view], k=cfg["normal_k"])
        out = Path("outputs") / f"{base}{suf}" / "points"
        out.mkdir(parents=True, exist_ok=True)
        save_points_ply(out / "points.ply", p, normals=nrm, colors=cols)
        shutil.copy(src / "cameras.npz", out / "cameras.npz")
        (out / "pointmap.json").write_text(json.dumps({"raw_from": args.run, "source": key, **info}, indent=2))
    # how far the point head is from the known-focal unprojection, pixel by pixel (VGGT frame, relative to depth)
    d = np.linalg.norm(pred["world_points"].reshape(-1, 3)[keep] - pred["points"].reshape(-1, 3)[keep], axis=1)
    z = pred["depth"].reshape(-1)[keep]
    info["pmap_vs_dunp_rel_median"] = float(np.median(d / z))
    (Path("outputs") / f"{base}_pmap" / "points" / "pointmap.json").write_text(
        json.dumps({"raw_from": args.run, "source": "world_points", **info}, indent=2))
    print(f"[pointmap] {base}: {info['pixels_kept']:,} pixels; point head vs unprojected depth median "
          f"{100 * info['pmap_vs_dunp_rel_median']:.2f}% of depth")


if __name__ == "__main__":
    main()
