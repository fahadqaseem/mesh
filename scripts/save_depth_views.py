"""Keep only what the multi-view checks need from a VGGT raw file, in the same state the TSDF fusion sees it.

Usage: python scripts/save_depth_views.py --run <s>_craw --scene data/scenes/<s>
Reads outputs/<run>/points/raw.npz (run_points.py --keep-raw), applies the known focal and the pipeline's pixel filter
(src/photo2mesh/points/filter.py filter_pixels with the TSDF settings, exactly as fusion.py:101), and writes
outputs/<run>/points/views.npz: depth (S, H, W) float32, keep (S, H, W) = pixels the TSDF fused, fg (S, H, W) =
VGGT foreground (conf > min_conf, not eroded), extrinsic (S, 3, 4) camera-from-world, intrinsic (S, 3, 3), cam_centres,
object_size (as fusion.py:103). About 20 MB instead of ~150 MB; raw.npz can then be deleted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from photo2mesh.points.filter import filter_pixels
from photo2mesh.points.fusion import TSDF_DEFAULTS
from photo2mesh.points.vggt_points import set_known_intrinsics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/default.yaml"))["points"]
    out = Path("outputs") / args.run / "points"
    pred = dict(np.load(out / "raw.npz"))
    pred.pop("view_ids", None)
    meta = json.loads((Path(args.scene) / "gt_transform.json").read_text())
    set_known_intrinsics(pred, pred["depth"].shape[2], meta["yfov_deg"])
    t = {**TSDF_DEFAULTS, **(cfg.get("tsdf") or {})}
    keep = filter_pixels(pred, {**cfg, "mv_rel": t["mv_rel"]}).reshape(pred["depth"].shape)
    kept = pred["points"][keep]
    size = float((np.percentile(kept, 99, axis=0) - np.percentile(kept, 1, axis=0)).max())
    np.savez_compressed(out / "views.npz", depth=pred["depth"].astype(np.float32), keep=keep,
                        fg=pred["conf"] > cfg["min_conf"], extrinsic=pred["extrinsic"], intrinsic=pred["intrinsic"],
                        cam_centres=pred["cam_centres"], object_size=size)
    print(f"[views] {args.run}: {keep.sum():,} fused pixels, object size {size:.4f} -> {out / 'views.npz'}")


if __name__ == "__main__":
    main()
