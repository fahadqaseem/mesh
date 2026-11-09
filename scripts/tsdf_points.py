"""Experiment B (route a, no GPU): fuse the VGGT depth maps of a scene in a TSDF volume (Open3D, CPU), extract the
surface, and sample points from it, instead of concatenating the per-view points.

Usage:
    python scripts/tsdf_points.py --images data/scenes/<s>/images_tex --raw-from <run> --run <s>_xb_tsdf \
        [--set KEY=VALUE ...] [--voxel-frac 0.005] [--trunc-vox 6] [--min-views 1] [--n 200000]
Reads outputs/<raw-from>/points/raw.npz (VGGT output saved by scripts/run_points.py --keep-raw).
Pixels: the same per-pixel filters as run_points (configs/default.yaml points.*); override with --set, e.g. mv_rel=0 to
fuse every eroded foreground pixel and let the fusion average the noise.
Voxel = voxel_frac * object size, where the object size is the bbox of the kept points (1st to 99th percentile per axis).
Writes outputs/<run>/points/{points.ply, cameras.npz, tsdf_mesh.ply, tsdf.json}; align.py, points_eval.py and
mesh_eval.py then run unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from photo2mesh.io import save_points_ply
from photo2mesh.points.fusion import tsdf_points
from photo2mesh.points.vggt_points import set_known_intrinsics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--raw-from", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--voxel-frac", type=float, default=0.005)
    ap.add_argument("--trunc-vox", type=float, default=6.0)
    ap.add_argument("--min-views", type=int, default=1, help="extract the surface where at least this many views saw it")
    ap.add_argument("--min-part", type=float, default=0.01, help="drop pieces under this fraction of the area")
    ap.add_argument("--n", type=int, default=200_000)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))["points"]
    for kv in args.set:
        key, value = kv.split("=", 1)
        cfg[key] = yaml.safe_load(value)

    pred = dict(np.load(Path("outputs") / args.raw_from / "points" / "raw.npz"))
    view_ids = pred.pop("view_ids")
    intrinsic_pred = pred["intrinsic"].copy()
    if cfg.get("intrinsics", "predicted") == "known":
        meta = json.loads((Path(args.images).parent / "gt_transform.json").read_text())
        set_known_intrinsics(pred, meta["size"], meta["yfov_deg"])
    tcfg = {"voxel_frac": args.voxel_frac, "trunc_vox": args.trunc_vox, "min_views": args.min_views,
            "min_part": args.min_part, "mv_rel": cfg.get("mv_rel", 0.0)}
    try:
        pts, nrm, cols, mesh, tinfo = tsdf_points(pred, cfg, tcfg, args.n)
    except RuntimeError as e:
        raise SystemExit(f"[tsdf] {e}")

    out = Path("outputs") / args.run / "points"
    out.mkdir(parents=True, exist_ok=True)
    save_points_ply(out / "points.ply", pts, normals=nrm, colors=cols)
    mesh.export(out / "tsdf_mesh.ply")
    np.savez(out / "cameras.npz", extrinsic=pred["extrinsic"], intrinsic=pred["intrinsic"],
             cam_centres=pred["cam_centres"], view_ids=view_ids, intrinsic_pred=intrinsic_pred)
    info = {"raw_from": args.raw_from, "config": cfg, **tinfo}
    (out / "tsdf.json").write_text(json.dumps(info, indent=2))
    print(f"[tsdf] {info['pixels_fused']:,} pixels, voxel {info['voxel']:.5f} trunc {info['trunc']:.5f} -> "
          f"{info['faces']:,} faces ({info['faces_raw']:,} before dropping pieces) -> {len(pts):,} points in {out}")


if __name__ == "__main__":
    main()
