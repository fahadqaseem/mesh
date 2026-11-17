"""Fit a GT-free depth scale for one scene: the factor(s) on VGGT depth that make the views agree best.

Usage: python scripts/depth_scale.py --run <s>_ds --scene data/scenes/<s> [--eps 0.005] [--n 20000]
Reads outputs/<run>/points/raw.npz (VGGT, run_points.py --keep-raw) and only what the pipeline itself has: the VGGT
depth maps and cameras, the known focal, and the default pixel filter (the pixels the TSDF fuses, filter.py via
fusion.py:101). No GT is read. Fits one scale for all views and, from it, one per view
(src/photo2mesh/points/depth_scale.py). Writes outputs/<run>/points/depth_scale_global.json and depth_scale_view.json,
each {"k": ...} for scripts/tsdf_points.py --depth-scale.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from photo2mesh.points.depth_scale import DepthAgreement, fit_global, fit_per_view
from photo2mesh.points.filter import filter_pixels
from photo2mesh.points.fusion import TSDF_DEFAULTS
from photo2mesh.points.vggt_points import set_known_intrinsics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--scene", required=True, help="only gt_transform.json's yfov is read (the known focal)")
    ap.add_argument("--eps", type=float, default=0.005, help="truncation of the relative depth residual (0.005 chosen on pencilpot, 17 Nov)")
    ap.add_argument("--n", type=int, default=20_000, help="pixels sampled per view")
    args = ap.parse_args()
    t0 = time.time()
    cfg = yaml.safe_load(open("configs/default.yaml"))["points"]
    out = Path("outputs") / args.run / "points"
    pred = dict(np.load(out / "raw.npz"))
    pred.pop("view_ids", None)
    yfov = json.loads((Path(args.scene) / "gt_transform.json").read_text())["yfov_deg"]
    set_known_intrinsics(pred, pred["depth"].shape[2], yfov)
    t = {**TSDF_DEFAULTS, **(cfg.get("tsdf") or {})}
    keep = filter_pixels(pred, {**cfg, "mv_rel": t["mv_rel"]}).reshape(pred["depth"].shape)
    agr = DepthAgreement(pred["depth"], keep, pred["extrinsic"], pred["intrinsic"], n_per_view=args.n, eps=args.eps)
    g = fit_global(agr)
    v = fit_per_view(agr, g["k"])
    meta = {"eps": args.eps, "n_per_view": args.n, "seconds": round(time.time() - t0, 1)}
    (out / "depth_scale_global.json").write_text(json.dumps({**g, **meta}, indent=2))
    (out / "depth_scale_view.json").write_text(json.dumps({**v, "k_global": g["k"], **meta}, indent=2))
    print(f"[depth-scale] {args.run}: global k {g['k']:.4f} (cost {g['cost_at_1']:.5f} -> {g['cost']:.5f}); per view "
          + " ".join(f"{x:.3f}" for x in v["k"]) + f" (cost {v['cost']:.5f}); {meta['seconds']} s")


if __name__ == "__main__":
    main()
