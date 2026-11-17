"""Confidence ablation, pixel level: how many depth pixels each filter setting fuses, and how accurate they are.

Usage: python scripts/conf_pixels.py --run <s>_c50 --scene data/scenes/<s>
Reads outputs/<run>/points/raw.npz (VGGT, 8 views) and outputs/<run>/align/sim3.json (camera alignment of that run; all
variants share the same VGGT output, so the same cameras). For each variant (VARIANTS, applied on top of
configs/default.yaml points.* exactly as the TSDF fusion does, src/photo2mesh/points/fusion.py:101) it back-projects
every fused pixel with the known focal, moves it into the GT frame and measures its exact distance to the GT surface
(normalised frame, GT extent 1) and its signed offset along the GT face normal (> 0 = outside). Also scores the pixels a
variant adds over the default. GT is used only for scoring. Writes outputs/<run>/eval/conf_pixels.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

from photo2mesh.eval.align import apply_sim3
from photo2mesh.eval.protocol import unit_cube_transform
from photo2mesh.io import load_mesh
from photo2mesh.points.filter import filter_pixels
from photo2mesh.points.fusion import TSDF_DEFAULTS
from photo2mesh.points.vggt_points import set_known_intrinsics

VARIANTS = {"c50": {}, "c30": {"conf_percentile": 30}, "c10": {"conf_percentile": 10},
            "c10e0": {"conf_percentile": 10, "fg_erode": 0}}


def stats(sd: np.ndarray) -> dict:
    d = np.abs(sd)
    return {"n": int(len(sd)), "out@0.01": float(100 * (d > 0.01).mean()) if len(sd) else None,
            "out@0.02": float(100 * (d > 0.02).mean()) if len(sd) else None,
            "median_abs": float(np.median(d)) if len(sd) else None,
            "median_signed": float(np.median(sd)) if len(sd) else None,
            "outside_pct": float(100 * (sd > 0).mean()) if len(sd) else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/default.yaml"))["points"]
    run = Path("outputs") / args.run
    pred = dict(np.load(run / "points" / "raw.npz"))
    pred.pop("view_ids", None)
    meta = json.loads((Path(args.scene) / "gt_transform.json").read_text())
    set_known_intrinsics(pred, pred["depth"].shape[2], meta["yfov_deg"])
    t = {**TSDF_DEFAULTS, **(cfg.get("tsdf") or {})}
    sim = json.loads((run / "align" / "sim3.json").read_text())
    gt = load_mesh(Path(args.scene) / "gt_normalized.ply")
    centre, scale = unit_cube_transform(gt)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(((gt.vertices - centre) * scale).astype(np.float32)),
                        o3d.core.Tensor(np.asarray(gt.faces, np.uint32)))
    pts_all = pred["points"].reshape(-1, 3)
    conf = pred["conf"].reshape(-1)
    fg = conf > cfg["min_conf"]
    out = {"pixels_total": int(conf.size), "fg_pixels": int(fg.sum()), "fg_frac": float(fg.mean()),
           "conf_p50_all": float(np.percentile(conf, 50)), "conf_p30_all": float(np.percentile(conf, 30)),
           "conf_p10_all": float(np.percentile(conf, 10)), "variants": {}}
    keeps = {}
    for name, over in VARIANTS.items():
        c = {**cfg, **over, "mv_rel": t["mv_rel"]}
        keeps[name] = filter_pixels(pred, c)
    sd_all = np.full(conf.size, np.nan, np.float32)
    union = np.flatnonzero(np.any(np.stack(list(keeps.values())), 0))
    q = ((apply_sim3(pts_all[union], sim["s"], np.array(sim["R"]), np.array(sim["t"])) - centre) * scale).astype(np.float32)
    r = scene.compute_closest_points(o3d.core.Tensor(q))
    sd_all[union] = ((q - r["points"].numpy()) * r["primitive_normals"].numpy()).sum(1)
    for name, keep in keeps.items():
        added = keep & ~keeps["c50"]
        out["variants"][name] = {"fused_pixels": int(keep.sum()), "fused_over_fg": float(keep.sum() / fg.sum()),
                                 "all": stats(sd_all[keep]), "added_vs_c50": stats(sd_all[added])}
    (run / "eval").mkdir(parents=True, exist_ok=True)
    (run / "eval" / "conf_pixels.json").write_text(json.dumps(out, indent=2))
    print(f"[pixels] {args.run}: fg {out['fg_frac']:.3f} of image, conf p50/p30/p10 over all pixels "
          f"{out['conf_p50_all']:.3f}/{out['conf_p30_all']:.3f}/{out['conf_p10_all']:.3f}; fused " +
          ", ".join(f"{k} {v['fused_pixels']:,} (out@0.02 {v['all']['out@0.02']:.1f}%, added {v['added_vs_c50']['n']:,})"
                    for k, v in out["variants"].items()))


if __name__ == "__main__":
    main()
