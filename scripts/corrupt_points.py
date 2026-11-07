"""Experiment A: write a corrupted copy of the clean GT samples (the upper-bound input) to test MeshFlow's robustness.

Usage:
    python scripts/corrupt_points.py --scene data/scenes/<s> --out outputs/robustness/<s>/<cond> [options]
Options, applied in this order. Sizes are fractions of the object size L (largest side of the GT bounding box):
    --visible          keep only samples that the 8 render cameras given to VGGT can see (coverage of the real rig)
    --mask-from RUN    keep only samples covered by outputs/RUN/align/points_aligned.ply (that cloud's coverage pattern,
                       same test as the coverage statistic: within COVER_DIST of its footprint on the GT surface)
    --patches F        drop a fraction F of the samples in 4 patches (equal-radius balls around well-spread centres)
    --bias B           move every sample B along its outward normal
    --noise S          isotropic Gaussian noise, standard deviation S per axis
    --match-noise RUN  take S and B from outputs/RUN/eval/cloud_stats.json (a measured VGGT cloud)
    --outliers F       replace a fraction F of the points by points uniform in the GT bounding box
Source: outputs/<s>_gtsamples/align/points_aligned.ply (200k area-uniform samples with face normals).
Writes <out>/align/points_aligned.ply and <out>/corrupt.json. Deterministic for a given --seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from photo2mesh.eval.robustness import (COVER_DIST, OUTLIER_DIST, add_noise, covered, farthest_point_centres,
                                        footprint, object_size, offset_along_normals, patch_mask,
                                        replace_with_outliers, visible_mask)
from photo2mesh.io import load_mesh, load_points_ply, save_points_ply

N_PATCHES = 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=None, help="clean cloud (default outputs/<scene name>_gtsamples/align/...)")
    ap.add_argument("--visible", action="store_true")
    ap.add_argument("--mask-from", default=None)
    ap.add_argument("--patches", type=float, default=0.0)
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--match-noise", default=None)
    ap.add_argument("--outliers", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    scene = Path(args.scene)
    src = Path(args.src or f"outputs/{scene.name}_gtsamples/align/points_aligned.ply")
    gt = load_mesh(scene / "gt_normalized.ply")
    size = object_size(gt)
    cloud = load_points_ply(src)
    pts, nrm = cloud["points"].astype(np.float64), cloud["normals"].astype(np.float64)
    rng = np.random.default_rng(args.seed)
    log = {"src": str(src), "n_src": int(len(pts)), "L": size, "seed": args.seed}

    if args.visible:
        view_ids = yaml.safe_load(open(args.config))["points"]["view_ids"]
        meta = json.loads((scene / "gt_transform.json").read_text())
        keep = visible_mask(pts, gt, np.load(scene / "gt_cam_poses.npy")[view_ids], meta["yfov_deg"])
        pts, nrm = pts[keep], nrm[keep]
        log["visible"] = {"view_ids": view_ids, "kept": float(keep.mean())}
    if args.mask_from:
        ref = load_points_ply(Path("outputs") / args.mask_from / "align" / "points_aligned.ply")["points"]
        keep = covered(pts, footprint(ref, gt, OUTLIER_DIST * size), COVER_DIST * size)
        pts, nrm = pts[keep], nrm[keep]
        log["mask_from"] = {"run": args.mask_from, "cover_dist": COVER_DIST, "kept": float(keep.mean())}
    if args.patches:
        centres = farthest_point_centres(pts, N_PATCHES, np.random.default_rng(args.seed + 1000))
        keep = patch_mask(pts, args.patches, centres)
        pts, nrm = pts[keep], nrm[keep]
        log["patches"] = {"frac": args.patches, "n_patches": N_PATCHES}
    sigma, bias = args.noise, args.bias
    if args.match_noise:
        st = json.loads((Path("outputs") / args.match_noise / "eval" / "cloud_stats.json").read_text())
        sigma, bias = st["noise_sigma"], st["bias"]
        log["match_noise"] = args.match_noise
    if bias:
        pts = offset_along_normals(pts, nrm, bias * size)
    if sigma:
        pts = add_noise(pts, sigma * size, rng)
    log.update({"noise": sigma, "bias": bias})
    if args.outliers:
        pts, nrm, _ = replace_with_outliers(pts, nrm, args.outliers, gt.bounds[0], gt.bounds[1], rng)
        log["outliers"] = args.outliers

    out = Path(args.out)
    save_points_ply(out / "align" / "points_aligned.ply", pts, normals=nrm)
    log["n_out"] = int(len(pts))
    (out / "corrupt.json").write_text(json.dumps(log, indent=2))
    print(f"[corrupt] {log['n_out']:,} of {log['n_src']:,} points -> {out}  "
          + json.dumps({k: v for k, v in log.items() if k not in ("src", "n_src", "n_out", "L", "seed")}))


if __name__ == "__main__":
    main()
