"""Stage 1b: move the VGGT point cloud into the ground-truth frame of a rendered test scene.

Usage:
    python scripts/align.py --run pencilpot_299291_tex_v1 --scene data/scenes/pencilpot_299291
Reads  outputs/<run>/points/{points.ply, cameras.npz}, <scene>/{gt_cam_poses.npy, gt_normalized.ply}
Writes outputs/<run>/align/{sim3.json, points_aligned.ply}

Method: Umeyama similarity on camera centres (VGGT vs GT render cameras, same order),
then optional trimmed ICP against GT surface samples. Only valid for synthetic scenes
where GT cameras exist; real photos would need ICP alone or a different reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from photo2mesh.eval.align import (align_by_cameras, apply_sim3, camera_centres_from_extrinsic,
                                   compose_sim3, icp_sim3)
from photo2mesh.eval.metrics import sample_surface
from photo2mesh.io import load_mesh, load_points_ply, save_points_ply


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--icp", choices=["none", "rigid", "sim3"], default="none",
                    help="protocol v1 uses none (camera alignment only); ICP only for diagnostics")
    ap.add_argument("--icp-points", type=int, default=50_000, help="subsample for ICP speed")
    args = ap.parse_args()

    run_dir, scene = Path("outputs") / args.run, Path(args.scene)
    pts = load_points_ply(run_dir / "points" / "points.ply")
    cams = np.load(run_dir / "points" / "cameras.npz")
    gt_c2w = np.load(scene / "gt_cam_poses.npy")
    assert len(gt_c2w) == len(cams["extrinsic"]), "camera count differs between VGGT and GT"

    sim, cam_rms = align_by_cameras(camera_centres_from_extrinsic(cams["extrinsic"]), gt_c2w)
    report = {"camera_rms": cam_rms, "camera_scale": sim[0]}
    if args.icp != "none":
        rng = np.random.default_rng(0)
        sub = pts["points"][rng.choice(len(pts["points"]), min(args.icp_points, len(pts["points"])), replace=False)]
        gt_pts, _ = sample_surface(load_mesh(scene / "gt_normalized.ply"), 100_000)
        step, icp_err = icp_sim3(apply_sim3(sub, *sim), gt_pts, with_scale=args.icp == "sim3")
        sim = compose_sim3(step, sim)
        report.update({"icp": args.icp, "icp_mean_dist": icp_err, "icp_step_scale": step[0],
                       "icp_step_rot_deg": float(np.degrees(np.arccos(np.clip((np.trace(step[1]) - 1) / 2, -1, 1)))),
                       "icp_step_trans": float(np.linalg.norm(step[2]))})

    out = run_dir / "align"
    out.mkdir(parents=True, exist_ok=True)
    s, rot, t = sim
    normals = pts.get("normals")
    if normals is not None:
        normals = (normals @ rot.T).astype(np.float32)     # rotation only; scale and translation do not apply
    save_points_ply(out / "points_aligned.ply", apply_sim3(pts["points"], *sim), normals=normals,
                    colors=pts.get("colors"))
    (out / "sim3.json").write_text(json.dumps({"s": s, "R": rot.tolist(), "t": t.tolist(), **report}, indent=2))
    print("[align] " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in report.items()))


if __name__ == "__main__":
    main()
