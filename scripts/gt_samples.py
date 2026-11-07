"""Upper-bound input: clean GT surface samples, written where mesh_eval.py expects an aligned cloud.

Usage:
    python scripts/gt_samples.py --scene data/scenes/<scene> --run <scene>_gtsamples [--n 200000]
Writes outputs/<run>/align/points_aligned.ply (GT frame, with face normals). This is an upper bound for the mesher,
not a pipeline result: it shows what MeshFlow does with perfect input.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from photo2mesh.eval.metrics import sample_surface
from photo2mesh.io import load_mesh, save_points_ply


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=200_000)
    args = ap.parse_args()
    pts, nrm = sample_surface(load_mesh(Path(args.scene) / "gt_normalized.ply"), args.n, seed=3)
    out = Path("outputs") / args.run / "align"
    out.mkdir(parents=True, exist_ok=True)
    save_points_ply(out / "points_aligned.ply", pts, normals=nrm)
    print(f"[gt_samples] {len(pts):,} points -> {out / 'points_aligned.ply'}")


if __name__ == "__main__":
    main()
