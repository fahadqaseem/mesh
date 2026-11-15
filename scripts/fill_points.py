"""TSDF points plus Gaussian-kernel points only where the TSDF surface is missing (gap filling).

Usage:
    python scripts/fill_points.py --base <s>_tex_v3 --extra <s>_gsk --run <s>_tfill [--lo 0.03] [--hi 0.1] [--min-nbrs 0]
Reads outputs/<base>/align/points_aligned.ply and outputs/<extra>/align/points_aligned.ply (both in the GT frame, both
from the same VGGT pass), keeps the extra points whose distance to the nearest base point is in (lo, hi] x the base
cloud's largest extent (src/photo2mesh/points/fill.py; no GT used), and writes base + kept extra points (with normals)
to outputs/<run>/align/points_aligned.ply, plus sim3.json (copied from base, for points_eval.py) and fill.json.
Rules chosen on pencilpot only (13 Nov): primary 0.03 / 0.1 / 0 (best pilot Chamfer), secondary 0.02 / 0.05 / 0
(best pilot recall). CPU, seconds per object.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from photo2mesh.io import load_points_ply, save_points_ply
from photo2mesh.points.fill import gap_fill_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--lo", type=float, default=0.03)
    ap.add_argument("--hi", type=float, default=0.1)
    ap.add_argument("--min-nbrs", type=int, default=0)
    args = ap.parse_args()

    b = load_points_ply(Path("outputs") / args.base / "align" / "points_aligned.ply")
    e = load_points_ply(Path("outputs") / args.extra / "align" / "points_aligned.ply")
    keep = gap_fill_mask(b["points"], e["points"], args.lo, args.hi, args.min_nbrs)
    out = Path("outputs") / args.run / "align"
    out.mkdir(parents=True, exist_ok=True)
    save_points_ply(out / "points_aligned.ply", np.vstack([b["points"], e["points"][keep]]),
                    np.vstack([b["normals"], e["normals"][keep]]))
    shutil.copy(Path("outputs") / args.base / "align" / "sim3.json", out / "sim3.json")
    info = {"base": args.base, "extra": args.extra, "lo": args.lo, "hi": args.hi, "min_nbrs": args.min_nbrs,
            "base_points": int(len(b["points"])), "added": int(keep.sum()), "extra_points": int(len(e["points"]))}
    (out / "fill.json").write_text(json.dumps(info, indent=2))
    print(f"[fill] {args.base} + {keep.sum():,} of {len(keep):,} {args.extra} points -> {out}")


if __name__ == "__main__":
    main()
