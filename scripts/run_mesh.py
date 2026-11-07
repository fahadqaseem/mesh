"""Stage 2: point cloud -> mesh with the mesher named in the config (or --mesher).

Usage:
    python scripts/run_mesh.py --run bunny_v1                 # uses outputs/bunny_v1/points/points.ply
    python scripts/run_mesh.py --run bunny_v1 --points some.ply --mesher meshflow_cvpr
Writes outputs/<run>/mesh/<mesher>.obj
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from photo2mesh.mesh.base import get_mesher


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--points", default=None)
    ap.add_argument("--mesher", default=None)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))["mesh"]

    name = args.mesher or cfg["mesher"]
    points = Path(args.points or Path("outputs") / args.run / "points" / "points.ply")
    out = Path("outputs") / args.run / "mesh" / f"{name}.obj"

    mesher = get_mesher(name, **cfg.get(name, {}))
    mesh = mesher.run(points, out)
    print(f"[mesh] {name}: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces -> {out}")


if __name__ == "__main__":
    main()
