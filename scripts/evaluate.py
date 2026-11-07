"""Stage 3: compare a predicted mesh with ground truth (Chamfer, normal consistency, F1, counts).

Usage:
    python scripts/evaluate.py --pred outputs/bunny_v1/mesh/meshflow_cvpr.obj --gt data/gt/bunny.obj --run bunny_v1
Writes outputs/<run>/eval/metrics.json

Note: VGGT reconstructs in its own frame and scale. Until alignment is added
(ICP or using rendered GT cameras), only compare meshes that share a frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from photo2mesh.eval.metrics import evaluate_mesh
from photo2mesh.io import load_mesh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--tau", type=float, default=0.01)
    args = ap.parse_args()

    metrics = evaluate_mesh(load_mesh(args.pred), load_mesh(args.gt), n=args.n, tau=args.tau)
    out = Path("outputs") / args.run / "eval" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    for k, v in metrics.items():
        print(f"{k:>12}: {v:.5f}" if isinstance(v, float) else f"{k:>12}: {v}")


if __name__ == "__main__":
    main()
