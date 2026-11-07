"""Run the whole thing: photos -> points -> mesh (-> metrics if --gt given).

Usage:
    python scripts/run_pipeline.py --images data/scenes/bunny/images --run bunny_v1 [--gt data/gt/bunny.obj]
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def step(*cmd: str) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run([sys.executable, *cmd], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--mesher", default="meshflow_cvpr")
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()

    step("scripts/run_points.py", "--images", a.images, "--run", a.run, "--config", a.config)
    step("scripts/run_mesh.py", "--run", a.run, "--mesher", a.mesher, "--config", a.config)
    if a.gt:
        step("scripts/evaluate.py", "--pred", f"outputs/{a.run}/mesh/{a.mesher}.obj", "--gt", a.gt, "--run", a.run)
    print("\nDone. Log this run in docs/experiments.md")


if __name__ == "__main__":
    main()
