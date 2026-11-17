"""Queue for Task D: MeshFlow on 4,096 points we choose (random seed 0 / FPS / curvature-weighted FPS), set B, seed 42.

Usage: python scripts/make_select4k_queue.py -> outputs/select4k_queue/queue.tsv (run with scripts/overnight.sh)
Baseline (not rerun): <s>_tex_v3 meshes, MeshFlow's own random 4,096 of the 200k TSDF cloud, seed 42.
"""
from __future__ import annotations

from pathlib import Path

from make_points_queue import PY, SET_B

METHODS = ["random", "fps", "wfps"]


def main() -> None:
    jobs = []
    for s in SET_B:
        for m in METHODS:
            run = f"{s}_k4{m}"
            jobs += [(1, f"{run}/select", f"{PY} scripts/select4k.py --from {s}_dense --run {run} --method {m}"),
                     (1, f"{run}/mesh_s42", f"{PY} scripts/mesh_eval.py --points outputs/{run}/align/points_aligned.ply "
                                            f"--scene data/scenes/{s} --out outputs/{run}/mesh --seed 42")]
    out = Path("outputs/select4k_queue")
    out.mkdir(parents=True, exist_ok=True)
    (out / "queue.tsv").write_text("".join(f"{st}\t{j}\t{c}\n" for st, j, c in jobs))
    print(f"[queue] {len(jobs)} jobs -> {out / 'queue.tsv'}")


if __name__ == "__main__":
    main()
