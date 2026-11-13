"""Queue for the FPS-input test: MeshFlow on exactly 4,096 farthest-point-sampled points vs its own random pick.

Usage: python scripts/make_fps_queue.py   -> outputs/fps_queue/queue.tsv (run with scripts/overnight.sh)
Variants per shape (all from existing aligned clouds; no VGGT, no TSDF):
  <s>_fps   FPS 4,096 of the TSDF cloud (<s>_tex_v3)
  <s>_uni   TSDF + Gaussian-kernel cloud (<s>_tex_v3 + <s>_gsk, 400k), MeshFlow's random pick (control for _fpsu)
  <s>_fpsu  FPS 4,096 of that union
Baseline (already done, not rerun): <s>_tex_v3, MeshFlow's random 4,096 of the 200k TSDF cloud.
Each built cloud (and the baseline) also gets scripts/meshflow_input_stats.py (what MeshFlow sees, at 32^3).
Steps: 1 dev pencilpot + set A seed 42; 2 set A seeds 7 and 123; 3 set B seed 42.
"""
from __future__ import annotations

from pathlib import Path

from make_points_queue import DEV, PY, SET_A, SET_B

VARIANTS = [("fps", ["tex_v3"], False), ("uni", ["tex_v3", "gsk"], True), ("fpsu", ["tex_v3", "gsk"], False)]


def make(step: int, s: str, seeds: list[int], build: bool) -> list[tuple]:
    jobs = []
    if build:
        jobs.append((step, f"{s}_tex_v3/mf_input", f"{PY} scripts/meshflow_input_stats.py --run {s}_tex_v3 --scene data/scenes/{s}"))
    for v, srcs, no_fps in VARIANTS:
        run = f"{s}_{v}"
        if build:
            src = " ".join(f"--src {s}_{x}" for x in srcs)
            jobs.append((step, f"{run}/build", f"{PY} scripts/fps_points.py {src} --run {run}" + (" --no-fps" if no_fps else "")))
            jobs.append((step, f"{run}/mf_input", f"{PY} scripts/meshflow_input_stats.py --run {run} --scene data/scenes/{s}"))
        for seed in seeds:
            jobs.append((step, f"{run}/mesh_s{seed}", f"{PY} scripts/mesh_eval.py --points outputs/{run}/align/points_aligned.ply "
                                                      f"--scene data/scenes/{s} --out outputs/{run}/mesh --seed {seed}"))
    return jobs


def main() -> None:
    jobs: list[tuple] = []
    for s in [DEV] + SET_A:
        jobs += make(1, s, [42], True)
    for s in SET_A:
        jobs += make(2, s, [7, 123], False)
    for s in SET_B:
        jobs += make(3, s, [42], True)
    out = Path("outputs/fps_queue")
    out.mkdir(parents=True, exist_ok=True)
    (out / "queue.tsv").write_text("".join(f"{st}\t{jid}\t{cmd}\n" for st, jid, cmd in jobs))
    meshes = sum(1 for j in jobs if "/mesh_s" in j[1])
    print(f"[queue] {len(jobs)} jobs, {meshes} MeshFlow runs -> {out / 'queue.tsv'}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
