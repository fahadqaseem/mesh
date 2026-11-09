"""Write the job queue for the points-stage v3 headline rerun and the Gaussian-centres experiment.

Usage: python scripts/make_v3_queue.py {headline|gaussians}   -> outputs/v3_queue_<name>/queue.tsv
Run with scripts/overnight.sh (QUEUE_DIR=outputs/v3_queue_<name> REPORT=docs/results/2026-11-09_v3_gaussians.md
REPORT_CMD=scripts/v3_report.py REPORT_EXTRA=docs/results/figures COMMIT_PREFIX="v3 results").

headline (2026-11-09, Fahad: TSDF fusion becomes the default points stage):
  1. dev + set A: VGGT (8 views) -> TSDF (default config) -> align -> points eval -> stats -> MeshFlow seeds 42/7/123
  2. set B: the same with seed 42
gaussians: see make_gaussian_jobs (added when the msplat stage exists).
"""
from __future__ import annotations

import sys
from pathlib import Path

PY = ".venv/bin/python"
DEV = "pencilpot_299291"
SET_A = ["cylinder_472094", "shackbot_94155", "nsphere_186199", "wheel_76903", "presents_98021",
         "monsterring_1356640", "cryptex_105686"]
SET_B = [f"thingi_{i}" for i in Path("configs/step5_shapes.txt").read_text().split()]


def mesh(step: int, run: str, s: str, seed: int) -> tuple:
    return (step, f"{run}/mesh_s{seed}", f"{PY} scripts/mesh_eval.py --points outputs/{run}/align/points_aligned.ply "
                                         f"--scene data/scenes/{s} --out outputs/{run}/mesh --seed {seed}")


def headline(step: int, s: str, seeds: list[int]) -> list[tuple]:
    run = f"{s}_tex_v3"
    jobs = [(step, f"{run}/points", f"{PY} scripts/run_points.py --images data/scenes/{s}/images_tex --run {run}"),
            (step, f"{run}/same_as_tsdf_run", f"cmp outputs/{run}/points/points.ply outputs/{s}_xb_tsdf/points/points.ply")
            if s != DEV else None,
            (step, f"{run}/align", f"{PY} scripts/align.py --run {run} --scene data/scenes/{s}"),
            (step, f"{run}/points_eval", f"{PY} scripts/points_eval.py --run {run} --scene data/scenes/{s}"),
            (step, f"{run}/stats", f"{PY} scripts/cloud_stats.py --run-dir outputs/{run} --scene data/scenes/{s}")]
    return [j for j in jobs if j] + [mesh(step, run, s, seed) for seed in seeds]


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "headline"
    jobs: list[tuple] = []
    if name == "headline":
        for s in [DEV] + SET_A:
            jobs += headline(1, s, [42, 7, 123])
        for s in SET_B:
            jobs += headline(2, s, [42])
    else:
        from make_gaussian_jobs import gaussian_jobs   # noqa: E402  (scripts/ is on sys.path when run as a script)
        jobs = gaussian_jobs()
    out = Path(f"outputs/v3_queue_{name}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "queue.tsv").write_text("".join(f"{st}\t{jid}\t{cmd}\n" for st, jid, cmd in jobs))
    meshes = sum(1 for j in jobs if "/mesh_s" in j[1])
    print(f"[queue] {len(jobs)} jobs, {meshes} MeshFlow runs -> {out / 'queue.tsv'}")


if __name__ == "__main__":
    main()
