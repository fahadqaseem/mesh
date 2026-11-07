"""Write the overnight job queue (outputs/overnight/queue.tsv), most important step first.

Each line: step <TAB> job id <TAB> shell command. scripts/overnight.sh runs the lines one at a time, each in its own
process, and publishes the results doc after the last job of every step.

Steps (2026-11-07 plan from Fahad):
  1. 8 shapes: VGGT (8 views, default floater fixes) -> align -> points eval -> MeshFlow + eval (seed 42).
     Also the uncalibrated variant (VGGT's own focal) from the same VGGT pass.
  2. Upper bound: MeshFlow on clean GT surface samples, 8 shapes.
  3. Variance: MeshFlow seeds 7 and 123 on each step-1 cloud.
  4. Grey vs textured on 3 more shapes (+ pencilpot): grey renders, VGGT, MeshFlow seeds 42/7/123.
  5. If time remains: 12 more Thingi10K shapes (configs/step5_shapes.txt): render, then steps 1 and 2.
Usage: python scripts/make_overnight_queue.py
"""
from __future__ import annotations

from pathlib import Path

PY = ".venv/bin/python"
SHAPES = ["pencilpot_299291", "cylinder_472094", "shackbot_94155", "nsphere_186199", "wheel_76903",
          "presents_98021", "monsterring_1356640", "cryptex_105686"]
GREY_SHAPES = ["shackbot_94155", "presents_98021", "wheel_76903", "pencilpot_299291"]
SEEDS_EXTRA = [7, 123]


def mesh(run: str, scene: str, seed: int) -> str:
    return (f"{PY} scripts/mesh_eval.py --points outputs/{run}/align/points_aligned.ply "
            f"--scene data/scenes/{scene} --out outputs/{run}/mesh --seed {seed}")


def points_jobs(step: int, scene: str, variant: str, run: str, extra_vf: bool) -> list[tuple]:
    """VGGT (8 views) -> [uncalibrated variant] -> delete raw -> align -> points eval -> MeshFlow seed 42."""
    jobs = [(step, f"{run}/vggt", f"{PY} scripts/run_points.py --images data/scenes/{scene}/images_{variant} --run {run}"
                                  + (" --keep-raw" if extra_vf else ""))]
    runs = [run]
    if extra_vf:
        jobs.append((step, f"{run}_vf/points", f"{PY} scripts/run_points.py --images data/scenes/{scene}/images_{variant} "
                                               f"--run {run}_vf --reuse-raw --raw-from {run} --set intrinsics=predicted"))
        jobs.append((step, f"{run}/rm_raw", f"rm -f outputs/{run}/points/raw.npz"))
        runs.append(f"{run}_vf")
    for r in runs:
        jobs.append((step, f"{r}/align", f"{PY} scripts/align.py --run {r} --scene data/scenes/{scene}"))
        jobs.append((step, f"{r}/points_eval", f"{PY} scripts/points_eval.py --run {r} --scene data/scenes/{scene}"))
    for r in runs:
        jobs.append((step, f"{r}/mesh_s42", mesh(r, scene, 42)))
    return jobs


def gt_jobs(step: int, scene: str) -> list[tuple]:
    run = f"{scene}_gtsamples"
    return [(step, f"{run}/samples", f"{PY} scripts/gt_samples.py --scene data/scenes/{scene} --run {run}"),
            (step, f"{run}/mesh_s42", mesh(run, scene, 42))]


def main() -> None:
    jobs: list[tuple] = []
    for s in SHAPES:
        jobs += points_jobs(1, s, "tex", f"{s}_tex_8v", extra_vf=True)
    for s in SHAPES:
        jobs += gt_jobs(2, s)
    for s in SHAPES:
        for seed in SEEDS_EXTRA:
            jobs.append((3, f"{s}_tex_8v/mesh_s{seed}", mesh(f"{s}_tex_8v", s, seed)))
    for s in GREY_SHAPES:
        if s != "pencilpot_299291":   # pencilpot already has grey renders
            jobs.append((4, f"{s}/render_grey", f"{PY} scripts/render_views.py --mesh data/gt/{s}.stl "
                                                f"--out data/scenes/{s}/images_grey"))
        jobs += points_jobs(4, s, "grey", f"{s}_grey_8v", extra_vf=False)
        for seed in SEEDS_EXTRA:
            jobs.append((4, f"{s}_grey_8v/mesh_s{seed}", mesh(f"{s}_grey_8v", s, seed)))
    for fid in Path("configs/step5_shapes.txt").read_text().split():
        s = f"thingi_{fid}"
        jobs.append((5, f"{s}/render_tex", f"{PY} scripts/render_views.py --mesh data/gt/{s}.stl "
                                           f"--out data/scenes/{s}/images_tex --texture procedural"))
        jobs += points_jobs(5, s, "tex", f"{s}_tex_8v", extra_vf=False)
        jobs += gt_jobs(5, s)

    out = Path("outputs/overnight")
    out.mkdir(parents=True, exist_ok=True)
    (out / "queue.tsv").write_text("".join(f"{st}\t{jid}\t{cmd}\n" for st, jid, cmd in jobs))
    counts = {st: sum(1 for j in jobs if j[0] == st) for st in sorted({j[0] for j in jobs})}
    print(f"[queue] {len(jobs)} jobs -> {out / 'queue.tsv'}  per step: {counts}")


if __name__ == "__main__":
    main()
