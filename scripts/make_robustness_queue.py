"""Write the job queue for Experiments A and B (outputs/robustness_queue/queue.tsv), most important step first.

Run with scripts/overnight.sh (one job at a time, each in its own process, memory and disk checks, resumable):
    QUEUE_DIR=outputs/robustness_queue REPORT=docs/results/2026-11-08_robustness.md \
    REPORT_CMD=scripts/robustness_report.py REPORT_EXTRA=docs/results/figures COMMIT_PREFIX="Robustness results" \
    nohup caffeinate -is scripts/overnight.sh > outputs/robustness_queue/runner.log 2>&1 &

Steps (2026-11-07 plan from Fahad; pencilpot is the dev shape and stays out of every step):
  1. Experiment A, MeshFlow robustness (route b): corrupt the clean GT samples of set A (Gaussian noise 0.25/0.5/1/2% of
     the object size, outliers 1/5/10%, missing patches 10/25%), MeshFlow seed 42. Same statistics on the real clouds.
  2. Experiment B, better points without a GPU (route a), set A: one VGGT pass per shape (kept raw), then
     relaxed points (no multi-view filter; the control), TSDF fusion of the 8 depth maps (Open3D, CPU), and the VGGT
     point head (vs depth unprojection, both with VGGT's own focal). The filtered baseline and the depth-unprojection
     variant are last night's runs: the rerun is byte-identical (checked per shape by a cmp job).
  3. Experiment A decomposition: missing 50%, coverage of the camera rig (visible from the 8 views), the real VGGT
     coverage pattern on clean points, and that pattern plus VGGT's measured noise and bias.
  4. Experiment B on set B (clean test set): relaxed points and TSDF.
Usage: python scripts/make_robustness_queue.py
"""
from __future__ import annotations

from pathlib import Path

PY = ".venv/bin/python"
SET_A = ["cylinder_472094", "shackbot_94155", "nsphere_186199", "wheel_76903", "presents_98021",
         "monsterring_1356640", "cryptex_105686"]
SET_B = [f"thingi_{i}" for i in Path("configs/step5_shapes.txt").read_text().split()]
CORE = [("noise0.25", "--noise 0.0025"), ("noise0.5", "--noise 0.005"), ("noise1", "--noise 0.01"),
        ("noise2", "--noise 0.02"), ("out1", "--outliers 0.01"), ("out5", "--outliers 0.05"),
        ("out10", "--outliers 0.10"), ("miss10", "--patches 0.10"), ("miss25", "--patches 0.25")]
TSDF_ARGS = "--set mv_rel=0 --min-views 1 --trunc-vox 6 --voxel-frac 0.005"   # chosen on pencilpot (dev) only


def decomposition(s: str) -> list[tuple[str, str]]:
    return [("miss50", "--patches 0.50"), ("vis8", "--visible"), ("vmask", f"--mask-from {s}_tex_8v"),
            ("vmask_noise", f"--mask-from {s}_tex_8v --match-noise {s}_tex_8v")]


def stats(step: int, run_dir: str, scene: str) -> tuple:
    return (step, f"{run_dir.removeprefix('outputs/')}/stats",
            f"{PY} scripts/cloud_stats.py --run-dir {run_dir} --scene data/scenes/{scene}")


def mesh(step: int, run_dir: str, scene: str) -> tuple:
    return (step, f"{run_dir.removeprefix('outputs/')}/mesh_s42",
            f"{PY} scripts/mesh_eval.py --points {run_dir}/align/points_aligned.ply --scene data/scenes/{scene} "
            f"--out {run_dir}/mesh --seed 42")


def corrupt_jobs(step: int, s: str, conds: list[tuple[str, str]]) -> list[tuple]:
    jobs = []
    for name, opts in conds:
        d = f"outputs/robustness/{s}/{name}"
        jobs += [(step, f"robustness/{s}/{name}/corrupt",
                  f"{PY} scripts/corrupt_points.py --scene data/scenes/{s} --out {d} {opts}"),
                 stats(step, d, s), mesh(step, d, s)]
    return jobs


def variant_jobs(step: int, s: str, run: str, make: str) -> list[tuple]:
    """Make a points variant, then align, score the points, measure the cloud, mesh it."""
    return [(step, f"{run}/points", make),
            (step, f"{run}/align", f"{PY} scripts/align.py --run {run} --scene data/scenes/{s}"),
            (step, f"{run}/points_eval", f"{PY} scripts/points_eval.py --run {run} --scene data/scenes/{s}"),
            stats(step, f"outputs/{run}", s), mesh(step, f"outputs/{run}", s)]


def points_jobs(step: int, s: str, point_head: bool) -> list[tuple]:
    raw, img = f"{s}_xb", f"data/scenes/{s}/images_tex"
    jobs = [(step, f"{raw}/vggt", f"{PY} scripts/run_points.py --images {img} --run {raw} --keep-raw"),
            (step, f"{raw}/same_as_overnight",
             f"cmp outputs/{raw}/points/points.ply outputs/{s}_tex_8v/points/points.ply")]
    jobs += variant_jobs(step, s, f"{s}_xb_mv0", f"{PY} scripts/run_points.py --images {img} --run {s}_xb_mv0 "
                                                 f"--reuse-raw --raw-from {raw} --set mv_rel=0")
    jobs += variant_jobs(step, s, f"{s}_xb_tsdf", f"{PY} scripts/tsdf_points.py --images {img} --raw-from {raw} "
                                                  f"--run {s}_xb_tsdf {TSDF_ARGS}")
    if point_head:
        jobs += variant_jobs(step, s, f"{s}_xb_ph", f"{PY} scripts/run_points.py --images {img} --run {s}_xb_ph "
                                                    f"--reuse-raw --raw-from {raw} "
                                                    f"--set use_depth_unprojection=false intrinsics=predicted")
        jobs.append(stats(step, f"outputs/{s}_tex_8v_vf", s))
    jobs.append((step, f"{raw}/rm_raw", f"rm -f outputs/{raw}/points/raw.npz"))
    return jobs


def main() -> None:
    jobs: list[tuple] = []
    for s in SET_A:
        jobs += [stats(1, f"outputs/{s}_tex_8v", s), stats(1, f"outputs/{s}_gtsamples", s)]
    for s in SET_A:
        jobs += corrupt_jobs(1, s, CORE)
    for s in SET_A:
        jobs += points_jobs(2, s, point_head=True)
    for s in SET_A:
        jobs += corrupt_jobs(3, s, decomposition(s))
    for s in SET_B:
        jobs += [stats(4, f"outputs/{s}_tex_8v", s), stats(4, f"outputs/{s}_gtsamples", s)]
        jobs += points_jobs(4, s, point_head=False)

    out = Path("outputs/robustness_queue")
    out.mkdir(parents=True, exist_ok=True)
    (out / "queue.tsv").write_text("".join(f"{st}\t{jid}\t{cmd}\n" for st, jid, cmd in jobs))
    counts = {st: sum(1 for j in jobs if j[0] == st) for st in sorted({j[0] for j in jobs})}
    meshes = {st: sum(1 for j in jobs if j[0] == st and j[1].endswith("mesh_s42")) for st in counts}
    print(f"[queue] {len(jobs)} jobs -> {out / 'queue.tsv'}  per step: {counts}  MeshFlow runs: {meshes}")


if __name__ == "__main__":
    main()
