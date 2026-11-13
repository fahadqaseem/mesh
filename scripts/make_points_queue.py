"""Job queue for the points-first block (Prof. Zhong, 9 Nov: the point cloud is the main output).

Usage: python scripts/make_points_queue.py   -> outputs/points_queue/queue.tsv
Run:   QUEUE_DIR=outputs/points_queue REPORT=docs/results/2026-11-12_points.md REPORT_CMD=scripts/points_report.py \
       REPORT_EXTRA=docs/results/figures COMMIT_PREFIX="points results" nohup caffeinate -is scripts/overnight.sh \
       > outputs/points_queue/runner.log 2>&1 &
Steps (one heavy job at a time; the runner commits the report after each step):
  1. rescore the existing clouds (filtered v2, TSDF v3, Gaussian centres) with the points scorer, which now also
     reports F1@0.005 and all-points recall: dev + set A + set B. CPU, seconds each.
  2. Gaussian kernel sampling from the trained splats (opacity and opacity x area weighting, 200k points), points
     scores and cloud statistics; MeshFlow seed 42 on the opacity variant (secondary row).
  3. TSDF density sweep, set A: VGGT once per shape (<s>_tdraw, raw kept, checked identical to the v3 cloud), then voxel
     0.005 (current) / 0.0025 / 0.00125 of the object size (truncation 6 voxels) x 200k / 1M points. MeshFlow seed 42
     on the 200k variants. Large intermediates are deleted after scoring (raw.npz, fine fused meshes, 1M clouds).
  4. The same sweep on set B.
Settings were chosen on pencilpot (dev) only: see docs/results/2026-11-12_points.md, pilot section.
"""
from __future__ import annotations

from pathlib import Path

PY = ".venv/bin/python"
DEV = "pencilpot_299291"
SET_A = ["cylinder_472094", "shackbot_94155", "nsphere_186199", "wheel_76903", "presents_98021",
         "monsterring_1356640", "cryptex_105686"]
SET_B = [f"thingi_{i}" for i in Path("configs/step5_shapes.txt").read_text().split()]
VOXELS = [("td005", 0.005), ("td0025", 0.0025), ("td00125", 0.00125)]


def scored(step: int, run: str, s: str, mesh: bool) -> list[tuple]:
    jobs = [(step, f"{run}/align", f"{PY} scripts/align.py --run {run} --scene data/scenes/{s}"),
            (step, f"{run}/points_eval", f"{PY} scripts/points_eval.py --run {run} --scene data/scenes/{s}"),
            (step, f"{run}/stats", f"{PY} scripts/cloud_stats.py --run-dir outputs/{run} --scene data/scenes/{s}")]
    if mesh:
        jobs.append((step, f"{run}/mesh_s42", f"{PY} scripts/mesh_eval.py --points outputs/{run}/align/points_aligned.ply "
                                             f"--scene data/scenes/{s} --out outputs/{run}/mesh --seed 42"))
    return jobs


def rescore(step: int, s: str) -> list[tuple]:
    return [(step, f"{s}{suf}/points_eval_v12", f"{PY} scripts/points_eval.py --run {s}{suf} --scene data/scenes/{s}")
            for suf in ["_tex_8v", "_tex_v3", "_gs"]]


def kernels(step: int, s: str) -> list[tuple]:
    jobs = []
    for suf, w in [("_gsk", "opacity"), ("_gska", "opacity_area")]:
        jobs.append((step, f"{s}{suf}/sample", f"{PY} scripts/gaussian_kernel_points.py --from {s}_gs --run {s}{suf} "
                                               f"--weight {w}"))
        jobs += scored(step, f"{s}{suf}", s, mesh=(suf == "_gsk" and s != DEV))
    return jobs


def sweep(step: int, s: str) -> list[tuple]:
    img, raw = f"data/scenes/{s}/images_tex", f"{s}_tdraw"
    jobs = [(step, f"{raw}/vggt", f"{PY} scripts/run_points.py --images {img} --run {raw} --keep-raw"),
            (step, f"{raw}/same_as_v3", f"cmp outputs/{raw}/points/points.ply outputs/{s}_tex_v3/points/points.ply")]
    for name, vf in VOXELS:
        for n, suf in [(200_000, ""), (1_000_000, "n1m")]:
            run = f"{s}_{name}{suf}"
            jobs.append((step, f"{run}/tsdf", f"{PY} scripts/tsdf_points.py --images {img} --raw-from {raw} --run {run} "
                                              f"--voxel-frac {vf} --trunc-vox 6 --n {n}"))
            if run == f"{s}_td005":   # current voxel, 200k: must be the v3 default cloud (scored as <s>_tex_v3)
                jobs.append((step, f"{run}/same_as_v3",
                             f"cmp outputs/{run}/points/points.ply outputs/{s}_tex_v3/points/points.ply"))
                jobs.append((step, f"{run}/clean", f"rm -f outputs/{run}/points/points.ply outputs/{run}/points/tsdf_mesh.ply"))
                continue
            jobs += scored(step, run, s, mesh=(n == 200_000))
            clean = f"rm -f outputs/{run}/points/tsdf_mesh.ply"
            if n == 1_000_000:
                clean += f" outputs/{run}/points/points.ply outputs/{run}/align/points_aligned.ply"
            jobs.append((step, f"{run}/clean", clean))
    jobs.append((step, f"{raw}/clean", f"rm -f outputs/{raw}/points/raw.npz outputs/{raw}/points/points.ply "
                                       f"outputs/{raw}/points/tsdf_mesh.ply"))
    return jobs


def main() -> None:
    jobs: list[tuple] = []
    for s in [DEV] + SET_A + SET_B:
        jobs += rescore(1, s)
    for s in [DEV] + SET_A + SET_B:
        jobs += kernels(2, s)
    for s in SET_A:
        jobs += sweep(3, s)
    for s in SET_B:
        jobs += sweep(4, s)
    out = Path("outputs/points_queue")
    out.mkdir(parents=True, exist_ok=True)
    (out / "queue.tsv").write_text("".join(f"{st}\t{jid}\t{cmd}\n" for st, jid, cmd in jobs))
    meshes = sum(1 for j in jobs if j[1].endswith("/mesh_s42"))
    print(f"[queue] {len(jobs)} jobs, {meshes} MeshFlow runs -> {out / 'queue.tsv'}")


if __name__ == "__main__":
    main()
