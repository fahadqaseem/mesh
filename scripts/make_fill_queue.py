"""Queue for gap filling: TSDF points + Gaussian-kernel points only where the TSDF surface is missing.

Usage: python scripts/make_fill_queue.py   -> outputs/fill_queue/queue.tsv (run with scripts/overnight.sh)
Per shape (dev + set A + set B): <s>_tfill (band 0.03-0.1, primary) and <s>_tfill2 (band 0.02-0.05, secondary),
each built (scripts/fill_points.py), scored (points_eval, cloud_stats) and, for the primary only and not on the dev
shape, meshed with MeshFlow seed 42 (secondary row). Step 1: build and score everything; step 2: MeshFlow.
"""
from __future__ import annotations

from pathlib import Path

from make_points_queue import DEV, PY, SET_A, SET_B

RULES = [("tfill", 0.03, 0.1), ("tfill2", 0.02, 0.05)]


def main() -> None:
    jobs: list[tuple] = []
    for s in [DEV] + SET_A + SET_B:
        for name, lo, hi in RULES:
            run = f"{s}_{name}"
            jobs += [(1, f"{run}/build", f"{PY} scripts/fill_points.py --base {s}_tex_v3 --extra {s}_gsk --run {run} "
                                          f"--lo {lo} --hi {hi}"),
                     (1, f"{run}/points_eval", f"{PY} scripts/points_eval.py --run {run} --scene data/scenes/{s}"),
                     (1, f"{run}/stats", f"{PY} scripts/cloud_stats.py --run-dir outputs/{run} --scene data/scenes/{s}")]
    for s in SET_A + SET_B:
        run = f"{s}_tfill"
        jobs.append((2, f"{run}/mesh_s42", f"{PY} scripts/mesh_eval.py --points outputs/{run}/align/points_aligned.ply "
                                           f"--scene data/scenes/{s} --out outputs/{run}/mesh --seed 42"))
    out = Path("outputs/fill_queue")
    out.mkdir(parents=True, exist_ok=True)
    (out / "queue.tsv").write_text("".join(f"{st}\t{jid}\t{cmd}\n" for st, jid, cmd in jobs))
    print(f"[queue] {len(jobs)} jobs, {sum('/mesh_s' in j[1] for j in jobs)} MeshFlow runs -> {out / 'queue.tsv'}")


if __name__ == "__main__":
    main()
