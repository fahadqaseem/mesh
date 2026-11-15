"""Queue for the dense default (1M points at TSDF voxel 0.005) and the gap-filled cloud at the same density.

Usage: python scripts/make_dense_queue.py   -> outputs/dense_queue/queue.tsv (run with scripts/overnight.sh)
Per shape (dev + set A + set B), no VGGT and no fusion (the fused surface of <s>_tex_v3 is resampled):
  <s>_dense  1M points from <s>_tex_v3's TSDF surface (scripts/resample_tsdf.py; equals run_points.py's new default)
  <s>_gskd   1M Gaussian-kernel samples (same Gaussians as <s>_gsk; only its aligned cloud is kept)
  <s>_dfill  <s>_dense + <s>_gskd samples 3-10% from it (scripts/fill_points.py, the primary rule of 14 Nov)
each aligned (camera similarity) and scored (points_eval, cloud_stats).
"""
from __future__ import annotations

from pathlib import Path

from make_points_queue import DEV, PY, SET_A, SET_B


def scored(run: str, s: str) -> list[tuple]:
    return [(1, f"{run}/points_eval", f"{PY} scripts/points_eval.py --run {run} --scene data/scenes/{s}"),
            (1, f"{run}/stats", f"{PY} scripts/cloud_stats.py --run-dir outputs/{run} --scene data/scenes/{s}")]


def main() -> None:
    jobs: list[tuple] = []
    for s in [DEV] + SET_A + SET_B:
        d, k, f = f"{s}_dense", f"{s}_gskd", f"{s}_dfill"
        jobs += [(1, f"{d}/resample", f"{PY} scripts/resample_tsdf.py --from {s}_tex_v3 --run {d} --n 1000000"),
                 (1, f"{d}/align", f"{PY} scripts/align.py --run {d} --scene data/scenes/{s}")] + scored(d, s)
        jobs += [(1, f"{k}/sample", f"{PY} scripts/gaussian_kernel_points.py --from {s}_gs --run {k} --n 1000000"),
                 (1, f"{k}/align", f"{PY} scripts/align.py --run {k} --scene data/scenes/{s}"),
                 (1, f"{k}/clean", f"rm -f outputs/{k}/points/points.ply")]
        jobs += [(1, f"{f}/build", f"{PY} scripts/fill_points.py --base {d} --extra {k} --run {f} --lo 0.03 --hi 0.1")] + scored(f, s)
    out = Path("outputs/dense_queue")
    out.mkdir(parents=True, exist_ok=True)
    (out / "queue.tsv").write_text("".join(f"{st}\t{jid}\t{cmd}\n" for st, jid, cmd in jobs))
    print(f"[queue] {len(jobs)} jobs -> {out / 'queue.tsv'}")


if __name__ == "__main__":
    main()
