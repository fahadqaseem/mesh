"""Time CVPR MeshFlow on one point cloud and one device, phase by phase (load, sample, decode+extract).

Usage:
    python scripts/bench_meshflow.py --points outputs/<run>/align/points_aligned.ply --device mps --out outputs/<run>/mesh_bench
    add --check-cfg to also run guidance_scale=2.5 (the config default) and compare with 1.0
Writes <out>/meshflow_<device>.ply and <out>/bench_<device>.json. Run one device per process so peak memory is per device.

Why guidance_scale=1.0: without a reference image, MeshFlow's CFG compares two identical empty
conditions, so 2.5 only doubles the batch. --check-cfg measures whether the result really matches.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "meshflow"))


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def generate(pipe, points: str, device: str, steps: int, seed: int, guidance: float) -> tuple[dict, object, torch.Tensor]:
    """Mirror MeshFlowPipeline.run() with timing per phase."""
    t = {}
    torch.manual_seed(seed)                      # MeshFlow's point resampling uses the global torch RNG
    t0 = time.time()
    surface = pipe.sample_surface_points(points)
    rope = pipe.get_rope_cond(surface)
    visual = pipe.get_visual_cond(None, guidance_scale=guidance)
    sync(device)
    t["prepare_s"] = time.time() - t0
    t0 = time.time()
    latents = pipe.sample_latent(visual, rope, steps=steps, guidance_scale=guidance, seed=seed, disable_prog=True)
    sync(device)
    t["sample_s"] = time.time() - t0
    t["sample_per_step_s"] = t["sample_s"] / steps
    t0 = time.time()
    mesh = pipe.decode_latent(latents)
    sync(device)
    t["decode_extract_s"] = time.time() - t0
    return t, mesh, latents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--device", required=True, choices=["cpu", "mps", "cuda"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="ckpt/meshflow/meshflow")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--check-cfg", action="store_true")
    args = ap.parse_args()

    from meshflow.pipelines import MeshFlowPipeline

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pipe = MeshFlowPipeline.from_pretrained(args.ckpt, device="cpu" if args.device != "cuda" else "cuda", dtype=args.dtype)
    if args.device == "mps":
        pipe.to("mps")                            # from_pretrained forces CPU when CUDA is absent
    sync(args.device)
    report = {"device": args.device, "steps": args.steps, "seed": args.seed, "dtype": args.dtype,
              "load_s": time.time() - t0, "points": args.points}

    timing, mesh, latents = generate(pipe, args.points, args.device, args.steps, args.seed, guidance=1.0)
    report.update(timing)
    tm = mesh.to_trimesh()
    tm.export(out / f"meshflow_{args.device}.ply")
    report.update(n_vertices=int(len(tm.vertices)), n_faces=int(len(tm.faces)),
                  watertight=bool(tm.is_watertight))
    np.save(out / f"latents_{args.device}.npy", latents.float().cpu().numpy())

    if args.check_cfg:
        t_cfg, mesh_cfg, lat_cfg = generate(pipe, args.points, args.device, args.steps, args.seed, guidance=2.5)
        report["cfg2.5_sample_s"] = t_cfg["sample_s"]
        report["cfg2.5_vs_1.0_latent_max_abs_diff"] = float((lat_cfg - latents).abs().max())
        report["cfg2.5_n_faces"] = int(len(mesh_cfg.to_trimesh().faces))

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss       # bytes on macOS, KiB on Linux
    report["peak_rss_gb"] = peak / 1e9 if sys.platform == "darwin" else peak / 1e6
    (out / f"bench_{args.device}.json").write_text(json.dumps(report, indent=2))
    print("[bench] " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in report.items()))


if __name__ == "__main__":
    main()
