# Status

Snapshot, rewritten at the end of every work block. Last updated: 2026-11-07 15:30.

## Latest
- Robustness run finished: 600/600 jobs ok (12:23 to 15:02), 136 MeshFlow runs, no failures, no memory waits.
  Results: `docs/results/2026-11-08_robustness.md`. Friday summary updated (`docs/results/2026-11-13_friday_summary.md`).
- **Answer to "which corruption explains the gap": missing surface.** MeshFlow tolerates the real noise level (1% of
  the object size: Chamfer 2.74 to 3.25). The filtered VGGT clouds cover only 40 to 43% of the GT surface, while the 8
  cameras see 75%. Set A decomposition of the Chamfer gap: never seen by the rig 24%, lost by my multi-view filter
  48%, noise and bias 4%, rest 23%. Noise is what multiplies the vertices and causes the size bias.
- **TSDF fusion of the 8 VGGT depth maps (Open3D, CPU, about 20 s) is a clear win:** set B (test) Chamfer
  9.58 to 6.24, F1@0.02 0.436 to 0.598, vertices 1,440 to 933, better on 12 of 12; set A 10.32 to 5.98.
- VGGT point head = depth unprojection (no gain). Pencilpot stays the dev shape (it tuned the TSDF settings).

## What works
- Full pipeline on the Mac under the memory rules: render -> VGGT (8 views) -> camera alignment -> MeshFlow -> protocol v1.
- New: `scripts/tsdf_points.py` (fusion, deterministic), `scripts/corrupt_points.py`, `scripts/cloud_stats.py`,
  `scripts/robustness_report.py`, `scripts/make_robustness_queue.py`; `scripts/overnight.sh` runs any queue. 27 tests pass.
- VGGT on MPS is deterministic (reruns byte-identical on all 19 shapes).

## Blocked
- 16-view VGGT (8 vs 16) needs the lab GPU (Mac: 16 views ~19 GB of 24 GB). Note: 16 views on the same two rings add
  only 0 to 6% visible surface; views from below would add more.

## Gaussian splatting on Apple Silicon (feasibility, no install done)
- **Fits the Mac; no lab GPU needed** for our scenes (8 views at 518 px, one object).
- Easiest: **msplat** 1.1.4 (Metal-native 3DGS trainer, Apache-2.0, no PyTorch). `pip install msplat` resolves to a
  754 KB wheel for this venv (cp313, macOS arm64); checked with a dry run only. Reads COLMAP / nerfstudio scenes,
  exports PLY. Published speed: 7K iterations of a full Mip-NeRF 360 scene in about 70 s on an M4 Max; our M4 Pro GPU
  has 16 cores (vs 40) but our scenes have far fewer pixels, so minutes per object is the estimate (not measured).
- Memory: about 2 KB per Gaussian (OpenSplat's figure), so 100k to 300k Gaussians is 0.2 to 0.6 GB. Run it alone
  as a heavy job anyway.
- OpenSplat supports Metal (MPS) but needs Xcode + Metal toolchain, OpenCV and libtorch builds: a long install, skipped.
  Brush (wgpu/Metal) has only unofficial macOS builds. gsplat / nerfstudio splatfacto are CUDA only.
- Needed first: export VGGT cameras + points as a COLMAP or nerfstudio scene (small script). Caveat: 3DGS from 8
  views is sparse-view; Gaussian centres may float off the surface, which is what the comparison would measure.

## Current numbers (protocol v1, MeshFlow seed 42, mean ± std over shapes, dev shape excluded)
| stage | set | Chamfer | NC | F1@0.01 | F1@0.02 | #V | size |
|---|---|---|---|---|---|---|---|
| Pipeline mesh, TSDF-fused depth (new) | B (test, 12) | 6.24 ± 1.87 | 0.819 | 0.335 | 0.598 ± 0.146 | 933 | 1.09 |
| Pipeline mesh, TSDF-fused depth (new) | A (7) | 5.98 ± 1.94 | 0.812 | 0.386 | 0.688 ± 0.136 | 864 | 1.08 |
| Pipeline mesh, filtered points (default) | B (test, 12) | 9.58 ± 2.98 | 0.806 | 0.235 | 0.436 ± 0.141 | 1,440 | 1.06 |
| Pipeline mesh, filtered points (default) | A (7) | 10.32 ± 3.73 | 0.769 | 0.298 | 0.473 ± 0.181 | 1,682 | 1.05 |
| Upper bound: MeshFlow on clean GT points | B (12) | 2.93 ± 0.51 | 0.941 | 0.746 | 0.979 | 265 | 1.01 |
| Upper bound: MeshFlow on clean GT points | A (7) | 2.74 ± 0.63 | 0.908 | 0.815 | 0.979 | 256 | 1.02 |
- Chamfer floor of a perfect mesh: set B 2.37, set A 2.26. Seed spread: 0.30 Chamfer (set A).

## Next 3 steps
1. Make TSDF fusion the default points stage (after your OK, see below); rerun the headline with it.
2. Gaussian centres vs interpolated points with msplat on the Mac (export VGGT scene to COLMAP format first).
3. Unseen surface (24% of the gap): test views from below, or treat completion as the mesher-side contribution,
   depending on Prof. Zhong's answer to the strategy question.

## Open questions
- **Switch the default points stage to TSDF fusion?** It wins on 12 of 12 test shapes. I kept the default (filtered
  points) as the headline and report TSDF next to it, because changing the default changes every headline number.
- **Strategy (asked of Prof. Zhong in the Friday summary): points vs mesher.** New input for it: MeshFlow already
  handles realistic noise; what it cannot do is fill unseen surface. A mesher-side contribution would be completion
  and vertex economy under noise, not denoising.
- **Known focal as the headline** (also in the Friday summary). Uncalibrated numbers are reported next to it.
- **One frame for all metrics** (MeshFlow's [-0.5, 0.5] cube also for the F1 thresholds). My choice.
- **Coverage statistic uses the GT** (footprint within 1% of L). Diagnostic only; nothing feeds back into the pipeline.
- **Set B has one MeshFlow seed per shape.** The seed spread on set A is small compared with these effects.
