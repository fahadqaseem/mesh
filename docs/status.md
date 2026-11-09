# Status

Snapshot, rewritten at the end of every work block. Last updated: 2026-11-09 02:00.

## Latest
- **TSDF fusion is the default points stage** (your decision; `configs/default.yaml` points.fusion: tsdf; the old
  filtered points via `--set fusion=points`). Headline rerun 135/135 ok; every cloud byte-identical to the 7 Nov TSDF run.
- **Gaussian centres vs TSDF points vs filtered points** (msplat on the Mac, 114/114 ok): Gaussian centres match TSDF
  on Chamfer (set B 6.44 vs 6.24) but give lower NC (0.736 vs 0.819) and 3 times the vertices; both beat the filtered
  points (9.58). TSDF stays the default. Report: `docs/results/2026-11-09_v3_gaussians.md`.
- Friday summary updated (`docs/results/2026-11-13_friday_summary.md`, update 197 words).

## What works
- Pipeline on the Mac under the memory rules: render -> VGGT (8 views) -> TSDF fusion -> camera alignment -> MeshFlow
  -> protocol v1. Optional points stages: filtered points (v2), Gaussian centres (`scripts/gaussian_points.py`).
- 3D Gaussian splatting on the Mac: msplat 1.1.4 (installed), about 10 s and 0.6 GB per object at 8 views x 518 px.
- Queue tooling for any job list (`scripts/overnight.sh`); reports regenerate after each step. 30 tests pass.

## Blocked
- 16-view VGGT (8 vs 16) needs the lab GPU (Mac: 16 views ~19 GB of 24 GB).

## Current numbers (protocol v1, MeshFlow seed 42, mean ± std over shapes, dev shape excluded)
| stage | set | Chamfer | NC | F1@0.01 | F1@0.02 | #V | size |
|---|---|---|---|---|---|---|---|
| Pipeline mesh (TSDF points, default) | B (test, 12) | 6.24 ± 1.87 | 0.819 | 0.335 | 0.598 ± 0.146 | 933 | 1.09 |
| Pipeline mesh (TSDF points, default) | A (7) | 5.98 ± 1.94 | 0.812 | 0.386 | 0.688 ± 0.136 | 864 | 1.08 |
| Pipeline mesh, Gaussian centres | B (test, 12) | 6.44 ± 1.73 | 0.736 | 0.321 | 0.581 ± 0.151 | 2,920 | 1.16 |
| Pipeline mesh, Gaussian centres | A (7) | 6.34 ± 1.66 | 0.706 | 0.331 | 0.586 ± 0.180 | 3,116 | 1.16 |
| Pipeline mesh, filtered points (previous) | B (test, 12) | 9.58 ± 2.98 | 0.806 | 0.235 | 0.436 ± 0.141 | 1,440 | 1.06 |
| Pipeline mesh, filtered points (previous) | A (7) | 10.32 ± 3.73 | 0.769 | 0.298 | 0.473 ± 0.181 | 1,682 | 1.05 |
| Upper bound: MeshFlow on clean GT points | B (12) | 2.93 ± 0.51 | 0.941 | 0.746 | 0.979 | 265 | 1.01 |
| Upper bound: MeshFlow on clean GT points | A (7) | 2.74 ± 0.63 | 0.908 | 0.815 | 0.979 | 256 | 1.02 |
- Set A, 3 MeshFlow seeds: TSDF 5.83 vs filtered 10.23. Chamfer floor of a perfect mesh: set B 2.37, set A 2.26.

## Next 3 steps
1. Unseen surface (24% of the gap: the 8 cameras never see the underside): views from below, or mesher-side completion,
   depending on Prof. Zhong's answer to the strategy question.
2. 16 views on the lab GPU (8 vs 16).
3. Size bias (1.08 to 1.09 with TSDF; caused by noise via MeshFlow): try light smoothing of the fused surface on
   pencilpot only.

## Open questions
- **Strategy (in the Friday summary): points vs mesher.** Evidence so far: the points side gave a third of the
  Chamfer back cheaply (TSDF); Gaussian centres add nothing as a point source; what remains is mostly unseen surface,
  which only completion (mesher side) or more viewpoints can fix.
- **Known focal as the headline** (also in the Friday summary). Uncalibrated numbers are reported next to it.
- **Gaussian settings were chosen on pencilpot only** (100k initial points from the TSDF cloud, opacity >= 0.5), and the
  Gaussians start from the TSDF cloud, so they are not independent of it. An SfM-only start would need COLMAP features.
- **One frame for all metrics** (MeshFlow's [-0.5, 0.5] cube also for the F1 thresholds). My choice.
- **Set B has one MeshFlow seed per shape.** Seed spread on set A is small compared with these effects.
