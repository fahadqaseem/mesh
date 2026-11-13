# Status

Snapshot, rewritten at the end of every work block. Last updated: 2026-11-12 23:45.

## Latest
- **New direction (Prof. Zhong, 9 Nov): points side only; the point cloud is the main output.** Every points stage is
  now scored directly against the GT (Chamfer, Hausdorff, F1@0.005/0.01/0.02, NC, coverage, #points, plus a
  density-sensitive all-points recall); MeshFlow meshes are a secondary row. Report:
  `docs/results/2026-11-12_points.md` (866 jobs, 864 ok; the 2 not-ok jobs are a harmless 1/255 colour difference on
  wheel_76903, explained there).
- **TSDF density sweep:** finer voxels do not help with 8 views (0.0025: F1 about the same, NC 0.81 to 0.75; 0.00125:
  worse everywhere, 27 s, up to 14 GB). 1M points instead of 200k is cheap (+0.3 s) but only raises the all-points recall.
  The depth maps are the limit, not the fusion grid. The default stays at voxel 0.005 / 200k.
- **Gaussian kernel sampling** (`scripts/gaussian_kernel_points.py`): beats Gaussian centres on every shape and gives the
  highest coverage, trails TSDF on average, but wins on the thin shapes where TSDF is worst.
- **`docs/pipeline_explained.md`**: how each figure column was made; MeshFlow keeps 4,096 random points, voxelises at
  32^3, and its vertex count comes from generation (no QEM).
- **Dataset prep** (`scripts/prep_dataset.py`, ShapeNetCore v2 and Toys4K): written and tested on stand-in files; the
  real datasets are gated (see Blocked).

## What works
- Render -> VGGT (8 views) -> TSDF fusion -> camera alignment -> points scored vs GT -> MeshFlow -> protocol v1, all on
  the Mac under the memory rules. Points sources: TSDF (default), filtered (v2), Gaussian centres, Gaussian kernels.
- TSDF stage standalone from a saved raw depth file (`scripts/tsdf_points.py`), with timing and peak memory.
- Queue tooling (`scripts/overnight.sh`; disk and memory thresholds via MIN_DISK_GB / MIN_MEM_PCT). 33 tests pass.

## Blocked
- **ShapeNetCore v2 and Toys4K are gated.** Fahad must request access himself: ShapeNetCore, accept the terms at
  https://huggingface.co/datasets/ShapeNet/ShapeNetCore; Toys4K, the form linked from
  https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k. Then `python scripts/prep_dataset.py --dataset
  shapenet --root data/shapenet_raw --n 3` (and the same for toys4k) renders a few shapes; the full sets are not run yet.
- 16-view VGGT needs the lab GPU (Mac: 16 views ~19 GB of 24 GB).
- Disk: a pending macOS update snapshot takes several GB (free space went 21 -> 12 -> 19 GB). Installing or removing
  the update frees it.

## Current numbers (points vs GT, protocol v1, mean ± std over shapes, dev shape excluded)
| points stage | set | Chamfer | F1@0.01 | F1@0.02 | NC | coverage | #points |
|---|---|---|---|---|---|---|---|
| TSDF, voxel 0.005 (default) | B (12) | 5.61 ± 1.79 | 0.469 | 0.708 | 0.809 | 76% | 200k |
| TSDF, voxel 0.005 | A (7) | 5.58 ± 1.93 | 0.515 | 0.751 | 0.803 | 77% | 200k |
| TSDF, voxel 0.0025 | B (12) | 5.62 ± 1.78 | 0.480 | 0.709 | 0.754 | 76% | 200k |
| Gaussian kernels | B (12) | 5.80 ± 1.39 | 0.393 | 0.680 | 0.761 | 80% | 200k |
| Gaussian centres | B (12) | 5.74 ± 1.41 | 0.256 | 0.608 | 0.769 | 50% | 11.5k |
| filtered points (v2) | B (12) | 8.37 ± 3.03 | 0.379 | 0.560 | 0.810 | 43% | 186k |
- Secondary, MeshFlow on TSDF points (seed 42): set B Chamfer 6.24, F1@0.02 0.598, #V 933; upper bound on clean GT
  points 2.93 / 0.979 / 265.

## Next 3 steps
1. TSDF + Gaussian-kernel union (kernels where the TSDF surface is missing): pilot on pencilpot, then set A/B.
2. ShapeNet / Toys4K: a few shapes each once access is granted, then the points pipeline on them.
3. 16 views on the lab GPU (better depth maps are now the main lever for the points).

## Open questions
- **Dense output by default?** 1M points at voxel 0.005 costs +0.3 s and is safe on the Mac, but is only denser, not
  more accurate under the protocol metrics. For Prof. Zhong on Mon 16 Nov.
- **Which point metric leads?** The protocol subsamples (5k for Chamfer, 100k for F1), so it is blind to density; the
  all-points recall is the density-sensitive one. Proposal: keep Chamfer + F1@0.01 as the headline, report the
  all-points recall next to them.
- **Gaussian settings were chosen on pencilpot only**, and the Gaussians start from 100k TSDF points, so they are not
  independent of the TSDF cloud.
- **Known focal as the headline**; **one frame for all metrics** (MeshFlow's [-0.5, 0.5] cube). Unchanged.
