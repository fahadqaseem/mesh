# Status

Snapshot, rewritten at the end of every work block. Last updated: 2026-11-15 00:00 (block in progress: views, MeshFlow vertex control, pictures).

## Latest
- **Gap filling (14 Nov, 139/139 ok): best point cloud so far.** TSDF points plus the Gaussian-kernel samples that lie
  3-10% from the TSDF cloud (~20k added, no GT used): set B Chamfer 5.61 to 5.34 (better on 8 of 12), F1@0.02 0.708 to
  0.732, coverage 76% to 82%, all-points recall @0.02 +0.12 (12 of 12); F1@0.01 -0.004. As a MeshFlow input it is
  worse (6.24 to 6.70), so MeshFlow keeps the plain TSDF cloud. Report: `docs/results/2026-11-14_gap_fill.md`.
- **FPS input test (13 Nov, 242/242 ok):** handing MeshFlow exactly 4,096 farthest-point-sampled TSDF points (so its
  random pick keeps all of them) gives the same accuracy (set A 3 seeds Chamfer 5.88 vs 5.83; set B 6.31 vs 6.24) but
  ~3,800 vertices instead of 740 to 933: MeshFlow keeps nearly every token on evenly spread input. The TSDF + kernel
  union is worse with either pick. MeshFlow's random pick stays. Report: `docs/results/2026-11-13_fps_input.md`.
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
  the Mac under the memory rules. Points sources: TSDF (default), filtered (v2), Gaussian centres, Gaussian kernels, TSDF + kernel gap fill.
- TSDF stage standalone from a saved raw depth file (`scripts/tsdf_points.py`), with timing and peak memory.
- Queue tooling (`scripts/overnight.sh`; disk and memory thresholds via MIN_DISK_GB / MIN_MEM_PCT). 36 tests pass.

## Blocked
- **ShapeNetCore v2 and Toys4K are gated.** Fahad must request access himself: ShapeNetCore, accept the terms at
  https://huggingface.co/datasets/ShapeNet/ShapeNetCore; Toys4K, the form linked from
  https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k. Then `python scripts/prep_dataset.py --dataset
  shapenet --root data/shapenet_raw --n 3` (and the same for toys4k) renders a few shapes; the full sets are not run yet.
- 16-view VGGT needs the lab GPU (Mac: 16 views ~19 GB of 24 GB).
- Disk: a pending macOS update snapshot takes several GB (free space went 21 -> 12 -> 19 GB). Installing or removing
  the update frees it.

## Current numbers (points vs GT, protocol v1, mean ± std over shapes, dev shape excluded)
Dense output is the default since 15 Nov (1M points). "Recall all" = share of the GT surface within tau of any point.
| points stage | set | Chamfer | F1@0.01 | F1@0.02 | recall all @0.01 | recall all @0.02 | NC | coverage | #points |
|---|---|---|---|---|---|---|---|---|---|
| TSDF + gap fill, 1M base (best) | B (12) | 5.31 ± 1.54 | 0.466 | 0.732 | 0.550 | 0.827 | 0.808 | 85% | 1.10M |
| TSDF, 1M (default) | B (12) | 5.59 ± 1.80 | 0.469 | 0.708 | 0.485 | 0.682 | 0.809 | 77% | 1M |
| TSDF, 200k (previous default) | B (12) | 5.61 ± 1.79 | 0.469 | 0.708 | 0.473 | 0.677 | 0.809 | 76% | 200k |
| TSDF + gap fill, 1M base (best) | A (7) | 5.39 ± 1.74 | 0.510 | 0.763 | 0.578 | 0.830 | 0.800 | 84% | 1.06M |
| TSDF, 1M (default) | A (7) | 5.56 ± 1.91 | 0.516 | 0.751 | 0.532 | 0.735 | 0.801 | 77% | 1M |
| Gaussian kernels | B (12) | 5.80 ± 1.39 | 0.393 | 0.680 | 0.534 | 0.770 | 0.761 | 80% | 200k |
| filtered points (v2) | B (12) | 8.37 ± 3.03 | 0.379 | 0.560 | - | - | 0.810 | 43% | 186k |
- Secondary, MeshFlow on TSDF points (seed 42): set B Chamfer 6.24, F1@0.02 0.598, #V 933 (MeshFlow keeps a random 4,096
  either way, so 1M vs 200k does not change its input distribution); upper bound on clean GT points 2.93 / 0.979 / 265.

## Next 3 steps
1. Meeting Mon 16 Nov: show the points-first results; ask about the points default (gap fill) and the open questions.
2. ShapeNet / Toys4K: a few shapes each once access is granted, then the points pipeline on them.
3. 16 views on the lab GPU (better depth maps are now the main lever for the points).

## Open questions
- **Points default = TSDF + gap fill?** Better cloud on every main metric except F1@0.01 (-0.004), but it needs the
  3DGS step (~10 s per object with msplat, not bit-reproducible) and is a worse MeshFlow input. For Prof. Zhong.
- **MeshFlow's vertex count depends on how its 4,096 input points are spread** (random surface samples ~900, FPS
  ~3,800, same accuracy). Worth asking the MeshFlow authors or checking once their training code is out; it matters
  for any input we design for it.
- **Dense output by default?** 1M points at voxel 0.005 costs +0.3 s and is safe on the Mac, but is only denser, not
  more accurate under the protocol metrics. For Prof. Zhong on Mon 16 Nov.
- **Which point metric leads?** The protocol subsamples (5k for Chamfer, 100k for F1), so it is blind to density; the
  all-points recall is the density-sensitive one. Proposal: keep Chamfer + F1@0.01 as the headline, report the
  all-points recall next to them.
- **Gaussian settings were chosen on pencilpot only**, and the Gaussians start from 100k TSDF points, so they are not
  independent of the TSDF cloud.
- **Known focal as the headline**; **one frame for all metrics** (MeshFlow's [-0.5, 0.5] cube). Unchanged.
