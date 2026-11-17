# Status

Snapshot, rewritten at the end of every work block. Last updated: 2026-11-17 01:30.

## Latest
- **Friday follow-ups (17 Nov), `docs/results/2026-11-17_conf_ablation.md`; `docs/methods.pdf` exported for attaching.**
  - Alignment check (ICP vs GT, diagnostic only): camera centres fit to 0.7% of the orbit radius; rigid ICP lowers set B
    outliers > 0.02 only from 23.9% to 17.3%; ICP with scale to 7.0%, but its scale (0.970, < 1 on 12 of 12) absorbs an
    outward surface offset, not a camera error. The fused VGGT depth pixels already sit outside the GT (median +0.012
    of the object size, 80% outside); the TSDF cloud +0.0065. **The outliers are a VGGT depth bias (~1% too short), not
    alignment.**
  - Confidence ablation: conf_percentile 30 and 10 fuse exactly the same pixels as 50 on 11 of 12 shapes (the
    percentile over all pixels is the background value 1.0; VGGT's own foreground decides). Fusing every foreground
    pixel (no erosion) fills 4% of the GT gap and worsens Chamfer by 0.20; gap fill fills 46%. **Extra depth pixels do not
    fill the gaps; the gaps are occluded surface** (99.4% of gap points lie behind fused depth in some view).
  - Corrections: the 18 Nov outlier report said gap points sit on low-confidence filtered pixels (wrong, fixed);
    methods.md now says the percentile is inert.
- **Block of 16 Nov (for Prof. Zhong, due Fri 20 Nov 1 PM): all four tasks done.**
  - Task A, `docs/methods.md`: every stage from the code with file:line references; NOT VERIFIED where code cannot show it
    (Open3D's marching cubes and TSDF weighting, msplat internals, set B selection script, MeshFlow's training loss and
    the CVPR supplementary). The 4,096 vs 32,768 discrepancy: the 32,768 points are in the paper's main text (§3.4,
    arXiv v2) as the abandoned initial Shape2VecSet design; the released model conditions on 4,096 points voxelised at
    32^3. "Supplementary Section A" could not be read (CVF blocks automated access): **Fahad, please check it by hand.**
    Also corrected: MeshFlow's generation noise does not depend on the input pick (13 Nov report said it did).
  - Task B, `docs/results/2026-11-17_equal_count.md`: at exactly 1M points each, gap fill keeps its gains (set B Chamfer
    5.33 vs 5.58, recall all @0.02 0.825 vs 0.682) but adds outliers on every shape (> 0.02: 24.0% to 28.0%). The TSDF
    cloud itself has 24% of its points beyond 0.02.
  - Task C, `docs/results/2026-11-18_outliers.md` + figure: 78% of gap points are > 0.02 off the GT; no Gaussian property
    separates them (AUC <= 0.62); 91.5% lie where no view has fused depth (the filtered low-confidence pixels), so
    multi-view checks see little. Free space + silhouette is the only filter that improves both (set B outliers -0.7
    points, Chamfer -0.05); opacity / scale thresholds only delete points.
  - Task D, `docs/results/2026-11-19_seeded_4k.md`: seeded random, FPS and curvature-weighted FPS give the same mesh
    accuracy (set B Chamfer 6.20 / 6.20 / 6.47); FPS variants give ~3,900 vertices; seeded random is the clean baseline.
- **Block of 15 Nov done** (`docs/results/2026-11-15_dense.md`, summary on top):
  - Dense output (1M points) is the default; the dense gap-filled cloud is the best cloud: set B Chamfer 5.31 (was
    5.61), recall all @0.02 0.827 (was 0.677), coverage 85% (was 76%).
  - 12 and 16 VGGT views run safely on the laptop (memory guard, 40 runs, never red, lowest free 13%). 16 views: set B
    F1@0.01 0.469 to 0.552, NC 0.809 to 0.832, recall up on 12 of 12; but Chamfer 5.59 to 5.85 (far floaters). Default
    stays at 8 views until TSDF min_views is piloted for more views.
  - Skipped: MeshFlow vertex control (the public checkpoint has no trained num_verts layer: config lacks
    use_proj_cond_on_temb and model.pth has no t_embedder.cond_proj weights).
  - Picture: `docs/results/figures/2026-11-15_gapfill_compare.png` (TSDF vs gap fill on the 3 hardest set B shapes).
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
1. Before Fri 20 Nov: Fahad checks the CVPR supplementary (Section A) and the MeshFlow loss wording against the PDF.
2. GT-free depth-scale correction (one factor per scene from multi-view depth agreement) to remove the outward offset.
3. 12/16 views with TSDF min_views 2 (the gaps are unseen surface; more views is the lever), then gap fill on them.

## Open questions
- **Gap fill: keep, filter or drop?** Better Chamfer and recall, +4 points of outliers; the outliers sit in occluded
  regions no view sees, so filters cannot remove them without removing the gains, and lower confidence thresholds do
  not help (17 Nov). For Prof. Zhong.
- **Correct VGGT's depth bias?** A GT-free per-scene depth scale is cheap to try; it would change the points stage
  (and every number since 9 Nov), so it needs his OK before becoming the default.
- **16 views as the default?** Safe on the laptop and better on F1, NC and recall, worse on Chamfer until the
  floaters are handled. For Prof. Zhong; my plan is to pilot min_views first.
- **Points default = TSDF + gap fill?** Better cloud on every main metric except F1@0.01 (-0.004), but it needs the
  3DGS step (~10 s per object with msplat, not bit-reproducible) and is a worse MeshFlow input. For Prof. Zhong.
- **MeshFlow's vertex count depends on how its 4,096 input points are spread** (random surface samples ~900, FPS
  ~3,800, same accuracy). Worth asking the MeshFlow authors or checking once their training code is out; it matters
  for any input we design for it.
- **Which point metric leads?** The protocol subsamples (5k for Chamfer, 100k for F1), so it is blind to density; the
  all-points recall is the density-sensitive one. Proposal: keep Chamfer + F1@0.01 as the headline, report the
  all-points recall next to them.
- **Gaussian settings were chosen on pencilpot only**, and the Gaussians start from 100k TSDF points, so they are not
  independent of the TSDF cloud.
- **Known focal as the headline**; **one frame for all metrics** (MeshFlow's [-0.5, 0.5] cube). Unchanged.
