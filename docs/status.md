# Status

Snapshot, rewritten at the end of every work block. Last updated: 2026-11-07 12:35 (interim; full rewrite when the queue ends).

## Latest
- **Running now (started 12:23, expected to finish about 15:15):** the robustness queue, 600 jobs, 136 MeshFlow runs
  (`scripts/make_robustness_queue.py`, log `outputs/robustness_queue/runner.log`). Step 1 Experiment A (MeshFlow on
  corrupted clean points, set A), step 2 Experiment B on set A (TSDF fusion, relaxed filter, point head), step 3 the
  decomposition, step 4 Experiment B on set B. Results land in `docs/results/2026-11-08_robustness.md` after each step.
- First finding (dev shape pencilpot): the multi-view filter keeps only 29% of the GT surface (VGGT's raw depth covers
  77%, the rig sees 75%). TSDF fusion of all eroded pixels keeps 72% and gave mesh Chamfer 7.05 vs 10.25 on pencilpot.
- Overnight run finished: 215/215 jobs ok (01:27 to 02:53), no failures, no memory waits. Full tables:
  `docs/results/2026-11-07_overnight.md`. Friday summary for Prof. Zhong (13 Nov): `docs/results/2026-11-13_friday_summary.md`.
- Pencilpot is the dev shape (used to tune the filters): shown in every table, excluded from all means. Set A = the other
  7 original shapes; set B = 12 new shapes picked automatically, the clean test set.

## What works
- Full pipeline on the Mac under the memory rules: render (Thingi10K, procedural colour) -> VGGT (8 views) -> alignment
  from the render cameras -> CVPR MeshFlow -> protocol-v1 scores (`docs/eval_protocol.md`).
- Overnight queue tooling: `scripts/make_overnight_queue.py`, `scripts/overnight.sh` (one job at a time, memory and disk
  checks, resumable), `scripts/overnight_report.py`.
- MeshFlow hybrid (flow sampling on MPS, extraction on CPU): ~51 s per mesh. 20 tests pass.

## Blocked
- 16-view VGGT (and the 8 vs 16 comparison) needs the lab GPU (Mac memory: 16 views ~19 GB of 24 GB).

## Current numbers (protocol v1, MeshFlow seed 42, mean ± std over shapes, dev shape excluded)
| stage | set | Chamfer | NC | F1@0.01 | F1@0.02 | #V | size |
|---|---|---|---|---|---|---|---|
| Pipeline mesh | B (test, 12) | 9.58 ± 2.98 | 0.806 | 0.235 | 0.436 ± 0.141 | 1,440 | 1.06 |
| Pipeline mesh | A (7) | 10.32 ± 3.73 | 0.769 | 0.298 | 0.473 ± 0.181 | 1,682 | 1.05 |
| Pipeline mesh, VGGT's own focal | A (7) | 11.77 ± 4.70 | 0.774 | 0.235 | 0.406 | 739 | 1.10 |
| Upper bound: MeshFlow on clean GT points | B (12) | 2.93 ± 0.51 | 0.941 | 0.746 | 0.979 | 265 | 1.01 |
| Upper bound: MeshFlow on clean GT points | A (7) | 2.74 ± 0.63 | 0.908 | 0.815 | 0.979 | 256 | 1.02 |
- Chamfer of a perfect mesh (sampling floor): set B 2.37, set A 2.26. Seed spread is small (Chamfer 0.30 across seeds vs
  3.53 across shapes). Texture is required: on grey renders VGGT's camera poses failed on the symmetric shapes.
- Main finding: MeshFlow is near the floor on clean input; the VGGT points are the bottleneck (and they make MeshFlow
  output 5-6x more vertices).

## Next 3 steps
1. Points-stage quality: VGGT point head vs depth unprojection; per-view depth scale check against the multi-view
   consensus; cloud thickness.
2. Find the 5-6% size bias (MeshFlow alone adds 1-2%; the rest comes from the cloud).
3. Lab GPU: 16 views (8 vs 16), then Gaussian centres vs interpolated points.

## Open questions
- **Known focal as the headline** (also asked of Prof. Zhong in the Friday summary). VGGT's focal is short on 18 of 19
  shapes (mean 0.93x). The known focal improves set A Chamfer 11.8 -> 10.3 and F1@0.02 0.41 -> 0.47. Uncalibrated numbers
  are reported next to it. OK as the headline, or should it be uncalibrated?
- **One frame for all metrics.** MeshFlow's [-0.5, 0.5] unit cube is used for Chamfer/Hausdorff and also for the F1
  thresholds (rather than a unit sphere for F1). My choice, so all numbers share one frame.
- **Filters.** The plain depth-edge filter did not help; the default uses a 2 px silhouette erosion plus a multi-view
  depth-consistency check. Thresholds were tuned on pencilpot only (now the dev shape); the test set shows no overfitting.
- **8 views = 0,2,4,6 (20 deg ring) + 9,11,13,15 (50 deg ring).** Every other camera per ring.
- **Set B has one MeshFlow seed per shape.** Seed spread on set A is small, so I kept one; 3 seeds would add ~25 min.
