# Status

Snapshot, rewritten at the end of every work block. Last updated: 2026-11-07 01:50.

## Running now: overnight queue (started 01:27, expected finish ~03:15, at the latest ~04:00 with memory waits)
- `nohup caffeinate -is scripts/overnight.sh` (log `outputs/overnight/runner.log`); queue `outputs/overnight/queue.tsv`
  (215 jobs, built by `scripts/make_overnight_queue.py`). One job at a time, each in its own process; before every job:
  free memory >= 30% and free disk >= 15 GB, else wait 5 min, skip after 3 checks; 30 min cap per job; failures are
  logged and skipped. Progress: `outputs/overnight/progress.txt`; per-job logs in `outputs/overnight/jobs/`.
- After each step the results doc `docs/results/2026-11-07_overnight.md` is regenerated, committed and pushed.
- Steps (most important first):
  1. 8 shapes: VGGT (8 views, default v2) -> align (cameras only) -> points eval -> MeshFlow seed 42 + eval.
     Also the uncalibrated variant (VGGT's own focal) from the same VGGT pass. (~21 min)
  2. Upper bound: MeshFlow on clean GT surface samples, 8 shapes. (~9 min)
  3. Seed variance: MeshFlow seeds 7 and 123 on every step-1 cloud. (~17 min)
  4. Grey vs textured: shackbot, presents, wheel (+ pencilpot), 3 seeds each. (~15 min)
  5. 12 more Thingi10K shapes (`configs/step5_shapes.txt`): render, steps 1 and 2. (~36 min)
- If it stopped: rerun the same command; finished jobs are skipped.
- Progress at 01:50: step 1 done and published (72/72 jobs ok), step 2 running; no failed or skipped jobs; free memory 78-81%.
- Reporting (Fahad, 01:45): pencilpot is the dev shape (used for tuning), shown but excluded from all means. Headline
  means over set A (the other 7 original shapes) and set B (the 12 step-5 shapes, clean test set). The Friday summary
  (docs/results/2026-11-13_friday_summary.md) is written when the queue finishes.
- Step 1, set A (7 shapes, seed 42): pipeline mesh CD 10.32 ± 3.73, NC 0.769, F1@0.01 0.298, F1@0.02 0.473, #V 1,682,
  size 1.05; with VGGT's own focal CD 11.77, F1@0.02 0.406, size 1.10. VGGT focal ratio 0.84-1.01 (mean 0.94).

## What works
- Full pipeline on the Mac under the memory rules: render -> VGGT (8 views) -> camera alignment -> MeshFlow -> protocol-v1 scores.
- Evaluation protocol v1 written down (`docs/eval_protocol.md`: MeshFlow paper's CD/HD, plus NC, F1@0.01/0.02, #V/#F, G).
- MeshFlow hybrid (DiT on MPS, extraction on CPU): ~51 s per mesh, identical output to all-MPS.
- 20 tests pass.

## Blocked
- 16-view VGGT (and the 8 vs 16 comparison) waits for the lab GPU (Mac memory: 16 views ~19 GB).

## Current numbers (pencilpot, protocol v1, MeshFlow mean of seeds 42/7/123)
| input | CD | HD | NC | F1@0.01 | F1@0.02 | size |
|---|---|---|---|---|---|---|
| old: 16 views, VGGT focal, no filter | 10.64 | 37.3 | 0.791 | 0.147 | 0.318 | 1.35 |
| new default: 8 views, known focal, multi-view >= 2, erode 2 | 10.02 | 31.0 | 0.786 | 0.148 | 0.339 | 1.06 |
- CD floor (perfect mesh) for pencilpot: 2.79. Overnight step 2 gives the MeshFlow upper bound on clean input.
- Main findings today: (1) VGGT's focal is 3-8% short on textured renders, which distorts the cloud; (2) floaters are
  mostly multi-view-inconsistent depth, not silhouette flying pixels; a multi-view check removes the halo.
- DIAGNOSTIC ONLY (never a result): the earlier "floaters removed using GT" input; it showed the input stage is the bottleneck.

## Next 3 steps (after the overnight run)
1. Read the overnight report; write the summary for Friday (results table with mean and spread, upper bound, seed spread).
2. Depth quality: the pipeline is far from the clean-input upper bound. Try VGGT's point head vs depth unprojection,
   and a per-view depth scale check against the multi-view consensus.
3. Lab GPU: 16-view VGGT for the 8 vs 16 comparison.

## Open questions (decisions I made overnight; all reversible)
- **Known focal as the headline.** VGGT's predicted focal is 3-8% short on textured renders; with camera-only alignment
  this costs a lot (pencilpot mesh F1@0.02 0.30 -> 0.16). I use the render focal (as EXIF would give for real photos)
  for the headline and report VGGT's own focal as the "uncalibrated" variant next to it. OK, or should the headline be
  uncalibrated?
- **Edge-pixel fix.** The plain depth-discontinuity filter did not help (most floaters are not silhouette pixels). The
  default now uses a 2 px silhouette erosion plus a multi-view depth-consistency check (>= 2 other views agree within
  1%). Thresholds were picked on pencilpot only (with the GT used for diagnosis) and kept fixed for all other shapes.
- **One frame for all metrics.** MeshFlow's [-0.5, 0.5] unit cube is used for CD/HD and also for the F1 thresholds
  (rather than a unit sphere for F1), so all numbers share one frame.
- **8 views = 0,2,4,6 (20 deg ring) + 9,11,13,15 (50 deg ring).** Every other camera per ring.
- **Step 5 shapes** were picked automatically: clean, watertight, 1k-20k faces, thinnest/longest extent >= 0.35.
- **Grey vs textured** uses shackbot, presents, wheel, plus pencilpot again (redone with the new pipeline).
