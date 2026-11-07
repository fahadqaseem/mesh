# Status

Snapshot, rewritten at the end of every work block. Last updated: 2026-11-07.

## What works
- End to end on the Mac: render (Thingi10K, v1 cameras + procedural texture) -> VGGT points -> alignment to GT -> CVPR MeshFlow mesh -> scores.
- CVPR MeshFlow with the real checkpoint on MPS and CPU (~95 s per mesh either way; MPS 5.5 GB, CPU 10 GB peak).
- VGGT points for all 8 test scenes (16 views, capped to 200k points each). Camera alignment to GT ~0.017 RMS (rig radius 3).
- Evaluation of points and meshes (Chamfer, NC, F1, F1 ceiling, normal_outward). 14 tests pass.
- Memory rules applied: one heavy job at a time, 200k point cap, raw.npz opt-in (outputs/ 1.7 GB -> 53 MB).

## Blocked
- **Floater fix needs a VGGT rerun**, and the memory rules cap VGGT on the Mac at 8 views, while the scenes have 16.
  Waiting on Fahad (see Open questions). Until then, no new VGGT runs.
- 16-view VGGT, and anything that keeps VGGT and MeshFlow in memory together, waits for the lab GPU (Mac crash on 2026-11-07).

## Current numbers (pencilpot, MeshFlow on MPS, rigid ICP, n=100k, tau=0.01)
| MeshFlow input | Chamfer | NC | F1 | size vs GT |
|---|---|---|---|---|
| VGGT points as is | 0.0334 | 0.731 | 0.088 | 0.86 |
| VGGT, floaters removed (oracle, uses GT) | 0.0115 | 0.884 | 0.369 | 0.98 |
| GT surface samples (upper bound) | 0.0004 | 0.955 | 0.488 | 1.02 |
- Main finding: floaters in the VGGT cloud (flying pixels at depth edges) cause MeshFlow's spikes and the ~14% shrink.
  The input stage is the bottleneck, not MeshFlow.
- VGGT points only, pencilpot grey: chamfer 0.00524, NC 0.847, F1 0.599. Textured: pending (points table).
- MeshFlow timing per mesh: MPS 93 s (DiT 1.81 s/step, extraction 33 s); CPU 100 s (DiT 3.24 s/step, extraction 1.6 s).

## Next 3 steps
1. Depth-edge filter in the VGGT stage (drop flying pixels), then rerun VGGT (view count per your answer) and MeshFlow.
2. Points table for all 8 scenes (grey vs textured evidence); MeshFlow on all 8 scenes; first results table for Friday.
3. Seed variance for MeshFlow (the mesh changes a lot between runs), so differences in the table can be judged.

## Open questions
- **8 vs 16 views for the VGGT rerun.** The floater fix needs new VGGT runs. Options: (a) 8 views on the Mac (4 per ring,
  every other camera), which is allowed by the memory rules but differs from your 16-view setup; (b) wait for the lab GPU
  and keep 16. Safest for now: no new VGGT runs; I work on things that do not need them.
- **Metric protocol.** F1 at tau 0.01 (unit sphere, 100k samples) is capped at 0.93-1.00 even for a perfect mesh, and it
  punishes lightweight meshes: MeshFlow on clean input gets chamfer 4e-4 but F1 0.49 (flat facets). What normalisation,
  sample count and tau does HD-PEA use? Worth asking Hongbo, so our numbers line up with the lab's.
- **Scale in scoring.** Meshes are scored after rigid ICP (no scale), so size errors count against them. Keep it that way?
- (Answered 2026-11-06: procedural texture now, Google Scanned Objects in December; Y-up + 2 rings x 8 views.)
