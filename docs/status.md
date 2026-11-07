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

## Current numbers (pencilpot, MeshFlow on MPS; old protocol: rigid ICP, n=100k, tau=0.01; to be rescored)
| MeshFlow input | Chamfer | NC | F1 | size vs GT |
|---|---|---|---|---|
| VGGT points as is | 0.0334 | 0.731 | 0.088 | 0.86 |
| DIAGNOSTIC ONLY, not a result: VGGT with floaters removed using the GT | 0.0115 | 0.884 | 0.369 | 0.98 |
| GT surface samples (upper bound) | 0.0004 | 0.955 | 0.488 | 1.02 |
- Main finding: floaters in the VGGT cloud (flying pixels at depth edges) cause MeshFlow's spikes and the ~14% shrink.
  The input stage is the bottleneck, not MeshFlow.
- VGGT points vs GT, all 8 scenes: F1 0.53-0.75 (ceiling 0.93-1.00), NC 0.75-0.85, normals outward 83-93%.
  Grey vs textured (pencilpot): textured is not better (F1 0.569 vs 0.600 grey, chamfer 0.0077 vs 0.0052). Weak evidence: one shape.
- MeshFlow timing per mesh: MPS 93 s (DiT 1.81 s/step, extraction 33 s); CPU 100 s (DiT 3.24 s/step, extraction 1.6 s).

## Next 3 steps
1. Depth-edge filter in the VGGT stage (drop flying pixels), then rerun VGGT (view count per your answer) and MeshFlow.
2. MeshFlow on all 8 scenes and the first results table for Friday (points table is done: outputs/points_table.md).
3. Seed variance for MeshFlow (the mesh changes a lot between runs), so differences in the table can be judged.

## Open questions
- None open. Answered 2026-11-07 (see decisions.md): 8-view VGGT on the Mac now, 16 on the lab GPU later; no HD-PEA
  dependency; scoring protocol in docs/eval_protocol.md; scale errors stay in the score (camera alignment only, no ICP);
  texture stays the default; the GT-cleaned input is diagnostic only; MPS sampling + CPU extraction if the hybrid checks out.
