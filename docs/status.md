# Status

Snapshot, rewritten at the end of every work block. Last updated: 2026-11-07.

## What works
- End to end on the Mac: render (Thingi10K, v1 cameras + procedural texture) -> VGGT points -> alignment to GT -> CVPR MeshFlow mesh.
  First real MeshFlow mesh exists (pencilpot, MPS). It is recognisable but has artefacts (see numbers).
- VGGT points for all 8 test scenes (16 views each, plus grey pencilpot).
- Alignment: camera-centre Umeyama + trimmed ICP. VGGT cameras match the GT rig to ~0.017 RMS (rig radius 3).
- Point-cloud and mesh evaluation (Chamfer, NC, F1, plus normal_outward and the F1 ceiling). 14 tests pass.
- MeshFlow weights downloaded (ckpt/meshflow/meshflow/).

## Blocked
- **Heavy work on the Mac is memory-limited.** The Mac crashed on 2026-11-07 (~00:15): 16-view VGGT (~19 GB) followed by
  MeshFlow (4.5 GB weights) exhausted memory and swap. New Mac memory rules in CLAUDE.md. 16-view VGGT runs and anything
  that loads VGGT and MeshFlow together have to wait for the lab GPU.
- CPU vs MPS MeshFlow timing was lost in the crash. Rerunning it one device at a time.

## Current numbers
- VGGT pencilpot grey vs GT: chamfer 0.00524, NC 0.847, F1 0.599 (precision 0.668, recall 0.544), normals outward 89%.
  Textured scores: pending (points table).
- VGGT on the Mac: 45-65 s per 16-view scene, ~19 GB peak.
- First MeshFlow mesh (pencilpot textured, MPS): 1752 verts / 9640 faces, not watertight. After rigid ICP:
  chamfer 0.0308, NC 0.737, F1 0.131. Spike triangles, a split through the mouth, ~13% too small.
- F1 ceiling (perfect reconstruction, tau 0.01, 100k samples): 0.93-1.00 depending on the shape.

## Next 3 steps
1. MeshFlow CPU vs MPS timing on pencilpot (one process per device, memory check first).
2. Oracle-input ablation: MeshFlow on clean GT surface samples. Do the artefacts come from the VGGT input or from MeshFlow?
3. Points table for all scenes (grey vs textured evidence), then MeshFlow on all 8 scenes.

## Open questions
- **8 vs 16 views.** You chose 16 views (2 rings x 8), but the new memory rule caps VGGT on the Mac at 8 views.
  The 8 existing 16-view point clouds are already done, so nothing needs rerunning now. For new VGGT runs on the Mac,
  should I use 8 views (4 per ring), or wait for the lab GPU for 16? Safest for now: keep the existing 16-view results
  and run no new VGGT jobs until you decide.
- **Metric protocol.** At tau = 0.01 on a unit sphere with 100k samples, F1 is capped at 0.93-1.00 even for a perfect mesh.
  What normalisation, sample count and tau does HD-PEA use? Worth asking Hongbo so our numbers match the lab's.
- **MeshFlow scale.** Its output is ~13% smaller than the input cloud. Should scoring use rigid ICP (current) or allow
  scale? With scale, a size error would hide; without it, scores include it. Currently rigid only.
- (Earlier, answered by Fahad on 2026-11-06: procedural texture now, Google Scanned Objects in December; Y-up + 2 rings.)
