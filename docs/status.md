# Status

Snapshot, rewritten at the end of every work block. Last updated: 2026-11-06.

## What works
- Mac environment (M4 Pro, 24 GB, Python 3.13 venv): `pytest` passes (8 tests).
- CVPR MeshFlow code runs on CPU and MPS (checked with random weights only). The adapter now actually uses MPS.
- VGGT installed (`pip install --no-deps -e third_party/vggt`; its `numpy<2` pin cannot install on Python 3.13, and it works with numpy 2.5).
- `render_views.py` renders offscreen on macOS. 8 Thingi10K test shapes are rendered (12 views each) in `data/scenes/`.

## Blocked
- **MeshFlow weights**: `facebook/meshflow` on Hugging Face is gated. Access was requested on 2026-11-06 and is pending
  (check huggingface.co/settings/gated-repos). After approval: `hf auth login`, then download to `ckpt/meshflow/`.

## In progress
- First VGGT run (`pencilpot_v1`) on the MacBook (MPS).

## Current numbers
- MeshFlow at roughly real size, random weights: DiT 1.7 s/step on MPS vs 3.3 s/step on CPU; peak RSS ~4.7 GB.
- No mesh-quality numbers yet (needs real weights, plus alignment for evaluate.py).

## Next 3 steps
1. Finish VGGT on pencilpot, inspect `points.ply` (density, normals pointing outward), then run the other 7 scenes.
2. Add alignment (GT render cameras or ICP) so `evaluate.py` scores mean something.
3. When the weights arrive: run MeshFlow on the VGGT points and build the first results table.

## Open questions
- Renders are plain grey with no texture. That is hard for VGGT, and it means colour cannot be tested yet.
  Should we add textured or vertex-coloured test shapes (e.g. Objaverse or Google Scanned Objects) for the colour loop?
- Thingi STLs are Z-up but the camera ring orbits around Y, and there is a single ring at 20° elevation, so the tops
  and bottoms of some shapes are barely seen. Fix (rotate to Y-up, add a second ring) before the results table?
  For now the safest option is to keep the current renders and note it.
