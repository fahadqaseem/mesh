# Handoff from the claude.ai planning session (2026-11-06)

Read this once at the start, along with CLAUDE.md, then work from docs/experiments.md.

## What exists
- Full scaffold: points (VGGT wrapper, filtering, PCA normals oriented to cameras), swappable meshers
  (MeshFlow CVPR adapter, HD-PEA stub), metrics (Chamfer, NC, F1, counts), stage scripts, config.
- 8 unit tests pass (normals, metrics, PLY I/O). VGGT and MeshFlow have NOT been run yet.
- Adapters were written against the real APIs (VGGT and facebookresearch/meshflow repos were read).

## Facts worth remembering
- MeshFlow uses xyz only and normalises the cloud itself (bounding sphere, size 2). Normals and colours are ignored by it.
- MeshFlow API: `MeshFlowPipeline.from_pretrained(ckpt_dir, device, dtype)`, then
  `run(mesh=<ply path>, image=None, steps, seed, num_verts=...)`, then `.to_trimesh()`.
- VGGT: depth + camera unprojection is used by default (authors say it beats the point head).
- HD-PEA needs xyz + normals (6 values per point); stage 1 already writes those.
- VGGT output is in its own frame and scale, so evaluate.py scores are meaningless until alignment is added
  (use GT render cameras from data/scenes/<name>/gt_cam_poses.npy, or ICP).

## Immediate next steps (in order)
1. Run `sh _setup/install.sh` if `_setup/` still exists, then the README setup, then `pytest`.
2. Smoke test MeshFlow alone on a sample point cloud (biggest risk: CUDA-only dependencies on the Mac).
   If it cannot run on MPS/CPU, note it in experiments.md; full runs wait for the lab GPU.
3. Render 5 to 10 small Thingi10K shapes, run VGGT, inspect points.ply in MeshLab.
4. Add alignment, then produce the first results table for the Friday meeting.

## Timeline target
Preliminary table (5 to 10 shapes + Gaussian centres vs interpolation) by about Dec 4.
