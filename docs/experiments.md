# Experiment log

One entry per run. Newest on top. Claude: append here after every run, and keep "Next up" current.

## Next up
0. [ ] Get access to the gated HF weights (huggingface.co/facebook/meshflow), `hf auth login`, download to
       `ckpt/meshflow/`, rerun the MeshFlow smoke test with the real checkpoint.
1. [x] Set up the env on the Mac (README "Setup"), run `pytest`. (2026-11-06: 8 passed; MeshFlow code runs on CPU + MPS)
2. [ ] Download a handful of small Thingi10K meshes into `data/gt/` (5 to 10 simple shapes).
3. [ ] Render 12 views of each with `scripts/render_views.py`.
4. [ ] Run VGGT on one scene, open `points.ply` in MeshLab, check normals point outward.
5. [ ] Run CVPR MeshFlow on that `points.ply` (CPU/MPS may be slow; move to lab GPU if so).
6. [ ] Add alignment (ICP, or use the GT render cameras) so `evaluate.py` scores are meaningful.
7. [ ] First table: Chamfer / NC / F1 / #verts for 5 shapes. Bring it to Friday's meeting.
8. [ ] Experiment: Gaussian centres vs interpolated points as MeshFlow input.

## Template
```
### YYYY-MM-DD  <run name>
Goal:
Input: (scene, #views)
Config: (diff from default.yaml)
Result: chamfer=  nc=  f1=  verts=  faces=  time=
Notes / what I saw:
Next:
```

## Runs

### 2026-11-06  meshflow_mac_smoke (random weights)
Goal: check that CVPR MeshFlow runs on the MacBook (M4 Pro, 24 GB) before the lab GPU is available.
Input: synthetic noisy sphere point cloud (20k pts, no faces), as a stand-in for points.ply.
Config: real checkpoint is gated on HF (facebook/meshflow, access requested), so this used a hand-made
  config.yaml + empty model.pth (random init). Small: 512 latents, width 384, 2 layers. Full-size guess:
  4096 latents, width 1152, DiT 24 layers, VAE 8+8 layers. fp32, 4 steps. Script: scratchpad smoke_meshflow.py.
Result:
  - from_pretrained -> run -> to_trimesh works on CPU and on MPS (no CUDA-only deps; plain SDPA attention).
  - Full size, DiT: CPU 3.3 s/step, MPS 1.7 s/step. VAE decode: CPU 0.9 s, MPS 0.5 s. Peak RSS ~4.7 GB.
  - End-to-end with random weights: CPU 105 s, MPS 737 s, dominated by extract_mesh on millions of junk
    faces (untrained adjacency). Expect this to be much smaller with real weights; recheck.
  - Bug found and fixed: MeshFlow's from_pretrained forces CPU whenever CUDA is absent, so "mps" was
    silently ignored. Adapter (meshflow_cvpr.py) now calls pipeline.to(device) after loading.
Notes / what I saw: half precision is disabled off-CUDA (autocast warns and falls back to fp32; harmless).
  Estimate for a real 28-step run on MPS: ~50 s DiT + decode/extract. Fine for small Mac tests.
Next: get HF access -> `hf auth login` -> `hf download facebook/meshflow --local-dir ckpt/meshflow`,
  rerun on a real point cloud, compare CPU vs MPS extract_mesh time.
