# Experiment log

One entry per run. Newest on top. Claude: append here after every run, and keep "Next up" current.

## Next up
1. [ ] MeshFlow CPU vs MPS timing with the real checkpoint on pencilpot (one process per device, memory check first).
2. [ ] Oracle-input ablation: MeshFlow on clean GT surface samples vs on VGGT points. Tells whether the spikes and
       split in pencilpot come from noisy input or from MeshFlow itself.
3. [ ] Points table for all 8 scenes (`scripts/points_table.py --suffix tex_v1` + pencilpot grey): grey vs textured evidence.
4. [ ] Memory rules: cap saved points (~200k), delete raw.npz. Future VGGT runs on the Mac use at most 8 views (see status.md).
5. [ ] MeshFlow on all 8 scenes, then the first results table (Chamfer / NC / F1 / #verts, with F1 ceiling). Bring it to Friday's meeting.
6. [ ] Experiment: Gaussian centres vs interpolated points as MeshFlow input.
7. [ ] December (with the colour render-back loop, priority 2): add Google Scanned Objects for real textures.

Done: env + pytest (11-06); MeshFlow weights downloaded (11-07); 8 Thingi10K shapes rendered v1 (Y-up, 2 rings x 8,
procedural texture); VGGT on all 8 scenes (16 views); alignment stage (camera Umeyama + ICP); point-cloud evaluation.

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

### 2026-11-07  pencilpot_299291_tex_v1 / meshflow_mps  (first real MeshFlow mesh)
Goal: first run of CVPR MeshFlow with the real checkpoint on a VGGT point cloud; CPU vs MPS timing.
Command: `python scripts/bench_meshflow.py --points outputs/pencilpot_299291_tex_v1/align/points_aligned.ply --device mps --out outputs/pencilpot_299291_tex_v1/mesh_bench --check-cfg`
Input: VGGT points (textured pencilpot, 16 views, 2.06M points), aligned to the GT frame. MeshFlow resamples 4096 points itself.
Config: ckpt/meshflow/meshflow (real config: DiT width 1536, 16 layers; VAE 1024 wide, 8+8 layers; 4096 latents).
  fp32 (no half precision off CUDA), 28 steps, seed 42, guidance_scale 1.0 (no reference image).
Result: verts=1752 faces=9640, not watertight. vs GT (unit sphere, n=100k, tau=0.01):
  as output: chamfer=0.0395 nc=0.662 f1=0.027 | after rigid ICP (3.4 deg, 0.21 shift): chamfer=0.0308 nc=0.737 f1=0.131.
  time: not recorded (the Mac crashed before the timing JSON was written; see next entry). CPU run never started.
Notes / what I saw: the shape is clearly recognisable (eye holes, ears, mouth, open top), so the pipeline works end to end.
  But: long spike triangles, a horizontal split through the mouth, ~13% smaller than GT, and 9640 faces on 1752 verts
  (5.5 faces per vertex vs ~2 for a clean closed mesh). Picture: outputs/pencilpot_299291_tex_v1/mesh_bench/compare_gt_meshflow_points.png
Next: CPU vs MPS timing again (one process at a time); oracle-input ablation to see whether noisy VGGT points cause the artefacts.

### 2026-11-07  CRASH: kernel panic from memory exhaustion (~00:15)
What happened: the Mac (24 GB) froze and restarted (watchdog timeout, memory and swap exhausted, disk ~96% full).
  Sequence: VGGT batch of 8 scenes x 16 views (each process ~19 GB peak, one at a time), then the pencilpot textured VGGT
  rerun, then MeshFlow (4.5 GB checkpoint) on MPS. The MeshFlow run wrote its mesh at 00:11 and crashed during the second
  pass that --check-cfg adds (guidance 2.5, which doubles the batch). Swap was already ~15 GB used during the VGGT batch.
Lost: MeshFlow timing JSON, the CPU timing run, the guidance 2.5 vs 1.0 check. Nothing in data/ or the repo was lost.
Takeaway: 16-view VGGT plus MeshFlow does not fit this Mac back to back. New "Mac memory rules" in CLAUDE.md:
  one heavy job at a time, VGGT at most 8 views, cap saved points, check memory before each run.

### 2026-11-07  vggt_v1 batch  (8 scenes, 16 views, render v1)
Goal: VGGT points for every test scene with the v1 camera setup; pencilpot both grey and textured.
Command: `scripts/vggt_batch.sh` (one process per scene, sequential) -> outputs/vggt_batch.tsv
Input: data/scenes/<scene>/images_tex (and images_grey for pencilpot), 16 views at 518 px.
Config: default.yaml (conf_percentile 50, with the fixed floor bound), depth unprojection, fp32 on MPS.
Result (points kept / seconds / peak memory GB):
  pencilpot grey 2.04M / 56 / 18.4, pencilpot tex 2.06M / 55 / 19.2, cryptex 1.88M / 55 / 18.8, cylinder 0.95M / 45 / 17.1,
  monsterring 2.04M / 57 / 19.1, nsphere 1.66M / 61 / 18.4, presents 2.04M / 65 / 19.0, shackbot 1.93M / 62 / 18.3, wheel 2.04M / 56 / 18.9.
  pencilpot grey vs GT (aligned): chamfer=0.00524 nc=0.847 f1=0.599 (precision 0.668, recall 0.544), normals outward 89%.
  Camera alignment is accurate: VGGT camera centres fit the GT rig to 0.015-0.019 RMS (rig radius 3); ICP then moves < 0.4 deg.
  Textured scores and the full table: pending (scripts/points_table.py; run lost to my own cleanup mistake, then the crash).
Notes / what I saw: 16 views needs ~19 GB, too close to the 24 GB limit (see the crash entry).
  Wrong-sign normals (11%) grow with distance from the surface (8% within 0.01, 39% beyond 0.03): noise, not an orientation bug.
  F1 at tau=0.01 with 100k samples is capped below 1 even for a perfect reconstruction (0.93-1.00 by shape), because
  sample spacing is close to tau. points_table.py reports this ceiling next to each F1.
Next: points table (grey vs textured), then MeshFlow on every scene.

### 2026-11-06  pencilpot_v1  (first VGGT run; confidence-filter bug)
Goal: first VGGT run on the Mac (render v0: 12 views, single ring, grey).
Command: `python scripts/run_points.py --images data/scenes/pencilpot_299291/images --run pencilpot_v1`
Result: 4.3 min including the 5 GB weight download; peak ~21 GB. Kept 3.11M of 3.22M pixels: 56% were white background.
Notes / what I saw: bug. VGGT gives background confidence exactly 1.0; with >50% background the 50th percentile is 1.0
  and `conf >= 1.0` kept everything. Fixed (floor is now a strict bound, with tests): 1.49M points, white share 56% -> 8%.
  run_points.py now saves raw.npz and has --reuse-raw for re-filtering without VGGT.
Next: re-render with the v1 camera setup and texture (decided by Fahad).

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
