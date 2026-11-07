# Experiment log

One entry per run. Newest on top. Claude: append here after every run, and keep "Next up" current.

## Next up
1. [ ] Depth-edge (flying pixel) filter in run_points.py, then VGGT rerun with 8 views (4 per ring), one scene at a time.
1b.[ ] Write docs/eval_protocol.md; rescore without ICP (camera alignment only), F1 at 0.01 and 0.02.
2. [x] Points table for all 8 scenes (11-07). Grey vs textured: textured not better on pencilpot; check one more shape grey.
3. [ ] MeshFlow on all 8 scenes (after the floater fix), then the first results table with the F1 ceiling. For Friday's meeting.
4. [ ] Seed variance: MeshFlow with 3 to 5 seeds on one input, so differences between runs can be judged.
5. [ ] Adapter: DiT on MPS, decode/extract on CPU (~30 s saved per mesh), default if it checks out. Guidance 2.5 vs 1.0 check (own process).
5b.[ ] Grey vs textured on 2 to 3 more shapes (when convenient).
5c.[ ] Lab GPU later: VGGT with 16 views, as an 8 vs 16 comparison.
6. [ ] Experiment: Gaussian centres vs interpolated points as MeshFlow input.
7. [ ] December (with the colour render-back loop, priority 2): add Google Scanned Objects for real textures.

Done: env + pytest (11-06); MeshFlow weights downloaded (11-07); 8 Thingi10K shapes rendered v1 (Y-up, 2 rings x 8,
procedural texture); VGGT on all 8 scenes (16 views); alignment stage (camera Umeyama + ICP); point-cloud evaluation;
MeshFlow CPU vs MPS timing (11-07); input ablation on pencilpot (11-07); point cap 200k + raw.npz opt-in.

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

### 2026-11-07  points_table_v1  (VGGT points vs GT, all scenes; grey vs textured)
Goal: stage-1 numbers for every scene, and Fahad's grey vs textured evidence on pencilpot.
Command: `python scripts/points_table.py --runs pencilpot_299291_grey_v1 --suffix tex_v1` (aligns each run, then scores)
Input: VGGT v1 runs (16 views), capped at 200k points. Config: default.yaml; camera Umeyama + sim3 ICP; n=100k, tau=0.01.
Result:
Points stage vs GT (n=100000, tau=0.01, GT in unit sphere)

| run | #points | chamfer | normal_cons | normal_outward | precision | recall | f1 | f1_ceiling | camera_rms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pencilpot_299291_grey_v1 | 200,000 | 5.21e-03 | 0.846 | 0.894 | 0.667 | 0.544 | 0.600 | 0.928 | 0.019 |
| cryptex_105686_tex_v1 | 200,000 | 1.93e-03 | 0.747 | 0.831 | 0.539 | 0.579 | 0.558 | 0.966 | 0.019 |
| cylinder_472094_tex_v1 | 200,000 | 1.16e-02 | 0.803 | 0.837 | 0.510 | 0.724 | 0.599 | 1.000 | 0.025 |
| monsterring_1356640_tex_v1 | 200,000 | 2.34e-02 | 0.785 | 0.934 | 0.785 | 0.633 | 0.701 | 0.958 | 0.014 |
| nsphere_186199_tex_v1 | 200,000 | 1.27e-02 | 0.757 | 0.851 | 0.545 | 0.507 | 0.525 | 0.993 | 0.033 |
| pencilpot_299291_tex_v1 | 200,000 | 7.68e-03 | 0.837 | 0.879 | 0.617 | 0.528 | 0.569 | 0.928 | 0.015 |
| presents_98021_tex_v1 | 200,000 | 6.24e-03 | 0.800 | 0.862 | 0.704 | 0.647 | 0.674 | 0.956 | 0.016 |
| shackbot_94155_tex_v1 | 200,000 | 2.08e-02 | 0.768 | 0.878 | 0.555 | 0.534 | 0.544 | 0.974 | 0.027 |
| wheel_76903_tex_v1 | 200,000 | 1.00e-02 | 0.791 | 0.931 | 0.875 | 0.651 | 0.747 | 0.958 | 0.011 |

Notes / what I saw:
  - Grey vs textured (pencilpot): textured is NOT better. chamfer 0.0077 vs 0.0052 grey, F1 0.569 vs 0.600, NC 0.837 vs 0.846.
    One shape and one run each, so this is weak evidence. Possible reasons: shading already gives VGGT enough features on
    this shape, or the smooth sine pattern adds little. Worth a second shape before drawing a conclusion.
  - F1 is 0.53-0.75 against a ceiling of 0.93-1.00. Recall (0.51-0.72) is limited by unseen surfaces; precision by floaters.
  - Chamfer varies 12x across shapes (0.002-0.023) and is dominated by floaters (squared distances); not a fair shape-to-shape
    comparison until the floater fix.
  - Normals point outward 83-93% of the time; wrong ones are mostly noisy points (see vggt_v1 entry).
Takeaway: stage 1 works on every shape, but floaters dominate the error; the depth-edge filter is the next lever.
Next: floater fix (waiting on the view-count answer), then rerun this table.

### 2026-11-07  pencilpot_299291_tex_v1 / ablation  (what limits MeshFlow: input or mesher?)
Goal: find the cause of the spikes, split and shrink in the first MeshFlow mesh.
Command: `python scripts/bench_meshflow.py --points <input>.ply --device mps --out outputs/pencilpot_299291_tex_v1/ablation/<name>`
Input (all in the GT frame, MeshFlow resamples 4096 points from each):
  (a) VGGT points as is (200k cap); (b) DIAGNOSTIC ONLY, not a result: the same with floaters removed using the GT (points > 0.03 from the
  GT surface dropped, 6.3% of points); (c) 200k clean GT surface samples (upper bound for the mesher).
Config: real checkpoint, fp32, MPS, 28 steps, seed 42, guidance 1.0. Scored after rigid ICP (no scale), n=100k, tau=0.01.
Result:
  (a) VGGT as is:         chamfer=0.0334  nc=0.731  f1=0.088  verts=3770 faces=11763  size vs GT 0.86  spikes
  (b) DIAGNOSTIC ONLY (GT-cleaned): chamfer=0.0115  nc=0.884  f1=0.369  verts=1442 faces=7449   size vs GT 0.98  no spikes; split at mouth, ragged holes
  (c) GT samples:         chamfer=0.0004  nc=0.955  f1=0.488  verts=361  faces=746    size vs GT 1.02  clean low-poly mesh
  Picture: outputs/pencilpot_299291_tex_v1/ablation/compare_inputs.png
Notes / what I saw:
  - The VGGT cloud has a halo of floaters (max radius 1.47 vs 1.00 for GT; extents 2.2 vs 1.7). MeshFlow fits the whole
    cloud into its unit sphere, so the real surface shrinks (~0.86), and floaters among the 4096 samples become spikes.
  - Local density filters do not remove the halo (statistical k=32 std=0.5 still leaves max radius 1.25, and loses 6.5% of
    the surface). Floaters are dense: most likely flying pixels at depth edges (silhouettes). Fix belongs in the VGGT
    stage: drop pixels at depth discontinuities or erode the foreground mask. That needs a VGGT rerun.
  - With clean input, MeshFlow is good (chamfer 4e-4, NC 0.955), but F1@0.01 is only 0.49: flat facets of a lightweight
    mesh sit > 0.01 from a curved surface. F1 at this tau penalises exactly the lightweight meshes we want (metric question).
Takeaway: the input stage, not MeshFlow, is the bottleneck now. This is evidence for the stage-wise ablation argument.
Next: depth-edge filter in run_points.py + VGGT rerun (view count pending Fahad's answer), then MeshFlow on all scenes.

### 2026-11-07  pencilpot_299291_tex_v1 / meshflow CPU vs MPS  (real checkpoint, timing)
Goal: CPU vs MPS timing of CVPR MeshFlow on the Mac (rerun after the crash, one process per device).
Command: `python scripts/bench_meshflow.py --points outputs/pencilpot_299291_tex_v1/align/points_aligned.ply --device {mps,cpu} --out outputs/pencilpot_299291_tex_v1/mesh_bench`
Input: VGGT textured pencilpot, aligned, capped at 200k points (the crash run used the 2M cloud).
Config: real checkpoint, fp32, 28 steps, seed 42, guidance 1.0.
Result:
  | device | load | sampling (28 steps) | per step | decode+extract | total | peak RSS | verts / faces |
  | MPS    | 9.5 s | 50.7 s | 1.81 s | 32.7 s | ~93 s  | 5.5 GB | 3770 / 11763 |
  | CPU    | 7.4 s | 90.8 s | 3.24 s | 1.6 s  | ~100 s | 10.1 GB | 3558 / 22208 |
  Scores after rigid ICP: MPS chamfer=0.0334 f1=0.088; CPU chamfer=0.0284 f1=0.100; crash-run (2M input) chamfer=0.0308 f1=0.131.
Notes / what I saw: MPS is 1.8x faster for the DiT, but mesh extraction is 20x slower on MPS, so totals are similar.
  Best option on the Mac: DiT on MPS, decode/extract on CPU (~62 s expected; not implemented yet).
  CPU and MPS meshes differ for the same seed (different devices draw different noise), and the mesh changes a lot with the
  4096-point resample of the input (1752 vs 3770 verts). Seed-to-seed variance must be measured before comparing numbers.
  The guidance 2.5 vs 1.0 check is still not done (it was the pass running during the crash).
Next: see the ablation entry above.

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
