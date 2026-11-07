# Decisions

Things agreed with Prof. Zhong, with the date. Add new ones at the bottom.

| Date | Decision | Why |
|---|---|---|
| 2026-11-05 | Priority 1: VGGT + CVPR MeshFlow end to end, small Thingi10K first | Get a working baseline and numbers before anything new |
| 2026-11-05 | Image to points is not the contribution; the mesh is | The novelty is anisotropic, lightweight meshes with geometry + texture |
| 2026-11-05 | Try Gaussian centres vs interpolated points as input | See which point source gives the mesher better input |
| 2026-11-05 | Priority 2: colour render-back loop (render mesh, compare to photos) | Adds texture fidelity and a self-check signal |
| 2026-11-05 | Priority 3 (after January): generative / flow model | Only after the modular pipeline works |
| 2026-11-05 | Keep the pipeline modular (swap the mesher); be ready to defend two stages vs a unified model | Reviewers will compare to LATO, MeshCraft |
| 2026-11-05 | HD-PEA plugs in as a second mesher once Hongbo shares code | It needs xyz + normals, which stage 1 already writes |
| 2026-11-05 | Weekly meetings Fridays 1 pm from Nov 13; lab GPU workstation later | Heavy runs move to the lab |
| 2026-11-06 | Test set v0: 8 clean Thingi10K shapes (ids 472094, 94155, 186199, 299291, 76903, 98021, 1356640, 105686), 12 views each | Single component, watertight, no self-intersections, 2k to 40k faces; flat parts dropped by eye; range from simple to thin and detailed |
| 2026-11-06 | Cheap texture now: procedural vertex colours (seeded 3D sine waves on a subdivided render copy) on Thingi shapes; run pencilpot grey vs textured as evidence | Grey renders give VGGT nothing to match and leave colour untested; Fahad's call, keep it cheap until the colour loop |
| 2026-11-06 | Google Scanned Objects deferred to the colour render-back loop (priority 2, December) | Real textures matter for that loop, not for the first geometry table |
| 2026-11-06 | Camera setup v1: meshes rotated to Y-up; two rings at 20° and 50° elevation, 8 views each (16 total, second ring offset half a step). Old renders kept in data/scenes_v0_12view_ring/ | v0 orbited the wrong axis (Thingi STL is Z-up) on a single 20° ring, so tops and bottoms were barely seen; fixed before the results table (Fahad) |
| 2026-11-07 | Mac memory rules: one heavy job at a time, VGGT at most 8 views on the Mac, cap saved points, check memory first | Kernel panic from memory exhaustion (16-view VGGT ~19 GB, then MeshFlow 4.5 GB); Fahad added the rules to CLAUDE.md |
| 2026-11-07 | MeshFlow runs with guidance_scale 1.0 when no reference image is given | With image=None, CFG compares two identical empty conditions, so 2.5 only doubles compute and memory. Numerical check still pending |
| 2026-11-07 | Next fix goes in the points stage (remove flying pixels at depth edges), not in the mesher | Input ablation on pencilpot: removing floaters cut MeshFlow Chamfer 3x and raised F1 4x; on clean GT samples MeshFlow is good (Chamfer 4e-4). Result-driven |
| 2026-11-07 | VGGT on the Mac: 8 views (4 per ring), rerun now with the edge-pixel fix, one scene at a time. Never split 16 views into chunks. 16 views later on the lab GPU, as an 8 vs 16 comparison | Mac memory limit (16 views ~19 GB); chunks would give inconsistent frames (Fahad) |
| 2026-11-07 | Work independently of Hongbo / HD-PEA for now. HD-PEA stays a placeholder mesher; MeshFlow is the only mesher | Fahad |
| 2026-11-07 | Scoring protocol: MeshFlow paper's protocol if it states one, otherwise unit sphere, 100k samples, F1 at tau 0.01 and 0.02, plus Chamfer, NC and vertex count in every table. Written in docs/eval_protocol.md | Fahad; one fixed protocol so tables stay comparable |
| 2026-11-07 | Scale errors stay in the score: similarity alignment from the GT cameras only, no re-fitting (no ICP) after meshing | Fahad; ICP would hide errors of the pipeline |
| 2026-11-07 | Texture stays the default. Grey vs textured on 2 to 3 more shapes when convenient | Fahad; one-shape evidence is weak, and texture is needed for the colour loop |
| 2026-11-07 | The "floaters removed using GT" input is diagnostic only: labelled that way everywhere, never used as a result | Fahad; it uses the GT |
| 2026-11-07 | Default MeshFlow device split: MPS for flow sampling, CPU for mesh extraction, if the ~62 s hybrid checks out | Fahad; extraction is 20x slower on MPS |
