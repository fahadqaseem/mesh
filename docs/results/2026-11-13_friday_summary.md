# Friday summary for Prof. Zhong, 13 Nov 2026

Fahad Qaseem Khawar. Numbers from the overnight run of 7 Nov (full tables: `docs/results/2026-11-07_overnight.md`).

## Update

**Done this week**
- The pipeline runs end to end on my MacBook: Thingi10K renders, VGGT (8 views), CVPR MeshFlow.
- 12 held-out shapes: Chamfer 9.6 (perfect: 2.4), F1@0.02 0.44, about 1,440 vertices. MeshFlow on clean GT points: Chamfer 2.9, F1@0.02 0.98.
- Fusing VGGT's depth maps (TSDF, CPU) instead of filtering points: Chamfer 6.2, F1@0.02 0.60, about 930 vertices, better on all 12.

**What I learned**
- Robustness tests: MeshFlow tolerates realistic noise; missing surface explains most of the gap. My filter kept 43% of the surface.
- Noise is what makes MeshFlow output 5 to 6 times more vertices.
- VGGT's focal is about 7% short; texture is needed for its poses.

**Questions**
- The gap is in the points, not the mesher. Should I (a) improve the points (your Gaussian centres idea), or (b) make the mesher robust to noisy points from photos, keeping the contribution on the mesh side?
- Is the known focal (calibrated cameras) acceptable for our setting?
- 16 views exceeds the Mac's memory: when can I use the lab GPU?

**Next week**
- TSDF as the default points stage, Gaussian centres (fits the Mac), 16 views on the lab GPU.

## Headline numbers

Protocol (`docs/eval_protocol.md`): MeshFlow paper's Chamfer and Hausdorff (x100, GT normalised to [-0.5, 0.5]), plus
normal consistency and F1 at 0.01 and 0.02 (100k samples). The prediction is placed in the GT frame only through the
render cameras (no ICP after meshing), so scale and position errors count. Size = mesh extent / GT extent (1 = right).

Shape sets: pencilpot was used to tune the point filters, so it is the **dev shape**: shown, but not in any mean.
**Set A** = the other 7 original shapes. **Set B** = 12 new Thingi10K shapes picked automatically and never looked at
before the run: the clean test set. Mean ± std over shapes, MeshFlow seed 42. Lower Chamfer and Hausdorff are better. "Pipeline mesh" rows without a
label use the filtered points (current default); the TSDF rows are from the 7 Nov robustness run
(`docs/results/2026-11-08_robustness.md`; TSDF settings chosen on pencilpot only).

| stage | set | n | Chamfer | Hausdorff | NC | F1@0.01 | F1@0.02 | #V | size |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pipeline mesh | **B (test)** | 12 | **9.58 ± 2.98** | 29.1 ± 11.0 | **0.806 ± 0.044** | 0.235 ± 0.099 | **0.436 ± 0.141** | **1,440 ± 799** | 1.06 ± 0.03 |
| Pipeline mesh | A | 7 | 10.32 ± 3.73 | 28.0 ± 9.9 | 0.769 ± 0.071 | 0.298 ± 0.154 | 0.473 ± 0.181 | 1,682 ± 1,202 | 1.05 ± 0.02 |
| Pipeline mesh | dev (pencilpot) | 1 | 10.25 | 31.2 | 0.776 | 0.148 | 0.305 | 918 | 1.08 |
| Pipeline mesh, TSDF-fused depth (new) | **B (test)** | 12 | **6.24 ± 1.87** | 22.7 ± 10.3 | **0.819 ± 0.039** | 0.335 ± 0.131 | **0.598 ± 0.146** | **933 ± 379** | 1.09 ± 0.04 |
| Pipeline mesh, TSDF-fused depth (new) | A | 7 | 5.98 ± 1.94 | 22.0 ± 9.8 | 0.812 ± 0.083 | 0.386 ± 0.095 | 0.688 ± 0.136 | 864 ± 395 | 1.08 ± 0.04 |
| Pipeline mesh, TSDF-fused depth (new) | dev (pencilpot) | 1 | 7.05 | 26.8 | 0.849 | 0.259 | 0.509 | 804 | 1.06 |
| Pipeline mesh, VGGT's own focal | A | 7 | 11.77 ± 4.70 | 30.9 ± 15.0 | 0.774 ± 0.093 | 0.235 ± 0.164 | 0.406 ± 0.182 | 739 ± 450 | 1.10 ± 0.05 |
| VGGT points, before meshing | B (test) | 12 | 8.37 ± 3.03 | 27.6 ± 11.7 | 0.810 ± 0.031 | 0.379 ± 0.166 | 0.560 ± 0.150 | (186k points) | - |
| VGGT points, before meshing | A | 7 | 9.32 ± 3.46 | 27.1 ± 10.6 | 0.800 ± 0.051 | 0.414 ± 0.181 | 0.559 ± 0.153 | (179k points) | - |
| Upper bound: MeshFlow on clean GT points | B (test) | 12 | 2.93 ± 0.51 | 4.4 ± 0.8 | 0.941 ± 0.031 | 0.746 ± 0.190 | 0.979 ± 0.025 | 265 ± 124 | 1.01 ± 0.02 |
| Upper bound: MeshFlow on clean GT points | A | 7 | 2.74 ± 0.63 | 4.2 ± 1.1 | 0.908 ± 0.102 | 0.815 ± 0.149 | 0.979 ± 0.040 | 256 ± 162 | 1.02 ± 0.01 |
| Upper bound | dev (pencilpot) | 1 | 3.19 | 5.1 | 0.955 | 0.820 | 0.991 | 361 | 1.02 |

Chamfer of a perfect mesh under this protocol (sampling floor): set B 2.37, set A 2.26.

## Other findings

- **Robustness of MeshFlow (set A, clean GT points corrupted on purpose):** noise of 1% of the object size costs
  little (Chamfer 2.74 to 3.25), outliers matter only at 10% (7.1), missing surface matters most (3.1 / 4.4 / 7.3 at
  10 / 25 / 50% missing). Splitting the real gap (clean 2.74 to real cloud 10.32): surface the 8 cameras never see 24%,
  visible surface lost in my points stage 48%, noise and bias 4%, rest 23%. The multi-view filter was the culprit: it
  kept 40 to 43% of the surface although the cameras see 75%. Noise, not missing surface, multiplies the vertices
  (1% noise: 4.9x) and inflates the size (1.05 at 1%). Details: `docs/results/2026-11-08_robustness.md`.
- **VGGT point head vs depth unprojection:** no difference (set A, VGGT's own focal: Chamfer 11.91 vs 11.77).
- **Seed spread is small** (set A, MeshFlow seeds 42, 7, 123): std across seeds vs across shapes is 0.30 vs 3.53 for
  Chamfer and 0.022 vs 0.176 for F1@0.02. Vertex count varies more with the seed (553 vs 1,061). 3-seed means for set A:
  Chamfer 10.23, F1@0.02 0.491.
- **Texture matters for VGGT's cameras** (shackbot, presents, wheel; textured minus grey): points F1@0.02 +0.50, mesh
  F1@0.02 +0.38. On grey presents and grey wheel, VGGT's camera poses failed (camera fit error 2.4 on a rig of radius 3,
  against 0.013 to 0.020 textured): symmetric shapes look the same from different sides. On the dev shape pencilpot,
  which is not symmetric, grey was slightly better on F1 (mesh F1@0.02 0.42 vs 0.34) and slightly worse on Chamfer
  (10.41 vs 10.02).
- **Focal length:** VGGT's focal was short on 18 of 19 shapes (mean ratio 0.93). The known focal improves set A from
  Chamfer 11.8 to 10.3 and F1@0.02 0.41 to 0.47, and the size error from 10% to 5%.
- **Size bias:** pipeline meshes are 5 to 6% too large on both sets. MeshFlow alone adds 1 to 2% (clean input), so most of
  the excess comes from the point cloud.
- **Runtime on the Mac (M4 Pro, 24 GB):** VGGT with 8 views about 25 s; MeshFlow about 51 s per mesh (flow sampling on the
  Apple GPU, mesh extraction on the CPU). The whole overnight queue (215 jobs) took 86 minutes with no failures.

## Caveats

- Synthetic renders of Thingi10K shapes with a procedural colour pattern, not real photos.
- 8 views because of the Mac's memory; 16 views (and the 8 vs 16 comparison) waits for the lab GPU.
- The headline uses the known render focal (calibrated setting); VGGT's own focal is reported next to it.
- Set B has one MeshFlow seed per shape; set A has three.
- Colour is not evaluated yet (render-back loop is priority 2).
