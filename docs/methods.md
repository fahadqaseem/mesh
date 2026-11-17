# Methods: what each pipeline stage does, from the code

Written 16 Nov 2026 for Prof. Zhong (requested at the 16 Nov meeting, due Fri 20 Nov). Every statement is checked
against the code at commit `53f2a67` (our repo), `third_party/meshflow` at `55f56f6`, `third_party/vggt` at `a288dd0`,
and the MeshFlow checkpoint `ckpt/meshflow/meshflow/{config.yaml, model.pth}`. References are `file:line`. Anything
that cannot be checked in code we can read (compiled libraries, the papers' unreleased training code, the selection of
the test shapes) is marked **NOT VERIFIED**. Software: Python 3.13, torch 2.15.0, Open3D 0.20.0, trimesh 5.2.0, scipy
1.18.1, msplat 1.1.4 (Apple M4 Pro, 24 GB).

Current default pipeline: render 16 views -> VGGT on 8 of them -> pixel filter -> TSDF fusion -> 1M points sampled from
the fused surface -> camera-based similarity alignment -> scored as points (protocol v1). Optional: Gaussian kernels and
gap fill (best cloud), MeshFlow (secondary).

## 1. Input data

**Shapes.** 20 meshes from Thingi10K [Zhou and Jacobson 2016], stored as STL in `data/gt/`:
- **pencilpot_299291**: development shape. Every setting below that was tuned was tuned on it alone, and it is in no
  mean (`docs/decisions.md:34`).
- **Set A (7)**: cylinder_472094, shackbot_94155, nsphere_186199, wheel_76903, presents_98021, monsterring_1356640,
  cryptex_105686 (`scripts/make_points_queue.py:24`). Chosen by hand on 6 Nov: "single component, watertight, no
  self-intersections, 2k to 40k faces; flat parts dropped by eye" (`docs/decisions.md:15`).
- **Set B (12, clean test set)**: Thingi10K ids in `configs/step5_shapes.txt` (1146183, 84620, 77949, 593381, 73155,
  40550, 290506, 105924, 103356, 59706, 697228, 94668). The log says they were "picked automatically: clean, 1k-20k
  faces, not flat" (`docs/experiments.md:183`). **NOT VERIFIED:** the selection script and the download of the STL files
  are not in the repo; only the resulting list is.

**Normalisation** (`scripts/render_views.py:58-67`): rotate Z-up STL to Y-up (`--up z`, default, `:122`), translate the
bbox centre to the origin, scale so the farthest vertex is at distance 1 (unit sphere). Saved as `gt_normalized.ply`
with the transform in `gt_transform.json` (`:166-170`).

**Colour for rendering only** (`:84-90`, `--texture procedural`, seed 0, `:125`): a subdivided copy (max edge 0.015) is
painted with seeded 3D sine waves; the GT geometry is not changed.

**Rendering** (`scripts/render_views.py:116-172`, pyrender):
- Resolution: 518 x 518 px (`--size 518`, `:126`), white background, ambient light 0.3 (`:131`), one directional light
  of intensity 3 at each camera (`:158`).
- Camera: perspective, vertical field of view 40 deg (`:148`). Focal length in pixels = (518 / 2) / tan(20 deg) = **711.6
  px**, principal point at the image centre (this is the "known focal" of section 2, `src/photo2mesh/points/vggt_points.py:119-121`).
- Layout: 16 views = 2 rings x 8 (`--n 8`, `--elevations 20,50`, `:120-121`), radius 3, all looking at the origin
  (`:39-46`). Ring 1 at 20 deg elevation, azimuths 0, 45, ..., 315 deg; ring 2 at 50 deg, rotated by half a step
  (22.5 deg, `:49-55`). View ids 0-7 = ring 1, 8-15 = ring 2.
- **8 views used by default**: ids 0, 2, 4, 6, 9, 11, 13, 15 (`configs/default.yaml:12`), i.e. 4 per ring, ring 1 at
  azimuths 0/90/180/270 deg and ring 2 at 67.5/157.5/247.5/337.5 deg. 12 views (15 Nov experiment) = those plus 1, 5,
  10, 14 (`scripts/make_views_queue.py:17`); 16 views = all.
- The rig never looks from below (lowest elevation 20 deg), so undersides are unobserved.

## 2. VGGT

**Citation:** J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, D. Novotny. *VGGT: Visual Geometry Grounded
Transformer.* CVPR 2025. Weights `facebook/VGGT-1B` (`src/photo2mesh/points/vggt_points.py:63`).

**Input:** the selected views, resized by VGGT's own loader to width 518 ("crop" mode) (`vggt_points.py:15-43`; at 518 it
calls `vggt.utils.load_fn.load_and_preprocess_images` unchanged, `:27-28`). fp32 on MPS (bf16 only on recent CUDA, `:62`).

**What we take from it** (`vggt_points.py:73-92`):
- **Camera poses:** extrinsics from the pose encoding (`pose_encoding_to_extri_intri`, `:73`). Used for fusion and for
  alignment.
- **Intrinsics:** VGGT's predicted focal is **replaced** by the known render focal (`intrinsics: known`,
  `configs/default.yaml:19`; `scripts/run_points.py:82-85`; `vggt_points.py:110-123`). VGGT's own estimate is kept only
  for diagnostics (`run_points.py:75`).
- **Depth maps:** the depth head's output, re-unprojected with the known focal (`use_depth_unprojection: true`,
  `configs/default.yaml:20`; `vggt_points.py:76-79`, `:123`).
- **Point map: not used.** The point head's `world_points` are returned (`:91-92`) but only used when
  `use_depth_unprojection` is false, which is not our setting (`run_points.py:76-81`).
- **Confidence: used.** With depth unprojection the confidence is the depth head's `depth_conf` (`vggt_points.py:79`).

**Pixel filter before fusion** (`src/photo2mesh/points/filter.py:75-95`, called from `fusion.py:101`):
1. Confidence: keep pixels with conf >= the 50th percentile **over all pixels of all views together** and conf > 1.0
   (`filter.py:8-16`; `configs/default.yaml:21-22`). VGGT confidences are 1 + exp(x) >= 1 and background sits at
   exactly 1.0, hence the strict bound (`filter.py:11-13`). **In practice the percentile does nothing on our renders**
   (measured 17 Nov, `docs/results/2026-11-17_conf_ablation.md`): VGGT's foreground (conf > 1.0) covers less than
   half of the image on 11 of 12 set B shapes, so the 50th percentile over all pixels is the background value 1.0
   and the binding rule is conf > 1.0, i.e. every pixel VGGT itself calls foreground. conf_percentile 30 or 10
   fuses exactly the same pixels.
2. Silhouette erosion: the foreground mask (conf > 1.0) is eroded by 2 px to drop the object's rim (`fg_erode: 2`,
   `configs/default.yaml:16`; `filter.py:84-87`, `:43-44`). The depth-edge test itself is off (`edge_rel: 0.0`,
   `configs/default.yaml:14`).
3. Multi-view consistency: **off for TSDF** (`points.tsdf.mv_rel: 0.0`, `configs/default.yaml:6`; it replaces the
   points-path value, `fusion.py:101`).

## 3. Depth fusion (TSDF), surface extraction and point sampling

**Citations:** B. Curless and M. Levoy. *A Volumetric Method for Building Complex Models from Range Images.* SIGGRAPH
1996 (weighted signed-distance averaging). R. A. Newcombe et al. *KinectFusion: Real-Time Dense Surface Mapping and
Tracking.* ISMAR 2011: its projective truncated-SDF integration applies; its tracking (ICP pose estimation) does
**not** (our poses come from VGGT). Open3D: Q.-Y. Zhou, J. Park, V. Koltun, *Open3D: A Modern Library for 3D Data
Processing*, arXiv 2018. W. E. Lorensen and H. E. Cline. *Marching Cubes.* SIGGRAPH 1987.

**Implementation** (`src/photo2mesh/points/fusion.py:14-41`): Open3D tensor API **`open3d.t.geometry.VoxelBlockGrid`**
on the CPU (`:25-28`), attributes tsdf / weight / color (float32), `block_resolution=8`, `block_count=20000`. (Open3D
0.20's legacy `ScalableTSDFVolume` integrates nothing on Apple silicon, `:20`.)
- **Object size** = largest extent of the 1st-99th percentile bbox of the kept pixels' 3D points (`fusion.py:103`).
- **Voxel size** = 0.005 x object size (`voxel_frac`, `configs/default.yaml:7`; `fusion.py:104`), about 0.003 in our
  units (e.g. wheel_76903: 0.00312).
- **sdf_trunc** = 6 voxels (`trunc_vox: 6`, `configs/default.yaml:8`), passed to Open3D as the truncation-voxel
  multiplier (`fusion.py:37-38`): about 0.019 (wheel_76903: 0.0187).
- **depth_trunc (depth_max)** = 1.5 x the largest kept depth (`fusion.py:29`); `depth_scale = 1.0` (`:37-38`).
  Rejected pixels are set to depth 0 = no measurement (`fusion.py:106`).
- **Weighting:** we pass no per-pixel weights; every integrated observation is handled by Open3D's default.
  **NOT VERIFIED** (compiled C++): that Open3D's tensor `integrate` uses weight 1 per observation and a running
  weighted average (the Curless-Levoy scheme with uniform weights).
- Colour images are VGGT's resized input images (`vggt_points.py:87`), fused as the colour attribute.

**Surface extraction** (`fusion.py:39`): `vbg.extract_triangle_mesh(weight_threshold = min_views - 0.5)` with
`min_views: 1` (`configs/default.yaml:9`), i.e. surface wherever at least one view observed the voxel.
**NOT VERIFIED** in code we can read: that `extract_triangle_mesh` is marching cubes (Open3D documents it as such; the
algorithm is in compiled code). Then: vertices/faces put into a canonical order (Open3D's multithreaded output order
varies between runs, `fusion.py:44-58`), and connected pieces with less than 1% of the total area dropped
(`min_part: 0.01`, `configs/default.yaml:10`; `fusion.py:75-85`, `:111`).

**Point sampling** (`fusion.py:61-72`, `:114-118`): **area-uniform random sampling** of the fused triangles (face
picked with probability proportional to area, uniform barycentric point in it), **not** Poisson-disk.
`numpy.random.default_rng(0)` (seed 0). Count **n = 1,000,000** (`points.tsdf.n_points`, `configs/default.yaml:11`, the
default since 15 Nov; 200,000 before; `run_points.py:90-91`). Normals = the face normal, flipped when it points away from
every camera (`fusion.py:115-118`). Colour = the face's mean vertex colour.

**Alignment to the GT frame** (`scripts/align.py:43`, `src/photo2mesh/eval/align.py`): one similarity transform
(scale, rotation, translation) fitted between VGGT's camera centres and the GT render camera centres (Umeyama,
S. Umeyama, PAMI 1991). No ICP (`--icp none` is the protocol default, `align.py:30-31`). Normals are rotated only (`:60`).

## 4. 3D Gaussian Splatting

**Citations:** B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis. *3D Gaussian Splatting for Real-Time Radiance Field
Rendering.* ACM TOG 42(4), SIGGRAPH 2023. Implementation: **msplat 1.1.4** (https://github.com/rayanht/msplat), Metal,
compiled (`_core.cpython-313-darwin.so`).

`scripts/gaussian_points.py`:
- **Scene:** the 8 images and VGGT's cameras (known focal) written as a COLMAP model, one PINHOLE camera per image
  (`:51-63`). Trained in VGGT's frame (`keep_crs=True`, `:68`).
- **Initialisation: TSDF points**, not VGGT's raw points: 100,000 points drawn uniformly at random (seed 0) from the
  200k-point TSDF cloud `<s>_tex_v3` (`--init-n 100000`, `:36`; `:58-59`), with their colours. The init size was chosen on
  pencilpot (`:36`). So the Gaussians are **not independent** of the TSDF cloud.
- **Iterations: 7,000** (`--iters`, `:35`; msplat's default is 30,000). Background white `[1, 1, 1]` (`:68`).
- **Densification:** `densify_grad_thresh = 0.0002` (`:39`, `:69`, equal to msplat's default). All other settings are
  msplat 1.1.4 defaults (read from `msplat.TrainingConfig()` on 16 Nov): `refine_every = 100`, `warmup_length = 500`,
  `reset_alpha_every = 30`, `densify_size_thresh = 0.01`, `split_screen_size = 0.05`, `stop_screen_size_at = 4000`,
  `num_downscales = 2`, `resolution_schedule = 3000`, `sh_degree = 3`, `sh_degree_interval = 1000`, `ssim_weight = 0.2`.
  **NOT VERIFIED:** what exactly these do inside msplat (pruning rule, opacity-reset value, when densification stops);
  the names follow OpenSplat/gsplat conventions, but the code is compiled. msplat exposes no opacity-pruning threshold.
- **Final Gaussian count** (`gaussians.json`, 20 shapes): **17,055 to 37,495, median 26,753.** Training ~10 s per object.
  Not bit-reproducible (the same settings gave 16.0k and 16.2k Gaussians on pencilpot, `docs/results/2026-11-09_v3_gaussians.md`).
- **Kept for points** (`:79-81`; `src/photo2mesh/points/gaussians.py:99-110`): opacity = sigmoid(stored logit) >= **0.5**;
  largest axis = max exp(stored log-scale) <= **0.05 x object size** (object size = 1st-99th percentile bbox of the
  initial cloud, `:79`). Kept: 7,906 to 17,560 per shape. Opacity 0.5 and scale 0.05 were chosen on pencilpot.

## 5. Gaussian kernel sampling

`src/photo2mesh/points/gaussians.py:128-162`, called by `scripts/gaussian_kernel_points.py:46-47`:
- Same filter as the centres (opacity >= 0.5, largest axis <= 0.05 x object size; `gaussians.py:141-144`).
- **Which Gaussian:** each sample picks a Gaussian independently, with probability proportional to its **opacity**
  (`weight="opacity"`, default; `:147`, `:153-154`). So the number of points per Gaussian is **not fixed**: it is
  multinomial with expectation n x alpha_i / sum(alpha). At n = 1M from 7,906 to 17,560 Gaussians, that is **about 83 points
  per Gaussian on average**.
- **Where in the Gaussian:** x = mu + R diag(s) z, z ~ N(0, I), i.e. a sample of **N(mu, Sigma)** with Sigma = R diag(s)^2 R^T
  (`:155`, `:160`). **Truncated per axis at 2 sigma**: any |z_i| > 2 is redrawn until all are within 2 (`:156-159`;
  `--trunc 2.0`), so the support is a box of +-2 sigma along the Gaussian's own axes (not an ellipsoid).
  Quaternions are normalised first (`:118-119`).
- **Seed:** `numpy.random.default_rng(0)` (`gaussian_kernel_points.py:47`).
- **Normals:** the PCA normal of the source Gaussian's centre, oriented towards the nearest camera
  (`--normals centre`, default, `gaussian_kernel_points.py:35`, `:51-58`); chosen on pencilpot (NC 0.78 vs 0.69 for the
  shortest axis).
- Counts used: 200,000 (`_gsk`, gap fill of 14 Nov) and 1,000,000 (`_gskd`, dense gap fill of 15 Nov).

## 6. Gap fill

`src/photo2mesh/points/fill.py:8-24`, `scripts/fill_points.py`:
- Both clouds are first aligned to the GT frame with the same camera similarity (section 3).
- **Measured against:** the **nearest TSDF point** (k-d tree on the TSDF cloud, `fill.py:18`), not the TSDF surface or
  its SDF. Distances are divided by the **TSDF cloud's own largest bbox extent** (`fill.py:17`); no GT is used.
- **Rule:** keep a kernel sample when 0.03 < d / extent <= 0.10 (`--lo 0.03 --hi 0.1`, `fill_points.py`; `fill.py:19`).
  No neighbour filter (`min_nbrs = 0`). The band was chosen on pencilpot only (`docs/results/2026-11-14_gap_fill.md`).
- **Output** = all TSDF points + the kept kernel samples, with normals.
- **Point counts** (dense version: 1M TSDF points + 1M kernel samples; and the 200k version):

| shape | TSDF (1M base) | kernel points added | total (dense) | added (200k version) |
|---|---:|---:|---:|---:|
| pencilpot_299291 (dev) | 1,000,000 | 76,768 | 1,076,768 | 15,974 |
| cylinder_472094 (A) | 1,000,000 | 51,277 | 1,051,277 | 10,551 |
| shackbot_94155 (A) | 1,000,000 | 167,259 | 1,167,259 | 34,451 |
| nsphere_186199 (A) | 1,000,000 | 59,282 | 1,059,282 | 12,677 |
| wheel_76903 (A) | 1,000,000 | 63,199 | 1,063,199 | 12,929 |
| presents_98021 (A) | 1,000,000 | 89,886 | 1,089,886 | 18,537 |
| monsterring_1356640 (A) | 1,000,000 | 10,724 | 1,010,724 | 2,298 |
| cryptex_105686 (A) | 1,000,000 | 11,321 | 1,011,321 | 2,389 |
| thingi_1146183 (B) | 1,000,000 | 70,616 | 1,070,616 | 14,686 |
| thingi_84620 (B) | 1,000,000 | 52,595 | 1,052,595 | 11,009 |
| thingi_77949 (B) | 1,000,000 | 82,028 | 1,082,028 | 17,304 |
| thingi_593381 (B) | 1,000,000 | 81,602 | 1,081,602 | 17,269 |
| thingi_73155 (B) | 1,000,000 | 51,746 | 1,051,746 | 10,601 |
| thingi_40550 (B) | 1,000,000 | 79,667 | 1,079,667 | 16,492 |
| thingi_290506 (B) | 1,000,000 | 109,037 | 1,109,037 | 22,395 |
| thingi_105924 (B) | 1,000,000 | 69,269 | 1,069,269 | 14,459 |
| thingi_103356 (B) | 1,000,000 | 333,399 | 1,333,399 | 68,793 |
| thingi_59706 (B) | 1,000,000 | 41,551 | 1,041,551 | 8,552 |
| thingi_697228 (B) | 1,000,000 | 148,347 | 1,148,347 | 30,582 |
| thingi_94668 (B) | 1,000,000 | 64,779 | 1,064,779 | 13,289 |

## 7. Selecting 4,096 points for MeshFlow

**Default path (random):** our adapter calls `pipe.sample_surface_points(path)` (`src/photo2mesh/mesh/meshflow_cvpr.py:87`)
-> `Mesh.load_rope_points` (`third_party/meshflow/meshflow/pipelines/meshflow_pipeline.py:275-282`) ->
1. read xyz only (`third_party/meshflow/meshflow/utils/mesh.py:115-131`; normals and colours are dropped);
2. normalise the **full** cloud: bbox centre to the origin, farthest point at radius 1 ("bsphere", size 2;
   `mesh.py:42-52`, `:428`);
3. **random selection without replacement**: `torch.randperm(N, device=...)[:4096]` (`mesh.py:67-75`, `:429`;
   4,096 = `data.n_verts`, `ckpt/meshflow/meshflow/config.yaml:2`).

- **Seeded: yes.** Our adapter calls `torch.manual_seed(seed)` immediately before (`meshflow_cvpr.py:84`), seed 42 by
  default (`configs/default.yaml:34`). In the hybrid setting the points live on MPS, so the permutation is drawn by the
  MPS generator (`meshflow_cvpr.py:71-72`, `meshflow_pipeline.py:282`). **NOT VERIFIED:** that MPS `randperm` is
  bit-reproducible across runs and torch versions.
- **Same subset for every method: no.** The permutation depends only on (seed, N, device), so two clouds with the same
  N get the same *indices*, but those index different points; clouds with different N (200k, 1M, 1.08M gap fill) get
  unrelated permutations. There is no shared subset across methods.
- The generation noise does **not** depend on this pick: `sample_latent` uses its own generator,
  `torch.Generator(device).manual_seed(seed)` (`meshflow_pipeline.py:371`). (Our 13 Nov FPS report said the opposite;
  that sentence was wrong and is corrected there.)

**13 Nov FPS path** (`scripts/fps_points.py`, `src/photo2mesh/points/fps.py:7-26`): exact farthest-point sampling (greedy;
start = the point farthest from the centroid, `fps.py:18`; deterministic, no seed) of **exactly 4,096** points from the
aligned cloud (or the concatenation of TSDF and kernel clouds, `fps_points.py:37-40`). With N = 4,096 the `randperm` above
is only a permutation, so every FPS point reaches MeshFlow. Result: same accuracy, ~3,800 vertices instead of ~900
(`docs/results/2026-11-13_fps_input.md`); not used by default.

## 8. MeshFlow

**Citation:** W. Li, A. Toisoul, T. Monnier, R. Shapovalov, ..., R. Ranjan, P. Tan, A. Vedaldi. *MeshFlow: Efficient
Artistic Mesh Generation via MeshVAE and Flow-based Diffusion Transformer.* CVPR 2026 (Highlight). arXiv:2606.04621v2.
Code: facebookresearch/meshflow (`third_party/meshflow`, inference only), weights `facebook/meshflow`.

**Input spec (as we run it):** a point cloud file; only xyz is read (`mesh.py:115-131`); 4,096 points after the random
selection of section 7; **voxelised at 32^3** (`voxelize_resolution: 32`, `config.yaml:13`): each point is replaced by
the integer index of its cell in [-1, 1]^3, `floor((p + 1) / 2 x 32)` clipped to 0..31
(`third_party/meshflow/meshflow/models/meshflow_dit.py:19-36`; `meshflow_pipeline.py:293-296`). Only these indices condition
the DiT, as 3D rotary position embeddings. One cell = 1/16 of the bounding-sphere diameter (3.2% to 4.8% of the object's
largest extent on our shapes). No image condition (zero visual condition, `meshflow_cvpr.py:89`).

**Sampling:** flow-matching DiT, 28 steps (`configs/default.yaml:33`), guidance 1.0 (`meshflow_cvpr.py:89-91`; the
checkpoint's default is 2.5, `config.yaml:11`; at 1.0 classifier-free guidance is off, `meshflow_pipeline.py:369`),
4,096 latent tokens of 64 channels (`config.yaml:22-23`, `:47`). DiT on MPS, VAE decoder and extraction on CPU, fp32
(`meshflow_cvpr.py:53-58`, `:69-74`).

**Output spec / mesh extraction** (`third_party/meshflow/meshflow/models/meshflow_vae.py:725-756`): the VAE decodes each
token into a vertex position, normal, mask logit and adjacency embedding. Vertices = tokens with **mask > 0** (`:739`).
Edge (i, j) when the "spacetime" distance d = |x_s,i - x_s,j|^2 - |x_t,i - x_t,j|^2 (first / second half of the
embedding, `:720-723`) satisfies sigmoid(d - tau) > 0.5, i.e. d > tau, with **tau = 0.6** fixed (`config.yaml:28-29`;
`:749-750`). Faces = cycles of 3 to 20 edges in that graph (`max_cycle_size=20`, `:729`; `mesh.py:206-275`), hole filling off
(`meshflow_cvpr.py:98`). The mesh is mapped back with the exact inverse of the input normalisation (`meshflow_cvpr.py:108-109`).
**No marching cubes, no QEM or decimation.**

**Vertex count:** at most the number of tokens, **4,096** (one vertex per kept token; `input_size: 4096`,
`config.yaml:47`; `meshflow_vae.py:739`). Observed over our 369 MeshFlow meshes: 53 to 4,056, never above 4,096 (after
protocol v1 merges duplicate vertices, `src/photo2mesh/eval/protocol.py:60`). The count is the mask head's decision:
~250 on clean GT samples, ~900 on TSDF points, ~3,800 on FPS input. **It cannot be requested with the released
checkpoint:** the `num_verts` control needs `denoiser_model.use_proj_cond_on_temb: true` and a trained
`t_embedder.cond_proj` layer (`meshflow_dit.py:722-730`; `meshflow_pipeline.py:119-132`). The config does not set the flag
(`config.yaml:46-60`), and `model.pth` has no `cond_proj` weights (checked 15 Nov by listing its keys).

**Training loss (paper only; the repo has no training code).** Quoted from arXiv:2606.04621v2 (HTML downloaded 17 Nov;
text and equations checked verbatim against the page source, including the LaTeX of Eq. 3-5):
- MeshVAE (§3.2 "Training Objective", Eq. 3-4): "The VAE is trained to minimize the sum of reconstruction loss
  L_rec and KL regularization loss L_kl"; L = L_mask + L_v + L_n + L_adj + lambda_kl L_kl, with BCE on the vertex mask,
  MSE on valid vertices and normals, and an adjacency loss
  L_adj = L_pos + lambda_neg R_neg/pos L_neg, L_pos = -1/|E| sum_{(i,j) in E} log sigma(d(e_i, e_j) - tau),
  L_neg = -1/|not E| sum_{(i,j) not in E} log sigma(tau - d(e_i, e_j)). lambda_kl = 1e-3 (arXiv appendix §F; CVPR supplementary §A: "the KL divergence weight is λkl = 10−3").
  The text calls L_rec "a weighted sum of four distinct components", but Eq. 3 writes L_mask, L_v, L_n and L_adj with no
  weights (only lambda_kl and, inside L_adj, lambda_neg = 0.01 per the supplementary Section A).
- DiT (§3.4, Eq. 5): conditional flow matching [Lipman et al., ICLR 2023],
  L_CFM = E_{t, x0, eps} || v_theta(RoPE3D(x_t, c_vox), t) - (eps - x0) ||^2 with x_t = (1 - t) x0 + t eps; logit-normal
  timestep sampling "during the final phase of training".

**The 4,096 vs 32,768 discrepancy: the paper's model and the released checkpoint differ.** Sources, quoted exactly:
- **CVPR 2026 supplementary, Section A "Additional Implementation Details"** (`MeshFlow_CVPR2026_supplemental.pdf`, page
  1 of 5; local copy in `docs/papers/`, not in git):
  - Data: "We utilize a proprietary dataset of approximately 600k high-quality 3D models from professional artists."
  - Shape condition: "We uniformly sampled 32,768 points for shape conditioning, which were then fed into a similarly
    pre-trained shape encoder with a structure similar to Shape2VecSet [7] to obtain 2048 tokens. The resulting tokens
    and the vertex count serve as conditional inputs for our Diffusion Transformer."
  - DiT: "The DiT uses the same standard transformer with 18 blocks and a 1024 hidden size, amounting to 427M
    parameters."
  - RoPE: "The RoPE can be applied to either the center coordinates or the indices of the encoded point set. [...] The
    results presented in the paper therefore employ indices for RoPE."
- **Main paper, arXiv:2606.04621v2 §3.4 "The Choice of Shape Information Injection"** (HTML, downloaded 17 Nov and
  read verbatim; the CVPR camera-ready main paper was not available to compare): "In our initial design, we uniformly
  sampled 32768 points for shape conditioning and then fed these points into a pre-trained shape encoder with an
  architecture analogous to Shape2VecSet [51], yielding 2048 shape tokens. We then leveraged cross-attention (CA) to
  condition our DiT on these tokens and further added a vertex count condition in the tilmestep embedding. We found
  that although this paradigm facilitates straightforward inference, yet it requires prolonged training to ensure good
  model performance." It then describes the GT-vertex RoPE scheme: "we coarsely voxelize the ground truth vertices
  during training to provide coarse shape priors in the RoPE embeddings. Ultimately, we found that a voxel resolution
  of 32 strikes a reasonable trade-off between having an artist-like topology and keeping shape details." The paper
  calls the 32,768-point encoder its "initial design"; it does not say in so many words that the encoder was dropped.
- **arXiv v2 appendix §F** (the arXiv counterpart of Section A) does not contain the 32,768-point sentence and gives a
  different DiT: "The DiT uses the same standard transformer with 16 blocks and a 1536 hidden size, amounting to 895M
  parameters." The main text §4.1 of the same arXiv version says "18 blocks and a 1024 hidden layer size, amounting to
  427M parameters".
- **Released code and checkpoint:** the DiT has `layers: 16`, `width: 1536` (`ckpt/meshflow/meshflow/config.yaml:51-52`)
  and 895.6M parameters (counted from `model.pth`, 17 Nov), i.e. the arXiv §F model, not the 427M model of the CVPR
  supplementary and §4.1. `model.pth` holds only `mesh_model` (the MeshVAE) and `denoiser_model` (the DiT): **no
  point-cloud shape encoder, no 2048-token cross-attention condition and no trained vertex-count layer** (see vertex
  count above). The input is 4,096 points (`data.n_verts: 4096`, `config.yaml:2`; `denoiser_model.input_size: 4096`,
  `config.yaml:47`; README: "Input meshes should respect the configured vertex budget (`mesh_model.num_latents`, 4096
  by default)", `third_party/meshflow/README.md:190`), voxelised at 32^3 for the RoPE positions.
- **Plainly: the model described in the CVPR supplementary (32,768 uniform points -> Shape2VecSet-like encoder -> 2048
  tokens, plus the vertex count as a condition, 427M-parameter DiT, trained on ~600k proprietary artist meshes) is not
  the released checkpoint** (4,096 points voxelised at 32^3 as RoPE positions, no encoder, no vertex-count input,
  895.6M-parameter DiT). Numbers in the paper therefore do not directly describe the model we run; we can only cite the
  paper for the method family and the released code for what we use.
- Also in the paper (§3.4): "We also provide the number of vertices concatenated with the time embedding". The released
  checkpoint does not have this input (see vertex count above).

## 9. Metrics: protocol v1

`src/photo2mesh/eval/protocol.py`, `docs/eval_protocol.md`. Prediction already in the GT frame (camera alignment or
MeshFlow's exact inverse; never ICP).
- **Normalisation** (`protocol.py:21-24`): GT and prediction together, x -> (x - c) / L with c = GT bbox centre and L =
  GT's largest bbox extent, so the GT fits [-0.5, 0.5] (as in MeshFlow's evaluation).
- **Chamfer and Hausdorff** (`:27-31`): CD = mean_a min_b |a - b| + mean_b min_a |a - b| (L2, not squared), HD = the
  larger of the two maxima; both **x100** (`:67`, `:88`). **5,000** samples per side (`N_CD`, `:15`; MeshFlow's
  `evaluate.py` default).
- **F1, precision, recall** at **tau = 0.01 and 0.02** (meshes, `:17`) and also **0.005** (points, `:18`); **NC** = mean
  |cos| between each sample's normal and its nearest neighbour's on the other side, symmetrised (`:38`). On **100,000**
  samples per side (`N_DENSE`, `:16`).
- **Sampling:** meshes are sampled area-uniformly (`src/photo2mesh/eval/metrics.py:15-25`), prediction seed 0, GT seed 1
  (`protocol.py:64-66`). Point clouds: random subsets of the cloud stand in for surface samples, 5,000 and 100,000
  points, seed 0 (`:83-89`), so a denser cloud of the same surface scores the same.
- **Density-sensitive extras for points** (`:91-96`): P_all, R_all, F1_all at each tau using **every** point of the
  cloud against the 100,000 GT samples.
- **Coverage** (`scripts/cloud_stats.py`): share of the GT surface within 1% of the footprint of the cloud's inliers.
- Mesh-only: vertex and face counts after merging duplicates (`:60`, `:68`), triangle quality G (`:48-55`), size ratio.

## Summary table

| stage | method | citation | key parameters (code) | our own choice? |
|---|---|---|---|---|
| Data | Thingi10K STL meshes, 20 shapes | Zhou and Jacobson 2016 | dev 1 + set A 7 + set B 12; unit sphere, Y-up | yes (shape choice; set B selection NOT VERIFIED) |
| Rendering | pyrender, 2 rings x 8 | - | 518 px, yfov 40 deg (f = 711.6 px), elev 20/50 deg, r = 3, white bg | yes |
| Views | 8 of 16 | - | ids 0,2,4,6,9,11,13,15 | yes (Mac memory rule; 16 views tested 15 Nov) |
| Cameras + depth | VGGT-1B | Wang et al., CVPR 2025 | depth head + pose head, 518 px, fp32; point head unused | VGGT given; known focal 711.6 px is ours |
| Pixel filter | confidence + rim erosion | - | conf >= 50th pct and > 1.0; 2 px erosion; no multi-view check | yes (tuned on pencilpot) |
| Fusion | TSDF, Open3D VoxelBlockGrid (CPU) | Curless and Levoy 1996; Newcombe et al. 2011 (integration only); Zhou et al. 2018 | voxel 0.005 x size, trunc 6 voxels, depth_max 1.5 x max depth, weights Open3D default (NOT VERIFIED) | yes (values tuned on pencilpot) |
| Surface | Open3D extract_triangle_mesh | Lorensen and Cline 1987 (marching cubes; NOT VERIFIED in code) | weight threshold 0.5 (>= 1 view); drop pieces < 1% area | yes (min_views, min_part) |
| Points | area-uniform random surface samples | - | n = 1,000,000, seed 0, face normals | yes (1M since 15 Nov, Prof. Zhong) |
| Alignment | Umeyama similarity on camera centres | Umeyama 1991 | no ICP | yes |
| 3DGS | msplat 1.1.4 (Metal) | Kerbl et al. 2023 | init 100k TSDF points, 7,000 it, densify grad 2e-4, other msplat defaults; 17k to 37k Gaussians | yes (init, iters); internals NOT VERIFIED |
| Kernel points | sample N(mu, Sigma), per-axis trunc 2 sigma | - | opacity >= 0.5, max axis <= 0.05 x size, pick prob. proportional to opacity, seed 0, ~83 pts per Gaussian at 1M | yes |
| Gap fill | distance band to nearest TSDF point | - | 3% < d / extent <= 10%, no GT | yes (band chosen on pencilpot) |
| 4K selection | MeshFlow's randperm (default) or FPS | Qi et al. 2017 (FPS, PointNet++) | bsphere normalise, 4,096, seed 42 (global torch RNG, MPS) | randomness MeshFlow's; seed ours; FPS ours (not default) |
| Mesher | MeshFlow (CVPR 2026) | Li et al. 2026 | 32^3 voxel RoPE, 28 steps, guidance 1.0, tau 0.6, cycles <= 20, V <= 4,096 | guidance 1.0 and device split ours; released checkpoint differs from the paper's model (section 8) |
| Metrics | protocol v1 | MeshFlow evaluation (CD/HD x100, 5k) | [-0.5, 0.5] by GT bbox; CD/HD 5k; F1/NC 100k at 0.005/0.01/0.02 | yes (protocol), CD definition MeshFlow's |
