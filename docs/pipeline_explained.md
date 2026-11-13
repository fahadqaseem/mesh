# How the pipeline works, and how each figure column was made

Written 12 Nov 2026 for the meeting with Prof. Zhong (Mon 16 Nov). Every statement below is checked against the code
in this repo (paths given) or MeshFlow's code in `third_party/meshflow` (read only, never edited).

## The figure

`docs/results/figures/2026-11-09_meshflow_outputs.png`: two shapes (pencilpot, the tuning shape; thingi_59706, a set B
test shape never used for tuning), six columns. The vertex counts in the column titles match the result files exactly
(checked 12 Nov). The picture itself was rendered outside this repo's scripts; the meshes it shows are these files:

| column | file shown | points that went into MeshFlow | marching cubes? | vertices (pencilpot / 59706) |
|---|---|---|---|---:|
| Input photo (1 of 8) | `data/scenes/<s>/images_tex/view_XXX.png` | (one of the 8 photos VGGT gets) | no | - |
| Before: filtered points | `outputs/<s>_tex_8v/mesh/meshflow_s42.ply` | v2 cloud: VGGT depth pixels after the filters (below), concatenated over the 8 views | no | 918 / 806 |
| Now: TSDF points | `outputs/<s>_tex_v3/mesh/meshflow_s42.ply` | v3 cloud: 200k points sampled from the TSDF-fused surface | **yes, in the points stage** (to get the fused surface the points are sampled from) | 982 / 753 |
| Gaussian centres | `outputs/<s>_gs/mesh/meshflow_s42.ply` | centres of the opaque, compact Gaussians of a 3DGS model trained on the 8 photos | no (the 3DGS is initialised from 100k v3 points, whose surface came from marching cubes) | 2,697 / 1,892 |
| Ceiling: perfect points | `outputs/<s>_gtsamples/mesh/meshflow_s42.ply` | 200k points sampled from the true surface (`scripts/gt_samples.py`, seed 3) | no | 361 / 158 |
| True shape | `data/scenes/<s>/gt_normalized.ply` | - (the Thingi10K mesh itself, only rotated, centred and scaled) | no | 8,988 / 536 |

All five MeshFlow columns use the same MeshFlow checkpoint, seed 42, 28 flow steps, guidance 1.0, and the same
scoring. **Only the input point cloud differs.** No column uses QEM or any other mesh simplification, anywhere.

## Step by step

### 0. Test scenes (`scripts/render_views.py`)
- GT mesh (Thingi10K STL) rotated to Y-up, centred on its bbox, scaled into the unit sphere -> `gt_normalized.ply`.
- Colour: a subdivided copy is painted with a procedural pattern (seeded 3D sine waves) for rendering only; the GT
  geometry is unchanged.
- 16 views with pyrender, 518 x 518 px, vertical FOV 40°, white background: two rings of 8 cameras at 20° and 50°
  elevation, radius 3. The cameras (`gt_cam_poses.npy`) are kept for alignment and scoring.

### 1. Points stage (`scripts/run_points.py`, `configs/default.yaml` points.*)
- **VGGT** (MPS, deterministic) on 8 of the 16 views (view ids 0, 2, 4, 6, 9, 11, 13, 15: 4 per ring; the Mac memory
  limit). Output per view: depth map, confidence, camera. Known focal: the depth maps are re-unprojected with the
  render focal (as EXIF would give for real photos).
- **Pixel filters** (both v2 and v3): confidence at or above the 50th percentile and above 1.0 (background sits at
  exactly 1.0); foreground silhouette eroded by 2 px (removes flying pixels at the object's outline).
- **v2, filtered points** (the "Before" column): additionally a multi-view check (a pixel is kept only if at least 2
  other views see a depth within 1% of its own at its projection), then statistical outlier removal (16 neighbours, 2 std), PCA normals
  (20 neighbours, oriented to the camera), and a random cap of 200k points. No surface is built.
- **v3, TSDF points** (default since 9 Nov, the "Now" column): the filtered depth maps (no multi-view check: the fusion
  averages the views instead) are fused in a TSDF volume with Open3D on the CPU (`src/photo2mesh/points/fusion.py`):
  voxel = 0.5% of the object size, truncation 6 voxels. **The surface is extracted with marching cubes**
  (Open3D `VoxelBlockGrid.extract_triangle_mesh`), pieces under 1% of the area are dropped, and 200k points are sampled
  area-uniformly from that surface, with face normals. This is the only marching cubes in the pipeline.
- **Gaussian centres** (`scripts/gaussian_points.py`): the 8 photos and VGGT's cameras are written as a COLMAP model,
  with 100k points of the v3 cloud as the initial cloud; msplat (Metal) trains 3D Gaussian splatting for 7,000
  iterations; the centres of Gaussians with opacity >= 0.5 and largest axis <= 5% of the object are the points
  (about 11k to 15k per object), PCA normals.
- **Gaussian kernels** (new, 12 Nov, `scripts/gaussian_kernel_points.py`): from the same trained Gaussians, 200k points
  drawn from each Gaussian's own covariance (Gaussian picked with probability proportional to its opacity, sample
  truncated at 2 sigma). Not in the figure.

### 2. Alignment (`scripts/align.py`)
One similarity transform (scale, rotation, translation) fitted between VGGT's camera centres and the GT render camera
centres (Umeyama). No ICP, so any scale or pose error of the pipeline stays in the score.

### 3. MeshFlow (CVPR 2026, `src/photo2mesh/mesh/meshflow_cvpr.py` wrapping `third_party/meshflow`)
What happens to the input cloud, in order:
1. **Only xyz is used.** Normals and colours in the `.ply` are ignored.
2. **Normalisation:** bbox centre to the origin, max distance from it scaled to 1 (`normalize_by="bsphere"`, size 2).
3. **Random resampling to 4,096 points** (`Mesh.load_rope_points` -> `_resample_point_cloud`: `torch.randperm`,
   seeded by our MeshFlow seed). `num_verts` / `n_verts` = 4,096 in `ckpt/meshflow/meshflow/config.yaml`.
   So a 200k-point cloud and a 1M-point cloud of the same surface give MeshFlow the same kind of input, and every
   point beyond 4,096 is thrown away.
4. **Voxelisation at 32^3** (`voxelize_pc`, `voxelize_resolution: 32`): each of the 4,096 points is replaced by the
   integer index of the voxel it falls in. A cell is 1/16 of the bounding sphere's diameter: 3.2% to 4.8% of the
   object's largest extent on our 20 shapes (median 3.6%), so about 6 to 9 times the TSDF voxel. Only these indices
   condition the model (as rotary position embeddings). Detail finer than a cell cannot reach MeshFlow.
5. **Generation:** a flow-matching transformer (DiT, 28 steps) samples 4,096 latent tokens. The VAE decoder turns
   each token into a candidate vertex: position, normal, a **mask** value and an adjacency embedding.
6. **Mesh extraction, no marching cubes:** tokens with mask > 0 become the vertices; an edge joins two vertices when
   the "spacetime" distance of their adjacency embeddings (squared distance of the space part minus squared
   distance of the time part) is above a fixed threshold tau (`compute_spacetime_distances`); faces are the short
   cycles (up to 20 edges) of that edge graph (`extract_mesh_from_verts_normals_edges`, hole filling off).
7. **Vertex count:** the number of tokens the mask head keeps (at most 4,096). This is MeshFlow's own generation, not
   a simplification of a dense mesh: there is no QEM / decimation step in MeshFlow or in our code. That is why the
   "ceiling" mesh has fewer vertices (361) than a noisy input gives (982 for TSDF, 2,697 for Gaussian centres): the
   model places more vertices when its voxel input looks rougher.
8. The mesh is mapped back by the exact inverse of step 2 (computed from the input cloud, not fitted).

### 4. Scoring (`src/photo2mesh/eval/protocol.py`, `docs/eval_protocol.md`)
GT and prediction normalised together into MeshFlow's [-0.5, 0.5] frame (GT bbox). Chamfer and Hausdorff x100 on
5,000 samples per surface; NC and F1 at 0.01 / 0.02 on 100,000. Point clouds are scored with the same metrics (random
subsamples of the cloud stand in for surface samples), plus coverage and point count; since 12 Nov also F1@0.005
and recall with all points (see `docs/results/2026-11-12_points.md`).
