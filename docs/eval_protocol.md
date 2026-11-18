# Evaluation protocol (v1, 2026-11-07)

Decided by Fahad on 2026-11-07: use the MeshFlow paper's protocol where it states one; otherwise unit-sphere
normalisation, 100k samples, F1 at tau 0.01 and 0.02, Chamfer, normal consistency and vertex count. Scale errors stay
in the score. Implemented in `src/photo2mesh/eval/protocol.py`. Every results table uses this protocol.

## 1. Getting the prediction into the GT frame (no re-fitting)
- Test scenes are rendered from `<scene>/gt_normalized.ply`; the render cameras `gt_cam_poses.npy` live in that frame.
- **Points (VGGT):** one similarity transform (scale, rotation, translation) fitted between the VGGT camera centres and the
  GT render camera centres (Umeyama). This is the only fit. No ICP. (`scripts/align.py --icp none`, now the default.)
- **Meshes (MeshFlow):** MeshFlow normalises its input cloud itself (bounding-box centre, max radius 1). The output mesh is
  mapped back by the exact inverse of that normalisation (computed from the input cloud, not fitted to the GT). No ICP.
- So any scale, position or orientation error of the pipeline counts against it.

## 2. Normalisation for scoring
- MeshFlow paper: "the Chamfer Distance (CD) between the reconstructed mesh and the GT mesh, which are normalized within
  [−0.5,0.5]"; its `evaluate.py` centres the GT on its bounding-box centre and scales the max extent to 1, and applies the
  same transform to the prediction.
- We use that one frame for **every** metric (GT bbox centre, GT max extent = 1, same transform on the prediction).
  F1 thresholds below are in this frame.

## 3. Metrics (every table)
| column | definition | samples | source |
|---|---|---|---|
| `CD` | Chamfer as in MeshFlow: mean L2 nearest-neighbour distance pred->GT plus GT->pred (not squared), x100 | 5,000 per surface | MeshFlow paper + `evaluate.py` (`point_cloud_utils.chamfer_distance`; ours matches it, see tests) |
| `HD` | Hausdorff: max of the two directed max nearest-neighbour distances, x100 | 5,000 per surface | MeshFlow paper (Table 2) |
| `NC` | normal consistency: mean abs cosine between matched normals, both directions averaged | 100,000 | Fahad (MeshFlow omits NC) |
| `F1@0.01`, `F1@0.02` | F-score of points within tau of the other surface (precision pred->GT, recall GT->pred) | 100,000 | Fahad |
| `#V`, `#F` | vertex and face count of the predicted mesh (after merging duplicate vertices) | - | Fahad, HD-PEA |
| `G` | mean triangle quality G = 2*sqrt(3)*S/(p*h) (S area, p half-perimeter, h longest edge; 1 = equilateral) | all faces | HD-PEA (isotropic criterion) |
| `time` | wall-clock seconds for the stage (MeshFlow: sampling + mesh extraction, as in the MeshFlow paper) | - | MeshFlow, HD-PEA |
- Point clouds (stage 1) get the same metrics except `#F` and `G`; `#V` is the number of points.
- Surface samples use fixed seeds (5,000-sample metrics: seed 0 / 1 for pred / GT; 100k metrics: seed 0 / 1).
- `CD floor` and `F1 ceiling`: what a perfect reconstruction scores (two independent GT samplings). At 5,000 samples the
  CD floor is ~2-2.5 (x100) for our shapes, so CD differences near the floor mean little. (MeshFlow's own Toys4K CD of
  2.33 is close to such a floor.) Both are reported per shape next to the scores.
- Tables give mean and spread (std, min-max) across shapes, and across seeds where there are several.

## 4. What we do not match exactly (known differences)
- MeshFlow samples 5,000 points from the GT on Toys4K and does not say how many from the prediction; we use 5,000 for both.
- MeshFlow does not state the input point count or noise for its shape-conditioned results; our input is the VGGT cloud
  (capped at 200k points; MeshFlow resamples 4,096).

## 5. HD-PEA paper (for later comparison; no dependency on it now)
"Learning Manifolds in High-D Point Embedding for Anisotropic Surface Approximation from Unstructured Point Clouds",
Li, Zhu, Guo, Wang, Hua, Zhong, arXiv 2607.28855.
- Accuracy: CD (reported x10^-5, so likely squared distances; not defined in the text), F1 (threshold not stated),
  NC, HD (x10^-2).
- Mesh quality: triangle quality G = 2*sqrt(3)*S/(p*h) averaged over triangles (isotropic criterion), plus an anisotropic
  version using the metric Q = sqrt(M). Element counts #V_out, #f_out. Average runtime.
- Data: 80 Thingi10K test models (40K input points), 75 synthetic scans (Myles et al.), plus TetWeave and ScanNet.
  Noise tests: Gaussian, std 0.001 / 0.003 / 0.005. The normalisation for evaluation is not stated.
- To compare later: also report squared CD (`cd_sq` is kept in the JSON files), and ask for or infer their F1 threshold.

## 6. Evaluation consistency, reports of 16-19 Nov (checked 17 Nov)

Asked by Prof. Zhong (email of 17 Nov). `scripts/eval_audit.py` checks this from the files on disk, for every cloud each
report scores:
- the alignment file has no ICP step;
- the score file says 1,000,000 points, seed 0;
- the GT mesh is the same file.

**The same for every protocol v1 row of these reports:**
- **Normalisation:** GT bbox centre, GT largest extent = 1, the same transform on the prediction
  (`protocol.py` `unit_cube_transform`).
- **Alignment:** one Umeyama sim3 on VGGT vs GT camera centres (`scripts/align.py`, no ICP), computed in each run's own
  folder. A rerun of VGGT (`_ds`, `_c50`, `_pm`) gives the same cameras and the same sim3 as `<s>_dense`.
- **Point count:** every cloud is scored on exactly 1,000,000 points drawn with `numpy.random.default_rng(0)`
  (`scripts/equal_count.py` `score`, called directly or through `ds_report.score_new`). Clouds of more than 1M points
  lose the excess at random. The TSDF clouds have exactly 1M points for any view count, so for them the draw only
  reorders.
- **Metrics and samples:** protocol v1 `score_points`:
  - Chamfer / Hausdorff on 5,000 cloud points vs 5,000 GT samples (seed 1);
  - F1 and NC on 100,000 vs 100,000 (GT seed 1);
  - "recall all" with every point vs the 100,000 GT samples.
- **Extra columns, computed the same way everywhere:**
  - outliers = exact distance to the GT triangles (Open3D `RaycastingScene`), on all 1M points;
  - signed offset = median signed distance along the GT face normal, on 100,000 points (`default_rng(0)`);
  - "GT gap filled" = GT surface samples (100,000, sampler seed 0) farther than 0.02 from the current 8-view cloud,
    tested against all points of the other cloud.

| report | clouds checked | alignment | points scored | n before the draw | GT | problems |
|---|---:|---|---|---|---|---|
| 2026-11-17_equal_count | 40 | cameras only (camera RMS 0.013-0.064) | 1,000,000, seed 0 | 1,000,000-1,333,399 | data/scenes | none |
| 2026-11-17_conf_ablation | 72 | cameras only (0.013-0.027) | 1,000,000, seed 0 | 1,000,000-1,333,399 | data/scenes | none |
| 2026-11-18_depth_scale | 72 | cameras only (0.008-0.034) | 1,000,000, seed 0 | 1,000,000 | data/scenes (+ scenes_low, identical) | none |
| 2026-11-18_depth_scale (pencilpot tuning) | 12 | cameras only (0.019) | 1,000,000, seed 0 | 1,000,000 | data/scenes | none |
| 2026-11-19_views_focal | 96 | cameras only (0.010-0.034) | 1,000,000, seed 0 | 1,000,000 | data/scenes (+ scenes_low, identical) | none |
| 2026-11-19_views_focal (pencilpot tuning) | 5 | cameras only (0.019) | 1,000,000, seed 0 | 1,000,000 | data/scenes | none |
| 2026-11-19_seeded_4k | 36 meshes | inputs from the camera-aligned `<s>_dense`; mesh mapped back by MeshFlow's exact inverse normalisation | mesh protocol: 5,000 / 100,000 surface samples | 4,096 input points | data/scenes | none (mesh rows, see exception 3) |
| 2026-11-18_outliers | 20 filled clouds per filter | cameras only; kernel samples moved with `<s>_gskd`'s sim3, identical to `<s>_dense`'s | TSDF 1M + kept gap points, not redrawn | 1,000,000 + gap points | data/scenes | exception 2 |

Camera RMS is in GT units, where the orbit radius is 3. Runs on the low-view rig (`_lowA`, `_lowB`, `_v12r3`) align
with `data/scenes_low/<s>`. Its `gt_normalized.ply` is byte-identical to `data/scenes/<s>`'s, and its first 16 render
cameras are identical, so these runs share the frame and the GT.

**Exceptions:**
1. **ICP rows are not protocol v1.** They are labelled "diagnostic" or "upper bound, NOT protocol v1" in
   2026-11-17_conf_ablation (§1), 2026-11-18_depth_scale and 2026-11-19_views_focal. They come from
   `scripts/icp_check.py`, which uses the GT to refine the alignment.
2. **2026-11-18_outliers scores TSDF 1M + the kept gap points without redrawing to 1M** (1.0M plus 0 to ~0.3M points,
   depending on the filter). Its filled clouds therefore have more points than the 1M rows elsewhere. The protocol
   metrics subsample (5k / 100k), so this affects only "recall all" and the outlier share. The 17 Nov equal-count report
   gives the 1M version of the unfiltered gap fill.
3. **2026-11-19_seeded_4k scores MeshFlow meshes,** with the mesh protocol (5,000 / 100,000 samples on both surfaces,
   `score_mesh`). Its numbers are not comparable with the point-cloud rows.
4. **The "cameras only" row of the ICP table** (2026-11-17_conf_ablation §1) is scored by `icp_check.py` on the cloud in
   file order, not on the seed-0 1M draw. Its 5,000-point Chamfer subsample therefore differs: Chamfer 5.59 vs 5.58 in
   every other report for the same cloud. The status table's 5.59 comes from the 15 Nov `points_eval.py` run (200k-era
   scorer, file order). This is the 5k-subsample noise (~0.1), not a different cloud.
5. **Wording fixed:** the 2026-11-19_views_focal header said "more-view clouds subsampled at random". The 12- and 16-view
   TSDF clouds already have exactly 1M points, so the draw only reorders them. The numbers do not change; the header
   (and `scripts/vf_report.py`) now say so.
6. **Two different GT sample sets.** The "GT gap" test uses the protocol sampler with seed 0. F1 and recall use seed 1.
   Both are 100,000 uniform samples of the same GT surface; neither is mixed with the other within a column.
