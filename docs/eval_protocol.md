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
