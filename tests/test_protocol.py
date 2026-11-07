import numpy as np
import trimesh

from photo2mesh.eval.protocol import chamfer_hausdorff, score_mesh, score_points, triangle_quality


def test_chamfer_hausdorff_definition():
    # matches point_cloud_utils (MeshFlow's evaluate.py): mean L2 both ways, summed; Hausdorff = max of maxima
    rng = np.random.default_rng(0)
    a, b = rng.random((500, 3)), rng.random((700, 3)) * 1.3
    cd, hd = chamfer_hausdorff(a, b)
    assert abs(cd - 0.244974) < 1e-5      # value from point_cloud_utils 0.34.0 on this exact data
    assert abs(hd - 0.524014) < 1e-5


def test_identical_mesh_scores_well_and_scale_errors_count():
    gt = trimesh.creation.icosphere(subdivisions=4)
    good = score_mesh(gt, gt)
    # CD has a sampling floor (~2.5 here at 5k samples, x100), so "good" means close to that floor, not zero
    assert good["CD"] < 3.0 and good["F1@0.02"] > 0.95 and good["NC"] > 0.98 and abs(good["size_ratio"] - 1) < 1e-6
    small = gt.copy()
    small.apply_scale(0.86)                # the shrink we saw with MeshFlow
    bad = score_mesh(small, gt)
    assert bad["CD"] > 2 * good["CD"] and bad["F1@0.02"] < 0.1 and abs(bad["size_ratio"] - 0.86) < 1e-3


def test_points_on_surface_score_well():
    gt = trimesh.creation.icosphere(subdivisions=4)
    pts, nrm = trimesh.sample.sample_surface(gt, 200_000, seed=5)
    r = score_points(pts, gt.face_normals[nrm], gt)
    assert r["F1@0.02"] > 0.95 and r["normal_outward"] > 0.99


def test_triangle_quality():
    eq = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0]], [[0, 1, 2]])
    sliver = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0.5, 0.01, 0]], [[0, 1, 2]])
    assert abs(triangle_quality(eq) - 1.0) < 1e-6
    assert triangle_quality(sliver) < 0.05
