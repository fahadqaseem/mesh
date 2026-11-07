import trimesh

from photo2mesh.eval.metrics import evaluate_mesh


def test_identical_meshes_score_perfectly():
    m = trimesh.creation.icosphere(subdivisions=3)
    r = evaluate_mesh(m, m, n=20_000, tau=0.02)
    assert r["chamfer"] < 1e-3
    assert r["normal_cons"] > 0.98
    assert r["f1"] > 0.99


def test_shifted_mesh_scores_worse():
    gt = trimesh.creation.icosphere(subdivisions=3)
    pred = gt.copy()
    pred.apply_translation([0.2, 0, 0])
    good = evaluate_mesh(gt, gt, n=20_000)
    bad = evaluate_mesh(pred, gt, n=20_000)
    assert bad["chamfer"] > good["chamfer"]
    assert bad["f1"] < good["f1"]
