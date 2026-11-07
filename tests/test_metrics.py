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


def test_points_on_the_surface_score_perfectly_and_face_outward():
    import numpy as np
    from photo2mesh.eval.metrics import evaluate_points, sample_surface
    gt = trimesh.creation.icosphere(subdivisions=4)
    pts, nrm = sample_surface(gt, 50_000, seed=1)
    r = evaluate_points(pts, nrm, gt, n=20_000, tau=0.05)   # tau above sample spacing (~0.025)
    assert r["f1"] > 0.99 and r["normal_outward"] > 0.99
    r_flip = evaluate_points(pts, -nrm, gt, n=20_000, tau=0.05)
    assert r_flip["normal_outward"] < 0.01 and abs(r_flip["normal_cons"] - r["normal_cons"]) < 1e-6
    half = pts[:, 0] > 0                       # only half the object seen: precision ok, recall ~half
    r_half = evaluate_points(pts[half], nrm[half], gt, n=20_000, tau=0.05)
    assert r_half["precision"] > 0.99 and r_half["recall"] < 0.6
