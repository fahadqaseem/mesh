import numpy as np
from scipy.spatial.transform import Rotation

from photo2mesh.eval.align import (align_by_cameras, apply_sim3, camera_centres_from_extrinsic,
                                   icp_sim3, umeyama)


def _random_sim3(seed=0):
    rng = np.random.default_rng(seed)
    return float(rng.uniform(0.3, 3.0)), Rotation.random(random_state=seed).as_matrix(), rng.normal(size=3)


def test_umeyama_recovers_known_similarity():
    pts = np.random.default_rng(1).normal(size=(50, 3))
    s, r, t = _random_sim3()
    s2, r2, t2 = umeyama(pts, apply_sim3(pts, s, r, t))
    assert abs(s2 - s) < 1e-5 and np.allclose(r2, r, atol=1e-5) and np.allclose(t2, t, atol=1e-4)


def test_camera_alignment_from_extrinsics():
    # GT cameras on a ring; "predicted" cameras are the same rig in another frame and scale
    a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    gt_centres = np.c_[3 * np.sin(a), np.where(np.arange(16) < 8, 1.0, 2.3), 3 * np.cos(a)]
    gt_c2w = np.tile(np.eye(4), (16, 1, 1))
    gt_c2w[:, :3, 3] = gt_centres
    s, r, t = _random_sim3(2)
    pred_centres = apply_sim3(gt_centres, s, r, t)
    # build extrinsics (camera-from-world) whose centres are pred_centres, and check the round trip
    rots = Rotation.random(16, random_state=3).as_matrix()
    ext = np.concatenate([rots, -np.einsum("sij,sj->si", rots, pred_centres)[:, :, None]], axis=2)
    assert np.allclose(camera_centres_from_extrinsic(ext), pred_centres, atol=1e-5)
    sim, rms = align_by_cameras(camera_centres_from_extrinsic(ext), gt_c2w)
    assert rms < 1e-4
    assert np.allclose(apply_sim3(pred_centres, *sim), gt_centres, atol=1e-4)


def test_icp_refines_small_misalignment_with_outliers():
    rng = np.random.default_rng(4)
    dst = rng.normal(size=(3000, 3))
    dst /= np.linalg.norm(dst, axis=1, keepdims=True)
    dst *= np.array([1.0, 0.6, 0.4])                               # ellipsoid: no symmetric ambiguity
    small = (1.03, Rotation.from_euler("xyz", [4, -3, 5], degrees=True).as_matrix(), np.array([0.03, -0.02, 0.01]))
    src = apply_sim3(dst[:2000], *small)
    src = np.vstack([src, rng.uniform(-2, 2, size=(100, 3))])      # 5% outliers
    sim, err = icp_sim3(src, dst, iters=50)
    back = apply_sim3(src[:2000], *sim)
    assert np.abs(back - dst[:2000]).max() < 0.02
