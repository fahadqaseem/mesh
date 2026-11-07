import numpy as np

from photo2mesh.points.normals import estimate_normals, orient_towards, pca_normals


def _plane(n=500, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1, 1, (n, 2))
    return np.c_[xy, np.zeros(n)].astype(np.float32)


def test_plane_normals_are_z():
    n = pca_normals(_plane(), k=16)
    assert np.allclose(np.abs(n[:, 2]), 1.0, atol=1e-3)


def test_orient_towards_camera():
    pts = _plane()
    n = -np.abs(pca_normals(pts, k=16))          # all pointing down
    cam = np.tile([0.0, 0.0, 5.0], (len(pts), 1))  # camera above
    out = orient_towards(n, pts, cam)
    assert (out[:, 2] > 0).all()


def test_sphere_normals_point_outward_without_cameras():
    rng = np.random.default_rng(1)
    p = rng.normal(size=(2000, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    n = estimate_normals(p.astype(np.float32), viewpoints=None, k=20)
    assert ((n * p).sum(1) > 0.9).mean() > 0.95
