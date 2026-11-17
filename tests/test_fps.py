import numpy as np
from scipy.spatial import cKDTree

from photo2mesh.points.fps import farthest_point_sample


def test_fps_unique_and_spread():
    rng = np.random.default_rng(0)
    # clumpy cloud: 90% of the points in a small blob, 10% spread over the unit cube
    pts = np.vstack([rng.normal(0, 0.02, (9000, 3)), rng.uniform(-0.5, 0.5, (1000, 3))])
    idx = farthest_point_sample(pts, 200)
    assert len(np.unique(idx)) == 200
    rand = rng.choice(len(pts), 200, replace=False)
    # FPS covers the cube far better than a random pick (smaller worst-case gap to the cloud)
    gap = lambda sel: cKDTree(pts[sel]).query(pts)[0].max()
    assert gap(idx) < 0.5 * gap(rand)


def test_fps_similarity_invariant_and_small_k():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(500, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    assert np.array_equal(farthest_point_sample(pts, 50), farthest_point_sample(2.5 * pts @ q.T + 1.0, 50))
    assert np.array_equal(farthest_point_sample(pts, 600), np.arange(500))


def test_weighted_fps_prefers_heavy_points():
    rng = np.random.default_rng(2)
    pts = rng.uniform(-1, 1, (4000, 3))
    w = np.where(pts[:, 0] > 0, 4.0, 1.0)            # right half 4x more important
    idx = farthest_point_sample(pts, 400, weights=w)
    assert (pts[idx, 0] > 0).mean() > 0.6             # unweighted FPS would give ~0.5
    assert len(np.unique(idx)) == 400


def test_surface_variation_plane_vs_noise():
    from photo2mesh.points.fps import surface_variation
    rng = np.random.default_rng(3)
    plane = np.c_[rng.uniform(0, 1, (2000, 2)), np.zeros(2000)]
    blob = rng.normal(5, 0.1, (2000, 3))
    v = surface_variation(np.vstack([plane, blob]), k=16)
    assert v[:2000].max() < 1e-6 and v[2000:].mean() > 0.1
