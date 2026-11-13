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
