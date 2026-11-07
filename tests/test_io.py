import numpy as np

from photo2mesh.io import load_points_ply, save_points_ply
from photo2mesh.points.filter import normalize_to_unit_sphere, voxel_downsample


def test_ply_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    p = rng.normal(size=(100, 3)).astype(np.float32)
    n = rng.normal(size=(100, 3)).astype(np.float32)
    c = rng.uniform(size=(100, 3)).astype(np.float32)
    path = tmp_path / "x.ply"
    save_points_ply(path, p, normals=n, colors=c)
    d = load_points_ply(path)
    assert np.allclose(d["points"], p, atol=1e-6)
    assert np.allclose(d["normals"], n, atol=1e-6)
    assert np.allclose(d["colors"], c, atol=1 / 255 + 1e-6)


def test_normalize_unit_sphere():
    p = np.random.default_rng(0).normal(size=(500, 3)) * 7 + 3
    q, _, _ = normalize_to_unit_sphere(p)
    assert np.linalg.norm(q, axis=1).max() <= 1.0 + 1e-6


def test_voxel_downsample_reduces():
    p = np.random.default_rng(0).uniform(size=(5000, 3))
    out = voxel_downsample(p, 0.1)
    pts = out[0] if isinstance(out, tuple) else out
    assert len(pts) <= 1000
