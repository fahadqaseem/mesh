import numpy as np
import pytest
import trimesh

from photo2mesh.eval.metrics import sample_surface
from photo2mesh.eval.robustness import (add_noise, cloud_stats, farthest_point_centres, offset_along_normals,
                                        patch_mask, replace_with_outliers, visible_mask)


def _sphere():
    return trimesh.creation.icosphere(subdivisions=5, radius=1.0)    # L = 2, so protocol units are half the raw units


def test_noise_has_the_requested_sigma():
    rng = np.random.default_rng(0)
    out = add_noise(np.zeros((200_000, 3)), 0.02, rng)
    assert abs(out.std() - 0.02) < 2e-4


def test_outliers_replace_an_exact_fraction_inside_the_box():
    rng = np.random.default_rng(0)
    pts, nrm = np.zeros((10_000, 3)), np.tile([0.0, 0.0, 1.0], (10_000, 1))
    out, n_out, is_out = replace_with_outliers(pts, nrm, 0.05, np.full(3, -1.0), np.full(3, 1.0), rng)
    assert is_out.sum() == 500
    assert (np.abs(out[is_out]) <= 1).all() and (out[~is_out] == 0).all()
    assert np.allclose(np.linalg.norm(n_out, axis=1), 1.0)


def test_patches_drop_an_exact_nested_fraction():
    pts, _ = sample_surface(_sphere(), 20_000, seed=0)
    centres = farthest_point_centres(pts, 4, np.random.default_rng(0))
    k10, k25 = patch_mask(pts, 0.10, centres), patch_mask(pts, 0.25, centres)
    assert (~k10).sum() == 2_000 and (~k25).sum() == 5_000
    assert not (k25 & ~k10).any()                      # everything dropped at 10% is also dropped at 25%


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_stats_of_clean_noisy_biased_and_outlier_clouds():
    pytest.importorskip("open3d")
    gt = _sphere()
    pts, nrm = sample_surface(gt, 100_000, seed=3)
    rng = np.random.default_rng(0)
    clean = cloud_stats(pts, gt)
    assert clean["noise_sigma"] < 2e-3 and clean["outlier_frac"] == 0 and clean["coverage"] > 0.999
    noisy = cloud_stats(add_noise(pts, 0.02, rng), gt)                 # raw 0.02 = protocol 0.01 (L = 2)
    assert abs(noisy["noise_sigma"] - 0.01) < 1e-3 and abs(noisy["bias"]) < 5e-4
    assert noisy["coverage"] > 0.999                                   # coverage does not count noise as holes
    half = pts[:, 2] > 0                                               # drop the lower hemisphere
    assert abs(cloud_stats(pts[half], gt)["coverage"] - 0.5) < 0.01
    biased = cloud_stats(offset_along_normals(pts, nrm, 0.02), gt)
    assert abs(biased["bias"] - 0.01) < 1e-3
    out_pts, _, _ = replace_with_outliers(pts, nrm, 0.10, gt.bounds[0], gt.bounds[1], rng)
    outl = cloud_stats(out_pts, gt)
    assert 0.06 < outl["outlier_frac"] <= 0.10                         # some uniform points land near the surface
    assert outl["noise_sigma"] < 2e-3                                  # robust: outliers do not inflate the noise


def test_visible_cap_from_one_camera():
    pytest.importorskip("open3d")
    gt = _sphere()
    pts, _ = sample_surface(gt, 50_000, seed=0)
    pose = np.eye(4)
    pose[2, 3] = 3.0                                   # on +z looking down -z at the origin (OpenGL)
    seen = visible_mask(pts, gt, pose[None], yfov_deg=60.0)
    # a sphere seen from distance 3 shows the cap cos(theta) > 1/3, i.e. (1 - 1/3) / 2 of its area
    assert abs(seen.mean() - 1.0 / 3.0) < 0.01
