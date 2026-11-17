import numpy as np

from photo2mesh.points.depth_scale import DepthAgreement, fit_global, fit_per_view


def sphere_views(n_views: int = 8, size: int = 64, radius: float = 0.5, dist: float = 3.0):
    """Exact depth maps of a sphere at the origin seen by cameras on a ring (OpenCV camera-from-world extrinsics)."""
    f = size / 2 / np.tan(np.deg2rad(20))
    K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])
    depth, ext = np.zeros((n_views, size, size)), np.zeros((n_views, 3, 4))
    for i in range(n_views):
        a, el = 2 * np.pi * i / n_views, np.deg2rad(20 if i % 2 else 50)
        c = dist * np.array([np.cos(el) * np.cos(a), np.sin(el), np.cos(el) * np.sin(a)])
        fwd = -c / np.linalg.norm(c)
        right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd])
        ext[i] = np.c_[R, -R @ c]
        v, u = np.mgrid[0:size, 0:size]
        rays = np.c_[u.ravel(), v.ravel(), np.ones(size * size)] @ np.linalg.inv(K).T   # camera frame, z = 1
        dw = rays @ R                                                                                # world directions
        b = 2 * dw @ c
        a2 = (dw ** 2).sum(1)
        disc = b ** 2 - 4 * a2 * (c @ c - radius ** 2)
        t = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a2), 0.0)
        depth[i] = t.reshape(size, size)                                                            # z depth (ray has z = 1)
    return depth, depth > 0, ext, np.repeat(K[None], n_views, 0)


def test_global_scale_recovered():
    depth, keep, ext, K = sphere_views()
    agr = DepthAgreement(depth * 0.97, keep, ext, K, n_per_view=3000)
    fit = fit_global(agr)
    assert abs(fit["k"] - 1 / 0.97) < 0.004
    assert fit["cost"] < fit["cost_at_1"]


def test_per_view_scale_recovered():
    depth, keep, ext, K = sphere_views()
    true = np.array([1.0, 0.98, 1.02, 0.99, 1.01, 0.97, 1.0, 1.03])
    agr = DepthAgreement(depth / true[:, None, None], keep, ext, K, n_per_view=3000)
    k = np.array(fit_per_view(agr, fit_global(agr)["k"])["k"])
    assert np.abs(k - true).max() < 0.01


def test_joint_scale_and_focal_recovered():
    from photo2mesh.points.depth_scale import fit_joint
    depth, keep, ext, K = sphere_views(size=96)
    wrong = K.copy()
    wrong[:, :2, :2] /= 0.95                 # the fit is given a focal 1/0.95 too long; the truth is phi = 0.95
    fit = fit_joint(DepthAgreement(depth * 0.97, keep, ext, wrong, n_per_view=3000))
    assert abs(fit["phi"] - 0.95) < 0.01
    assert abs(fit["k"] - 1 / 0.97) < 0.01
