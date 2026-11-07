import numpy as np
import pytest

from photo2mesh.points.vggt_points import use_point_head


def look_at_opencv(eye: np.ndarray) -> np.ndarray:
    """(3, 4) camera-from-world, OpenCV convention (x right, y down, z forward), looking at the origin."""
    z = -eye / np.linalg.norm(eye)
    x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rot = np.stack([x, y, z])                       # rows: camera axes in world coordinates
    return np.c_[rot, -rot @ eye]


def sphere_views(n_views: int = 6, size: int = 96, f: float = 110.0, dist: float = 3.0):
    """Exact depth maps (z in the camera) of the unit sphere seen from a ring of cameras; 0 where the ray misses."""
    k = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])
    u, v = np.meshgrid(np.arange(size), np.arange(size))
    rays_cam = np.stack([(u - size / 2) / f, (v - size / 2) / f, np.ones_like(u, float)], -1)   # z = 1
    depth, ext = [], []
    for a in np.linspace(0, 2 * np.pi, n_views, endpoint=False):
        e = look_at_opencv(dist * np.array([np.sin(a), 0.3, np.cos(a)]))
        rot, t = e[:, :3], e[:, 3]
        centre = -rot.T @ t
        d = rays_cam @ rot                            # ray directions in world (z-scaled), (H, W, 3)
        b = (d * centre).sum(-1)
        a2 = (d * d).sum(-1)
        disc = b ** 2 - a2 * ((centre ** 2).sum() - 1.0)
        z = np.where(disc > 0, (-b - np.sqrt(np.clip(disc, 0, None))) / a2, 0.0)    # depth along z = ray parameter
        depth.append(z)
        ext.append(e)
    return np.array(depth, np.float32), np.array(ext), np.repeat(k[None], n_views, 0)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_tsdf_fusion_recovers_a_sphere():
    pytest.importorskip("open3d")
    from photo2mesh.points.fusion import fuse, sample_with_colours

    depth, ext, k = sphere_views()
    colours = np.full(depth.shape + (3,), 0.5, np.float32)
    mesh = fuse(depth, colours, ext, k, voxel=0.02, trunc_vox=4, min_views=1)
    pts, nrm, cols = sample_with_colours(mesh, 20_000)
    r = np.linalg.norm(pts, axis=1)
    assert np.abs(r - 1.0).mean() < 0.01                       # on the sphere to about half a voxel
    assert (np.einsum("ij,ij->i", nrm, pts / r[:, None]) > 0).mean() > 0.95 or \
           (np.einsum("ij,ij->i", nrm, pts / r[:, None]) < 0).mean() > 0.95      # consistently oriented
    assert np.allclose(cols, 0.5, atol=0.01)
    seen = fuse(depth, colours, ext, k, voxel=0.02, trunc_vox=4, min_views=3)
    assert 0 < seen.area < mesh.area                           # needing 3 views keeps less surface
    again = fuse(depth, colours, ext, k, voxel=0.02, trunc_vox=4, min_views=1)
    assert np.array_equal(again.vertices, mesh.vertices) and np.array_equal(again.faces, mesh.faces)   # reproducible
    assert np.isclose(again.area, mesh.area)


def test_point_head_depth_is_z_in_each_camera():
    depth, ext, k = sphere_views(n_views=3, size=32)
    s, h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    world = []
    for i in range(s):
        cam = np.stack([(u - k[i, 0, 2]) / k[i, 0, 0] * depth[i], (v - k[i, 1, 2]) / k[i, 1, 1] * depth[i], depth[i]], -1)
        world.append((cam - ext[i, :, 3]) @ ext[i, :, :3])     # X = R^T (x_cam - t)
    pred = {"world_points": np.array(world, np.float32), "world_points_conf": np.ones((s, h, w), np.float32),
            "extrinsic": ext.astype(np.float32)}
    use_point_head(pred)
    assert np.allclose(pred["depth"], depth, atol=1e-4)
    assert pred["points"].shape == (s, h, w, 3) and pred["conf"].shape == (s, h, w)
