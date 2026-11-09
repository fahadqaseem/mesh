import numpy as np

from photo2mesh.points.gaussians import (gaussian_centres, qvec_to_rotmat, read_colmap_images, read_ply_vertices,
                                         rotmat_to_qvec, write_colmap)


def _random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_quaternion_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        r = _random_rotation(rng)
        q = rotmat_to_qvec(r)
        assert q[0] >= 0 and np.isclose(np.linalg.norm(q), 1)
        assert np.allclose(qvec_to_rotmat(q), r, atol=1e-9)


def test_colmap_images_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    ext = np.stack([np.c_[_random_rotation(rng), rng.normal(size=3)] for _ in range(3)])
    k = np.repeat(np.array([[700.0, 0, 259], [0, 700, 259], [0, 0, 1]])[None], 3, 0)
    names = [f"view_{i:03d}.png" for i in range(3)]
    write_colmap(tmp_path, names, ext, k, (518, 518), rng.normal(size=(10, 3)), rng.random((10, 3)))
    back = read_colmap_images(tmp_path / "images.bin")
    assert [b[0] for b in back] == names
    for (_, q, t), e in zip(back, ext):
        assert np.allclose(qvec_to_rotmat(q), e[:, :3]) and np.allclose(t, e[:, 3])
    assert (tmp_path / "points3D.bin").stat().st_size == 8 + 10 * 51     # id 8 + xyz 24 + rgb 3 + error 8 + track length 8


def test_gaussian_ply_and_opacity_filter(tmp_path):
    fields = [(n, "<f4") for n in ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"]]
    v = np.zeros(3, dtype=fields)
    v["x"] = [0, 1, 2]
    v["opacity"] = [2.0, -2.0, 0.0]                  # sigmoid: 0.88, 0.12, 0.5
    v["scale_0"] = np.log([0.01, 0.01, 0.5])
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 3\n" + \
             "".join(f"property float {n}\n" for n, _ in fields) + "end_header\n"
    path = tmp_path / "s.ply"
    path.write_bytes(header.encode() + v.tobytes())
    back = read_ply_vertices(path)
    xyz, op, sc = gaussian_centres(back, min_opacity=0.5)
    assert np.allclose(xyz[:, 0], [0, 2])
    xyz, _, _ = gaussian_centres(back, min_opacity=0.5, max_scale=0.1)
    assert np.allclose(xyz[:, 0], [0])
