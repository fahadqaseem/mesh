"""Gaussian splatting as a points stage (decision 5: Gaussian centres vs interpolated points).

Pipeline per object: write the VGGT cameras and an initial cloud as a binary COLMAP model, train 3D Gaussian
splatting on the Mac with msplat (Metal), then use the centres of the opaque Gaussians as MeshFlow's input points.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def rotmat_to_qvec(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), COLMAP's convention (w >= 0)."""
    r = np.asarray(r, np.float64)
    k = np.array([[r[0, 0] - r[1, 1] - r[2, 2], 0, 0, 0],
                  [r[1, 0] + r[0, 1], r[1, 1] - r[0, 0] - r[2, 2], 0, 0],
                  [r[2, 0] + r[0, 2], r[2, 1] + r[1, 2], r[2, 2] - r[0, 0] - r[1, 1], 0],
                  [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], r[0, 0] + r[1, 1] + r[2, 2]]]) / 3.0
    vals, vecs = np.linalg.eigh(k)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    return q if q[0] >= 0 else -q


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def write_colmap(sparse_dir: Path, names: list[str], extrinsic: np.ndarray, intrinsic: np.ndarray, size: tuple[int, int],
                 points: np.ndarray, colours: np.ndarray) -> None:
    """Binary COLMAP model (cameras.bin, images.bin, points3D.bin), one PINHOLE camera per image.

    extrinsic (S, 3, 4) camera-from-world (OpenCV, as COLMAP), intrinsic (S, 3, 3), size (width, height),
    points (N, 3) initial cloud in the same world frame, colours (N, 3) in [0, 1]. No 2D-3D tracks.
    """
    sparse_dir = Path(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    w, h = size
    with open(sparse_dir / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(names)))
        for i, k in enumerate(intrinsic):
            f.write(struct.pack("<iiQQ", i + 1, 1, w, h))                       # model 1 = PINHOLE
            f.write(struct.pack("<4d", k[0, 0], k[1, 1], k[0, 2], k[1, 2]))
    with open(sparse_dir / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(names)))
        for i, (name, e) in enumerate(zip(names, extrinsic)):
            f.write(struct.pack("<i", i + 1))
            f.write(struct.pack("<4d", *rotmat_to_qvec(e[:, :3])))
            f.write(struct.pack("<3d", *np.asarray(e[:, 3], np.float64)))
            f.write(struct.pack("<i", i + 1))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))                                       # no 2D points
    rgb = np.clip(np.asarray(colours) * 255.0, 0, 255).astype(np.uint8)
    with open(sparse_dir / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for i, (p, c) in enumerate(zip(np.asarray(points, np.float64), rgb)):
            f.write(struct.pack("<Q3d3BdQ", i + 1, *p, *c, 0.0, 0))


def read_colmap_images(path: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(name, qvec, tvec) per image from images.bin (used by the tests)."""
    out = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(4)
            q = np.array(struct.unpack("<4d", f.read(32)))
            t = np.array(struct.unpack("<3d", f.read(24)))
            f.read(4)
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (n2d,) = struct.unpack("<Q", f.read(8))
            f.read(24 * n2d)
            out.append((name.decode(), q, t))
    return out


_PLY = {"float": "f4", "double": "f8", "uchar": "u1", "int": "i4", "uint": "u4", "short": "i2", "ushort": "u2"}


def read_ply_vertices(path: Path) -> np.ndarray:
    """All vertex properties of a binary little-endian PLY as a structured array (e.g. a 3DGS splat file)."""
    with open(path, "rb") as f:
        fields, n = [], 0
        while (line := f.readline().decode("ascii").strip()) != "end_header":
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                n = int(parts[2])
            elif parts[0] == "property" and n and parts[1] != "list":
                fields.append((parts[2], "<" + _PLY[parts[1]]))
        return np.frombuffer(f.read(n * np.dtype(fields).itemsize), dtype=fields, count=n)


def gaussian_centres(v: np.ndarray, min_opacity: float = 0.5, max_scale: float | None = None):
    """Centres (N, 3), opacity and largest scale of the Gaussians kept: opacity >= min_opacity, largest axis <= max_scale.

    3DGS files store opacity as a logit and scales as logs.
    """
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    opacity = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
    scale = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1).astype(np.float64)).max(1)
    keep = opacity >= min_opacity
    if max_scale is not None:
        keep &= scale <= max_scale
    return xyz[keep], opacity[keep], scale[keep]
