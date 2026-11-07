"""Point cloud and mesh I/O with normals and colours."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def save_points_ply(path: str | Path, points: np.ndarray, normals: np.ndarray | None = None,
                    colors: np.ndarray | None = None) -> None:
    """Write a binary little-endian PLY point cloud.

    Args:
        points: (N, 3) float xyz.
        normals: optional (N, 3) unit normals.
        colors: optional (N, 3) RGB in [0, 1] or [0, 255].
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=np.float32)
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    cols = [pts[:, 0], pts[:, 1], pts[:, 2]]
    if normals is not None:
        n = np.asarray(normals, dtype=np.float32)
        fields += [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
        cols += [n[:, 0], n[:, 1], n[:, 2]]
    if colors is not None:
        c = np.asarray(colors, dtype=np.float32)
        if c.max() <= 1.0:
            c = c * 255.0
        c = np.clip(c, 0, 255).astype(np.uint8)
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        cols += [c[:, 0], c[:, 1], c[:, 2]]
    data = np.empty(len(pts), dtype=fields)
    for (name, _), col in zip(fields, cols):
        data[name] = col
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(pts)}"]
    type_names = {"f4": "float", "u1": "uchar"}
    header += [f"property {type_names[t]} {name}" for name, t in fields]
    header += ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(data.tobytes())


_PLY_TYPES = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1",
              "uint8": "u1", "char": "i1", "int8": "i1", "short": "i2", "ushort": "u2",
              "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4"}


def load_points_ply(path: str | Path) -> dict[str, np.ndarray]:
    """Read a PLY point cloud (binary little-endian or ASCII vertex element).

    Returns a dict with 'points' and, if present, 'normals' and 'colors' (RGB in [0, 1]).
    Falls back to trimesh for anything unusual (and then returns points only).
    """
    path = Path(path)
    with open(path, "rb") as f:
        header, props, n_vert, fmt = [], [], 0, None
        in_vertex = False
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header.append(line)
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element"):
                _, name, count = line.split()
                in_vertex = name == "vertex"
                if in_vertex:
                    n_vert = int(count)
            elif line.startswith("property") and in_vertex:
                parts = line.split()
                if parts[1] == "list":
                    props = None
                    break
                props.append((parts[2], _PLY_TYPES[parts[1]]))
            elif line == "end_header" or not line and len(header) > 200:
                break
        if props is None or fmt not in ("binary_little_endian", "ascii"):
            cloud = trimesh.load(str(path), process=False)
            return {"points": np.asarray(cloud.vertices, dtype=np.float32)}
        dtype = np.dtype([(n, "<" + t) for n, t in props])
        if fmt == "binary_little_endian":
            data = np.frombuffer(f.read(dtype.itemsize * n_vert), dtype=dtype, count=n_vert)
        else:
            rows = np.loadtxt(f, max_rows=n_vert, ndmin=2)
            data = np.empty(n_vert, dtype=dtype)
            for i, (n, _) in enumerate(props):
                data[n] = rows[:, i]
    names = data.dtype.names
    out = {"points": np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)}
    if {"nx", "ny", "nz"} <= set(names):
        out["normals"] = np.stack([data["nx"], data["ny"], data["nz"]], axis=1).astype(np.float32)
    if {"red", "green", "blue"} <= set(names):
        rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.float32)
        out["colors"] = rgb / 255.0 if data["red"].dtype == np.uint8 else rgb
    return out


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh file (obj/ply/glb/stl) as a single Trimesh."""
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load a triangle mesh from {path}")
    return mesh
