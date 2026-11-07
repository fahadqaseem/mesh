"""Depth-map fusion for the points stage: TSDF (Open3D, CPU) and surface sampling.

Experiment B (route a): instead of concatenating the per-view VGGT points, fuse the depth maps into one surface and
sample points from it. Used by scripts/tsdf_points.py.
"""
from __future__ import annotations

import numpy as np
import trimesh


def fuse(depth: np.ndarray, colors: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray, voxel: float,
         trunc_vox: float, min_views: int) -> trimesh.Trimesh:
    """TSDF fusion of masked depth maps (0 = no depth) with Open3D's VoxelBlockGrid on the CPU.

    extrinsic (S, 3, 4) camera-from-world (OpenCV, as Open3D expects), intrinsic (S, 3, 3). The surface is extracted
    where at least min_views views observed the voxel (Open3D keeps weight > threshold, so threshold = min_views - 0.5).
    Open3D 0.20's legacy ScalableTSDFVolume integrates nothing on Apple silicon, so the tensor API is used.
    """
    import open3d as o3d

    o3c = o3d.core
    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight", "color"),
                                        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
                                        attr_channels=((1), (1), (3)), voxel_size=float(voxel), block_resolution=8,
                                        block_count=20000, device=o3c.Device("CPU:0"))
    depth_max = float(depth.max()) * 1.5
    for i in range(len(depth)):
        d_img = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(depth[i], np.float32)))
        c_img = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(colors[i], np.float32)))
        k = o3c.Tensor(np.asarray(intrinsic[i], np.float64))
        ext = np.eye(4)
        ext[:3] = extrinsic[i]
        e = o3c.Tensor(ext)
        blocks = vbg.compute_unique_block_coordinates(d_img, k, e, 1.0, depth_max, float(trunc_vox))
        vbg.integrate(blocks, d_img, c_img, k, k, e, 1.0, depth_max, float(trunc_vox))
    m = vbg.extract_triangle_mesh(weight_threshold=float(min_views) - 0.5).to_legacy()
    cols = (np.asarray(m.vertex_colors) * 255).clip(0, 255).astype(np.uint8) if m.has_vertex_colors() else None
    return canonical_order(np.asarray(m.vertices), np.asarray(m.triangles), cols)


def canonical_order(vertices: np.ndarray, faces: np.ndarray, colours: np.ndarray | None = None) -> trimesh.Trimesh:
    """The same mesh with vertices and faces in a fixed order.

    Open3D's multithreaded extraction returns the same surface in a different order on every run, which would change
    the sampled points. Vertices are sorted (duplicates merged), each face is rotated to start at its smallest index
    (orientation kept), and faces are sorted; degenerate and repeated faces are dropped.
    """
    v, first, inv = np.unique(np.asarray(vertices, np.float64), axis=0, return_index=True, return_inverse=True)
    f = inv.reshape(-1)[np.asarray(faces)]
    r = np.argmin(f, axis=1)
    f = np.stack([f[np.arange(len(f)), (r + i) % 3] for i in range(3)], axis=1)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    f = np.unique(f, axis=0)                          # sorted rows
    cols = None if colours is None else np.asarray(colours)[first]
    return trimesh.Trimesh(v, f, vertex_colors=cols, process=False)


def sample_with_colours(mesh: trimesh.Trimesh, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-uniform surface samples with face normals and the face's mean vertex colour (RGB in [0, 1])."""
    rng = np.random.default_rng(seed)
    areas = mesh.area_faces
    face = rng.choice(len(areas), size=n, p=areas / areas.sum())
    tri = mesh.triangles[face]
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])
    cols = mesh.visual.vertex_colors[mesh.faces[face], :3].astype(np.float64).mean(axis=1) / 255.0
    return pts.astype(np.float32), mesh.face_normals[face].astype(np.float32), cols.astype(np.float32)


def drop_small_parts(mesh: trimesh.Trimesh, min_frac: float) -> trimesh.Trimesh:
    """Remove connected pieces with less than min_frac of the total area (fused floaters)."""
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    total = sum(p.area for p in parts)
    keep = [p for p in parts if p.area >= min_frac * total]
    return trimesh.util.concatenate(keep)
