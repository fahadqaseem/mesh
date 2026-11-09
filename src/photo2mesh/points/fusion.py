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


TSDF_DEFAULTS = {"mv_rel": 0.0, "voxel_frac": 0.005, "trunc_vox": 6.0, "min_views": 1, "min_part": 0.01}


def tsdf_points(pred: dict, cfg: dict, tsdf_cfg: dict | None = None, n: int = 200_000):
    """Points stage v3: fuse the filtered depth maps in a TSDF and sample n points from the fused surface.

    pred: VGGT maps (intrinsics already final, e.g. known focal); cfg: points config (pixel filters);
    tsdf_cfg: overrides of TSDF_DEFAULTS. Its mv_rel replaces the points config's multi-view filter (0 = off: the
    fusion averages the views instead). Returns (points, normals, colours, fused mesh, info dict).
    """
    from .filter import filter_pixels

    t = {**TSDF_DEFAULTS, **(tsdf_cfg or {})}
    keep = filter_pixels(pred, {**cfg, "mv_rel": t["mv_rel"]}).reshape(pred["depth"].shape)
    kept = pred["points"][keep]
    size = float((np.percentile(kept, 99, axis=0) - np.percentile(kept, 1, axis=0)).max())
    voxel = t["voxel_frac"] * size
    mesh = fuse(np.where(keep, pred["depth"], 0.0), pred["colors"], pred["extrinsic"], pred["intrinsic"], voxel,
                t["trunc_vox"], t["min_views"])
    if len(mesh.faces) == 0:
        raise RuntimeError("TSDF: empty surface (lower min_views or raise trunc_vox)")
    n_faces_raw = len(mesh.faces)
    mesh = drop_small_parts(mesh, t["min_part"])
    pts, nrm, cols = sample_with_colours(mesh, n, seed=0)
    # the fused surface faces the cameras; flip the few normals that point away from every camera
    to_cam = pred["cam_centres"][None] - pts[:, None]
    away = (np.einsum("nsj,nj->ns", to_cam, nrm) < 0).all(axis=1)
    nrm[away] *= -1
    info = {**t, "pixels_fused": int(keep.sum()), "object_size": size, "voxel": voxel, "trunc": t["trunc_vox"] * voxel,
            "faces_raw": n_faces_raw, "faces": int(len(mesh.faces)), "n": int(len(pts))}
    return pts, nrm, cols, mesh, info
