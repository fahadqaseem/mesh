"""Make a test scene: render a ground-truth mesh (e.g. from Thingi10K) into photos around it.

This gives us photos where the true shape is known, so the whole pipeline can be scored.

Usage:
    python scripts/render_views.py --mesh data/gt/bunny.stl --out data/scenes/bunny/images_tex --texture procedural
    (--texture original renders the file's own materials, e.g. ShapeNet; see scripts/prep_dataset.py)
Writes, next to the images folder (the scene dir):
    gt_cam_poses.npy   (V, 4, 4) camera-to-world, OpenGL convention, in the normalised frame
    gt_normalized.ply  the GT mesh in that same frame (rotated to Y-up, centred, unit sphere)
    gt_transform.json  how the raw mesh was mapped into that frame, plus the camera rig (shared by every images folder)
and, inside the images folder: render.json (texture and seed of that folder's images)
Needs: pip install pyrender  (on a headless Linux box also set PYOPENGL_PLATFORM=egl)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

Z_UP_TO_Y_UP = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])   # (x, y, z) -> (x, z, -y)


def look_at(eye: np.ndarray, target: np.ndarray, up=np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """Camera-to-world pose (OpenGL convention: camera looks down -Z)."""
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, eye
    return pose


def ring_poses(n: int, radius: float, elevation_deg: float = 20.0, azimuth_offset: float = 0.0) -> list[np.ndarray]:
    """``n`` cameras on a horizontal ring around the Y axis, all looking at the origin."""
    elev = np.deg2rad(elevation_deg)
    poses = []
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False) + azimuth_offset:
        eye = radius * np.array([np.cos(elev) * np.sin(a), np.sin(elev), np.cos(elev) * np.cos(a)])
        poses.append(look_at(eye, np.zeros(3)))
    return poses


def multi_ring_poses(n_per_ring: int, elevations: list[float], radius: float) -> list[np.ndarray]:
    """Several rings; each ring is rotated by half a step from the previous one for better coverage."""
    step = 2 * np.pi / n_per_ring
    poses = []
    for i, e in enumerate(elevations):
        poses += ring_poses(n_per_ring, radius, e, azimuth_offset=(i % 2) * step / 2)
    return poses


def normalise(mesh: trimesh.Trimesh, up: str) -> dict:
    """Rotate to Y-up, centre on the bbox centre, scale into the unit sphere (in place). Returns the transform."""
    rot = Z_UP_TO_Y_UP if up == "z" else np.eye(3)
    mesh.vertices = mesh.vertices @ rot.T
    centre = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-centre)
    scale = 1.0 / np.linalg.norm(mesh.vertices, axis=1).max()
    mesh.apply_scale(scale)
    return {"rotation": rot.tolist(), "centre_after_rotation": centre.tolist(), "scale": float(scale),
            "note": "x_norm = scale * (R @ x_raw - centre)"}


def procedural_colours(points: np.ndarray, seed: int = 0, n_waves: int = 6) -> np.ndarray:
    """Smooth but feature-rich RGB pattern from random 3D sine waves (deterministic for a seed). Returns uint8 (N, 3)."""
    rng = np.random.default_rng(seed)
    rgb = np.zeros((len(points), 3))
    for c in range(3):
        for _ in range(n_waves):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            freq = rng.uniform(2.0, 10.0)          # waves per unit length (object fits the unit sphere)
            rgb[:, c] += np.sin(2 * np.pi * freq * points @ d + rng.uniform(0, 2 * np.pi))
    rgb = (rgb - rgb.min(0)) / (np.ptp(rgb, axis=0) + 1e-9)
    return (40 + 200 * rgb).astype(np.uint8)       # avoid pure black/white (white is the background)


def textured_copy(mesh: trimesh.Trimesh, seed: int, max_edge: float = 0.015) -> trimesh.Trimesh:
    """Subdivide (same surface, finer vertices) and paint procedural vertex colours. Used for rendering only."""
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=max_edge, max_iter=20)
    fine = trimesh.Trimesh(v, f, process=False)
    fine.visual.vertex_colors = np.c_[procedural_colours(fine.vertices, seed), np.full(len(v), 255, np.uint8)]
    return fine


def baked_copy(geom: trimesh.Trimesh, max_edge: float = 0.015, max_faces: int = 3_000_000) -> trimesh.Trimesh:
    """The part's own colour as vertex colours: UV textures are sampled at the vertices of a subdivided copy.

    pyrender's GL texture upload fails here (pyrender pins PyOpenGL 3.1.0, which breaks under Python 3.13 / numpy 2),
    so textures are baked the same way the procedural Thingi10K texture is applied. Untextured materials give their
    flat colour. Used for rendering only; the GT geometry is the original mesh.
    """
    vis = geom.visual
    image = getattr(getattr(vis, "material", None), "image", None) if vis.kind == "texture" else None
    if image is None or getattr(vis, "uv", None) is None:
        out = geom.copy()
        out.visual = vis.to_color() if vis.kind == "texture" else vis
        return out
    v, f, uv = geom.vertices, geom.faces, np.asarray(vis.uv, np.float64)
    while len(f) * 4 <= max_faces:
        edges = np.linalg.norm(v[f[:, [1, 2, 0]]] - v[f], axis=2).max()
        if edges <= max_edge:
            break
        v, f, attrs = trimesh.remesh.subdivide(v, f, vertex_attributes={"uv": uv})
        uv = attrs["uv"]
    cols = trimesh.visual.uv_to_color(uv, image)
    return trimesh.Trimesh(v, f, vertex_colors=cols, process=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True, help="images folder; metadata goes to its parent (the scene dir)")
    ap.add_argument("--n", type=int, default=8, help="views per ring")
    ap.add_argument("--elevations", default="20,50", help="comma-separated ring elevations in degrees")
    ap.add_argument("--up", choices=["z", "y"], default="z", help="up axis of the input mesh (Thingi10K STL: z)")
    ap.add_argument("--texture", choices=["none", "procedural", "original"], default="none",
                    help="original: render the file's own materials / UV textures (e.g. ShapeNet OBJ + MTL)")
    ap.add_argument("--seed", type=int, default=0, help="procedural texture seed")
    ap.add_argument("--size", type=int, default=518)   # VGGT's native resolution
    args = ap.parse_args()

    import pyrender
    from PIL import Image

    mesh = trimesh.load(args.mesh, force="mesh")
    transform = normalise(mesh, args.up)
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.3, 0.3, 0.3])
    if args.texture == "original":
        # same normalisation applied to every textured part of the file, so the render matches gt_normalized.ply
        src = trimesh.load(args.mesh, force="scene")
        m = np.eye(4)
        m[:3, :3] = transform["scale"] * np.asarray(transform["rotation"])
        m[:3, 3] = -transform["scale"] * np.asarray(transform["centre_after_rotation"])
        for geom in src.dump(concatenate=False):
            g = geom.copy()
            g.apply_transform(m)
            scene.add(pyrender.Mesh.from_trimesh(baked_copy(g), smooth=False))
    else:
        render_mesh = textured_copy(mesh, args.seed) if args.texture == "procedural" else mesh
        scene.add(pyrender.Mesh.from_trimesh(render_mesh, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(40))
    renderer = pyrender.OffscreenRenderer(args.size, args.size)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    elevations = [float(e) for e in args.elevations.split(",")]
    poses = multi_ring_poses(args.n, elevations, radius=3.0)
    for i, pose in enumerate(poses):
        cam_node = scene.add(cam, pose=pose)
        light = scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
        color, _ = renderer.render(scene)
        Image.fromarray(color).save(out / f"view_{i:03d}.png")
        scene.remove_node(cam_node)
        scene.remove_node(light)
    renderer.delete()

    scene_dir = out.parent
    np.save(scene_dir / "gt_cam_poses.npy", np.stack(poses))
    mesh.export(scene_dir / "gt_normalized.ply")
    # scene-level: geometry + camera rig only (identical for grey and textured renders of the same mesh)
    meta = {**transform, "source_mesh": str(args.mesh), "up": args.up, "elevations": elevations,
            "views_per_ring": args.n, "yfov_deg": 40, "size": args.size}
    (scene_dir / "gt_transform.json").write_text(json.dumps(meta, indent=2))
    # folder-level: what these particular images look like
    (out / "render.json").write_text(json.dumps({"texture": args.texture, "seed": args.seed}, indent=2))
    print(f"[render] {len(poses)} views ({args.texture}) -> {out}")


if __name__ == "__main__":
    main()
