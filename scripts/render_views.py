"""Make a test scene: render a ground-truth mesh (e.g. from Thingi10K) into N photos around it.

This gives us photos where the true shape is known, so the whole pipeline can be scored.

Usage:
    python scripts/render_views.py --mesh data/gt/bunny.obj --out data/scenes/bunny/images --n 12
Needs: pip install pyrender  (on a headless Linux box also set PYOPENGL_PLATFORM=egl)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


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


def ring_poses(n: int, radius: float, elevation_deg: float = 20.0) -> list[np.ndarray]:
    elev = np.deg2rad(elevation_deg)
    poses = []
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        eye = radius * np.array([np.cos(elev) * np.sin(a), np.sin(elev), np.cos(elev) * np.cos(a)])
        poses.append(look_at(eye, np.zeros(3)))
    return poses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--size", type=int, default=518)   # VGGT's native resolution
    args = ap.parse_args()

    import pyrender
    from PIL import Image

    mesh = trimesh.load(args.mesh, force="mesh")
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale(1.0 / np.linalg.norm(mesh.vertices, axis=1).max())   # unit sphere

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.3, 0.3, 0.3])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(40))
    renderer = pyrender.OffscreenRenderer(args.size, args.size)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    poses = ring_poses(args.n, radius=3.0)
    for i, pose in enumerate(poses):
        cam_node = scene.add(cam, pose=pose)
        light = scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
        color, _ = renderer.render(scene)
        Image.fromarray(color).save(out / f"view_{i:03d}.png")
        scene.remove_node(cam_node)
        scene.remove_node(light)
    np.save(out.parent / "gt_cam_poses.npy", np.stack(poses))
    renderer.delete()
    print(f"[render] {len(poses)} views -> {out}")


if __name__ == "__main__":
    main()
