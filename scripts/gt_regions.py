"""Label GT surface regions (Prof. Zhong, 17 Nov, item 2). The labels use the GT, so they only score; no filter sees them.

Usage: python scripts/gt_regions.py [--shapes ...]          -> outputs/<s>_dense/eval/gt_regions.npz
GT surface samples: the 100,000 points of the protocol sampler (metrics.sample_surface, seed 0) in protocol v1's
normalised frame (GT bbox centre, largest extent 1), with their face normals. Input views: the 8 default views
(configs/default.yaml points.view_ids) of data/scenes/<s>/gt_cam_poses.npy (OpenGL camera-to-world, 518 px, yfov 40).
Regions (not exclusive):
  unobserved: no input view sees the sample (outside every image, or the ray from the camera hits the GT surface
      earlier, tolerance 0.002 of the extent).
  boundary: some input view sees the sample and its pixel lies within 3 px of the GT silhouette or of an occlusion edge
      (neighbouring GT depths differ by more than 2%) in the GT depth image of that view. The default pixel filter
      erodes VGGT's foreground by 2 px, so this is where the input loses pixels.
  thin: local thickness < 0.03 of the extent. Thickness = median ray distance from the sample to the opposite GT
      surface over 5 rays (the inward normal and 4 rays at 20 deg around it); inward = the side whose point at 1e-4
      is inside the GT (Open3D compute_occupancy, ray parity), so badly oriented faces do not matter.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from scipy.ndimage import binary_dilation

from photo2mesh.eval.metrics import sample_surface

sys.path.insert(0, str(Path(__file__).parent))
from conf_report import normalised  # noqa: E402
from make_points_queue import DEV, SET_A, SET_B  # noqa: E402

THIN, EDGE_PX, EDGE_REL, VIS_TOL = 0.03, 3, 0.02, 0.002


def scene_of(g) -> o3d.t.geometry.RaycastingScene:
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(np.asarray(g.vertices, np.float32)), o3d.core.Tensor(np.asarray(g.faces, np.uint32)))
    return sc


def cast(sc, origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    r = np.concatenate([origins, dirs], 1).astype(np.float32)
    return sc.cast_rays(o3d.core.Tensor(r))["t_hit"].numpy()


def cameras(shape: str, centre: np.ndarray, scale: float):
    """Input-view cameras in the protocol frame: (centres (V, 3), world-to-camera rotations (V, 3, 3), focal, size)."""
    import json
    sd = Path("data/scenes") / shape
    meta = json.loads((sd / "gt_transform.json").read_text())
    ids = yaml.safe_load(open("configs/default.yaml"))["points"]["view_ids"]
    c2w = np.load(sd / "gt_cam_poses.npy")[ids]
    size = int(meta["size"])
    f = (size / 2) / np.tan(np.deg2rad(meta["yfov_deg"]) / 2)
    return (c2w[:, :3, 3] - centre) * scale, np.transpose(c2w[:, :3, :3], (0, 2, 1)), f, size


def project(p: np.ndarray, c: np.ndarray, r: np.ndarray, f: float, size: int):
    """Pixel (u, v) of points p in an OpenGL camera (looks down -z, y up) and whether it is in front and inside."""
    x = (p - c) @ r.T
    z = -x[:, 2]
    ok = z > 1e-6
    u = np.where(ok, size / 2 + f * x[:, 0] / np.maximum(z, 1e-6), -1)
    v = np.where(ok, size / 2 - f * x[:, 1] / np.maximum(z, 1e-6), -1)
    return u, v, ok & (u >= 0) & (u < size) & (v >= 0) & (v < size)


def depth_image(sc, c: np.ndarray, r: np.ndarray, f: float, size: int) -> np.ndarray:
    """GT ray distance per pixel (inf = background)."""
    v, u = np.mgrid[0:size, 0:size] + 0.5
    d = np.stack([(u - size / 2) / f, -(v - size / 2) / f, -np.ones_like(u)], -1).reshape(-1, 3)
    d = d @ r                                                  # camera -> world (r is world-to-camera)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return cast(sc, np.broadcast_to(c, d.shape), d).reshape(size, size)


def edge_mask(t: np.ndarray) -> np.ndarray:
    """Pixels within EDGE_PX of the silhouette or an occlusion edge."""
    fg = np.isfinite(t)
    tt = np.where(fg, t, 0)
    e = np.zeros_like(fg)
    for ax in (0, 1):
        a, b = np.take(tt, range(0, t.shape[ax] - 1), ax), np.take(tt, range(1, t.shape[ax]), ax)
        fa, fb = np.take(fg, range(0, t.shape[ax] - 1), ax), np.take(fg, range(1, t.shape[ax]), ax)
        jump = (fa != fb) | (fa & fb & (np.abs(a - b) > EDGE_REL * np.minimum(a, b)))
        sl0 = [slice(None)] * 2
        sl1 = [slice(None)] * 2
        sl0[ax], sl1[ax] = slice(0, -1), slice(1, None)
        e[tuple(sl0)] |= jump
        e[tuple(sl1)] |= jump
    return binary_dilation(e, iterations=EDGE_PX) & fg


def thickness(sc, p: np.ndarray, n: np.ndarray) -> np.ndarray:
    eps = 1e-4
    occ = sc.compute_occupancy(o3d.core.Tensor((p - eps * n).astype(np.float32)), nsamples=5).numpy() > 0.5
    inward = np.where(occ[:, None], -n, n)
    a = np.cross(inward, np.where(np.abs(inward[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = np.cross(inward, a)
    th = np.deg2rad(20)
    dirs = [inward] + [np.cos(th) * inward + np.sin(th) * (s1 * a + s2 * b) / np.sqrt(s1 * s1 + s2 * s2)
                       for s1, s2 in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
    t = np.stack([cast(sc, p + eps * d, d) for d in dirs], 1)
    return np.median(t, 1)


def label(shape: str) -> dict:
    _, g, centre, scale = normalised(shape)
    sc = scene_of(g)
    p, n = sample_surface(g, 100_000)
    p, n = p.astype(np.float64), n.astype(np.float64)
    cs, rs, f, size = cameras(shape, centre, scale)
    n_vis = np.zeros(len(p), np.int32)
    boundary = np.zeros(len(p), bool)
    for c, r in zip(cs, rs):
        u, v, ins = project(p, c, r, f, size)
        d = p - c
        dist = np.linalg.norm(d, axis=1)
        hit = cast(sc, np.broadcast_to(c, p.shape), d / dist[:, None])
        vis = ins & (hit > dist - VIS_TOL)
        n_vis += vis
        em = edge_mask(depth_image(sc, c, r, f, size))
        iu, iv = np.clip(u.astype(int), 0, size - 1), np.clip(v.astype(int), 0, size - 1)
        boundary |= vis & em[iv, iu]
    th = thickness(sc, p, n)
    out = {"p": p.astype(np.float32), "n": n.astype(np.float32), "n_vis": n_vis, "thickness": th.astype(np.float32),
           "unobserved": n_vis == 0, "boundary": boundary, "thin": th < THIN}
    o = Path("outputs") / f"{shape}_dense" / "eval"
    o.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(o / "gt_regions.npz", **out)
    print(f"[regions] {shape}: unobserved {100 * out['unobserved'].mean():.1f}%  boundary {100 * boundary.mean():.1f}%  "
          f"thin {100 * out['thin'].mean():.1f}%", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="*", default=None)
    args = ap.parse_args()
    for s in args.shapes or [DEV] + SET_A + SET_B:
        label(s)


if __name__ == "__main__":
    main()
