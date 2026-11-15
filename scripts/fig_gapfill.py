"""Figure: dense TSDF cloud vs dense gap-filled cloud, coloured by distance to the GT, for the shapes where TSDF is worst.

Usage: python scripts/fig_gapfill.py [--out docs/results/figures/2026-11-15_gapfill_compare.png]
Rows: thingi_103356, thingi_77949, thingi_593381 (set B). Columns: TSDF (1M, default), TSDF + gap fill (1M base), GT.
Same style as docs/results/figures/2026-11-12_points_compare.png: colour = distance to the GT surface in % of the GT's
largest extent (teal = on the surface, orange = 3% or more off it). Panel numbers: "on surface" = share of the cloud's
points within 1% of the GT; "covers" = share of the GT surface within 1% of any point (the all-points recall @0.01).
Light: CPU, matplotlib only (orthographic view, points drawn back to front; the GT is flat-shaded triangles).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import trimesh
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree

from photo2mesh.io import load_points_ply

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SHAPES = [("thingi_103356", "thin parts"), ("thingi_77949", "thin parts"), ("thingi_593381", "detailed")]
COLS = [("_dense", "TSDF (default)"), ("_dfill", "TSDF + gap fill")]
CMAP = LinearSegmentedColormap.from_list("err", ["#1a9e8f", "#8cbf5f", "#e6b422", "#ee8a35", "#e8603c"])
VMAX, SHOW = 3.0, 60_000          # colour scale max (% of extent); points drawn per panel


def view_matrix(elev: float = 22.0, azim: float = 35.0) -> np.ndarray:
    """Rows: screen x, screen y, depth (towards the viewer), for a Y-up scene."""
    a, e = np.deg2rad(azim), np.deg2rad(elev)
    fwd = np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])       # from object to camera
    right = np.cross([0.0, 1.0, 0.0], fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    return np.stack([right, up, fwd])


def draw_points(ax, pts: np.ndarray, err: np.ndarray, V: np.ndarray, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(pts), min(SHOW, len(pts)), replace=False)
    q, e = pts[pick] @ V.T, err[pick]
    order = np.argsort(q[:, 2])                        # far first
    ax.scatter(q[order, 0], q[order, 1], c=e[order], cmap=CMAP, vmin=0, vmax=VMAX, s=0.15, linewidths=0, rasterized=True)


def draw_mesh(ax, mesh: trimesh.Trimesh, V: np.ndarray) -> None:
    tri = mesh.triangles @ V.T
    n = mesh.face_normals @ V.T
    front = n[:, 2] > 0
    shade = 0.35 + 0.65 * np.clip(n[front] @ np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8]), 0, 1)
    t = tri[front]
    order = np.argsort(t[:, :, 2].mean(1))
    base = np.array([0.55, 0.62, 0.80])
    ax.add_collection(PolyCollection(t[order][:, :, :2], facecolors=np.clip(base[None] * shade[order, None], 0, 1),
                                     edgecolors="face", linewidths=0.3, rasterized=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results/figures/2026-11-15_gapfill_compare.png")
    out = Path(ap.parse_args().out)
    V = view_matrix()
    fig, axes = plt.subplots(len(SHAPES), len(COLS) + 1, figsize=(10.5, 4.1 * len(SHAPES)), gridspec_kw={"hspace": 0.35})
    for r, (s, kind) in enumerate(SHAPES):
        gt = trimesh.load(f"data/scenes/{s}/gt_normalized.ply", force="mesh")
        ext = float(np.ptp(gt.bounds, axis=0).max())
        g, _ = trimesh.sample.sample_surface(gt, 100_000, seed=0)
        tg = cKDTree(g)
        lims = None
        for c, (suf, title) in enumerate(COLS):
            pts = load_points_ply(f"outputs/{s}{suf}/align/points_aligned.ply")["points"].astype(np.float64)
            err = 100.0 * tg.query(pts)[0] / ext
            on = (err <= 1.0).mean()
            covers = (cKDTree(pts).query(g)[0] <= 0.01 * ext).mean()
            ax = axes[r, c]
            draw_points(ax, pts, err, V)
            ax.set_title(f"{title}\n{len(pts):,} points\non surface {100 * on:.0f}% · covers {100 * covers:.0f}%", fontsize=9)
            lims = lims or (ax.get_xlim(), ax.get_ylim())
        draw_mesh(axes[r, -1], gt, V)
        axes[r, -1].set_title("True shape", fontsize=9)
        q = gt.vertices @ V.T
        pad = 0.06 * ext
        for ax in axes[r]:
            ax.set_xlim(q[:, 0].min() - pad, q[:, 0].max() + pad)
            ax.set_ylim(q[:, 1].min() - pad, q[:, 1].max() + pad)
            ax.set_aspect("equal")
            ax.axis("off")
        axes[r, 0].text(-0.08, 0.5, f"Test shape {s.split('_')[1]}\n({kind})", transform=axes[r, 0].transAxes,
                        rotation=90, ha="center", va="center", fontsize=9, fontweight="bold")
    fig.suptitle("Dense point clouds coloured by error (teal = on the surface, orange = off it)", fontweight="bold", fontsize=11)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, VMAX))
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.025, pad=0.03, aspect=50)
    cb.set_label("distance to the true surface (% of the object's size)", fontsize=8)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
