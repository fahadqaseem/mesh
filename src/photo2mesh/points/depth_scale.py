"""GT-free depth scale correction: rescale VGGT depth maps so that the views agree with each other.

VGGT predicts cameras and depth with separate heads; if the depth is slightly too short or too long relative to the
camera baseline, surfaces seen from different views do not coincide. Scaling depth by k (per scene, or k_i per view)
with the cameras fixed changes that agreement, so k can be fitted from the views alone (no GT anywhere).

Agreement: for kept pixels of view i (back-projected with k_i) projected into every other view j, the relative depth
residual r = (k_j d_j(u, v) - z_j) / z_j at the pixel they land on (kept pixels of j only). The cost is the truncated
mean min(|r|, eps): pairs that see different surfaces (occlusion) cost eps whatever k is, so they do not pull the fit.
"""
from __future__ import annotations

import numpy as np


class DepthAgreement:
    """Pre-sampled pixels of each view; evaluate the multi-view cost for a vector of per-view depth scales."""

    def __init__(self, depth: np.ndarray, keep: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray,
                 n_per_view: int = 20_000, seed: int = 0, eps: float = 0.01) -> None:
        self.depth, self.keep = np.asarray(depth, np.float64), np.asarray(keep, bool)
        self.R, self.t = np.asarray(extrinsic)[:, :, :3].astype(np.float64), np.asarray(extrinsic)[:, :, 3].astype(np.float64)
        self.K = np.asarray(intrinsic, np.float64)
        self.eps = eps
        rng = np.random.default_rng(seed)
        self.cam_pts = []                                   # per view: (n, 3) camera-frame points at k = 1
        for i in range(len(depth)):
            v, u = np.nonzero(self.keep[i])
            sel = rng.choice(len(u), min(n_per_view, len(u)), replace=False)
            u, v = u[sel], v[sel]
            d = self.depth[i, v, u]
            ki = np.linalg.inv(self.K[i])
            self.cam_pts.append((np.c_[u, v, np.ones(len(u))] @ ki.T) * d[:, None])

    def residuals(self, k: np.ndarray, i: int, j: int) -> np.ndarray:
        """Relative depth residuals of view i's pixels seen in view j (only where j has a kept pixel)."""
        s, h, w = self.depth.shape
        xw = (k[i] * self.cam_pts[i] - self.t[i]) @ self.R[i]          # world = R^T (x_cam - t)
        xc = xw @ self.R[j].T + self.t[j]
        z = xc[:, 2]
        ok = z > 1e-9
        uv = (xc[ok] / z[ok, None]) @ self.K[j].T
        u0, v0 = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
        ins = (u0 >= 0) & (u0 < w - 1) & (v0 >= 0) & (v0 < h - 1)
        u0, v0, fu, fv, zz = u0[ins], v0[ins], uv[ins, 0] - u0[ins], uv[ins, 1] - v0[ins], z[ok][ins]
        kp, dp = self.keep[j], self.depth[j]
        kept = kp[v0, u0] & kp[v0, u0 + 1] & kp[v0 + 1, u0] & kp[v0 + 1, u0 + 1]   # bilinear only inside kept pixels
        u0, v0, fu, fv, zz = u0[kept], v0[kept], fu[kept], fv[kept], zz[kept]
        d = ((1 - fu) * (1 - fv) * dp[v0, u0] + fu * (1 - fv) * dp[v0, u0 + 1]
             + (1 - fu) * fv * dp[v0 + 1, u0] + fu * fv * dp[v0 + 1, u0 + 1])
        return (k[j] * d - zz) / zz

    def cost(self, k: np.ndarray, views: list[int] | None = None) -> float:
        """Truncated mean |r| over ordered view pairs (all pairs, or pairs involving `views`)."""
        s = len(self.depth)
        tot, n = 0.0, 0
        for i in range(s):
            for j in range(s):
                if i == j or (views is not None and i not in views and j not in views):
                    continue
                r = self.residuals(k, i, j)
                tot += np.minimum(np.abs(r), self.eps).sum()
                n += len(r)
        return tot / max(n, 1)


def _scan(f, lo: float, hi: float, step: float) -> tuple[float, np.ndarray, np.ndarray]:
    grid = np.arange(lo, hi + step / 2, step)
    vals = np.array([f(g) for g in grid])
    return float(grid[int(np.argmin(vals))]), grid, vals


def fit_global(agr: DepthAgreement, lo: float = 0.9, hi: float = 1.1) -> dict:
    """One depth scale for all views: coarse grid (0.005), then fine grid (0.0005) around the best."""
    s = len(agr.depth)
    f = lambda g: agr.cost(np.full(s, g))
    k0, grid, vals = _scan(f, lo, hi, 0.005)
    k1, g2, v2 = _scan(f, k0 - 0.005, k0 + 0.005, 0.0005)
    return {"k": k1, "cost": float(f(k1)), "cost_at_1": float(f(1.0)), "grid": grid.tolist(), "curve": vals.tolist()}


def fit_per_view(agr: DepthAgreement, k_init: float, sweeps: int = 3, half: float = 0.03) -> dict:
    """Per-view scales by coordinate descent from the global fit (each view scanned with the others fixed)."""
    s = len(agr.depth)
    k = np.full(s, k_init)
    for _ in range(sweeps):
        for i in range(s):
            def f(g, i=i):
                kk = k.copy()
                kk[i] = g
                return agr.cost(kk, views=[i])
            k[i], _, _ = _scan(f, k[i] - half, k[i] + half, 0.0025)
            k[i], _, _ = _scan(f, k[i] - 0.0025, k[i] + 0.0025, 0.0005)
    return {"k": k.tolist(), "cost": float(agr.cost(k))}
