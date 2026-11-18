"""Gaussian gap fill by GT region (item 2) and points per Gaussian (item 3), Prof. Zhong's email of 17 Nov.

Usage: python scripts/zhong_gauss.py regions [--shapes ...]   -> outputs/<s>_dfill/eval/zhong_regions.json
       python scripts/zhong_gauss.py ppg [--shapes ...]       -> outputs/<s>_dfill/eval/zhong_ppg.json, zhong_size.npz
Inputs as scripts/gapfill_outliers.py (Shape: trained splats <s>_gs, sim3 of <s>_gskd, 1M TSDF points <s>_dense, VGGT
views <s>_craw) plus outputs/<s>_dense/eval/gt_regions.npz (scripts/gt_regions.py). GT only scores: exact distance to
the GT triangles and the region labels of the nearest GT sample; no filter or sampler reads the GT.

regions: for opacity >= o in {0.3, 0.5, 0.7, 0.9} and largest axis <= a in {0.01, 0.02, 0.05} x object size, redraw the
  1M opacity-weighted kernel samples from the Gaussians passing the filter (sample_kernels, seed 0, 2 sigma), select the
  gap points as the pipeline does (3-10% of the TSDF cloud's extent from the nearest TSDF point), and report per GT
  region: points added (region of the nearest GT sample), precision (share within 0.02 of the GT surface) and GT area
  filled (region GT samples > 0.02 from the TSDF cloud that come within 0.02 of an added point). o 0.5 / a 0.05 is the
  current gap fill.
ppg: N = Gaussians passing the default filter (opacity >= 0.5, largest axis <= 0.05). For k in {20, 50, 83, 150} and the
  same total T = k N: (a) opacity-weighted multinomial (current rule), (b) exactly k per Gaussian, (c) area-weighted
  multinomial (weight s1 s2 of the two largest axes), (d) surface-aligned: opacity-weighted counts, points on the disk
  of the two largest axes (smallest axis set to 0). Each cloud: protocol v1 (Chamfer, F1, NC), exact-distance precision
  and outliers, density uniformity (CV of the nearest-neighbour distance), and as a gap fill (added points, their
  precision, % of the TSDF gap filled). Also, for the current 1M draw, per sample: source Gaussian largest axis, axis
  ratio (smallest / largest), number of views that see its centre, and whether it is an outlier (> 0.02).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from photo2mesh.eval.protocol import score_points
from photo2mesh.points.fill import gap_fill_mask
from photo2mesh.points.gaussians import sample_kernels, sample_kernels_counts

sys.path.insert(0, str(Path(__file__).parent))
from equal_count import surface_distance  # noqa: E402
from gapfill_outliers import HI, LO, Shape  # noqa: E402
from make_points_queue import DEV, SET_A, SET_B  # noqa: E402

OPAC, AXES, KS = (0.3, 0.5, 0.7, 0.9), (0.01, 0.02, 0.05), (20, 50, 83, 150)
REGIONS = ("boundary", "thin", "unobserved", "rest", "all")


class Ctx:
    """A Shape plus the GT region labels and the TSDF gap, in protocol v1's normalised frame."""

    def __init__(self, s: str):
        self.sh = sh = Shape(s)
        r = np.load(Path("outputs") / f"{s}_dense" / "eval" / "gt_regions.npz")
        self.gp = r["p"].astype(np.float64)
        lab = {k: r[k] for k in ("boundary", "thin", "unobserved")}
        lab["rest"] = ~(lab["boundary"] | lab["thin"] | lab["unobserved"])
        lab["all"] = np.ones(len(self.gp), bool)
        self.lab = lab
        self.gtree = cKDTree(self.gp)
        self.base_n = self.norm(sh.base)
        self.gap = cKDTree(self.base_n).query(self.gp, distance_upper_bound=0.05)[0] > 0.02
        self.kept = (sh.opacity >= sh.g["min_opacity"]) & (sh.S.max(1) <= sh.g["max_scale"] * sh.g["object_size"])

    def norm(self, x_gt: np.ndarray) -> np.ndarray:
        return (x_gt - self.sh.centre) * self.sh.scale

    def normals(self, pick: np.ndarray, axis_nrm: np.ndarray) -> np.ndarray:
        """GT-frame normals: the source Gaussian's centre PCA normal (as the gap fill), its shortest axis if missing."""
        n = self.sh.gauss_normal[pick].copy()
        miss = ~n.any(1)
        n[miss] = axis_nrm[miss] @ self.sh.sim[1].T
        return n

    def as_fill(self, x_gt: np.ndarray) -> dict:
        """Gap points of a kernel cloud (pipeline rule) and what they bring, per region."""
        keep = gap_fill_mask(self.sh.base, x_gt, LO, HI)
        a = self.norm(x_gt[keep])
        out = {"n_added": int(keep.sum())}
        if keep.any():
            d = surface_distance(a, self.sh.gt_n)
            reg = self.gtree.query(a)[1]
            near = cKDTree(a).query(self.gp[self.gap], distance_upper_bound=0.05)[0] <= 0.02
        else:
            d, reg, near = np.zeros(0), np.zeros(0, int), np.zeros(int(self.gap.sum()), bool)
        filled = np.zeros(len(self.gp), bool)
        filled[np.flatnonzero(self.gap)[near]] = True
        for k in REGIONS:
            m = self.lab[k][reg]
            area = self.lab[k]
            out[k] = {"added": int(m.sum()), "precision": float(100 * (d[m] <= 0.02).mean()) if m.any() else None,
                      "area": float(100 * area.mean()), "gap_of_region": float(100 * self.gap[area].mean()),
                      "filled_of_gap": float(100 * filled[area].sum() / max(1, (self.gap & area).sum())),
                      "filled_of_area": float(100 * filled[area].mean())}
        return out, keep


def regions(s: str) -> dict:
    c = Ctx(s)
    sh = c.sh
    res = {}
    for o in OPAC:
        for a in AXES:
            pts, nrm, pick = sample_kernels(sh.v, 1_000_000, o, a * sh.g["object_size"], "opacity", 2.0, seed=0)
            x_gt = sh.to_gt(pts)
            fill, keep = c.as_fill(x_gt)
            base_d = sh.base_d
            add = c.norm(x_gt[keep])
            union = np.vstack([sh.base, x_gt[keep]])
            un = np.vstack([sh.base_n, c.normals(pick[keep], nrm[keep])])
            r = score_points(union, un, sh.gt)
            d_all = np.concatenate([base_d, surface_distance(add, sh.gt_n)]) if len(add) else base_d
            fill.update({"gaussians": int(len(np.unique(pick))), "CD": r["CD"], "F1@0.01": r["F1@0.01"],
                         "R_all@0.02": r["R_all@0.02"], "out@0.02": float(100 * (d_all > 0.02).mean())})
            res[f"o{o}_a{a}"] = fill
            print(f"[regions] {s} o {o} a {a}: +{fill['n_added']:,}  prec {fill['all']['precision']}  "
                  f"filled {fill['all']['filled_of_gap']:.1f}%  CD {r['CD']:.2f}", flush=True)
    o = Path("outputs") / f"{s}_dfill" / "eval"
    (o / "zhong_regions.json").write_text(json.dumps(res, indent=2))
    return res


def cloud_metrics(c: Ctx, pts: np.ndarray, nrm_axis: np.ndarray, pick: np.ndarray) -> dict:
    sh = c.sh
    x_gt = sh.to_gt(pts)
    xn = c.norm(x_gt)
    r = score_points(x_gt, c.normals(pick, nrm_axis), sh.gt)
    d = surface_distance(xn, sh.gt_n)
    nn = cKDTree(xn).query(xn, k=2)[0][:, 1]
    fill, _ = c.as_fill(x_gt)
    per_g = np.bincount(pick, minlength=len(sh.opacity))[c.kept]
    return {"n": int(len(pts)), "CD": r["CD"], "F1@0.01": r["F1@0.01"], "NC": r["NC"], "R_all@0.02": r["R_all@0.02"],
            "precision@0.02": float(100 * (d <= 0.02).mean()), "out@0.01": float(100 * (d > 0.01).mean()),
            "out@0.02": float(100 * (d > 0.02).mean()), "nn_cv": float(nn.std() / nn.mean()),
            "nn_mean": float(nn.mean()), "per_gaussian_mean": float(per_g.mean()), "per_gaussian_cv": float(per_g.std() / per_g.mean()),
            "gaussians_with_0": int((per_g == 0).sum()), "fill_added": fill["n_added"],
            "fill_precision": fill["all"]["precision"], "gap_filled": fill["all"]["filled_of_gap"]}


def ppg(s: str) -> dict:
    c = Ctx(s)
    sh = c.sh
    idx = np.flatnonzero(c.kept)
    nk = len(idx)
    op = sh.opacity[idx]
    ss = np.sort(sh.S[idx], 1)
    area = ss[:, 1] * ss[:, 2]
    res = {"N_kept": int(nk), "n_current": 1_000_000, "mean_current": 1_000_000 / nk,
           "opacity": {"mean": float(op.mean()), "min": float(op.min()), "max": float(op.max())},
           "expected_count_current": {"min": float(1e6 * op.min() / op.sum()), "max": float(1e6 * op.max() / op.sum())}}
    full = lambda cnt: np.bincount(idx, weights=cnt, minlength=len(sh.opacity)).astype(np.int64)
    for k in KS:
        t = k * nk
        rng = np.random.default_rng(0)
        variants = {
            "a_opacity": sample_kernels(sh.v, t, sh.g["min_opacity"], sh.g["max_scale"] * sh.g["object_size"], "opacity", 2.0, seed=0),
            "b_fixed": sample_kernels_counts(sh.v, full(np.full(nk, k)), seed=0),
            "c_area": sample_kernels_counts(sh.v, full(rng.multinomial(t, area / area.sum())), seed=0),
            "d_disk": sample_kernels_counts(sh.v, full(np.random.default_rng(1).multinomial(t, op / op.sum())), disk=True, seed=0)}
        for name, (pts, nrm, pick) in variants.items():
            res[f"k{k}_{name}"] = m = cloud_metrics(c, pts, nrm, pick)
            print(f"[ppg] {s} k {k} {name}: n {m['n']:,} CD {m['CD']:.2f} prec {m['precision@0.02']:.1f} "
                  f"nn_cv {m['nn_cv']:.2f} gap filled {m['gap_filled']:.1f}%", flush=True)
    # outliers vs Gaussian size and visibility: the current draw (1M, as the dense gap fill)
    pts, _, pick = sh.kernels(2.0)
    x_gt = sh.to_gt(pts)
    d = surface_distance(c.norm(x_gt), sh.gt_n)
    gap = gap_fill_mask(sh.base, x_gt, LO, HI)
    ids = np.unique(pick)
    cen = np.stack([sh.v["x"], sh.v["y"], sh.v["z"]], 1).astype(np.float64)[ids]
    mv = sh.multiview(cen)
    occluded = (np.isfinite(mv["front"]) & (mv["front"] < -0.02)).sum(0)
    n_see = np.zeros(len(sh.opacity), np.int32)
    n_see[ids] = mv["n_vis"] - occluded
    o = Path("outputs") / f"{s}_dfill" / "eval"
    np.savez_compressed(o / "zhong_size.npz", out=d > 0.02, gap=gap,
                        max_axis=(sh.S.max(1) / sh.g["object_size"])[pick].astype(np.float32),
                        ratio=(sh.S.min(1) / sh.S.max(1))[pick].astype(np.float32), n_see=n_see[pick].astype(np.int8))
    (o / "zhong_ppg.json").write_text(json.dumps(res, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["regions", "ppg"])
    ap.add_argument("--shapes", nargs="*", default=None)
    args = ap.parse_args()
    for s in args.shapes or [DEV] + SET_A + SET_B:
        (regions if args.what == "regions" else ppg)(s)


if __name__ == "__main__":
    main()
