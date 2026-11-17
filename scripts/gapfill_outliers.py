"""Task C: where the gap-fill outliers come from, and principled filters against them (no GT inside any filter).

Usage:
    python scripts/gapfill_outliers.py --pilot                 # pencilpot only, full grid of filter settings
    python scripts/gapfill_outliers.py --chosen <json>         # dev + set A + set B with the settings chosen on pencilpot
Inputs per shape: outputs/<s>_gs/gaussians/splat.ply + points/gaussians.json (trained 3DGS), outputs/<s>_gskd/align/sim3.json
(VGGT -> GT frame), outputs/<s>_dense/align/points_aligned.ply (1M TSDF points, GT frame), outputs/<s>_craw/points/views.npz
(the 8 filtered VGGT depth maps, silhouettes and cameras; scripts/save_depth_views.py), data/scenes/<s>/gt_normalized.ply.

1. Redraws the 1M kernel samples exactly as the dense gap fill did (gaussians.py sample_kernels, seed 0, opacity-weighted,
   2-sigma truncation; checked against <s>_gskd), keeping each sample's source Gaussian and its position in sigma units.
2. Selects the gap points as the pipeline does (fill.py: 3-10% of the TSDF cloud's extent from the nearest TSDF point).
3. GT is used ONLY to score: exact distance of every point to the GT surface (protocol v1 frame, GT extent 1).
4. Filters on the gap points, all computed from the reconstruction only (VGGT frame):
   (a) kernel truncation at 1.5 and 1 sigma instead of 2 (samples redrawn, gap selection redone);
   (b) source Gaussian opacity >= o and/or largest axis <= s x object size;
   (c) multi-view free space / silhouette: project the point into each view; a view votes against it if the point lands
       on background (outside VGGT's foreground, conf <= 1) or in front of the fused depth by more than a margin
       m x object size; reject when at least k views vote against;
   (d) projective TSDF value at the point (the uniform-weight average of truncated d_obs - z over the views that observed
       it, as in the fusion, trunc = 6 voxels = 0.03 x object size), keep if |tsdf| < tau x trunc; points no view observed
       (no fused depth there) have no TSDF value and are kept.
5. For each filter: the filled cloud (all TSDF points + kept gap points) scored with protocol v1 + outliers (exact
   distance > 0.01 / 0.02), and the precision of the added points alone.
Writes outputs/<s>_dfill/eval/outliers_<tag>.json and outputs/<s>_dfill/eval/gap_points.npz (per gap point: GT distance,
features, filter decisions) for the report and the figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from photo2mesh.eval.protocol import score_points, unit_cube_transform
from photo2mesh.io import load_mesh, load_points_ply
from photo2mesh.points.fill import gap_fill_mask
from photo2mesh.points.gaussians import kernel_frames, read_ply_vertices, sample_kernels

sys.path.insert(0, str(Path(__file__).parent))
from equal_count import surface_distance  # noqa: E402
from make_points_queue import DEV, SET_A, SET_B  # noqa: E402

N_KERNEL, LO, HI = 1_000_000, 0.03, 0.10
PILOT = {"a": [{"trunc": 1.5}, {"trunc": 1.0}],
         "b": [{"min_op": 0.7}, {"min_op": 0.9}, {"max_s": 0.02}, {"max_s": 0.01}, {"min_op": 0.7, "max_s": 0.02}],
         "c": [{"m": m, "k": k} for m in (0.01, 0.02, 0.03) for k in (1, 2)],
         "d": [{"tau": 0.5}, {"tau": 0.9}]}


class Shape:
    """Everything one shape needs, loaded once."""

    def __init__(self, s: str):
        self.s = s
        o = Path("outputs")
        self.v = read_ply_vertices(o / f"{s}_gs" / "gaussians" / "splat.ply")
        self.g = json.loads((o / f"{s}_gs" / "points" / "gaussians.json").read_text())
        sim = json.loads((o / f"{s}_gskd" / "align" / "sim3.json").read_text())
        self.sim = (sim["s"], np.array(sim["R"]), np.array(sim["t"]))
        base = load_points_ply(o / f"{s}_dense" / "align" / "points_aligned.ply")
        self.base, self.base_n = base["points"].astype(np.float64), base["normals"]
        vw = np.load(o / f"{s}_craw" / "points" / "views.npz")
        self.views = {k: vw[k] for k in vw.files}
        cams = np.load(o / f"{s}_tex_v3" / "points" / "cameras.npz")
        assert np.allclose(cams["extrinsic"], self.views["extrinsic"], atol=1e-4), "VGGT rerun cameras differ from tex_v3"
        self.gt = load_mesh(Path("data/scenes") / s / "gt_normalized.ply")
        self.centre, self.scale = unit_cube_transform(self.gt)
        gt_n = self.gt.copy()
        gt_n.vertices = (self.gt.vertices - self.centre) * self.scale
        self.gt_n = gt_n
        self.base_d = surface_distance((self.base - self.centre) * self.scale, gt_n)
        self.R, self.S = kernel_frames(self.v)
        self.opacity = 1.0 / (1.0 + np.exp(-self.v["opacity"].astype(np.float64)))
        self.size_vggt = float(self.views["object_size"])
        # per-Gaussian normal: as the 1M samples of <s>_gskd carry it (centre PCA normal, gaussian_kernel_points.py)
        gskd = load_points_ply(o / f"{s}_gskd" / "align" / "points_aligned.ply")
        pts2, _, pick2 = self.kernels(2.0)
        assert np.allclose(self.to_gt(pts2), gskd["points"], atol=1e-4), "redrawn kernel samples differ from <s>_gskd"
        self.normal_of = {}
        _, first = np.unique(pick2, return_index=True)
        self.gauss_normal = np.zeros((len(self.opacity), 3))
        self.gauss_normal[pick2[first]] = gskd["normals"][first]

    def kernels(self, trunc: float):
        g = self.g
        return sample_kernels(self.v, N_KERNEL, g["min_opacity"], g["max_scale"] * g["object_size"], "opacity", trunc, seed=0)

    def to_gt(self, x: np.ndarray) -> np.ndarray:
        s, r, t = self.sim
        return s * x @ r.T + t

    def gap(self, trunc: float) -> dict:
        """Gap points of the dense gap fill drawn at this truncation, with features (VGGT and GT frames)."""
        pts, _, pick = self.kernels(trunc)
        x_gt = self.to_gt(pts)
        keep = gap_fill_mask(self.base, x_gt, LO, HI)
        x, xg, pk = pts[keep], x_gt[keep], pick[keep]
        z = np.einsum("nji,nj->ni", self.R[pk], x - np.stack([self.v["x"], self.v["y"], self.v["z"]], 1)[pk]) / self.S[pk]
        d_tsdf = cKDTree(self.base).query(xg)[0] / float(np.ptp(self.base, axis=0).max())
        out = {"x": x, "x_gt": xg, "pick": pk, "normals": self.gauss_normal[pk],
               "gt_dist": surface_distance((xg - self.centre) * self.scale, self.gt_n),
               "opacity": self.opacity[pk], "max_scale": self.S[pk].max(1) / self.g["object_size"],
               "flatness": self.S[pk].min(1) / self.S[pk].max(1), "sigma": np.abs(z).max(1), "d_tsdf": d_tsdf}
        out.update(self.multiview(x))
        return out

    def multiview(self, x: np.ndarray) -> dict:
        """Per point and view: background hit, free-space margin (signed, in object sizes), TSDF contribution."""
        vw = self.views
        s_, h, w = vw["depth"].shape
        trunc = 0.03 * self.size_vggt                            # 6 voxels x 0.005 x object size (fusion.py)
        sil = np.zeros(len(x), np.int32)
        front = np.full((s_, len(x)), -np.inf)                   # (d_obs - z) / size where observed, else -inf
        acc, cnt, vis = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x), np.int32)
        for j in range(s_):
            cam = x @ vw["extrinsic"][j, :, :3].T + vw["extrinsic"][j, :, 3]
            zc = cam[:, 2]
            ok = zc > 1e-6
            u = np.full(len(x), -1)
            v = np.full(len(x), -1)
            u[ok] = np.round(vw["intrinsic"][j, 0, 0] * cam[ok, 0] / zc[ok] + vw["intrinsic"][j, 0, 2]).astype(int)
            v[ok] = np.round(vw["intrinsic"][j, 1, 1] * cam[ok, 1] / zc[ok] + vw["intrinsic"][j, 1, 2]).astype(int)
            inside = ok & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            i = np.flatnonzero(inside)
            fg = vw["fg"][j, v[i], u[i]]
            sil[i[~fg]] += 1
            vis[i[fg]] += 1
            obs = vw["keep"][j, v[i], u[i]]
            io = i[obs]
            sdf = vw["depth"][j, v[io], u[io]] - zc[io]
            front[j, io] = sdf / self.size_vggt
            upd = sdf >= -trunc
            acc[io[upd]] += np.minimum(sdf[upd], trunc)
            cnt[io[upd]] += 1
        tsdf = np.where(cnt > 0, acc / np.maximum(cnt, 1) / trunc, np.nan)
        return {"sil": sil, "front": front, "tsdf": tsdf, "n_obs": cnt.astype(np.int32), "n_vis": vis}


def keep_mask(gp: dict, family: str, p: dict) -> np.ndarray:
    if family == "b":
        k = np.ones(len(gp["gt_dist"]), bool)
        if "min_op" in p:
            k &= gp["opacity"] >= p["min_op"]
        if "max_s" in p:
            k &= gp["max_scale"] <= p["max_s"]
        return k
    if family == "c":
        votes = gp["sil"] + (gp["front"] > p["m"]).sum(0)
        return votes < p["k"]
    if family == "d":
        return ~(np.abs(gp["tsdf"]) >= p["tau"])            # NaN (no observation) -> kept
    return np.ones(len(gp["gt_dist"]), bool)


def evaluate(sh: Shape, gp: dict, keep: np.ndarray) -> dict:
    pts = np.vstack([sh.base, gp["x_gt"][keep]])
    nrm = np.vstack([sh.base_n, gp["normals"][keep]])
    r = score_points(pts, nrm, sh.gt)
    d = np.concatenate([sh.base_d, gp["gt_dist"][keep]])
    ga = gp["gt_dist"][keep]
    return {"added": int(keep.sum()), "n": int(len(pts)), **{k: r[k] for k in ["CD", "HD", "F1@0.01", "F1@0.02", "NC",
            "R_all@0.01", "R_all@0.02"]}, "out@0.01": float(100 * (d > 0.01).mean()), "out@0.02": float(100 * (d > 0.02).mean()),
            "added_prec@0.02": float(100 * (ga <= 0.02).mean()) if len(ga) else float("nan"),
            "added_prec@0.01": float(100 * (ga <= 0.01).mean()) if len(ga) else float("nan")}


def run_shape(s: str, settings: dict, tag: str) -> dict:
    sh = Shape(s)
    gp2 = sh.gap(2.0)
    res = {"TSDF only": evaluate(sh, gp2, np.zeros(len(gp2["gt_dist"]), bool)),
           "gap fill (current, 2 sigma)": evaluate(sh, gp2, np.ones(len(gp2["gt_dist"]), bool))}
    decisions = {}
    for fam, plist in settings.items():
        for p in plist:
            name = f"({fam}) " + ", ".join(f"{k} {v}" for k, v in p.items())
            if fam == "a":
                gpa = sh.gap(p["trunc"])
                res[name] = evaluate(sh, gpa, np.ones(len(gpa["gt_dist"]), bool))
            else:
                k = keep_mask(gp2, fam, p)
                decisions[name] = k
                res[name] = evaluate(sh, gp2, k)
    o = Path("outputs") / f"{s}_dfill" / "eval"
    o.mkdir(parents=True, exist_ok=True)
    (o / f"outliers_{tag}.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(o / f"gap_points_{tag}.npz", x_gt=gp2["x_gt"].astype(np.float32), gt_dist=gp2["gt_dist"],
                        opacity=gp2["opacity"], max_scale=gp2["max_scale"], flatness=gp2["flatness"], sigma=gp2["sigma"],
                        d_tsdf=gp2["d_tsdf"], sil=gp2["sil"], n_vis=gp2["n_vis"], n_obs=gp2["n_obs"], tsdf=gp2["tsdf"],
                        front_max=np.max(gp2["front"], axis=0),
                        **{f"keep_{i}": k for i, k in enumerate(decisions.values())}, filter_names=np.array(list(decisions)))
    print(f"[outliers] {s}: {len(gp2['gt_dist']):,} gap points; " +
          "; ".join(f"{n}: +{r['added']:,} CD {r['CD']:.2f} out@0.02 {r['out@0.02']:.1f}%" for n, r in list(res.items())[:3]))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--chosen", default=None, help="json file {family: [settings]} chosen on pencilpot")
    ap.add_argument("--shapes", nargs="*", default=None)
    args = ap.parse_args()
    if args.pilot:
        run_shape(DEV, PILOT, "pilot")
        return
    chosen = json.loads(Path(args.chosen).read_text())
    for s in args.shapes or [DEV] + SET_A + SET_B:
        run_shape(s, chosen, "chosen")


if __name__ == "__main__":
    main()
