"""Report for Prof. Zhong's email of 17 Nov (five items) -> docs/results/2026-11-19_zhong_followups.md + two figures.

Usage: python scripts/zhong_report.py [--score-only] [--out docs/results/2026-11-19_zhong_followups.md]
Inputs:
  item 1: outputs/<s>_{pmap,dunp}/align/points_aligned.ply (scripts/pm_run.sh), outputs/<s>_dense (current TSDF cloud).
          Scored here at exactly 1M points (numpy default_rng(0); a cloud with fewer points is drawn with replacement,
          which leaves every metric's expectation unchanged), protocol v1 + exact-distance outliers + signed offset;
          written to outputs/<run>/eval/{equal_count,signed_offset}.json like every other 1M score.
  items 2-3: outputs/<s>_dfill/eval/{zhong_regions.json, zhong_ppg.json, zhong_size.npz} (scripts/zhong_gauss.py),
          outputs/<s>_dense/eval/gt_regions.npz (scripts/gt_regions.py).
Hand-written blocks between <!-- tag --> markers survive a rerun.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from photo2mesh.eval.protocol import score_points, unit_cube_transform
from photo2mesh.io import load_mesh, load_points_ply

sys.path.insert(0, str(Path(__file__).parent))
from conf_report import normalised, signed_offset  # noqa: E402
from ds_report import load, ms  # noqa: E402
from equal_count import surface_distance  # noqa: E402
from make_points_queue import SET_A, SET_B  # noqa: E402
from zhong_gauss import AXES, KS, OPAC  # noqa: E402

O = Path("outputs")
FIG = Path("docs/results/figures")
SETS = [("Set B (12 shapes, clean test set)", SET_B), ("Set A (7 shapes)", SET_A)]


def score_1m(shape: str, suf: str, n: int = 1_000_000, seed: int = 0) -> dict:
    """Protocol v1 + outliers + signed offset on exactly n points (with replacement if the cloud is smaller)."""
    run = O / f"{shape}{suf}"
    f, fo = run / "eval" / "equal_count.json", run / "eval" / "signed_offset.json"
    if f.exists() and fo.exists():
        return load(shape, suf)
    pts = load_points_ply(run / "align" / "points_aligned.ply")
    m = len(pts["points"])
    idx = np.random.default_rng(seed).choice(m, n, replace=m < n)
    p, nrm = pts["points"][idx].astype(np.float64), pts["normals"][idx]
    gt = load_mesh(Path("data/scenes") / shape / "gt_normalized.ply")
    out = {"run": run.name, "n_source": int(m), "n": n, "seed": seed, "replace": bool(m < n), **score_points(p, nrm, gt)}
    centre, scale = unit_cube_transform(gt)
    _, g, _, _ = normalised(shape)
    d = surface_distance((p - centre) * scale, g)
    out.update({"out@0.01": float(100 * (d > 0.01).mean()), "out@0.02": float(100 * (d > 0.02).mean()),
                "dist_mean": float(d.mean()), "dist_p95": float(np.percentile(d, 95))})
    (run / "eval").mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=2))
    q = (pts["points"].astype(np.float64) - centre) * scale
    off = signed_offset(q[np.random.default_rng(0).choice(m, 100_000, replace=False)], g)
    fo.write_text(json.dumps({"signed_offset": off}))
    print(f"[score] {run.name}: {m:,} -> {n:,}  CD {out['CD']:.2f}  out@0.02 {out['out@0.02']:.1f}%  offset {off:+.4f}", flush=True)
    return load(shape, suf)


PM_COLS = [("CD", "Chamfer", 2), ("F1@0.01", "F1@0.01", 3), ("F1@0.02", "F1@0.02", 3), ("NC", "NC", 3),
           ("R_all@0.02", "recall all @0.02", 3), ("out@0.01", "outliers > 0.01 (%)", 1), ("out@0.02", "outliers > 0.02 (%)", 1),
           ("offset", "signed offset", 4)]
PM_ROWS = [("_pmap", "(a) VGGT point head (world_points), no fusion"),
           ("_dunp", "(b) depth maps unprojected with the known focal, no fusion"),
           ("_dense", "(c) TSDF fusion of (b) (current default)")]


def paired(base: list[dict], new: list[dict], lbl: str) -> str:
    parts = []
    for k, l, d in PM_COLS:
        get = (lambda r: abs(r["offset"])) if k == "offset" else (lambda r, k=k: r[k])
        dv = np.array([get(b) - get(a) for a, b in zip(base, new)])
        lower = k in ("CD", "offset") or k.startswith("out")
        better = (dv < 0).sum() if lower else (dv > 0).sum()
        parts.append(f"{'|signed offset|' if k == 'offset' else l} {dv.mean():+.{d}f} (better on {better} of {len(dv)})")
    return f"- {lbl}: " + "; ".join(parts) + "."


def item1() -> list[str]:
    L = ["## 1. VGGT point map vs unprojected depth vs TSDF", "",
         "**Method.** One VGGT pass per shape on the 8 default views, rerun under the memory guard with the raw output kept "
         "(`scripts/pm_run.sh`; the rerun has the same cameras, the same fused-pixel count and the same alignment as the 15 Nov "
         "`<s>_dense` run on every shape). `scripts/pointmap_clouds.py` keeps **the same pixels for (a) and (b)**: the default "
         "TSDF pixel filter (conf > 1, 2 px foreground erosion; the percentile is inert, 17 Nov), computed once from the depth "
         "head. (a) = the point head's `world_points` at those pixels, in VGGT's world frame. The point head cannot take the "
         "known focal, so it carries VGGT's own focal implicitly. (b) = the depth head's depth at those pixels, unprojected "
         "with the known focal. These are exactly the points the TSDF fuses. (c) = the current TSDF cloud (1M points "
         "sampled from the fused surface). Normals for (a) and (b): PCA over 20 neighbours, flipped towards the camera that "
         "saw the point (the v2 points stage). All three use the same camera alignment (Umeyama on VGGT's camera centres, "
         "one sim3, no ICP) and protocol v1 at **exactly 1,000,000 points** (`default_rng(0)`). (a) and (b) have 0.42M to "
         "1.07M kept pixels, so most are drawn with replacement. This repeats points but leaves every metric's "
         "expectation unchanged: CD and F1 use random subsamples, recall uses the point set, outliers are a share.", ""]
    R = {(s, x): score_1m(s, x) for _, sh in SETS for s in sh for x, _ in PM_ROWS}
    info = {s: json.loads((O / f"{s}_pmap" / "points" / "pointmap.json").read_text()) for _, sh in SETS for s in sh}
    for name, sh in SETS:
        L += [f"**{name}**", "", "| points | " + " | ".join(c[1] for c in PM_COLS) + " |", "|---|" + "---:|" * len(PM_COLS)]
        for x, lbl in PM_ROWS:
            L.append(f"| {lbl} | " + " | ".join(ms([R[(s, x)][k] for s in sh], d) for k, _, d in PM_COLS) + " |")
        L += ["", "Paired per shape:", "",
              paired([R[(s, "_dunp")] for s in sh], [R[(s, "_dense")] for s in sh], "TSDF (c) minus unprojected depth (b), the effect of fusion"),
              paired([R[(s, "_dunp")] for s in sh], [R[(s, "_pmap")] for s in sh], "point head (a) minus unprojected depth (b)"), ""]
    L += ["**Per shape** (Chamfer / outliers > 0.02 % / signed offset)", "",
          "| shape | kept pixels | point head vs (b), median (% of depth) | (a) point head | (b) unprojected depth | (c) TSDF |",
          "|---|---:|---:|---:|---:|---:|"]
    cell = lambda r: f"{r['CD']:.2f} / {r['out@0.02']:.1f} / {r['offset']:+.4f}"
    for name, sh in SETS:
        for s in sh:
            L.append(f"| {s}{' (B)' if s in SET_B else ''} | {info[s]['pixels_kept']:,} | {100 * info[s]['pmap_vs_dunp_rel_median']:.2f} | "
                     + " | ".join(cell(R[(s, x)]) for x, _ in PM_ROWS) + " |")
    return L + ["", "<!-- interp1 -->", "<!-- /interp1 -->", ""]


REG = [("boundary", "boundary / silhouette-adjacent"), ("thin", "thin (< 3% of extent)"), ("unobserved", "unobserved (no view)"),
       ("rest", "rest (none of the three)"), ("all", "whole surface")]


def item2() -> list[str]:
    shapes = SET_B + SET_A
    Z = {s: json.loads((O / f"{s}_dfill" / "eval" / "zhong_regions.json").read_text()) for s in shapes}
    L = ["## 2. Opacity / scale thresholds by GT region", "",
         "**Regions** (`scripts/gt_regions.py`). They are labelled on the 100k GT surface samples of the protocol sampler; "
         "the GT is used only for these labels and for scoring. No filter or sampler reads it. Regions may overlap.",
         "- **boundary**: some input view sees the sample within 3 px of the GT silhouette or of an occlusion edge (neighbouring GT "
         "depths differ by > 2%). The input filter erodes VGGT's foreground by 2 px, so this is where pixels are lost.",
         "- **thin**: local thickness < 0.03 of the extent. Thickness = median distance to the opposite GT surface along the "
         "inward normal and 4 rays at 20 deg around it. Inward is decided by an occupancy test, so face orientation does "
         "not matter.",
         "- **unobserved**: none of the 8 input views sees the sample (outside every image, or occluded; GT ray casting from "
         "the render cameras; silhouettes checked against the renders, IoU 0.99).", "",
         "**Thresholds.** For each opacity >= o and largest axis <= a x object size, the 1M opacity-weighted kernel samples "
         "are redrawn from the Gaussians that pass (seed 0, 2 sigma). The gap points are then selected exactly as the "
         "pipeline does: 3-10% of the TSDF cloud's extent from the nearest TSDF point. o 0.5 / a 0.05 is the current gap fill. "
         "Per region:",
         "- **added**: gap points whose nearest GT sample is in the region, mean per shape;",
         "- **precision**: share of them within 0.02 of the GT surface (exact distance), pooled over shapes;",
         "- **filled**: share of the region's TSDF gap (GT samples > 0.02 from the TSDF cloud) that now has an added point within "
         "0.02, mean over shapes.", ""]
    for name, sh in [("Set B (12 shapes)", SET_B), ("Set A (7 shapes)", SET_A)]:
        L += [f"**{name}**: region sizes, % of the GT surface (mean): " + ", ".join(
            f"{lbl.split(' (')[0]} {np.mean([Z[s]['o0.5_a0.05'][k]['area'] for s in sh]):.1f} "
            f"(TSDF gap inside it {np.nanmean([Z[s]['o0.5_a0.05'][k]['gap_of_region'] for s in sh]):.0f}%)" for k, lbl in REG[:4]) + ".", ""]
        L += ["| opacity >= | max axis <= | " + " | ".join(f"{lbl}: added / precision % / filled %" for _, lbl in REG) +
              " | filled cloud: Chamfer / outliers > 0.02 % |", "|---:|---:|" + "---:|" * (len(REG) + 1)]
        for o in OPAC:
            for a in AXES:
                key = f"o{o}_a{a}"
                cells = []
                for k, _ in REG:
                    add = np.array([Z[s][key][k]["added"] for s in sh])
                    prec = [Z[s][key][k]["precision"] for s in sh]
                    pooled = sum(p * n for p, n in zip(prec, add) if p is not None) / max(1, add.sum())
                    cells.append(f"{add.mean():,.0f} / {pooled:.0f} / {np.mean([Z[s][key][k]['filled_of_gap'] for s in sh]):.1f}")
                cur = " (current)" if key == "o0.5_a0.05" else ""
                L.append(f"| {o}{cur} | {a} | " + " | ".join(cells) +
                         f" | {np.mean([Z[s][key]['CD'] for s in sh]):.2f} / {np.mean([Z[s][key]['out@0.02'] for s in sh]):.1f} |")
        tsdf_cd = np.mean([load(s, "_dense")["CD"] for s in sh])
        tsdf_out = np.mean([load(s, "_dense")["out@0.02"] for s in sh])
        L += ["", f"TSDF alone (no gap points), same shapes: Chamfer {tsdf_cd:.2f}, outliers > 0.02 {tsdf_out:.1f}%. (The "
              "filled-cloud columns score TSDF 1M + all added points, not redrawn to 1M, as in the 18 Nov outlier report.)", ""]
    L += [f"![GT regions](figures/2026-11-19_regions.png)", "",
          "*GT surface samples coloured by region (priority where they overlap: unobserved > thin > boundary). Grey = rest.*", "",
          "<!-- interp2 -->", "<!-- /interp2 -->", ""]
    return L


def fig_regions(shapes: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"rest": "#c9c9c9", "boundary": "#2a78d6", "thin": "#e8833a", "unobserved": "#a23b8f"}
    fig = plt.figure(figsize=(12, 4.2 * len(shapes)))
    for i, s in enumerate(shapes):
        r = np.load(O / f"{s}_dense" / "eval" / "gt_regions.npz")
        lab = np.full(len(r["p"]), "rest", object)
        lab[r["boundary"]] = "boundary"
        lab[r["thin"]] = "thin"
        lab[r["unobserved"]] = "unobserved"
        sub = np.random.default_rng(0).choice(len(lab), 40_000, replace=False)
        p, lab = r["p"][sub], lab[sub]
        for j, (el, az) in enumerate([(25, 30), (25, 210), (-35, 120)]):
            ax = fig.add_subplot(len(shapes), 3, 3 * i + j + 1, projection="3d")
            for k in ["rest", "boundary", "thin", "unobserved"]:
                m = lab == k
                ax.scatter(p[m, 0], p[m, 2], p[m, 1], s=0.3, c=col[k], label=k, depthshade=False, rasterized=True)
            ax.view_init(el, az)
            ax.set_box_aspect((1, 1, 1))
            ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_zlim(-0.5, 0.5)
            ax.set_axis_off()
            sh = r
            if j == 0:
                ax.set_title(f"{s}: boundary {100 * sh['boundary'].mean():.0f}%, thin {100 * sh['thin'].mean():.0f}%, "
                             f"unobserved {100 * sh['unobserved'].mean():.0f}%", fontsize=9, loc="left")
            else:
                ax.set_title("from below" if el < 0 else "opposite side", fontsize=8, loc="left")
    h = [plt.Line2D([], [], marker="o", ls="", color=col[k], label=k) for k in ["boundary", "thin", "unobserved", "rest"]]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "2026-11-19_regions.png", dpi=110)
    plt.close(fig)


VAR = [("a_opacity", "(a) opacity-weighted (current rule)"), ("b_fixed", "(b) fixed k per Gaussian"),
       ("c_area", "(c) area-weighted (s1 s2)"), ("d_disk", "(d) surface-aligned disk, opacity-weighted")]
PPG_COLS = [("n", "points", 0), ("CD", "Chamfer", 2), ("F1@0.01", "F1@0.01", 3), ("NC", "NC", 3),
            ("precision@0.02", "precision (%)", 1), ("out@0.01", "outliers > 0.01 (%)", 1), ("nn_cv", "NN distance CV", 2),
            ("per_gaussian_cv", "per-Gaussian count CV", 2), ("gap_filled", "gap filled (%)", 1), ("fill_precision", "gap points' precision (%)", 1)]


def item3() -> list[str]:
    P = {s: json.loads((O / f"{s}_dfill" / "eval" / "zhong_ppg.json").read_text()) for s in SET_B + SET_A}
    nk = np.array([P[s]["N_kept"] for s in SET_B + SET_A])
    mean_c = 1e6 / nk
    L = ["## 3. Points per Gaussian", "",
         "**Why the average is ~83.** The gap fill draws n = 1,000,000 kernel samples (`gaussians.py` `sample_kernels`). Each "
         "sample picks one Gaussian from those passing the filter (opacity >= 0.5, largest axis <= 0.05 x object size). "
         "The probability is proportional to its opacity, p_i = alpha_i / sum_j alpha_j. So the counts are one multinomial "
         "draw, (c_1, ..., c_N) ~ Multinomial(n, p), and E[c_i] = n alpha_i / sum_j alpha_j. Averaged over the N kept "
         "Gaussians, the mean count is **exactly n / N** whatever the weights, because the counts sum to n. The weights "
         "only move points between Gaussians. "
         f"Over the 19 shapes N = {nk.min():,} to {nk.max():,} (median {int(np.median(nk)):,}), so n / N = "
         f"{mean_c.min():.0f} to {mean_c.max():.0f}, **mean {mean_c.mean():.0f}** (set B {np.mean(1e6 / np.array([P[s]['N_kept'] for s in SET_B])):.0f}). "
         "Opacity runs only from 0.5 to 1, so the expected count of a single Gaussian stays within a factor of 2: "
         f"{np.mean([P[s]['expected_count_current']['min'] for s in SET_B + SET_A]):.0f} to "
         f"{np.mean([P[s]['expected_count_current']['max'] for s in SET_B + SET_A]):.0f} (means over shapes). The count "
         "ignores the Gaussian's size, so a large splat and a tiny one get about the same number of points, and big "
         "splats are sampled sparsely.", "",
         "**Comparison at the same total count.** For each k in {20, 50, 83, 150} the total is T = k x N (per shape), and four "
         "allocations of T samples over the same kept Gaussians are compared:",
         "- (a) opacity-weighted multinomial (current rule);",
         "- (b) exactly k per Gaussian;",
         "- (c) area-weighted multinomial, count proportional to s1 s2 of the two largest axes (uniform density);",
         "- (d) the opacity-weighted counts of (a), with points drawn on the disk of the two largest axes (smallest axis z = 0, "
         "surface-aligned).",
         "Kernel sampling as before (2 sigma truncation; normals = the source Gaussian's centre PCA normal). Metrics on the "
         "whole kernel cloud, in protocol v1's frame:",
         "- protocol v1 (Chamfer, F1@0.01, NC);",
         "- precision = share within 0.02 of the GT surface (exact distance);",
         "- **density uniformity** = coefficient of variation (std / mean) of each point's nearest-neighbour distance, lower "
         "= more uniform;",
         "- as a gap fill (3-10% band from the TSDF cloud): % of the TSDF gap filled and the precision of the added points.", ""]
    for name, sh in SETS:
        L += [f"**{name}**", "", "| total | allocation | " + " | ".join(c[1] for c in PPG_COLS) + " |",
              "|---|---|" + "---:|" * len(PPG_COLS)]
        for k in KS:
            for v, lbl in VAR:
                rs = [P[s][f"k{k}_{v}"] for s in sh]
                L.append(f"| {k} N | {lbl} | " + " | ".join(
                    (f"{np.mean([r[c] for r in rs]):,.0f}" if c == "n" else ms([r[c] for r in rs], d)) for c, _, d in PPG_COLS) + " |")
        L.append("")
    L += ["**Outlier rate vs Gaussian size and visibility** (current draw: the 1M opacity-weighted samples of the dense "
          "gap fill; outlier = exact distance > 0.02; all 19 shapes pooled; bins with < 1,000 samples not drawn). Views "
          "that see the centre = the 8 VGGT views in which the Gaussian's centre falls on VGGT's foreground and is not more "
          "than 2% of the object size behind the fused depth there.", "",
          "![outliers vs Gaussian](figures/2026-11-19_outliers_vs_gaussian.png)", ""]
    L += size_table() + ["", "<!-- interp3 -->", "<!-- /interp3 -->", ""]
    return L


BINS = {"max_axis": [0, 0.005, 0.01, 0.02, 0.03, 0.05], "ratio": [0, 0.05, 0.1, 0.2, 0.4, 1.0001]}


def size_data():
    out = {k: [] for k in ["out", "gap", "max_axis", "ratio", "n_see"]}
    for s in SET_B + SET_A:
        z = np.load(O / f"{s}_dfill" / "eval" / "zhong_size.npz")
        for k in out:
            out[k].append(z[k])
    return {k: np.concatenate(v) for k, v in out.items()}


def binned(x, y, edges):
    b = np.digitize(x, edges[1:-1])
    return [(int((b == i).sum()), float(100 * y[b == i].mean()) if (b == i).any() else np.nan) for i in range(len(edges) - 1)]


def size_table() -> list[str]:
    d = size_data()
    L = ["| feature | bin | samples | outlier rate, all samples (%) | gap points in bin | outlier rate, gap points (%) |",
         "|---|---|---:|---:|---:|---:|"]
    for feat, lbl in [("max_axis", "largest axis (x object size)"), ("ratio", "axis ratio (smallest / largest)")]:
        e = BINS[feat]
        a, g = binned(d[feat], d["out"], e), binned(d[feat][d["gap"]], d["out"][d["gap"]], e)
        for i in range(len(e) - 1):
            L.append(f"| {lbl} | {e[i]:g}-{min(e[i + 1], 1):g} | {a[i][0]:,} | {a[i][1]:.1f} | {g[i][0]:,} | {g[i][1]:.1f} |")
    for v in range(9):
        m = d["n_see"] == v
        mg = m & d["gap"]
        L.append(f"| views that see the centre | {v} | {m.sum():,} | {100 * d['out'][m].mean() if m.any() else float('nan'):.1f} | "
                 f"{mg.sum():,} | {100 * d['out'][mg].mean() if mg.sum() else float('nan'):.1f} |")
    return L


def fig_size() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = size_data()
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    series = [("all kernel samples", np.ones(len(d["out"]), bool), "#2a78d6"), ("gap points", d["gap"], "#e8833a")]
    for ax, feat, lbl in [(axs[0], "max_axis", "largest axis (x object size)"), (axs[1], "ratio", "axis ratio (smallest / largest)")]:
        e = np.array(BINS[feat])
        mid = (e[:-1] + np.minimum(e[1:], 1)) / 2
        for name, m, c in series:
            r = binned(d[feat][m], d["out"][m], list(e))
            ok = np.array([n >= 1000 for n, _ in r])
            ax.plot(mid[ok], np.array([v for _, v in r])[ok], marker="o", lw=2, ms=6, color=c, label=name)
        ax.set_xlabel(lbl)
        ax.set_xticks(mid, [f"{a:g}-{min(b, 1):g}" for a, b in zip(e[:-1], e[1:])], fontsize=8, rotation=20)
    for name, m, c in series:
        xs, ys = [], []
        for v in range(9):
            mm = m & (d["n_see"] == v)
            if mm.sum() >= 1000:
                xs.append(v)
                ys.append(100 * d["out"][mm].mean())
        axs[2].plot(xs, ys, marker="o", lw=2, ms=6, color=c, label=name)
    axs[2].set_xlabel("views that see the Gaussian's centre (of 8)")
    axs[0].set_ylabel("outliers > 0.02 (% of samples)")
    for ax in axs:
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Kernel-sample outlier rate vs source Gaussian (19 shapes pooled, current 1M draw)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "2026-11-19_outliers_vs_gaussian.png", dpi=130)
    plt.close(fig)


def item45() -> list[str]:
    return ["## 4. MeshFlow: paper vs released checkpoint", "",
            "One page with a quote or file:line for each claim: [`docs/meshflow_public_vs_paper.md`](../meshflow_public_vs_paper.md).", "",
            "<!-- item4 -->", "<!-- /item4 -->", "",
            "## 5. Evaluation consistency", "",
            "Section 6 of [`docs/eval_protocol.md`](../eval_protocol.md), checked from the files on disk by `scripts/eval_audit.py`.", "",
            "<!-- item5 -->", "<!-- /item5 -->", ""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results/2026-11-19_zhong_followups.md")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--items", default="12345")
    args = ap.parse_args()
    if args.score_only:
        item1()
        return
    L = ["# Prof. Zhong's follow-ups of 17 Nov: point map vs TSDF, Gaussian thresholds by region, points per Gaussian, "
         "MeshFlow paper vs release, evaluation consistency", "",
         "Fahad Qaseem Khawar, 19 Nov 2026. Generated by `scripts/zhong_report.py`; interpretation by hand. Set B = 12 new "
         "shapes (clean test set), set A = 7 original shapes. Pencilpot (dev) is in no table and no mean, and nothing "
         "was tuned. Mean ± std over shapes. Protocol v1: camera alignment only (no ICP), normalised frame with GT extent 1. "
         "Outliers = exact distance to the GT triangles; signed offset = median distance along the GT face normal (> 0 = "
         "outside).", "", "<!-- summary -->", "<!-- /summary -->", ""]
    if "1" in args.items:
        L += item1()
    if "2" in args.items:
        fig_regions(["thingi_94668", "thingi_103356", "wheel_76903"])
        L += item2()
    if "3" in args.items:
        fig_size()
        L += item3()
    L += item45()
    text = "\n".join(L) + "\n"
    if Path(args.out).exists():
        old = Path(args.out).read_text()
        for tag in ["summary", "interp1", "interp2", "interp3", "item4", "item5"]:
            m = re.search(f"<!-- {tag} -->.*?<!-- /{tag} -->", old, re.S)
            if m:
                text = re.sub(f"<!-- {tag} -->.*?<!-- /{tag} -->", lambda _: m.group(0), text, flags=re.S)
    Path(args.out).write_text(text)
    print(f"[report] {args.out}")


if __name__ == "__main__":
    main()
