"""Regenerate docs/results/2026-11-18_outliers.md (Task C) and its figure from scripts/gapfill_outliers.py outputs.

Usage: python scripts/outliers_report.py [--out docs/results/2026-11-18_outliers.md]
Everything from "<!-- manual -->" onwards in the existing doc is kept.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent))
from make_points_queue import DEV, SET_A, SET_B  # noqa: E402

MANUAL = "<!-- manual -->"
OUT = Path("outputs")
COLS = [("added", "points added", 0), ("CD", "Chamfer", 2), ("F1@0.01", "F1@0.01", 3), ("F1@0.02", "F1@0.02", 3),
        ("NC", "NC", 3), ("R_all@0.02", "recall all @0.02", 3), ("out@0.01", "outliers > 0.01 (%)", 2),
        ("out@0.02", "outliers > 0.02 (%)", 2), ("added_prec@0.02", "added points within 0.02 (%)", 1)]
FEATS = [("opacity", "source opacity", +1), ("max_scale", "source largest axis (x object size)", -1),
         ("flatness", "source flatness (min / max axis)", -1), ("sigma", "position in the kernel (max |z|, sigma)", -1),
         ("d_tsdf", "distance to nearest TSDF point (x extent)", -1), ("n_vis", "views seeing it on the object", +1),
         ("sil", "views seeing it on background", -1), ("front_max", "max in-front-of-depth margin (x size)", -1),
         ("abs_tsdf", "|projective TSDF| (x trunc)", -1), ("n_obs", "views with fused depth there", +1)]
FIG = Path("docs/results/figures/2026-11-18_outliers.png")


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def gap_points(shapes: list[str]) -> dict:
    out = {}
    for s in shapes:
        f = OUT / f"{s}_dfill" / "eval" / "gap_points_chosen.npz"
        if f.exists():
            z = np.load(f, allow_pickle=False)
            d = {k: z[k] for k in z.files}
            d["abs_tsdf"] = np.abs(d["tsdf"])
            out[s] = d
    return out


def diagnosis(L: list[str]) -> None:
    gp = gap_points(SET_A + SET_B)
    if not gp:
        return
    allp = {k: np.concatenate([g[k] for g in gp.values()]) for k, _, _ in FEATS + [("gt_dist", "", 0)]}
    bad = allp["gt_dist"] > 0.02
    n = len(bad)
    L += ["## 1. Where the outliers come from", "",
          f"All gap points of the dense gap fill on set A and set B pooled: {n:,} points, of which {100 * bad.mean():.1f}% lie more "
          "than 0.02 from the GT surface (GT used here only to label them). For each property: the outlier rate in its four "
          "quartile bins (lowest to highest value), and the AUC for telling outliers apart (0.5 = no information; listed so "
          "that above 0.5 means the property separates them in the expected direction, given in the last column). Points "
          "without a projective TSDF value (no view has fused depth there) are left out of that row.", "",
          "| property | Q1 | Q2 | Q3 | Q4 | AUC | outliers are ... |", "|---|---:|---:|---:|---:|---:|---|"]
    for k, label, sign in FEATS:
        v = allp[k].astype(float)
        ok = np.isfinite(v)
        vv, bb = v[ok], bad[ok]
        if len(np.unique(vv)) < 2:
            continue
        qs = np.quantile(vv, [0.25, 0.5, 0.75])
        bins = np.digitize(vv, qs)
        rates = [f"{100 * bb[bins == i].mean():.0f}%" if (bins == i).any() else "-" for i in range(4)]
        auc = mannwhitneyu(vv[bb], vv[~bb]).statistic / (bb.sum() * (~bb).sum()) if bb.any() and (~bb).any() else np.nan
        direction = "higher" if auc >= 0.5 else "lower"
        L.append(f"| {label} | " + " | ".join(rates) + f" | {max(auc, 1 - auc):.2f} | {direction} |")
    L.append("")
    per = [(s, 100 * (g['gt_dist'] > 0.02).mean(), len(g["gt_dist"])) for s, g in gp.items()]
    L += ["Outlier rate of the gap points per shape (> 0.02): " +
          ", ".join(f"{s.replace('thingi_', '')} {r:.0f}% of {n:,}" for s, r, n in per) + ".", ""]


def table(rows: dict, names: list[str], shapes: list[str], ref: str) -> list[str]:
    L = ["| filter | " + " | ".join(c[1] for c in COLS) + " |", "|---|" + "---:|" * len(COLS)]
    for nm in names:
        vals = [r[nm] for s, r in rows.items() if s in shapes and nm in r]
        if not vals:
            continue
        cells = []
        for k, _, d in COLS:
            a = np.array([v[k] for v in vals], float)
            a = a[np.isfinite(a)]
            cells.append(f"{a.mean():,.{d}f}" if len(a) else "-")
        L.append(f"| {nm} | " + " | ".join(cells) + " |")
    return L


def paired(rows: dict, shapes: list[str], a: str, b: str, key: str) -> str:
    d = np.array([rows[s][b][key] - rows[s][a][key] for s in shapes if s in rows])
    if not len(d):
        return "-"
    better = (d < 0).sum() if key in ("CD", "HD") or key.startswith("out") else (d > 0).sum()
    return f"{d.mean():+.2f} (better on {better} of {len(d)})"


def figure(rows_names: list[str], best: str) -> bool:
    shapes = ["thingi_103356", "thingi_77949", "thingi_593381"]
    if not all((OUT / f"{s}_dfill" / "eval" / "gap_points_chosen.npz").exists() for s in shapes):
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh

    from fig_gapfill import view_matrix
    from photo2mesh.io import load_points_ply
    V = view_matrix()
    fig, axes = plt.subplots(len(shapes), 2, figsize=(8.0, 4.1 * len(shapes)), gridspec_kw={"hspace": 0.35})
    rng = np.random.default_rng(0)
    for r, s in enumerate(shapes):
        z = np.load(OUT / f"{s}_dfill" / "eval" / "gap_points_chosen.npz")
        names = list(z["filter_names"])
        base = load_points_ply(OUT / f"{s}_dense" / "align" / "points_aligned.ply")["points"]
        gt = trimesh.load(f"data/scenes/{s}/gt_normalized.ply", force="mesh")
        ext = float(np.ptp(gt.bounds, axis=0).max())
        bsub = base[rng.choice(len(base), 40_000, replace=False)] @ V.T
        x = z["x_gt"] @ V.T
        bad = z["gt_dist"] > 0.02
        keeps = [np.ones(len(bad), bool), z[f"keep_{names.index(best)}"]]
        titles = ["gap fill (current)", f"gap fill + {best}"]
        for c, (keep, title) in enumerate(zip(keeps, titles)):
            ax = axes[r, c]
            ax.scatter(bsub[:, 0], bsub[:, 1], s=0.1, c="#c9c9c4", linewidths=0, rasterized=True)
            good = keep & ~bad
            out = keep & bad
            ax.scatter(x[good, 0], x[good, 1], s=0.6, c="#1a9e8f", linewidths=0, rasterized=True)
            ax.scatter(x[out, 0], x[out, 1], s=0.9, c="#d6402b", linewidths=0, rasterized=True)
            ax.set_title(f"{title}\n{keep.sum():,} gap points · {out.sum():,} outliers "
                         f"({100 * out.sum() / max(keep.sum(), 1):.0f}%)", fontsize=9)
            q = gt.vertices @ V.T
            pad = 0.06 * ext
            ax.set_xlim(q[:, 0].min() - pad, q[:, 0].max() + pad)
            ax.set_ylim(q[:, 1].min() - pad, q[:, 1].max() + pad)
            ax.set_aspect("equal")
            ax.axis("off")
        axes[r, 0].text(-0.08, 0.5, f"Test shape {s.split('_')[1]}", transform=axes[r, 0].transAxes, rotation=90,
                        ha="center", va="center", fontsize=9, fontweight="bold")
    fig.suptitle("Gap-fill points: red = more than 2% of the object's size from the true surface, teal = within 2%;\n"
                 "grey = TSDF points (subsample). Same view as 2026-11-15_gapfill_compare.png.", fontsize=10)
    fig.savefig(FIG, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results/2026-11-18_outliers.md")
    ap.add_argument("--best", default=None, help="filter name shown in the figure")
    args = ap.parse_args()
    out = Path(args.out)
    old = out.read_text() if out.exists() else ""
    manual = old[old.index(MANUAL):] if MANUAL in old else f"{MANUAL}\n## Interpretation\n\n(written when the runs are done)\n"
    L = ["# Gap-fill outliers: where they come from and principled filters", "",
         "Fahad Qaseem Khawar, 18 Nov 2026. Tables regenerated by `scripts/outliers_report.py` from "
         "`scripts/gapfill_outliers.py`; the interpretation at the end is written by hand. Base: the dense default (1M TSDF "
         "points) plus the gap points of the dense gap fill (1M kernel samples, kept when 3-10% of the extent from the nearest "
         "TSDF point; `docs/methods.md` sections 5-6). **No filter uses the GT**; the GT only labels and scores. Filter "
         "settings were chosen on pencilpot only (section 2). Outliers = exact distance to the GT surface above 0.01 / 0.02 "
         "in the normalised frame (GT extent 1), as in `docs/results/2026-11-17_equal_count.md`. Clouds are scored at their own "
         "size (1M TSDF points + the kept gap points).", ""]
    diagnosis(L)
    pilot = load(OUT / f"{DEV}_dfill" / "eval" / "outliers_pilot.json")
    if pilot:
        L += ["## 2. Pilot on pencilpot (dev): all filter settings", "",
              "(a) redraws the kernels at a tighter truncation (the current one is already 2 sigma); (b) source-Gaussian "
              "opacity / size; (c) multi-view free space and silhouette (m = margin in object sizes, k = views that must "
              "object); (d) projective TSDF (tau in truncation units). Carried to sets A and B "
              "(`configs/outlier_filters_chosen.json`): per family, the setting with the lowest share of points beyond 0.02 "
              "among those whose Chamfer stays below the TSDF-only cloud's (the fill must still help); for (b) also opacity "
              ">= 0.9, to answer the opacity question directly. Chamfer differences of ~0.1 between near-identical clouds "
              "are sampling noise of the 5,000-point protocol (e.g. (c) m 0.01 vs 0.02 keep 70,656 vs 70,687 points but "
              "score 5.86 vs 5.93).", ""] + table({DEV: pilot}, list(pilot), [DEV], "TSDF only") + [""]
    rows = {s: r for s in [DEV] + SET_A + SET_B if (r := load(OUT / f"{s}_dfill" / "eval" / "outliers_chosen.json"))}
    if any(s in rows for s in SET_A + SET_B):
        names = list(next(iter(rows.values())))
        L += ["## 3. Sets A and B with the chosen settings", ""]
        for title, shapes in [("Set B (12 shapes, clean test set)", SET_B), ("Set A (7 shapes)", SET_A)]:
            L += [f"**{title}** (mean over shapes)", ""] + table(rows, names, shapes, "TSDF only") + [""]
            L += ["Paired per shape against the current gap fill: " + "; ".join(
                f"{nm}: outliers > 0.02 {paired(rows, shapes, 'gap fill (current, 2 sigma)', nm, 'out@0.02')}, "
                f"Chamfer {paired(rows, shapes, 'gap fill (current, 2 sigma)', nm, 'CD')}"
                for nm in names[2:]) + ".", ""]
        best = args.best or names[2]
        if figure(names, best):
            L += [f"## 4. Picture", "", f"![gap-fill outliers](figures/{FIG.name})", "",
                  f"Left: all gap points of the current gap fill; right: those kept by {best}. Same view and zoom as "
                  "`figures/2026-11-15_gapfill_compare.png` (the meeting picture).", ""]
    out.write_text("\n".join(L) + "\n" + manual)
    print(f"[report] {out}")


if __name__ == "__main__":
    main()
