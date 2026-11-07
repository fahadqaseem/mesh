"""Regenerate docs/results/2026-11-08_robustness.md and its figures from outputs/ (Experiments A and B).

Usage: python scripts/robustness_report.py [--out docs/results/2026-11-08_robustness.md]
Called by scripts/overnight.sh after every queue step; safe on partial results (missing runs are left out and the
number of shapes behind each mean is shown). Everything from the line "<!-- manual -->" onwards in the existing doc
is kept as it is (hand-written interpretation).
"""
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import numpy as np

SET_A = ["cylinder_472094", "shackbot_94155", "nsphere_186199", "wheel_76903", "presents_98021",
         "monsterring_1356640", "cryptex_105686"]
SET_B = [f"thingi_{i}" for i in Path("configs/step5_shapes.txt").read_text().split()]
DEV = "pencilpot_299291"
OUT = Path("outputs")
FIG_DIR = Path("docs/results/figures")
MANUAL = "<!-- manual -->"

NOISE = [("noise0.25", 0.25), ("noise0.5", 0.5), ("noise1", 1.0), ("noise2", 2.0)]
OUTL = [("out1", 1.0), ("out5", 5.0), ("out10", 10.0)]
MISS = [("miss10", 10.0), ("miss25", 25.0), ("miss50", 50.0)]
MESH_COLS = [("CD", "Chamfer", 2), ("HD", "Hausdorff", 1), ("NC", "NC", 3), ("F1@0.01", "F1@0.01", 3),
             ("F1@0.02", "F1@0.02", 3), ("V", "#V", 0), ("size_ratio", "size", 2)]
STAT_COLS = [("n_points", "points", 0), ("noise_sigma", "noise σ (% of L)", 2), ("bias", "bias (% of L)", 2),
             ("outlier_frac", "outliers (%)", 1), ("coverage", "coverage (%)", 1)]
PCT = {"noise_sigma", "bias", "outlier_frac", "coverage"}     # stored as fractions of L / fractions; shown in %

# chart tokens (dataviz reference palette, light surface; slots 1-3 validated all-pairs)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


# ----------------------------------------------------------------------------------------------- data
def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def run_data(run_dir: Path) -> dict:
    """Everything known about one run: mesh scores (seed 42), cloud statistics, point scores."""
    return {"mesh": load(run_dir / "mesh" / "meshflow_s42.json"), "stats": load(run_dir / "eval" / "cloud_stats.json"),
            "points": load(run_dir / "eval" / "points_v1.json")}


def rob(shape: str, cond: str) -> dict:
    return run_data(OUT / "robustness" / shape / cond)


def pipe(shape: str, suffix: str) -> dict:
    return run_data(OUT / f"{shape}{suffix}")


def value(d: dict, part: str, key: str) -> float | None:
    v = (d.get(part) or {}).get(key)
    if v is None:
        return None
    return 100.0 * v if (part == "stats" and key in PCT) else float(v)


def collect(runs: list[dict], part: str, key: str) -> np.ndarray:
    vals = [value(r, part, key) for r in runs]
    return np.array([v for v in vals if v is not None], float)


def sd(arr: np.ndarray) -> float:
    """Sample standard deviation (ddof=1), as in the overnight report and the Friday summary."""
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def fmt(arr: np.ndarray, digits: int, n_full: int, spread: bool = True, unit: str = "") -> str:
    if len(arr) == 0:
        return "-"
    m = arr.mean()
    s = f"{m:,.{digits}f}{unit}"
    if spread and len(arr) > 1:
        s += f" ± {sd(arr):,.{digits}f}"
    if len(arr) < n_full:
        s += f" (n={len(arr)})"
    return s


def cells(runs: list[dict], cols: list[tuple], part: str, n_full: int) -> list[str]:
    return [fmt(collect(runs, part, k), d, n_full) for k, _, d in cols]


def line(items: list[str]) -> str:
    return "| " + " | ".join(items) + " |"


def row(label: str, runs: list[dict], cols: list[tuple], part: str, n_full: int, extra: list[str] | None = None) -> str:
    return line([label] + (extra or []) + cells(runs, cols, part, n_full))


def header(first: list[str], cols: list[tuple]) -> str:
    names = first + [c[1] for c in cols]
    return "| " + " | ".join(names) + " |\n|" + "|".join(["---"] + ["---:"] * (len(names) - 1)) + "|"


def queue_progress() -> str:
    q, st = OUT / "robustness_queue" / "queue.tsv", OUT / "robustness_queue" / "status.tsv"
    if not q.exists():
        return ""
    total = sum(1 for _ in q.open())
    rows = [ln.rstrip("\n").split("\t") for ln in st.open()] if st.exists() else []
    last = {}
    for r in rows:
        if len(r) >= 4:
            last[r[2]] = r[3]
    ok = sum(1 for v in last.values() if v == "ok")
    bad = sorted(k for k, v in last.items() if v != "ok")
    s = f"Queue: {ok} of {total} jobs done"
    if bad:
        s += f"; not ok: {', '.join(bad)}"
    return s + f" (as of {rows[-1][0] if rows else '-'})."


# ----------------------------------------------------------------------------------------------- figures
def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.titlecolor": INK, "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED, "legend.frameon": False, "legend.labelcolor": INK2,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round"})
    return plt


def dot(ax, x, y, color, size=6.5, zorder=4, **kw):
    """Marker with a surface-coloured ring (stays legible over lines)."""
    return ax.plot(x, y, "o", ms=size, mfc=color, mec=SURFACE, mew=1.3, zorder=zorder, **kw)


def fig_degradation(path: Path) -> bool:
    """3 x 3 small multiples: (noise, outliers, missing) x (Chamfer, F1@0.02, #V), set A."""
    clean = [pipe(s, "_gtsamples") for s in SET_A]
    if not any(r["mesh"] for r in [rob(s, c) for s in SET_A for c, _ in NOISE]):
        return False
    plt = plot_style()
    cols = [("Gaussian noise", NOISE, "stats", "noise_sigma", lambda v: v, "measured noise σ (% of object size)"),
            ("Outliers", OUTL, "stats", "outlier_frac", lambda v: v, "measured outliers (% of points > 5% off)"),
            ("Missing regions", MISS, "stats", "coverage", lambda v: 100.0 - v, "missing surface (100 − coverage, %)")]
    rows_ = [("CD", "Chamfer (lower is better)"), ("F1@0.02", "F1@0.02 (higher is better)"), ("V", "vertices")]
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.4), sharey="row")
    vggt = [pipe(s, "_tex_8v") for s in SET_A]
    for j, (title, conds, part, key, tx, xlabel) in enumerate(cols):
        for i, (mkey, ylabel) in enumerate(rows_):
            ax = axes[i, j]
            xs_all, ys_all = [], []
            for s, c0 in zip(SET_A, clean):
                pts = [(tx(value(c0, part, key)) if value(c0, part, key) is not None else 0.0, value(c0, "mesh", mkey))]
                for cond, _ in conds:
                    r = rob(s, cond)
                    xv, yv = value(r, part, key), value(r, "mesh", mkey)
                    if xv is not None and yv is not None:
                        pts.append((tx(xv), yv))
                pts = [p for p in pts if p[1] is not None]
                if len(pts) > 1:
                    ax.plot(*zip(*pts), color=MUTED, lw=0.7, alpha=0.45, zorder=1)
            for cond in [None] + [c for c, _ in conds]:
                runs = clean if cond is None else [rob(s, cond) for s in SET_A]
                ok = [r for r in runs if r["mesh"] and (cond is None or r["stats"])]
                if len(ok) < len(SET_A):
                    continue          # mean curve only over complete conditions
                xv = np.array([tx(value(r, part, key)) if r["stats"] else 0.0 for r in ok])
                yv = np.array([value(r, "mesh", mkey) for r in ok])
                xs_all.append(xv.mean())
                ys_all.append((yv.mean(), sd(yv)))
            if xs_all:
                m = np.array([v[0] for v in ys_all])
                spread = np.array([v[1] for v in ys_all])
                ax.fill_between(xs_all, m - spread, m + spread, color=BLUE, alpha=0.10, lw=0, zorder=2)
                ax.plot(xs_all, m, color=BLUE, lw=1.8, zorder=3)
                dot(ax, xs_all, m, BLUE, size=5.5)
            vx = [tx(value(r, part, key)) for r in vggt if r["stats"] and r["mesh"]]
            vy = [value(r, "mesh", mkey) for r in vggt if r["stats"] and r["mesh"]]
            if vx:
                ax.plot(vx, vy, "o", ms=3.5, color=ORANGE, alpha=0.55, mew=0, zorder=4)
                dot(ax, [np.mean(vx)], [np.mean(vy)], ORANGE, size=7.5, zorder=5)
                if i == 0:
                    ax.annotate("VGGT", (np.mean(vx), np.mean(vy)), xytext=(6, 4), textcoords="offset points",
                                color=INK2, fontsize=8)
            if key == "coverage":   # real coverage patterns on clean points (decomposition runs)
                for cond, lab in [("vis8", "rig"), ("vmask", "VGGT pattern")]:
                    runs = [rob(s, cond) for s in SET_A]
                    ok = [r for r in runs if r["stats"] and r["mesh"]]
                    if len(ok) == len(SET_A):
                        x0 = np.mean([tx(value(r, part, key)) for r in ok])
                        y0 = np.mean([value(r, "mesh", mkey) for r in ok])
                        dot(ax, [x0], [y0], AQUA, size=7, zorder=5)
                        if i == 0:
                            ax.annotate(lab, (x0, y0), xytext=(5, -10), textcoords="offset points", color=INK2,
                                        fontsize=8)
            if i == 0:
                ax.set_title(title, loc="left")
            if i == 2:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            ax.set_xlim(left=-0.03 * ax.get_xlim()[1])
    handles = [plt.Line2D([], [], color=BLUE, lw=1.8, marker="o", ms=5, mec=SURFACE,
                          label="MeshFlow on corrupted clean points, set A mean ± std"),
               plt.Line2D([], [], color=MUTED, lw=0.7, alpha=0.6, label="single shapes"),
               plt.Line2D([], [], color=ORANGE, lw=0, marker="o", ms=7, mec=SURFACE,
                          label="real VGGT clouds (small: shapes, large: mean)"),
               plt.Line2D([], [], color=AQUA, lw=0, marker="o", ms=7, mec=SURFACE,
                          label="clean points with a real coverage pattern")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=8, handlelength=2.2)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return True


LADDER = [("clean GT points", lambda s: pipe(s, "_gtsamples")),
          ("visible from the 8 views", lambda s: rob(s, "vis8")),
          ("VGGT coverage pattern", lambda s: rob(s, "vmask")),
          ("+ VGGT noise and bias", lambda s: rob(s, "vmask_noise")),
          ("real VGGT cloud", lambda s: pipe(s, "_tex_8v"))]


def fig_ladder(path: Path) -> bool:
    if not all(f(s)["mesh"] for _, f in LADDER for s in SET_A):
        return False
    plt = plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3))
    x = np.arange(len(LADDER))
    for ax, (mkey, ylabel) in zip(axes, [("CD", "Chamfer (lower is better)"), ("F1@0.02", "F1@0.02 (higher is better)")]):
        per = np.array([[value(f(s), "mesh", mkey) for _, f in LADDER] for s in SET_A])
        for r in per:
            ax.plot(x, r, color=MUTED, lw=0.7, alpha=0.45, zorder=1)
        ax.plot(x, per.mean(0), color=BLUE, lw=1.8, zorder=3)
        dot(ax, x, per.mean(0), BLUE, size=6)
        ax.set_xticks(x, [n for n, _ in LADDER], rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    axes[0].set_title("From clean points to the real VGGT cloud, one corruption at a time (set A mean; gray: shapes)",
                      loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return True


VARIANTS = [("filtered points (current)", "_tex_8v"), ("relaxed points (no multi-view filter)", "_xb_mv0"),
            ("TSDF fusion of the depth maps", "_xb_tsdf"), ("upper bound: clean GT points", "_gtsamples")]


def fig_variants(path: Path) -> bool:
    if not any(pipe(s, "_xb_tsdf")["mesh"] for s in SET_A):
        return False
    plt = plot_style()
    metrics = [("CD", "mesh Chamfer (lower is better)"), ("F1@0.02", "mesh F1@0.02 (higher is better)"),
               ("V", "mesh vertices")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=True)
    y = np.arange(len(VARIANTS))[::-1].astype(float)
    for ax, (mkey, xlabel) in zip(axes, metrics):
        for k, (shapes, color, off) in enumerate([(SET_A, BLUE, 0.13), (SET_B, ORANGE, -0.13)]):
            for yi, (_, suf) in zip(y, VARIANTS):
                vals = collect([pipe(s, suf) for s in shapes], "mesh", mkey)
                if len(vals) == 0:
                    continue
                ax.plot([vals.mean() - sd(vals), vals.mean() + sd(vals)], [yi + off] * 2, color=color, lw=1.2,
                        alpha=0.55, zorder=2)
                dot(ax, [vals.mean()], [yi + off], color, size=6.5)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [n for n, _ in VARIANTS])
    handles = [plt.Line2D([], [], color=BLUE, lw=1.2, marker="o", ms=6.5, mec=SURFACE, label="set A (7 shapes)"),
               plt.Line2D([], [], color=ORANGE, lw=1.2, marker="o", ms=6.5, mec=SURFACE,
                          label="set B (12 shapes, clean test set)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.6, 1.0), fontsize=8)
    fig.text(0.01, 0.97, "Mean ± std over shapes, MeshFlow seed 42", color=INK2, fontsize=8, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------------------------- tables
def section_a(lines: list[str], figs: dict) -> None:
    n = len(SET_A)
    vggt = [pipe(s, "_tex_8v") for s in SET_A]
    clean = [pipe(s, "_gtsamples") for s in SET_A]
    lines += ["## A1. What the real VGGT clouds look like (set A)", "",
              "The same statistics for every cloud (`src/photo2mesh/eval/robustness.py`), in % of the object size L. "
              "Visible = share of the GT surface the 8 cameras can see at all (from the vis8 run).", "",
              header(["shape"], STAT_COLS + [("vis", "visible from rig (%)", 1)])]
    vis_all = []
    for s, r in zip(SET_A, vggt):
        vis = (load(OUT / "robustness" / s / "vis8" / "corrupt.json") or {}).get("visible", {}).get("kept")
        if vis is not None:
            vis_all.append(100 * vis)
        lines.append(line([s] + cells([r], STAT_COLS, "stats", 1) + [f"{100 * vis:.1f}" if vis is not None else "-"]))
    lines.append(line(["**mean ± std**"] + cells(vggt, STAT_COLS, "stats", n) + [fmt(np.array(vis_all), 1, n)]))
    lines += ["", "## A2. MeshFlow on corrupted clean points (set A)", "",
              "Corruptions of the 200k clean GT samples (`scripts/corrupt_points.py`); MeshFlow seed 42; mean ± std "
              "over the 7 shapes. Nominal = what was applied; measured = the statistic of the corrupted cloud "
              "(uniform outliers that land within 5% of the surface do not count as outliers).", "",
              header(["input", "nominal", "measured"], MESH_COLS)]
    lines.append(row("clean GT points (upper bound)", clean, MESH_COLS, "mesh", n, ["-", "-"]))
    for group, conds, key, unit in [("noise σ", NOISE, "noise_sigma", "%"), ("outliers", OUTL, "outlier_frac", "%"),
                                    ("missing", MISS, "coverage", "%")]:
        for cond, nom in conds:
            runs = [rob(s, cond) for s in SET_A]
            meas = collect(runs, "stats", key)
            if key == "coverage":
                meas = 100.0 - meas
            lines.append(row(f"{group} {nom:g}{unit}", runs, MESH_COLS, "mesh", n,
                             [f"{nom:g}%", fmt(meas, 2, n, spread=False, unit="%")]))
    lines.append(row("**real VGGT cloud (pipeline)**", vggt, MESH_COLS, "mesh", n, ["-", "see A1"]))
    lines.append("")
    if figs.get("degradation"):
        lines += [f"![Degradation curves]({figs['degradation']})", ""]
    lad = [[f(s) for s in SET_A] for _, f in LADDER]
    if any(r["mesh"] for r in lad[1]):
        lines += ["## A3. Which corruption explains the gap (set A)", "",
                  "One corruption at a time, from clean points to the real cloud. Rig = clean points the 8 render "
                  "cameras can see. VGGT coverage pattern = clean points where the VGGT cloud has points (same test as "
                  "the coverage statistic). Then that pattern plus the VGGT cloud's own measured noise and bias. "
                  "Share of gap = this step's Chamfer change / (real VGGT − clean).", "",
                  header(["stage", "share of gap (Chamfer)"], MESH_COLS + [("coverage", "coverage (%)", 1)])]
        cd = [collect(r, "mesh", "CD") for r in lad]
        gap = cd[-1].mean() - cd[0].mean() if len(cd[-1]) and len(cd[0]) else np.nan
        for k, ((name, _), runs) in enumerate(zip(LADDER, lad)):
            share = "-" if k == 0 or not len(cd[k]) or not len(cd[k - 1]) else \
                f"{100 * (cd[k].mean() - cd[k - 1].mean()) / gap:+.0f}%"
            cov = collect(runs, "stats", "coverage")
            lines.append(line([name, share] + cells(runs, MESH_COLS, "mesh", n) + [fmt(cov, 1, n)]))
        lines.append("")
        if figs.get("ladder"):
            lines += [f"![Decomposition]({figs['ladder']})", ""]


def section_correlations(lines: list[str]) -> None:
    """Spearman rank correlation, across the real VGGT clouds of sets A and B, between each statistic and mesh quality."""
    from scipy.stats import spearmanr

    runs = [pipe(s, "_tex_8v") for s in SET_A + SET_B]
    runs = [r for r in runs if r["stats"] and r["mesh"]]
    if len(runs) < 10:
        return
    lines += ["## A4. Across the real clouds (sets A and B)", "",
              f"Spearman rank correlation over the {len(runs)} real VGGT clouds (one per shape, filtered points, seed 42). "
              "A statistic that explains mesh quality correlates positively with Chamfer and negatively with F1.", "",
              "| statistic | range over the clouds | ρ with Chamfer | ρ with F1@0.02 | ρ with #V |", "|---|---:|---:|---:|---:|"]
    stats = [("noise σ (% of L)", "noise_sigma", 1.0), ("bias (% of L)", "bias", 1.0),
             ("outliers (%)", "outlier_frac", 1.0), ("missing surface (100 − coverage, %)", "coverage", -1.0)]
    for label, key, sign in stats:
        x = np.array([value(r, "stats", key) for r in runs]) * sign + (100.0 if sign < 0 else 0.0)
        cols = []
        for mkey in ("CD", "F1@0.02", "V"):
            y = np.array([value(r, "mesh", mkey) for r in runs])
            rho, pval = spearmanr(x, y)
            cols.append(f"{rho:+.2f} (p={pval:.2g})")
        lines.append(line([label, f"{x.min():.2f} to {x.max():.2f}"] + cols))
    lines.append("")


def paired(shapes: list[str], a: str, b: str, key: str) -> str:
    d = []
    for s in shapes:
        x, y = value(pipe(s, a), "mesh", key), value(pipe(s, b), "mesh", key)
        if x is not None and y is not None:
            d.append(y - x)
    if not d:
        return "-"
    d = np.array(d)
    better = (d < 0).sum() if key in ("CD", "HD") else (d > 0).sum()
    return f"{d.mean():+.3g} ± {sd(d):.2g} (better on {better} of {len(d)})"


def section_b(lines: list[str], figs: dict) -> None:
    if not any(pipe(s, "_xb_tsdf")["points"] for s in SET_A + SET_B):
        return
    pcols = [("CD", "points Chamfer", 2), ("F1@0.02", "points F1@0.02", 3)]
    lines += ["## B1. Better points without a GPU: TSDF fusion (sets A and B)", "",
              "One VGGT pass per shape (8 views, known focal). Filtered = the current pipeline (silhouette erosion + "
              "multi-view consistency); its rerun is byte-identical to last night's cloud, so last night's meshes are "
              "the baseline. Relaxed = the same without the multi-view filter (control). TSDF = every eroded "
              "foreground pixel fused in a TSDF volume (Open3D, CPU; voxel 0.5% of the object, truncation 6 voxels, "
              "surface wherever at least one view saw it; settings chosen on pencilpot only), then 200k points "
              "sampled from the fused surface. MeshFlow seed 42.", ""]
    for name, shapes in [("Set B (12 shapes, clean test set)", SET_B), ("Set A (7 shapes)", SET_A)]:
        n = len(shapes)
        lines += [f"**{name}: points**", "", header(["input"], STAT_COLS[1:] + pcols)]
        for label, suf in VARIANTS[:3]:
            runs = [pipe(s, suf) for s in shapes]
            lines.append(line([label] + cells(runs, STAT_COLS[1:], "stats", n) + cells(runs, pcols, "points", n)))
        lines += ["", f"**{name}: meshes**", "", header(["input"], MESH_COLS)]
        for label, suf in VARIANTS:
            lines.append(row(label, [pipe(s, suf) for s in shapes], MESH_COLS, "mesh", n))
        lines += ["", f"Paired per shape, TSDF minus filtered: Chamfer {paired(shapes, '_tex_8v', '_xb_tsdf', 'CD')}; "
                      f"F1@0.02 {paired(shapes, '_tex_8v', '_xb_tsdf', 'F1@0.02')}. Relaxed minus filtered: Chamfer "
                      f"{paired(shapes, '_tex_8v', '_xb_mv0', 'CD')}. TSDF minus relaxed: Chamfer "
                      f"{paired(shapes, '_xb_mv0', '_xb_tsdf', 'CD')}.", ""]
    if figs.get("variants"):
        lines += [f"![Points variants]({figs['variants']})", ""]
    if any(pipe(s, "_xb_ph")["mesh"] for s in SET_A):
        n = len(SET_A)
        lines += ["## B2. VGGT point head vs depth unprojection (set A, VGGT's own focal)", "",
                  "Same VGGT pass, same filters. Depth unprojection with VGGT's own focal is last night's "
                  "uncalibrated variant (rerun byte-identical). The known focal applies to depth unprojection only.", "",
                  "**Points**", "", header(["input"], STAT_COLS[1:] + pcols)]
        for label, suf in [("depth unprojection (VGGT focal)", "_tex_8v_vf"), ("point head", "_xb_ph"),
                           ("depth unprojection (known focal, headline)", "_tex_8v")]:
            runs = [pipe(s, suf) for s in SET_A]
            lines.append(line([label] + cells(runs, STAT_COLS[1:], "stats", n) + cells(runs, pcols, "points", n)))
        lines += ["", "**Meshes**", "", header(["input"], MESH_COLS)]
        for label, suf in [("depth unprojection (VGGT focal)", "_tex_8v_vf"), ("point head", "_xb_ph"),
                           ("depth unprojection (known focal, headline)", "_tex_8v")]:
            lines.append(row(label, [pipe(s, suf) for s in SET_A], MESH_COLS, "mesh", n))
        lines += ["", f"Paired per shape, point head minus depth unprojection (VGGT focal): Chamfer "
                      f"{paired(SET_A, '_tex_8v_vf', '_xb_ph', 'CD')}; F1@0.02 "
                      f"{paired(SET_A, '_tex_8v_vf', '_xb_ph', 'F1@0.02')}.", ""]


def section_dev_and_shapes(lines: list[str]) -> None:
    lines += ["## Dev shape (pencilpot): shown, never in a mean", "",
              "Pencilpot tuned the point filters (earlier) and the TSDF settings (this block).", "",
              header(["input"], MESH_COLS + [("coverage", "coverage (%)", 1)])]
    for label, suf in VARIANTS + [("point head (VGGT focal)", "_xb_ph"), ("depth unprojection (VGGT focal)", "_tex_8v_vf")]:
        r = pipe(DEV, suf)
        if r["mesh"]:
            lines.append(line([label] + cells([r], MESH_COLS, "mesh", 1) + [fmt(collect([r], "stats", "coverage"), 1, 1)]))
    lines += ["", "## Per shape (Chamfer / F1@0.02 of the MeshFlow mesh)", ""]
    conds = [("clean", lambda s: pipe(s, "_gtsamples"))] + \
            [(c, (lambda c: lambda s: rob(s, c))(c)) for c, _ in NOISE + OUTL + MISS] + \
            [(c, (lambda c: lambda s: rob(s, c))(c)) for c in ("vis8", "vmask", "vmask_noise")] + \
            [("VGGT", lambda s: pipe(s, "_tex_8v")), ("relaxed", lambda s: pipe(s, "_xb_mv0")),
             ("TSDF", lambda s: pipe(s, "_xb_tsdf")), ("point head", lambda s: pipe(s, "_xb_ph"))]
    lines += ["| shape | " + " | ".join(c for c, _ in conds) + " |", "|---|" + "---:|" * len(conds)]
    for s in SET_A + SET_B:
        items = []
        for _, f in conds:
            m = f(s)["mesh"]
            items.append(f"{m['CD']:.2f} / {m['F1@0.02']:.2f}" if m else "")
        if any(items):
            lines.append(f"| {s}{' (B)' if s in SET_B else ''} | " + " | ".join(items) + " |")
    lines.append("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results/2026-11-08_robustness.md")
    args = ap.parse_args()
    out = Path(args.out)
    manual = ""
    if out.exists() and MANUAL in out.read_text():
        manual = out.read_text()[out.read_text().index(MANUAL):]
    figs = {}
    for key, fn in [("degradation", fig_degradation), ("ladder", fig_ladder), ("variants", fig_variants)]:
        p = FIG_DIR / f"2026-11-08_{key}.png"
        if fn(p):
            figs[key] = os.path.relpath(p, out.parent)
    lines = ["# MeshFlow robustness and better points without a GPU (Experiments A and B)", "",
             "Fahad Qaseem Khawar, 7 to 8 Nov 2026. Tables and figures are regenerated by "
             "`scripts/robustness_report.py` after every queue step; the interpretation at the end is written by hand.",
             "", queue_progress(), "",
             "Protocol v1 (`docs/eval_protocol.md`): MeshFlow's Chamfer and Hausdorff (x100, GT scaled to [-0.5, 0.5]), "
             "normal consistency, F1 at 0.01 and 0.02 of the object size, vertex count, size = mesh extent / GT extent. "
             "Clouds reach the GT frame only through the render cameras. Set A = 7 original shapes, set B = 12 new "
             "shapes (clean test set); pencilpot is the dev shape and is in no mean.", ""]
    section_a(lines, figs)
    section_correlations(lines)
    section_b(lines, figs)
    section_dev_and_shapes(lines)
    text = "\n".join(lines) + "\n" + (manual or f"{MANUAL}\n## Interpretation\n\n(written when the queue has finished)\n")
    out.write_text(text)
    print(f"[report] {out} ({len(text.splitlines())} lines; figures: {', '.join(figs) or 'none yet'})")


if __name__ == "__main__":
    main()
