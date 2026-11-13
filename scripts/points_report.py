"""Regenerate docs/results/2026-11-12_points.md: the point cloud as the main output (Prof. Zhong, 9 Nov).

Usage: python scripts/points_report.py [--out docs/results/2026-11-12_points.md]
Safe on partial results (the number of shapes behind a mean is shown when some are missing). Everything from
"<!-- manual -->" onwards in the existing doc is kept (hand-written interpretation).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from robustness_report import (DEV, MANUAL, MESH_COLS, OUT, SET_A, SET_B, cells, collect, fmt, header, line, load,
                               pipe, sd)

PCOLS = [("CD", "Chamfer", 2), ("HD", "Hausdorff", 1), ("F1@0.005", "F1@0.005", 3), ("F1@0.01", "F1@0.01", 3),
         ("F1@0.02", "F1@0.02", 3), ("NC", "NC", 3)]
ALLCOLS = [("R_all@0.005", "recall, all points @0.005", 3), ("R_all@0.01", "recall, all points @0.01", 3)]
COV = [("coverage", "coverage (%)", 1)]
NPTS = [("V", "#points", 0)]
INPUTS = [("filtered points (v2, previous default)", "_tex_8v"), ("TSDF points (v3, default)", "_tex_v3"),
          ("Gaussian centres", "_gs"), ("Gaussian kernels, opacity-weighted", "_gsk"),
          ("Gaussian kernels, opacity x area", "_gska")]
SWEEP = [("0.005 (current)", "200k", "_tex_v3", "_td005"), ("0.005 (current)", "1M", "_td005n1m", "_td005n1m"),
         ("0.0025 (2x finer)", "200k", "_td0025", "_td0025"), ("0.0025 (2x finer)", "1M", "_td0025n1m", "_td0025n1m"),
         ("0.00125 (4x finer)", "200k", "_td00125", "_td00125"),
         ("0.00125 (4x finer)", "1M", "_td00125n1m", "_td00125n1m")]
MESH_INPUTS = [("TSDF points (v3, default)", "_tex_v3"), ("TSDF, voxel 0.0025", "_td0025"),
               ("TSDF, voxel 0.00125", "_td00125"), ("Gaussian kernels, opacity-weighted", "_gsk"),
               ("Gaussian centres", "_gs"), ("filtered points (v2)", "_tex_8v"),
               ("upper bound: clean GT points", "_gtsamples")]
SETS = [("Set B (12 shapes, clean test set)", SET_B), ("Set A (7 shapes)", SET_A)]
FIG = Path("docs/results/figures/2026-11-12_density.png")


def tsdf_info(shape: str, info_suffix: str) -> dict | None:
    return load(OUT / f"{shape}{info_suffix}" / "points" / "tsdf.json")


def runtime_cells(shapes: list[str], info_suffix: str, n_full: int) -> list[str]:
    """fusion + sampling seconds and peak memory of the fusion process (mean ± std over shapes)."""
    infos = [tsdf_info(s, info_suffix) for s in shapes]
    infos = [i for i in infos if i and "fuse_s" in i]
    t = np.array([i["fuse_s"] + i["sample_s"] for i in infos])
    m = np.array([i["peak_rss_gb"] for i in infos if "peak_rss_gb" in i])
    return [fmt(t, 1, n_full), fmt(np.array([m.max()]) if len(m) else m, 1, 1, spread=False)]


def vggt_seconds(shapes: list[str]) -> np.ndarray:
    v = [(tsdf_info(s, "_tdraw") or {}).get("vggt_s") for s in shapes]
    return np.array([x for x in v if x is not None])


def queue_line() -> str:
    q, st = OUT / "points_queue" / "queue.tsv", OUT / "points_queue" / "status.tsv"
    if not q.exists():
        return ""
    total = sum(1 for _ in q.open())
    last = {}
    for r in (ln.rstrip("\n").split("\t") for ln in st.open()) if st.exists() else []:
        if len(r) >= 4:
            last[r[2]] = r[3]
    bad = sorted(k for k, v in last.items() if v != "ok")
    return f"Queue: {sum(v == 'ok' for v in last.values())} of {total} jobs ok" + \
        (f"; not ok: {', '.join(bad)}" if bad else "") + "."


def paired(shapes: list[str], a: str, b: str, key: str) -> str:
    """b minus a per shape on a points metric, with how many shapes b is better on."""
    d = []
    for s in shapes:
        va, vb = (pipe(s, a)["points"] or {}).get(key), (pipe(s, b)["points"] or {}).get(key)
        if va is not None and vb is not None:
            d.append(vb - va)
    if not d:
        return "-"
    d = np.array(d)
    better = (d < 0).sum() if key in ("CD", "HD") else (d > 0).sum()
    return f"{d.mean():+.3g} ± {sd(d):.2g} (better on {better} of {len(d)})"


def figure(path: Path) -> bool:
    """Set B: protocol F1@0.01 and all-points recall vs number of points, and F1@0.01 vs runtime."""
    if not any(pipe(s, "_td0025")["points"] for s in SET_B):
        return False
    from robustness_report import INK2, MUTED, SURFACE, dot, plot_style
    plt = plot_style()
    voxel_colours = {"0.005": "#86b6ef", "0.0025": "#2a78d6", "0.00125": "#0d3f80"}    # one hue, light -> dark
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    panels = [("F1@0.01", "points F1@0.01 (protocol, 100k subsample)"),
              ("R_all@0.01", "recall @0.01 with all points")]
    for ax, (key, ylabel) in zip(axes[:2], panels):
        for vox, c in voxel_colours.items():
            xs, ys = [], []
            for label, npts, suf, _ in SWEEP:
                if label.startswith(vox + " "):
                    v = collect([pipe(s, suf) for s in SET_B], "points", key)
                    if len(v):
                        xs.append(2e5 if npts == "200k" else 1e6)
                        ys.append(v.mean())
            if xs:
                ax.plot(xs, ys, color=c, lw=2, zorder=2)
                dot(ax, xs, ys, c, size=7)
                ax.annotate(f"voxel {vox}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=7.5, color=INK2)
        for suf, mk, name in [("_gsk", "D", "Gaussian kernels"), ("_gs", "s", "Gaussian centres")]:
            g = collect([pipe(s, suf) for s in SET_B], "points", key)
            n = collect([pipe(s, suf) for s in SET_B], "points", "V")
            if len(g):
                ax.scatter([n.mean()], [g.mean()], marker=mk, s=40, color=MUTED, edgecolor=SURFACE, lw=1.5, zorder=3)
                ax.annotate(name, (n.mean(), g.mean()), xytext=(6, -2), textcoords="offset points", fontsize=7.5,
                            color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(8e3, 3e6)
        ax.set_xlabel("points in the cloud")
        ax.set_ylabel(ylabel)
    ax = axes[2]
    for vox, c in voxel_colours.items():
        for label, npts, suf, info in SWEEP:
            if label.startswith(vox + " ") and npts == "200k":
                infos = [tsdf_info(s, info) for s in SET_B]
                t = np.array([i["fuse_s"] + i["sample_s"] for i in infos if i and "fuse_s" in i])
                f1 = collect([pipe(s, suf) for s in SET_B], "points", "F1@0.01")
                if len(t) and len(f1):
                    dot(ax, [t.mean()], [f1.mean()], c, size=7)
                    ax.annotate(f"voxel {vox}", (t.mean(), f1.mean()), xytext=(6, 0), textcoords="offset points",
                                va="center", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("fusion + sampling time per object (s, CPU)")
    ax.set_ylabel("points F1@0.01 (200k points)")
    fig.text(0.01, 0.98, "Set B (12 test shapes), mean over shapes. VGGT (same for all) not included in the time.",
             color=INK2, fontsize=8, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results/2026-11-12_points.md")
    out = Path(ap.parse_args().out)
    text_old = out.read_text() if out.exists() else ""
    manual = text_old[text_old.index(MANUAL):] if MANUAL in text_old else ""
    L = ["# The point cloud as the main output: TSDF density sweep and Gaussian kernel sampling", "",
         "Fahad Qaseem Khawar, 12 Nov 2026. Direction from Prof. Zhong (9 Nov): points side only; dense, accurate, "
         "high-fidelity clouds; lightweight is not a priority yet. Tables and the figure are regenerated by "
         "`scripts/points_report.py`; the interpretation at the end is written by hand. Set A = 7 original shapes, "
         "set B = 12 new shapes (clean test set); pencilpot is the dev shape (all settings were chosen on it) and is "
         "in no mean. Mean ± std over shapes.", "", queue_line(), "",
         "**How the points are scored** (protocol v1, `docs/eval_protocol.md`, MeshFlow's frame: GT bbox centre, max "
         "extent 1, i.e. [-0.5, 0.5]): cloud aligned by the camera similarity only (no ICP). Chamfer and Hausdorff "
         "x100 on 5,000 random points of the cloud vs 5,000 GT surface samples (MeshFlow protocol). F1 at 0.005, 0.01 "
         "and 0.02 and NC on 100,000 random points vs 100,000 GT samples. Because both use random subsamples, a "
         "denser cloud of the same surface scores the same there; the **all-points recall** (share of the 100k GT "
         "samples within tau of any point of the cloud) is what extra density can buy. **Coverage**: share of the "
         "GT surface within 1% of the footprint of the cloud's inliers (points within 5% of the surface, projected "
         "onto it), as in the robustness report. F1@0.005 and the all-points recall are new (added 12 Nov).", ""]

    L += ["## 1. Points first: every points stage scored against the GT", ""]
    for name, shapes in SETS:
        n = len(shapes)
        L += [f"**{name}**", "", header(["points stage"], PCOLS + COV + NPTS)]
        for label, suf in INPUTS:
            runs = [pipe(s, suf) for s in shapes]
            L.append(line([label] + cells(runs, PCOLS, "points", n) + cells(runs, COV, "stats", n) +
                          cells(runs, NPTS, "points", n)))
        L += ["", f"Paired per shape: Gaussian kernels minus TSDF, F1@0.01 {paired(shapes, '_tex_v3', '_gsk', 'F1@0.01')}, "
                  f"Chamfer {paired(shapes, '_tex_v3', '_gsk', 'CD')}; Gaussian kernels minus Gaussian centres, "
                  f"F1@0.01 {paired(shapes, '_gs', '_gsk', 'F1@0.01')}.", ""]

    L += ["## 2. TSDF density sweep", "",
          "Same VGGT pass (8 views, known focal) per shape; the TSDF voxel is a fraction of the object size (bbox of "
          "the fused pixels, 1st to 99th percentile), truncation 6 voxels, and n points are sampled from the fused "
          "surface (area-uniform). 0.005 / 200k is the v3 default (the sweep reproduces it byte for byte, job "
          "`same_as_v3`). Time = fusion + small-piece removal + sampling, CPU, per object; peak = largest resident "
          "memory of the fusion process over the shapes.", ""]
    for name, shapes in SETS:
        n = len(shapes)
        L += [f"**{name}**", "", header(["voxel (x object size)", "points"], PCOLS + ALLCOLS + COV +
                                        [("t", "time (s)", 1), ("m", "peak (GB)", 1)])]
        for label, npts, suf, info in SWEEP:
            runs = [pipe(s, suf) for s in shapes]
            L.append(line([label, npts] + cells(runs, PCOLS + ALLCOLS, "points", n) + cells(runs, COV, "stats", n) +
                          runtime_cells(shapes, info, n)))
        v = vggt_seconds(shapes)
        L += ["", f"VGGT itself (MPS, 8 views): {fmt(v, 0, n)} s per object, the same for every row." if len(v) else "",
              ""]
    if figure(FIG):
        L += [f"![density sweep]({os.path.relpath(FIG, out.parent)})", ""]

    L += ["## 3. MeshFlow on these clouds (secondary)", "",
          "MeshFlow resamples every input to 4,096 points and voxelises them at 32^3, so it cannot see the density "
          "differences above; seed 42.", ""]
    for name, shapes in SETS:
        n = len(shapes)
        L += [f"**{name}**", "", header(["MeshFlow input"], MESH_COLS)]
        for label, suf in MESH_INPUTS:
            L.append(line([label] + cells([pipe(s, suf) for s in shapes], MESH_COLS, "mesh", n)))
        L.append("")

    L += ["## 4. Pilot on the dev shape (pencilpot): how the settings were chosen", "",
          header(["run", "what"], PCOLS + ALLCOLS + COV)]
    pilot = [("_tex_v3", "TSDF voxel 0.005, trunc 6 voxels, 200k (v3 default)"),
             ("_td0025", "voxel 0.0025, trunc 6 voxels"), ("_td0025t", "voxel 0.0025, trunc 12 voxels (same distance)"),
             ("_td00125", "voxel 0.00125, trunc 6 voxels"), ("_td00125t", "voxel 0.00125, trunc 24 voxels"),
             ("_td005n1m", "voxel 0.005, 1M points"), ("_td0025n1m", "voxel 0.0025, 1M points"),
             ("_gs", "Gaussian centres"), ("_gsk", "Gaussian kernels, opacity-weighted"),
             ("_gska", "Gaussian kernels, opacity x area")]
    for suf, what in pilot:
        r = pipe(DEV, suf)
        if r["points"]:
            L.append(line([suf.lstrip("_"), what] + cells([r], PCOLS + ALLCOLS, "points", 1) + cells([r], COV, "stats", 1)))
    L.append("")

    L += ["## 5. Per shape (points F1@0.01: TSDF v3 / TSDF voxel 0.0025 / Gaussian kernels)", "",
          "| shape | TSDF v3 | TSDF 0.0025 | Gaussian kernels |", "|---|---:|---:|---:|"]
    for s in SET_A + SET_B:
        vals = [(pipe(s, suf)["points"] or {}).get("F1@0.01") for suf in ["_tex_v3", "_td0025", "_gsk"]]
        L.append(line([s + (" (B)" if s in SET_B else "")] + [f"{v:.3f}" if v is not None else "" for v in vals]))
    L.append("")
    text = "\n".join(L) + "\n" + (manual or f"{MANUAL}\n## Interpretation\n\n(written when the queue has finished)\n")
    out.write_text(text)
    print(f"[report] {out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
