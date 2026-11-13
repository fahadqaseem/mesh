"""Regenerate docs/results/2026-11-13_fps_input.md: MeshFlow on 4,096 FPS points vs its own random pick.

Usage: python scripts/fps_report.py [--out docs/results/2026-11-13_fps_input.md]
Safe on partial results. Everything from "<!-- manual -->" onwards in the existing doc is kept.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from robustness_report import DEV, MANUAL, MESH_COLS, OUT, SET_A, SET_B, cells, header, line, load, sd

SEEDS = [42, 7, 123]
INPUTS = [("TSDF, MeshFlow's random 4,096 (current)", "_tex_v3"), ("TSDF, FPS 4,096", "_fps"),
          ("TSDF + kernels, MeshFlow's random 4,096", "_uni"), ("TSDF + kernels, FPS 4,096", "_fpsu")]
UPPER = ("upper bound: clean GT points", "_gtsamples")
IN_COLS = [("cells", "occupied 32^3 cells", 0), ("cell_recall", "GT cells hit", 3), ("cell_precision", "cells on GT", 3),
           ("off_surface", "points off surface (> 2%)", 3), ("gap_p95", "GT gap p95 (% of extent)", 2)]
PAIRS = [("_tex_v3", "_fps", "TSDF: FPS minus random"), ("_uni", "_fpsu", "TSDF + kernels: FPS minus random"),
         ("_tex_v3", "_fpsu", "TSDF + kernels FPS minus TSDF random (current)")]


def mesh(shape: str, suffix: str, seed: int) -> dict | None:
    return load(OUT / f"{shape}{suffix}" / "mesh" / f"meshflow_s{seed}.json")


def runs(shapes: list[str], suffix: str, seeds: list[int]) -> list[dict]:
    """One dict per shape: mesh metrics averaged over the given seeds (only if all exist)."""
    out = []
    for s in shapes:
        ms = [mesh(s, suffix, sd_) for sd_ in seeds]
        out.append({"mesh": {k: float(np.mean([m[k] for m in ms])) for k, _, _ in MESH_COLS} if all(ms) else None})
    return out


def paired(shapes: list[str], a: str, b: str, key: str, seeds: list[int]) -> str:
    """b minus a over every (shape, seed) pair where both exist."""
    d = [mesh(s, b, x)[key] - mesh(s, a, x)[key] for s in shapes for x in seeds if mesh(s, a, x) and mesh(s, b, x)]
    if not d:
        return "-"
    d = np.array(d, dtype=float)
    better = (d < 0).sum() if key in ("CD", "HD", "V") else (d > 0).sum()
    word = "lower" if key == "V" else "better"
    return f"{d.mean():+.3g} ± {sd(d):.2g} ({word} on {better} of {len(d)})"


def queue_line() -> str:
    q, st = OUT / "fps_queue" / "queue.tsv", OUT / "fps_queue" / "status.tsv"
    if not q.exists():
        return ""
    total = sum(1 for _ in q.open())
    last = {}
    for r in (ln.rstrip("\n").split("\t") for ln in st.open()) if st.exists() else []:
        if len(r) >= 4:
            last[r[2]] = r[3]
    bad = sorted(k for k, v in last.items() if v != "ok")
    return f"Queue: {sum(v == 'ok' for v in last.values())} of {total} jobs ok" + (f"; not ok: {', '.join(bad)}" if bad else "") + "."


def fig(path: Path) -> bool:
    if not any(mesh(s, "_fps", 42) for s in SET_A + SET_B):
        return False
    from robustness_report import BLUE, ORANGE, SURFACE, dot, plot_style
    plt = plt_ = plot_style()  # pyplot
    rows = INPUTS + [UPPER]
    metrics = [("CD", "mesh Chamfer (lower is better)"), ("F1@0.02", "mesh F1@0.02 (higher is better)"), ("V", "mesh vertices")]
    fig_, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=True)
    y = np.arange(len(rows))[::-1].astype(float)
    for ax, (k, xlabel) in zip(axes, metrics):
        for shapes, seeds, color, off in [(SET_A, SEEDS, BLUE, 0.13), (SET_B, [42], ORANGE, -0.13)]:
            for yi, (_, suf) in zip(y, rows):
                sds = [42] if suf == "_gtsamples" else seeds
                vals = np.array([r["mesh"][k] for r in runs(shapes, suf, sds) if r["mesh"]])
                if len(vals):
                    ax.plot([vals.mean() - sd(vals), vals.mean() + sd(vals)], [yi + off] * 2, color=color, lw=1.2,
                            alpha=0.55, zorder=2)
                    dot(ax, [vals.mean()], [yi + off], color, size=6.5)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [n for n, _ in rows])
    handles = [plt_.Line2D([], [], color=BLUE, lw=1.2, marker="o", ms=6.5, mec=SURFACE, label="set A (7 shapes, 3 seeds)"),
               plt_.Line2D([], [], color=ORANGE, lw=1.2, marker="o", ms=6.5, mec=SURFACE,
                           label="set B (12 shapes, clean test set, seed 42)")]
    fig_.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.62, 1.0), fontsize=8)
    fig_.text(0.01, 0.97, "MeshFlow input: mean ± std over shapes", color="#52514e", fontsize=8, va="top")
    fig_.tight_layout(rect=(0, 0, 1, 0.9))
    fig_.savefig(path, dpi=170)
    plt.close(fig_)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results/2026-11-13_fps_input.md")
    out = Path(ap.parse_args().out)
    old = out.read_text() if out.exists() else ""
    manual = old[old.index(MANUAL):] if MANUAL in old else ""
    L = ["# MeshFlow input: 4,096 farthest-point-sampled points vs MeshFlow's own random pick", "",
         "Fahad Qaseem Khawar, 13 Nov 2026. Tables and the figure are regenerated by `scripts/fps_report.py`; the "
         "interpretation at the end is written by hand. Protocol v1 mesh metrics (`docs/eval_protocol.md`). Set A = 7 "
         "original shapes, set B = 12 new shapes (clean test set); pencilpot is the dev shape and is in no mean.", "",
         queue_line(), "",
         "**Why:** MeshFlow's network takes exactly 4,096 input points (fixed by training). Its loader normalises the "
         "cloud and keeps `torch.randperm(N)[:4096]`, a random 2% of our 200k TSDF points. Handing it exactly 4,096 points "
         "chosen by farthest-point sampling (FPS, `scripts/fps_points.py`) makes that pick a permutation, so every FPS point "
         "reaches the network. FPS spreads the points evenly over the cloud, but it also seeks out the cloud's extremes, "
         "so stray points are more likely to be kept. **Inputs** (same VGGT pass, same alignment, no rerun of points): the TSDF cloud (`_tex_v3`, 200k); "
         "the TSDF cloud plus 200k Gaussian-kernel samples (`_gsk`, opacity-weighted), a simple union of 400k points. "
         "The current random-pick rows are the existing `_tex_v3` meshes (seeds 42, 7, 123). Note that the input "
         "size changes how many random numbers the pick consumes, so the same seed gives a different generation noise "
         "across inputs; the 3 seeds on set A are what separates the input effect from that.", ""]

    L += ["## 1. What MeshFlow sees", "",
          "`scripts/meshflow_input_stats.py`: the 4,096 points after MeshFlow's normalisation, voxelised at 32^3 as in "
          "`voxelize_pc`, against 100k GT surface samples in the same frame (random picks reproduced on the CPU with "
          "seed 42: same distribution as MeshFlow's MPS pick, not necessarily the same points).", ""]
    for name, shapes in [("Set B (12 shapes, clean test set)", SET_B), ("Set A (7 shapes)", SET_A)]:
        L += [f"**{name}**", "", header(["MeshFlow input"], IN_COLS)]
        for label, suf in INPUTS:
            rs = [{"in": load(OUT / f"{s}{suf}" / "eval" / "meshflow_input.json")} for s in shapes]
            L.append(line([label] + cells(rs, IN_COLS, "in", len(shapes))))
        L.append("")

    L += ["## 2. Meshes, MeshFlow seed 42", ""]
    for name, shapes in [("Set B (12 shapes, clean test set)", SET_B), ("Set A (7 shapes)", SET_A)]:
        n = len(shapes)
        L += [f"**{name}**", "", header(["MeshFlow input"], MESH_COLS)]
        for label, suf in INPUTS + [UPPER]:
            L.append(line([label] + cells(runs(shapes, suf, [42]), MESH_COLS, "mesh", n)))
        L += ["", "Paired per shape: " + "; ".join(
            f"{t}: Chamfer {paired(shapes, a, b, 'CD', [42])}, F1@0.02 {paired(shapes, a, b, 'F1@0.02', [42])}, "
            f"#V {paired(shapes, a, b, 'V', [42])}" for a, b, t in PAIRS) + ".", ""]

    n = len(SET_A)
    L += ["## 3. Set A, mean of 3 MeshFlow seeds (42, 7, 123)", "",
          "Each shape's metrics are first averaged over the 3 seeds; mean ± std over the 7 shapes. Paired differences "
          "use all 21 (shape, seed) pairs.", "", header(["MeshFlow input"], MESH_COLS)]
    for label, suf in INPUTS:
        L.append(line([label] + cells(runs(SET_A, suf, SEEDS), MESH_COLS, "mesh", n)))
    L += ["", "Paired: " + "; ".join(
        f"{t}: Chamfer {paired(SET_A, a, b, 'CD', SEEDS)}, F1@0.02 {paired(SET_A, a, b, 'F1@0.02', SEEDS)}, "
        f"#V {paired(SET_A, a, b, 'V', SEEDS)}" for a, b, t in PAIRS) + ".", ""]

    p = Path("docs/results/figures/2026-11-13_fps_input.png")
    if fig(p):
        L += [f"![FPS input]({os.path.relpath(p, out.parent)})", ""]

    d = [(label, mesh(DEV, suf, 42)) for label, suf in INPUTS]
    if any(m for _, m in d):
        L += ["## 4. Dev shape (pencilpot, not in means), seed 42", "", "| MeshFlow input | Chamfer | F1@0.02 | #V |",
              "|---|---:|---:|---:|"]
        L += [line([label, f"{m['CD']:.2f}", f"{m['F1@0.02']:.3f}", f"{m['V']:,}"]) for label, m in d if m]
        L.append("")

    L += ["## 5. Per shape (mesh Chamfer, seed 42)", "",
          "| shape | " + " | ".join(lbl for lbl, _ in INPUTS) + " |", "|---|" + "---:|" * len(INPUTS)]
    for s in SET_A + SET_B:
        items = [f"{m['CD']:.2f}" if (m := mesh(s, suf, 42)) else "" for _, suf in INPUTS]
        if any(items[1:]):
            L.append(line([s + (" (B)" if s in SET_B else "")] + items))
    L.append("")
    out.write_text("\n".join(L) + "\n" + (manual or f"{MANUAL}\n## Interpretation\n\n(written when the queue has finished)\n"))
    print(f"[report] {out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
