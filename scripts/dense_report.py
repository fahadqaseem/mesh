"""Regenerate docs/results/2026-11-15_dense.md: dense default, more views, MeshFlow vertex control, pictures.

Usage: python scripts/dense_report.py [--out docs/results/2026-11-15_dense.md]
Safe on partial results. Kept by hand: the summary between "<!-- summary -->" and "<!-- /summary -->" at the top, and
everything from "<!-- manual -->" onwards.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from robustness_report import DEV, MANUAL, OUT, SET_A, SET_B, cells, header, line, load, pipe, sd

SUM0, SUM1 = "<!-- summary -->", "<!-- /summary -->"
SETS = [("Set B (12 shapes, clean test set)", SET_B), ("Set A (7 shapes)", SET_A)]
HCOLS = [("CD", "Chamfer", 2), ("F1@0.01", "F1@0.01", 3), ("F1@0.02", "F1@0.02", 3),
         ("R_all@0.01", "recall, all points @0.01", 3), ("R_all@0.02", "recall, all points @0.02", 3), ("NC", "NC", 3)]
COV, NPTS = [("coverage", "coverage (%)", 1)], [("V", "#points", 0)]
HEAD = [("TSDF, 200k (previous default)", "_tex_v3"), ("TSDF, 1M (new default)", "_dense"),
        ("TSDF + gap fill, 200k base", "_tfill"), ("TSDF + gap fill, 1M base (new best)", "_dfill")]
VIEW_SHAPES = [DEV, "thingi_103356", "thingi_77949", "thingi_593381"]
VIEWS = [("8 views (default)", "_dense"), ("12 views", "_v12"), ("16 views", "_v16"), ("16 views, 392 px", "_v16s")]


def queue_line(name: str) -> str:
    q, st = OUT / name / "queue.tsv", OUT / name / "status.tsv"
    if not q.exists():
        return ""
    total = sum(1 for _ in q.open())
    last = {}
    for r in (ln.rstrip("\n").split("\t") for ln in st.open()) if st.exists() else []:
        if len(r) >= 4:
            last[r[2]] = r[3]
    bad = sorted(k for k, v in last.items() if v != "ok")
    return f"Queue `{name}`: {sum(v == 'ok' for v in last.values())} of {total} jobs ok" + \
        (f"; not ok: {', '.join(bad)}" if bad else "") + "."


def paired(shapes: list[str], a: str, b: str, key: str) -> str:
    d = [pipe(s, b)["points"][key] - pipe(s, a)["points"][key] for s in shapes
         if pipe(s, a)["points"] and pipe(s, b)["points"]]
    if not d:
        return "-"
    d = np.array(d)
    better = (d < 0).sum() if key in ("CD", "HD") else (d > 0).sum()
    return f"{d.mean():+.3g} ± {sd(d):.2g} (better on {better} of {len(d)})"


def section_dense(L: list[str]) -> None:
    L += ["## 1. Dense output is the default: 1M points at TSDF voxel 0.005", "",
          "Points vs GT, protocol v1 (Chamfer x100 on 5,000 points per side; F1 and NC on 100,000), plus the "
          "all-points recall (share of 100,000 GT samples within tau of any point of the cloud; the metric that "
          "density can move). The 1M TSDF clouds are resampled from the same fused surfaces as the 200k ones "
          "(`scripts/resample_tsdf.py`, identical to the 1M sweep cells of 12 Nov, checked on thingi_59706). Gap fill: "
          "kernel samples 3-10% from the TSDF cloud (rule of 14 Nov); at 1M the kernels are sampled at 1M too.", ""]
    for name, shapes in SETS:
        n = len(shapes)
        L += [f"**{name}**", "", header(["points"], HCOLS + COV + NPTS)]
        for label, suf in HEAD:
            runs = [pipe(s, suf) for s in shapes]
            L.append(line([label] + cells(runs, HCOLS, "points", n) + cells(runs, COV, "stats", n) +
                          cells(runs, NPTS, "points", n)))
        L += ["", "Paired per shape: dense gap fill minus dense TSDF: " +
              ", ".join(f"{k} {paired(shapes, '_dense', '_dfill', k)}" for k in ["CD", "F1@0.01", "F1@0.02", "R_all@0.01", "R_all@0.02", "NC"]) +
              "; dense minus 200k TSDF: " + ", ".join(f"{k} {paired(shapes, '_tex_v3', '_dense', k)}" for k in ["CD", "R_all@0.01", "R_all@0.02"]) + ".", ""]


def section_views(L: list[str]) -> None:
    if not any(pipe(s, "_v12")["points"] for s in VIEW_SHAPES + SET_A + SET_B):
        return
    L += ["## 2. More views on the laptop: 8 vs 12 vs 16", ""]
    vi = load(OUT / "views_log.json") or {}
    if vi.get("note"):
        L += [vi["note"], ""]
    cols = HCOLS + COV
    L += ["**Per shape (pencilpot + 3 set B shapes)**", "",
          "| shape | views | Chamfer | F1@0.01 | F1@0.02 | recall all @0.01 | recall all @0.02 | NC | coverage (%) | VGGT (s) | peak memory (GB) |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in VIEW_SHAPES:
        for label, suf in VIEWS:
            p, st = pipe(s, suf)["points"], pipe(s, suf)["stats"]
            if not p:
                continue
            info = load(OUT / f"{s}{suf}" / "points" / "tsdf.json") or {}
            if suf == "_dense":
                info = load(OUT / f"{s}_tdraw" / "points" / "tsdf.json") or info
            L.append(line([s, label] + [f"{p[k]:.3f}" if k != "CD" else f"{p[k]:.2f}" for k, _, _ in HCOLS] +
                          [f"{100 * st['coverage']:.1f}" if st else "", f"{info['vggt_s']:.0f}" if info.get("vggt_s") else "",
                           f"{info['peak_rss_gb']:.1f}" if info.get("peak_rss_gb") else ""]))
    L.append("")
    for name, shapes in SETS:
        if not all(pipe(s, "_v12")["points"] for s in shapes):
            continue
        n = len(shapes)
        L += [f"**{name}**", "", header(["views"], cols)]
        for label, suf in VIEWS:
            runs = [pipe(s, suf) for s in shapes]
            if any(r["points"] for r in runs):
                L.append(line([label] + cells(runs, HCOLS, "points", n) + cells(runs, COV, "stats", n)))
        L += ["", "Paired per shape, minus 8 views: " + "; ".join(
            f"{lbl}: " + ", ".join(f"{k} {paired(shapes, '_dense', suf, k)}" for k in ["CD", "F1@0.02", "R_all@0.02"])
            for lbl, suf in VIEWS[1:] if any(pipe(s, suf)["points"] for s in shapes)) + ".", ""]


def section_meshflow_verts(L: list[str], out: Path) -> None:
    info = load(OUT / "nverts_check.json")
    if not info:
        return
    L += ["## 3. MeshFlow vertex control", "", info.get("note", ""), ""]
    rows = info.get("rows") or []
    if rows:
        L += ["| MeshFlow input | num_verts | Chamfer | F1@0.02 | NC | #V (mean ± std over set A) |", "|---|---:|---:|---:|---:|---:|"]
        for r in rows:
            L.append(line([r["input"], f"{r['num_verts']:,}"] + [r.get(k, "-") for k in ["CD", "F1@0.02", "NC", "V"]]))
        L.append("")
    p = Path("docs/results/figures/2026-11-15_meshflow_nverts.png")
    if p.exists():
        L += [f"![MeshFlow vertex control]({os.path.relpath(p, out.parent)})", ""]


def section_pictures(L: list[str], out: Path) -> None:
    p = Path("docs/results/figures/2026-11-15_gapfill_compare.png")
    if not p.exists():
        return
    L += ["## 4. Pictures: TSDF vs gap-filled, coloured by distance to the GT", "",
          f"![TSDF vs gap fill]({os.path.relpath(p, out.parent)})", "",
          "thingi_103356, thingi_77949 and thingi_593381 (set B, the three shapes where TSDF is worst). Top: the dense TSDF "
          "cloud; bottom: the dense gap-filled cloud. Colour = distance to the GT surface (% of the GT's largest extent, "
          "clipped at 5%); GT surface regions with no point within 2% are drawn in grey on the GT.", ""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results/2026-11-15_dense.md")
    out = Path(ap.parse_args().out)
    old = out.read_text() if out.exists() else ""
    summary = old[old.index(SUM0):old.index(SUM1) + len(SUM1)] if SUM0 in old and SUM1 in old else \
        f"{SUM0}\n**Summary:** (written when the block is done)\n{SUM1}"
    manual = old[old.index(MANUAL):] if MANUAL in old else f"{MANUAL}\n## Interpretation\n\n(written when the block is done)\n"
    L = ["# Dense points by default, more views, MeshFlow vertex control", "",
         "Fahad Qaseem Khawar, 15 Nov 2026. Tables regenerated by `scripts/dense_report.py`; the summary and the "
         "interpretation are written by hand. Set A = 7 original shapes, set B = 12 new shapes (clean test set); "
         "pencilpot is the dev shape and is in no mean. Mean ± std over shapes.", "", summary, "",
         " ".join(x for x in [queue_line("dense_queue"), queue_line("views_queue"), queue_line("nverts_queue")] if x), ""]
    section_dense(L)
    section_views(L)
    section_meshflow_verts(L, out)
    section_pictures(L, out)
    out.write_text("\n".join(L) + "\n" + manual)
    print(f"[report] {out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
