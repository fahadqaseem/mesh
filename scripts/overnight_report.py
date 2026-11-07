"""Build docs/results/2026-11-07_overnight.md from the JSON files the overnight jobs wrote (protocol v1).

Usage: python scripts/overnight_report.py
Safe to run at any time: missing results show as "-", so partial runs still give a valid report.
"""
from __future__ import annotations

import csv
import json
import statistics as st
from datetime import datetime
from pathlib import Path

from make_overnight_queue import GREY_SHAPES, SEEDS_EXTRA, SHAPES

OUT = Path("docs/results/2026-11-07_overnight.md")
STEP5 = [f"thingi_{i}" for i in Path("configs/step5_shapes.txt").read_text().split()]
SEEDS = [42] + SEEDS_EXTRA
MESH_COLS = [("CD", 2), ("HD", 1), ("NC", 3), ("F1@0.01", 3), ("F1@0.02", 3), ("V", 0), ("F", 0), ("G", 3),
             ("size_ratio", 2), ("total_s", 0)]
PTS_COLS = [("CD", 2), ("HD", 1), ("NC", 3), ("F1@0.01", 3), ("F1@0.02", 3), ("normal_outward", 3)]


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def mesh_json(run: str, seed: int = 42) -> dict | None:
    return load(Path("outputs") / run / "mesh" / f"meshflow_s{seed}.json")


def points_json(run: str) -> dict | None:
    return load(Path("outputs") / run / "eval" / "points_v1.json")


def fmt(v, nd: int) -> str:
    if v is None:
        return "-"
    return f"{v:,.0f}" if nd == 0 else f"{v:.{nd}f}"


def summary(values: list[float], nd: int) -> str:
    """mean ± std (min-max) over the available values."""
    vals = [v for v in values if v is not None]
    if not vals:
        return "-"
    if len(vals) == 1:
        return fmt(vals[0], nd)
    return f"{fmt(st.mean(vals), nd)} ± {fmt(st.stdev(vals), nd)} ({fmt(min(vals), nd)}-{fmt(max(vals), nd)})"


def table(rows: list[tuple[str, dict | None]], cols: list[tuple[str, int]], extra: list[tuple[str, callable]] = ()) -> list[str]:
    head = ["shape"] + [c for c, _ in cols] + [e for e, _ in extra]
    lines = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    for name, d in rows:
        cells = [fmt(d.get(c) if d else None, nd) for c, nd in cols] + [fn(name, d) for _, fn in extra]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    n = sum(1 for _, d in rows if d)
    lines.append(f"| **mean ± std (min-max), n={n}** | " +
                 " | ".join(summary([d.get(c) for _, d in rows if d], nd) for c, nd in cols) +
                 " |" + " |" * len(extra))
    return lines


def floor_col(name: str, d: dict | None) -> str:
    return fmt(d.get("CD_floor") if d else None, 2)


def job_status() -> tuple[list[str], dict[int, dict[str, int]]]:
    path = Path("outputs/overnight/status.tsv")
    failures, counts = [], {}
    if path.exists():
        for row in csv.reader(path.open(), delimiter="\t"):
            if len(row) < 4:
                continue
            step, jid, status = int(row[1]), row[2], row[3]
            counts.setdefault(step, {}).setdefault(status, 0)
            counts[step][status] += 1
            if status != "ok":
                failures.append(f"- step {step} `{jid}`: {status} ({row[4] if len(row) > 4 else ''})")
    return failures, counts


def main() -> None:
    failures, counts = job_status()
    L = [f"# Overnight results, 2026-11-07", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/overnight_report.py` from `outputs/` (not in git).",
         "Protocol v1 (`docs/eval_protocol.md`): camera-only alignment, no ICP; GT in [-0.5, 0.5]; CD/HD x100 at 5k samples",
         "(MeshFlow definition, not squared); NC and F1 at 100k samples; size = mesh max extent / GT max extent (1 = right size).",
         "Pipeline default v2: 8 views (4 per ring), known render focal, multi-view consistency (>= 2 views), 2 px foreground",
         "erosion, 200k point cap; MeshFlow hybrid (DiT on MPS, extraction on CPU), 28 steps, guidance 1.0.",
         "`CD floor` = CD of a perfect reconstruction under this protocol (sampling floor). Lower CD/HD is better; higher NC/F1 is better.",
         "", "## Job status", ""]
    for step in sorted(counts):
        L.append(f"- step {step}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts[step].items())))
    L += (["", "Failed or skipped jobs:"] + failures) if failures else ["", "No failed or skipped jobs so far."]

    L += ["", "## Step 1: full pipeline on 8 shapes (textured, 8 views, MeshFlow seed 42)", "",
          "### Mesh (headline: known focal)", ""]
    L += table([(s, mesh_json(f"{s}_tex_8v")) for s in SHAPES], MESH_COLS, [("CD floor", floor_col)])
    L += ["", "### Mesh, uncalibrated variant (VGGT's own focal, same VGGT pass)", ""]
    L += table([(s, mesh_json(f"{s}_tex_8v_vf")) for s in SHAPES], MESH_COLS)
    L += ["", "### Points stage (VGGT cloud vs GT, known focal; #V = points kept)", ""]
    L += table([(s, points_json(f"{s}_tex_8v")) for s in SHAPES], PTS_COLS + [("V", 0)])
    L += ["", "### Points stage, uncalibrated variant", ""]
    L += table([(s, points_json(f"{s}_tex_8v_vf")) for s in SHAPES], PTS_COLS + [("V", 0)])

    L += ["", "## Step 2: upper bound, MeshFlow on clean GT surface samples (seed 42)", "",
          "Not a pipeline result: shows what MeshFlow does with perfect input.", ""]
    L += table([(s, mesh_json(f"{s}_gtsamples")) for s in SHAPES], MESH_COLS, [("CD floor", floor_col)])

    L += ["", "## Step 3: seed variance (MeshFlow seeds 42, 7, 123 on each step-1 cloud)", "",
          "Per shape: mean ± std over seeds. Last rows compare the spread across seeds with the spread across shapes.", ""]
    cols3 = [("CD", 2), ("HD", 1), ("NC", 3), ("F1@0.01", 3), ("F1@0.02", 3), ("V", 0), ("size_ratio", 2)]
    L += ["| shape | seeds | " + " | ".join(c for c, _ in cols3) + " |", "|---|---:|" + "---:|" * len(cols3)]
    per_shape_means, per_shape_stds = {c: [] for c, _ in cols3}, {c: [] for c, _ in cols3}
    for s in SHAPES:
        runs = [d for d in (mesh_json(f"{s}_tex_8v", seed) for seed in SEEDS) if d]
        cells = []
        for c, nd in cols3:
            vals = [d[c] for d in runs if d.get(c) is not None]
            if vals:
                per_shape_means[c].append(st.mean(vals))
                if len(vals) > 1:
                    per_shape_stds[c].append(st.stdev(vals))
            cells.append(summary(vals, nd).split(" (")[0])
        L.append(f"| {s} | {len(runs)} | " + " | ".join(cells) + " |")
    L.append("| **mean of per-shape means** | | " + " | ".join(fmt(st.mean(per_shape_means[c]), nd) if per_shape_means[c] else "-" for c, nd in cols3) + " |")
    L.append("| **avg std across seeds** | | " + " | ".join(fmt(st.mean(per_shape_stds[c]), nd if nd else 0) if per_shape_stds[c] else "-" for c, nd in cols3) + " |")
    L.append("| **std across shapes** | | " + " | ".join(fmt(st.stdev(per_shape_means[c]), nd if nd else 0) if len(per_shape_means[c]) > 1 else "-" for c, nd in cols3) + " |")

    L += ["", "## Step 4: grey vs textured (8 views, default pipeline)", "",
          "Points stage, then MeshFlow mean over seeds 42/7/123. Delta = textured - grey (for CD/HD lower is better).", ""]
    hdr = ["shape", "pts F1@0.02 grey", "pts F1@0.02 tex", "pts CD grey", "pts CD tex", "focal ratio grey", "focal ratio tex",
           "mesh CD grey", "mesh CD tex", "mesh F1@0.02 grey", "mesh F1@0.02 tex"]
    L += ["| " + " | ".join(hdr) + " |", "|---|" + "---:|" * (len(hdr) - 1)]
    for s in GREY_SHAPES:
        pg, pt = points_json(f"{s}_grey_8v"), points_json(f"{s}_tex_8v")
        mg = [d for d in (mesh_json(f"{s}_grey_8v", sd) for sd in SEEDS) if d]
        mt = [d for d in (mesh_json(f"{s}_tex_8v", sd) for sd in SEEDS) if d]
        fr = [focal_ratio(f"{s}_grey_8v", s), focal_ratio(f"{s}_tex_8v", s)]
        m = lambda ds, c: st.mean([d[c] for d in ds]) if ds else None
        L.append(f"| {s} | {fmt(pg and pg['F1@0.02'], 3)} | {fmt(pt and pt['F1@0.02'], 3)} | {fmt(pg and pg['CD'], 2)} | "
                 f"{fmt(pt and pt['CD'], 2)} | {fr[0]} | {fr[1]} | {fmt(m(mg, 'CD'), 2)} | {fmt(m(mt, 'CD'), 2)} | "
                 f"{fmt(m(mg, 'F1@0.02'), 3)} | {fmt(m(mt, 'F1@0.02'), 3)} |")

    L += ["", "## Step 5: 12 more Thingi10K shapes (textured, 8 views, seed 42)", "", "### Mesh", ""]
    L += table([(s, mesh_json(f"{s}_tex_8v")) for s in STEP5], MESH_COLS, [("CD floor", floor_col)])
    L += ["", "### Upper bound (GT surface samples)", ""]
    L += table([(s, mesh_json(f"{s}_gtsamples")) for s in STEP5], MESH_COLS)
    all_rows = [(s, mesh_json(f"{s}_tex_8v")) for s in SHAPES + STEP5]
    L += ["", "### All shapes so far (steps 1 + 5), pipeline mesh", ""]
    L.append("| n | " + " | ".join(c for c, _ in MESH_COLS) + " |")
    L.append("|---:|" + "---:|" * len(MESH_COLS))
    L.append(f"| {sum(1 for _, d in all_rows if d)} | " + " | ".join(summary([d.get(c) for _, d in all_rows if d], nd) for c, nd in MESH_COLS) + " |")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"[report] -> {OUT}")


def focal_ratio(run: str, scene: str) -> str:
    """VGGT's predicted focal / true render focal (cameras.npz keeps VGGT's estimate as intrinsic_pred)."""
    import numpy as np
    cams, meta = Path("outputs") / run / "points" / "cameras.npz", load(Path("data/scenes") / scene / "gt_transform.json")
    if not cams.exists() or not meta:
        return "-"
    c = np.load(cams)
    k = c["intrinsic_pred"] if "intrinsic_pred" in c else c["intrinsic"]
    f_true = (meta["size"] / 2) / np.tan(np.deg2rad(meta["yfov_deg"]) / 2)
    return f"{k[:, 1, 1].mean() / f_true:.3f}"


if __name__ == "__main__":
    main()
