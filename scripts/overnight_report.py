"""Build docs/results/2026-11-07_overnight.md from the JSON files the overnight jobs wrote (protocol v1).

Usage: python scripts/overnight_report.py [--out PATH]

Shape sets:
  dev    pencilpot_299291: used to tune the filter thresholds, so it is shown in every table but excluded from all means.
  set A  the other 7 original shapes: not used for tuning (they were looked at in earlier runs).
  set B  the 12 step-5 shapes: picked automatically and never looked at before this run; the clean test set.
Safe to run at any time: missing results show as "-", so partial runs still give a valid report.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from datetime import datetime
from pathlib import Path

import numpy as np

from make_overnight_queue import GREY_SHAPES, SEEDS_EXTRA, SHAPES

OUT = Path("docs/results/2026-11-07_overnight.md")
DEV = "pencilpot_299291"
SET_A = [s for s in SHAPES if s != DEV]
SET_B = [f"thingi_{i}" for i in Path("configs/step5_shapes.txt").read_text().split()]
GREY_TEST = [s for s in GREY_SHAPES if s != DEV]
SEEDS = [42] + SEEDS_EXTRA
MESH_COLS = [("CD", 2), ("HD", 1), ("NC", 3), ("F1@0.01", 3), ("F1@0.02", 3), ("V", 0), ("F", 0), ("G", 3),
             ("size_ratio", 2), ("total_s", 0)]
PTS_COLS = [("CD", 2), ("HD", 1), ("NC", 3), ("F1@0.01", 3), ("F1@0.02", 3), ("normal_outward", 3), ("V", 0)]
HEAD_COLS = [("CD", 2), ("HD", 1), ("NC", 3), ("F1@0.01", 3), ("F1@0.02", 3), ("V", 0), ("size_ratio", 2)]


# ---------- loading and formatting ----------

def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def mesh_json(run: str, seed: int = 42) -> dict | None:
    return load(Path("outputs") / run / "mesh" / f"meshflow_s{seed}.json")


def points_json(run: str) -> dict | None:
    return load(Path("outputs") / run / "eval" / "points_v1.json")


def seed_mean(run: str, require_all: bool = False) -> dict | None:
    """Per-metric mean over the seeds of one run. None if no seed finished, or (require_all) if any seed is missing."""
    runs = [d for d in (mesh_json(run, sd) for sd in SEEDS) if d]
    if not runs or (require_all and len(runs) < len(SEEDS)):
        return None
    keys = [k for k, _ in MESH_COLS if all(isinstance(d.get(k), (int, float)) for d in runs)]
    out = {k: st.mean(d[k] for d in runs) for k in keys}
    out["n_seeds"] = len(runs)
    return out


def fmt(v, nd: int) -> str:
    if v is None:
        return "-"
    return f"{v:,.0f}" if nd == 0 else f"{v:.{nd}f}"


def stats(values: list, nd: int, spread: bool = True) -> str:
    """mean ± std (min-max) over the values that exist."""
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return "-"
    if len(vals) == 1 or not spread:
        return fmt(st.mean(vals), nd)
    return f"{fmt(st.mean(vals), nd)} ± {fmt(st.stdev(vals), nd)} ({fmt(min(vals), nd)}-{fmt(max(vals), nd)})"


def name(shape: str) -> str:
    return f"{shape} (dev, not in means)" if shape == DEV else shape


def table(rows: list[tuple[str, dict | None]], cols: list[tuple[str, int]], groups: list[tuple[str, list[str]]],
          extra: list[tuple[str, callable]] = ()) -> list[str]:
    """Per-shape rows, then one summary row per group (mean ± std (min-max) over the shapes in that group)."""
    head = ["shape"] + [c for c, _ in cols] + [e for e, _ in extra]
    lines = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    for shape, d in rows:
        cells = [fmt(d.get(c) if d else None, nd) for c, nd in cols] + [fn(shape, d) for _, fn in extra]
        lines.append(f"| {name(shape)} | " + " | ".join(cells) + " |")
    for label, members in groups:
        ds = [d for s, d in rows if s in members and d]
        lines.append(f"| **{label}, n={len(ds)}** | " + " | ".join(stats([d.get(c) for d in ds], nd) for c, nd in cols)
                     + " |" + " |" * len(extra))
    return lines


def floor_col(_shape: str, d: dict | None) -> str:
    return fmt(d.get("CD_floor") if d else None, 2)


def focal_ratio(run: str, scene: str) -> float | None:
    """VGGT's predicted focal / true render focal (cameras.npz keeps VGGT's estimate as intrinsic_pred)."""
    cams, meta = Path("outputs") / run / "points" / "cameras.npz", load(Path("data/scenes") / scene / "gt_transform.json")
    if not cams.exists() or not meta:
        return None
    c = np.load(cams)
    k = c["intrinsic_pred"] if "intrinsic_pred" in c else c["intrinsic"]
    f_true = (meta["size"] / 2) / np.tan(np.deg2rad(meta["yfov_deg"]) / 2)
    return float(k[:, 1, 1].mean() / f_true)


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


def queue_counts() -> dict[int, int]:
    path = Path("outputs/overnight/queue.tsv")
    counts: dict[int, int] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                counts[int(line.split("\t")[0])] = counts.get(int(line.split("\t")[0]), 0) + 1
    return counts


# ---------- sections ----------

def headline() -> list[str]:
    rows = [
        ("Pipeline mesh, known focal (seed 42)", "A", [mesh_json(f"{s}_tex_8v") for s in SET_A]),
        ("Pipeline mesh, known focal (seed 42)", "B test", [mesh_json(f"{s}_tex_8v") for s in SET_B]),
        ("Pipeline mesh, known focal (mean of 3 seeds per shape)", "A", [seed_mean(f"{s}_tex_8v", True) for s in SET_A]),
        ("Pipeline mesh, VGGT focal (uncalibrated, seed 42)", "A", [mesh_json(f"{s}_tex_8v_vf") for s in SET_A]),
        ("Upper bound: MeshFlow on clean GT samples (seed 42)", "A", [mesh_json(f"{s}_gtsamples") for s in SET_A]),
        ("Upper bound: MeshFlow on clean GT samples (seed 42)", "B test", [mesh_json(f"{s}_gtsamples") for s in SET_B]),
        ("Points stage: VGGT cloud before meshing (#V = points)", "A", [points_json(f"{s}_tex_8v") for s in SET_A]),
        ("Points stage: VGGT cloud before meshing (#V = points)", "B test", [points_json(f"{s}_tex_8v") for s in SET_B]),
    ]
    L = ["| stage | set | n | " + " | ".join(c for c, _ in HEAD_COLS) + " |", "|---|---|---:|" + "---:|" * len(HEAD_COLS)]
    for label, set_name, ds in rows:
        ds = [d for d in ds if d]
        L.append(f"| {label} | {set_name} | {len(ds)} | " + " | ".join(stats([d.get(c) for d in ds], nd) for c, nd in HEAD_COLS) + " |")
    floors_a = [d.get("CD_floor") for d in (mesh_json(f"{s}_tex_8v") for s in SET_A) if d]
    floors_b = [d.get("CD_floor") for d in (mesh_json(f"{s}_tex_8v") for s in SET_B) if d]
    L += ["", f"CD floor (what a perfect mesh scores under this protocol): set A {stats(floors_a, 2)}, set B {stats(floors_b, 2)}."]
    return L


def seed_spread_line() -> list[str]:
    per_shape = {c: ([], []) for c in ("CD", "F1@0.02", "V")}
    for s in SET_A:
        runs = [d for d in (mesh_json(f"{s}_tex_8v", sd) for sd in SEEDS) if d]
        for c, (means, stds) in per_shape.items():
            vals = [d[c] for d in runs if isinstance(d.get(c), (int, float))]
            if len(vals) == len(SEEDS):
                means.append(st.mean(vals))
                stds.append(st.stdev(vals))
    parts = []
    for c, (means, stds) in per_shape.items():
        nd = 0 if c == "V" else (2 if c == "CD" else 3)
        if len(means) > 1:
            parts.append(f"{c}: across seeds {fmt(st.mean(stds), nd)} vs across shapes {fmt(st.stdev(means), nd)}")
    n = len(per_shape["CD"][0])
    return [f"Seed spread (set A, {n} shapes with all {len(SEEDS)} seeds; average std across seeds vs std across shapes): "
            + ("; ".join(parts) if parts else "pending") + "."]


def grey_tex_line() -> list[str]:
    d_pts, d_mesh_cd, d_mesh_f1, fr_g, fr_t = [], [], [], [], []
    for s in GREY_TEST:
        pg, pt = points_json(f"{s}_grey_8v"), points_json(f"{s}_tex_8v")
        mg, mt = seed_mean(f"{s}_grey_8v", True), seed_mean(f"{s}_tex_8v", True)
        if pg and pt:
            d_pts.append(pt["F1@0.02"] - pg["F1@0.02"])
        if mg and mt:
            d_mesh_cd.append(mt["CD"] - mg["CD"])
            d_mesh_f1.append(mt["F1@0.02"] - mg["F1@0.02"])
        fr_g.append(focal_ratio(f"{s}_grey_8v", s))
        fr_t.append(focal_ratio(f"{s}_tex_8v", s))
    return [f"Grey vs textured ({len(d_pts)} of {len(GREY_TEST)} shapes, dev shape excluded; textured minus grey): "
            f"points F1@0.02 {stats(d_pts, 3)}; mesh CD {stats(d_mesh_cd, 2)}; mesh F1@0.02 {stats(d_mesh_f1, 3)}; "
            f"VGGT focal ratio grey {stats(fr_g, 3)} vs textured {stats(fr_t, 3)}."]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    out = Path(ap.parse_args().out)

    failures, counts = job_status()
    planned = queue_counts()
    groups_a = [("mean ± std (min-max), set A", SET_A)]
    groups_b = [("mean ± std (min-max), set B (test)", SET_B)]
    L = ["# Overnight results, 2026-11-07", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/overnight_report.py` from `outputs/` (not in git).", "",
         "Protocol v1 (`docs/eval_protocol.md`): camera-only alignment, no ICP; GT in [-0.5, 0.5]; CD/HD x100 at 5k samples",
         "(MeshFlow definition, not squared); NC and F1 at 100k samples; size = mesh max extent / GT max extent (1 = right size).",
         "Pipeline default v2: 8 views (4 per ring), known render focal, multi-view consistency (>= 2 views), 2 px foreground",
         "erosion, 200k point cap; MeshFlow hybrid (DiT on MPS, extraction on CPU), 28 steps, guidance 1.0.",
         "Lower CD/HD is better; higher NC/F1 is better.", "",
         f"Shape sets: **dev** = {DEV} (used to tune the filter thresholds; shown, but excluded from every mean). "
         f"**Set A** = the other {len(SET_A)} original shapes. **Set B** = the {len(SET_B)} step-5 shapes, picked automatically "
         "and never looked at before this run: the clean test set.", "",
         "## Headline (mean ± std (min-max) over shapes; dev shape excluded)", ""]
    L += headline()
    L += [""] + seed_spread_line() + [""] + grey_tex_line()

    L += ["", "## Job status", ""]
    for step in sorted(planned):
        done = ", ".join(f"{k} {v}" for k, v in sorted(counts.get(step, {}).items())) or "not started"
        L.append(f"- step {step}: {sum(counts.get(step, {}).values())} of {planned[step]} jobs run ({done})")
    L += (["", "Failed or skipped jobs:"] + failures) if failures else ["", "No failed or skipped jobs so far."]

    L += ["", "## Step 1: full pipeline, original 8 shapes (textured, 8 views, MeshFlow seed 42)", "",
          "### Mesh (headline: known focal)", ""]
    L += table([(s, mesh_json(f"{s}_tex_8v")) for s in SHAPES], MESH_COLS, groups_a, [("CD floor", floor_col)])
    L += ["", "### Mesh, uncalibrated variant (VGGT's own focal, same VGGT pass)", ""]
    L += table([(s, mesh_json(f"{s}_tex_8v_vf")) for s in SHAPES], MESH_COLS, groups_a,
               [("VGGT focal ratio", lambda s, d: fmt(focal_ratio(f"{s}_tex_8v", s), 3))])
    L += ["", "### Points stage (VGGT cloud vs GT, known focal; V = points kept)", ""]
    L += table([(s, points_json(f"{s}_tex_8v")) for s in SHAPES], PTS_COLS, groups_a)
    L += ["", "### Points stage, uncalibrated variant", ""]
    L += table([(s, points_json(f"{s}_tex_8v_vf")) for s in SHAPES], PTS_COLS, groups_a)

    L += ["", "## Step 2: upper bound, MeshFlow on clean GT surface samples (seed 42)", "",
          "Not a pipeline result: shows what MeshFlow does with perfect input.", ""]
    L += table([(s, mesh_json(f"{s}_gtsamples")) for s in SHAPES], MESH_COLS, groups_a, [("CD floor", floor_col)])

    L += ["", "## Step 3: seed variance (MeshFlow seeds 42, 7, 123 on each step-1 cloud)", "",
          "Per shape: mean ± std over seeds.", ""]
    cols3 = [("CD", 2), ("HD", 1), ("NC", 3), ("F1@0.01", 3), ("F1@0.02", 3), ("V", 0), ("size_ratio", 2)]
    L += ["| shape | seeds | " + " | ".join(c for c, _ in cols3) + " |", "|---|---:|" + "---:|" * len(cols3)]
    for s in SHAPES:
        runs = [d for d in (mesh_json(f"{s}_tex_8v", sd) for sd in SEEDS) if d]
        cells = [stats([d.get(c) for d in runs], nd).split(" (")[0] for c, nd in cols3]
        L.append(f"| {name(s)} | {len(runs)} | " + " | ".join(cells) + " |")
    means = [seed_mean(f"{s}_tex_8v", True) for s in SET_A]
    means = [m for m in means if m]
    L.append(f"| **set A: mean ± std of per-shape means, n={len(means)}** | | "
             + " | ".join(stats([m.get(c) for m in means], nd).split(" (")[0] for c, nd in cols3) + " |")
    L += [""] + seed_spread_line()

    L += ["", "## Step 4: grey vs textured (8 views, default pipeline)", "",
          "Points stage, then MeshFlow mean over seeds 42/7/123. Focal ratio = VGGT's predicted focal / true focal.", ""]
    hdr = ["shape", "pts F1@0.02 grey", "pts F1@0.02 tex", "pts CD grey", "pts CD tex", "focal ratio grey",
           "focal ratio tex", "mesh CD grey", "mesh CD tex", "mesh F1@0.02 grey", "mesh F1@0.02 tex"]
    L += ["| " + " | ".join(hdr) + " |", "|---|" + "---:|" * (len(hdr) - 1)]
    for s in GREY_SHAPES:
        pg, pt = points_json(f"{s}_grey_8v"), points_json(f"{s}_tex_8v")
        mg, mt = seed_mean(f"{s}_grey_8v"), seed_mean(f"{s}_tex_8v")
        g = lambda d, c, nd: fmt(d.get(c) if d else None, nd)
        L.append(f"| {name(s)} | {g(pg, 'F1@0.02', 3)} | {g(pt, 'F1@0.02', 3)} | {g(pg, 'CD', 2)} | {g(pt, 'CD', 2)} | "
                 f"{fmt(focal_ratio(f'{s}_grey_8v', s), 3)} | {fmt(focal_ratio(f'{s}_tex_8v', s), 3)} | "
                 f"{g(mg, 'CD', 2)} | {g(mt, 'CD', 2)} | {g(mg, 'F1@0.02', 3)} | {g(mt, 'F1@0.02', 3)} |")
    L += [""] + grey_tex_line()

    L += ["", "## Step 5: clean test set, 12 new Thingi10K shapes (textured, 8 views, seed 42)", "", "### Mesh", ""]
    L += table([(s, mesh_json(f"{s}_tex_8v")) for s in SET_B], MESH_COLS, groups_b, [("CD floor", floor_col)])
    L += ["", "### Points stage", ""]
    L += table([(s, points_json(f"{s}_tex_8v")) for s in SET_B], PTS_COLS, groups_b)
    L += ["", "### Upper bound (MeshFlow on clean GT samples)", ""]
    L += table([(s, mesh_json(f"{s}_gtsamples")) for s in SET_B], MESH_COLS, groups_b)

    both = [(s, mesh_json(f"{s}_tex_8v")) for s in SET_A + SET_B]
    L += ["", "## All non-dev shapes (set A + set B), pipeline mesh, seed 42", "",
          "| n | " + " | ".join(c for c, _ in MESH_COLS) + " |", "|---:|" + "---:|" * len(MESH_COLS),
          f"| {sum(1 for _, d in both if d)} | " + " | ".join(stats([d.get(c) for _, d in both if d], nd) for c, nd in MESH_COLS) + " |"]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"[report] -> {out}")


if __name__ == "__main__":
    main()
