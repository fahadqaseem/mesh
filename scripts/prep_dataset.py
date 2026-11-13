"""Make test scenes (multi-view renders + GT) from ShapeNetCore v2 or Toys4K, like the Thingi10K scenes.

Usage:
    python scripts/prep_dataset.py --dataset shapenet --root data/shapenet_raw [--n 3] [--seed 0] [--list-only]
    python scripts/prep_dataset.py --dataset toys4k --root data/toys4k_raw/toys4k_obj_files [--n 3] [--seed 0]
Picks n models at random (seeded; one per category first, so the pick spans categories), writes the list to
configs/<dataset>_shapes.txt and renders each one with scripts/render_views.py (one process per shape) into
data/scenes/<dataset>_<category>_<id>/images_tex: 16 views (two rings of 8, 20° and 50°), 518 px, white background,
plus gt_normalized.ply, gt_cam_poses.npy and gt_transform.json (same rig and frame conventions as Thingi10K).

Layouts and access (neither dataset is downloaded by this script; both are gated):
- ShapeNetCore v2: <root>/<synset>/<model_id>/models/model_normalized.obj (+ .mtl, ../images/*). Y-up already
  (--up y). Rendered with its own materials and UV textures (--texture original). Access: accept the terms at
  https://huggingface.co/datasets/ShapeNet/ShapeNetCore (per-synset zips), then
  `hf download ShapeNet/ShapeNetCore <synset>.zip --repo-type dataset --local-dir data/shapenet_raw` and unzip.
- Toys4K: OBJ files without materials (toys4k_obj_files.zip, after the access form at
  https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k). Any *.obj under <root> counts; the category is the
  first folder below <root> and the id the file's parent folder. Blender OBJ exports are Y-up (--up y). No
  materials, so the procedural texture is used (--texture procedural), as for Thingi10K.
Both are messier than our Thingi10K picks (not watertight, inner faces, several parts): the script prints faces,
parts and watertightness per model so bad picks can be swapped. Light: pyrender only (no VGGT, no MeshFlow).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh


def find_models(dataset: str, root: Path) -> list[tuple[str, str, Path]]:
    """(category, model id, path) for every model under root, sorted."""
    out = []
    if dataset == "shapenet":
        for p in sorted(root.glob("*/*/models/model_normalized.obj")):
            out.append((p.parents[2].name, p.parents[1].name, p))
    else:
        for p in sorted(root.rglob("*.obj")):
            rel = p.relative_to(root).parts
            out.append((rel[0] if len(rel) > 1 else "misc", p.parent.name if len(rel) > 1 else p.stem, p))
    return out


def pick(models: list[tuple[str, str, Path]], n: int, seed: int) -> list[tuple[str, str, Path]]:
    """n models, seeded: shuffle the categories, take one random model from each in turn until n are chosen."""
    rng = np.random.default_rng(seed)
    by_cat: dict[str, list] = {}
    for m in models:
        by_cat.setdefault(m[0], []).append(m)
    cats = sorted(by_cat)
    rng.shuffle(cats)
    for c in cats:
        rng.shuffle(by_cat[c])
    chosen, i = [], 0
    while len(chosen) < min(n, len(models)):
        c = cats[i % len(cats)]
        if by_cat[c]:
            chosen.append(by_cat[c].pop())
        i += 1
    return chosen


def scene_name(dataset: str, cat: str, mid: str) -> str:
    safe = lambda s: re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")
    return f"{dataset}_{safe(cat)}_{safe(mid)[:16]}"


def describe(path: Path) -> str:
    m = trimesh.load(path, force="mesh")
    parts = len(trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(m.faces)), min_len=1))
    return f"{len(m.faces):,} faces, {parts} parts, watertight={m.is_watertight}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["shapenet", "toys4k"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list-only", action="store_true", help="write the list, do not render")
    ap.add_argument("--scenes", default="data/scenes")
    args = ap.parse_args()

    models = find_models(args.dataset, Path(args.root))
    if not models:
        raise SystemExit(f"no {args.dataset} models under {args.root} (see the layout in this script's docstring)")
    chosen = pick(models, args.n, args.seed)
    lst = Path(f"configs/{args.dataset}_shapes.txt")
    lst.write_text("".join(f"{scene_name(args.dataset, c, i)}\t{p}\n" for c, i, p in chosen))
    print(f"[prep] {len(models):,} models in {len({m[0] for m in models})} categories; picked {len(chosen)} -> {lst}")
    flags = ["--up", "y", "--texture", "original" if args.dataset == "shapenet" else "procedural"]
    for c, i, p in chosen:
        name = scene_name(args.dataset, c, i)
        print(f"[prep] {name}: {describe(p)}")
        if args.list_only:
            continue
        cmd = [sys.executable, "scripts/render_views.py", "--mesh", str(p),
               "--out", f"{args.scenes}/{name}/images_tex", *flags]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
