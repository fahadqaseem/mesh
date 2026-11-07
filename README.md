# photo2mesh

Multi-view photos → point cloud (VGGT) → lightweight, high-fidelity mesh (CVPR 2026 MeshFlow now, HD-PEA later).
PhD project with Prof. Zichun Zhong, Wayne State University.

```
photos ──VGGT──▶ points + normals + colours ──mesher──▶ mesh ──▶ metrics (Chamfer, NC, F1, #verts)
                 (scripts/run_points.py)     (scripts/run_mesh.py)   (scripts/evaluate.py)
```

## Setup (Mac, first time)

```bash
cd "/Users/fahad/Downloads/To keep/2026/11 Nov/mesh"
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # should pass with no GPU

# heavy parts (VGGT + MeshFlow)
pip install -e ".[gpu,render]"
git clone https://github.com/facebookresearch/vggt.git third_party/vggt
pip install -e third_party/vggt
git clone https://github.com/facebookresearch/meshflow.git third_party/meshflow
pip install diffusers einops omegaconf   # MeshFlow's own requirements.txt pins CUDA torch 2.8; skip it on the Mac
# weights are gated: request access at huggingface.co/facebook/meshflow first, then
hf auth login
hf download facebook/meshflow --local-dir ckpt/meshflow   # files land in ckpt/meshflow/meshflow/
```

VGGT weights download automatically the first time (`facebook/VGGT-1B`, about 5 GB).

On the Mac, code runs on MPS (Apple GPU) or CPU automatically. It will be slow; that is fine for
small tests. Big runs go to the lab GPU workstation (same commands, it picks CUDA by itself).

## Run

```bash
# 1. make a test scene from a known mesh
python scripts/render_views.py --mesh data/gt/bunny.obj --out data/scenes/bunny/images --n 12
# 2. whole pipeline
python scripts/run_pipeline.py --images data/scenes/bunny/images --run bunny_v1 --gt data/gt/bunny.obj
```

Outputs land in `outputs/<run>/{points,mesh,eval}/`.

## Layout

```
src/photo2mesh/
  points/   VGGT wrapper, filtering, normals
  mesh/     swappable meshers (meshflow_cvpr, hdpea)
  eval/     metrics
  io.py     PLY / mesh read-write
scripts/    one script per stage + run_pipeline.py
configs/    default.yaml
docs/       experiments.md (log), decisions.md, related_work.md
.claude/    slash commands: /weekly-update, /log-run
```

## Working with Claude Code

Open the folder in VS Code, start Claude Code in the terminal (`claude`). It reads `CLAUDE.md`
first, so it knows the goal, the decisions and the conventions. Useful:
- `/log-run bunny_v1` after a run
- `/weekly-update` on Thursday before the Friday meeting

## Moving to the lab GPU

```bash
git clone git@github.com:<you>/photo2mesh.git && cd photo2mesh
# repeat Setup (venv, third_party, ckpt); data/ and outputs/ are not in git, copy with rsync if needed
```
