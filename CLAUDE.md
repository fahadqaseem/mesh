# CLAUDE.md: photo2mesh

Read this first in every session. It is the project brief for Claude Code.

## The project

**Goal:** an end-to-end pipeline from ordinary multi-view **photos** to a **lightweight, high-fidelity mesh** that carries both **geometry and texture (colour)**, ending in an **anisotropic** mesh (dense where the shape needs it, sparse elsewhere) for accuracy-critical uses: simulation, CAD, 3D printing, autonomous driving.

- **Owner:** Fahad Qaseem Khawar (Wayne State University)
- **Advisor:** Prof. Zichun Zhong (Wayne State, geometric modeling and neural 3D)
- **Weekly meeting:** Fridays, 1 pm

## Pipeline

```
multi-view photos
  → [points]  VGGT (or COLMAP + 3DGS): cameras, depth, dense point cloud
  → [bridge]  filter + normals + colour: clean dense points with oriented normals and RGB
  → [mesh]    swappable mesher:
                • CVPR 2026 MeshFlow (now: validates the pipeline; artist-style mesh)
                • HD-PEA, Hongbo Li / Zhong lab (later: anisotropic, high fidelity)
  → [loop]    render the mesh back with colour, compare with the input photos (open problem)
```

## Decisions so far (from meetings with Prof. Zhong)

1. **Priority 1:** get the pipeline working end to end with **VGGT + CVPR MeshFlow** (both have public code). Small data first (Thingi10K renders), then scale up.
2. Use the **CVPR 2026 MeshFlow** (facebookresearch/meshflow; point cloud in). Not the SIGGRAPH 2026 MeshFlow (starts from noise, no input condition).
3. **The image-to-points stage is not our contribution.** Use the state of the art (VGGT). The research is in the mesh: anisotropic, high fidelity, geometry + texture.
4. **HD-PEA** (input: 6 values per point, xyz + normal) is plugged in once its code is shared. It is not public yet.
5. Try **Gaussian centres vs interpolated points** as the point input, and compare.
6. **Priority 2:** a closed loop that renders the mesh back **with colour** and compares with the photos.
7. **Priority 3 (after January):** a flow-matching / generative model for the anisotropic mesh stage.
8. Keep components **modular and swappable** (the "chair" argument), but be ready to defend why two components beat one unified model (reviewers will ask). Ablations of each stage support this.

## Repo map

- `src/photo2mesh/points/`: VGGT wrapper, point filtering, normal estimation
- `src/photo2mesh/mesh/`: `Mesher` interface plus adapters (`meshflow_cvpr.py`, `hdpea.py` stub)
- `src/photo2mesh/eval/`: metrics (Chamfer, normal consistency, F1, vertex count)
- `src/photo2mesh/io.py`: PLY read/write with normals and colours
- `scripts/`: command-line steps (render views, points, mesh, evaluate, full pipeline)
- `configs/default.yaml`: all tunable settings
- `docs/`: `experiments.md` (log every run), `decisions.md`, `related_work.md`
- `third_party/`: external repos cloned here (gitignored; never edit them)

## Conventions

- **Compute:**
  - Light runs on the MacBook (device auto-selects CUDA, then MPS, then CPU).
  - Heavy runs go on the lab GPU workstation later.
  - Never hard-code `cuda`.
- **Never modify `third_party/`.** Wrap external code in adapters under `src/photo2mesh/mesh/` or `points/`.
- **Every stage reads and writes files** (`.ply`, `.npz`, `.json`) under `outputs/<run_name>/<stage>/`, so any stage can be swapped or rerun alone.
- **Log every experiment** in `docs/experiments.md`: date, command, config, data, result numbers, one-line takeaway.
- **Same metrics as HD-PEA:** Chamfer distance, normal consistency, F1, vertex count, runtime. That way results compare directly with the lab's numbers.
- **Keep it simple:** small functions, type hints, docstrings on public functions, tests for anything numeric (`pytest`).
- Don't commit data, checkpoints or outputs (see `.gitignore`).

## Current task

See `docs/experiments.md` → "Next up". Update it at the end of every session.
