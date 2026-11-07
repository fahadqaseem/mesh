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

## Working rules: git and logging (read every session)

Fahad does strategy in a separate claude.ai chat that reads this folder directly. Keep the files below
current so that chat always sees the real state without being told.

**Git**
- Commit after every meaningful step (a fix, a new script, a finished run, a log update). Small commits, clear messages.
- Push to origin main after each commit (or at least at the end of every work block). Never leave a session with unpushed work.
- Never commit data/, outputs/, ckpt/, third_party/, .venv/ or any weights (already in .gitignore). Check `git status` before committing.
- Never force push or rewrite history.

**Logs**
- docs/experiments.md: one entry per run (template inside), newest on top. Keep "Next up" accurate.
- docs/status.md: a short snapshot, rewritten (not appended) at the end of every work block:
  date, what works, what is blocked and why, current numbers, next 3 steps, open questions for Fahad or Prof. Zhong.
- docs/decisions.md: add a row whenever a direction is chosen (by Fahad, Prof. Zhong, or a result that forces it).
- If a strategy question comes up that Fahad should decide, write it under "Open questions" in status.md and
  carry on with the safest option instead of stopping.

## Mac memory rules (added 2026-11-07 after a kernel panic)

The MacBook (M4 Pro, 24 GB unified memory, disk ~96% full) froze and restarted at about 00:15 on 2026-11-07:
a watchdog timeout from memory and swap exhaustion. VGGT on 16 views peaked at ~19 GB, and the 4.5 GB MeshFlow
checkpoint was loaded on top of that. Until heavy work moves to the lab GPU:
- Run ONE heavy job at a time (VGGT, MeshFlow, ICP on millions of points). Never run things in parallel or in the background alongside another heavy job.
- VGGT on the Mac: at most 8 views at a time, or process views in chunks. Free memory between scenes (separate processes, one per scene).
- Downsample before saving points (voxel ~0.005 in unit scale, or cap at ~200k points). 2M-point clouds are wasteful; MeshFlow resamples anyway.
- Load MeshFlow in one process per run and exit afterwards; do not keep CPU and MPS copies alive together.
- Before a heavy run, check free memory (`vm_stat` / `memory_pressure`). If pressure is high, stop and say so instead of starting.
- Delete raw.npz and other large intermediates that can be regenerated; keep outputs/ small.
- Anything that needs more than this goes to the lab GPU. Note it in status.md under "Blocked".
