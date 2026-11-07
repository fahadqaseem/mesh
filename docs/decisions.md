# Decisions

Things agreed with Prof. Zhong, with the date. Add new ones at the bottom.

| Date | Decision | Why |
|---|---|---|
| 2026-11-05 | Priority 1: VGGT + CVPR MeshFlow end to end, small Thingi10K first | Get a working baseline and numbers before anything new |
| 2026-11-05 | Image to points is not the contribution; the mesh is | The novelty is anisotropic, lightweight meshes with geometry + texture |
| 2026-11-05 | Try Gaussian centres vs interpolated points as input | See which point source gives the mesher better input |
| 2026-11-05 | Priority 2: colour render-back loop (render mesh, compare to photos) | Adds texture fidelity and a self-check signal |
| 2026-11-05 | Priority 3 (after January): generative / flow model | Only after the modular pipeline works |
| 2026-11-05 | Keep the pipeline modular (swap the mesher); be ready to defend two stages vs a unified model | Reviewers will compare to LATO, MeshCraft |
| 2026-11-05 | HD-PEA plugs in as a second mesher once Hongbo shares code | It needs xyz + normals, which stage 1 already writes |
| 2026-11-05 | Weekly meetings Fridays 1 pm from Nov 13; lab GPU workstation later | Heavy runs move to the lab |
