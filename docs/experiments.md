# Experiment log

One entry per run. Newest on top. Claude: append here after every run, and keep "Next up" current.

## Next up
1. [ ] Set up the env on the Mac (README "Setup"), run `pytest`.
2. [ ] Download a handful of small Thingi10K meshes into `data/gt/` (5 to 10 simple shapes).
3. [ ] Render 12 views of each with `scripts/render_views.py`.
4. [ ] Run VGGT on one scene, open `points.ply` in MeshLab, check normals point outward.
5. [ ] Run CVPR MeshFlow on that `points.ply` (CPU/MPS may be slow; move to lab GPU if so).
6. [ ] Add alignment (ICP, or use the GT render cameras) so `evaluate.py` scores are meaningful.
7. [ ] First table: Chamfer / NC / F1 / #verts for 5 shapes. Bring it to Friday's meeting.
8. [ ] Experiment: Gaussian centres vs interpolated points as MeshFlow input.

## Template
```
### YYYY-MM-DD  <run name>
Goal:
Input: (scene, #views)
Config: (diff from default.yaml)
Result: chamfer=  nc=  f1=  verts=  faces=  time=
Notes / what I saw:
Next:
```

## Runs
(none yet)
