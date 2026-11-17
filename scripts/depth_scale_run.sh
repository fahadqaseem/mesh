#!/bin/bash
# Depth scale correction (18 Nov) on set B: per shape, rerun VGGT (8 default views) under scripts/mem_guard.py with
# raw.npz kept, align the default cloud (<s>_ds, identical to <s>_dense), fit the GT-free depth scale
# (scripts/depth_scale.py, eps 0.005 chosen on pencilpot), fuse with the global (<s>_dsg) and per-view (<s>_dsv) scale,
# align, then delete raw.npz and the TSDF meshes. One heavy job at a time.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for s in $(sed 's/^/thingi_/' configs/step5_shapes.txt); do
  [ -f outputs/${s}_dsv/align/points_aligned.ply ] && { echo "$s done already"; continue; }
  mkdir -p outputs/${s}_ds/points
  $PY scripts/mem_guard.py --log outputs/${s}_ds/points/mem_guard.json -- \
    $PY scripts/run_points.py --images data/scenes/$s/images_tex --run ${s}_ds --keep-raw > outputs/${s}_ds/run.log 2>&1
  rc=$?
  [ $rc = 99 ] && { echo "$s ABORTED (memory)"; rm -f outputs/${s}_ds/points/raw.npz; exit 99; }
  [ $rc != 0 ] && { echo "$s VGGT rc=$rc"; continue; }
  $PY scripts/align.py --run ${s}_ds --scene data/scenes/$s > /dev/null
  $PY scripts/depth_scale.py --run ${s}_ds --scene data/scenes/$s | tail -1
  for m in "dsg global" "dsv view"; do set -- $m
    $PY scripts/tsdf_points.py --images data/scenes/$s/images_tex --raw-from ${s}_ds --run ${s}_$1 \
      --depth-scale outputs/${s}_ds/points/depth_scale_$2.json > /dev/null
    $PY scripts/align.py --run ${s}_$1 --scene data/scenes/$s > /dev/null
    rm -f outputs/${s}_$1/points/tsdf_mesh.ply outputs/${s}_$1/points/points.ply
  done
  rm -f outputs/${s}_ds/points/raw.npz outputs/${s}_ds/points/tsdf_mesh.ply outputs/${s}_ds/points/points.ply
  echo "$s ok"
done
echo DS_DONE
