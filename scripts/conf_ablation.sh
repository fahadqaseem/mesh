#!/bin/bash
# Confidence ablation (17 Nov): rerun VGGT (8 default views, default settings) per set B shape, keep raw.npz, fuse the
# same depth maps with conf_percentile 50 (= the default, <s>_c50 from run_points.py), 30, 10, and 10 without the 2 px
# foreground erosion (scripts/tsdf_points.py), align each, score the pixels (scripts/conf_pixels.py), then delete
# raw.npz and the TSDF meshes. VGGT runs one shape at a time under scripts/mem_guard.py (Mac memory rules).
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for s in $(sed 's/^/thingi_/' configs/step5_shapes.txt); do
  [ -f outputs/${s}_c50/eval/conf_pixels.json ] && { echo "$s done already"; continue; }
  mkdir -p outputs/${s}_c50/points
  $PY scripts/mem_guard.py --log outputs/${s}_c50/points/mem_guard.json -- \
    $PY scripts/run_points.py --images data/scenes/$s/images_tex --run ${s}_c50 --keep-raw > outputs/${s}_c50/run.log 2>&1
  rc=$?
  [ $rc = 99 ] && { echo "$s ABORTED (memory)"; rm -f outputs/${s}_c50/points/raw.npz; exit 99; }
  [ $rc != 0 ] && { echo "$s VGGT rc=$rc"; continue; }
  $PY scripts/align.py --run ${s}_c50 --scene data/scenes/$s | tail -1
  for v in "c30 conf_percentile=30" "c10 conf_percentile=10" "c10e0 conf_percentile=10 fg_erode=0"; do
    set -- $v; name=$1; shift
    $PY scripts/tsdf_points.py --images data/scenes/$s/images_tex --raw-from ${s}_c50 --run ${s}_$name --set "$@" | tail -1
    $PY scripts/align.py --run ${s}_$name --scene data/scenes/$s | tail -1
    rm -f outputs/${s}_$name/points/tsdf_mesh.ply outputs/${s}_$name/points/points.ply
  done
  $PY scripts/conf_pixels.py --run ${s}_c50 --scene data/scenes/$s | tail -1
  rm -f outputs/${s}_c50/points/raw.npz outputs/${s}_c50/points/tsdf_mesh.ply outputs/${s}_c50/points/points.ply
  echo "$s ok"
done
echo CONF_DONE
