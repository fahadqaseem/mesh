#!/bin/bash
# Task C (18 Nov): rerun VGGT (8 default views) per shape to recover the filtered depth maps, silhouettes and cameras the
# multi-view outlier filters need. One shape at a time under scripts/mem_guard.py (Mac memory rules); each raw.npz is
# reduced at once to outputs/<s>_craw/points/views.npz (~20 MB, scripts/save_depth_views.py) and deleted.
cd "$(dirname "$0")/.."
for s in pencilpot_299291 cylinder_472094 shackbot_94155 nsphere_186199 wheel_76903 presents_98021 monsterring_1356640 cryptex_105686 $(sed 's/^/thingi_/' configs/step5_shapes.txt); do
  [ -f outputs/${s}_craw/points/views.npz ] && { echo "$s done already"; continue; }
  mkdir -p outputs/${s}_craw/points
  .venv/bin/python scripts/mem_guard.py --log outputs/${s}_craw/points/mem_guard.json -- \
    .venv/bin/python scripts/run_points.py --images data/scenes/$s/images_tex --run ${s}_craw --keep-raw > outputs/${s}_craw/run.log 2>&1
  rc=$?
  [ $rc = 0 ] && .venv/bin/python scripts/save_depth_views.py --run ${s}_craw --scene data/scenes/$s | tail -1
  rm -f outputs/${s}_craw/points/points.ply outputs/${s}_craw/points/tsdf_mesh.ply outputs/${s}_craw/points/raw.npz
  echo "$s rc=$rc"
  [ $rc = 99 ] && { echo ABORTED; exit 99; }
done
echo CRAW_DONE
