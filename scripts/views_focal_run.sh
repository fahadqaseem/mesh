#!/bin/bash
# 19 Nov block, set B, one VGGT job at a time under scripts/mem_guard.py:
#  (1) <s>_v12r3: 12 views over three rings, 4 per ring (data/scenes_low/<s>, ids 0,2,4,6 = 20 deg az 0/90/180/270;
#      9,11,13,15 = 50 deg az 67.5/157.5/247.5/337.5; 17,19,21,23 = -10 deg az 45/135/225/315), default TSDF, 1M points.
#  (2) joint GT-free fit of depth scale and focal: default VGGT rerun with raw.npz kept (<s>_ds), scripts/depth_scale.py
#      --joint (eps 0.005, chosen on pencilpot), fused as <s>_jf; plus <s>_vfocal (VGGT's own predicted focal, GT-free
#      reference). raw.npz deleted afterwards.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for s in $(sed 's/^/thingi_/' configs/step5_shapes.txt); do
  r=${s}_v12r3
  if [ ! -f outputs/$r/align/points_aligned.ply ]; then
    mkdir -p outputs/$r/points
    $PY scripts/mem_guard.py --log outputs/$r/points/mem_guard.json -- \
      $PY scripts/run_points.py --images data/scenes_low/$s/images_tex --run $r --view-ids 0,2,4,6,9,11,13,15,17,19,21,23 > outputs/$r/run.log 2>&1
    rc=$?
    [ $rc = 99 ] && { echo "$r ABORTED (memory)"; exit 99; }
    [ $rc = 0 ] && { $PY scripts/align.py --run $r --scene data/scenes_low/$s | tail -1; rm -f outputs/$r/points/tsdf_mesh.ply outputs/$r/points/points.ply; } || echo "$r rc=$rc"
  fi
  if [ ! -f outputs/${s}_vfocal/align/points_aligned.ply ]; then
    $PY scripts/mem_guard.py --log outputs/${s}_ds/points/mem_guard_jf.json -- \
      $PY scripts/run_points.py --images data/scenes/$s/images_tex --run ${s}_ds --keep-raw > outputs/${s}_ds/run_jf.log 2>&1
    rc=$?
    [ $rc = 99 ] && { echo "${s}_ds ABORTED (memory)"; rm -f outputs/${s}_ds/points/raw.npz; exit 99; }
    [ $rc != 0 ] && { echo "${s}_ds rc=$rc"; continue; }
    $PY scripts/depth_scale.py --run ${s}_ds --scene data/scenes/$s --joint | tail -1
    f=outputs/${s}_ds/points/depth_scale_joint.json
    $PY scripts/tsdf_points.py --images data/scenes/$s/images_tex --raw-from ${s}_ds --run ${s}_jf --depth-scale $f --focal-scale $f > /dev/null
    $PY scripts/tsdf_points.py --images data/scenes/$s/images_tex --raw-from ${s}_ds --run ${s}_vfocal --set intrinsics=predicted > /dev/null
    for x in jf vfocal; do $PY scripts/align.py --run ${s}_$x --scene data/scenes/$s > /dev/null
      rm -f outputs/${s}_$x/points/tsdf_mesh.ply outputs/${s}_$x/points/points.ply; done
    rm -f outputs/${s}_ds/points/raw.npz outputs/${s}_ds/points/tsdf_mesh.ply outputs/${s}_ds/points/points.ply
  fi
  echo "$s ok"
done
echo VF_DONE
