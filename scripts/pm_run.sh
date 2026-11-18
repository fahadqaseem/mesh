#!/bin/bash
# Prof. Zhong item 1 (17 Nov email): VGGT point map vs unprojected depth vs TSDF, set A (7) + set B (12).
# Per shape, one heavy job at a time: VGGT on the 8 default views under scripts/mem_guard.py with raw.npz kept (<s>_pm;
# its TSDF cloud is checked against <s>_dense), then scripts/pointmap_clouds.py writes <s>_pmap (point head) and
# <s>_dunp (known-focal unprojected depth, same pixels), all three aligned with the camera alignment of protocol v1.
# raw.npz, TSDF meshes and unaligned clouds are deleted afterwards.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SHAPES="cylinder_472094 shackbot_94155 nsphere_186199 wheel_76903 presents_98021 monsterring_1356640 cryptex_105686 $(sed 's/^/thingi_/' configs/step5_shapes.txt | tr '\n' ' ')"
for s in $SHAPES; do
  [ -f outputs/${s}_dunp/align/points_aligned.ply ] && { echo "$s done already"; continue; }
  mkdir -p outputs/${s}_pm/points
  $PY scripts/mem_guard.py --log outputs/${s}_pm/points/mem_guard.json -- \
    $PY scripts/run_points.py --images data/scenes/$s/images_tex --run ${s}_pm --keep-raw > outputs/${s}_pm/run.log 2>&1
  rc=$?
  [ $rc = 99 ] && { echo "$s ABORTED (memory)"; rm -f outputs/${s}_pm/points/raw.npz; exit 99; }
  [ $rc != 0 ] && { echo "$s VGGT rc=$rc"; continue; }
  $PY scripts/pointmap_clouds.py --run ${s}_pm --scene data/scenes/$s
  for x in pm pmap dunp; do $PY scripts/align.py --run ${s}_$x --scene data/scenes/$s > /dev/null; done
  a=$(md5 -q outputs/${s}_pm/align/points_aligned.ply); b=$(md5 -q outputs/${s}_dense/align/points_aligned.ply)
  [ "$a" = "$b" ] && echo "$s TSDF identical to _dense" || echo "$s TSDF DIFFERS from _dense"
  rm -f outputs/${s}_pm/points/raw.npz outputs/${s}_pm/points/tsdf_mesh.ply outputs/${s}_pm/points/points.ply \
        outputs/${s}_pm/align/points_aligned.ply outputs/${s}_pmap/points/points.ply outputs/${s}_dunp/points/points.ply
  echo "$s ok"
done
echo PM_DONE
