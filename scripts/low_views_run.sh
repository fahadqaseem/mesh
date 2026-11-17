#!/bin/bash
# Low views (18 Nov): render a third ring at -10 deg elevation (8 views, same azimuths as the 20 deg ring) next to the
# usual 20 / 50 deg rings into data/scenes_low/<s> (24 views; ids 0-7 = 20 deg, 8-15 = 50 deg, 16-23 = -10 deg; the
# first 16 are the same renders as data/scenes/<s>), then run VGGT on two 8-view sets that include low views:
#   <s>_lowA  4 at 20 deg (az 0/90/180/270) + 4 at -10 deg (az 45/135/225/315), i.e. the 50 deg ring swapped for -10 deg
#   <s>_lowB  2 at 50 deg (az 67.5/247.5) + 4 at 20 deg + 2 at -10 deg (az 135/315)
# each fused to 1M TSDF points (default settings) and aligned. One VGGT run at a time under scripts/mem_guard.py.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for s in $(sed 's/^/thingi_/' configs/step5_shapes.txt); do
  d=data/scenes_low/$s
  [ -f $d/images_tex/view_023.png ] || $PY scripts/render_views.py --mesh data/gt/$s.stl --out $d/images_tex \
    --elevations 20,50,-10 --texture procedural --seed 0 | tail -1
  for v in "lowA 0,2,4,6,17,19,21,23" "lowB 9,13,0,2,4,6,19,23"; do set -- $v; r=${s}_$1
    [ -f outputs/$r/align/points_aligned.ply ] && continue
    mkdir -p outputs/$r/points
    $PY scripts/mem_guard.py --log outputs/$r/points/mem_guard.json -- \
      $PY scripts/run_points.py --images $d/images_tex --run $r --view-ids $2 > outputs/$r/run.log 2>&1
    rc=$?
    [ $rc = 99 ] && { echo "$r ABORTED (memory)"; exit 99; }
    [ $rc != 0 ] && { echo "$r VGGT rc=$rc"; continue; }
    $PY scripts/align.py --run $r --scene $d | tail -1
    rm -f outputs/$r/points/tsdf_mesh.ply outputs/$r/points/points.ply
  done
  echo "$s ok"
done
echo LOW_DONE
