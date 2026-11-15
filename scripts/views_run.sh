#!/bin/bash
# One guarded VGGT run with more views, then align + score. Usage: scripts/views_run.sh <shape> <suffix> <view_ids|all> [img_size]
# Writes outputs/<shape><suffix>/points/mem_guard.json. Exit 99 = aborted by the memory guard (memory pressure red).
set -u
S=$1; SUF=$2; IDS=$3; SIZE=${4:-518}; RUN=${S}${SUF}; PY=.venv/bin/python
mkdir -p outputs/$RUN/points
$PY scripts/mem_guard.py --log outputs/$RUN/points/mem_guard.json -- \
    $PY scripts/run_points.py --images data/scenes/$S/images_tex --run $RUN --view-ids "$IDS" --set img_size=$SIZE || exit $?
$PY scripts/align.py --run $RUN --scene data/scenes/$S >/dev/null && \
$PY scripts/points_eval.py --run $RUN --scene data/scenes/$S && \
$PY scripts/cloud_stats.py --run-dir outputs/$RUN --scene data/scenes/$S >/dev/null
