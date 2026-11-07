#!/bin/zsh
# Run the VGGT points stage on several scenes, one at a time (VGGT needs ~19 GB at 16 views on the Mac).
# Usage: scripts/vggt_batch.sh <scene>:<variant> ...     e.g. pencilpot_299291:tex wheel_76903:tex
#        scripts/vggt_batch.sh                           (all scenes in data/scenes, textured)
# Appends one line per run to outputs/vggt_batch.tsv
cd "${0:A:h}/.."
runs=("$@")
if (( ${#runs} == 0 )); then for d in data/scenes/*(/); do runs+=("${d:t}:tex"); done; fi
mkdir -p outputs
for r in $runs; do
  n="${r%%:*}"; v="${r##*:}"; run="${n}_${v}_v1"
  mkdir -p "outputs/$run"
  /usr/bin/time -l .venv/bin/python scripts/run_points.py --images "data/scenes/$n/images_$v" --run "$run" > "outputs/$run/points.log" 2>&1
  rc=$?
  pts=$(grep -o '\[points\] [0-9,]*' "outputs/$run/points.log" | awk '{print $2}')
  secs=$(awk '/ real /{print $1}' "outputs/$run/points.log")
  peak=$(awk '/peak memory footprint/{printf "%.1f", $1/1e9}' "outputs/$run/points.log")
  printf '%s\trc=%s\tpoints=%s\tsecs=%s\tpeakGB=%s\n' "$run" "$rc" "$pts" "$secs" "$peak" | tee -a outputs/vggt_batch.tsv
done
echo DONE
