#!/bin/bash
# Overnight queue runner (Mac memory rules): one job at a time, each in its own process.
#
# Usage (from the repo root):
#   .venv/bin/python scripts/make_overnight_queue.py
#   nohup caffeinate -is scripts/overnight.sh > outputs/overnight/runner.log 2>&1 &
# Other queues set QUEUE_DIR, REPORT, REPORT_CMD, REPORT_EXTRA (more paths to commit) and COMMIT_PREFIX, e.g. the
# robustness queue (see scripts/make_robustness_queue.py).
#
# Before every job: memory pressure and free disk. If free memory < 30% or free disk < 15 GB, wait 5 minutes and
# recheck; after 3 failed checks the job is skipped. A failed job is logged and the queue moves on. Jobs already
# marked ok in status.tsv are skipped, so the runner can be restarted safely. After the last job of each step the
# results doc (docs/results/2026-11-07_overnight.md) is regenerated, committed (that file only) and pushed.
# Logs: outputs/overnight/{runner.log, status.tsv, progress.txt, jobs/<job>.log}

cd "$(dirname "$0")/.." || exit 1
DIR=${QUEUE_DIR:-outputs/overnight}
QUEUE=$DIR/queue.tsv
STATUS=$DIR/status.tsv
REPORT=${REPORT:-docs/results/2026-11-07_overnight.md}
REPORT_CMD=${REPORT_CMD:-scripts/overnight_report.py}
REPORT_EXTRA=${REPORT_EXTRA:-}
COMMIT_PREFIX=${COMMIT_PREFIX:-Overnight results}
MIN_MEM_PCT=30
MIN_DISK_GB=15
WAIT_S=300
TRIES=3
JOB_TIMEOUT_S=1800
mkdir -p "$DIR/jobs"
touch "$STATUS"

log() { echo "$(date '+%F %T') $*"; }
mem_free_pct() { memory_pressure 2>/dev/null | awk '/free percentage/ {gsub("%", "", $NF); print $NF}'; }
disk_free_gb() { df -g /System/Volumes/Data | awk 'NR == 2 {print $4}'; }

resources_ok() {   # returns 0 when there is enough memory and disk; sets RES_INFO
  local mem disk
  mem=$(mem_free_pct); disk=$(disk_free_gb)
  RES_INFO="mem_free=${mem:-?}% disk_free=${disk:-?}GB"
  [ -n "$mem" ] && [ -n "$disk" ] && [ "$mem" -ge "$MIN_MEM_PCT" ] && [ "$disk" -ge "$MIN_DISK_GB" ]
}

run_with_timeout() {   # run "$1" in a new process with a time cap; output to $2
  bash -c "$1" < /dev/null > "$2" 2>&1 &
  local pid=$!
  # watchdog: kills the job after JOB_TIMEOUT_S; when stopped it also stops its own sleep (no orphans, no open pipes)
  ( trap 'kill "$s" 2>/dev/null; exit 0' TERM; sleep "$JOB_TIMEOUT_S" & s=$!; wait "$s"; kill -TERM "$pid" 2>/dev/null ) \
    < /dev/null > /dev/null 2>&1 &
  local watchdog=$!
  wait "$pid"
  local rc=$?
  kill "$watchdog" 2>/dev/null
  wait "$watchdog" 2>/dev/null
  return $rc
}

publish() {   # regenerate the results doc and push it (only that file)
  local step=$1
  log "publish after step $step"
  .venv/bin/python "$REPORT_CMD" >> "$DIR/report.log" 2>&1 || log "report failed (see report.log)"
  # shellcheck disable=SC2086
  git add "$REPORT" $REPORT_EXTRA >> "$DIR/git.log" 2>&1
  # shellcheck disable=SC2086
  if git commit -m "$COMMIT_PREFIX: step $step done ($(date '+%H:%M'))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "$REPORT" $REPORT_EXTRA >> "$DIR/git.log" 2>&1; then
    git push origin main >> "$DIR/git.log" 2>&1 || log "git push failed (see git.log)"
  else
    log "nothing to commit or commit failed (see git.log)"
  fi
}

total=$(wc -l < "$QUEUE" | tr -d ' ')
n=0
prev_step=""
log "start: $total jobs"
while IFS=$'\t' read -r -u 3 step jid cmd; do
  n=$((n + 1))
  if [ -n "$prev_step" ] && [ "$step" != "$prev_step" ]; then publish "$prev_step"; fi
  prev_step=$step
  echo "$(date '+%F %T') job $n/$total step $step: $jid" > "$DIR/progress.txt"
  if awk -F'\t' -v j="$jid" '$3 == j && $4 == "ok" {found = 1} END {exit !found}' "$STATUS"; then
    log "[$n/$total] $jid already ok, skipping"
    continue
  fi
  tries=0
  until resources_ok; do
    tries=$((tries + 1))
    log "[$n/$total] $jid waiting: $RES_INFO (check $tries/$TRIES)"
    if [ "$tries" -ge "$TRIES" ]; then break; fi
    sleep "$WAIT_S"
  done
  if [ "$tries" -ge "$TRIES" ]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$step" "$jid" "skipped" "$RES_INFO" >> "$STATUS"
    log "[$n/$total] $jid SKIPPED ($RES_INFO)"
    continue
  fi
  joblog="$DIR/jobs/$(echo "$jid" | tr '/' '__').log"
  start=$(date +%s)
  log "[$n/$total] $jid start ($RES_INFO)"
  run_with_timeout "$cmd" "$joblog"
  rc=$?
  secs=$(( $(date +%s) - start ))
  if [ "$rc" -eq 0 ]; then status=ok; else status="failed"; fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$step" "$jid" "$status" "rc=$rc secs=$secs $RES_INFO" >> "$STATUS"
  log "[$n/$total] $jid $status rc=$rc ${secs}s"
done 3< "$QUEUE"
[ -n "$prev_step" ] && publish "$prev_step"
echo "$(date '+%F %T') DONE ($total jobs)" > "$DIR/progress.txt"
log "done"
