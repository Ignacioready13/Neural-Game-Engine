#!/usr/bin/env bash
# Sequential night queue: wait for the GPU to be free, then run each job to
# completion, one at a time, each wrapped in the energy logger.  Written so the
# night's ordered plan is reproducible from one file (and so the GPU is never
# idle waiting on me, and never shared between two jobs).
#
# Usage:
#   neural/night_queue.sh job1 "run_id,purpose,notes,--command,arg,arg"
set -uo pipefail
cd "$(dirname "$0")"   # the neural/ directory
PY=.venv/bin/python
LOG=logs/queue.log

MIN_FREE_MIB=2500      # the display shares this 8 GB GPU; a job needs real headroom

wait_for_gpu() {
  # A job is safe to start when no other process holds a CUDA context *and* the
  # driver has actually given the memory back. Checking only the process list raced
  # the previous job's teardown once: `score_flow96d` started 2 s after training
  # ended, found 149 MB free of 8 GB, and died with a CUDA OOM on its first
  # allocation.
  local busy free_mib wait_s=0
  while true; do
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ "${busy:-1}" -eq 0 ] && [ "${free_mib:-0}" -ge "$MIN_FREE_MIB" ]; then
      [ "$wait_s" -gt 0 ] && sleep 20   # settle
      return 0
    fi
    echo "[queue $(date +%H:%M:%S)] GPU busy ($busy procs, ${free_mib} MiB free), waiting..." >> "$LOG"
    sleep 30
    wait_s=$((wait_s + 30))
  done
}

# Each argument after the first is "<run_id>|<purpose>|<notes>|<command...>"
for spec in "$@"; do
  IFS='|' read -r run_id purpose notes cmd <<< "$spec"
  wait_for_gpu
  echo "[queue $(date +%H:%M:%S)] starting $run_id: $cmd" >> "$LOG"
  # shellcheck disable=SC2086
  ./memcap.sh --mem 15G -- $PY powerlog.py --run-id "$run_id" --purpose "$purpose" \
      --notes "$notes" -- $cmd >> "logs/$run_id.log" 2>&1
  echo "[queue $(date +%H:%M:%S)] finished $run_id (exit $?)" >> "$LOG"
done
echo "[queue $(date +%H:%M:%S)] all jobs done" >> "$LOG"
