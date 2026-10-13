#!/usr/bin/env bash
# One obvious way to start the player.
#
#   ./play.sh                      newest registered model, windowed
#   ./play.sh --model 5a           pick a model by name
#   ./play.sh --list-models        what is registered
#   ./play.sh --help               every flag
#
# It runs the player inside memcap.sh so a runaway allocation (e.g. memory-mapping
# the 18 GB packed set when --packed is used) can only OOM-kill this job, never the
# editor scope. Add --no-memcap as the first argument to skip that wrapper.
set -euo pipefail
cd "$(dirname "$0")"

usememcap=1
if [[ "${1:-}" == "--no-memcap" ]]; then
  usememcap=0
  shift
fi

if [[ $usememcap -eq 1 && -x ./memcap.sh ]]; then
  exec ./memcap.sh --mem "${PLAY_MEM:-12G}" -- .venv/bin/python play.py "$@"
fi
exec .venv/bin/python play.py "$@"
