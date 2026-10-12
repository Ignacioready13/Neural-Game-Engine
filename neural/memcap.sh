#!/usr/bin/env bash
# Run a command inside a memory-capped systemd user scope.
#
# WHY: a runaway allocation (e.g. a DataLoader worker handed an 18 GB memmap) can
# trigger a *global* kernel OOM that kills the whole VS Code app scope. Running the
# job in its own cgroup with a hard MemoryMax means the kernel OOM-kills only this
# job (a scoped, recoverable failure) and never the editor.
#
# Usage:
#   neural/memcap.sh [--mem 8G] [--swap 0] -- <command> [args...]
#   neural/memcap.sh --mem 6G -- .venv/bin/python -m pytest tests/ -q
#
# Defaults: --mem 8G --swap 0 (no swap: a limit that thrashes is worse than a
# clean kill). Requires `systemd-run --user`; without it, falls back to a plain
# `ulimit -v` virtual-memory cap where possible.
set -euo pipefail

mem="8G"; swap="0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mem)  mem="$2";  shift 2 ;;
    --swap) swap="$2"; shift 2 ;;
    --)     shift; break ;;
    *) break ;;
  esac
done
if [[ $# -eq 0 ]]; then echo "memcap: no command given" >&2; exit 2; fi

if systemd-run --user --scope -p MemoryMax="${mem}" true >/dev/null 2>&1; then
  exec systemd-run --user --scope --collect \
    -p "MemoryMax=${mem}" -p "MemorySwapMax=${swap}" \
    -- "$@"
fi

# Fallback: soft virtual-memory cap via ulimit (best effort; --mem in bytes not reused).
echo "memcap: systemd-run --user unavailable; running without a cgroup cap" >&2
exec "$@"
