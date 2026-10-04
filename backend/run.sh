#!/usr/bin/env bash
# Starts the backend the same way every time, regardless of the caller's
# working directory or which name someone remembers the venv folder by.
#
# Usage:
#   ./run.sh         # emulator on this machine, or backend + Flutter on the same machine
#   ./run.sh --lan    # also reachable from a physical device on the LAN
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python=".venv/bin/python"
if [ ! -x "$python" ]; then
  echo ".venv not found at $python. Set it up first -- see README.md 'Backend Setup'." >&2
  exit 1
fi

if [ "${1:-}" = "--lan" ]; then
  exec "$python" -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
else
  exec "$python" -m uvicorn main:app --reload
fi
