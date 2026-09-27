#!/bin/bash
# Runs the sidecar and the UI in the browser, without building the app.
#   scripts/start_browser.sh          use the real models
#   scripts/start_browser.sh --demo   demo mode, no models needed
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${DIARIZATION_PYTHON:-$HOME/Library/Application Support/DualTrackDiarization/venv/bin/python}"

if [[ "${1:-}" == "--demo" ]]; then
  export DIARIZATION_MOCK=1
fi

if [[ ! -x "$PY" ]]; then
  echo "找不到 Python 環境，請先執行 scripts/setup_mac.sh"
  exit 1
fi

(cd "$REPO/python-sidecar" && "$PY" main.py) &
SIDECAR_PID=$!
trap 'kill $SIDECAR_PID 2>/dev/null' EXIT

cd "$REPO"
(sleep 3 && open "http://127.0.0.1:1420" 2>/dev/null || true) &
npm run dev
