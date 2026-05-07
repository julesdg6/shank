#!/bin/sh
set -eu

APP_DIR="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "$APP_DIR"

python worker/worker_loop.py &
worker_pid=$!

cleanup() {
  kill "$worker_pid" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

uvicorn api.main:app --host 0.0.0.0 --port 8080
