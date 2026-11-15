#!/usr/bin/env bash
# Start Shroud (the real server, real models). Open http://localhost:8000
cd "$(dirname "$0")/.."
exec .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port "${PORT:-8000}"
