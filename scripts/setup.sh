#!/usr/bin/env bash
# One-time setup: venv, dependencies, model weights, fonts. Everything is cached on disk; nothing is fetched at runtime.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3.11}
[ -d .venv ] || $PY -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/pip install -q --no-deps facenet-pytorch==2.6.0
.venv/bin/python scripts/fetch_assets.py
echo "setup done"
