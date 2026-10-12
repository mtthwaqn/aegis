#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps -e .
if [[ ! -e .env ]]; then
    cp .env.example .env
fi
printf '\nReady. Activate with: source .venv/bin/activate\n'
