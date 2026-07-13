#!/usr/bin/env bash
# Verification entry point (SOT-1671): forbidden-term lint + syntax + tests.
# Run from the repo root. Uses venv/bin/python when present (engine tests
# need it locally); falls back to python3 (engine tests self-skip).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=python3
[ -x venv/bin/python ] && PY=venv/bin/python

echo "== forbidden-term lint =="
"$PY" scripts/lint_hardcoded_cards.py

echo "== syntax check =="
"$PY" -m compileall -q agents eval scripts tests train

echo "== unit tests =="
"$PY" -m unittest discover -s tests -t . -v

echo "ALL CHECKS PASSED"
