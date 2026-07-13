#!/bin/bash
# Pack a Kaggle submission: main.py + deck.csv + agents/ + cg/ at the archive top level.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
[ -d cg ] && [ -f main.py ] && [ -f deck.csv ] && [ -d agents ] || { echo "missing cg/ or main.py or deck.csv or agents/ (run setup_engine.sh)"; exit 1; }
tar -czf submission.tar.gz main.py deck.csv agents cg
echo "wrote $REPO/submission.tar.gz"; tar -tzf submission.tar.gz | head
