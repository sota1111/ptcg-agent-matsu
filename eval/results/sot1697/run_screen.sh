#!/usr/bin/env bash
# SOT-1697 deck-parallel screening: candidate vs champion, seats {0,1} x 5 deck-slices.
set -u
cd /workspaces/ptcg-agent-matsu
PY=venv/bin/python
BASE=eval/results/sot1697
name="$1"; cfg="$2"
mkdir -p "$BASE/$name"
pids=()
for seat in 0 1; do
  for off in 0 5 10 15 20; do
    $PY -u eval/bench_configs.py --match-index "$seat" \
        --deck-offset "$off" --deck-limit 5 \
        --candidate "$cfg" \
        --json "$BASE/$name/s${seat}_o${off}.json" \
        > "$BASE/$name/s${seat}_o${off}.log" 2>&1 &
    pids+=($!)
  done
done
for p in "${pids[@]}"; do wait "$p"; done
echo "SCREEN_DONE $name"
