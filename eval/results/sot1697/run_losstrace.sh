#!/usr/bin/env bash
# SOT-1697 parallel loss-trace launcher: NSHARDS independent seeds, each runs
# matsu (in-process) vs take & ume for --n matches/opponent. Shards are seeded
# apart so their mirror deck schedules are independent samples we later merge
# with `loss_trace_matsu.py --aggregate`. Foreground: waits for all shards.
set -u
cd /workspaces/ptcg-agent-matsu
PY=venv/bin/python
BASE=eval/results/sot1697/loss
NSHARDS=${1:-16}
N=${2:-3}
mkdir -p "$BASE"
pids=()
for ((s=0; s<NSHARDS; s++)); do
  seed=$((20260716 + s * 1000))
  $PY -u eval/loss_trace_matsu.py --n "$N" --opponents take ume \
      --seed "$seed" --json "$BASE/shard_${s}.json" \
      > "$BASE/shard_${s}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "LOSSTRACE_DONE shards=$NSHARDS n=$N"
