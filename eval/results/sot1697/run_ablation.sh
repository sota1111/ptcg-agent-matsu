#!/usr/bin/env bash
# SOT-1697 ablation launcher: candidate configs vs champion, 25-deck mirror.
set -u
cd /workspaces/ptcg-agent-matsu
PY=venv/bin/python
BASE=eval/results/sot1697
NSHARDS=${1:-2}          # shards per candidate (each = 25 matches, seat s%2)
declare -A CAND=(
  [depth2]='{"max_tree_depth": 2}'
  [worlds6]='{"n_worlds": 6}'
  [roots8]='{"max_root_actions": 8}'
  [depth2_roots8]='{"max_tree_depth": 2, "max_root_actions": 8}'
  [worlds6_depth2]='{"n_worlds": 6, "max_tree_depth": 2}'
  [selfcal]='{}'
)
for name in "${!CAND[@]}"; do
  mkdir -p "$BASE/$name"
  for ((s=0; s<NSHARDS; s++)); do
    $PY -u eval/bench_configs.py --match-index "$s" \
        --candidate "${CAND[$name]}" \
        --json "$BASE/$name/shard_$s.json" \
        > "$BASE/$name/shard_$s.log" 2>&1 &
  done
done
wait
echo "ABLATION_DONE"
