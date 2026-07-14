#!/usr/bin/env bash
# SOT-1679 mctsV benchmark driver — value network + rollout Early Cutoff.
#
# Arms (all other planner parameters = SOT-1673 adopted 基準構成):
#   BASE  : 基準mcts — HeuristicEvaluator, rollouts run to terminal
#           (rollout_turns=100/depth=200; the paper's plain "mcts")
#   MCTSV : value_net + Early Cutoff (rollout_cutoff min_steps=20 — rollouts
#           play ≥20 steps, stop at the next turn boundary, and the network's
#           win probability replaces the rollout result; paper §III-B4/IV-B)
#   LEAFV : value_net as a leaf replacement (rollout_turns=2, SOT-1674's
#           value-reliant construction) — the integration-method A/B arm
#
# Pairings: mctsv_vs_base (headline, paper: mctsV 82.0% vs mcts),
# leafv_vs_base (A/B arm), mctsv_vs_leafv (direct A/B). N per pairing =
# SHARDS * N_PER (default 50*10 = 500), side-alternating, agent seeds derived
# per shard (engine RNG not seedable, ASSUMPTIONS.md A-9). Reproduce any
# shard with the one-liner appended to eval/results/value_net/commands.txt.
#
# Usage (from the repo root; slices can be run separately and re-aggregated):
#   PAIRINGS_FILTER=mctsv_vs_base SHARD_FROM=1 SHARD_TO=25 \
#     bash eval/run_value_net.sh
#   bash eval/run_value_net.sh            # everything missing, then aggregate
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python
OUT=eval/results/value_net
SHARDS=${SHARDS:-50}
N_PER=${N_PER:-10}
JOBS=${JOBS:-20}
SHARD_FROM=${SHARD_FROM:-1}
SHARD_TO=${SHARD_TO:-$SHARDS}
PAIRINGS_FILTER=${PAIRINGS_FILTER:-}
mkdir -p "$OUT"

COMMON='"max_root_actions":6,"max_tree_depth":1,"n_worlds":4,"time_budget_s":0.8,"deviate_margin":0.1'
BASE='{"evaluator":"heuristic","rollout_turns":100,"rollout_depth":200,'$COMMON'}'
MCTSV='{"evaluator":"value_net","rollout_cutoff":{"min_steps":20},"rollout_turns":100,"rollout_depth":200,'$COMMON'}'
LEAFV='{"evaluator":"value_net","rollout_turns":2,"rollout_depth":60,'$COMMON'}'

# name  agent_a  config_a  agent_b  config_b  seed_base
PAIRINGS=(
  "mctsv_vs_base|mcts|$MCTSV|mcts|$BASE|6179100"
  "leafv_vs_base|mcts|$LEAFV|mcts|$BASE|6179200"
  "mctsv_vs_leafv|mcts|$MCTSV|mcts|$LEAFV|6179300"
)

CMDS=$(mktemp)
for pairing in "${PAIRINGS[@]}"; do
  IFS='|' read -r name a ca b cb seed_base <<< "$pairing"
  [ -n "$PAIRINGS_FILTER" ] && [ "$name" != "$PAIRINGS_FILTER" ] && continue
  for s in $(seq "$SHARD_FROM" "$SHARD_TO"); do
    [ -s "$OUT/${name}_shard_${s}.json" ] && continue  # already done
    cmd="$PY eval/bench.py --agent-a $a --agent-b $b --n $N_PER --seed $((seed_base + s))"
    [ -n "$ca" ] && cmd="$cmd --config-a '$ca'"
    [ -n "$cb" ] && cmd="$cmd --config-b '$cb'"
    cmd="$cmd --json $OUT/${name}_shard_${s}.json > $OUT/${name}_shard_${s}.log 2>&1"
    echo "$cmd" >> "$CMDS"
    echo "$cmd" >> "$OUT/commands.txt"
  done
done

echo "running $(wc -l < "$CMDS") shards with $JOBS parallel jobs..."
# -d '\n' disables xargs quote processing (the JSON configs contain quotes)
[ -s "$CMDS" ] && xargs -d '\n' -P "$JOBS" -I{} bash -c '{}' < "$CMDS"
rm -f "$CMDS"

echo
for pairing in "${PAIRINGS[@]}"; do
  IFS='|' read -r name _ <<< "$pairing"
  shards=("$OUT/${name}"_shard_*.json)
  [ -e "${shards[0]}" ] || continue
  "$PY" eval/aggregate_shards.py "$OUT/${name}.json" "${shards[@]}"
done
