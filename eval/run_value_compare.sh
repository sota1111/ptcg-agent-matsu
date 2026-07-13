#!/usr/bin/env bash
# SOT-1674 value-function comparison driver — heuristic vs learned evaluator.
#
# Arms (all other planner parameters = SOT-1672 adopted config):
#   H-V  : MCTS + HeuristicEvaluator, value-reliant config (rollout_turns=2 —
#          rollouts stop at the second turn boundary and the LEAF EVALUATOR
#          scores the state, so the value function actually drives play)
#   L-V  : MCTS + LearnedEvaluator, same value-reliant config
#   BASE : SOT-1672 adopted 基準構成 (rollout_turns=100 — rollouts usually
#          reach a terminal, the evaluator is almost never consulted)
#   L-B  : LearnedEvaluator inside the adopted 基準構成 (regression check)
#
# Pairings: each arm vs Random / Greedy / BASE, plus the direct L-V vs H-V
# head-to-head. N matches per pairing = SHARDS * N_PER (default 10*20 = 200),
# side-alternating, agent seeds derived per shard (engine RNG not seedable,
# ASSUMPTIONS.md A-9). Reproduce any single shard with the one-liner printed
# into eval/results/value_compare/commands.txt.
#
# Usage (from the repo root, ~15 min on 24 cores):
#   bash eval/run_value_compare.sh [SHARDS=10] [N_PER=20] [JOBS=20]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python
OUT=eval/results/value_compare
SHARDS=${SHARDS:-10}
N_PER=${N_PER:-20}
JOBS=${JOBS:-20}
mkdir -p "$OUT"

COMMON='"max_root_actions":6,"max_tree_depth":1,"n_worlds":4,"time_budget_s":0.8,"deviate_margin":0.1'
HV='{"evaluator":"heuristic","rollout_turns":2,"rollout_depth":60,'$COMMON'}'
LV='{"evaluator":"learned","rollout_turns":2,"rollout_depth":60,'$COMMON'}'
BASE='{"evaluator":"heuristic","rollout_turns":100,"rollout_depth":200,'$COMMON'}'
LB='{"evaluator":"learned","rollout_turns":100,"rollout_depth":200,'$COMMON'}'

# name  agent_a  config_a  agent_b  config_b  seed_base
PAIRINGS=(
  "hv_vs_random|mcts|$HV|random||6174100"
  "lv_vs_random|mcts|$LV|random||6174200"
  "hv_vs_greedy|mcts|$HV|greedy||6174300"
  "lv_vs_greedy|mcts|$LV|greedy||6174400"
  "lv_vs_hv|mcts|$LV|mcts|$HV|6174500"
  "hv_vs_base|mcts|$HV|mcts|$BASE|6174600"
  "lv_vs_base|mcts|$LV|mcts|$BASE|6174700"
  "lbase_vs_greedy|mcts|$LB|greedy||6174800"
)

CMDS="$OUT/commands.txt"
: > "$CMDS"
for pairing in "${PAIRINGS[@]}"; do
  IFS='|' read -r name a ca b cb seed_base <<< "$pairing"
  for s in $(seq 1 "$SHARDS"); do
    cmd="$PY eval/bench.py --agent-a $a --agent-b $b --n $N_PER --seed $((seed_base + s))"
    [ -n "$ca" ] && cmd="$cmd --config-a '$ca'"
    [ -n "$cb" ] && cmd="$cmd --config-b '$cb'"
    cmd="$cmd --json $OUT/${name}_shard_${s}.json > $OUT/${name}_shard_${s}.log 2>&1"
    echo "$cmd" >> "$CMDS"
  done
done

echo "running $(wc -l < "$CMDS") shards with $JOBS parallel jobs..."
# -d '\n' disables xargs quote processing (the JSON configs contain quotes)
xargs -d '\n' -P "$JOBS" -I{} bash -c '{}' < "$CMDS"

echo
for pairing in "${PAIRINGS[@]}"; do
  IFS='|' read -r name _ <<< "$pairing"
  "$PY" eval/aggregate_shards.py "$OUT/${name}.json" "$OUT/${name}"_shard_*.json
done
