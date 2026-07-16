#!/usr/bin/env bash
# SOT-1697 final N=400 candidate-vs-champion bench (16 seat-shards x 25 decks),
# fanned out over deck-halves and throttled. Usage: run_final.sh <name> <cfgjson>
set -u
cd /workspaces/ptcg-agent-matsu
PY=venv/bin/python
BASE=eval/results/sot1697
name="$1"; cfg="$2"; PAR=${3:-22}
outdir="$BASE/final_$name"
mkdir -p "$outdir"
# Build task list: 16 match-indices x 2 deck-halves = 32 tasks (400 matches).
tasks=()
for s in $(seq 0 15); do
  for half in "0 13" "13 12"; do
    set -- $half; off=$1; lim=$2
    tasks+=("$s|$off|$lim")
  done
done
printf '%s\n' "${tasks[@]}" | xargs -P "$PAR" -I{} bash -c '
  t="{}"; IFS="|" read s off lim <<< "$t"
  '"$PY"' -u eval/bench_configs.py --match-index "$s" \
    --deck-offset "$off" --deck-limit "$lim" \
    --candidate '"'"$cfg"'"' \
    --json "'"$outdir"'/s${s}_o${off}.json" \
    > "'"$outdir"'/s${s}_o${off}.log" 2>&1
'
echo "FINAL_DONE $name"
