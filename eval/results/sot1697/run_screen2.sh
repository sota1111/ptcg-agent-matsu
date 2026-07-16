#!/usr/bin/env bash
# SOT-1697 candidate-vs-champion screening, finely sliced for 24-core parallel.
# Both seats x 25 decks (N=50) split into deck slices of 3 so each process is a
# ~5-6 min chunk that fits a foreground window. Usage: run_screen2.sh <name> <cfgjson>
set -u
cd /workspaces/ptcg-agent-matsu
PY=venv/bin/python
BASE=eval/results/sot1697
name="$1"; cfg="$2"
outdir="$BASE/scr_$name"
mkdir -p "$outdir"
pids=()
for seat in 0 1; do
  for off in 0 3 6 9 12 15 18 21; do
    lim=3; [ "$off" -ge 21 ] && lim=4   # 21..24 = 4 decks (covers all 25)
    $PY -u eval/bench_configs.py --match-index "$seat" \
        --deck-offset "$off" --deck-limit "$lim" \
        --candidate "$cfg" \
        --json "$outdir/s${seat}_o${off}.json" \
        > "$outdir/s${seat}_o${off}.log" 2>&1 &
    pids+=($!)
  done
done
for p in "${pids[@]}"; do wait "$p"; done
echo "SCREEN2_DONE $name"
