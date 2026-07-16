#!/usr/bin/env bash
# SOT-1697 final N=400 gate for the best-screened candidate (decklow2:
# eval_weights deck_low=-0.2 @14) vs champion. Reuses the 50 match-index {0,1}
# matches already in scr_decklow2/ and adds match-index 2..15 (=350 matches) so
# the aggregate over BOTH dirs is 16 matches/deck x 25 decks = 400.
# Resumable: a slice whose JSON already exists is skipped, so repeated bounded
# runs converge. Fine 3-deck slices keep every task well under a few minutes.
set -u
cd /workspaces/ptcg-agent-matsu
export PY=venv/bin/python
export CFG='{"eval_weights": {"deck_low": -0.2, "deck_low_at": 14}}'
export OUTDIR=eval/results/sot1697/final_decklow2
mkdir -p "$OUTDIR"
PAR=${1:-22}
tasks=()
for s in $(seq 2 15); do
  for off in 0 3 6 9 12 15 18 21 24; do
    lim=3; [ "$off" -eq 24 ] && lim=1
    out="$OUTDIR/s${s}_o${off}.json"
    [ -s "$out" ] && continue
    tasks+=("$s $off $lim")
  done
done
echo "remaining tasks: ${#tasks[@]}"
[ "${#tasks[@]}" -eq 0 ] && { echo "FINAL2_ALL_DONE"; exit 0; }
run_one() {
  read s off lim <<< "$1"
  "$PY" -u eval/bench_configs.py --match-index "$s" \
    --deck-offset "$off" --deck-limit "$lim" \
    --candidate "$CFG" \
    --json "$OUTDIR/s${s}_o${off}.json" \
    > "$OUTDIR/s${s}_o${off}.log" 2>&1
}
export -f run_one
printf '%s\n' "${tasks[@]}" | xargs -P "$PAR" -I{} bash -c 'run_one "$@"' _ {}
echo "FINAL2_BATCH_DONE remaining_before=${#tasks[@]}"
