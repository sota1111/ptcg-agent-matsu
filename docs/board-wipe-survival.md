# Board-wipe KPI and smooth survival candidate (SOT-1883)

## What changed

The config-vs-config A/B artifact now classifies each completed loss as
`board_wipe` when the losing terminal board has neither an Active nor a Bench
Pokémon. It records the following separately for candidate and champion:

- board-wipe count;
- board-wipe rate among that side's losses;
- board-wipe avoidance rate over completed matches.

The classifier uses only terminal board structure. Prize, deck-out, effects,
and unknown terminal paths remain `other`, avoiding card-name heuristics.

The evaluated candidate replaces SOT-1878's binary wipe cliff with
`board_survival: 1.0`. The feature uses the opponent's strongest
energy-payable attack from Active or Bench, the defender's remaining HP, and
the number of surviving Active/Bench replacement routes. A bounded HP-margin
tie-break makes it continuous while keeping the calculation local to the leaf
evaluator.

The feature is disabled by default (`board_survival: 0.0`), so the champion is
unchanged unless the candidate clears every promotion gate.

## Small-N screen

Candidate configuration:

```json
{
  "eval_weights": {
    "deck_low": -0.22,
    "deck_low_at": 12,
    "deck_low_prize_gate": 3,
    "board_survival": 1.0
  }
}
```

The screen used one match on each of the frozen 25-deck rotation baseline,
seed `20260723`.

| Metric | Candidate | Champion |
| --- | ---: | ---: |
| Wins | 15 | 10 |
| Win rate | 0.6000 | 0.4000 |
| Wilson 95% CI (candidate) | [0.4074, 0.7660] | — |
| board_wipe count | 1 | 0 |
| board_wipe rate in losses | 0.1000 | 0.0000 |
| board_wipe avoidance rate | 0.9600 | 1.0000 |
| sims/sec | 308.4 | 180.8 |
| faults | 0 combined | 0 combined |

Promotion requires all of: candidate Wilson 95% lower bound greater than
0.5, fault total 0, and candidate/champion sims/sec ratio at least 0.9. The
candidate passed the fault and throughput gates (ratio 1.706), but its Wilson
lower bound was 0.4074. The small-N screen therefore failed.

Machine-readable evidence:

- `eval/results/sot1883/screen/final.json`
- `eval/results/sot1883/screen/shard_0.json` through `shard_4.json`

## Large-N, champion, and Kaggle decision

Large-N confirmation was intentionally not run because the small-N screen did
not pass every promotion gate. The candidate was not applied to
`main.CHAMPION_CONFIG`; therefore no Kaggle resubmission was permitted or
needed.

The one-match smoke artifacts under `eval/results/sot1883/smoke/` validate the
artifact schema separately and are not promotion evidence.
