# Next-turn board-wipe risk (SOT-1878)

## Implementation

`HeuristicEvaluator` can assign a configurable penalty when every visible
Pokémon on one side is within the opponent's highest reachable attack damage.
The estimate uses remaining HP, energy-payable attacks from both Active and
Bench attackers, and the presence of a known surviving switch target. Hidden
Pokémon make the estimate conservative: they prevent a definite board-wipe
classification.

The feature is disabled by default (`board_wipe: 0.0`). The tested candidate
used `board_wipe: -2.0` in addition to every weight in the current champion
profile, so the A/B comparison isolated this feature.

## Reproduction coverage

`tests/test_value.py::TestNextTurnBoardWipeRisk` covers:

- a replay-shaped exposed board versus a board with an 80 HP survivor;
- attacks that cannot be paid with attached Energy;
- a charged Bench attacker as a possible switch response;
- exact preservation of the champion value while the weight is disabled.

## Promotion evidence

| Stage | N | Candidate–champion | Wilson 95% CI | Faults | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| small-N screen | 25 | 10–15 (40.0%) | [0.2340, 0.5926] | 0 | reject |
| large-N confirm | — | not run | — | — | screen did not pass |

The fixed promotion gate is Wilson 95% lower bound **> 0.5**, fault total 0,
and no throughput regression. The candidate failed the statistical gate, so
large-N confirmation was intentionally skipped.

A matched three-deck throughput probe recorded 107.3 candidate simulations/s
versus 358.2 champion simulations/s (ratio 0.300), with fault total 0 and no
budget violations. This independently fails the no-regression requirement.
The detailed machine-readable artifacts are:

- `eval/results/sot1878/screen/final.json`
- `eval/results/sot1878/throughput/final.json`

## Promotion and Kaggle decision

The champion profile remains unchanged because the candidate failed both the
Wilson and throughput gates. Consequently there is no champion update and,
per the issue's conditional requirement, no Kaggle resubmission.
