# SOT-1704 — root-level deck-out guard evaluation

## Decision

Keep the current champion configuration. The proposed root-level pure-draw
guard remains available behind `PlannerConfig.deck_guard_threshold`, whose
default is `0` (disabled), but it is not enabled in `main.py`.

The two small-N screens both lost to the champion, so neither candidate was
eligible for the N=400 promotion gate. This follows the specified two-stage
rule: run the full gate only for a promising screen candidate, and require a
Wilson 95% lower bound above 0.5 for promotion.

## Fresh loss trace against current 竹 and 梅

Before implementing the guard, `eval/loss_trace_matsu.py` was rerun with 48
matches per opponent using eight independent seed shards. The aggregate is
stored in `eval/results/sot1704/baseline_loss_trace.json`.

| Opponent | Record | Deck-out losses | Other losses | Faults |
| --- | ---: | ---: | ---: | ---: |
| 竹 (`take`) | 39–9 | 6 / 9 (66.7%) | no-active 1, prize-race 2 | 0 |
| 梅 (`ume`) | 42–6 | 4 / 6 (66.7%) | prize-race 2 | 0 |

All ten deck-out losses reported `search_healthy=true`, and all records had
zero budget violations, planner fallbacks, and engine faults. Deck-out
therefore remained the dominant measured loss cause against both updated
opponents.

## Guard design

At root candidate enumeration, when the player's remaining deck is at or
below the configured threshold, PLAY and ABILITY options classified from the
card master's effect text as pure draw are removed before applying
`max_root_actions`. Determinization, tree search, evaluation, and
`deviate_margin` are unchanged. Direct lethal attacks are retained, and if
filtering would remove every candidate, the original ordering is restored.

Unit coverage includes the threshold boundary, lethal preservation, and the
all-filtered fallback. With the default threshold of zero, existing champion
behavior is unchanged.

## Small-N screens

Each candidate played 50 mirror matches against the champion across all 25
decks with seat order swapped. Seeds were independently sharded; every screen
recorded zero rejects, exceptions, unfinished games, fallbacks, planner
fallbacks, degraded decisions, and budget violations.

| Threshold | Candidate record | Win rate | Wilson 95% interval | Decision |
| ---: | ---: | ---: | ---: | --- |
| 4 | 22–28 | 0.44 | [0.3116, 0.5769] | reject |
| 6 | 20–30 | 0.40 | [0.2761, 0.5382] | reject |

Aggregates are in
`eval/results/sot1704/screen_t4_complete/final.json` and
`eval/results/sot1704/screen_t6_complete/final.json`. Since neither screen
beat the champion even at point estimate, no candidate advanced to N=400.
`CHAMPION_CONFIG` was not changed.
