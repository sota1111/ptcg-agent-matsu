# Context scoring prior / rollout (SOT-1748)

`PlannerConfig.context_scorer=True` enables the Take-derived context bands for
root and in-tree PUCT priors. `PlannerConfig.rollout="context"` uses the same
fast scorer during rollout. Both switches default to their previous values
(`False` and `"greedy"`), so the champion MCTS remains exactly reproducible.

The adapter works only from Matsu's information-set `View`; the engine remains
the source of legal options. Unknown option types keep the existing GreedyAgent
score, and MCTS/agent fallback paths are unchanged.

## Holdout A/B

Command (real cabt engine, alternating seats):

```bash
venv/bin/python eval/bench.py --agent-a mcts --agent-b mcts --n 12 \
  --seed 1748001 \
  --config-a '{"n_worlds":1,"time_budget_s":0.05,"max_iterations":40,"context_scorer":true,"rollout":"context"}' \
  --config-b '{"n_worlds":1,"time_budget_s":0.05,"max_iterations":40,"context_scorer":false,"rollout":"greedy"}' \
  --json eval/results/sot1748/holdout_context_vs_baseline.json
```

Context won 6/12: win rate 0.500, Wilson 95% CI [0.254, 0.746]. There were
zero illegal-action rejects, exceptions, fallbacks, degraded searches, and
budget violations. Maximum planner move latency was 41.0 ms under the shared
50 ms budget; mean decision latency was 28.1 ms. This small holdout establishes
non-regression, not superiority; a larger league is needed before champion
promotion.
