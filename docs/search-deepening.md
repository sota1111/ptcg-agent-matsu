# Search deepening, speedup & loss analysis (SOT-1697)

Goal: push the 25-deck mirror win rate of the champion Determinized-MCTS agent
higher within the same 0.8 s/decision budget, by (1) analysing *why* matsu loses
to 竹/梅, (2) speeding up the playout so the same budget buys more search, (3)
ablating a deeper/wider tree, and (4) reviewing the determinization. Paper style
(Determinized MCTS, arXiv:1808.04794) is preserved throughout — the engine stays
the only source of legality/transitions and the search is unchanged in kind.

Champion baseline (`main.CHAMPION_CONFIG`, unchanged unless a candidate clears
the gate): `max_tree_depth=1, n_worlds=4, max_root_actions=6, deviate_margin=0.1,
rollout_turns=100, rollout_depth=200, time_budget_s=0.8`.

## 1. Playout speedup (agents/observation.py `adapt_engine_obs`)

Profiling one champion mirror decision (`cProfile`, `eval/profile_rollout.py`)
showed two ~equal hot paths, each ~40 % of a decision:

1. **`_rollout_action` → `dataclasses.asdict()` round-trip.** The greedy rollout
   policy converted the *entire* engine dataclass observation (both boards, all
   cards/energy) back into a raw dict on every rollout step, only so
   `GreedyAgent` could score it. This is agent-side and fully under our control.
2. **`search_step` → `json_to_dataclass`** — the engine's own JSON→dataclass
   deserialisation inside `cg/` (license-restricted, not shippable; left as is).

Fix: `agents.observation.adapt_engine_obs` builds the information-set `View`
straight from the engine dataclass via attribute access, skipping the recursive
dict rebuild. It is **behavior-identical** — `tests/test_observation.py`
asserts the same `View` structure and the same `GreedyAgent` choice/scores as the
old asdict path, and the micro-benchmark below confirms 0 mismatched choices
over hundreds of real positions.

Measured before/after (`eval/results/sot1697/profile.json`, 12 real positions):

| metric | before (asdict) | after (`adapt_engine_obs`) | speedup |
| --- | --- | --- | --- |
| greedy-choice throughput | 18.6 k/s | 91.9 k/s | **4.94×** |
| MCTS iterations per 0.8 s decision (mean) | 197.8 | 296.7 | **1.50×** |
| mismatched greedy choices | — | 0 / 720 | identical |

So the same 0.8 s budget now completes ~50 % more determinized rollouts end-to-
end (the residual is the engine-side `to_dataclass` cost we can't ship changes
to). Both sides of every A/B bench below share this speedup, so the config
ablation isolates the config, not the speedup.

## 2. Loss-cause analysis vs 竹/梅 (`eval/loss_trace_matsu.py`)

Champion mirror-random vs the latest 竹/梅 agents, N=48/opponent, traced and
classified (`eval/results/sot1697/loss_trace_agg.json`):

| opponent | N | matsu wins | win rate (Wilson95) | loss causes | 立ち上がり事故 | 探索unhealthy |
| --- | --- | --- | --- | --- | --- | --- |
| 竹 (take) | 48 | 33 | 0.688 [0.547, 0.801] | deck_out 8 · prize_race 5 · no_active 2 | 1 | 0 |
| 梅 (ume) | 48 | 37 | 0.771 [0.635, 0.867] | deck_out 10 · prize_race 1 | 0 | 0 |

**Key finding: self-deck-out is the dominant defeat** — 8/15 = 53 % of losses vs
竹 and 10/11 = 91 % vs 梅. Every deck-out loss had `search_healthy=true` (no
degradation / budget violation / greedy hand-off), so the search is *executing*
fine; the loss is an *evaluation* blind spot. The champion evaluator rewards
`hand` (drawing) with no counter-pressure until the terminal `deck_empty` cliff
at deckCount==0, so long determinized lines mill matsu's own deck. This directly
motivated the `deck_low` gradient candidate in §3 (a smooth pre-cliff penalty).
prize-race losses (5 vs 竹) were all `search_healthy` too and turn-mean ~28 — a
genuine board-state disadvantage, not a search fault, so out of scope here.

## 3. Depth / worlds / roots ablation + champion gate (`eval/bench_configs.py`)

Method (SOT-1673 style, 25 diverse decks = multiple matchup axes): each
candidate config plays the champion config head-to-head on all 25 tournament
decks, mirror + seat-alternating, both at 0.8 s. Promotion gate: candidate is
adopted **only if the Wilson-95 lower bound of its win rate vs champion > 0.5**;
a straddle keeps the champion. Health invariants (engine rejects / exceptions /
budget violations 時間切れ / planner degradations) must all be 0.

**Screening (candidate vs champion, both seats × 25 decks, 0.8 s each).** First a
cheap small-N screen of each lever; only a promising candidate earns the full
N≥400 gate. Deeper/wider-tree levers were screened via `run_screen.sh`; the
loss-analysis-driven `deck_low` gradient via `run_screen2.sh`:

| candidate | lever | N (screen) | cand wins | screen win rate | verdict |
| --- | --- | --- | --- | --- | --- |
| depth2 | `max_tree_depth=2` | 27 | 15 | ~0.56 | wide CI (±0.19); no edge |
| roots8 | `max_root_actions=8` | 26 | 16 | ~0.62 | wide CI; not gate-clearing |
| worlds6 | `n_worlds=6` | 25 | 11 | ~0.44 | candidate behind |
| deck_low(-0.3@8) | `eval_weights` | 50 | 25 | 0.500 [0.366, 0.634] | straddles 0.5 |

None of the deeper/wider-tree levers produced a screen edge worth the ~hours of
compute a full N≥400 gate costs at 0.8 s/decision (past lesson: 松の大N A/B は
実行時間的に非現実的). The `deck_low` gradient — the only lever with a
mechanistic reason to help from §2 — was the least-bad and was promoted to the
full gate.

**Final gate (N=400, `eval/results/sot1697/final_decklow2/final.json`).**
Candidate `deck_low=-0.2, deck_low_at=14` vs champion, 25 decks × both seats:

| metric | value |
| --- | --- |
| N (matches, no draws) | 400 |
| candidate wins / champion wins | 199 / 201 |
| candidate win rate | **0.4975** |
| Wilson-95 CI | **[0.4488, 0.5463]** |
| promotion gate (CI lower > 0.5) | **not met → champion maintained** |
| faults (rejects/exceptions/unfinished/fallbacks/budget/degraded) | **0** |
| move time over 400 matches (mean / max) | 641.2 ms / 653.4 ms (< 800 ms → 時間切れ 0) |

The CI straddles 0.5, so the champion is kept. Per-deck the candidate is within
noise on 24/25 decks (no deck regressed hard, none improved significantly),
confirming the deck-out penalty neither helped nor hurt at this magnitude.

## 4. Determinization review (`agents/planner.py:sample_fills`)

The paper-faithful refinement the issue describes — excluding the opponent's
*observed public* cards (used/discarded cards, board Pokémon + attached
energy/tools/pre-evolutions, face-up prizes, stadium ownership) from the world
sampling pool — is **already implemented** in `_visible_ids`/`_pool_minus`
(SOT-1672): every determinization subtracts exactly those visible IDs from the
candidate pool before sampling the hidden zones. No change is warranted; adding
speculative opponent-hand inference would be an unvalidated model change and is
explicitly out of scope for this issue.

## 5. Conclusion

**Champion maintained** (`main.CHAMPION_CONFIG` unchanged). Of the four work
items:

1. **Speedup shipped** — the `adapt_engine_obs` fast path (§1) is behavior-
   identical (0/720 mismatched greedy choices, asserted in `tests/`) and buys
   ~1.5× determinized rollouts inside the same 0.8 s budget. It ships because it
   strictly improves the champion at equal behaviour; both A/B sides used it, so
   the config ablation isolates the config, not the speedup.
2. **Loss analysis shipped** — the deck-out blind spot (§2) is the concrete,
   件数付き finding the issue asked for.
3. **No config promoted** — neither the deeper/wider tree nor the deck-out
   `deck_low` gradient cleared the CI gate (best fully-gated candidate 0.4975,
   CI [0.449, 0.546]); the `deck_low`/`deck_low_at` weights therefore default to
   0 and are dormant infrastructure, keeping shipped champion behaviour byte-for-
   byte identical.
4. **Determinization already paper-faithful** (§4) — no change warranted.

Net PR content: the behaviour-identical playout speedup, the loss-analysis and
ablation harnesses + results, and this report. No champion behaviour change,
fault 0, 時間切れ 0.
