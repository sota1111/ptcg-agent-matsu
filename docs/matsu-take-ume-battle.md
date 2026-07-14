# 松竹梅 cross-repo battle (SOT-1681)

Round-robin battle between the three sibling projects' **Kaggle submission
agents** (each `main.agent` playing its own `deck.csv`) on the shared cabt
engine. Harness: [`eval/battle_matsu_take_ume.py`](../eval/battle_matsu_take_ume.py)
(+ [`eval/agent_server.py`](../eval/agent_server.py)); helpers unit-tested in
`tests/test_battle_matsu_take_ume.py`.

```bash
venv/bin/python eval/battle_matsu_take_ume.py --n 500 --json /tmp/sot1681.json
```

## What was battled — implementation state confirmed first

The issue asks to confirm all implementations are complete before battling. Each
repo was at its latest merged `main` with a clean tree; the contestant is that
project's **actual submission entry point** (`main.agent`):

| 銘 | repo | submission agent (`main.py`) | deck |
| --- | --- | --- | --- |
| 松 matsu | `ptcg-agent-matsu` | `GreedyAgent` (SOT-1671 baseline) | `deck.csv` |
| 竹 take  | `ptcg-agent-take`  | `RuleBasedAgent` (scoring policy) | `deck.csv` |
| 梅 ume   | `ptcg-agent-ume`   | random legal move (baseline) | `deck.csv` |

> Note: matsu and ume still ship **baseline** submissions — matsu's advanced
> stack (MCTS / turn-solver / learned value, SOT-1672/1674/1677) and ume's
> `RuleAgent`/R4 exist in their `agents/` packages but are **not wired into
> `main.py`**. This battle reflects what each project currently ships. Re-point
> those `main.py` entries and re-run to compare best-vs-best.

## Results — N=500 matches per pairing (1500 total), 0 faults

Each pairing is played seat-alternating (先後入替); win rate excludes draws;
Wilson 95% CI.

| pairing | wins | win rate (row vs col) | 95% CI | 先手勝率 |
| --- | --- | --- | --- | --- |
| matsu vs take | 282 – 218 | **matsu 0.564** | [0.520, 0.607] | 0.648 |
| matsu vs ume  | 458 – 42  | **matsu 0.916** | [0.888, 0.937] | 0.774 |
| take vs ume   | 439 – 61  | **take 0.878**  | [0.846, 0.904] | 0.726 |

### Standings

| rank | 銘 | W–L | win rate | 95% CI |
| --- | --- | --- | --- | --- |
| 1 | 松 matsu | 740–260 | **0.740** | [0.712, 0.766] |
| 2 | 竹 take  | 657–343 | **0.657** | [0.627, 0.686] |
| 3 | 梅 ume   | 103–897 | **0.103** | [0.086, 0.123] |

**松 > 竹 > 梅**, and every head-to-head CI clears 0.5 — the ordering is
statistically decisive at N=500. matsu's `GreedyAgent` beats take's tuned
`RuleBasedAgent` (0.564, CI lower bound 0.520 > 0.5); both crush ume's random
baseline. Zero faults across all 1500 matches (no illegal moves, agent
exceptions, or dead servers) — all three submissions are functional.

**先手 (first-player) advantage is large and consistent** (0.65–0.77 across
pairings), which is why the harness swaps seats every match.

## Method notes

- **Isolation.** The three repos' `agents` packages have colliding module names
  (`base`, `random_agent`, `search_agent`), so they cannot co-exist in one
  interpreter. Each contestant runs in its own subprocess (its repo/venv) behind
  a line-delimited JSON protocol; the host owns only this repo's `cg.game`
  (a process-global single battle, matches run sequentially).
- **Agent-bound decks.** `battle_start(seat0.deck, seat1.deck)` binds each deck
  to its agent; seat-alternation swaps agent+deck together, so it is an exact
  先後 swap, not a deck swap.
- **Reproducibility.** The cabt engine has no seed API (ASSUMPTIONS A-9), so
  outcomes vary run-to-run; conclusions are statistical (CI separation), not
  bit-exact. Faults (illegal move / agent exception / dead server) are charged
  to the offending agent as a loss and the batch continues.
