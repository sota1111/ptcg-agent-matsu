# 松竹梅 cross-repo battle (SOT-1681)

Round-robin battle between the three sibling projects' **Kaggle submission
agents** (`main.agent`) on the shared cabt engine. Harness:
[`eval/battle_matsu_take_ume.py`](../eval/battle_matsu_take_ume.py)
(+ [`eval/agent_server.py`](../eval/agent_server.py)); helpers unit-tested in
`tests/test_battle_matsu_take_ume.py`.

```bash
# Primary: fair 25-deck mirror-random (isolates agent skill from deck luck)
venv/bin/python eval/battle_matsu_take_ume.py --n 8 \
    --decks-dir decks/initial --deck-mode mirror --seed 20260716001 \
    --json shard.json
# Aggregate independent shards into one decisive report
venv/bin/python eval/battle_matsu_take_ume.py --aggregate shard_*.json --json aggregate.json
```

## What was battled — latest submission agents

Each repo was at its latest merged `main` (松 PR#12 / 竹 PR#25 / 梅 PR#19, all
after the SOT-1692 強化) with a clean tree; the contestant is that project's
**actual submission entry point** (`main.agent`):

| 銘 | repo | submission agent (`main.py`) | notes |
| --- | --- | --- | --- |
| 松 matsu | `ptcg-agent-matsu` | `SubmissionAgent` = champion **MCTS** + time governor (SOT-1693) | 0.8s/決定 pred, Greedy fallback |
| 竹 take  | `ptcg-agent-take`  | `RuleBasedAgent` (25-deck 汎化 scoring, SOT-1694) | deck-free |
| 梅 ume   | `ptcg-agent-ume`   | `HarnessAgent` = PPO policy + **MCTS** (SOT-1695) | `time_limit_s=0.4` |

This is a rematch of the earlier PR#9 battle (which ran the then-shipping
**baselines** — matsu `GreedyAgent`, ume random — on each repo's own champion
`deck.csv`). Two things changed here, per the human's question *"did you use the
25 decks randomly?"*:

1. **Latest agents** (the advanced MCTS / PPO stacks, now wired into `main.py`).
2. **Fair 25-deck random decks** instead of each repo's hand-picked `deck.csv`,
   so the ranking reflects piloting skill across a diverse metagame rather than an
   agent+deck package.

## Primary result — 25-deck **mirror-random**, N=96 matches/pairing (288 total), 0 faults

Both contestants pilot the **same** randomly-drawn tournament deck (from the 25
decks in `decks/initial`, SOT-1684), seats alternated every match, so deck
strength cancels and only **agent skill** is measured. 12 independent seeded
shards (`--seed 20260716001..012`, n=8 each) aggregated. Win rate excludes draws;
Wilson 95% CI.

| pairing | wins | win rate (row vs col) | 95% CI | 先手勝率 |
| --- | --- | --- | --- | --- |
| matsu vs take | 79 – 17 | **matsu 0.823** | [0.735, 0.886] | 0.448 |
| matsu vs ume  | 73 – 23 | **matsu 0.760** | [0.666, 0.835] | 0.500 |
| take vs ume   | 80 – 16 | **take 0.833**  | [0.746, 0.895] | 0.552 |

### Standings

| rank | 銘 | W–L | win rate | 95% CI |
| --- | --- | --- | --- | --- |
| 1 | 松 matsu | 152–40 | **0.792** | [0.729, 0.843] |
| 2 | 竹 take  | 97–95  | **0.505** | [0.435, 0.575] |
| 3 | 梅 ume   | 39–153 | **0.203** | [0.152, 0.266] |

**松 > 竹 > 梅**, and every head-to-head CI clears 0.5 with a clear gap — the
ordering is statistically decisive on neutral decks. matsu's champion **MCTS**
dominates both opponents (0.82 / 0.76); take's rule-based agent sits at ~0.50
overall (it splits with ume-crushing wins and matsu losses); ume's PPO+MCTS is
weakest here (0.20). **Zero faults** across all 288 matches — all three
submissions run cleanly under per-match deck injection.

Because both sides play the same deck, the earlier suspicion that the PR#9 result
was a deck confound is **rejected**: matsu is strongest even when the deck
advantage is removed. **先手 (first-player) advantage shrank markedly** vs PR#9
(now ~0.45–0.55, was 0.65–0.77): stronger, deck-synced agents convert the mirror
matchup on skill rather than tempo.

### Deck-sync (why this is fair for MCTS)

松 MCTS determinizes from its `deck.csv` and 梅's MCTS/harness reads `deck.csv` at
construction, so a random deck is only fair if each agent *plans with the deck the
engine actually deals it*. The harness launches each contestant in a per-match
**sandbox cwd** (symlinks to the repo, plus a writable copy of `deck.csv`) and, on
each deck change, sends a `__set_deck__` control message: the server rewrites the
sandbox `deck.csv` and `importlib.reload(main)` to rebuild `main.agent` from the
new deck — no source change to any repo's `main.py`, and the sandbox copy
guarantees a sibling repo's committed `deck.csv` is never touched. 竹 is deck-free
and unaffected.

## Secondary context — PR#9 baseline battle (own champion decks)

For reference, the earlier run (baseline agents, each on its own `deck.csv`, N=500)
gave 松 0.740 > 竹 0.657 > 梅 0.103. Same ordering, but that measured
agent+deck packages of the *baseline* submissions; the table above supersedes it
for "which current agent is strongest".

## Method notes

- **Isolation.** The three repos' `agents` packages have colliding module names
  (`base`, `random_agent`, `search_agent`), so they cannot co-exist in one
  interpreter. Each contestant runs in its own subprocess (its repo/venv) behind
  a line-delimited JSON protocol; the host owns only this repo's `cg.game`
  (a process-global single battle, matches run sequentially).
- **Mirror vs independent.** `--deck-mode mirror` (primary) draws one deck per
  先後 pair and gives it to both agents, cancelling deck strength.
  `--deck-mode independent` draws a fresh deck per contestant per match
  (tournament-like, higher variance).
- **Fairness.** Seat-alternation (先後入替) every match; `battle_start` binds each
  deck to its agent so a mirror pair is an exact 先後 swap.
- **Reproducibility.** The cabt engine has no seed API (ASSUMPTIONS A-9), so
  engine shuffles vary run-to-run; `--seed` only fixes the deck-selection RNG.
  Conclusions are statistical (CI separation), not bit-exact. More matches (not a
  fixed seed) is how the CIs are made decisive — hence the shard+`--aggregate`
  workflow. Faults (illegal move / agent exception / dead server) are charged to
  the offending agent as a loss and the batch continues.
</content>
</invoke>
