# ptcg-agent-matsu

Agent & local evaluation environment for the **Pokémon TCG AI Battle Challenge** (Kaggle).

- Competition (Simulation): https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
- Competition (Strategy):   https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy

## ⚠️ License note
The battle engine (`cg/`, `libcg.so`) and card data (`data/`) are **competition-use-only and must not
be redistributed**. They are **gitignored** and never committed. Only our own code
(`main.py`, `deck.csv`, `eval/`, `scripts/`) lives in git.

## Layout
```
main.py            # submission entry: agent(obs_dict) -> list[int]  (tracked)
deck.csv           # our 60-card deck                                (tracked)
agents/            # 4-layer agent package (SOT-1671):               (tracked)
                   #   observation.py = [1] Observation Adapter
                   #   actions.py     = [2] Action Enumerator
                   #   random_agent.py / greedy_agent.py = baselines
                   #   rng.py (seeded RNG) / cards.py (attribute features)
eval/run_match.py  # local self-play match runner                   (tracked)
eval/bench.py      # N-match benchmark (win rate + Wilson CI)       (tracked)
tests/             # unittest suite (engine tests self-skip w/o cg/) (tracked)
scripts/           # setup + build + check helpers                  (tracked)
cg/                # cabt engine bindings (gitignored, license)
data/              # card CSVs (gitignored, license)
```

## Setup
```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
bash scripts/setup_engine.sh          # copies cg/ + data/ from the Kaggle download
venv/bin/python eval/run_match.py     # run one local self-play match
```

## Verify (lint + tests) — also run by CI
```bash
bash scripts/check.sh                 # forbidden-term lint + syntax + unittest
venv/bin/python eval/bench.py --agent-a greedy --agent-b random --n 1000
```
The forbidden-term linter (`scripts/lint_hardcoded_cards.py`) rejects hardcoded
card names / card IDs in agent code and global `random`/`np.random` use in
`agents/` — evaluation must derive from card attributes and all agent
randomness from an injected seed. Baseline results: `docs/baselines.md`.

## Build a submission
```bash
bash scripts/build_submission.sh      # -> submission.tar.gz (main.py + deck.csv + agents/ + cg/)
```
