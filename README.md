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
git submodule update --init --recursive
bash scripts/check_core_compatibility.sh
python3 -m venv venv && venv/bin/pip install -r requirements.txt
bash scripts/setup_engine.sh          # copies cg/ + data/ from the Kaggle download
venv/bin/python eval/run_match.py     # run one local self-play match
```

## Shared core dependency

This repository consumes [`ptcg-agent-core`](vendor/ptcg-agent-core) as a
pinned Git submodule, using the same integration boundary as the other PTCG
agents. Core owns algorithm-independent contracts and the shared
[Kaggle submission guide](vendor/ptcg-agent-core/docs/kaggle-submission.md).
Matsu continues to own its Python adapter, deck, search/policy code, and
evaluation logic; those are deliberately not shared.

The pinned commit keeps setup and submission builds reproducible. To update
core, review its schema versions and release notes, then run:

```bash
git -C vendor/ptcg-agent-core fetch origin main
git -C vendor/ptcg-agent-core checkout origin/main
bash scripts/check_core_compatibility.sh
bash scripts/check.sh
git add vendor/ptcg-agent-core
```

Commit the gitlink update together with the compatibility results. If the new
core is incompatible, restore the previous gitlink with
`git checkout -- vendor/ptcg-agent-core`, run
`git submodule update --init`, and re-run both checks. Do not use an unreviewed
moving branch for a submission build.

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

The builder follows the core-owned submission layout and checks that required
top-level files exist while development files, credentials, Git metadata, and
the core checkout remain outside the archive. For authentication, submission,
result checks, and troubleshooting, follow the
[shared core guide](vendor/ptcg-agent-core/docs/kaggle-submission.md).
