# Promoted Matsu runtime profile

SOT-1868 promotes `matsu-stability-control-v1`, evaluated in SOT-1848 and
the SOT-1851 ensemble league, into the real Kaggle entrypoint. The canonical,
submission-bundled contract is `agents/stability_profile.py`; `main.py`
applies it before constructing `MctsAgent`.

Runtime mapping:

- 250 ms search budget, depth 5, UCT constant 0.72;
- 0.28 root deviation margin to suppress noisy departures from the stable
  greedy prior;
- card-agnostic low-deck reserve penalty of 0.22 below 12 cards while at
  least three prizes remain;
- layered highest-value/legal fallbacks and the 600-second time governor are
  retained.

The submission archive test verifies the profile module is bundled. Runtime
promotion evidence under `eval/results/sot1868/` records the repository commit,
deck SHA-256, fixed seeds, seat reversal, execution conditions, and safety
counters used for the final gate.
