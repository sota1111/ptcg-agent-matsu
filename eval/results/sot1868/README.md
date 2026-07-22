# SOT-1868 runtime promotion evidence

`promotion.json` is the machine-readable gate. It pins the implementation
commit, unchanged deck hash, submission archive hash, agent/deck-schedule
seeds, seat reversal, runtime conditions, and safety counters.

`runtime-smoke.json` is a real cabt process match against Sol using the same
deck in both seats. The engine does not expose shuffle seeding, so the fixed
seed guarantee applies to the agent (`AGENT_SEED` default) and deck schedule;
this limitation is explicit in both artifacts. The two reversed-seat games
finished with zero faults, unfinished games, illegal actions, or timeouts.

The statistical old-version A/B promotion decision is reused from the accepted
SOT-1848 artifact: 33/40 (82.5%), Wilson 95% lower bound 68.05% > 50%.
