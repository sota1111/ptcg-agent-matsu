# Teacher distillation and champion league

`train/distill.py` consumes versioned JSONL positions containing the encoded
state, all legal-action features, MCTS root visit counts, and the terminal value
from the root-player perspective. Faulted or unfinished trajectories are passed
to `TeacherDatasetWriter.append_trajectory(..., valid=False)` and are excluded.

Each dataset manifest records its SHA-256, teacher identity, seed/config supplied
by the generator, and git revision. Training writes an inference model, a
`distillation-run/v1` manifest with code/data/config/model provenance, and an
atomic epoch-boundary checkpoint. Resume is refused if dataset hash, dimensions,
seed, learning rate, hidden size, or config differs.

`eval/champion_league.py` reads candidate-perspective match JSONL. Opponents are
named `champion` or `history/<artifact-id>`; a fixed holdout deck suite is thus a
versioned history entry rather than an implicit global. The promotion decision
requires a Wilson 95% lower bound above 0.50, zero faults/timeouts, candidate mean
latency no more than 110% of champion, and every configured opponent. It emits a
machine-readable `champion-promotion/v1` artifact and exits non-zero on rejection,
which makes the same gate usable in CI.
