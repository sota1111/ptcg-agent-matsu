"""Determinized-MCTS agent (SOT-1672) — wires the planner into the SOT-1671
agent contract.

Per decision, `choose` derives an independent per-decision Rng stream
(`rng.child(f"plan{decision_count}")`) so reproducibility is per-decision:
the same seed and the same observation sequence give the same action, no
matter how much randomness an earlier decision consumed.

Time-budget awareness: every decision (including forced fast-path ones) is
timed; `budget_violations` counts decisions that exceeded
`config.time_budget_s` (the planner itself stops searching at 80% of the
budget — the margin absorbs bookkeeping overhead). The bench reports this
counter; the acceptance criterion is 0.

Failure containment: if the planner cannot search (engine absent, fills
inconsistent with an exotic observation), it degrades to its greedy prior;
if `plan` raises, this class falls back to a plain GreedyAgent choice
(`planner_fallbacks`); BaseAgent.act still guards the outermost layer with
the random-legal fallback. All three counters are expected to stay 0 on the
known card pool.
"""
import time

from .base import BaseAgent
from .cards import shared_index
from .evaluator import HeuristicEvaluator, make_evaluator
from .greedy_agent import GreedyAgent
from .observation import View
from .planner import MctsPlanner, PlannerConfig


class MctsAgent(BaseAgent):
    """Determinized MCTS over the engine search API (anytime, budget-aware)."""

    def __init__(self, seed: int, deck=None, card_index=None, config=None,
                 evaluator=None, backend=None, clock=time.perf_counter,
                 **config_overrides):
        super().__init__(seed, deck)
        # `eval_weights` rides in the JSON config overrides (SOT-1673
        # ablation: bench.py --config-a can only pass constructor kwargs)
        # and overrides HeuristicEvaluator feature weights.
        eval_weights = config_overrides.pop("eval_weights", None)
        if eval_weights and evaluator is None:
            evaluator = HeuristicEvaluator(weights=eval_weights)
        self.config = config or PlannerConfig(**config_overrides)
        # Resolve the card master eagerly: the lazy singleton load would
        # otherwise land inside the first TIMED decision (budget criterion).
        self._card_index = card_index if card_index is not None \
            else shared_index()
        # String specs ("heuristic" / "learned") come from bench --config
        # JSON (SOT-1674); Evaluator instances pass through unchanged.
        if evaluator is not None and not hasattr(evaluator, "evaluate"):
            evaluator = make_evaluator(evaluator, card_index=self._card_index)
        self._evaluator = evaluator
        self._backend = backend
        self._clock = clock
        self._planner = None
        self._greedy_fallback = None
        self.planner_fallbacks = 0
        self.budget_violations = 0
        self.move_times = []

    @property
    def planner(self) -> MctsPlanner:
        if self._planner is None:
            if self._card_index is None:
                self._card_index = shared_index()
            self._planner = MctsPlanner(
                own_deck=self._deck, config=self.config,
                evaluator=self._evaluator, backend=self._backend,
                card_index=self._card_index, clock=self._clock)
        return self._planner

    @property
    def degraded_count(self) -> int:
        return self._planner.degraded_count if self._planner else 0

    def choose(self, view: View) -> list:
        t0 = self._clock()
        rng = self.rng.child(f"plan{self.decision_count}")
        try:
            action = self.planner.plan(view, rng)
        except Exception:
            self.planner_fallbacks += 1
            if self._greedy_fallback is None:
                self._greedy_fallback = GreedyAgent(
                    seed=self.seed, card_index=self._card_index)
            action = self._greedy_fallback.choose(view)
        elapsed = self._clock() - t0
        self.move_times.append(elapsed)
        if elapsed > self.config.time_budget_s:
            self.budget_violations += 1
        return action
