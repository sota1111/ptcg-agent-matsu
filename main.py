import os
import time

from agents import GreedyAgent, MctsAgent, actions
from agents.observation import adapt
from agents.rng import Rng

# Agent seed: externally injectable (SOT-1671 RNG discipline); the default
# only fixes the tie-break/fallback stream, the engine shuffles independently.
_DEFAULT_SEED = 20260713

# Champion Determinized-MCTS configuration (SOT-1672 docs/mcts-design.md §7,
# confirmed by the SOT-1673 ablation; unspecified fields keep PlannerConfig
# defaults: uct_c=1.4, rollout="greedy", heuristic evaluator DEFAULT_WEIGHTS).
CHAMPION_CONFIG = {
    "max_root_actions": 6,
    "max_tree_depth": 1,
    "rollout_turns": 100,
    "rollout_depth": 200,
    "n_worlds": 4,
    "time_budget_s": 0.8,
    "deviate_margin": 0.1,
}

# Remaining-time-aware budget control (ASSUMPTIONS A-1: ~10 min total clock
# per player per match; A-2: no per-move limit). Thresholds are on THIS
# agent's cumulative act() wall-clock; crossing one shrinks the per-decision
# search budget, and past the last one the agent stops searching and hands
# off to Greedy. The 500-match champion bench (docs/mcts-design.md §7) spent
# ~10s of search per match, so in a healthy match the schedule never fires —
# it bounds tail risk (pathological matches / slow submission hardware) far
# away from the 600s loss-on-timeout line.
MATCH_TIME_ALLOWANCE_S = 600.0
BUDGET_SCHEDULE = (
    (300.0, 0.8),   # < 300s spent: champion budget
    (420.0, 0.4),   # 300-420s: half budget
    (510.0, 0.2),   # 420-510s: quarter budget
)                   # >= 510s: Greedy handoff (no search)


class SubmissionAgent:
    """Submission wrapper: champion MCTS + time governor + layered fallbacks.

    Failure containment (Validation Episode Error prevention), innermost
    first: MctsAgent already degrades on its own (planner exception -> greedy
    prior -> random-legal in BaseAgent.act); if its act() nevertheless
    raises, this wrapper falls back to a GreedyAgent, and if THAT raises, to
    a legal action built straight from the raw observation. The initial deck
    call (select is None) always returns the 60-card deck.
    """

    def __init__(self, seed, deck, clock=time.perf_counter, card_index=None):
        self.seed = int(seed)
        self._deck = list(deck)
        self._clock = clock
        self._mcts = MctsAgent(self.seed, deck=self._deck,
                               card_index=card_index, **CHAMPION_CONFIG)
        self._greedy = GreedyAgent(seed=self.seed, deck=self._deck,
                                   card_index=card_index)
        self._rng = Rng(self.seed).child("submission-last-resort")
        self.think_time_s = 0.0   # cumulative act() wall-clock (time governor)
        self.move_times = []      # per-decision wall-clock (bench reporting)
        self.greedy_handoffs = 0  # decisions made by Greedy after exhaustion
        self.emergency_fallbacks = 0  # act()-level exceptions caught here

    # Counters proxied from the inner agents so benches see one namespace.
    @property
    def fallback_count(self):
        return self._mcts.fallback_count + self._greedy.fallback_count

    @property
    def decision_count(self):
        return self._mcts.decision_count + self._greedy.decision_count

    @property
    def budget_violations(self):
        return self._mcts.budget_violations

    @property
    def planner_fallbacks(self):
        return self._mcts.planner_fallbacks

    @property
    def degraded_count(self):
        return self._mcts.degraded_count

    def current_budget(self):
        """Per-decision search budget for the current cumulative clock.

        None means the search allowance is exhausted: hand off to Greedy.
        """
        for spent_limit, budget in BUDGET_SCHEDULE:
            if self.think_time_s < spent_limit:
                return budget
        return None

    def act(self, obs_dict):
        t0 = self._clock()
        try:
            return self._act_inner(obs_dict)
        finally:
            elapsed = self._clock() - t0
            self.think_time_s += elapsed
            if self._is_decision(obs_dict):
                self.move_times.append(elapsed)

    @staticmethod
    def _is_decision(obs_dict):
        try:
            return (obs_dict or {}).get("select") is not None
        except Exception:
            return False

    def _act_inner(self, obs_dict):
        budget = self.current_budget()
        if budget is None:
            if self._is_decision(obs_dict):
                self.greedy_handoffs += 1
            return self._greedy_act(obs_dict)
        self._mcts.config.time_budget_s = budget
        try:
            return self._mcts.act(obs_dict)
        except Exception:
            self.emergency_fallbacks += 1
            return self._greedy_act(obs_dict)

    def _greedy_act(self, obs_dict):
        try:
            return self._greedy.act(obs_dict)
        except Exception:
            self.emergency_fallbacks += 1
            return self._last_resort(obs_dict)

    def _last_resort(self, obs_dict):
        """Legal action from the raw dict alone (no agent code in the path)."""
        sel = (obs_dict or {}).get("select")
        if sel is None:
            return list(self._deck)
        try:
            return actions.random_action(adapt(obs_dict).select, self._rng)
        except Exception:
            options = sel.get("option") or []
            lo = max(int(sel.get("minCount") or 0), 1)
            lo = min(lo, len(options))
            return list(range(lo))


_agent: SubmissionAgent | None = None


def read_deck_csv() -> list[int]:
    """Read deck.csv.

    Returns:
        list[int]: A list of card IDs in the deck.
    """
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck


def agent(obs_dict: dict) -> list[int]:
    """Pokémon Trading Card Game Agent (champion Determinized MCTS, SOT-1693).

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.
    On the initial call obs.select is None and the 60-card deck is returned.

    Returns:
        list[int]: A list of option index.
    """
    global _agent
    if _agent is None:
        seed = int(os.environ.get("AGENT_SEED", _DEFAULT_SEED))
        _agent = SubmissionAgent(seed=seed, deck=read_deck_csv())
    return _agent.act(obs_dict)
