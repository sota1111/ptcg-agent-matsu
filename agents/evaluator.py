"""Leaf evaluator (SOT-1672) — layer [4] of the 4-layer architecture.

`Evaluator.evaluate(obs, root_player)` maps a battle observation to an
estimated win probability in [0, 1] for `root_player`. The planner
(agents/planner.py) calls it on non-terminal rollout leaves; a learned value
function can replace `HeuristicEvaluator` by implementing the same interface
(SOT-1674).

Observations are duck-typed: any object shaped like `cg.api.Observation`
(the engine's search states) works, as do test doubles built from
SimpleNamespace. Every feature derives from state/card ATTRIBUTES visible in
the observation (prizes, HP, energy counts, hand/deck sizes) — per-card
weight tables keyed by card ID/name are forbidden
(scripts/lint_hardcoded_cards.py).
"""
import math

# Feature weights (externally overridable for SOT-1673 ablation). Scores are
# per-side; the value is a logistic squash of (score_me - score_opp).
DEFAULT_WEIGHTS = {
    "prize_taken": 2.0,   # per prize card this side has taken (dominant term)
    "pokemon": 0.3,       # per Pokémon this side has in play
    "energy": 0.2,        # per Energy attached on this side
    "hp": 0.004,          # per HP point this side has in play
    "hand": 0.06,         # per card in hand
    "deck_empty": -3.0,   # this side loses at its next turn start (deck-out)
    "scale": 0.6,         # logistic scale on the score difference
}

PRIZE_START = 6  # PRIZE_SIZE (ptcgProgram 22/Core.h:14)


class Evaluator:
    """Value interface: estimated win probability for `root_player`."""

    def evaluate(self, obs, root_player: int) -> float:
        raise NotImplementedError


class HeuristicEvaluator(Evaluator):
    """Card-attribute heuristic value; terminal results are exact."""

    def __init__(self, weights: dict | None = None):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    def evaluate(self, obs, root_player: int) -> float:
        current = getattr(obs, "current", None)
        if current is None:
            return 0.5
        result = getattr(current, "result", -1)
        if result is not None and result != -1:
            if result == root_player:
                return 1.0
            if result == 1 - root_player:
                return 0.0
            return 0.5  # draw (result == 2) or unknown future value
        players = getattr(current, "players", None) or ()
        if len(players) < 2:
            return 0.5
        diff = (self._side_score(players[root_player])
                - self._side_score(players[1 - root_player]))
        return 1.0 / (1.0 + math.exp(-self.weights["scale"] * diff))

    def _side_score(self, p) -> float:
        w = self.weights
        prize = getattr(p, "prize", None) or ()
        score = w["prize_taken"] * max(0, PRIZE_START - len(prize))
        hp_total = 0
        pokemon = 0
        energy = 0
        in_play = list(getattr(p, "active", None) or ())
        in_play += list(getattr(p, "bench", None) or ())
        for pk in in_play:
            if pk is None:  # facedown Pokémon: presence known, stats hidden
                pokemon += 1
                continue
            pokemon += 1
            hp_total += getattr(pk, "hp", 0) or 0
            energy += len(getattr(pk, "energies", None) or ())
        score += w["pokemon"] * pokemon
        score += w["energy"] * energy
        score += w["hp"] * hp_total
        score += w["hand"] * (getattr(p, "handCount", 0) or 0)
        if (getattr(p, "deckCount", 0) or 0) == 0:
            score += w["deck_empty"]
        return score
