"""Fast context-aware option scoring adapted from the Take agent.

The adapter deliberately consumes Matsu's information-set ``View`` instead of
importing Take or the engine bindings.  It therefore stays cheap enough for
PUCT priors and rollouts, and unknown options safely retain the established
greedy score.
"""


class ContextScorer:
    """Overlay Take-style action bands on Matsu's complete greedy scorer."""

    def __init__(self, greedy):
        self.greedy = greedy

    def score_options(self, view) -> list[float]:
        scores = self.greedy.score_options(view)
        if view.select is None:
            return scores
        return [self._score(view, option, fallback)
                for option, fallback in zip(view.select.options, scores)]

    def choose(self, view) -> list[int]:
        sel = view.select
        scores = self.score_options(view)
        lo = max(0, min(sel.min_count, len(scores)))
        hi = max(lo, min(sel.max_count, len(scores)))
        # Take's context policy commits minimally for costs and maximally for
        # known benefit contexts.  The greedy chooser remains the authority for
        # count semantics; only substitute its count when it is legal.
        greedy_choice = self.greedy.choose(view)
        k = len(greedy_choice) if lo <= len(greedy_choice) <= hi else lo
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return sorted(order[:k])

    def _score(self, view, option, fallback: float) -> float:
        t = option.type
        raw = option.raw
        # Take scoring bands: terminal KO > development > attack > END.
        if t == 13:  # ATTACK
            attack = self.greedy._attack_score(view, raw.get("attackId"))
            return (10_000.0 if attack > 300.0 else 500.0) + attack
        if t == 9:   # EVOLVE
            return 2_800.0 + fallback
        if t == 8:   # ATTACH
            active_bonus = 200.0 if raw.get("inPlayArea") == 4 else 0.0
            return 2_400.0 + active_bonus + fallback
        if t == 7:   # PLAY (Greedy already distinguishes Supporter/Item/Basic)
            return 2_000.0 + fallback
        if t == 10:  # ABILITY
            return 1_800.0 + fallback
        if t == 12:  # RETREAT: keep conservative Take fallback behaviour
            return -0.5
        if t == 14:  # END
            return 0.0
        return fallback
