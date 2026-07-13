"""Agent base class (SOT-1671).

`act(obs_dict)` implements the submission contract (main.py:22-36):
- initial call with `select == None` -> return the 60-card deck;
- otherwise -> a legal list of option indices.

Subclasses implement `choose(view) -> list[int]`. Whatever happens inside
`choose` (unknown enum values, unknown cards, scoring bugs), `act` degrades
to a uniformly random LEGAL action instead of crashing or submitting an
illegal move; such degradations are counted in `fallback_count` so benches
and tests can assert they stay at zero on the known card pool.
"""
from . import actions
from .observation import View, adapt
from .rng import Rng


class BaseAgent:
    def __init__(self, seed: int, deck=None):
        self.seed = int(seed)
        self.rng = Rng(self.seed).child(type(self).__name__)
        self._deck = list(deck) if deck is not None else None
        self.fallback_count = 0
        self.decision_count = 0

    def choose(self, view: View) -> list:
        """Return a legal action for the given view (override in subclasses)."""
        raise NotImplementedError

    def act(self, obs_dict: dict) -> list:
        view = adapt(obs_dict)
        if view.select is None:
            # Initial deck selection (only on the submission harness).
            if self._deck is None:
                raise ValueError("select is None but no deck was provided")
            return list(self._deck)
        self.decision_count += 1
        try:
            return actions.validate(view.select, self.choose(view))
        except Exception:
            self.fallback_count += 1
            return actions.random_action(view.select, self.rng)
