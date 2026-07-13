"""Uniform-random baseline agent (SOT-1671)."""
from . import actions
from .base import BaseAgent
from .observation import View


class RandomAgent(BaseAgent):
    """Picks a uniformly random legal action from the engine's option list."""

    def choose(self, view: View) -> list:
        return actions.random_action(view.select, self.rng)
