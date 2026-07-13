"""PTCG battle agents (SOT-1671) — 4-layer architecture, layers [1]+[2]
plus Random/Greedy baselines.

- observation.py : [1] Observation Adapter (raw obs dict -> information-set View)
- actions.py     : [2] Action Enumerator (obs.select is the single source of truth)
- random_agent.py / greedy_agent.py : baseline policies
- rng.py         : single externally-seeded RNG (no global random)
- cards.py       : card-attribute feature index (unknown IDs -> defaults)
"""
from .base import BaseAgent
from .greedy_agent import GreedyAgent
from .random_agent import RandomAgent
from .rng import Rng

AGENT_TYPES = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
}


def make_agent(name: str, seed: int, deck=None, **kwargs) -> BaseAgent:
    """Factory: agent name -> instance. Raises KeyError for unknown names."""
    return AGENT_TYPES[name](seed=seed, deck=deck, **kwargs)
