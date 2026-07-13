"""PTCG battle agents — 4-layer architecture (SOT-1671 layers [1]+[2],
SOT-1672 layers [3]+[4]) plus Random/Greedy baselines.

- observation.py : [1] Observation Adapter (raw obs dict -> information-set View)
- actions.py     : [2] Action Enumerator (obs.select is the single source of truth)
- planner.py     : [3] Determinized MCTS planner (engine search API, anytime)
- evaluator.py   : [4] leaf value interface + heuristic implementation
- random_agent.py / greedy_agent.py / mcts_agent.py : policies
- rng.py         : single externally-seeded RNG (no global random)
- cards.py       : card-attribute feature index (unknown IDs -> defaults)
"""
from .base import BaseAgent
from .greedy_agent import GreedyAgent
from .mcts_agent import MctsAgent
from .random_agent import RandomAgent
from .rng import Rng

AGENT_TYPES = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "mcts": MctsAgent,
}


def make_agent(name: str, seed: int, deck=None, **kwargs) -> BaseAgent:
    """Factory: agent name -> instance. Raises KeyError for unknown names."""
    return AGENT_TYPES[name](seed=seed, deck=deck, **kwargs)
