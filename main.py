import os

from agents import GreedyAgent

# Agent seed: externally injectable (SOT-1671 RNG discipline); the default
# only fixes the tie-break/fallback stream, the engine shuffles independently.
_DEFAULT_SEED = 20260713

_agent: GreedyAgent | None = None


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
    """Pokémon Trading Card Game Agent (GreedyAgent baseline, SOT-1671).

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.
    On the initial call obs.select is None and the 60-card deck is returned.

    Returns:
        list[int]: A list of option index.
    """
    global _agent
    if _agent is None:
        seed = int(os.environ.get("AGENT_SEED", _DEFAULT_SEED))
        _agent = GreedyAgent(seed=seed, deck=read_deck_csv())
    return _agent.act(obs_dict)
