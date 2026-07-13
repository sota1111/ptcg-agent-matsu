"""Engine-dependent tests (SOT-1671).

Skipped automatically when the cabt engine bindings (cg/, license-restricted,
gitignored) are not present — e.g. on CI.

Reproducibility note (ASSUMPTIONS.md A-9): the engine's internal RNG cannot
be seeded externally, so "same seed -> same play" is specified and tested at
the agent boundary: same seed + same observation sequence -> same actions.
"""
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)  # libcg.so / deck.csv resolve relative to the repo root

try:
    from cg import game  # noqa: F401
    HAS_ENGINE = True
except Exception:  # pragma: no cover - engine absent (CI)
    HAS_ENGINE = False

from agents import GreedyAgent, RandomAgent, make_agent


def load_deck():
    with open(os.path.join(REPO, "deck.csv")) as f:
        return [int(x) for x in f.read().split("\n")[:60]]


def play_and_record(agent0, agent1, max_steps=100000):
    """Play one engine match; return per-player [(obs, action), ...]."""
    deck = load_deck()
    records = ([], [])
    obs, start = game.battle_start(deck, deck)
    assert obs is not None, "battle_start failed"
    try:
        steps = 0
        while steps < max_steps:
            current = obs.get("current") or {}
            if current.get("result", -1) != -1:
                return records, current["result"]
            idx = current.get("yourIndex", 0)
            agent = agent0 if idx == 0 else agent1
            action = agent.act(obs)
            records[idx].append((obs, action))
            obs = game.battle_select(action)
            steps += 1
        raise AssertionError("match did not finish")
    finally:
        game.battle_finish()


@unittest.skipUnless(HAS_ENGINE, "cabt engine (cg/) not available")
class TestEngineReproducibility(unittest.TestCase):
    def test_same_seed_same_observations_same_actions(self):
        for name in ("random", "greedy"):
            with self.subTest(agent=name):
                records, _ = play_and_record(make_agent(name, seed=11),
                                             make_agent(name, seed=22))
                # Fresh agents with the same seeds replay the recorded
                # observation streams and must reproduce every action.
                for idx, seed in ((0, 11), (1, 22)):
                    replay = make_agent(name, seed=seed)
                    for obs, action in records[idx]:
                        self.assertEqual(replay.act(obs), action)

    def test_different_seed_diverges_for_random(self):
        records, _ = play_and_record(RandomAgent(seed=11),
                                     RandomAgent(seed=22))
        merged = records[0] + records[1]
        replay = RandomAgent(seed=99)
        replayed = [replay.act(obs) for obs, _ in merged]
        recorded = [action for _, action in merged]
        self.assertNotEqual(replayed, recorded)

    def test_no_fallbacks_on_known_card_pool(self):
        agent0, agent1 = GreedyAgent(seed=1), GreedyAgent(seed=2)
        play_and_record(agent0, agent1)
        self.assertEqual(agent0.fallback_count, 0)
        self.assertEqual(agent1.fallback_count, 0)
        self.assertGreater(agent0.decision_count, 0)


if __name__ == "__main__":
    unittest.main()
