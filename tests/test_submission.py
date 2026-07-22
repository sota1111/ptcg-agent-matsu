"""SubmissionAgent (main.py, SOT-1693) — engine-free unit tests.

Covers the champion-config wiring, the remaining-time budget governor
(budget steps down as cumulative think time grows, Greedy handoff at
exhaustion), and the layered fallbacks (MCTS exception -> Greedy -> raw
legal action), including the initial deck call.
"""
import unittest

import main as submission
from main import BUDGET_SCHEDULE, CHAMPION_CONFIG, SubmissionAgent
from tests import support

DECK = list(range(1, 61))


def make_submission_agent(clock=None):
    """Engine-free SubmissionAgent (synthetic card master, fake clock)."""
    return SubmissionAgent(seed=1, deck=DECK, clock=clock or FakeClock(),
                           card_index=support.synthetic_card_index())


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class StubAgent:
    """Records calls; optionally raises. Mimics the BaseAgent surface."""

    def __init__(self, action=(0,), raises=False):
        from types import SimpleNamespace
        self.action = list(action)
        self.raises = raises
        self.calls = 0
        self.fallback_count = 0
        self.decision_count = 0
        self.config = SimpleNamespace(time_budget_s=None)
        self.budget_violations = 0
        self.planner_fallbacks = 0
        self.degraded_count = 0

    def act(self, obs):
        self.calls += 1
        if self.raises:
            raise RuntimeError("stub failure")
        return list(self.action)


def decision_obs():
    return support.observation(support.select([{"type": 0}, {"type": 0}]))


class TestChampionConfig(unittest.TestCase):
    def test_mcts_core_uses_champion_config(self):
        agent = make_submission_agent()
        for key, value in CHAMPION_CONFIG.items():
            self.assertEqual(getattr(agent._mcts.config, key), value, key)
        # Unpinned fields keep the documented PlannerConfig defaults.
        self.assertEqual(agent._mcts.config.uct_c, 1.4)
        self.assertEqual(agent._mcts.config.rollout, "greedy")

    def test_module_entrypoint_builds_submission_agent(self):
        self.assertIsNone(submission._agent)  # lazy until first agent() call


class TestBudgetGovernor(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.agent = make_submission_agent(clock=self.clock)
        self.stub = StubAgent()
        self.agent._mcts = self.stub

    def test_budget_steps_down_with_cumulative_think_time(self):
        expected = [(0.0, 0.8), (299.9, 0.8), (300.0, 0.4), (419.9, 0.4),
                    (420.0, 0.2), (509.9, 0.2)]
        for spent, budget in expected:
            self.agent.think_time_s = spent
            self.assertEqual(self.agent.current_budget(), budget, spent)
        self.agent.think_time_s = 510.0
        self.assertIsNone(self.agent.current_budget())

    def test_budget_is_applied_to_the_mcts_config(self):
        self.agent.think_time_s = 350.0
        self.agent.act(decision_obs())
        self.assertEqual(self.stub.config.time_budget_s, 0.4)
        self.assertEqual(self.stub.calls, 1)

    def test_exhausted_clock_hands_off_to_greedy(self):
        self.agent.think_time_s = BUDGET_SCHEDULE[-1][0]
        action = self.agent.act(decision_obs())
        self.assertEqual(self.stub.calls, 0)  # search never invoked
        self.assertEqual(self.agent.greedy_handoffs, 1)
        self.assertIsInstance(action, list)

    def test_think_time_accumulates_from_the_clock(self):
        original = self.stub.act

        def slow_act(obs):
            self.clock.t += 1.5
            return original(obs)

        self.stub.act = slow_act
        self.agent.act(decision_obs())
        self.assertAlmostEqual(self.agent.think_time_s, 1.5)
        self.assertEqual(len(self.agent.move_times), 1)
        self.assertAlmostEqual(self.agent.move_times[0], 1.5)


class TestFallbackChain(unittest.TestCase):
    def setUp(self):
        self.agent = make_submission_agent()

    def test_mcts_exception_falls_back_to_greedy(self):
        self.agent._mcts = StubAgent(raises=True)
        action = self.agent.act(decision_obs())
        self.assertEqual(self.agent.emergency_fallbacks, 1)
        self.assertIsInstance(action, list)
        self.assertTrue(all(0 <= i < 2 for i in action))

    def test_double_failure_falls_back_to_raw_legal_action(self):
        self.agent._mcts = StubAgent(raises=True)
        self.agent._greedy = StubAgent(raises=True)
        obs = decision_obs()
        action = self.agent.act(obs)
        self.assertEqual(self.agent.emergency_fallbacks, 2)
        sel = obs["select"]
        self.assertTrue(sel["minCount"] <= len(action) <= sel["maxCount"])
        self.assertTrue(all(0 <= i < len(sel["option"]) for i in action))
        self.assertEqual(len(set(action)), len(action))

    def test_initial_deck_call_returns_deck_even_when_agents_fail(self):
        self.agent._mcts = StubAgent(raises=True)
        self.agent._greedy = StubAgent(raises=True)
        obs = support.observation(None)
        self.assertEqual(self.agent.act(obs), DECK)
        # The deck call is not a decision: no move time is recorded.
        self.assertEqual(self.agent.move_times, [])

    def test_counters_proxy_the_inner_agents(self):
        agent = make_submission_agent()
        self.assertEqual(agent.fallback_count, 0)
        self.assertEqual(agent.budget_violations, 0)
        self.assertEqual(agent.planner_fallbacks, 0)
        self.assertEqual(agent.degraded_count, 0)


class TestBenchDecksHelpers(unittest.TestCase):
    def test_wilson_ci_matches_bench(self):
        from eval.bench_decks import wilson_ci
        lo, hi = wilson_ci(309, 500)
        self.assertAlmostEqual(lo, 0.5747, places=3)
        self.assertAlmostEqual(hi, 0.6595, places=3)

    def test_discover_decks_orders_by_numeric_prefix(self):
        import os
        from eval.bench_decks import discover_decks
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        decks = discover_decks(os.path.join(repo, "decks", "rotation_baseline"))
        self.assertEqual(len(decks), 25)
        names = [os.path.basename(p) for p in decks]
        self.assertTrue(names[0].startswith("01_"))
        self.assertTrue(names[-1].startswith("25_"))


if __name__ == "__main__":
    unittest.main()
