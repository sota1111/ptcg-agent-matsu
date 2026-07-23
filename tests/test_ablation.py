"""SOT-1673 ablation harness tests — engine-independent.

Covers the pure parts: the eval_weights -> HeuristicEvaluator wiring through
MctsAgent's JSON-config surface (bench.py --config-a), and the ablation
driver's cell matrix / seed derivation invariants.
"""
import importlib.util
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from agents.cards import CardIndex
from agents.evaluator import DEFAULT_WEIGHTS, HeuristicEvaluator
from agents.mcts_agent import MctsAgent


def load_ablation():
    spec = importlib.util.spec_from_file_location(
        "ablation", os.path.join(REPO, "eval", "ablation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestEvalWeightsWiring(unittest.TestCase):
    def test_eval_weights_override_builds_evaluator(self):
        agent = MctsAgent(seed=1, deck=list(range(1, 61)),
                          card_index=CardIndex(),
                          eval_weights={"prize_taken": 5.0})
        self.assertIsInstance(agent._evaluator, HeuristicEvaluator)
        self.assertEqual(agent._evaluator.weights["prize_taken"], 5.0)
        # unlisted weights keep their defaults
        self.assertEqual(agent._evaluator.weights["hand"],
                         DEFAULT_WEIGHTS["hand"])
        # eval_weights must not leak into PlannerConfig
        self.assertFalse(hasattr(agent.config, "eval_weights"))

    def test_no_eval_weights_keeps_planner_default(self):
        agent = MctsAgent(seed=1, deck=list(range(1, 61)),
                          card_index=CardIndex(), n_worlds=2)
        self.assertIsNone(agent._evaluator)
        self.assertEqual(agent.config.n_worlds, 2)
        self.assertEqual(agent.search_iterations, 0)

    def test_explicit_evaluator_wins_over_eval_weights(self):
        sentinel = HeuristicEvaluator(weights={"prize_taken": 9.0})
        agent = MctsAgent(seed=1, deck=list(range(1, 61)),
                          card_index=CardIndex(), evaluator=sentinel,
                          eval_weights={"prize_taken": 5.0})
        self.assertIs(agent._evaluator, sentinel)


class TestAblationMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ab = load_ablation()

    def test_required_axes_present(self):
        axes = {axis for axis, _, _ in self.ab.CONFIGS}
        self.assertLessEqual(
            {"baseline", "n_worlds", "uct_c", "rollout", "eval_weights"},
            axes)
        opponents = {name for name, _ in self.ab.OPPONENTS}
        self.assertEqual({"random", "greedy", "mcts_base"}, opponents)

    def test_baseline_is_sot1672_adopted_config(self):
        self.assertEqual(self.ab.BASELINE["n_worlds"], 4)
        self.assertEqual(self.ab.BASELINE["time_budget_s"], 0.8)
        self.assertEqual(self.ab.BASELINE["deviate_margin"], 0.1)
        base_row = next(ov for _, name, ov in self.ab.CONFIGS
                        if name == "baseline")
        self.assertEqual(base_row, {})

    def test_cell_names_unique_and_slugs_safe(self):
        names = [name for _, name, _ in self.ab.CONFIGS]
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            for opp, _ in self.ab.OPPONENTS:
                slug = self.ab.cell_slug(name, opp)
                self.assertRegex(slug, r"^[\w.=-]+$")

    def test_cell_seed_deterministic_and_distinct(self):
        seeds = {}
        for _, name, _ in self.ab.CONFIGS:
            for opp, _ in self.ab.OPPONENTS:
                s = self.ab.cell_seed(94000, name, opp)
                self.assertEqual(s, self.ab.cell_seed(94000, name, opp))
                seeds[(name, opp)] = s
        # shards use base..base+shards-1: cell bases must not collide
        # within a realistic shard width
        values = sorted(seeds.values())
        for a, b in zip(values, values[1:]):
            self.assertGreaterEqual(b - a, 100)


if __name__ == "__main__":
    unittest.main()
