"""SOT-1679 tests — MLP value network, trainer determinism, Early Cutoff.

Engine-independent: the planner runs against a scripted turn-advancing
backend double, the evaluator against synthetic models, and the trainer on
generated separable data (same conventions as tests/test_value.py and
tests/test_mcts.py).
"""
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from agents import evaluator as evaluator_mod
from agents import make_agent
from agents.evaluator import LearnedEvaluator, make_evaluator
from agents.features import FEATURE_NAMES
from agents.observation import adapt
from agents.planner import MctsPlanner, PlannerConfig
from agents.rng import Rng
from tests.support import observation, player, pokemon, select, \
    synthetic_card_index


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _obs(me=None, opp=None, **state):
    return observation(select([{"type": 0}]), me=me, opp=opp, **state)


def _mlp_model(layers, **overrides):
    model = {
        "feature_names": list(FEATURE_NAMES),
        "layers": layers,
        "mean": [0.0] * len(FEATURE_NAMES),
        "std": [1.0] * len(FEATURE_NAMES),
    }
    model.update(overrides)
    return model


def _zero_hidden(n_hidden=2):
    k = len(FEATURE_NAMES)
    return [{"w": [[0.0] * k for _ in range(n_hidden)], "b": [0.0] * n_hidden},
            {"w": [[0.0] * n_hidden], "b": [0.0]}]


class TestMlpEvaluator(unittest.TestCase):
    def test_zero_net_is_agnostic(self):
        ev = LearnedEvaluator(model=_mlp_model(_zero_hidden()))
        self.assertEqual(ev.evaluate(_to_namespace(_obs()), 0), 0.5)

    def test_output_bias_flows_through_sigmoid(self):
        layers = _zero_hidden()
        layers[-1]["b"] = [2.0]
        ev = LearnedEvaluator(model=_mlp_model(layers))
        p = ev.evaluate(_to_namespace(_obs()), 0)
        self.assertAlmostEqual(p, 1.0 / (1.0 + math.exp(-2.0)), places=9)

    def test_forward_matches_hand_computation(self):
        k = len(FEATURE_NAMES)
        i = FEATURE_NAMES.index("my_prizes_taken")
        w1 = [0.0] * k
        w1[i] = 0.25
        layers = [{"w": [w1], "b": [0.1]}, {"w": [[1.5]], "b": [-0.2]}]
        ev = LearnedEvaluator(model=_mlp_model(layers))
        obs = _to_namespace(_obs(me=player(prize=2)))  # 4 prizes taken
        h = math.tanh(0.1 + 0.25 * 4.0)
        z = -0.2 + 1.5 * h
        self.assertAlmostEqual(ev.evaluate(obs, 0),
                               1.0 / (1.0 + math.exp(-z)), places=9)

    def test_prize_lead_orders_states(self):
        k = len(FEATURE_NAMES)
        w1 = [0.0] * k
        w1[FEATURE_NAMES.index("my_prizes_taken")] = 0.1
        layers = [{"w": [w1], "b": [0.0]}, {"w": [[1.0]], "b": [0.0]}]
        ev = LearnedEvaluator(model=_mlp_model(layers))
        ahead = ev.evaluate(_to_namespace(_obs(me=player(prize=2))), 0)
        behind = ev.evaluate(_to_namespace(_obs(me=player(prize=6))), 0)
        self.assertGreater(ahead, behind)

    def test_terminal_results_are_exact(self):
        ev = LearnedEvaluator(model=_mlp_model(_zero_hidden()))
        for result, expected in ((0, 1.0), (1, 0.0), (2, 0.5)):
            obs = _to_namespace(_obs(result=result))
            self.assertEqual(ev.evaluate(obs, 0), expected)

    def test_unknown_cards_fall_back_without_crashing(self):
        ev = LearnedEvaluator(model=_mlp_model(_zero_hidden()))
        obs = _to_namespace(_obs(me=player(active=[pokemon(999999)])))
        p = ev.evaluate(obs, 0)  # no card index: neutral attribute defaults
        self.assertTrue(0.0 < p < 1.0)

    def test_layer_shape_mismatch_raises(self):
        bad = _zero_hidden()
        bad[0]["w"][0] = [0.0] * 3  # row shorter than the feature set
        with self.assertRaises(ValueError):
            LearnedEvaluator(model=_mlp_model(bad))

    def test_multi_logit_output_raises(self):
        k = len(FEATURE_NAMES)
        layers = [{"w": [[0.0] * k, [0.0] * k], "b": [0.0, 0.0]}]
        with self.assertRaises(ValueError):
            LearnedEvaluator(model=_mlp_model(layers))

    def test_make_evaluator_value_net_loads_net_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as f:
            json.dump(_mlp_model(_zero_hidden()), f)
        try:
            with mock.patch.object(evaluator_mod, "DEFAULT_NET_PATH", f.name):
                ev = make_evaluator("value_net")
            self.assertIsInstance(ev, LearnedEvaluator)
            self.assertIsNotNone(ev.layers)
        finally:
            os.unlink(f.name)


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "train_value", os.path.join(REPO, "train", "train_value.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _separable(n, seed):
    rng = Rng(seed).child("data")
    k = len(FEATURE_NAMES)
    out = []
    for _ in range(n):
        x = array("d", [rng.random() * 2 - 1 for _ in range(k)])
        y = float(x[0] + 0.5 * x[1] > 0)
        out.append((y, 0.5, 5, x))
    return out


class TestMlpTrainer(unittest.TestCase):
    def setUp(self):
        self.tv = _load_trainer()

    def test_parse_arch(self):
        self.assertIsNone(self.tv.parse_arch("linear"))
        self.assertEqual(self.tv.parse_arch("64-32"), [64, 32])
        self.assertEqual(self.tv.parse_arch("256-128-64"), [256, 128, 64])
        for bad in ("", "0", "a-b", "-8"):
            with self.assertRaises(SystemExit):
                self.tv.parse_arch(bad)

    def _fit(self, seed=61679, epochs=3, checkpoint=None, n=600):
        train = _separable(n, 7)
        mean, std = self.tv.standardizer(train)
        layers = self.tv.fit_mlp(train, mean, std, [8], epochs, lr=0.01,
                                 l2=1e-6, seed=seed, batch=32,
                                 checkpoint=checkpoint)
        return layers, mean, std

    def test_fit_mlp_learns_separable_data(self):
        layers, mean, std = self._fit()
        holdout = _separable(300, 11)
        metrics = self.tv.evaluate_holdout(holdout, mean, std, layers=layers)
        self.assertGreater(metrics["acc_learned"], 0.9)

    def test_same_seed_same_model(self):
        a, _, _ = self._fit(seed=61679)
        b, _, _ = self._fit(seed=61679)
        self.assertEqual(a, b)
        c, _, _ = self._fit(seed=61680)
        self.assertNotEqual(a, c)

    def test_checkpoint_resume_reproduces_single_run(self):
        straight, _, _ = self._fit(epochs=3)
        with tempfile.TemporaryDirectory() as tmp:
            ck = os.path.join(tmp, "ck.json")
            self._fit(epochs=2, checkpoint=ck)   # interrupted run
            resumed, _, _ = self._fit(epochs=3, checkpoint=ck)
        self.assertEqual(straight, resumed)

    def test_checkpoint_config_mismatch_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck = os.path.join(tmp, "ck.json")
            self._fit(epochs=1, checkpoint=ck)
            with self.assertRaises(SystemExit):
                self._fit(epochs=1, checkpoint=ck, seed=99)


# --- Early Cutoff (planner) --------------------------------------------------

class _TurnBackend:
    """Deterministic double whose match never ends: the turn counter advances
    every second step, so rollouts are bounded only by the planner's own
    rules (depth / turn cap / Early Cutoff)."""

    def __init__(self):
        self.next_sid = 0
        self.steps = 0

    def _obs(self):
        sel = SimpleNamespace(option=[SimpleNamespace(type=13, number=0)] * 3,
                              minCount=1, maxCount=1, context=0)
        return SimpleNamespace(
            select=sel,
            current=SimpleNamespace(result=-1, yourIndex=0,
                                    turn=1 + self.steps // 2))

    def begin(self, raw_obs, fills, manual_coin=True):
        self.next_sid += 1
        return self.next_sid, self._obs()

    def step(self, sid, action):
        self.next_sid += 1
        self.steps += 1
        return self.next_sid, self._obs()

    def release(self, sid):
        pass

    def end(self):
        pass


class _SpyEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, obs, root_player):
        self.calls.append(getattr(obs.current, "turn", None))
        return 0.5


def _main_view(n_options=3):
    opts = [{"type": 13, "attackId": 201 + i, "number": 0}
            for i in range(n_options)]
    return adapt(observation(
        select(opts, sel_type=0, context=0, min_count=1, max_count=1),
        me=player(active=[pokemon(101, energies=[3])]),
        opp=player(active=[pokemon(102)])))


def _planner(backend, evaluator, **config_overrides):
    config = PlannerConfig(
        n_worlds=1, max_iterations=1, max_tree_depth=0, rollout="random",
        rollout_depth=30, rollout_turns=100, **config_overrides)
    return MctsPlanner(own_deck=[101] * 60, config=config,
                       evaluator=evaluator, backend=backend,
                       card_index=synthetic_card_index(),
                       clock=lambda: 0.0)


class TestEarlyCutoff(unittest.TestCase):
    def test_default_config_has_no_cutoff(self):
        self.assertIsNone(PlannerConfig().rollout_cutoff)

    def test_cutoff_stops_rollout_at_turn_boundary(self):
        backend = _TurnBackend()
        spy = _SpyEvaluator()
        planner = _planner(backend, spy,
                           rollout_cutoff={"min_steps": 4})
        planner.plan(_main_view(), Rng(7))
        self.assertEqual(planner.rollout_cutoffs, 1)
        self.assertEqual(len(spy.calls), 1)
        # The 4th step lands in turn 3; the boundary to turn 4 fires after 2
        # more steps — 6 in total, far below the 30-step depth cap.
        self.assertEqual(backend.steps, 6)

    def test_without_cutoff_rollout_runs_to_depth(self):
        backend = _TurnBackend()
        spy = _SpyEvaluator()
        planner = _planner(backend, spy)  # rollout_cutoff=None (default)
        planner.plan(_main_view(), Rng(7))
        self.assertEqual(planner.rollout_cutoffs, 0)
        # The full 30-step rollout runs (the turn cap never binds).
        self.assertEqual(backend.steps, 30)

    def test_cutoff_evaluates_state_after_boundary(self):
        backend = _TurnBackend()
        spy = _SpyEvaluator()
        planner = _planner(backend, spy, rollout_cutoff={"min_steps": 4})
        planner.plan(_main_view(), Rng(7))
        # min_steps landed in turn 3 (steps//2 + 1); the evaluated state is
        # the first one PAST that turn.
        self.assertEqual(spy.calls, [4])

    def test_cutoff_config_reaches_planner_via_make_agent(self):
        agent = make_agent("mcts", seed=1, deck=[101] * 60,
                           rollout_cutoff={"min_steps": 5})
        self.assertEqual(agent.config.rollout_cutoff, {"min_steps": 5})
        self.assertEqual(agent.rollout_cutoffs, 0)  # planner not built yet


if __name__ == "__main__":
    unittest.main()
