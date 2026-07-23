"""SOT-1674 tests — value features, LearnedEvaluator, trainer.

Engine-independent: observations are synthetic (dict-shaped like live-match
raw observations, and SimpleNamespace-shaped like engine search states), the
card master is tests/support.py's synthetic index, and the trainer is
exercised on generated separable data.
"""
import importlib.util
import math
import os
import sys
import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from agents import evaluator as evaluator_mod
from agents.evaluator import (HeuristicEvaluator, LearnedEvaluator,
                              make_evaluator)
from agents.features import FEATURE_NAMES, featurize
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


def _model(weights=None, **overrides):
    model = {
        "feature_names": list(FEATURE_NAMES),
        "weights": weights or [0.0] * len(FEATURE_NAMES),
        "bias": 0.0,
        "mean": [0.0] * len(FEATURE_NAMES),
        "std": [1.0] * len(FEATURE_NAMES),
    }
    model.update(overrides)
    return model


class TestFeaturize(unittest.TestCase):
    def test_vector_matches_feature_names(self):
        x = featurize(_obs(), 0)
        self.assertEqual(len(x), len(FEATURE_NAMES))
        self.assertTrue(all(isinstance(v, float) for v in x))

    def test_dict_and_namespace_shapes_agree(self):
        """Live-match dicts and engine search dataclasses must featurize
        identically — training/inference feature parity."""
        obs = _obs(me=player(active=[pokemon(101, hp=70, max_hp=100,
                                             energies=(0, 0))],
                             prize=4, deck_count=0),
                   opp=player(bench=[pokemon(102)], hand_count=2))
        for root in (0, 1):
            self.assertEqual(featurize(obs, root, synthetic_card_index()),
                             featurize(_to_namespace(obs), root,
                                       synthetic_card_index()))

    def test_perspective_flips_sides(self):
        obs = _obs(me=player(prize=3), opp=player(prize=6))
        x0 = featurize(obs, 0)
        x1 = featurize(obs, 1)
        side = (len(FEATURE_NAMES) - 2) // 2  # my_*/opp_* blocks swap
        self.assertEqual(x0[:side], x1[side:2 * side])
        self.assertEqual(x0[side:2 * side], x1[:side])
        self.assertEqual(x0[0], 3.0)  # prizes taken by side 0
        self.assertEqual(x1[0], 0.0)
        self.assertNotEqual(x0[-1], x1[-1])  # my_turn flag flips

    def test_card_attribute_features_use_index(self):
        obs = _obs(me=player(active=[pokemon(101)]))
        with_index = featurize(obs, 0, synthetic_card_index())
        without = featurize(obs, 0, None)
        i = FEATURE_NAMES.index("my_active_max_attack")
        self.assertEqual(with_index[i], 0.5)   # attack 201 damage 50 / 100
        self.assertEqual(without[i], 0.0)      # unknown card -> default
        self.assertNotEqual(with_index, without)

    def test_degenerate_observations_do_not_crash(self):
        for obs in ({}, {"current": None}, {"current": {"players": []}}):
            self.assertEqual(len(featurize(obs, 0)), len(FEATURE_NAMES))


class TestHeuristicDeckLow(unittest.TestCase):
    """SOT-1697 deck-preservation gradient: off by default (champion identity),
    penalises a thin own deck when enabled via eval_weights."""

    def _state(self, my_deck, my_prize=6):
        obs = _obs(me=player(deck_count=my_deck, prize=my_prize,
                             active=[pokemon(101, hp=90)]),
                   opp=player(deck_count=20, active=[pokemon(102, hp=90)]))
        return _to_namespace(obs)

    def test_default_off_is_identity(self):
        ev = HeuristicEvaluator()
        # deck_low defaults to 0 => a thin deck and a full deck score the same.
        self.assertAlmostEqual(ev.evaluate(self._state(2), 0),
                               ev.evaluate(self._state(30), 0))

    def test_enabled_penalises_thin_own_deck(self):
        ev = HeuristicEvaluator(weights={"deck_low": -0.5, "deck_low_at": 8})
        thin = ev.evaluate(self._state(2), 0)   # 6 cards below threshold
        full = ev.evaluate(self._state(30), 0)  # above threshold, no penalty
        self.assertLess(thin, full)
        # No penalty once the deck is at/above the threshold.
        self.assertAlmostEqual(ev.evaluate(self._state(8), 0),
                               ev.evaluate(self._state(30), 0))

    def test_empty_deck_cliff_unchanged(self):
        # Ramp kept below the deck_empty cliff (max at d=1 is -0.3*7=-2.1 > -3.0)
        # so the terminal deck-out stays the worst non-terminal state.
        ev = HeuristicEvaluator(weights={"deck_low": -0.3, "deck_low_at": 8})
        # deckCount==0 still takes the terminal deck_empty cliff, not the ramp.
        empty = ev.evaluate(self._state(0), 0)
        one = ev.evaluate(self._state(1), 0)
        self.assertLess(empty, one)

    def test_prize_gate_preserves_endgame_dig(self):
        ev = HeuristicEvaluator(weights={"deck_low": -0.2,
                                         "deck_low_at": 14,
                                         "deck_low_prize_gate": 3})
        far = ev.evaluate(self._state(4, my_prize=3), 0)
        near = ev.evaluate(self._state(4, my_prize=2), 0)
        self.assertLess(far, near)


class TestNextTurnBoardWipeRisk(unittest.TestCase):
    """SOT-1878 replay-shaped board-wipe regression coverage."""

    def _state(self, my_board, opp_active, opp_bench=()):
        return _to_namespace(_obs(
            me=player(active=my_board[:1], bench=my_board[1:]),
            opp=player(active=[opp_active], bench=opp_bench)))

    def test_candidate_penalises_board_where_every_pokemon_is_reachable(self):
        cards = synthetic_card_index()
        ev = HeuristicEvaluator(
            weights={"board_wipe": -2.0}, card_index=cards)
        exposed = self._state(
            [pokemon(101, hp=40), pokemon(101, hp=50)],
            pokemon(101, energies=(0, 0, 0)))  # payable 50-damage attack
        survivor = self._state(
            [pokemon(101, hp=40), pokemon(101, hp=80)],
            pokemon(101, energies=(0, 0, 0)))
        self.assertEqual(ev.board_wipe_risk(
            exposed.current.players[0], exposed.current.players[1]), 1.0)
        self.assertEqual(ev.board_wipe_risk(
            survivor.current.players[0], survivor.current.players[1]), 0.0)
        self.assertLess(ev.evaluate(exposed, 0), ev.evaluate(survivor, 0))

    def test_unpayable_attack_is_not_reachable(self):
        cards = synthetic_card_index()
        ev = HeuristicEvaluator(
            weights={"board_wipe": -2.0}, card_index=cards)
        state = self._state(
            [pokemon(101, hp=40)],
            pokemon(101, energies=()))  # attack 201 requires one Energy
        self.assertEqual(ev.board_wipe_risk(
            state.current.players[0], state.current.players[1]), 0.0)

    def test_charged_bench_attacker_models_switch_response(self):
        cards = synthetic_card_index()
        ev = HeuristicEvaluator(
            weights={"board_wipe": -2.0}, card_index=cards)
        state = self._state(
            [pokemon(101, hp=25)],
            pokemon(102, energies=()),
            [pokemon(102, energies=(0, 0))])  # bench reaches 30
        self.assertEqual(ev.board_wipe_risk(
            state.current.players[0], state.current.players[1]), 1.0)

    def test_default_weight_preserves_champion_value(self):
        cards = synthetic_card_index()
        state = self._state(
            [pokemon(101, hp=40)],
            pokemon(101, energies=(0, 0, 0)))
        self.assertAlmostEqual(
            HeuristicEvaluator(card_index=cards).evaluate(state, 0),
            HeuristicEvaluator(weights={"board_wipe": 0.0},
                               card_index=cards).evaluate(state, 0))

    def test_smooth_survival_rewards_hp_margin_and_bench_escape(self):
        cards = synthetic_card_index()
        ev = HeuristicEvaluator(
            weights={"board_survival": 1.0}, card_index=cards)
        attacker = pokemon(101, energies=(0, 0, 0))  # reaches 50 damage
        wiped = self._state([pokemon(101, hp=40)], attacker)
        survivor = self._state([pokemon(101, hp=80)], attacker)
        bench_escape = self._state(
            [pokemon(101, hp=40), pokemon(101, hp=80)], attacker)
        self.assertEqual(ev.board_survival(
            wiped.current.players[0], wiped.current.players[1]), 0.0)
        self.assertGreater(ev.evaluate(survivor, 0),
                           ev.evaluate(wiped, 0))
        self.assertGreater(ev.evaluate(bench_escape, 0),
                           ev.evaluate(wiped, 0))

    def test_smooth_survival_is_disabled_by_default(self):
        cards = synthetic_card_index()
        state = self._state(
            [pokemon(101, hp=80)],
            pokemon(101, energies=(0, 0, 0)))
        self.assertAlmostEqual(
            HeuristicEvaluator(card_index=cards).evaluate(state, 0),
            HeuristicEvaluator(weights={"board_survival": 0.0},
                               card_index=cards).evaluate(state, 0))


class TestLearnedEvaluator(unittest.TestCase):
    def test_terminal_results_are_exact(self):
        ev = LearnedEvaluator(model=_model())
        for result, expected in ((0, 1.0), (1, 0.0), (2, 0.5)):
            obs = _to_namespace(_obs(result=result))
            self.assertEqual(ev.evaluate(obs, 0), expected)
            self.assertEqual(ev.evaluate(obs, 1), 1.0 - expected
                             if result != 2 else 0.5)

    def test_prize_weight_orders_states(self):
        weights = [0.0] * len(FEATURE_NAMES)
        weights[FEATURE_NAMES.index("my_prizes_taken")] = 1.0
        ev = LearnedEvaluator(model=_model(weights=weights))
        ahead = ev.evaluate(_to_namespace(_obs(me=player(prize=2))), 0)
        behind = ev.evaluate(_to_namespace(_obs(me=player(prize=6))), 0)
        self.assertGreater(ahead, behind)
        self.assertTrue(0.0 < behind < ahead < 1.0)

    def test_zero_model_is_agnostic(self):
        ev = LearnedEvaluator(model=_model())
        self.assertEqual(ev.evaluate(_to_namespace(_obs()), 0), 0.5)

    def test_feature_mismatch_raises(self):
        with self.assertRaises(ValueError):
            LearnedEvaluator(model={"feature_names": ["stale"],
                                    "weights": [1.0]})

    def test_make_evaluator_resolution(self):
        self.assertIsInstance(make_evaluator(None), HeuristicEvaluator)
        self.assertIsInstance(make_evaluator("heuristic"), HeuristicEvaluator)
        ev = HeuristicEvaluator()
        self.assertIs(make_evaluator(ev), ev)
        with self.assertRaises(ValueError):
            make_evaluator("nonsense")

    def test_make_evaluator_learned_loads_model_file(self):
        import json
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as f:
            json.dump(_model(), f)
        try:
            with mock.patch.object(evaluator_mod, "DEFAULT_MODEL_PATH",
                                   f.name):
                ev = make_evaluator("learned")
            self.assertIsInstance(ev, LearnedEvaluator)
        finally:
            os.unlink(f.name)


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "train_value", os.path.join(REPO, "train", "train_value.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTrainer(unittest.TestCase):
    def test_fit_learns_separable_data(self):
        tv = _load_trainer()
        from agents.rng import Rng
        rng = Rng(7).child("data")
        k = len(FEATURE_NAMES)

        def example():
            x = array("d", [rng.random() * 2 - 1 for _ in range(k)])
            y = float(x[0] + 0.5 * x[1] > 0)
            return (y, 0.5, 5, x)

        train = [example() for _ in range(2000)]
        holdout = [example() for _ in range(500)]
        mean, std = tv.standardizer(train)
        w, b = tv.fit(train, mean, std, epochs=3, lr0=0.1, l2=1e-6, seed=7)
        metrics = tv.evaluate_holdout(holdout, mean, std, w, b)
        self.assertGreater(metrics["acc_learned"], 0.9)
        self.assertLess(metrics["logloss_learned"],
                        metrics["logloss_heuristic"])  # h logged as 0.5

    def test_logloss_acc(self):
        tv = _load_trainer()
        loss, acc = tv.logloss_acc([(1.0, 0.9), (0.0, 0.2)])
        self.assertAlmostEqual(
            loss, -(math.log(0.9) + math.log(0.8)) / 2, places=6)
        self.assertEqual(acc, 1.0)


if __name__ == "__main__":
    unittest.main()
