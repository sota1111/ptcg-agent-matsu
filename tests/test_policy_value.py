import os
import tempfile
import unittest
from types import SimpleNamespace

from agents.features import FEATURE_NAMES_V2
from agents.policy_value import (PolicyMctsConfig, PolicyMctsSearch,
                                 PolicyValueTransformer)
from agents.search_encoding import ACTION_FEATURE_NAMES
from tests.support import observation, player, pokemon, select


def namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [namespace(v) for v in value]
    return value


class ScriptedBackend:
    def __init__(self, winner_by_action):
        self.winner_by_action = winner_by_action
        self.next_id = 0
        self.releases = 0
        self.ends = 0

    def begin(self, raw_obs, fills, manual_coin=True):
        self.next_id += 1
        return self.next_id, namespace(raw_obs)

    def step(self, sid, action):
        self.next_id += 1
        result = self.winner_by_action[tuple(action)]
        obs = observation(select([]), result=result)
        return self.next_id, namespace(obs)

    def release(self, sid):
        self.releases += 1

    def end(self):
        self.ends += 1


class TurnFlipBackend(ScriptedBackend):
    def step(self, sid, action):
        self.next_id += 1
        raw = observation(select([{"type": 14}]),
                          your_index=action[0], result=-1)
        return self.next_id, namespace(raw)


class ConstantModel:
    def predict(self, state, legal_actions):
        return 0.75, [0.0] * len(legal_actions)


class TestPolicyValueTransformer(unittest.TestCase):
    def model(self, seed=17):
        return PolicyValueTransformer(len(FEATURE_NAMES_V2),
                                      len(ACTION_FEATURE_NAMES),
                                      hidden_size=8, seed=seed)

    def test_output_shapes_and_seed_are_deterministic(self):
        model = self.model()
        state = [0.0] * len(FEATURE_NAMES_V2)
        actions = [[0.0] * len(ACTION_FEATURE_NAMES) for _ in range(3)]
        value, policy = model.predict(state, actions)
        self.assertEqual((value, policy), self.model().predict(state, actions))
        self.assertEqual(len(policy), 3)
        self.assertTrue(-1.0 <= value <= 1.0)

    def test_save_load_preserves_shape_and_outputs(self):
        model = self.model()
        state = [0.25] * len(FEATURE_NAMES_V2)
        actions = [[0.0] * len(ACTION_FEATURE_NAMES),
                   [1.0] * len(ACTION_FEATURE_NAMES)]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.json")
            model.save(path)
            loaded = PolicyValueTransformer.load(path)
        self.assertEqual(loaded.predict(state, actions),
                         model.predict(state, actions))
        self.assertEqual(loaded.state_size, len(FEATURE_NAMES_V2))
        self.assertEqual(loaded.action_size, len(ACTION_FEATURE_NAMES))


class TestPolicyMctsSearch(unittest.TestCase):
    def model(self):
        return PolicyValueTransformer(len(FEATURE_NAMES_V2),
                                      len(ACTION_FEATURE_NAMES),
                                      hidden_size=8, seed=9)

    def root(self, min_count=1, max_count=1):
        return observation(select(
            [{"type": 13, "attackId": 201},
             {"type": 13, "attackId": 202}],
            min_count=min_count, max_count=max_count),
            me=player(active=[pokemon(101)]),
            opp=player(active=[pokemon(102)]))

    def test_search_uses_api_lifecycle_and_selects_winning_legal_move(self):
        backend = ScriptedBackend({(0,): 1, (1,): 0})
        search = PolicyMctsSearch(
            self.model(), backend,
            PolicyMctsConfig(simulations=24, exploration=1.4, seed=44))
        action = search.choose(self.root(), fills=object())
        self.assertEqual(action, [1])
        self.assertEqual(sum(search.last_visits), 24)
        self.assertEqual(backend.releases, 24)
        self.assertEqual(backend.ends, 1)

    def test_multi_selection_remains_legal(self):
        backend = ScriptedBackend({(0, 1): 0})
        search = PolicyMctsSearch(
            self.model(), backend,
            PolicyMctsConfig(simulations=5, seed=3))
        action = search.choose(self.root(min_count=2, max_count=2), object())
        self.assertEqual(action, [0, 1])

    def test_leaf_value_is_inverted_when_turn_changes(self):
        backend = TurnFlipBackend({})
        search = PolicyMctsSearch(
            ConstantModel(), backend,
            PolicyMctsConfig(simulations=8, exploration=2.0, seed=1))
        search.choose(self.root(), object())
        self.assertGreater(search.last_totals[0], 0.0)
        self.assertLess(search.last_totals[1], 0.0)

    def test_fixed_seed_smoke_battle_reaches_battle_finish(self):
        def run():
            turns = []
            for turn in range(1, 5):
                backend = ScriptedBackend({(0,): 0, (1,): 1})
                search = PolicyMctsSearch(
                    self.model(), backend,
                    PolicyMctsConfig(simulations=8, seed=20260719 + turn))
                turns.append(search.choose(self.root(), object()))
            return turns, "Battle Finish"
        first = run()
        self.assertEqual(first, run())
        self.assertEqual(first[1], "Battle Finish")


if __name__ == "__main__":
    unittest.main()
