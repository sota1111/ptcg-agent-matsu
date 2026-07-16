import unittest
from types import SimpleNamespace

from agents.greedy_agent import GreedyAgent
from agents.observation import adapt, adapt_engine_obs
from tests.support import (card, observation, player, pokemon, select,
                           synthetic_card_index)


def _to_ns(value):
    """Recursively mirror a raw observation dict as attribute-access objects,
    the shape the engine's dataclass Observation presents. Lets us exercise the
    dataclass fast path (adapt_engine_obs) without importing the engine."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_ns(v) for v in value]
    return value


class TestObservationAdapter(unittest.TestCase):
    def test_perspective_your_index_1(self):
        me = player(hand=[card(101)], deck_count=30,
                    active=[pokemon(101, hp=90)])
        opp = player(hand=None, deck_count=40, active=[pokemon(102)])
        obs = observation(select([{"type": 14}]), me=me, opp=opp,
                          your_index=1)
        view = adapt(obs)
        self.assertEqual(view.your_index, 1)
        self.assertEqual(view.me.deck_count, 30)
        self.assertEqual(view.me.hand_card_ids, [101])
        self.assertEqual(view.opp.deck_count, 40)
        self.assertIsNone(view.opp.hand_card_ids)  # opponent hand hidden
        self.assertEqual(view.me.active[0].card_id, 101)
        self.assertEqual(view.opp.active[0].card_id, 102)

    def test_information_set_hidden_zones(self):
        me = player(hand=[card(101)], prize=6)
        obs = observation(select([{"type": 14}]), me=me)
        view = adapt(obs)
        self.assertEqual(view.me.prize_count, 6)
        self.assertEqual(view.me.prize_known_ids, [])  # all facedown
        # facedown opponent active is represented as None
        obs2 = observation(select([{"type": 14}]),
                           opp=player(active=[None], hand=None))
        self.assertEqual(adapt(obs2).opp.active, [None])

    def test_unknown_enum_values_preserved(self):
        sel = select([{"type": 999, "mystery": 1}], sel_type=777, context=888)
        view = adapt(observation(sel))
        self.assertEqual(view.select.type, 777)
        self.assertEqual(view.select.context, 888)
        self.assertEqual(view.select.options[0].type, 999)
        self.assertEqual(view.select.options[0].raw["mystery"], 1)

    def test_missing_and_null_fields_do_not_crash(self):
        self.assertIsNone(adapt({}).select)
        self.assertEqual(adapt({}).result, -1)
        self.assertEqual(adapt({"current": None}).me.deck_count, 0)
        view = adapt({"select": {"option": None}, "current": {"players": []}})
        self.assertEqual(view.select.options, [])
        self.assertEqual(view.me.hand_count, 0)

    def test_initial_deck_selection_select_none(self):
        view = adapt({"select": None, "logs": [], "current": None})
        self.assertIsNone(view.select)

    def test_find_pokemon(self):
        me = player(active=[pokemon(101, hp=80)], bench=[pokemon(102)])
        opp = player(active=[pokemon(102, hp=40)], hand=None)
        view = adapt(observation(select([{"type": 14}]), me=me, opp=opp))
        self.assertEqual(view.find_pokemon(0, 4, 0).card_id, 101)  # my active
        self.assertEqual(view.find_pokemon(0, 5, 0).card_id, 102)  # my bench
        self.assertEqual(view.find_pokemon(1, 4, 0).hp, 40)        # opp active
        self.assertIsNone(view.find_pokemon(0, 4, 5))              # bad index
        self.assertIsNone(view.find_pokemon(0, 99, 0))             # bad area


class TestAdaptEngineObs(unittest.TestCase):
    """adapt_engine_obs (SOT-1697 rollout fast path) must be behavior-identical
    to adapt() on the asdict round-trip: same View structure, same greedy pick."""

    def _obs(self):
        me = player(hand=[card(101), card(103)], deck_count=30,
                    active=[pokemon(101, hp=90)], bench=[pokemon(102, hp=40)],
                    discard=[card(102)])
        opp = player(hand=None, deck_count=40, active=[pokemon(102, hp=30)],
                     bench=[], prize=4)
        options = [
            {"type": 13, "attackId": 201, "area": 4, "index": 0,
             "playerIndex": 0},                              # ATTACK
            {"type": 7, "index": 0},                          # PLAY
            {"type": 8, "inPlayArea": 4},                     # ATTACH active
            {"type": 14},                                     # END
        ]
        return observation(select(options, context=1, min_count=1,
                                  max_count=1), me=me, opp=opp)

    def test_structure_matches_dict_adapter(self):
        obs = self._obs()
        v1 = adapt(obs)
        v2 = adapt_engine_obs(_to_ns(obs))
        self.assertEqual(v1.your_index, v2.your_index)
        self.assertEqual(v1.turn, v2.turn)
        self.assertEqual(v1.me, v2.me)      # dataclass deep equality
        self.assertEqual(v1.opp, v2.opp)
        self.assertEqual(v1.stadium_card_ids, v2.stadium_card_ids)
        self.assertEqual(v1.select.context, v2.select.context)
        self.assertEqual(v1.select.min_count, v2.select.min_count)
        self.assertEqual(v1.select.max_count, v2.select.max_count)
        self.assertEqual(v1.select.deck_card_ids, v2.select.deck_card_ids)
        self.assertEqual(v1.select.options, v2.select.options)  # incl. .raw

    def test_greedy_choice_identical(self):
        greedy = GreedyAgent(seed=0, card_index=synthetic_card_index())
        obs = self._obs()
        v1, v2 = adapt(obs), adapt_engine_obs(_to_ns(obs))
        self.assertEqual(greedy.choose(v1), greedy.choose(v2))
        self.assertEqual(greedy.score_options(v1), greedy.score_options(v2))

    def test_facedown_and_hidden_zones(self):
        obs = observation(select([{"type": 14}]),
                          me=player(active=[pokemon(101)], prize=6),
                          opp=player(active=[None], hand=None, bench=[]))
        v1, v2 = adapt(obs), adapt_engine_obs(_to_ns(obs))
        self.assertEqual(v2.opp.active, [None])       # facedown -> None
        self.assertIsNone(v2.opp.hand_card_ids)       # opponent hand hidden
        self.assertEqual(v1.me.prize_count, v2.me.prize_count)
        self.assertEqual(v1.opp, v2.opp)

    def test_initial_deck_selection_select_none(self):
        obs = _to_ns({"select": None, "logs": [], "current": None})
        view = adapt_engine_obs(obs)
        self.assertIsNone(view.select)
        self.assertEqual(view.result, -1)
        self.assertEqual(view.me.deck_count, 0)


if __name__ == "__main__":
    unittest.main()
