import unittest

from agents.observation import adapt
from tests.support import card, observation, player, pokemon, select


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


if __name__ == "__main__":
    unittest.main()
