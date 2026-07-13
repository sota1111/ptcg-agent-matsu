import unittest

from agents import make_agent
from agents.base import BaseAgent
from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from tests.support import (card, observation, player, pokemon, select,
                           synthetic_card_index)


def greedy(seed=1):
    return GreedyAgent(seed=seed, card_index=synthetic_card_index())


def main_select(options):
    return select(options, sel_type=0, context=0, min_count=1, max_count=1)


class TestBaseAgent(unittest.TestCase):
    def test_deck_returned_when_select_none(self):
        deck = list(range(1, 61))
        agent = RandomAgent(seed=1, deck=deck)
        self.assertEqual(agent.act({"select": None, "current": None}), deck)
        with self.assertRaises(ValueError):
            RandomAgent(seed=1).act({"select": None, "current": None})

    def test_broken_choose_falls_back_to_legal_random(self):
        class Broken(BaseAgent):
            def choose(self, view):
                raise RuntimeError("boom")

        agent = Broken(seed=5)
        obs = observation(main_select([{"type": 14}, {"type": 12}]))
        action = agent.act(obs)
        self.assertEqual(len(action), 1)
        self.assertIn(action[0], (0, 1))
        self.assertEqual(agent.fallback_count, 1)

    def test_factory(self):
        self.assertIsInstance(make_agent("random", seed=1), RandomAgent)
        self.assertIsInstance(make_agent("greedy", seed=1), GreedyAgent)
        with self.assertRaises(KeyError):
            make_agent("nope", seed=1)


class TestRandomAgent(unittest.TestCase):
    def test_actions_always_legal(self):
        agent = RandomAgent(seed=9)
        obs = observation(select([{"type": 3, "index": i} for i in range(6)],
                                 min_count=0, max_count=3))
        for _ in range(200):
            action = agent.act(obs)
            self.assertTrue(0 <= len(action) <= 3)
            self.assertEqual(len(set(action)), len(action))
        self.assertEqual(agent.fallback_count, 0)


class TestGreedyAgent(unittest.TestCase):
    def test_main_prefers_ko_attack_over_weaker_attack(self):
        # Opp active 102 (Fire, weak to Water, hp 40, ex) vs our Water attacker
        # 101: attack 201 (50 dmg -> 100 with weakness -> KO) beats attack 202
        # (30 dmg) and END.
        me = player(hand=[], active=[pokemon(101, hp=120)])
        opp = player(hand=None, active=[pokemon(102, hp=40, max_hp=60)])
        obs = observation(main_select([
            {"type": 14},                       # END
            {"type": 13, "attackId": 202},      # weaker ATTACK
            {"type": 13, "attackId": 201},      # KO ATTACK
        ]), me=me, opp=opp)
        self.assertEqual(greedy().act(obs), [2])

    def test_main_develops_before_attacking(self):
        # Attacking ends the turn -> playing the supporter must come first;
        # the attack stays available on the next MAIN selection.
        me = player(hand=[card(103)], active=[pokemon(101, hp=120)])
        opp = player(hand=None, active=[pokemon(102, hp=40, max_hp=60)])
        obs = observation(main_select([
            {"type": 14},                       # END
            {"type": 7, "index": 0},            # PLAY supporter
            {"type": 13, "attackId": 201},      # ATTACK (KO available)
        ]), me=me, opp=opp)
        self.assertEqual(greedy().act(obs), [1])

    def test_main_plays_instead_of_end(self):
        me = player(hand=[card(103)], active=[pokemon(101)])
        obs = observation(main_select([{"type": 14},
                                       {"type": 7, "index": 0}]), me=me)
        self.assertEqual(greedy().act(obs), [1])

    def test_cost_context_discards_lowest_value_and_min_count(self):
        # DISCARD (context 8): hand has strong 101 and weak 103;
        # min_count=1, max_count=2 -> discard exactly one, the supporter 103.
        me = player(hand=[card(101), card(103)])
        obs = observation(select(
            [{"type": 3, "area": 2, "index": 0, "playerIndex": 0},
             {"type": 3, "area": 2, "index": 1, "playerIndex": 0}],
            sel_type=1, context=8, min_count=1, max_count=2), me=me)
        self.assertEqual(greedy().act(obs), [1])

    def test_gain_context_takes_max_count_best_first(self):
        # TO_HAND (context 7) from deck: take max_count, prefer 101 over 103.
        deck_cards = [card(103), card(101), card(103)]
        obs = observation(select(
            [{"type": 3, "area": 1, "index": i, "playerIndex": 0}
             for i in range(3)],
            sel_type=1, context=7, min_count=1, max_count=2, deck=deck_cards))
        self.assertEqual(greedy().act(obs), [0, 1])

    def test_damage_target_prefers_low_hp_high_prize(self):
        # DAMAGE (context 15): opp active hp 90 vs opp bench hp 20 -> bench.
        opp = player(hand=None, active=[pokemon(101, hp=90)],
                     bench=[pokemon(102, hp=20, max_hp=60)])
        obs = observation(select(
            [{"type": 3, "area": 4, "index": 0, "playerIndex": 1},
             {"type": 3, "area": 5, "index": 0, "playerIndex": 1}],
            sel_type=1, context=15), opp=opp)
        self.assertEqual(greedy().act(obs), [1])

    def test_heal_target_prefers_most_damaged(self):
        me = player(active=[pokemon(101, hp=100, max_hp=120)],
                    bench=[pokemon(102, hp=10, max_hp=60)])
        obs = observation(select(
            [{"type": 3, "area": 4, "index": 0, "playerIndex": 0},
             {"type": 3, "area": 5, "index": 0, "playerIndex": 0}],
            sel_type=1, context=17), me=me)
        self.assertEqual(greedy().act(obs), [1])

    def test_yes_no_policy(self):
        yes_no = [{"type": 2}, {"type": 1}]  # NO first, YES second
        # ACTIVATE (43) -> YES ; unknown context (999) -> NO
        obs_yes = observation(select(yes_no, sel_type=9, context=43))
        obs_unknown = observation(select(yes_no, sel_type=9, context=999))
        self.assertEqual(greedy().act(obs_yes), [1])
        self.assertEqual(greedy().act(obs_unknown), [0])

    def test_count_context(self):
        numbers = [{"type": 0, "number": n} for n in (1, 3, 2)]
        # DRAW_COUNT (38) -> max number; unknown count context -> min number
        obs_draw = observation(select(numbers, sel_type=8, context=38))
        obs_unknown = observation(select(numbers, sel_type=8, context=777))
        self.assertEqual(greedy().act(obs_draw), [1])
        self.assertEqual(greedy().act(obs_unknown), [0])

    def test_unknown_cards_and_option_types_never_crash(self):
        # Unknown card IDs everywhere + an unknown option type + unknown enums.
        me = player(hand=[card(987654)], active=[pokemon(999999, hp=50)])
        opp = player(hand=None, active=[pokemon(888888, hp=50)])
        obs = observation(select(
            [{"type": 999, "weird": True},
             {"type": 13, "attackId": 55555},
             {"type": 14}],
            sel_type=0, context=0), me=me, opp=opp)
        agent = greedy()
        action = agent.act(obs)
        self.assertEqual(len(action), 1)
        self.assertEqual(agent.fallback_count, 0)  # handled, not fallback

    def test_unknown_select_context_stays_legal(self):
        # min_count 0 + unknown context -> commit to nothing (k = lo = 0).
        obs = observation(select(
            [{"type": 3, "area": 2, "index": 0, "playerIndex": 0}],
            sel_type=1, context=12345, min_count=0, max_count=1))
        agent = greedy()
        self.assertEqual(agent.act(obs), [])
        self.assertEqual(agent.fallback_count, 0)

    def test_deterministic_same_seed_same_obs(self):
        me = player(hand=[card(101), card(103)], active=[pokemon(101)])
        obs = observation(main_select(
            [{"type": 14}, {"type": 7, "index": 0}, {"type": 7, "index": 1}]),
            me=me)
        a, b = greedy(seed=42), greedy(seed=42)
        for _ in range(20):
            self.assertEqual(a.act(obs), b.act(obs))


if __name__ == "__main__":
    unittest.main()
