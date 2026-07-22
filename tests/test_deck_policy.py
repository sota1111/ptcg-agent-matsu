"""Deck-specific search allocation selected by SOT-1733 evidence."""
import glob
import os
import unittest

from agents.deck_policy import deck_id, search_overrides


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    with open(path) as f:
        return [int(x) for x in f.read().splitlines()[:60]]


class TestDeckPolicy(unittest.TestCase):
    def test_all_25_decks_are_deliberately_classified(self):
        paths = glob.glob(os.path.join(REPO, "decks", "rotation_baseline", "[0-9][0-9]_*.csv"))
        enabled = {int(os.path.basename(p)[:2])
                   for p in paths if search_overrides(load(p))}
        self.assertEqual(len(paths), 25)
        self.assertEqual(enabled, set(range(1, 26)) - {6, 8, 11, 21, 22})

    def test_deck_id_ignores_csv_order(self):
        deck = list(range(60))
        self.assertEqual(deck_id(deck), deck_id(list(reversed(deck))))

    def test_unknown_deck_keeps_champion_policy(self):
        self.assertEqual(search_overrides(list(range(1000, 1060))), {})

    def test_overrides_are_not_shared_mutable_state(self):
        deck = load(os.path.join(REPO, "decks", "initial", "01_dragapult.csv"))
        first = search_overrides(deck)
        first["eval_weights"]["deck_low"] = 999
        self.assertEqual(search_overrides(deck)["eval_weights"]["deck_low"], -0.2)


if __name__ == "__main__":
    unittest.main()
