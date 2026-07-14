"""Unit tests for the 松竹梅 battle harness pure helpers (SOT-1681).

Engine-free: only the stdlib aggregation logic (Wilson CI, per-pairing tally,
round-robin standings) is exercised. The subprocess/engine parts are covered by
running ``eval/battle_matsu_take_ume.py`` itself.
"""
import unittest

from eval.battle_matsu_take_ume import PairResult, standings, wilson_ci


class TestWilsonCI(unittest.TestCase):
    def test_no_evidence(self):
        self.assertEqual(wilson_ci(0, 0), (0.0, 1.0))

    def test_known_value(self):
        lo, hi = wilson_ci(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)

    def test_clamped(self):
        lo, hi = wilson_ci(10, 10)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)


class TestPairResult(unittest.TestCase):
    def test_win_attribution_respects_seat(self):
        pr = PairResult(a="matsu", b="take")
        # matsu first (seat 0), seat 0 wins => matsu win.
        pr.record(a_first=True, result=0, steps=10, first_player=0, fault_seat=None)
        # take first (seat 0), seat 0 wins => take win.
        pr.record(a_first=False, result=0, steps=12, first_player=0, fault_seat=None)
        self.assertEqual(pr.a_wins, 1)
        self.assertEqual(pr.b_wins, 1)
        self.assertEqual(pr.decided, 2)

    def test_draw_and_unfinished_excluded(self):
        pr = PairResult(a="matsu", b="ume")
        pr.record(a_first=True, result=2, steps=5, first_player=0, fault_seat=None)
        pr.record(a_first=True, result=-1, steps=99999, first_player=0, fault_seat=None)
        self.assertEqual(pr.decided, 0)
        self.assertEqual(pr.draws, 1)
        self.assertEqual(pr.unfinished, 1)
        self.assertIsNone(pr.to_dict()["a_win_rate"])

    def test_fault_attributed_to_seat_occupant(self):
        pr = PairResult(a="matsu", b="take")
        # take first (seat 0); seat 1 (=matsu) faults => matsu loses, fault on matsu.
        pr.record(a_first=False, result=0, steps=3, first_player=0, fault_seat=1)
        d = pr.to_dict()
        self.assertEqual(d["faults"]["matsu"], 1)
        self.assertEqual(d["faults"]["take"], 0)
        self.assertEqual(pr.b_wins, 1)  # take won

    def test_first_player_win_rate(self):
        pr = PairResult(a="matsu", b="take")
        pr.record(a_first=True, result=0, steps=1, first_player=0, fault_seat=None)  # first won
        pr.record(a_first=True, result=1, steps=1, first_player=0, fault_seat=None)  # second won
        self.assertEqual(pr.to_dict()["first_player_win_rate"], 0.5)


class TestStandings(unittest.TestCase):
    def test_round_robin_totals_and_sort(self):
        # matsu beats take 3-1; matsu beats ume 4-0; take beats ume 2-2.
        p1 = PairResult(a="matsu", b="take", a_wins=3, b_wins=1)
        p2 = PairResult(a="matsu", b="ume", a_wins=4, b_wins=0)
        p3 = PairResult(a="take", b="ume", a_wins=2, b_wins=2)
        table = standings([p1, p2, p3], ["matsu", "take", "ume"])
        row = {r["contestant"]: r for r in table}
        self.assertEqual(row["matsu"]["wins"], 7)
        self.assertEqual(row["matsu"]["losses"], 1)
        self.assertEqual(row["take"]["wins"], 3)   # 1 (vs matsu) + 2 (vs ume)
        self.assertEqual(row["ume"]["losses"], 6)  # 4 (matsu) + 2 (take)
        self.assertEqual(table[0]["contestant"], "matsu")  # sorted best-first


if __name__ == "__main__":
    unittest.main()
