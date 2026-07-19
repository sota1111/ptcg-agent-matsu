"""Unit tests for the 松竹梅 battle harness pure helpers (SOT-1681).

Engine-free: only the stdlib aggregation logic (Wilson CI, per-pairing tally,
round-robin standings) is exercised. The subprocess/engine parts are covered by
running ``eval/battle_matsu_take_ume.py`` itself.
"""
import os
import random
import tempfile
import unittest

from eval.battle_matsu_take_ume import (
    PairResult,
    aggregate_reports,
    build_deck_schedule,
    resolve_deck,
    deck_usage,
    make_sandbox,
    standings,
    wilson_ci,
)


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


class TestDeckSchedule(unittest.TestCase):
    POOL = [f"{i:02d}_deck.csv" for i in range(1, 26)]

    def test_mirror_pairs_share_one_deck(self):
        sched = build_deck_schedule(6, self.POOL, "mirror", random.Random(0))
        self.assertEqual(len(sched), 6)
        # both contestants play the same deck each match ...
        for a, b in sched:
            self.assertEqual(a, b)
        # ... and each 先後 pair (2k, 2k+1) reuses one deck so the swap cancels it.
        self.assertEqual(sched[0], sched[1])
        self.assertEqual(sched[2], sched[3])
        self.assertEqual(sched[4], sched[5])

    def test_mirror_odd_n_has_lone_final_match(self):
        sched = build_deck_schedule(5, self.POOL, "mirror", random.Random(1))
        self.assertEqual(len(sched), 5)
        self.assertEqual(sched[0], sched[1])
        self.assertEqual(sched[2], sched[3])

    def test_independent_draws_per_contestant(self):
        sched = build_deck_schedule(40, self.POOL, "independent", random.Random(2))
        self.assertEqual(len(sched), 40)
        # over 40 matches the two slots differ at least once (independent draws).
        self.assertTrue(any(a != b for a, b in sched))
        for a, b in sched:
            self.assertIn(a, self.POOL)
            self.assertIn(b, self.POOL)

    def test_seeded_schedule_is_reproducible(self):
        s1 = build_deck_schedule(10, self.POOL, "mirror", random.Random(42))
        s2 = build_deck_schedule(10, self.POOL, "mirror", random.Random(42))
        self.assertEqual(s1, s2)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            build_deck_schedule(2, self.POOL, "bogus", random.Random(0))

    def test_deck_usage_counts_all_slots(self):
        sched = build_deck_schedule(4, self.POOL, "mirror", random.Random(3))
        usage = deck_usage(sched)
        # 4 matches * 2 contestant-slots = 8 slots counted.
        self.assertEqual(sum(usage.values()), 8)

    def test_resolve_deck_by_id(self):
        with tempfile.TemporaryDirectory() as root:
            expected = os.path.join(root, "07_example.csv")
            with open(expected, "w", encoding="utf-8") as fh:
                fh.write("1\n")
            self.assertEqual(resolve_deck(root, "07"), expected)

    def test_resolve_deck_rejects_unknown_id(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(SystemExit):
                resolve_deck(root, "99")


class TestAggregateReports(unittest.TestCase):
    @staticmethod
    def _shard(a_wins, b_wins, seed):
        pr = PairResult(a="matsu", b="take")
        pr.n = a_wins + b_wins
        pr.a_wins, pr.b_wins = a_wins, b_wins
        pr.first_decided, pr.first_wins = pr.n, a_wins
        pr.steps_total = pr.n * 10
        contestants = [{"label": lb, "kanji": lb, "repo": lb}
                       for lb in ("matsu", "take")]
        return {
            "n_per_pairing": pr.n,
            "deck_selection": {"mode": "mirror", "random": True,
                               "pool_size": 25, "pool": [], "seed": seed},
            "contestants": contestants,
            "pairings": [pr.to_dict()],
            "standings": standings([pr], ["matsu", "take"]),
        }

    def test_shards_sum_exactly(self):
        rep = aggregate_reports([self._shard(6, 4, 1), self._shard(5, 5, 2)])
        self.assertEqual(rep["aggregated_from"], 2)
        self.assertEqual(rep["n_per_pairing"], 20)
        pd = rep["pairings"][0]
        self.assertEqual(pd["a_wins"], 11)
        self.assertEqual(pd["b_wins"], 9)
        self.assertEqual(pd["a_win_rate"], 0.55)
        self.assertEqual(rep["deck_selection"]["seed"], [1, 2])
        # standings reflect the summed tally
        top = rep["standings"][0]
        self.assertEqual(top["contestant"], "matsu")
        self.assertEqual(top["wins"], 11)

    def test_from_dict_roundtrip(self):
        pr = PairResult(a="matsu", b="ume", n=4, a_wins=3, b_wins=1,
                        first_decided=4, first_wins=2, steps_total=40)
        back = PairResult.from_dict(pr.to_dict())
        self.assertEqual(back.a_wins, 3)
        self.assertEqual(back.b_wins, 1)
        self.assertEqual(back.first_wins, 2)
        self.assertEqual(back.steps_total, 40)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_reports([])


class TestSandbox(unittest.TestCase):
    """The deck-sync sandbox must never let a deck rewrite reach the repo."""

    def _fake_repo(self) -> str:
        repo = tempfile.mkdtemp(prefix="sot1681_repo_")
        with open(os.path.join(repo, "deck.csv"), "w", encoding="utf-8") as fh:
            fh.write("1\n2\n3\n")
        os.mkdir(os.path.join(repo, "agents"))
        with open(os.path.join(repo, "main.py"), "w", encoding="utf-8") as fh:
            fh.write("# main\n")
        return repo

    def test_deck_csv_is_a_real_copy_not_a_symlink(self):
        repo = self._fake_repo()
        sb = make_sandbox(repo)
        deck = os.path.join(sb, "deck.csv")
        self.assertTrue(os.path.isfile(deck))
        self.assertFalse(os.path.islink(deck))  # rewrites must stay in the sandbox

    def test_other_entries_are_symlinks(self):
        repo = self._fake_repo()
        sb = make_sandbox(repo)
        self.assertTrue(os.path.islink(os.path.join(sb, "main.py")))
        self.assertTrue(os.path.islink(os.path.join(sb, "agents")))

    def test_rewriting_sandbox_deck_leaves_repo_deck_untouched(self):
        repo = self._fake_repo()
        sb = make_sandbox(repo)
        with open(os.path.join(sb, "deck.csv"), "w", encoding="utf-8") as fh:
            fh.write("99\n98\n97\n")  # simulate a __set_deck__ rewrite
        with open(os.path.join(repo, "deck.csv"), encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "1\n2\n3\n")


if __name__ == "__main__":
    unittest.main()
