"""SOT-1708 KPI recording/report tests — engine-independent.

Covers the pure parts of eval/kpi.py and eval/kpi_report.py: loss
classification from a terminal observation, KPI record construction from
per-match dicts, converters from existing bench report shapes, JSONL history
round-trip, and the trend judgement (改善/悪化/横ばい, fault gate).
"""
import json
import os
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from eval import kpi, kpi_report


def match(deck="01_x.csv", won=False, lost=False, draw=False, cause=None,
          **over):
    rec = {
        "deck": deck, "matsu_won": won, "matsu_lost": lost, "draw": draw,
        "unfinished": not (won or lost or draw), "loss_cause": cause,
        "rejects": 0, "exceptions": 0, "fallbacks": 0,
        "budget_violations": 0, "planner_fallbacks": 0, "degraded": 0,
        "emergency_fallbacks": 0, "greedy_handoffs": 0,
        "move_time_ms_sum": 100.0, "move_count": 10,
        "move_time_ms_max": 30.0,
    }
    rec.update(over)
    return rec


def terminal(my_deck=10, opp_prize=3, active=(1,), bench=(2,)):
    return {"players": [
        {"deckCount": my_deck, "active": list(active), "bench": list(bench),
         "prize": [1] * 4},
        {"deckCount": 20, "active": [9], "bench": [8], "prize": [1] * opp_prize},
    ]}


class TestClassifyLoss(unittest.TestCase):
    def test_no_active_wins_over_deck_out(self):
        cur = terminal(my_deck=0, active=(None,), bench=())
        self.assertEqual(kpi.classify_loss(cur, 0), "no_active")

    def test_deck_out(self):
        cur = terminal(my_deck=0, opp_prize=2)
        self.assertEqual(kpi.classify_loss(cur, 0), "deck_out")

    def test_prize_race(self):
        cur = terminal(opp_prize=0)
        self.assertEqual(kpi.classify_loss(cur, 0), "prize_race")

    def test_other(self):
        self.assertEqual(kpi.classify_loss(terminal(), 0), "other")

    def test_seat_1_uses_second_player(self):
        cur = {"players": [
            {"deckCount": 30, "active": [1], "bench": [], "prize": [1]},
            {"deckCount": 0, "active": [2], "bench": [], "prize": [1, 1]},
        ]}
        self.assertEqual(kpi.classify_loss(cur, 1), "deck_out")


class TestBuildRecord(unittest.TestCase):
    def matches(self):
        return [
            match("01_a.csv", won=True),
            match("02_b.csv", won=True),
            match("03_c.csv", lost=True, cause="deck_out"),
            match("04_d.csv", draw=True),
        ]

    def test_kpi_values(self):
        rec = kpi.build_record(self.matches(), issue="SOT-TEST", seed=1)
        k = rec["kpis"]
        self.assertEqual(rec["schema"], kpi.SCHEMA)
        self.assertEqual(rec["issue"], "SOT-TEST")
        self.assertEqual(rec["n_matches"], 4)
        self.assertEqual(rec["n_decks"], 4)
        wr = k["mirror_winrate_vs_greedy"]
        self.assertAlmostEqual(wr["value"], round(2 / 3, 4))
        self.assertEqual((wr["wins"], wr["losses"], wr["draws"]), (2, 1, 1))
        lo, hi = wr["ci95"]
        self.assertTrue(0.0 <= lo < 2 / 3 < hi <= 1.0)
        do = k["self_deck_out_loss_rate"]
        self.assertEqual(do["value"], 1.0)
        self.assertEqual(do["deck_out_losses"], 1)
        self.assertEqual(k["fault_total"]["value"], 0)
        dt = k["decision_time_mean_ms"]
        self.assertAlmostEqual(dt["value"], 10.0)  # 400ms / 40 moves
        self.assertEqual(dt["max_ms"], 30.0)
        self.assertEqual(dt["budget_violations"], 0)

    def test_faults_counted(self):
        ms = self.matches()
        ms[0]["rejects"] = 1
        ms[1]["budget_violations"] = 2
        rec = kpi.build_record(ms, issue="SOT-TEST")
        ft = rec["kpis"]["fault_total"]
        self.assertEqual(ft["value"], 3)
        self.assertEqual(ft["breakdown"]["rejects"], 1)
        self.assertEqual(
            rec["kpis"]["decision_time_mean_ms"]["budget_violations"], 2)

    def test_no_losses_gives_null_deck_out_rate(self):
        rec = kpi.build_record([match(won=True)], issue="SOT-TEST")
        self.assertIsNone(rec["kpis"]["self_deck_out_loss_rate"]["value"])


class TestConverters(unittest.TestCase):
    def test_from_bench_decks(self):
        report = {
            "issue": "SOT-1693", "n_matches": 50, "seed": 7,
            "wins_a_mcts": 30, "wins_b_greedy": 18, "draws": 2,
            "winrate_a_excl_draws": 30 / 48,
            "wilson95_excl_draws": [0.48, 0.75],
            "faults": {"rejects": 0, "exceptions": 0,
                       "budget_violations_a": 1},
            "per_deck": {"01_a.csv": {}, "02_b.csv": {}},
            "a_move_time_ms": {"mean": 812.5, "max": 4001.0},
        }
        rec = kpi.record_from_bench_decks(report)
        self.assertEqual(rec["issue"], "SOT-1693")
        self.assertEqual(rec["source"], "bench_decks")
        self.assertEqual(rec["n_decks"], 2)
        k = rec["kpis"]
        self.assertAlmostEqual(k["mirror_winrate_vs_greedy"]["value"], 0.625)
        self.assertIsNone(k["self_deck_out_loss_rate"]["value"])
        self.assertEqual(k["fault_total"]["value"], 1)
        self.assertEqual(k["decision_time_mean_ms"]["budget_violations"], 1)

    def test_from_bench(self):
        report = {
            "agent_a": "mcts", "agent_b": "greedy", "n_matches": 100,
            "seed": 3, "deck": "deck.csv",
            "wins_a": 60, "wins_b": 40, "draws": 0, "unfinished": 0,
            "rejects": 0, "exceptions": 0, "fallbacks_a": 0,
            "budget_violations_a": 0, "planner_fallbacks_a": 0,
            "degraded_count_a": 0,
            "winrate_a_excl_draws": 0.6,
            "wilson95_excl_draws": [0.5, 0.69],
            "time_per_decision_ms": {"mean": 5.2, "max": 40.0},
            "planner_move_max_ms": 900.0,
        }
        rec = kpi.record_from_bench(report, issue="SOT-X")
        self.assertEqual(rec["issue"], "SOT-X")
        self.assertEqual(rec["opponent"], "greedy")
        k = rec["kpis"]
        self.assertAlmostEqual(k["mirror_winrate_vs_greedy"]["value"], 0.6)
        self.assertEqual(k["fault_total"]["value"], 0)
        self.assertEqual(k["decision_time_mean_ms"]["max_ms"], 900.0)


class TestHistory(unittest.TestCase):
    def test_append_and_load_roundtrip(self):
        rec = kpi.build_record([match(won=True)], issue="SOT-TEST")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "kpi_history.jsonl")
            kpi.append_history(rec, path)
            kpi.append_history(rec, path)
            loaded = kpi.load_history(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["kpis"], rec["kpis"])
        self.assertEqual(kpi.load_history("/nonexistent/x.jsonl"), [])

    def test_history_line_is_single_json(self):
        rec = kpi.build_record([match(won=True)], issue="SOT-TEST")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "h.jsonl")
            kpi.append_history(rec, path)
            with open(path) as f:
                lines = f.read().splitlines()
        self.assertEqual(len(lines), 1)
        json.loads(lines[0])


def rec_with(values: dict) -> dict:
    return {"kpis": {name: {"value": v} for name, v in values.items()}}


class TestReportJudgement(unittest.TestCase):
    def test_winrate_up_is_improvement(self):
        self.assertEqual(kpi_report.judge(
            "mirror_winrate_vs_greedy", 0.60, 0.70), "改善")

    def test_deck_out_up_is_worse(self):
        self.assertEqual(kpi_report.judge(
            "self_deck_out_loss_rate", 0.40, 0.55), "悪化")

    def test_small_move_is_flat(self):
        self.assertEqual(kpi_report.judge(
            "mirror_winrate_vs_greedy", 0.700, 0.702), "横ばい")

    def test_fault_gate(self):
        self.assertEqual(kpi_report.judge("fault_total", 0, 0), "OK(=0)")
        self.assertTrue(kpi_report.judge("fault_total", 0, 2).startswith("NG"))

    def test_missing_value(self):
        self.assertEqual(kpi_report.judge(
            "self_deck_out_loss_rate", None, 0.5), "n/a")

    def test_compare_covers_all_kpis(self):
        prev = rec_with({"mirror_winrate_vs_greedy": 0.6,
                         "self_deck_out_loss_rate": 0.5,
                         "fault_total": 0, "decision_time_mean_ms": 800.0})
        latest = rec_with({"mirror_winrate_vs_greedy": 0.7,
                           "self_deck_out_loss_rate": 0.4,
                           "fault_total": 0, "decision_time_mean_ms": 900.0})
        c = kpi_report.compare(prev, latest)
        self.assertEqual(set(c), set(kpi.KPI_DIRECTIONS))
        self.assertEqual(c["mirror_winrate_vs_greedy"]["judgement"], "改善")
        self.assertEqual(c["self_deck_out_loss_rate"]["judgement"], "改善")
        self.assertEqual(c["decision_time_mean_ms"]["judgement"], "悪化")
        self.assertAlmostEqual(c["mirror_winrate_vs_greedy"]["delta"], 0.1)


if __name__ == "__main__":
    unittest.main()
