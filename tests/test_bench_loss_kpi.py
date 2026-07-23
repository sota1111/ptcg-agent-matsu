"""Terminal loss classification coverage for SOT-1883 A/B artifacts."""
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from eval.loss_kpi import terminal_loss_cause


class TestTerminalLossCause(unittest.TestCase):
    def test_empty_active_and_bench_is_board_wipe(self):
        current = {"players": [
            {"active": [], "bench": []},
            {"active": [{"id": 1}], "bench": []},
        ]}
        self.assertEqual(terminal_loss_cause(current, 0), "board_wipe")

    def test_surviving_bench_is_not_board_wipe(self):
        current = {"players": [
            {"active": [], "bench": [{"id": 2}]},
            {"active": [{"id": 1}], "bench": []},
        ]}
        self.assertEqual(terminal_loss_cause(current, 0), "other")

    def test_missing_terminal_shape_is_other(self):
        self.assertEqual(terminal_loss_cause({}, 0), "other")


if __name__ == "__main__":
    unittest.main()
