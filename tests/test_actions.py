import unittest

from agents import actions
from agents.observation import adapt
from agents.rng import Rng
from tests.support import observation, select


def _sel(n_options, min_count, max_count, **kw):
    opts = [{"type": 3, "area": 2, "index": i} for i in range(n_options)]
    view = adapt(observation(select(opts, min_count=min_count,
                                    max_count=max_count, **kw)))
    return view.select


class TestActionEnumerator(unittest.TestCase):
    def test_legal_indices_and_mask(self):
        sel = _sel(4, 1, 2)
        self.assertEqual(actions.legal_indices(sel), [0, 1, 2, 3])
        self.assertEqual(actions.legality_mask(sel), [True] * 4)

    def test_count_bounds_clamped(self):
        # engine promises maxCount <= len(option); clamp defensively anyway
        self.assertEqual(actions.count_bounds(_sel(3, 1, 10)), (1, 3))
        self.assertEqual(actions.count_bounds(_sel(3, 0, 2)), (0, 2))
        self.assertEqual(actions.count_bounds(_sel(0, 1, 1)), (0, 0))
        self.assertEqual(actions.count_bounds(_sel(3, -1, 2)), (0, 2))

    def test_validate_accepts_legal(self):
        sel = _sel(4, 1, 2)
        self.assertEqual(actions.validate(sel, [0]), [0])
        self.assertEqual(actions.validate(sel, [3, 1]), [3, 1])

    def test_validate_rejects_illegal(self):
        sel = _sel(4, 1, 2)
        for bad in ([], [0, 1, 2], [4], [-1], [1, 1], "x", [0.5]):
            with self.assertRaises(actions.IllegalActionError, msg=bad):
                actions.validate(sel, bad)

    def test_random_action_always_legal(self):
        rng = Rng(123)
        for n, lo, hi in [(1, 1, 1), (5, 0, 5), (7, 2, 3), (4, 0, 0)]:
            sel = _sel(n, lo, hi)
            for _ in range(200):
                action = actions.random_action(sel, rng)
                actions.validate(sel, action)
                self.assertTrue(lo <= len(action) <= min(hi, n))


if __name__ == "__main__":
    unittest.main()
