import unittest

from agents.rng import Rng


class TestRng(unittest.TestCase):
    def test_same_seed_same_sequence(self):
        a, b = Rng(42), Rng(42)
        self.assertEqual([a.randint(0, 1000) for _ in range(50)],
                         [b.randint(0, 1000) for _ in range(50)])

    def test_different_seeds_differ(self):
        a, b = Rng(1), Rng(2)
        self.assertNotEqual([a.randint(0, 10 ** 9) for _ in range(10)],
                            [b.randint(0, 10 ** 9) for _ in range(10)])

    def test_child_deterministic_and_independent(self):
        base = Rng(7)
        c1, c2 = Rng(7).child("x"), Rng(7).child("x")
        self.assertEqual(c1.seed, c2.seed)
        self.assertEqual([c1.randint(0, 1000) for _ in range(20)],
                         [c2.randint(0, 1000) for _ in range(20)])
        self.assertNotEqual(Rng(7).child("x").seed, Rng(7).child("y").seed)
        self.assertNotEqual(base.seed, base.child("x").seed)

    def test_sample_no_duplicates(self):
        r = Rng(3)
        for _ in range(100):
            picked = r.sample(range(10), 5)
            self.assertEqual(len(set(picked)), 5)


if __name__ == "__main__":
    unittest.main()
