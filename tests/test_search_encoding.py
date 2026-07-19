import unittest
from types import SimpleNamespace

from agents.features import FEATURE_NAMES_V2
from agents.search_encoding import (ACTION_FEATURE_NAMES, SearchSession,
                                    encode_actions, encode_state)
from tests.support import card, observation, player, pokemon, select


class TestStateEncoding(unittest.TestCase):
    def test_is_fixed_length_and_deterministic(self):
        obs = observation(
            select([{"type": 14}]),
            me=player(hand=[card(999999)], active=[pokemon(999999)],
                      bench=[]),
            opp=player(hand=None, active=[None], bench=[]),
        )
        first = encode_state(obs, 0)
        self.assertEqual(first, encode_state(obs, 0))
        self.assertEqual(len(first), len(FEATURE_NAMES_V2))
        self.assertTrue(all(isinstance(value, float) for value in first))

    def test_empty_observation_is_safe(self):
        encoded = encode_state({}, 0)
        self.assertEqual(len(encoded), len(FEATURE_NAMES_V2))
        self.assertTrue(all(value == value for value in encoded))  # no NaN


class TestActionEncoding(unittest.TestCase):
    def test_all_primary_option_types_and_unknown_are_encoded(self):
        options = [{"type": value, "index": value} for value in range(17)]
        options.append({"type": 999, "mystery": object()})
        encoded = encode_actions(select(options, context=48))
        self.assertEqual(len(encoded.features), 18)
        self.assertTrue(all(len(row) == len(ACTION_FEATURE_NAMES)
                            for row in encoded.features))
        self.assertEqual(encoded.features[0][0], 1.0)
        self.assertEqual(encoded.features[16][16], 1.0)
        self.assertEqual(encoded.features[17][17], 1.0)

    def test_multi_select_decodes_to_original_option_indices(self):
        encoded = encode_actions(select(
            [{"type": 3}, {"type": 3}, {"type": 3}],
            min_count=2, max_count=2,
        ))
        self.assertEqual(encoded.option_indices, (0, 1, 2))
        self.assertEqual(encoded.decode([0.1, 0.9, 0.8]), [1, 2])
        self.assertEqual(encoded.decode([1.0, 1.0, 0.0]), [0, 1])

    def test_empty_options_and_malformed_bounds_are_safe(self):
        encoded = encode_actions(select([], min_count=3, max_count=8))
        self.assertEqual(encoded.features, ())
        self.assertEqual(encoded.decode([]), [])


class TestSearchSession(unittest.TestCase):
    def test_end_is_called_when_search_work_raises(self):
        backend = SimpleNamespace(end_calls=0)
        backend.end = lambda: setattr(backend, "end_calls",
                                      backend.end_calls + 1)
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with SearchSession(backend):
                raise RuntimeError("boom")
        self.assertEqual(backend.end_calls, 1)

    def test_end_is_called_when_begin_raises(self):
        class Backend:
            end_calls = 0

            def begin(self, *args, **kwargs):
                raise ValueError("bad fill")

            def end(self):
                self.end_calls += 1

        backend = Backend()
        with self.assertRaisesRegex(ValueError, "bad fill"):
            with SearchSession(backend) as session:
                session.begin({}, object())
        self.assertEqual(backend.end_calls, 1)


if __name__ == "__main__":
    unittest.main()
