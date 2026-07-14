"""SOT-1676 tests — card embeddings (word2vec) and the v2 board vector.

Engine-independent: the card master is synthetic (SimpleNamespace shaped
like cg.api.CardData/Attack, with effect text), embeddings are trained
in-process on that master, and observations reuse tests/support.py.
"""
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from agents.cards import CardEmbeddings, DEFAULT_EMBED_DIM
from agents.evaluator import LearnedEvaluator
from agents.features import (FEATURE_NAMES, FEATURE_NAMES_V2,
                             feature_names_v2, featurize_v2, make_featurizer)
from tests.support import (card, observation, player, pokemon, select,
                           synthetic_card_index)


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "train_embeddings", os.path.join(REPO, "train", "train_embeddings.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _obs(me=None, opp=None, **state):
    return observation(select([{"type": 0}]), me=me, opp=opp, **state)


# Synthetic card master WITH effect text (test-only IDs, as in support.py).
# 101/102 are near-duplicates (same type/attack text); 103 is unrelated.
def _text_master():
    cards = [
        SimpleNamespace(cardId=101, name="alphamon", cardType=0,
                        retreatCost=1, hp=120, weakness=2, resistance=None,
                        energyType=3, basic=True, stage1=False, stage2=False,
                        ex=False, megaEx=False, tera=False, aceSpec=False,
                        evolvesFrom=None, skills=[], attacks=[201]),
        SimpleNamespace(cardId=102, name="alphamon ex", cardType=0,
                        retreatCost=1, hp=130, weakness=2, resistance=None,
                        energyType=3, basic=True, stage1=False, stage2=False,
                        ex=True, megaEx=False, tera=False, aceSpec=False,
                        evolvesFrom=None, skills=[], attacks=[201]),
        SimpleNamespace(cardId=103, name="draw helper", cardType=3,
                        retreatCost=0, hp=0, weakness=None, resistance=None,
                        energyType=0, basic=False, stage1=False, stage2=False,
                        ex=False, megaEx=False, tera=False, aceSpec=False,
                        evolvesFrom=None,
                        skills=[SimpleNamespace(
                            name="research",
                            text="Discard your hand and draw 7 cards.")],
                        attacks=[]),
    ]
    attacks = [
        SimpleNamespace(attackId=201, name="splash strike",
                        text="Flip a coin. If heads, this attack does 30 "
                             "more damage.", damage=50, energies=[3]),
    ]
    return cards, attacks


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class TestTrainEmbeddings(unittest.TestCase):
    def test_same_seed_reproduces_payload(self):
        te = _load_trainer()
        cards, attacks = _text_master()
        first = te.build_embeddings(cards, attacks, dim=6, epochs=3, seed=42)
        second = te.build_embeddings(cards, attacks, dim=6, epochs=3, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(json.dumps(first, sort_keys=True),
                         json.dumps(second, sort_keys=True))

    def test_different_seed_differs(self):
        te = _load_trainer()
        cards, attacks = _text_master()
        first = te.build_embeddings(cards, attacks, dim=6, epochs=3, seed=42)
        second = te.build_embeddings(cards, attacks, dim=6, epochs=3, seed=43)
        self.assertNotEqual(first["cards"], second["cards"])

    def test_payload_shape_and_dim(self):
        te = _load_trainer()
        cards, attacks = _text_master()
        payload = te.build_embeddings(cards, attacks, dim=6, epochs=2, seed=1)
        self.assertEqual(payload["dim"], 6)
        self.assertEqual(set(payload["cards"]), {"101", "102", "103"})
        for vec in payload["cards"].values():
            self.assertEqual(len(vec), 6)
            self.assertTrue(all(math.isfinite(v) for v in vec))
        self.assertEqual(len(payload["default"]), 6)

    def test_similar_cards_are_closer(self):
        """Qualitative property: near-duplicate cards (shared attack text /
        attributes) embed closer than an unrelated Trainer card."""
        te = _load_trainer()
        cards, attacks = _text_master()
        payload = te.build_embeddings(cards, attacks, dim=6, epochs=5, seed=7)
        near = _cosine(payload["cards"]["101"], payload["cards"]["102"])
        far = _cosine(payload["cards"]["101"], payload["cards"]["103"])
        self.assertGreater(near, far)


class TestCardEmbeddings(unittest.TestCase):
    def _table(self):
        return CardEmbeddings(dim=3, vectors={"101": [1.0, 2.0, 3.0]},
                              default=[0.5, 0.5, 0.5])

    def test_unknown_and_none_fall_back_to_default(self):
        emb = self._table()
        self.assertEqual(emb.vector(101), [1.0, 2.0, 3.0])
        self.assertEqual(emb.vector(999), [0.5, 0.5, 0.5])
        self.assertEqual(emb.vector(None), [0.5, 0.5, 0.5])

    def test_mean(self):
        emb = self._table()
        self.assertEqual(emb.mean([]), [0.0, 0.0, 0.0])
        self.assertEqual(emb.mean([101, 999]), [0.75, 1.25, 1.75])

    def test_dim_mismatch_raises(self):
        with self.assertRaises(ValueError):
            CardEmbeddings(dim=3, vectors={"101": [1.0, 2.0]})
        with self.assertRaises(ValueError):
            CardEmbeddings(dim=3, default=[1.0])

    def test_empty_and_load_roundtrip(self):
        self.assertEqual(CardEmbeddings.empty(4).vector(1), [0.0] * 4)
        payload = {"dim": 3, "cards": {"101": [1.0, 2.0, 3.0]},
                   "default": [0.5, 0.5, 0.5]}
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as f:
            json.dump(payload, f)
        try:
            emb = CardEmbeddings.load(f.name)
            self.assertEqual(emb.dim, 3)
            self.assertEqual(emb.vector(101), [1.0, 2.0, 3.0])
            self.assertEqual(emb.vector(7), [0.5, 0.5, 0.5])
        finally:
            os.unlink(f.name)


class TestFeaturizeV2(unittest.TestCase):
    def _emb(self):
        return CardEmbeddings(
            dim=4,
            vectors={"101": [1.0, 0.0, 0.0, 0.0], "102": [0.0, 1.0, 0.0, 0.0],
                     "103": [0.0, 0.0, 1.0, 0.0]},
            default=[0.25, 0.25, 0.25, 0.25])

    def test_vector_matches_feature_names(self):
        emb = self._emb()
        names = feature_names_v2(emb.dim)
        x = featurize_v2(_obs(), 0, synthetic_card_index(), emb)
        self.assertEqual(len(x), len(names))
        self.assertTrue(all(isinstance(v, float) for v in x))
        self.assertTrue(all(math.isfinite(v) for v in x))

    def test_default_dim_matches_module_constant(self):
        x = featurize_v2(_obs(), 0)
        self.assertEqual(len(x), len(FEATURE_NAMES_V2))
        self.assertEqual(len(feature_names_v2(DEFAULT_EMBED_DIM)),
                         len(FEATURE_NAMES_V2))

    def test_dict_and_namespace_shapes_agree(self):
        emb = self._emb()
        obs = _obs(me=player(active=[pokemon(101, hp=70, max_hp=100,
                                             energies=(0, 0))],
                             hand=[card(103)], discard=[card(102)],
                             prize=4, deck_count=0),
                   opp=player(bench=[pokemon(102)], hand_count=2))
        for root in (0, 1):
            self.assertEqual(
                featurize_v2(obs, root, synthetic_card_index(), emb),
                featurize_v2(_to_namespace(obs), root,
                             synthetic_card_index(), emb))

    def test_perspective_flips_sides(self):
        emb = self._emb()
        obs = _obs(me=player(prize=3, active=[pokemon(101)]),
                   opp=player(prize=6))
        names = feature_names_v2(emb.dim)
        x0 = featurize_v2(obs, 0, synthetic_card_index(), emb)
        x1 = featurize_v2(obs, 1, synthetic_card_index(), emb)
        side = (len(names) - (3 + emb.dim)) // 2  # my_/opp_ blocks swap
        self.assertEqual(x0[:side], x1[side:2 * side])
        self.assertEqual(x0[side:2 * side], x1[:side])
        self.assertNotEqual(x0[names.index("my_turn")],
                            x1[names.index("my_turn")])

    def test_slots_embeddings_and_fallbacks(self):
        emb = self._emb()
        names = feature_names_v2(emb.dim)
        obs = _obs(me=player(active=[pokemon(101)],
                             bench=[pokemon(999)]),   # unknown card ID
                   opp=player(active=[None]))         # facedown Active
        x = featurize_v2(obs, 0, synthetic_card_index(), emb)

        def block(prefix):
            i = names.index(f"{prefix}_emb0")
            return x[i:i + emb.dim]

        self.assertEqual(block("my_slot0"), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(block("my_slot1"), emb.default)  # unknown -> default
        self.assertEqual(block("opp_slot0"), emb.default)  # facedown
        self.assertEqual(x[names.index("opp_slot0_present")], 1.0)
        self.assertEqual(x[names.index("opp_slot0_hidden")], 1.0)
        self.assertEqual(x[names.index("my_slot2_present")], 0.0)
        self.assertEqual(block("my_slot2"), [0.0] * emb.dim)  # empty slot

    def test_hand_visibility_gates_embedding(self):
        emb = self._emb()
        names = feature_names_v2(emb.dim)
        obs = _obs(me=player(hand=[card(101), card(102)]),
                   opp=player(hand=None, hand_count=5))
        x = featurize_v2(obs, 0, synthetic_card_index(), emb)
        self.assertEqual(x[names.index("my_hand_known")], 1.0)
        i = names.index("my_hand_emb0")
        self.assertEqual(x[i:i + emb.dim], [0.5, 0.5, 0.0, 0.0])
        self.assertEqual(x[names.index("opp_hand_known")], 0.0)
        j = names.index("opp_hand_emb0")
        self.assertEqual(x[j:j + emb.dim], [0.0] * emb.dim)

    def test_stadium_embedding(self):
        emb = self._emb()
        names = feature_names_v2(emb.dim)
        with_stadium = featurize_v2(_obs(stadium=[card(103)]), 0,
                                    synthetic_card_index(), emb)
        without = featurize_v2(_obs(), 0, synthetic_card_index(), emb)
        i = names.index("stadium_present")
        self.assertEqual(with_stadium[i], 1.0)
        self.assertEqual(without[i], 0.0)
        j = names.index("stadium_emb0")
        self.assertEqual(with_stadium[j:j + emb.dim], [0.0, 0.0, 1.0, 0.0])
        self.assertEqual(without[j:j + emb.dim], [0.0] * emb.dim)

    def test_degenerate_observations_do_not_crash(self):
        for obs in ({}, {"current": None}, {"current": {"players": []}}):
            self.assertEqual(len(featurize_v2(obs, 0)),
                             len(FEATURE_NAMES_V2))

    def test_deterministic(self):
        emb = self._emb()
        obs = _obs(me=player(active=[pokemon(101)], hand=[card(102)]))
        self.assertEqual(featurize_v2(obs, 0, synthetic_card_index(), emb),
                         featurize_v2(obs, 0, synthetic_card_index(), emb))


class TestFeatureSetSelection(unittest.TestCase):
    def test_make_featurizer_v1_is_default(self):
        names, fn = make_featurizer(None)
        self.assertEqual(names, FEATURE_NAMES)
        names, _ = make_featurizer("v1")
        self.assertEqual(names, FEATURE_NAMES)
        with self.assertRaises(ValueError):
            make_featurizer("v3")

    def test_make_featurizer_v2_uses_injected_embeddings(self):
        emb = CardEmbeddings.empty(4)
        names, fn = make_featurizer("v2", embeddings=emb)
        self.assertEqual(names, feature_names_v2(4))
        x = fn(_obs(), 0, synthetic_card_index())
        self.assertEqual(len(x), len(names))

    def test_learned_evaluator_v2_model(self):
        emb = CardEmbeddings.empty(4)
        names = feature_names_v2(4)
        model = {"feature_set": "v2", "feature_names": list(names),
                 "weights": [0.0] * len(names), "bias": 0.0,
                 "mean": [0.0] * len(names), "std": [1.0] * len(names)}
        ev = LearnedEvaluator(model=model, embeddings=emb)
        value = ev.evaluate(_to_namespace(_obs()), 0)
        self.assertEqual(value, 0.5)

    def test_learned_evaluator_v2_name_mismatch_raises(self):
        emb = CardEmbeddings.empty(4)
        model = {"feature_set": "v2", "feature_names": ["stale"],
                 "weights": [1.0]}
        with self.assertRaises(ValueError):
            LearnedEvaluator(model=model, embeddings=emb)

    def test_learned_evaluator_v1_model_still_loads(self):
        model = {"feature_names": list(FEATURE_NAMES),
                 "weights": [0.0] * len(FEATURE_NAMES), "bias": 0.0}
        ev = LearnedEvaluator(model=model)
        self.assertEqual(ev.evaluate(_to_namespace(_obs()), 0), 0.5)


if __name__ == "__main__":
    unittest.main()
