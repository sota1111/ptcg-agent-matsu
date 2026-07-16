"""SOT-1678 tests — cheater-determinization self-play data generation.

Engine-independent: the engine is replaced by scripted doubles (the
generator loop by a FakeGame, the planner's search API by the SOT-1672
_ScriptedBackend), so these run on CI. Reproducibility is specified as in
tests/test_mcts.py: with engine responses held fixed, generation is a
deterministic function of the injected seed (ASSUMPTIONS.md A-9 — the real
engine's internal RNG is not injectable).

The fairness boundary (the cheater path must be unusable by battle-time
agents) is pinned here: agents/ never references the true-state source,
MctsAgent/make_agent cannot be configured with a fills hook, and the
submission archive does not include train/.
"""
import json
import os
import re
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from agents import make_agent
from agents.evaluator import HeuristicEvaluator
from agents.features import make_featurizer
from agents.planner import Fills, MctsPlanner, sample_fills
from agents.rng import Rng
from tests.support import (card, observation, player, pokemon, select,
                           synthetic_card_index)
from tests.test_mcts import _ScriptedBackend, main_view
from train import gen_selfplay
from train.cheater import CheaterMctsAgent, parse_true_state, true_fills


def true_player(deck_ids=(101, 103), hand_ids=(102,), prize_ids=(101,),
                active_id=101):
    """A player entry as visualize_data reports it: every zone face-up."""
    return {
        "deck": [card(i) for i in deck_ids],
        "hand": [card(i) for i in hand_ids],
        "prize": [card(i) for i in prize_ids],
        "active": [pokemon(active_id)] if active_id is not None else [],
        "bench": [], "discard": [], "deckCount": len(deck_ids),
        "handCount": len(hand_ids),
    }


def obs_for(your_index, turn, winner=-1):
    """Acting-player observation matching true_state() zone sizes."""
    me = player(active=[pokemon(101)], hand=[card(102)], deck_count=2,
                hand_count=1, prize=1)
    opp = player(active=[pokemon(101)], hand=None, deck_count=2,
                 hand_count=1, prize=1)
    return observation(
        select([{"type": 13, "attackId": 201, "number": 0}], min_count=1,
               max_count=1),
        me=me, opp=opp, your_index=your_index, turn=turn, result=winner)


def true_state():
    return {"players": [true_player(), true_player()], "yourIndex": 0}


class FakeGame:
    """Scripted engine double for the generation loop: a fixed observation
    sequence ending in a result, with a matching true-state dump."""

    def __init__(self, winner=0, turns=2):
        self._script = [obs_for(t % 2, t) for t in range(1, turns + 1)]
        self._script.append(obs_for(turns % 2, turns, winner=winner))
        self._i = 0
        self.finished = 0

    def battle_start(self, deck0, deck1):
        self._i = 0
        return self._script[0], None

    def battle_select(self, action):
        self._i += 1
        return self._script[self._i]

    def visualize_data(self):
        return json.dumps([{"current": true_state(), "selected": []}])

    def battle_finish(self):
        self.finished += 1


class FakeCheaterAgent:
    """act()/set_true_fills() double; deck only satisfies battle_start."""

    def __init__(self, seed):
        self.seed = seed
        self._deck = [101] * 60
        self.fills_seen = []

    def set_true_fills(self, fills):
        self.fills_seen.append(fills)

    def act(self, obs):
        return [0]


V1_FEATURIZE = make_featurizer("v1")[1]


class TestTrueFills(unittest.TestCase):
    def test_exact_zone_contents(self):
        obs = obs_for(your_index=0, turn=1)
        fills = true_fills(obs["current"], true_state())
        self.assertEqual(fills.my_deck, [101, 103])
        self.assertEqual(fills.my_prize, [101])
        self.assertEqual(fills.opp_deck, [101, 103])
        self.assertEqual(fills.opp_prize, [101])
        self.assertEqual(fills.opp_hand, [102])
        self.assertEqual(fills.opp_active, [])  # opponent Active is face-up

    def test_perspective_follows_your_index(self):
        state = true_state()
        state["players"][1] = true_player(hand_ids=(103,))
        obs = obs_for(your_index=1, turn=2)
        fills = true_fills(obs["current"], state)
        self.assertEqual(fills.my_deck, [101, 103])   # players[1] = "me"
        self.assertEqual(fills.opp_hand, [102])       # players[0] = opponent

    def test_facedown_opponent_active_filled_with_true_id(self):
        obs = obs_for(your_index=0, turn=1)
        obs["current"]["players"][1]["active"] = [None]
        state = true_state()
        state["players"][1]["active"] = [pokemon(102)]
        fills = true_fills(obs["current"], state)
        self.assertEqual(fills.opp_active, [102])

    def test_zone_size_mismatch_fails_loudly(self):
        obs = obs_for(your_index=0, turn=1)
        obs["current"]["players"][1]["handCount"] = 5  # stale state
        with self.assertRaises(ValueError):
            true_fills(obs["current"], true_state())

    def test_parse_true_state_takes_last_snapshot(self):
        payload = json.dumps([{"current": {"turn": 1}},
                              {"current": {"turn": 2}}])
        self.assertEqual(parse_true_state(payload), {"turn": 2})
        with self.assertRaises(ValueError):
            parse_true_state("[]")


class TestGenerateCheater(unittest.TestCase):
    def _generate(self, path, seed=61678001, winner=0, sample_p=1.0, n=3):
        return gen_selfplay.generate_cheater(
            FakeGame(winner=winner), FakeCheaterAgent, n, Rng(seed),
            synthetic_card_index(), HeuristicEvaluator(), V1_FEATURIZE,
            sample_p, path)

    def test_same_seed_same_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a.jsonl"), os.path.join(tmp, "b.jsonl")
            stats_a = self._generate(a)
            stats_b = self._generate(b)
            self.assertEqual(stats_a, stats_b)
            with open(a, "rb") as fa, open(b, "rb") as fb:
                self.assertEqual(fa.read(), fb.read())
            self.assertGreater(stats_a["examples"], 0)

    def test_both_perspectives_labelled_with_final_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.jsonl")
            self._generate(path, winner=1, n=1)
            with open(path) as f:
                records = [json.loads(line) for line in f]
            # p=1.0: every turn>0 state sampled, two perspectives each.
            self.assertEqual({r["who"] for r in records}, {0, 1})
            self.assertEqual(len(records) % 2, 0)
            for r in records:
                self.assertEqual(r["y"], float(r["who"] == 1))
                self.assertEqual(r["schema"], "matsu-battle-log-v2")
                self.assertIn(r["outcome"], ("win", "loss"))
                self.assertIn(r["reward"], (-1.0, 1.0))
                self.assertIsInstance(r["legal_actions"], list)
                self.assertIsInstance(r["action"], list)
                self.assertGreater(r["t"], 0)
                self.assertEqual(len(r["x"]), len(make_featurizer("v1")[0]))
            by_turn = {}
            for r in records:
                by_turn.setdefault(r["t"], {})[r["who"]] = r["x"]
            for pair in by_turn.values():
                # my_turn flag differs between the two perspectives.
                self.assertNotEqual(pair[0], pair[1])

    def test_sample_p_zero_records_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.jsonl")
            stats = self._generate(path, sample_p=0.0)
            self.assertEqual(stats["examples"], 0)
            self.assertEqual(stats["matches"], 3)

    def test_agents_receive_true_fills_every_decision(self):
        game = FakeGame()
        seen = []

        class Recorder(FakeCheaterAgent):
            def __init__(self, seed):
                super().__init__(seed)
                seen.append(self)

        with tempfile.TemporaryDirectory() as tmp:
            gen_selfplay.generate_cheater(
                game, Recorder, 1, Rng(1), synthetic_card_index(),
                HeuristicEvaluator(), V1_FEATURIZE, 1.0,
                os.path.join(tmp, "e.jsonl"))
        fills = [f for agent in seen for f in agent.fills_seen]
        self.assertEqual(len(fills), 2)  # one per non-terminal decision
        for f in fills:
            self.assertEqual(f.opp_hand, [102])
        self.assertEqual(game.finished, 1)

    def test_match_error_is_contained(self):
        class ExplodingGame(FakeGame):
            def visualize_data(self):
                raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "f.jsonl")
            stats = gen_selfplay.generate_cheater(
                ExplodingGame(), FakeCheaterAgent, 2, Rng(1),
                synthetic_card_index(), HeuristicEvaluator(), V1_FEATURIZE,
                1.0, path)
            self.assertEqual(stats["errors"], 2)
            self.assertEqual(stats["matches"], 0)


class _RecordingBackend(_ScriptedBackend):
    def __init__(self):
        super().__init__()
        self.fills = []

    def begin(self, raw_obs, fills, manual_coin=True):
        self.fills.append(fills)
        return super().begin(raw_obs, fills, manual_coin)


class TestCheaterMctsAgent(unittest.TestCase):
    CFG = dict(n_worlds=1, max_iterations=8, time_budget_s=30.0,
               max_root_actions=3, max_child_actions=3)

    def _agent(self, backend):
        return CheaterMctsAgent(seed=7, deck=[101] * 60,
                                card_index=synthetic_card_index(),
                                backend=backend, **self.CFG)

    def test_true_fills_reach_the_search(self):
        backend = _RecordingBackend()
        agent = self._agent(backend)
        fills = Fills([101] * 40, [101] * 6, [101] * 40, [101] * 6,
                      [101] * 5, [])
        agent.set_true_fills(fills)
        agent.act(main_view(3).raw)
        self.assertEqual(backend.fills, [fills])
        self.assertEqual(agent.degraded_count, 0)

    def test_without_fills_the_decision_degrades_not_crashes(self):
        agent = self._agent(_RecordingBackend())
        action = agent.act(main_view(3).raw)
        self.assertEqual(len(action), 1)  # legal greedy-prior action
        self.assertEqual(agent.degraded_count, 1)


class TestBattleAgentsCannotCheat(unittest.TestCase):
    """The fairness boundary: no battle-time path can reach the cheater."""

    def test_agents_package_never_touches_the_true_state_source(self):
        agents_dir = os.path.join(REPO, "agents")
        for name in sorted(os.listdir(agents_dir)):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(agents_dir, name)) as f:
                src = f.read()
            self.assertNotIn("visualize_data", src, name)
            self.assertIsNone(
                re.search(r"^\s*(?:import|from)\s+train\b", src, re.M), name)

    def test_mcts_agent_rejects_a_fills_hook(self):
        with self.assertRaises(TypeError):
            make_agent("mcts", seed=0, deck=[101] * 60,
                       fills_fn=lambda *a: None)

    def test_planner_defaults_to_information_set_sampling(self):
        planner = MctsPlanner(own_deck=[101] * 60,
                              card_index=synthetic_card_index())
        self.assertIs(planner._fills_fn, sample_fills)

    def test_submission_archive_excludes_train(self):
        with open(os.path.join(REPO, "scripts", "build_submission.sh")) as f:
            script = f.read()
        pack = next(line for line in script.splitlines()
                    if line.startswith("tar -czf"))
        self.assertNotIn("train", pack)
        for required in ("main.py", "deck.csv", "agents", "cg"):
            self.assertIn(required, pack)


if __name__ == "__main__":
    unittest.main()
