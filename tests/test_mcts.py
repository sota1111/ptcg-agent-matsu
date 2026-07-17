"""SOT-1672 tests — Determinized MCTS planner, evaluator, MctsAgent.

Engine-independent parts run everywhere (CI included); the reproducibility
and full-match tests need the cabt engine bindings (cg/, gitignored) and
skip automatically when absent, like tests/test_engine_repro.py.
"""
import os
import sys
import unittest
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from agents.evaluator import DEFAULT_WEIGHTS, HeuristicEvaluator
from agents.mcts_agent import MctsAgent
from agents.observation import adapt
from agents.planner import MctsPlanner, PlannerConfig, sample_fills
from tests.support import (observation, player, pokemon, select,
                           synthetic_card_index)

try:
    from cg import game  # noqa: F401
    HAS_ENGINE = True
except Exception:  # pragma: no cover - engine absent (CI)
    HAS_ENGINE = False


def eval_obs(me, opp, result=-1, your_index=0):
    """Engine-search-shaped (attribute-access) observation for the evaluator."""
    players = [me, opp] if your_index == 0 else [opp, me]
    return SimpleNamespace(current=SimpleNamespace(
        result=result, yourIndex=your_index, players=players))


def side(prize=6, active=(), bench=(), hand_count=5, deck_count=40):
    return SimpleNamespace(prize=[None] * prize, active=list(active),
                           bench=list(bench), handCount=hand_count,
                           deckCount=deck_count)


def mon(hp=100, energies=1):
    return SimpleNamespace(hp=hp, energies=[0] * energies)


class TestHeuristicEvaluator(unittest.TestCase):
    def test_terminal_results_are_exact(self):
        ev = HeuristicEvaluator()
        obs = eval_obs(side(), side(), result=0)
        self.assertEqual(ev.evaluate(obs, 0), 1.0)
        self.assertEqual(ev.evaluate(obs, 1), 0.0)
        draw = eval_obs(side(), side(), result=2)
        self.assertEqual(ev.evaluate(draw, 0), 0.5)

    def test_symmetric_position_is_half(self):
        ev = HeuristicEvaluator()
        obs = eval_obs(side(active=[mon()]), side(active=[mon()]))
        self.assertAlmostEqual(ev.evaluate(obs, 0), 0.5)
        self.assertAlmostEqual(ev.evaluate(obs, 1), 0.5)

    def test_prize_lead_dominates(self):
        ev = HeuristicEvaluator()
        ahead = eval_obs(side(prize=2, active=[mon()]),
                         side(prize=6, active=[mon()]))
        self.assertGreater(ev.evaluate(ahead, 0), 0.8)
        self.assertLess(ev.evaluate(ahead, 1), 0.2)

    def test_weights_are_externally_overridable(self):
        flat = HeuristicEvaluator({k: 0.0 for k in DEFAULT_WEIGHTS})
        ahead = eval_obs(side(prize=1), side(prize=6))
        self.assertAlmostEqual(flat.evaluate(ahead, 0), 0.5)

    def test_facedown_pokemon_counts_without_stats(self):
        ev = HeuristicEvaluator()
        obs = eval_obs(side(active=[None]), side())
        self.assertGreater(ev.evaluate(obs, 0), 0.5)


class TestSampleFills(unittest.TestCase):
    DECK = [101] * 20 + [102] * 20 + [103] * 20

    def rng(self, seed=7):
        from agents.rng import Rng
        return Rng(seed)

    def raw_obs(self, me=None, opp=None):
        return observation(select([]), me=me, opp=opp)

    def test_fill_sizes_match_visible_counts(self):
        me = player(active=[pokemon(101)], deck_count=30, hand_count=4,
                    prize=6, hand=[{"id": 103}] * 4)
        opp = player(active=[pokemon(102)], deck_count=25, hand_count=7,
                     prize=5)
        fills = sample_fills(self.raw_obs(me, opp), self.DECK, self.rng(),
                             synthetic_card_index())
        self.assertEqual(len(fills.my_deck), 30)
        self.assertEqual(len(fills.my_prize), 6)
        self.assertEqual(len(fills.opp_deck), 25)
        self.assertEqual(len(fills.opp_prize), 5)
        self.assertEqual(len(fills.opp_hand), 7)
        self.assertEqual(fills.opp_active, [])

    def test_visible_cards_are_excluded_from_own_pool(self):
        # 19 copies of 102 visible in my discard -> at most 1 more 102 in
        # my hidden zones (20 in deck total).
        me = player(discard=[{"id": 102}] * 19, deck_count=41, prize=0,
                    hand_count=0, hand=[])
        fills = sample_fills(self.raw_obs(me=me), self.DECK, self.rng(),
                             synthetic_card_index())
        self.assertLessEqual(fills.my_deck.count(102), 1)

    def test_facedown_opponent_active_predicted_as_basic(self):
        opp = player(active=[None], deck_count=30, hand_count=5)
        fills = sample_fills(self.raw_obs(opp=opp), self.DECK, self.rng(),
                             synthetic_card_index())
        self.assertEqual(len(fills.opp_active), 1)
        self.assertTrue(synthetic_card_index().card(fills.opp_active[0]).basic)

    def test_opponent_deck_fill_contains_a_basic(self):
        opp = player(deck_count=10, hand_count=5, prize=6)
        fills = sample_fills(self.raw_obs(opp=opp), self.DECK, self.rng(),
                             synthetic_card_index())
        idx = synthetic_card_index()
        self.assertTrue(any(idx.card(c).basic for c in fills.opp_deck))

    def test_same_rng_seed_same_fills(self):
        obs = self.raw_obs()
        a = sample_fills(obs, self.DECK, self.rng(3), synthetic_card_index())
        b = sample_fills(obs, self.DECK, self.rng(3), synthetic_card_index())
        self.assertEqual(a, b)


class _ExplodingBackend:
    """Planner backend double: every world creation fails."""
    calls = 0

    def begin(self, raw_obs, fills, manual_coin=True):
        self.calls += 1
        raise RuntimeError("no engine")

    def end(self):
        pass


def main_view(n_options=3):
    opts = [{"type": 13, "attackId": 201 + i, "number": 0} for i in
            range(n_options)]
    return adapt(observation(
        select(opts, sel_type=0, context=0, min_count=1, max_count=1),
        me=player(active=[pokemon(101, energies=[3])]),
        opp=player(active=[pokemon(102)])))


class TestMctsPlanner(unittest.TestCase):
    def planner(self, backend, **overrides):
        return MctsPlanner(own_deck=[101] * 60,
                           config=PlannerConfig(**overrides),
                           backend=backend,
                           card_index=synthetic_card_index())

    def rng(self, seed=5):
        from agents.rng import Rng
        return Rng(seed)

    def test_forced_selection_skips_search(self):
        view = adapt(observation(
            select([{"type": 1}], min_count=1, max_count=1)))
        # min == max == n(=1): only one legal action, backend never touched.
        planner = self.planner(backend=None)
        self.assertEqual(planner.plan(view, self.rng()), [0])
        self.assertTrue(planner.last_stats.get("forced"))

    def test_degrades_to_greedy_prior_when_no_world_builds(self):
        backend = _ExplodingBackend()
        planner = self.planner(backend, n_worlds=2)
        action = planner.plan(main_view(), self.rng())
        self.assertEqual(len(action), 1)
        self.assertEqual(planner.degraded_count, 1)
        self.assertTrue(planner.last_stats.get("degraded"))
        self.assertGreater(backend.calls, 0)

    def test_config_parameters_are_external(self):
        # SOT-1673 ablation points must be constructor-injectable.
        cfg = PlannerConfig(n_worlds=7, uct_c=0.3, rollout="random",
                            time_budget_s=1.5)
        planner = MctsPlanner(own_deck=[101] * 60, config=cfg, backend=None,
                              card_index=synthetic_card_index())
        self.assertEqual(planner.config.n_worlds, 7)
        self.assertEqual(planner.config.uct_c, 0.3)
        self.assertEqual(planner.config.rollout, "random")

    def test_deck_guard_threshold_boundary_filters_pure_draw(self):
        opts = [
            {"type": 7, "index": 0},  # supporter 103: pure draw
            {"type": 13, "attackId": 201},
        ]
        planner = self.planner(None, deck_guard_threshold=4)
        low = adapt(observation(
            select(opts),
            me=player(active=[pokemon(101)], hand=[{"id": 103}],
                      hand_count=1, deck_count=4),
            opp=player(active=[pokemon(102)])))
        above = adapt(observation(
            select(opts),
            me=player(active=[pokemon(101)], hand=[{"id": 103}],
                      hand_count=1, deck_count=5),
            opp=player(active=[pokemon(102)])))
        self.assertEqual(planner._root_candidates(low, self.rng())[0], [[1]])
        self.assertEqual(len(planner._root_candidates(above, self.rng())[0]),
                         2)

    def test_deck_guard_never_removes_lethal(self):
        view = adapt(observation(
            select([{"type": 7, "index": 0},
                    {"type": 13, "attackId": 201}]),
            me=player(active=[pokemon(101)], hand=[{"id": 103}],
                      hand_count=1, deck_count=1),
            opp=player(active=[pokemon(102, hp=40)], prize=1)))
        planner = self.planner(None, deck_guard_threshold=8)
        candidates, _ = planner._root_candidates(view, self.rng())
        self.assertEqual(candidates, [[1]])
        self.assertTrue(planner._is_lethal_option(view, 1))

    def test_deck_guard_falls_back_when_every_candidate_is_draw(self):
        view = adapt(observation(
            select([{"type": 7, "index": 0}]),
            me=player(active=[pokemon(101)], hand=[{"id": 103}],
                      hand_count=1, deck_count=1)))
        planner = self.planner(None, deck_guard_threshold=8)
        # Exercise the helper directly because a one-option engine selection
        # is forced and intentionally bypasses root enumeration.
        self.assertEqual(planner._guarded_root_order(view, [0]), [0])

    def test_zero_budget_still_returns_a_legal_action(self):
        # Anytime contract: deadline already passed -> greedy prior comes
        # back immediately (worlds may build, but no iteration runs).
        backend = _ExplodingBackend()
        planner = self.planner(backend, n_worlds=2)
        action = planner.plan(main_view(), self.rng(), budget_s=0.0)
        self.assertEqual(len(action), 1)
        self.assertIn(action[0], (0, 1, 2))

    def test_best_action_aggregates_visits_across_worlds(self):
        w1 = SimpleNamespace(root=SimpleNamespace(
            edges=[[[0], None, 5, 2.0], [[1], None, 9, 6.0]]))
        w2 = SimpleNamespace(root=SimpleNamespace(
            edges=[[[0], None, 6, 3.0], [[1], None, 4, 2.0]]))
        best = MctsPlanner._best_action([[0], [1]], [w1, w2])
        self.assertEqual(best, [1])  # 13 visits vs 11

    def test_deviate_margin_keeps_greedy_prior_without_evidence(self):
        w = SimpleNamespace(root=SimpleNamespace(
            edges=[[[0], None, 10, 5.5], [[1], None, 12, 6.7]]))
        # Challenger wins on visits but its mean (0.558) is within the
        # margin of the incumbent's (0.550) -> stay with the greedy prior.
        self.assertEqual(MctsPlanner._best_action([[0], [1]], [w]), [1])
        self.assertEqual(MctsPlanner._best_action([[0], [1]], [w], 0.05), [0])
        # A clear challenger still deviates.
        w2 = SimpleNamespace(root=SimpleNamespace(
            edges=[[[0], None, 10, 3.0], [[1], None, 12, 9.0]]))
        self.assertEqual(MctsPlanner._best_action([[0], [1]], [w2], 0.05),
                         [1])


class TestMctsAgent(unittest.TestCase):
    def test_planner_exception_falls_back_to_greedy(self):
        agent = MctsAgent(seed=1, deck=[101] * 60,
                          card_index=synthetic_card_index())

        class Boom:
            def plan(self, view, rng, budget_s=None):
                raise RuntimeError("boom")

        agent._planner = Boom()
        action = agent.act(main_view(2).raw)
        self.assertEqual(len(action), 1)
        self.assertEqual(agent.planner_fallbacks, 1)
        self.assertEqual(agent.fallback_count, 0)  # inner fallback caught it

    def test_budget_violations_counted_with_injected_clock(self):
        times = iter([0.0, 10.0])  # decision takes 10s > 0.1s budget

        class Instant:
            def plan(self, view, rng, budget_s=None):
                return [0]

        agent = MctsAgent(seed=1, deck=[101] * 60,
                          card_index=synthetic_card_index(),
                          clock=lambda: next(times))
        agent._planner = Instant()
        agent.act(main_view(2).raw)
        self.assertEqual(agent.budget_violations, 1)
        self.assertEqual(agent.move_times, [10.0])


class _ScriptedBackend:
    """Deterministic engine double: stepping action [1] wins for player 0,
    any other root action loses. Engine responses are a pure function of the
    action, so any run-to-run variation could only come from agent-side
    randomness — which must all be seed-derived (ASSUMPTIONS.md A-9)."""

    def __init__(self):
        self.next_sid = 0

    def begin(self, raw_obs, fills, manual_coin=True):
        self.next_sid += 1
        sel = SimpleNamespace(option=[SimpleNamespace(type=13, number=0)] * 3,
                              minCount=1, maxCount=1, context=0)
        obs = SimpleNamespace(
            select=sel,
            current=SimpleNamespace(result=-1, yourIndex=0, turn=1))
        return self.next_sid, obs

    def step(self, sid, action):
        self.next_sid += 1
        result = 0 if action == [1] else 1
        obs = SimpleNamespace(
            select=None,
            current=SimpleNamespace(result=result, yourIndex=0, turn=1))
        return self.next_sid, obs

    def release(self, sid):
        pass

    def end(self):
        pass


class TestAgentSideReproducibility(unittest.TestCase):
    """同一シード+同一局面→同一着手, scoped to agent-side randomness.

    The real search API consumes a non-injectable engine RNG (shuffle
    effects; docs/engine-facts.md §5, docs/mcts-design.md §9), so the
    reproducibility guarantee is specified with engine responses held fixed
    — every OTHER source of randomness (fill sampling, coin sampling,
    candidate generation, tie-break jitter) is exercised here and must be a
    deterministic function of the injected seed."""

    CFG = dict(n_worlds=3, max_iterations=16, time_budget_s=30.0,
               max_root_actions=3, max_child_actions=3)

    def _agent(self, seed):
        return MctsAgent(seed=seed, deck=[101] * 60,
                         card_index=synthetic_card_index(),
                         backend=_ScriptedBackend(), **self.CFG)

    def test_same_seed_same_observation_same_action(self):
        obs = main_view(3).raw
        actions = [self._agent(seed=7).act(obs) for _ in range(3)]
        self.assertEqual(actions[0], actions[1])
        self.assertEqual(actions[0], actions[2])

    def test_search_finds_the_scripted_win(self):
        for seed in (1, 2, 3):
            self.assertEqual(self._agent(seed).act(main_view(3).raw), [1])

    def test_repro_holds_across_a_decision_sequence(self):
        # Per-decision child streams: decision k must not depend on how
        # much randomness earlier decisions consumed beyond the stream name.
        obs_seq = [main_view(3).raw, main_view(2).raw, main_view(3).raw]
        a, b = self._agent(11), self._agent(11)
        self.assertEqual([a.act(o) for o in obs_seq],
                         [b.act(o) for o in obs_seq])


@unittest.skipUnless(HAS_ENGINE, "cabt engine (cg/) not available")
class TestMctsOnEngine(unittest.TestCase):
    """Same harness as tests/test_engine_repro.py, for the MCTS agent."""

    # Fast search config so a full engine match stays test-sized.
    REPRO_CFG = dict(n_worlds=2, max_iterations=4, time_budget_s=30.0,
                     rollout_turns=1, rollout_depth=20, max_root_actions=4,
                     max_child_actions=4)

    def _agent(self, seed):
        return MctsAgent(seed=seed, deck=self._deck(), **self.REPRO_CFG)

    @staticmethod
    def _deck():
        with open(os.path.join(REPO, "deck.csv")) as f:
            return [int(x) for x in f.read().split("\n")[:60]]

    def test_full_match_no_rejects_no_fallbacks(self):
        os.chdir(REPO)
        from tests.test_engine_repro import play_and_record
        a, b = self._agent(1), self._agent(2)
        _, result = play_and_record(a, b)
        self.assertIn(result, (0, 1, 2))
        for agent in (a, b):
            self.assertEqual(agent.fallback_count, 0)
            self.assertEqual(agent.planner_fallbacks, 0)
            self.assertGreater(agent.decision_count, 0)


if __name__ == "__main__":
    unittest.main()
