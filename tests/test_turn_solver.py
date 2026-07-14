"""SOT-1677 tests — gain-loss greedy TurnSolver and the mctsS planner
integration (macro-action expansion behind `PlannerConfig.solver`).

Engine-independent: all tests run against scripted backend doubles, like
tests/test_mcts.py. Legality is asserted per step against the select the
engine (double) offered — the solver must never emit an out-of-range index,
a duplicate, or a count outside [minCount, maxCount].
"""
import os
import sys
import unittest
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from agents.mcts_agent import MctsAgent
from agents.planner import MctsPlanner, PlannerConfig, _Node
from agents.rng import Rng
from agents.turn_solver import TurnSolver
from tests.support import synthetic_card_index

# OptionType ints (cg/api.py:120-187)
OT_PLAY, OT_ATTACH, OT_EVOLVE = 7, 8, 9
OT_ATTACK, OT_END = 13, 14
OT_YES, OT_NO = 1, 2
CTX_COIN = 46


def opt(type, **attrs):
    return SimpleNamespace(type=type, **attrs)


def sel(options, min_count=1, max_count=1, context=0):
    return SimpleNamespace(option=list(options), minCount=min_count,
                           maxCount=max_count, context=context)


def mon(card_id, hp=100):
    return SimpleNamespace(id=card_id, hp=hp)


def sobs(select, turn=1, actor=0, result=-1, my_active=(), opp_active=()):
    players = [SimpleNamespace(active=list(my_active)),
               SimpleNamespace(active=list(opp_active))]
    if actor == 1:
        players.reverse()
    return SimpleNamespace(select=select, current=SimpleNamespace(
        result=result, yourIndex=actor, turn=turn, players=players))


class ScriptBackend:
    """Steps through a fixed observation script, recording every action
    with the index of the select it answered."""

    def __init__(self, script):
        self.script = list(script)
        self.pos = 0
        self.steps = []      # (script position answered, action)
        self.released = []
        self.next_sid = 100

    def step(self, sid, action):
        self.steps.append((self.pos, list(action)))
        self.pos += 1
        self.next_sid += 1
        return self.next_sid, self.script[self.pos]

    def release(self, sid):
        self.released.append(sid)

    def end(self):
        pass


def assert_legal(testcase, script, steps):
    for pos, action in steps:
        s = script[pos].select
        n = len(s.option)
        lo = min(max(s.minCount, 0), min(s.maxCount, n))
        hi = min(s.maxCount, n)
        testcase.assertTrue(lo <= len(action) <= hi,
                            f"count {len(action)} outside [{lo},{hi}]")
        testcase.assertTrue(all(0 <= i < n for i in action), f"{action}")
        testcase.assertEqual(len(set(action)), len(action), f"{action}")


def solver(max_evals=64, **kw):
    return TurnSolver(synthetic_card_index(), max_evals=max_evals, **kw)


class TestTurnSolverWalk(unittest.TestCase):
    def turn_script(self):
        """Actor 0's turn: a development select, then an attack select,
        then the opponent's turn begins (boundary)."""
        return [
            sobs(sel([opt(OT_ATTACH, inPlayArea=4), opt(OT_EVOLVE),
                      opt(OT_END)])),
            sobs(sel([opt(OT_ATTACK, attackId=201), opt(OT_END)]),
                 my_active=[mon(101)], opp_active=[mon(102, hp=60)]),
            sobs(sel([opt(OT_END)]), turn=2, actor=1),
        ]

    def test_actions_all_legal_until_turn_end(self):
        script = self.turn_script()
        backend = ScriptBackend(script)
        res = solver().solve(backend, 0, script[0], Rng(1))
        self.assertEqual(res.stop, "turn_end")
        self.assertEqual(len(res.actions), 2)
        assert_legal(self, script, backend.steps)
        # Development chosen over END on the first select.
        self.assertNotIn(2, res.actions[0])

    def test_boundary_obs_is_returned_not_stepped(self):
        script = self.turn_script()
        backend = ScriptBackend(script)
        res = solver().solve(backend, 0, script[0], Rng(1))
        self.assertIs(res.obs, script[2])
        self.assertEqual(backend.pos, 2)  # the boundary select never stepped

    def test_same_seed_same_sequence_through_coins(self):
        def script():
            return [
                sobs(sel([opt(OT_ATTACH), opt(OT_EVOLVE), opt(OT_END)])),
                sobs(sel([opt(OT_YES), opt(OT_NO)], context=CTX_COIN)),
                sobs(sel([opt(OT_ATTACK, attackId=201), opt(OT_END)]),
                     my_active=[mon(101)], opp_active=[mon(102, hp=60)]),
                sobs(sel([opt(OT_END)]), turn=2, actor=1),
            ]
        runs = []
        for _ in range(2):
            s = script()
            runs.append(solver().solve(ScriptBackend(s), 0, s[0],
                                       Rng(42)).actions)
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(len(runs[0]), 3)  # coin resolved inside the macro

    def test_eval_cap_bounds_the_walk(self):
        many = [opt(OT_ATTACH) for _ in range(10)]
        script = [sobs(sel(many)) for _ in range(6)]
        backend = ScriptBackend(script)
        res = solver(max_evals=25).solve(backend, 0, script[0], Rng(1))
        self.assertEqual(res.stop, "eval_cap")
        self.assertLessEqual(res.evals, 25)
        self.assertEqual(len(res.actions), 2)  # 10 + 10, third would be 30

    def test_step_cap_bounds_the_walk(self):
        script = [sobs(sel([opt(OT_ATTACH)])) for _ in range(10)]
        backend = ScriptBackend(script)
        res = TurnSolver(synthetic_card_index(), max_evals=1000,
                         max_steps=3).solve(backend, 0, script[0], Rng(1))
        self.assertEqual(res.stop, "step_cap")
        self.assertEqual(len(res.actions), 3)

    def test_deadline_stops_anytime(self):
        script = [sobs(sel([opt(OT_ATTACH)])) for _ in range(5)]
        times = iter([0.0, 10.0, 10.1, 10.2, 10.3, 10.4])
        res = solver().solve(ScriptBackend(script), 0, script[0], Rng(1),
                             deadline=5.0, clock=lambda: next(times))
        self.assertEqual(res.stop, "deadline")
        self.assertEqual(len(res.actions), 1)

    def test_terminal_stops_immediately(self):
        script = [sobs(sel([opt(OT_END)]), result=0)]
        res = solver().solve(ScriptBackend(script), 0, script[0], Rng(1))
        self.assertEqual(res.stop, "terminal")
        self.assertEqual(res.actions, [])

    def test_intermediate_sids_released_final_kept(self):
        script = self.turn_script()
        backend = ScriptBackend(script)
        res = solver().solve(backend, 0, script[0], Rng(1),
                             release_initial=True)
        # Two steps: sids 101 (released as intermediate) and 102 (final).
        self.assertIn(0, backend.released)     # initial, release_initial=True
        self.assertIn(101, backend.released)
        self.assertNotIn(res.sid, backend.released)

    def test_initial_sid_kept_by_default(self):
        script = self.turn_script()
        backend = ScriptBackend(script)
        solver().solve(backend, 0, script[0], Rng(1))
        self.assertNotIn(0, backend.released)


class TestGainLossScoring(unittest.TestCase):
    def test_lethal_attack_beats_development(self):
        # 101 (Water, attack 201 dmg 50) vs 102 (weak to Water, hp 60):
        # 50x2 = 100 >= 60 -> lethal (+300 + 150*prize) dominates the
        # opportunity cost of the development options in the same select.
        s = sel([opt(OT_ATTACK, attackId=201), opt(OT_ATTACH),
                 opt(OT_EVOLVE), opt(OT_END)])
        obs = sobs(s, my_active=[mon(101)], opp_active=[mon(102, hp=60)])
        self.assertEqual(solver().choose(s, obs), [0])

    def test_development_beats_nonlethal_attack(self):
        # 102 (Fire, attack 202 dmg 30) vs 101 (no weakness, hp 120):
        # gain 30 - opportunity cost of 2 foregone development options < 0.
        s = sel([opt(OT_ATTACK, attackId=202), opt(OT_ATTACH),
                 opt(OT_EVOLVE), opt(OT_END)])
        obs = sobs(s, my_active=[mon(102)], opp_active=[mon(101, hp=120)])
        choice = solver().choose(s, obs)
        self.assertNotIn(0, choice)  # not the attack
        self.assertNotIn(3, choice)  # not END

    def test_end_chosen_when_it_is_the_only_option(self):
        s = sel([opt(OT_END)])
        self.assertEqual(solver().choose(s, sobs(s)), [0])

    def test_cost_context_pays_the_minimum(self):
        # DISCARD (context 8): free count 0..2 -> commit to the minimum.
        s = sel([opt(3), opt(3)], min_count=0, max_count=2, context=8)
        self.assertEqual(solver().choose(s, sobs(s)), [])

    def test_scoring_is_deterministic(self):
        s = sel([opt(OT_ATTACH), opt(OT_EVOLVE), opt(OT_PLAY)])
        obs = sobs(s)
        sv = solver()
        self.assertEqual(sv.choose(s, obs), sv.choose(s, obs))


class TestPlannerSolverIntegration(unittest.TestCase):
    def turn_script(self):
        return [
            sobs(sel([opt(OT_ATTACH), opt(OT_EVOLVE), opt(OT_END)])),
            sobs(sel([opt(OT_ATTACK, attackId=201), opt(OT_END)]),
                 my_active=[mon(101)], opp_active=[mon(102, hp=60)]),
            sobs(sel([opt(OT_END)]), turn=2, actor=1),
        ]

    def planner(self, backend, **overrides):
        return MctsPlanner(own_deck=[101] * 60,
                           config=PlannerConfig(**overrides),
                           backend=backend,
                           card_index=synthetic_card_index())

    def test_solver_defaults_off(self):
        cfg = PlannerConfig()
        self.assertFalse(cfg.solver)
        self.assertEqual(cfg.solver_max_evals, 64)

    def test_expand_steps_once_with_solver_off(self):
        script = self.turn_script()
        backend = ScriptBackend(script)
        planner = self.planner(backend)
        node = _Node(0, script[0], 0)
        child = planner._expand(node, [0], 0, Rng(3))
        self.assertEqual(len(backend.steps), 1)
        self.assertIs(child.obs, script[1])

    def test_expand_completes_the_turn_with_solver_on(self):
        script = self.turn_script()
        backend = ScriptBackend(script)
        planner = self.planner(backend, solver=True)
        node = _Node(0, script[0], 0)
        child = planner._expand(node, [0], 0, Rng(3))
        # Edge action + solver completion of the turn (the attack select).
        self.assertEqual(len(backend.steps), 2)
        self.assertIs(child.obs, script[2])
        assert_legal(self, script, backend.steps)
        self.assertNotIn(0, backend.released)  # parent node sid untouched

    def test_mcts_agent_accepts_solver_config_kwarg(self):
        # bench.py --config-a '{"solver": true}' arrives here as a kwarg.
        agent = MctsAgent(seed=1, deck=[101] * 60,
                          card_index=synthetic_card_index(), solver=True)
        self.assertTrue(agent.config.solver)

    def test_same_seed_same_action_with_solver_on(self):
        from tests.test_mcts import _ScriptedBackend, main_view
        cfg = dict(n_worlds=2, max_iterations=8, time_budget_s=30.0,
                   max_root_actions=3, max_child_actions=3, solver=True)
        obs = main_view(3).raw
        actions = [MctsAgent(seed=9, deck=[101] * 60,
                             card_index=synthetic_card_index(),
                             backend=_ScriptedBackend(), **cfg).act(obs)
                   for _ in range(2)]
        self.assertEqual(actions[0], actions[1])


if __name__ == "__main__":
    unittest.main()
