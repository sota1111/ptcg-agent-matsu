"""Determinized MCTS planner (SOT-1672) — layer [3] of the 4-layer architecture.

Built on the engine's determinization search API (cg/api.py:517-639,
docs/engine-facts.md §1.3). Each decision:

1. samples `n_worlds` determinizations — assignments of the hidden zones
   (own deck order / facedown prizes, opponent hand / deck / prizes /
   facedown Active) consistent with all visible information (discards,
   boards, face-up prizes, stadium ownership are subtracted from the
   candidate pool). N is an external parameter: n_worlds=1 is PIMC-style
   single-world search, larger N averages over hidden-information samples;
2. runs PUCT-guided tree search in each world THROUGH the engine (the
   engine's search API remains the single source of truth for legality and
   state transitions — this module never simulates rules itself);
3. aggregates root-action statistics across worlds and returns the action
   with the most total visits.

Chance events (coin flips) are made explicit with
`search_begin(manual_coin=True)` and sampled 50/50 from the injected agent
Rng, never left to the engine's internal non-injectable RNG. Coins are the
only engine randomness that CAN be captured this way: the search API keeps
its own `std::random_device`-seeded RNG for everything else (shuffle
effects; docs/engine-facts.md §5), so identical (observation, fills, action
sequence) can still yield different search outcomes — measured on the real
engine 2026-07-13, same-seed decisions diverged on 3/12 probed positions.
The same-seed reproducibility guarantee (ASSUMPTIONS.md A-9) is therefore
scoped to AGENT-side randomness: with engine responses held fixed, a
decision is a deterministic function of the seed (tests/test_mcts.py tests
this against a deterministic backend double). Design rationale and the
chance-node alternatives considered are recorded in docs/mcts-design.md
(Świechowski et al., arXiv:1808.04794).

Anytime contract: the search consumes at most `budget * budget_fraction`
(default 80%; the remaining 20% is margin for aggregation/bookkeeping) and
can be cut off at ANY point — before the first iteration completes the
best-so-far action is the greedy prior (root candidates are ordered by
GreedyAgent scores), so a legal, sensible action is always returned.
Setting `max_iterations` bounds the search by count instead, which removes
the wall clock from the decision path (used by the repro tests together
with a deterministic backend; the wall-clock cutoff alone cannot guarantee
identical visit counts).

All randomness flows through the per-decision `Rng` stream passed to
`plan()` (SOT-1671 rng規約); this module never touches global `random`.
"""
import math
import time
from dataclasses import asdict, dataclass, field

from . import actions
from .evaluator import HeuristicEvaluator
from .greedy_agent import (GreedyAgent, _COST_CONTEXTS, _COUNT_MAX_CONTEXTS,
                           _YES_CONTEXTS)
from .observation import View, adapt
from .turn_solver import TurnSolver

# --- OptionType tiers for the lightweight rollout policy ---------------------
# Mirrors cg/api.py:120-187 (values as plain ints, unknown values -> default).
# Ordering lesson from SOT-1671 (docs/baselines.md): development actions
# (EVOLVE/ABILITY/ATTACH/PLAY) must rank ABOVE attacking (attacks end the
# turn), and attacking above RETREAT/END.
_TIER = {
    9: 70.0,   # EVOLVE
    10: 60.0,  # ABILITY
    8: 45.0,   # ATTACH
    7: 40.0,   # PLAY
    13: 20.0,  # ATTACK
    15: 10.0,  # SKILL
    12: 4.0,   # RETREAT
    14: 0.0,   # END
}
_TIER_DEFAULT = 10.0  # unknown option types: between END and real actions
_OT_YES, _OT_NO, _OT_NUMBER = 1, 2, 0

_CTX_COIN = 46  # SelectContext.COIN_HEAD (cg/api.py:114) — chance node marker


@dataclass
class PlannerConfig:
    """External parameters (SOT-1673 ablation points)."""
    n_worlds: int = 3            # determinizations per decision (N)
    uct_c: float = 1.4           # PUCT exploration constant
    prior_temperature: float = 40.0  # softmax temp over greedy/tier scores
    rollout: str = "greedy"      # rollout policy: "greedy" | "heuristic" | "random"
    rollout_depth: int = 60      # rollout step cap before leaf evaluation
    rollout_turns: int = 2       # rollout turn-boundary cap (evaluate at the
                                 # start of a turn, when the board is settled)
    max_tree_depth: int = 4      # tree depth before switching to rollout
    max_root_actions: int = 12   # root candidate-action cap
    max_child_actions: int = 8   # in-tree candidate-action cap
    time_budget_s: float = 0.1   # per-decision wall-clock budget
    budget_fraction: float = 0.8 # search cutoff fraction (rest is margin)
    # Iteration bound. Normally the wall clock binds first; the default cap
    # only stops degenerate endgame spins on saturated trees. A LOW value
    # (that always binds before the clock) removes the wall clock from the
    # decision path (see the repro tests, which pair it with a
    # deterministic backend).
    max_iterations: int | None = 2000
    # Deviate from the greedy prior (candidate 0) only when the challenger's
    # pooled mean value beats the prior's by this margin. 0.0 keeps plain
    # visit-max aggregation; small positive values suppress false deviations
    # in the low-sample regime (few rollouts per alternative).
    deviate_margin: float = 0.0
    # mctsS (SOT-1677, arXiv:1808.04794 §III-B3): when True, every tree
    # expansion becomes a macro action — the edge's action is stepped and
    # the gain-loss greedy TurnSolver then completes the acting player's
    # turn, so one tree level spans one turn. False keeps the existing
    # one-select-per-edge behavior exactly.
    solver: bool = False
    solver_max_evals: int = 64   # option-evaluation cap per solver run


@dataclass
class Fills:
    """Predicted hidden-zone card IDs passed to search_begin."""
    my_deck: list
    my_prize: list
    opp_deck: list
    opp_prize: list
    opp_hand: list
    opp_active: list


class EngineBackend:
    """cg.api search wrapper (imported lazily; absent on CI)."""

    def __init__(self):
        from cg import api
        self._api = api

    def begin(self, raw_obs: dict, fills: Fills, manual_coin: bool = True):
        obs = self._api.to_observation_class(raw_obs)
        st = self._api.search_begin(
            obs, fills.my_deck, fills.my_prize, fills.opp_deck,
            fills.opp_prize, fills.opp_hand, fills.opp_active,
            manual_coin=manual_coin)
        return st.searchId, st.observation

    def step(self, sid: int, action: list):
        st = self._api.search_step(sid, list(action))
        return st.searchId, st.observation

    def release(self, sid: int) -> None:
        self._api.search_release(sid)

    def end(self) -> None:
        self._api.search_end()


# --- determinization ----------------------------------------------------------

def _card_ids_of(cards) -> list:
    return [c.get("id") for c in (cards or ()) if isinstance(c, dict)]


def _visible_ids(player: dict, stadium_ids: list) -> list:
    """All card IDs visibly owned by `player` (subtracted from its pool)."""
    ids = _card_ids_of(player.get("hand"))
    ids += _card_ids_of(player.get("discard"))
    for pk in list(player.get("active") or ()) + list(player.get("bench") or ()):
        if not isinstance(pk, dict):
            continue  # facedown Pokémon: identity unknown, stays in the pool
        ids.append(pk.get("id"))
        ids += _card_ids_of(pk.get("energyCards"))
        ids += _card_ids_of(pk.get("tools"))
        ids += _card_ids_of(pk.get("preEvolution"))
    ids += _card_ids_of([c for c in (player.get("prize") or ())
                         if isinstance(c, dict)])  # face-up prizes only
    ids += stadium_ids
    return [i for i in ids if i is not None]


def _pool_minus(deck: list, seen: list) -> list:
    pool = list(deck)
    for cid in seen:
        if cid in pool:
            pool.remove(cid)
    return pool


def sample_fills(raw_obs: dict, own_deck: list, rng, card_index) -> Fills:
    """One determinization consistent with the visible information.

    Opponent model: MIRROR — the opponent is assumed to play the same
    60-card list as us (exactly true in self-play benches; an opponent-deck
    estimator can replace this in Phase 3+, see docs/mcts-design.md). If the
    mirror pool does not cover the required counts (non-mirror opponent),
    the pool is padded by resampling its own items — sizes then still
    satisfy the search_begin length contract.
    """
    current = raw_obs.get("current") or {}
    players = current.get("players") or ({}, {})
    yi = current.get("yourIndex", 0)
    me, opp = players[yi], players[1 - yi]

    # Stadium cards belong to whoever played them (Card.playerIndex).
    stadium = [c for c in (current.get("stadium") or ()) if isinstance(c, dict)]
    my_stadium = [c.get("id") for c in stadium if c.get("playerIndex") == yi]
    opp_stadium = [c.get("id") for c in stadium if c.get("playerIndex") != yi]

    my_pool = _pool_minus(own_deck, _visible_ids(me, my_stadium))
    opp_pool = _pool_minus(own_deck, _visible_ids(opp, opp_stadium))
    rng.shuffle(my_pool)
    rng.shuffle(opp_pool)

    my_deck_n = me.get("deckCount", 0) or 0
    my_prize = list(me.get("prize") or ())
    opp_deck_n = opp.get("deckCount", 0) or 0
    opp_prize_n = len(opp.get("prize") or ())
    opp_hand_n = opp.get("handCount", 0) or 0

    def take(pool, n):
        while len(pool) < n:  # non-mirror opponent: pad by resampling
            pool.append(rng.choice(own_deck))
        out, rest = pool[:n], pool[n:]
        return out, rest

    my_deck_fill, my_pool = take(my_pool, my_deck_n)
    # Keep face-up prize slots at their positions; fill facedown slots.
    my_prize_fill = []
    for slot in my_prize:
        if isinstance(slot, dict) and slot.get("id") is not None:
            my_prize_fill.append(slot["id"])
        else:
            picked, my_pool = take(my_pool, 1)
            my_prize_fill.append(picked[0])

    opp_deck_fill, opp_pool = take(opp_pool, opp_deck_n)
    opp_prize_fill, opp_pool = take(opp_pool, opp_prize_n)
    opp_hand_fill, opp_pool = take(opp_pool, opp_hand_n)

    # search_begin requires >=1 Basic Pokémon in the opponent deck at setup.
    if opp_deck_fill and not any(card_index.card(c).basic for c in opp_deck_fill):
        for src in (opp_hand_fill, opp_prize_fill):
            j = next((k for k, c in enumerate(src)
                      if card_index.card(c).basic), None)
            if j is not None:
                opp_deck_fill[0], src[j] = src[j], opp_deck_fill[0]
                break

    # Facedown opponent Active must be predicted with a Pokémon card ID.
    opp_active_fill = []
    opp_active = list(opp.get("active") or ())
    if opp_active and opp_active[0] is None:
        candidates = opp_hand_fill + opp_deck_fill + list(own_deck)
        pick = next((c for c in candidates if card_index.card(c).basic), None)
        if pick is None:
            pick = next((c for c in candidates
                         if card_index.card(c).card_type == 0), own_deck[0])
        opp_active_fill = [pick]

    return Fills(my_deck_fill, my_prize_fill, opp_deck_fill,
                 opp_prize_fill, opp_hand_fill, opp_active_fill)


# --- duck-typed accessors over engine search observations ---------------------

def _obs_result(obs) -> int:
    current = getattr(obs, "current", None)
    r = getattr(current, "result", -1) if current is not None else -1
    return -1 if r is None else r


def _obs_actor(obs) -> int:
    current = getattr(obs, "current", None)
    return getattr(current, "yourIndex", 0) if current is not None else 0


def _sel_bounds(sel) -> tuple:
    n = len(sel.option or ())
    hi = min(max(sel.maxCount, 0), n)
    lo = min(max(sel.minCount, 0), hi)
    return n, lo, hi


def _terminal_value(obs, root_player: int):
    r = _obs_result(obs)
    if r == -1:
        return None
    if r == root_player:
        return 1.0
    if r == 1 - root_player:
        return 0.0
    return 0.5  # draw


# --- tree ---------------------------------------------------------------------

class _Node:
    __slots__ = ("sid", "obs", "actor", "terminal", "edges", "priors")

    def __init__(self, sid, obs, root_player):
        self.sid = sid
        self.obs = obs
        self.actor = _obs_actor(obs)
        self.terminal = _terminal_value(obs, root_player)
        self.edges = []   # [action(list), child(_Node|None), visits, value_sum]
        self.priors = []


class _World:
    __slots__ = ("root", "iterations")

    def __init__(self, root):
        self.root = root
        self.iterations = 0


def _softmax(scores, temperature) -> list:
    if not scores:
        return []
    t = max(temperature, 1e-6)
    m = max(scores)
    exps = [math.exp((s - m) / t) for s in scores]
    z = sum(exps)
    return [e / z for e in exps]


class MctsPlanner:
    """Determinized, anytime, PUCT-guided MCTS over the engine search API."""

    def __init__(self, own_deck, config=None, evaluator=None, backend=None,
                 card_index=None, clock=time.perf_counter, fills_fn=None):
        if not own_deck:
            raise ValueError("MctsPlanner requires the agent's 60-card deck")
        # Determinization source: `fills_fn(raw_obs, own_deck, rng, card_index)
        # -> Fills` replaces sample_fills per world. Constructor-injection only
        # (SOT-1678 training-data generation); MctsAgent never sets it, so a
        # battle agent always samples its worlds from the information set.
        self._fills_fn = fills_fn or sample_fills
        self.config = config or PlannerConfig()
        self.evaluator = evaluator or HeuristicEvaluator()
        self._own_deck = list(own_deck)
        self._backend = backend
        self._card_index = card_index
        self._clock = clock
        self._greedy = GreedyAgent(seed=0, card_index=card_index)
        self._solver = None
        self.degraded_count = 0   # decisions answered by the greedy prior
        self.last_stats = {}

    @property
    def backend(self):
        if self._backend is None:
            self._backend = EngineBackend()
        return self._backend

    @property
    def cards(self):
        if self._card_index is None:
            self._card_index = self._greedy.cards
        return self._card_index

    @property
    def solver(self) -> TurnSolver:
        if self._solver is None:
            self._solver = TurnSolver(self.cards,
                                      max_evals=self.config.solver_max_evals)
        return self._solver

    # ---- public API -----------------------------------------------------

    def plan(self, view: View, rng, budget_s: float | None = None) -> list:
        cfg = self.config
        budget = cfg.time_budget_s if budget_s is None else budget_s
        t0 = self._clock()
        deadline = t0 + budget * cfg.budget_fraction

        candidates, priors = self._root_candidates(view, rng)
        if len(candidates) == 1:
            self.last_stats = {"iterations": 0, "worlds": 0, "forced": True}
            return list(candidates[0])

        root_player = view.your_index
        worlds = []
        iterations = 0
        try:
            for _ in range(max(1, cfg.n_worlds)):
                if worlds and self._clock() >= deadline:
                    break
                try:
                    worlds.append(self._make_world(
                        view.raw, candidates, priors, root_player, rng))
                except Exception:
                    continue  # inconsistent fills etc.: skip this world
            if not worlds:
                self.degraded_count += 1
                self.last_stats = {"iterations": 0, "worlds": 0,
                                   "degraded": True}
                return list(candidates[0])  # greedy prior (anytime default)

            while self._clock() < deadline:
                if cfg.max_iterations is not None \
                        and iterations >= cfg.max_iterations:
                    break
                self._iterate(worlds[iterations % len(worlds)],
                              root_player, rng, deadline)
                iterations += 1
        finally:
            try:
                self.backend.end()
            except Exception:
                pass

        best = self._best_action(candidates, worlds, cfg.deviate_margin)
        self.last_stats = {"iterations": iterations, "worlds": len(worlds),
                           "elapsed_s": self._clock() - t0}
        return list(best)

    # ---- root candidates --------------------------------------------------

    def _root_candidates(self, view: View, rng) -> tuple:
        """Candidate root actions ordered by greedy prior (index 0 = prior
        best), with softmax priors. Forced selections return one candidate."""
        cfg = self.config
        sel = view.select
        lo, hi = actions.count_bounds(sel)
        n = len(sel.options)
        if lo == hi and lo in (0, n):  # forced: empty or take-everything
            return [sorted(range(lo))] if lo == 0 else [list(range(n))], [1.0]
        scores = self._greedy.score_options(view)
        order = sorted(range(n), key=lambda i: (-scores[i], i))
        if lo == hi == 1:
            picked = order[:cfg.max_root_actions]
            return ([[i] for i in picked],
                    _softmax([scores[i] for i in picked],
                             cfg.prior_temperature))
        # Count selection: greedy's own pick first, then extreme-count
        # top-score sets, then random legal samples for diversity.
        cands = [tuple(sorted(self._greedy.choose(view)))]
        for k in (lo, hi):
            cands.append(tuple(sorted(order[:k])))
        attempts = 0
        while len(set(cands)) < cfg.max_root_actions and attempts < 40:
            attempts += 1
            k = rng.randint(lo, hi)
            cands.append(tuple(sorted(rng.sample(range(n), k))))
        uniq = list(dict.fromkeys(cands))[:cfg.max_root_actions]
        prior_scores = [sum(scores[i] for i in a) for a in uniq]
        return ([list(a) for a in uniq],
                _softmax(prior_scores, cfg.prior_temperature))

    # ---- worlds -----------------------------------------------------------

    def _make_world(self, raw_obs, candidates, priors, root_player, rng):
        fills = self._fills_fn(raw_obs, self._own_deck, rng, self.cards)
        sid, obs = self.backend.begin(raw_obs, fills, manual_coin=True)
        root = _Node(sid, obs, root_player)
        if root.terminal is None:
            n_opts = len(root.obs.select.option or ())
            for a in candidates:
                if any(i >= n_opts for i in a):
                    raise ValueError("world root select mismatch")
        root.edges = [[list(a), None, 0, 0.0] for a in candidates]
        root.priors = list(priors)
        return _World(root)

    # ---- one PUCT iteration -------------------------------------------------

    def _iterate(self, world, root_player, rng, deadline):
        node = world.root
        path = []
        depth = 0
        value = None
        while True:
            if node.terminal is not None:
                value = node.terminal
                break
            if depth >= self.config.max_tree_depth or not node.edges:
                value = self._rollout(node, root_player, rng, deadline)
                break
            edge = self._select_edge(node, root_player)
            path.append(edge)
            if edge[1] is None:
                child = self._expand(node, edge[0], root_player, rng,
                                     deadline)
                edge[1] = child
                value = (child.terminal if child.terminal is not None
                         else self._rollout(child, root_player, rng, deadline))
                break
            node = edge[1]
            depth += 1
        for edge in path:
            edge[2] += 1
            edge[3] += value
        world.iterations += 1

    def _select_edge(self, node, root_player):
        c = self.config.uct_c
        total = sum(e[2] for e in node.edges) + 1
        sqrt_total = math.sqrt(total)
        best, best_key = None, None
        for i, e in enumerate(node.edges):
            q = (e[3] / e[2]) if e[2] else 0.5
            if node.actor != root_player:
                q = 1.0 - q
            prior = node.priors[i] if i < len(node.priors) else 0.0
            key = q + c * prior * sqrt_total / (1 + e[2])
            if best_key is None or key > best_key:
                best, best_key = e, key
        return best

    def _expand(self, node, action, root_player, rng, deadline=None):
        sid, obs = self.backend.step(node.sid, action)
        sid, obs = self._resolve_chance(sid, obs, rng)
        if self.config.solver:
            # mctsS macro action (SOT-1677): the gain-loss TurnSolver
            # completes the acting player's turn, so this edge spans one
            # turn instead of one select. Anytime: the solver stops at the
            # deadline / eval cap and the node is built wherever it stopped.
            res = self.solver.solve(self.backend, sid, obs, rng,
                                    deadline=deadline, clock=self._clock,
                                    release_initial=True)
            sid, obs = self._resolve_chance(res.sid, res.obs, rng)
        child = _Node(sid, obs, root_player)
        if child.terminal is None:
            cands, priors = self._node_candidates(child.obs, rng)
            child.edges = [[list(a), None, 0, 0.0] for a in cands]
            child.priors = priors
        return child

    def _resolve_chance(self, sid, obs, rng):
        """Coin selects are chance nodes: sample uniformly from the agent
        Rng and step past them (one sampled outcome per tree edge; rollouts
        resample on every traversal — docs/mcts-design.md)."""
        while _obs_result(obs) == -1 and obs.select is not None \
                and getattr(obs.select, "context", None) == _CTX_COIN:
            n, lo, hi = _sel_bounds(obs.select)
            k = max(lo, min(1, hi))
            action = sorted(rng.sample(range(n), k)) if n else []
            prev = sid
            sid, obs = self.backend.step(sid, action)
            self.backend.release(prev)
        return sid, obs

    def _node_candidates(self, obs, rng) -> tuple:
        cfg = self.config
        sel = obs.select
        n, lo, hi = _sel_bounds(sel)
        if lo == hi and lo in (0, n):
            return ([list(range(n)) if lo else []], [1.0])
        scores = [self._tier_score(sel, opt, rng) for opt in sel.option]
        order = sorted(range(n), key=lambda i: (-scores[i], i))
        if lo == hi == 1:
            picked = order[:cfg.max_child_actions]
            return ([[i] for i in picked],
                    _softmax([scores[i] for i in picked],
                             cfg.prior_temperature))
        k_pref = self._preferred_count(sel, lo, hi)
        cands = [tuple(sorted(order[:k_pref]))]
        for k in (lo, hi):
            cands.append(tuple(sorted(order[:k])))
        attempts = 0
        while len(set(cands)) < cfg.max_child_actions and attempts < 20:
            attempts += 1
            k = rng.randint(lo, hi)
            cands.append(tuple(sorted(rng.sample(range(n), k))))
        uniq = list(dict.fromkeys(cands))[:cfg.max_child_actions]
        prior_scores = [sum(scores[i] for i in a) for a in uniq]
        return ([list(a) for a in uniq],
                _softmax(prior_scores, cfg.prior_temperature))

    # ---- rollout ------------------------------------------------------------

    def _rollout(self, node, root_player, rng, deadline) -> float:
        sid, obs = node.sid, node.obs
        start_turn = getattr(getattr(obs, "current", None), "turn", 0) or 0
        turn_cap = start_turn + self.config.rollout_turns
        transients = []
        try:
            for _ in range(self.config.rollout_depth):
                if _obs_result(obs) != -1 or obs.select is None:
                    break
                if (getattr(obs.current, "turn", 0) or 0) >= turn_cap:
                    break
                if self._clock() >= deadline:
                    break
                action = self._rollout_action(obs, rng)
                sid, obs = self.backend.step(sid, action)
                transients.append(sid)
            terminal = _terminal_value(obs, root_player)
            if terminal is not None:
                return terminal
            return self.evaluator.evaluate(obs, root_player)
        finally:
            for t in transients:
                try:
                    self.backend.release(t)
                except Exception:
                    pass

    def _rollout_action(self, obs, rng) -> list:
        sel = obs.select
        n, lo, hi = _sel_bounds(sel)
        if n == 0:
            return []
        context = getattr(sel, "context", -1)
        if context == _CTX_COIN:  # chance: uniform, resampled every rollout
            return sorted(rng.sample(range(n), max(lo, min(1, hi))))
        if self.config.rollout == "random":
            return sorted(rng.sample(range(n), rng.randint(lo, hi)))
        if self.config.rollout == "greedy":
            # Full-strength GreedyAgent for BOTH sides (deterministic; only
            # coin outcomes above keep rollouts stochastic). asdict() maps
            # the engine's dataclass observation back to the raw-dict shape
            # the [1] Observation Adapter expects.
            try:
                view = adapt({"select": asdict(sel),
                              "current": asdict(obs.current)})
                return self._greedy.choose(view)
            except Exception:
                pass  # non-dataclass double etc.: use the tier policy below
        scores = [self._tier_score(sel, opt, rng, obs) for opt in sel.option]
        order = sorted(range(n), key=lambda i: (-scores[i], i))
        k = self._preferred_count(sel, lo, hi)
        return sorted(order[:k])

    def _tier_score(self, sel, opt, rng, obs=None) -> float:
        """Type-tier heuristic score + small random tiebreak jitter.

        ATTACK options additionally get a card-attribute damage estimate
        (weakness x2 / resistance -30 / KO + prize bonus, as in
        GreedyAgent._attack_score) so rollouts take lethal instead of
        picking attacks blindly — all from CardIndex attribute lookups,
        cheap enough for the rollout hot loop.
        """
        t = getattr(opt, "type", -1)
        context = getattr(sel, "context", -1)
        if t == _OT_YES:
            base = 30.0 if context in _YES_CONTEXTS else -30.0
        elif t == _OT_NO:
            base = -30.0 if context in _YES_CONTEXTS else 30.0
        elif t == _OT_NUMBER:
            number = getattr(opt, "number", 0) or 0
            base = float(number if context in _COUNT_MAX_CONTEXTS else -number)
        elif t == 13 and obs is not None:  # ATTACK
            base = 20.0 + 0.01 * self._attack_estimate(obs, opt)
        elif t == 8:  # ATTACH: prefer the Active Pokémon (AreaType.ACTIVE=4)
            base = _TIER[8] + (10.0 if getattr(opt, "inPlayArea", None) == 4
                               else 0.0)
        else:
            base = _TIER.get(t, _TIER_DEFAULT)
        return base + rng.random()

    def _attack_estimate(self, obs, opt) -> float:
        cards = self.cards
        damage = float(cards.attack(getattr(opt, "attackId", None)).damage)
        current = obs.current
        actor = getattr(current, "yourIndex", 0)
        players = getattr(current, "players", None) or ()
        if len(players) < 2:
            return damage
        me, opp = players[actor], players[1 - actor]
        my_active = list(getattr(me, "active", None) or ())
        attacker_type = -1
        if my_active and my_active[0] is not None:
            attacker_type = cards.card(
                getattr(my_active[0], "id", None)).energy_type
        opp_active = list(getattr(opp, "active", None) or ())
        defender = opp_active[0] if opp_active else None
        if defender is not None:
            d = cards.card(getattr(defender, "id", None))
            if d.weakness is not None and d.weakness == attacker_type:
                damage *= 2
            elif d.resistance is not None and d.resistance == attacker_type:
                damage = max(0.0, damage - 30.0)
            hp = getattr(defender, "hp", 0) or 0
            if 0 < hp <= damage:
                damage += 300.0 + 150.0 * d.prize_value  # KO bonus
        return damage

    @staticmethod
    def _preferred_count(sel, lo, hi) -> int:
        context = getattr(sel, "context", -1)
        if context in _COST_CONTEXTS:
            return lo
        if context in _COUNT_MAX_CONTEXTS:
            return hi
        if isinstance(context, int) and 0 <= context <= 48:
            return hi
        return lo  # unknown context: commit to as little as allowed

    # ---- aggregation ----------------------------------------------------------

    @staticmethod
    def _best_action(candidates, worlds, deviate_margin: float = 0.0) -> list:
        totals = [[0, 0.0] for _ in candidates]
        for world in worlds:
            for i, edge in enumerate(world.root.edges):
                totals[i][0] += edge[2]
                totals[i][1] += edge[3]
        best_i = 0
        best_key = None
        for i, (visits, value) in enumerate(totals):
            key = (visits, value, -i)  # ties: prefer the greedy-prior order
            if best_key is None or key > best_key:
                best_i, best_key = i, key
        if deviate_margin > 0.0 and best_i != 0 and totals[0][0]:
            challenger = totals[best_i][1] / totals[best_i][0]
            incumbent = totals[0][1] / totals[0][0]
            if challenger < incumbent + deviate_margin:
                best_i = 0  # not enough evidence to leave the greedy prior
        return candidates[best_i]
