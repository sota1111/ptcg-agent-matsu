"""Turn solver (SOT-1677) — gain-loss greedy intra-turn action sequencing.

Reproduces Świechowski et al. (arXiv:1808.04794) §III-B3 "Board Solver" for
cabt: instead of letting tree search branch over every intra-turn ordering
(energy attach / evolve / items / retreat / attack …), a greedy solver
builds the WHOLE remaining turn as one action sequence, scored step by step
with a gain-loss measure derived only from card attributes. The MCTS
planner uses it as a macro action (one tree edge = candidate action +
solver-completed turn) when `PlannerConfig.solver` is on — the paper's
mctsS variant.

Design constraints (issue SOT-1677):
- Legality comes ONLY from the engine's select (Action Enumerator contract,
  agents/actions.py): the solver reorders/picks among the offered options
  and never invents indices, so it cannot emit an illegal move.
- Pure-function scoring: `score_option` reads card attributes through
  CardIndex (no card names/IDs) and the current select; no state is
  mutated. The only randomness is coin sampling from the injected Rng, so
  the same seed yields the same action sequence against fixed engine
  responses (ASSUMPTIONS.md A-9 scope).
- Bounded work, analogous to the paper's <=64 attack-pair evaluations:
  option scoring stops extending the sequence once `max_evals` option
  evaluations are spent; `max_steps` and an optional deadline bound the
  walk itself (anytime — stopping early just yields a shorter macro).

Gain-loss scoring. Each option's score is gain − loss:
- gain: attack damage estimate (weakness x2 / resistance −30, KO + prize
  bonus), value of cards entering play (evolve/play/attach), heal amounts,
  beneficial counts — all from CardIndex attributes.
- loss: resources paid in cost contexts (discards, energy returned),
  detrimental counts, and — for turn-ending options (ATTACK/END) — an
  OPPORTUNITY cost proportional to the development options still available
  in the same select. That last term is what sequences the turn: develop
  first while development has positive net value, attack when its gain
  (especially lethal) dominates what is foregone, END only when nothing
  else is worth doing. This replaces the fixed type-tier ordering
  (SOT-1671 lesson) with a quantity derived from the current option list.
"""
import time
from dataclasses import dataclass

from .greedy_agent import (_COST_CONTEXTS, _COUNT_MAX_CONTEXTS,
                           _HEAL_TARGET_CONTEXTS, _YES_CONTEXTS)

# OptionType values (cg/api.py:120-187) as plain ints, as elsewhere in the
# package (importable without the engine library).
_OT_NUMBER, _OT_YES, _OT_NO = 0, 1, 2
_OT_TOOL_CARD, _OT_ENERGY_CARD, _OT_ENERGY = 4, 5, 6
_OT_PLAY, _OT_ATTACH, _OT_EVOLVE, _OT_ABILITY = 7, 8, 9, 10
_OT_RETREAT, _OT_ATTACK, _OT_END, _OT_SKILL = 12, 13, 14, 15
_AREA_ACTIVE = 4  # AreaType.ACTIVE (cg/api.py:11-23)
_CTX_COIN = 46    # SelectContext.COIN_HEAD — chance, not a decision

# Option types that develop the board (playing them does not end the turn).
_DEVELOPMENT_TYPES = frozenset(
    {_OT_PLAY, _OT_ATTACH, _OT_EVOLVE, _OT_ABILITY, _OT_SKILL})
# Per-foregone-option opportunity cost charged to turn-ending options.
_OPPORTUNITY_COST = 25.0


def _obs_result(obs) -> int:
    current = getattr(obs, "current", None)
    r = getattr(current, "result", -1) if current is not None else -1
    return -1 if r is None else r


def _obs_actor(obs) -> int:
    current = getattr(obs, "current", None)
    return getattr(current, "yourIndex", 0) if current is not None else 0


def _obs_turn(obs) -> int:
    current = getattr(obs, "current", None)
    return (getattr(current, "turn", 0) or 0) if current is not None else 0


def _sel_bounds(sel) -> tuple:
    n = len(sel.option or ())
    hi = min(max(getattr(sel, "maxCount", 0) or 0, 0), n)
    lo = min(max(getattr(sel, "minCount", 0) or 0, 0), hi)
    return n, lo, hi


@dataclass
class SolveResult:
    """One solver run: the macro action sequence and where it stopped."""
    actions: list   # list[list[int]] — one entry per engine select stepped
    sid: int        # search id at the stop point (NOT released)
    obs: object     # observation at the stop point
    evals: int      # option evaluations spent (<= max_evals)
    stop: str       # terminal | turn_end | eval_cap | deadline | step_cap


class TurnSolver:
    """Gain-loss greedy sequencer over the engine's legal selects."""

    def __init__(self, card_index, max_evals: int = 64, max_steps: int = 30):
        self.cards = card_index
        self.max_evals = max_evals
        self.max_steps = max_steps

    # ---- pure per-option scoring -----------------------------------------

    def score_option(self, sel, opt, obs) -> float:
        return self._gain(sel, opt, obs) - self._loss(sel, opt)

    def _gain(self, sel, opt, obs) -> float:
        t = getattr(opt, "type", -1)
        context = getattr(sel, "context", -1)
        if t == _OT_ATTACK:
            return self._attack_gain(obs, opt)
        if t == _OT_EVOLVE:
            return 60.0 + 0.2 * self._entering_value(opt)
        if t == _OT_ABILITY:
            return 55.0
        if t == _OT_PLAY:
            return 40.0 + 0.1 * self._entering_value(opt)
        if t == _OT_ATTACH:
            return 50.0 + (10.0 if getattr(opt, "inPlayArea", None)
                           == _AREA_ACTIVE else 0.0)
        if t == _OT_SKILL:
            return 15.0
        if t == _OT_RETREAT:
            return 4.0
        if t == _OT_END:
            return 0.0
        if t == _OT_YES:
            return 30.0 if context in _YES_CONTEXTS else 0.0
        if t == _OT_NO:
            return 0.0 if context in _YES_CONTEXTS else 30.0
        if t == _OT_NUMBER:
            number = float(getattr(opt, "number", 0) or 0)
            return number if context in _COUNT_MAX_CONTEXTS else 0.0
        if t in (_OT_TOOL_CARD, _OT_ENERGY_CARD, _OT_ENERGY):
            count = float(getattr(opt, "count", 0) or 1)
            return 0.0 if context in _COST_CONTEXTS else count
        # CARD and unknown types: neutral positive so a real target always
        # beats nothing; heal contexts have no per-option amount attribute,
        # cost contexts are handled by _loss.
        if context in _HEAL_TARGET_CONTEXTS:
            return 20.0
        return 10.0

    def _loss(self, sel, opt) -> float:
        t = getattr(opt, "type", -1)
        context = getattr(sel, "context", -1)
        if t in (_OT_ATTACK, _OT_END):
            # Opportunity cost of ending the turn: what development the
            # same select still offers and would be foregone.
            foregone = sum(1 for o in (sel.option or ())
                           if getattr(o, "type", -1) in _DEVELOPMENT_TYPES)
            return _OPPORTUNITY_COST * foregone + (5.0 if t == _OT_END
                                                   else 0.0)
        if context in _COST_CONTEXTS:
            if t == _OT_NUMBER:
                return float(getattr(opt, "number", 0) or 0)
            if t in (_OT_TOOL_CARD, _OT_ENERGY_CARD, _OT_ENERGY):
                return float(getattr(opt, "count", 0) or 1)
            return 0.1 * self._entering_value(opt)  # pay the cheapest card
        if t == _OT_NUMBER and context not in _COUNT_MAX_CONTEXTS:
            return float(getattr(opt, "number", 0) or 0)
        if t == _OT_RETREAT:
            return 2.0
        return 0.0

    def _attack_gain(self, obs, opt) -> float:
        """Damage estimate from card attributes (weakness x2, resistance
        −30, KO + prize bonus) — the paper's per-attack gain measure."""
        cards = self.cards
        damage = float(cards.attack(getattr(opt, "attackId", None)).damage)
        current = getattr(obs, "current", None)
        players = getattr(current, "players", None) or ()
        if len(players) < 2:
            return damage
        actor = _obs_actor(obs)
        me, opp = players[actor], players[1 - actor]
        attacker_type = -1
        my_active = list(getattr(me, "active", None) or ())
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
                damage += 300.0 + 150.0 * d.prize_value  # lethal
        return damage

    def _entering_value(self, opt) -> float:
        """Attribute value of the card an option references, when the
        engine exposes a card id on the option; neutral otherwise."""
        f = self.cards.card(getattr(opt, "cardId", None))
        value = 0.4 * f.hp + float(f.max_attack_damage) - 2.0 * f.retreat_cost
        if f.stage1:
            value += 15.0
        elif f.stage2:
            value += 25.0
        return value

    # ---- per-select greedy choice ----------------------------------------

    def choose(self, sel, obs) -> list:
        """Greedy legal action for one select: rank options by gain-loss,
        take the preferred count. Deterministic (index tie-break)."""
        n, lo, hi = _sel_bounds(sel)
        if n == 0 or (lo == hi == 0):
            return []
        scores = [self.score_option(sel, opt, obs) for opt in sel.option]
        order = sorted(range(n), key=lambda i: (-scores[i], i))
        if lo == hi:
            k = lo
        else:
            context = getattr(sel, "context", -1)
            if context in _COST_CONTEXTS:
                k = lo
            elif context in _COUNT_MAX_CONTEXTS:
                k = hi
            else:
                # Free choice: take every net-positive option allowed.
                positive = sum(1 for s in scores if s > 0.0)
                k = min(hi, max(lo, positive))
        return sorted(order[:k])

    # ---- turn walk --------------------------------------------------------

    def solve(self, backend, sid, obs, rng, deadline=None,
              clock=time.perf_counter, release_initial=False) -> SolveResult:
        """Greedily play out the CURRENT actor's remaining turn through
        `backend`. Steps coin selects as sampled chance (they belong to no
        actor); stops at terminal states, turn/actor boundaries, the
        evaluation cap, the step cap, or the deadline. Intermediate search
        states are released; the initial `sid` only when `release_initial`
        (i.e. the caller treats it as transient too) AND the solver stepped
        past it. The final state is never released."""
        initial_sid = sid
        start_actor, start_turn = _obs_actor(obs), _obs_turn(obs)
        actions = []
        evals = 0
        stop = "step_cap"
        for _ in range(self.max_steps):
            if _obs_result(obs) != -1 or getattr(obs, "select", None) is None:
                stop = "terminal"
                break
            sel = obs.select
            n, lo, hi = _sel_bounds(sel)
            if getattr(sel, "context", None) == _CTX_COIN:
                # Chance node: uniform sample, no evaluation cost.
                action = sorted(rng.sample(range(n), max(lo, min(1, hi)))) \
                    if n else []
            elif _obs_actor(obs) != start_actor or _obs_turn(obs) != start_turn:
                stop = "turn_end"
                break
            elif deadline is not None and clock() >= deadline:
                stop = "deadline"
                break
            elif evals + n > self.max_evals:
                stop = "eval_cap"
                break
            else:
                evals += n
                action = self.choose(sel, obs)
            prev = sid
            sid, obs = backend.step(sid, action)
            actions.append(action)
            if prev != initial_sid or release_initial:
                try:
                    backend.release(prev)
                except Exception:
                    pass
        return SolveResult(actions, sid, obs, evals, stop)
