"""Cheater determinization for training-data generation (SOT-1678).

Reproduces Świechowski et al. (arXiv:1808.04794) §III-B1: when GENERATING
learning data (and only then), the MCTS players determinize with the TRUE
hidden state — the opponent's hand, both decks and both prize sets — instead
of sampling them from the information set. The true state comes from the
local engine's `cg.game.visualize_data()` dump, which exposes every zone of
the LIVE battle (verified 2026-07-14: deck order, hands and facedown prizes
all match the observation counts).

Fairness boundary (competition submissions must not cheat):
- This module lives in train/ and is NOT packaged into the submission
  (scripts/build_submission.sh archives main.py + deck.csv + agents/ + cg/).
- Nothing under agents/ imports it or calls visualize_data; the only hook is
  MctsPlanner's constructor-injected `fills_fn`, which `agents.make_agent` /
  `MctsAgent` cannot reach (their config kwargs go to PlannerConfig, which
  has no such field). tests/test_cheater.py pins all of this.
- A submitted agent could not use this path even intentionally:
  visualize_data() reads the local battle pointer that only our own
  battle_start() loop owns; the Kaggle harness never hands it to an agent.
"""
import json

from agents.mcts_agent import MctsAgent
from agents.planner import Fills, MctsPlanner


def parse_true_state(visualize_json: str) -> dict:
    """`cg.game.visualize_data()` payload -> the CURRENT true state dict.

    The payload is a list with one snapshot per select so far; the last
    entry's "current" is the full present state (both hands, both decks in
    order, prize identities — nothing masked).
    """
    snapshots = json.loads(visualize_json)
    if not snapshots:
        raise ValueError("empty visualize_data payload")
    state = snapshots[-1].get("current")
    if not isinstance(state, dict):
        raise ValueError("visualize_data snapshot has no current state")
    return state


def _ids(cards) -> list:
    return [c.get("id") for c in (cards or ()) if isinstance(c, dict)]


def true_fills(obs_current: dict, state_current: dict) -> Fills:
    """True-state Fills for the acting player of `obs_current`.

    `obs_current` is the acting player's observation "current" dict (defines
    the perspective and which zones the engine expects predictions for);
    `state_current` is the matching parse_true_state() dict. Zone sizes are
    checked against the observation so a stale/mismatched state fails loudly
    instead of feeding search_begin inconsistent fills.
    """
    yi = obs_current.get("yourIndex", 0)
    tplayers = list(state_current.get("players") or ())
    oplayers = list(obs_current.get("players") or ())
    if len(tplayers) < 2 or len(oplayers) < 2:
        raise ValueError("true state / observation lacks two players")
    me_t, opp_t = tplayers[yi], tplayers[1 - yi]
    me_o, opp_o = oplayers[yi], oplayers[1 - yi]

    fills = Fills(
        my_deck=_ids(me_t.get("deck")),
        my_prize=_ids(me_t.get("prize")),
        opp_deck=_ids(opp_t.get("deck")),
        opp_prize=_ids(opp_t.get("prize")),
        opp_hand=_ids(opp_t.get("hand")),
        opp_active=[],
    )
    expected = (
        (fills.my_deck, me_o.get("deckCount", 0) or 0, "my_deck"),
        (fills.my_prize, len(me_o.get("prize") or ()), "my_prize"),
        (fills.opp_deck, opp_o.get("deckCount", 0) or 0, "opp_deck"),
        (fills.opp_prize, len(opp_o.get("prize") or ()), "opp_prize"),
        (fills.opp_hand, opp_o.get("handCount", 0) or 0, "opp_hand"),
    )
    for got, want, name in expected:
        if len(got) != want:
            raise ValueError(
                f"true-state {name} has {len(got)} cards, observation "
                f"expects {want}")

    # Facedown opponent Active (None in the observation): predict it with
    # its TRUE identity from the state dump.
    obs_active = list(opp_o.get("active") or ())
    if obs_active and obs_active[0] is None:
        true_active = [p for p in (opp_t.get("active") or ())
                       if isinstance(p, dict)]
        if not true_active or true_active[0].get("id") is None:
            raise ValueError("facedown opponent Active missing in true state")
        fills.opp_active.append(true_active[0]["id"])
    return fills


class CheaterMctsAgent(MctsAgent):
    """MctsAgent whose determinizations are the true hidden state.

    The generation loop calls `set_true_fills()` with the current decision's
    true_fills() before each act(); the planner then builds every world from
    that single true state (n_worlds=1 is the natural setting — with exact
    fills there is no hidden-information distribution left to average over).
    A decision without fills raises inside the world build, which the planner
    contains as a degraded (greedy-prior) decision — surfaced through the
    inherited degraded_count so generation stats can report it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._true_fills = None

    def set_true_fills(self, fills: Fills) -> None:
        self._true_fills = fills

    def _cheater_fills(self, raw_obs, own_deck, rng, card_index) -> Fills:
        if self._true_fills is None:
            raise ValueError("true fills not set for this decision")
        return self._true_fills

    @property
    def planner(self) -> MctsPlanner:
        if self._planner is None:
            self._planner = MctsPlanner(
                own_deck=self._deck, config=self.config,
                evaluator=self._evaluator, backend=self._backend,
                card_index=self._card_index, clock=self._clock,
                fills_fn=self._cheater_fills)
        return self._planner
