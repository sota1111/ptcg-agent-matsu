"""Greedy one-ply heuristic baseline (SOT-1671).

Scores each option the engine offers and picks the best-scoring legal
selection. All evaluation terms derive ONLY from card attributes (HP, attack
damage, energy requirements, prize impact ex/megaEx, stages, retreat cost)
via `agents.cards.CardIndex` — no per-card weight tables, no card-name/ID
special cases. Unknown cards, attacks, enum values and contexts degrade to
neutral defaults or (at worst) the random-legal fallback in BaseAgent.act —
never a crash, never an illegal move.

Enum values below mirror cg/api.py (SelectType:55-66, SelectContext:68-118,
OptionType:120-187, AreaType:11-23) as plain ints so this package stays
importable without the engine shared library. Unknown/new values simply miss
these tables and take the documented fallbacks.
"""
from . import actions
from .base import BaseAgent
from .cards import CardFeatures, shared_index
from .observation import OptionView, View

# --- SelectType (cg/api.py:55-66) ---
_ST_YES_NO = 9
_ST_COUNT = 8

# --- OptionType (cg/api.py:120-187) ---
_OT_NUMBER = 0
_OT_YES = 1
_OT_NO = 2
_OT_CARD = 3
_OT_TOOL_CARD = 4
_OT_ENERGY_CARD = 5
_OT_ENERGY = 6
_OT_PLAY = 7
_OT_ATTACH = 8
_OT_EVOLVE = 9
_OT_ABILITY = 10
_OT_DISCARD = 11
_OT_RETREAT = 12
_OT_ATTACK = 13
_OT_END = 14
_OT_SKILL = 15
_OT_SPECIAL_CONDITION = 16

# --- AreaType (cg/api.py:11-23) ---
_AREA_DECK = 1
_AREA_HAND = 2
_AREA_DISCARD = 3
_AREA_ACTIVE = 4
_AREA_BENCH = 5
_AREA_PRIZE = 6
_AREA_STADIUM = 7
_AREA_LOOKING = 12

# --- CardType (cg/api.py:39-46) ---
_CT_POKEMON = 0
_CT_ITEM = 1
_CT_SUPPORTER = 3
_CT_STADIUM = 4

# --- SelectContext (cg/api.py:68-118) grouped by selection intent ---
# "Cost" contexts: we are paying (discarding/returning our resources) ->
# select as few and as low-value as allowed.
_COST_CONTEXTS = frozenset({
    8,   # DISCARD
    9,   # TO_DECK
    10,  # TO_DECK_BOTTOM
    11,  # TO_PRIZE
    23,  # DETACH_FROM
    26,  # DISCARD_ENERGY_CARD
    27,  # DISCARD_TOOL_CARD
    29,  # DISCARD_CARD_OR_ATTACHED_CARD
    30,  # DISCARD_ENERGY
    31,  # TO_HAND_ENERGY
    32,  # TO_DECK_ENERGY
})
# Contexts targeting a Pokémon to damage/weaken -> prefer the opponent's
# Pokémon closest to a KO (and worth the most prizes).
_DAMAGE_TARGET_CONTEXTS = frozenset({
    13,  # DAMAGE_COUNTER
    14,  # DAMAGE_COUNTER_ANY
    15,  # DAMAGE
    20,  # DEVOLVE
    36,  # DISABLE_ATTACK (attack of the strongest threat)
})
# Contexts targeting one of our Pokémon to heal -> prefer the most damaged.
_HEAL_TARGET_CONTEXTS = frozenset({
    16,  # REMOVE_DAMAGE_COUNTER
    17,  # HEAL
})
# COUNT contexts where a larger number is beneficial (cg/api.py:107-109).
_COUNT_MAX_CONTEXTS = frozenset({
    38,  # DRAW_COUNT
    39,  # DAMAGE_COUNTER_COUNT
    40,  # REMOVE_DAMAGE_COUNTER_COUNT
})
# YES/NO decisions by context (cg/api.py:110-115); unknown context -> NO
# (never activate effects we cannot evaluate).
_YES_CONTEXTS = frozenset({
    41,  # IS_FIRST: go first
    42,  # MULLIGAN: draw extra cards on opponent mulligan
    43,  # ACTIVATE
    44,  # FIRST_EFFECT
    46,  # COIN_HEAD
})
# Special-condition severity (SpecialConditionType cg/api.py:48-53);
# unknown values score 0.
_SPECIAL_CONDITION_SEVERITY = {
    3: 5,  # PARALYZE
    2: 4,  # SLEEP
    4: 3,  # CONFUSE
    0: 2,  # POISON
    1: 1,  # BURN
}


class GreedyAgent(BaseAgent):
    """One-ply greedy agent over the engine's legal options."""

    def __init__(self, seed: int, deck=None, card_index=None):
        super().__init__(seed, deck)
        self._card_index = card_index

    @property
    def cards(self):
        if self._card_index is None:
            self._card_index = shared_index()
        return self._card_index

    # ---- top-level policy ----------------------------------------------

    def choose(self, view: View) -> list:
        sel = view.select
        lo, hi = actions.count_bounds(sel)
        scores = [self._score_option(view, opt) for opt in sel.options]
        if lo == hi:
            k = lo
        elif sel.context in _COST_CONTEXTS:
            k = lo
        elif sel.context in _COUNT_MAX_CONTEXTS or self._is_known_context(sel.context):
            k = hi
        else:
            k = lo  # unknown context: commit to as little as allowed
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return sorted(ranked[:k])

    def score_options(self, view: View) -> list:
        """Per-option scores for the current selection (used as the root
        prior by the MCTS planner, SOT-1672)."""
        return [self._score_option(view, opt) for opt in view.select.options]

    @staticmethod
    def _is_known_context(context) -> bool:
        # SelectContext values shipped with the engine (cg/api.py:68-118).
        return isinstance(context, int) and 0 <= context <= 48

    # ---- option scoring --------------------------------------------------

    def _score_option(self, view: View, opt: OptionView) -> float:
        t = opt.type
        raw = opt.raw
        if t == _OT_ATTACK:
            # Attacking ends the turn, so it must rank BELOW all development
            # actions (play/attach/evolve/ability) and be chosen only once
            # nothing else is worth doing — but ABOVE retreat/end. The
            # damage estimate only ranks attacks against each other.
            return 20.0 + 0.01 * self._attack_score(view, raw.get("attackId"))
        if t == _OT_ABILITY:
            return 60.0
        if t == _OT_EVOLVE:
            return 70.0 + 0.2 * self._card_value(
                self._resolve_card_id(view, raw.get("area"), raw.get("index"),
                                       raw.get("playerIndex", view.your_index)))
        if t == _OT_PLAY:
            return self._play_score(view, raw.get("index"))
        if t == _OT_ATTACH:
            bonus = 10.0 if raw.get("inPlayArea") == _AREA_ACTIVE else 0.0
            return 45.0 + bonus
        if t == _OT_RETREAT:
            return 4.0
        if t == _OT_END:
            return 0.0
        if t == _OT_YES:
            return 1.0 if view.select.context in _YES_CONTEXTS else -1.0
        if t == _OT_NO:
            return -1.0 if view.select.context in _YES_CONTEXTS else 1.0
        if t == _OT_NUMBER:
            number = raw.get("number") or 0
            return float(number if view.select.context in _COUNT_MAX_CONTEXTS
                         else -number)
        if t == _OT_SPECIAL_CONDITION:
            return float(_SPECIAL_CONDITION_SEVERITY.get(
                raw.get("specialConditionType"), 0))
        if t == _OT_CARD:
            return self._card_target_score(view, raw)
        if t in (_OT_TOOL_CARD, _OT_ENERGY_CARD, _OT_ENERGY):
            value = float(raw.get("count") or 1)
            return -value if view.select.context in _COST_CONTEXTS else value
        if t == _OT_SKILL:
            return 10.0
        return 10.0  # unknown option type: between END and real actions

    def _play_score(self, view: View, hand_index) -> float:
        card = self._hand_card(view, hand_index)
        base = 40.0
        if card.card_type == _CT_SUPPORTER:
            base += 25.0
        elif card.card_type == _CT_ITEM:
            base += 20.0
        elif card.card_type == _CT_STADIUM:
            base += 10.0
        elif card.card_type == _CT_POKEMON and card.basic:
            base += 15.0
        return base + 0.05 * self._features_value(card)

    def _card_target_score(self, view: View, raw: dict) -> float:
        context = view.select.context
        player_index = raw.get("playerIndex", view.your_index)
        area, index = raw.get("area"), raw.get("index")
        if context in _DAMAGE_TARGET_CONTEXTS:
            pokemon = view.find_pokemon(player_index, area, index)
            if pokemon is None:
                return 50.0
            prize = self.cards.card(pokemon.card_id).prize_value
            return 200.0 - float(pokemon.hp) + 20.0 * prize
        if context in _HEAL_TARGET_CONTEXTS:
            pokemon = view.find_pokemon(player_index, area, index)
            if pokemon is None:
                return 0.0
            return float(pokemon.max_hp - pokemon.hp)
        value = self._card_value(
            self._resolve_card_id(view, area, index, player_index))
        return -value if context in _COST_CONTEXTS else value

    # ---- feature helpers --------------------------------------------------

    def _attack_score(self, view: View, attack_id) -> float:
        attack = self.cards.attack(attack_id)
        damage = float(attack.damage)
        attacker_type = -1
        if view.me.active and view.me.active[0] is not None:
            attacker_type = self.cards.card(view.me.active[0].card_id).energy_type
        defender = view.opp.active[0] if view.opp.active else None
        if defender is not None:
            d = self.cards.card(defender.card_id)
            if d.weakness is not None and d.weakness == attacker_type:
                damage *= 2
            elif d.resistance is not None and d.resistance == attacker_type:
                damage = max(0.0, damage - 30.0)
            if 0 < defender.hp <= damage:
                return damage + 300.0 + 150.0 * d.prize_value  # KO bonus
        return damage

    def _card_value(self, card_id) -> float:
        return self._features_value(self.cards.card(card_id))

    def _features_value(self, f: CardFeatures) -> float:
        value = 0.4 * f.hp + float(f.max_attack_damage) - 2.0 * f.retreat_cost
        if f.card_type == _CT_POKEMON:
            value += 10.0
        if f.stage1:
            value += 15.0
        elif f.stage2:
            value += 25.0
        return value

    def _hand_card(self, view: View, hand_index) -> CardFeatures:
        hand = view.me.hand_card_ids or []
        if hand_index is None or not (0 <= hand_index < len(hand)):
            return self.cards.card(None)
        return self.cards.card(hand[hand_index])

    def _resolve_card_id(self, view: View, area, index, player_index):
        """(area, index, playerIndex) option reference -> card ID or None."""
        side = view.me if player_index == view.your_index else view.opp
        if index is None:
            return None
        sources = {
            _AREA_HAND: side.hand_card_ids,
            _AREA_DISCARD: side.discard_card_ids,
            _AREA_DECK: view.select.deck_card_ids if view.select else None,
            _AREA_PRIZE: None,  # prizes are facedown; stay unknown
            _AREA_STADIUM: view.stadium_card_ids,
            _AREA_LOOKING: view.looking_card_ids,
        }
        if area in (_AREA_ACTIVE, _AREA_BENCH):
            pokemon = view.find_pokemon(player_index, area, index)
            return pokemon.card_id if pokemon is not None else None
        cards = sources.get(area)
        if cards is None or not (0 <= index < len(cards)):
            return None
        return cards[index]
