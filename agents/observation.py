"""Observation Adapter (SOT-1671) — layer [1] of the 4-layer architecture.

Converts the raw cabt observation dict into an information-set view from the
acting player's perspective (`View`): only information visible to the agent
(own hand, public board, counts, face-up prizes) is represented — hidden
zones stay as counts / None, exactly as the engine reports them.

The adapter is defensive by design: it reads the raw dict with defaults and
keeps enum values as plain ints, so unknown enum values or newly appended
attributes (cg/api.py:118,328) pass through without crashing.
"""
from dataclasses import dataclass, field


def _card_id(card_dict) -> int | None:
    """Card ID from a raw Card dict; None for facedown/absent cards."""
    if not isinstance(card_dict, dict):
        return None
    return card_dict.get("id")


@dataclass
class PokemonView:
    card_id: int | None       # None if the Pokémon is facedown
    serial: int | None
    hp: int
    max_hp: int
    appear_this_turn: bool
    energies: list            # EnergyType ints (unknown values preserved)
    energy_card_ids: list
    tool_ids: list
    pre_evolution_ids: list


@dataclass
class SideView:
    active: list              # list[PokemonView | None] (size 0 or 1)
    bench: list               # list[PokemonView]
    bench_max: int
    deck_count: int
    hand_count: int
    hand_card_ids: list | None   # None for the opponent (hidden)
    discard_card_ids: list
    prize_count: int
    prize_known_ids: list        # IDs of face-up prizes only
    poisoned: bool = False
    burned: bool = False
    asleep: bool = False
    paralyzed: bool = False
    confused: bool = False


@dataclass
class OptionView:
    index: int
    type: int                 # OptionType int (unknown values preserved)
    raw: dict


@dataclass
class SelectView:
    type: int                 # SelectType int
    context: int              # SelectContext int
    min_count: int
    max_count: int
    remain_damage_counter: int
    remain_energy_cost: int
    options: list             # list[OptionView]
    deck_card_ids: list | None
    context_card_id: int | None
    effect_card_id: int | None
    raw: dict = field(default_factory=dict)


@dataclass
class View:
    your_index: int
    turn: int
    turn_action_count: int
    first_player: int
    result: int               # -1 while the battle is running
    supporter_played: bool
    stadium_played: bool
    energy_attached: bool
    retreated: bool
    stadium_card_ids: list
    looking_card_ids: list | None
    me: SideView
    opp: SideView
    select: SelectView | None  # None only at the initial deck selection
    logs: list                 # raw log dicts
    raw: dict                  # the original observation dict

    def find_pokemon(self, player_index: int, area, index) -> PokemonView | None:
        """Resolve an (area, index) option reference to a Pokémon in play.

        AreaType.ACTIVE == 4, AreaType.BENCH == 5 (cg/api.py:11-23).
        Returns None when the reference cannot be resolved.
        """
        side = self.me if player_index == self.your_index else self.opp
        if area == 4:
            pokemons = side.active
        elif area == 5:
            pokemons = side.bench
        else:
            return None
        if index is None or not (0 <= index < len(pokemons)):
            return None
        return pokemons[index]


def _adapt_pokemon(p) -> PokemonView | None:
    if not isinstance(p, dict):
        return None  # facedown Pokémon are reported as None
    return PokemonView(
        card_id=p.get("id"),
        serial=p.get("serial"),
        hp=p.get("hp", 0) or 0,
        max_hp=p.get("maxHp", 0) or 0,
        appear_this_turn=bool(p.get("appearThisTurn", False)),
        energies=list(p.get("energies") or ()),
        energy_card_ids=[_card_id(c) for c in (p.get("energyCards") or ())],
        tool_ids=[_card_id(c) for c in (p.get("tools") or ())],
        pre_evolution_ids=[_card_id(c) for c in (p.get("preEvolution") or ())],
    )


def _adapt_side(p) -> SideView:
    p = p if isinstance(p, dict) else {}
    hand = p.get("hand")
    prize = p.get("prize") or ()
    return SideView(
        active=[_adapt_pokemon(x) for x in (p.get("active") or ())],
        bench=[pv for pv in (_adapt_pokemon(x) for x in (p.get("bench") or ()))
               if pv is not None],
        bench_max=p.get("benchMax", 0) or 0,
        deck_count=p.get("deckCount", 0) or 0,
        hand_count=p.get("handCount", 0) or 0,
        hand_card_ids=None if hand is None else [_card_id(c) for c in hand],
        discard_card_ids=[_card_id(c) for c in (p.get("discard") or ())],
        prize_count=len(prize),
        prize_known_ids=[_card_id(c) for c in prize if isinstance(c, dict)],
        poisoned=bool(p.get("poisoned", False)),
        burned=bool(p.get("burned", False)),
        asleep=bool(p.get("asleep", False)),
        paralyzed=bool(p.get("paralyzed", False)),
        confused=bool(p.get("confused", False)),
    )


def _adapt_select(sel) -> SelectView | None:
    if not isinstance(sel, dict):
        return None
    deck = sel.get("deck")
    options = sel.get("option") or ()
    return SelectView(
        type=sel.get("type", -1),
        context=sel.get("context", -1),
        min_count=sel.get("minCount", 0) or 0,
        max_count=sel.get("maxCount", 0) or 0,
        remain_damage_counter=sel.get("remainDamageCounter", 0) or 0,
        remain_energy_cost=sel.get("remainEnergyCost", 0) or 0,
        options=[OptionView(index=i,
                            type=(o.get("type", -1) if isinstance(o, dict) else -1),
                            raw=(o if isinstance(o, dict) else {}))
                 for i, o in enumerate(options)],
        deck_card_ids=None if deck is None else [_card_id(c) for c in deck],
        context_card_id=_card_id(sel.get("contextCard")),
        effect_card_id=_card_id(sel.get("effect")),
        raw=sel,
    )


# --------------------------------------------------------------------------- #
# Fast path: build the View straight from the engine's dataclass Observation
# (agents.planner rollout hot loop, SOT-1697). The rollout used to convert the
# whole search-API observation back to a raw dict with dataclasses.asdict()
# just so GreedyAgent could score it; profiling showed that recursive asdict()
# round-trip was ~40% of a champion decision. These helpers mirror the dict
# adapters above field-for-field but read dataclass attributes directly, so the
# resulting View is identical (tests/test_observation.py asserts equality with
# the asdict path) while skipping the deep dict rebuild.
# --------------------------------------------------------------------------- #
def _obj_card_id(card) -> int | None:
    """Card ID from an engine Card dataclass; None for facedown/absent cards."""
    return getattr(card, "id", None) if card is not None else None


def _adapt_pokemon_obj(p) -> PokemonView | None:
    if p is None:
        return None  # facedown Pokémon are reported as None
    return PokemonView(
        card_id=getattr(p, "id", None),
        serial=getattr(p, "serial", None),
        hp=getattr(p, "hp", 0) or 0,
        max_hp=getattr(p, "maxHp", 0) or 0,
        appear_this_turn=bool(getattr(p, "appearThisTurn", False)),
        energies=list(getattr(p, "energies", None) or ()),
        energy_card_ids=[_obj_card_id(c)
                         for c in (getattr(p, "energyCards", None) or ())],
        tool_ids=[_obj_card_id(c) for c in (getattr(p, "tools", None) or ())],
        pre_evolution_ids=[_obj_card_id(c)
                           for c in (getattr(p, "preEvolution", None) or ())],
    )


def _adapt_side_obj(p) -> SideView:
    hand = getattr(p, "hand", None)
    prize = getattr(p, "prize", None) or ()
    return SideView(
        active=[_adapt_pokemon_obj(x) for x in (getattr(p, "active", None) or ())],
        bench=[pv for pv in (_adapt_pokemon_obj(x)
                             for x in (getattr(p, "bench", None) or ()))
               if pv is not None],
        bench_max=getattr(p, "benchMax", 0) or 0,
        deck_count=getattr(p, "deckCount", 0) or 0,
        hand_count=getattr(p, "handCount", 0) or 0,
        hand_card_ids=None if hand is None else [_obj_card_id(c) for c in hand],
        discard_card_ids=[_obj_card_id(c)
                          for c in (getattr(p, "discard", None) or ())],
        prize_count=len(prize),
        prize_known_ids=[_obj_card_id(c) for c in prize if c is not None],
        poisoned=bool(getattr(p, "poisoned", False)),
        burned=bool(getattr(p, "burned", False)),
        asleep=bool(getattr(p, "asleep", False)),
        paralyzed=bool(getattr(p, "paralyzed", False)),
        confused=bool(getattr(p, "confused", False)),
    )


def _adapt_select_obj(sel) -> SelectView | None:
    if sel is None:
        return None
    deck = getattr(sel, "deck", None)
    options = getattr(sel, "option", None) or ()
    return SelectView(
        type=getattr(sel, "type", -1),
        context=getattr(sel, "context", -1),
        min_count=getattr(sel, "minCount", 0) or 0,
        max_count=getattr(sel, "maxCount", 0) or 0,
        remain_damage_counter=getattr(sel, "remainDamageCounter", 0) or 0,
        remain_energy_cost=getattr(sel, "remainEnergyCost", 0) or 0,
        # OptionView.raw exposes the option's scalar fields (attackId, area,
        # index, ...) via .get(); an engine Option is a flat dataclass, so its
        # __dict__ is exactly the dict GreedyAgent read from the asdict path.
        options=[OptionView(index=i,
                            type=getattr(o, "type", -1),
                            raw=vars(o) if hasattr(o, "__dict__") else {})
                 for i, o in enumerate(options)],
        deck_card_ids=None if deck is None else [_obj_card_id(c) for c in deck],
        context_card_id=_obj_card_id(getattr(sel, "contextCard", None)),
        effect_card_id=_obj_card_id(getattr(sel, "effect", None)),
        raw={},
    )


def adapt_engine_obs(obs) -> View:
    """Engine dataclass Observation -> `View` (rollout fast path, SOT-1697).

    Equivalent to ``adapt({"select": asdict(sel), "current": asdict(current)})``
    but without the recursive dict rebuild. ``obs`` is the search-API
    ``Observation`` dataclass (``obs.select``, ``obs.current``).
    """
    current = getattr(obs, "current", None)
    players = (getattr(current, "players", None) or ()) if current is not None else ()
    your_index = getattr(current, "yourIndex", 0) if current is not None else 0
    if not isinstance(your_index, int) or your_index not in (0, 1):
        your_index = 0
    sides = [_adapt_side_obj(players[i]) if i < len(players) else _adapt_side({})
             for i in (your_index, 1 - your_index)]
    looking = getattr(current, "looking", None) if current is not None else None
    result = getattr(current, "result", -1) if current is not None else -1
    return View(
        your_index=your_index,
        turn=(getattr(current, "turn", 0) or 0) if current is not None else 0,
        turn_action_count=(getattr(current, "turnActionCount", 0) or 0)
        if current is not None else 0,
        first_player=getattr(current, "firstPlayer", -1) if current is not None else -1,
        result=result if result is not None else -1,
        supporter_played=bool(getattr(current, "supporterPlayed", False)),
        stadium_played=bool(getattr(current, "stadiumPlayed", False)),
        energy_attached=bool(getattr(current, "energyAttached", False)),
        retreated=bool(getattr(current, "retreated", False)),
        stadium_card_ids=[_obj_card_id(c)
                          for c in ((getattr(current, "stadium", None) or ())
                                    if current is not None else ())],
        looking_card_ids=(None if looking is None
                          else [_obj_card_id(c) for c in looking]),
        me=sides[0],
        opp=sides[1],
        select=_adapt_select_obj(getattr(obs, "select", None)),
        logs=list(getattr(obs, "logs", None) or ()),
        raw=obs,
    )


def adapt(obs_dict: dict) -> View:
    """Raw observation dict -> information-set `View` for the acting player."""
    obs = obs_dict if isinstance(obs_dict, dict) else {}
    current = obs.get("current")
    current = current if isinstance(current, dict) else {}
    players = current.get("players") or ()
    your_index = current.get("yourIndex", 0)
    if not isinstance(your_index, int) or your_index not in (0, 1):
        your_index = 0
    sides = [_adapt_side(players[i]) if i < len(players) else _adapt_side({})
             for i in (your_index, 1 - your_index)]
    looking = current.get("looking")
    return View(
        your_index=your_index,
        turn=current.get("turn", 0) or 0,
        turn_action_count=current.get("turnActionCount", 0) or 0,
        first_player=current.get("firstPlayer", -1),
        result=current.get("result", -1) if current.get("result") is not None else -1,
        supporter_played=bool(current.get("supporterPlayed", False)),
        stadium_played=bool(current.get("stadiumPlayed", False)),
        energy_attached=bool(current.get("energyAttached", False)),
        retreated=bool(current.get("retreated", False)),
        stadium_card_ids=[_card_id(c) for c in (current.get("stadium") or ())],
        looking_card_ids=(None if looking is None
                          else [_card_id(c) for c in looking]),
        me=sides[0],
        opp=sides[1],
        select=_adapt_select(obs.get("select")),
        logs=list(obs.get("logs") or ()),
        raw=obs,
    )
