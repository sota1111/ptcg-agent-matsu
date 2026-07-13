"""Value-function features (SOT-1674) — shared by training and inference.

`featurize(obs, root_player, card_index)` maps a battle observation to a
fixed-length vector of floats from `root_player`'s perspective. It is the
single feature definition: train/gen_selfplay.py calls it on live-match
observations (raw dicts) and LearnedEvaluator calls it on engine search
states (`cg.api.Observation` dataclasses), so every accessor is duck-typed
over both shapes (dict key or attribute).

Every feature derives from state counts or card ATTRIBUTES via
agents.cards.CardIndex (HP, max attack damage, prize value) — per-card
weight tables keyed by card ID/name are forbidden
(scripts/lint_hardcoded_cards.py). Unknown card IDs fall back to
CardIndex's neutral defaults, so the vector never crashes on new cards.
"""
from .cards import CardIndex

PRIZE_START = 6   # PRIZE_SIZE (ptcgProgram 22/Core.h:14)
TURN_SCALE = 30.0  # engine draws around turn 30 (BattleData.h:66-74)

# Neutral index used when no card master is available (features that need
# attribute lookups then take their defaults instead of crashing).
_EMPTY_INDEX = CardIndex()

_SIDE_FEATURES = (
    "prizes_taken",        # cards this side has taken (0-6, dominant signal)
    "pokemon_in_play",     # Active + Bench count
    "bench_count",
    "energy_total",        # Energy attached across the board
    "hp_total",            # current HP in play
    "damage_total",        # max_hp - hp in play (accumulated damage)
    "hand_count",
    "deck_count",
    "deck_empty",          # loses at its next draw (deck-out)
    "active_hp",
    "active_energy",
    "active_max_attack",   # best attack damage of the Active (card attribute)
    "prize_risk",          # sum of prize_value of Pokémon in play (ex/megaEx)
    "status_damage",       # Active Poisoned/Burned (damage-over-time)
    "status_disable",      # Active Asleep/Paralyzed/Confused (action-denial)
)

FEATURE_NAMES = tuple(
    [f"my_{name}" for name in _SIDE_FEATURES]
    + [f"opp_{name}" for name in _SIDE_FEATURES]
    + ["turn", "my_turn"]
)


def _get(obj, name, default=None):
    """Duck-typed field access: dict key or object attribute."""
    if isinstance(obj, dict):
        value = obj.get(name, default)
    else:
        value = getattr(obj, name, default)
    return default if value is None else value


def _side_features(side, cards: CardIndex) -> list:
    prize = _get(side, "prize", ()) or ()
    in_play = list(_get(side, "active", ()) or ())
    in_play += list(_get(side, "bench", ()) or ())

    hp_total = 0.0
    damage_total = 0.0
    energy_total = 0.0
    prize_risk = 0.0
    pokemon = 0
    for pk in in_play:
        pokemon += 1
        if pk is None:  # facedown: presence known, stats hidden
            prize_risk += 1.0
            continue
        hp = float(_get(pk, "hp", 0))
        max_hp = float(_get(pk, "maxHp", 0))
        hp_total += hp
        damage_total += max(0.0, max_hp - hp)
        energy_total += float(len(_get(pk, "energies", ()) or ()))
        prize_risk += float(cards.card(_get(pk, "id")).prize_value)

    active = list(_get(side, "active", ()) or ())
    active_pk = active[0] if active else None
    if active_pk is not None:
        active_hp = float(_get(active_pk, "hp", 0))
        active_energy = float(len(_get(active_pk, "energies", ()) or ()))
        active_max_attack = float(
            cards.card(_get(active_pk, "id")).max_attack_damage)
    else:
        active_hp = active_energy = active_max_attack = 0.0

    deck_count = float(_get(side, "deckCount", 0))
    status_damage = float(bool(_get(side, "poisoned", False))
                          or bool(_get(side, "burned", False)))
    status_disable = float(bool(_get(side, "asleep", False))
                           or bool(_get(side, "paralyzed", False))
                           or bool(_get(side, "confused", False)))
    return [
        float(max(0, PRIZE_START - len(prize))),
        float(pokemon),
        float(len(_get(side, "bench", ()) or ())),
        energy_total,
        hp_total / 100.0,
        damage_total / 100.0,
        float(_get(side, "handCount", 0)),
        deck_count / 10.0,
        float(deck_count == 0),
        active_hp / 100.0,
        active_energy,
        active_max_attack / 100.0,
        prize_risk,
        status_damage,
        status_disable,
    ]


def featurize(obs, root_player: int, card_index: CardIndex | None = None) -> list:
    """Observation -> feature vector (floats) from `root_player`'s side.

    Accepts either the raw observation dict handed to agents in a live match
    or the engine's dataclass search observation; both players' entries carry
    the same visible-count fields (hidden zones are counts either way).
    """
    cards = card_index if card_index is not None else _EMPTY_INDEX
    current = _get(obs, "current", {}) or {}
    players = list(_get(current, "players", ()) or ())
    while len(players) < 2:
        players.append({})
    me = players[root_player] if root_player < len(players) else {}
    opp = players[1 - root_player] if 1 - root_player < len(players) else {}
    turn = float(_get(current, "turn", 0))
    actor = _get(current, "yourIndex", 0)
    return (_side_features(me, cards)
            + _side_features(opp, cards)
            + [min(turn, TURN_SCALE) / TURN_SCALE,
               float(actor == root_player)])
