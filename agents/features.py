"""Value-function features (SOT-1674 v1, SOT-1676 v2) — shared by training
and inference.

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

`featurize_v2` (SOT-1676) is the second, extended extractor reproducing
arXiv:1808.04794 §IV-B: on top of the v1 side scalars it adds per-slot
(Active + 5 Bench) low-level attributes and learned card embeddings
(agents.cards.CardEmbeddings), hand/discard embedding means, and the
stadium embedding. Both extractors coexist; models pick one via their
`feature_set` field (see make_featurizer / evaluator.LearnedEvaluator).
"""
from .cards import CardIndex, CardEmbeddings, DEFAULT_EMBED_DIM, \
    shared_embeddings

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


# ---------------------------------------------------------------------------
# v2 extended board vector (SOT-1676, arXiv:1808.04794 §IV-B)

BENCH_SLOTS = 5   # engine benchMax (tests/support.py, PlayerState.benchMax)
V2_SLOTS = 1 + BENCH_SLOTS  # Active + Bench, fixed positions

_SLOT_FEATURES = (
    "present",      # a Pokémon occupies this slot
    "hidden",       # facedown: presence known, identity/stats hidden
    "hp",
    "max_hp",
    "damage",       # max_hp - hp
    "energy",       # attached Energy count
    "retreat",      # retreat cost (card attribute)
    "prize_value",  # prizes the opponent takes on KO (ex/megaEx)
    "max_attack",   # best attack damage (card attribute)
    "basic",
    "stage1",
    "stage2",
    "tera",
    "has_ability",
    "appeared",     # played this turn
)

_EMPTY_EMBEDDINGS = CardEmbeddings.empty()


def feature_names_v2(dim: int = DEFAULT_EMBED_DIM) -> tuple:
    """Feature names for the v2 vector at a given embedding dimension."""
    names = []
    for prefix in ("my_", "opp_"):
        names += [f"{prefix}{name}" for name in _SIDE_FEATURES]
        for slot in range(V2_SLOTS):
            names += [f"{prefix}slot{slot}_{name}" for name in _SLOT_FEATURES]
            names += [f"{prefix}slot{slot}_emb{j}" for j in range(dim)]
        names += [f"{prefix}hand_known"]
        names += [f"{prefix}hand_emb{j}" for j in range(dim)]
        names += [f"{prefix}discard_count"]
        names += [f"{prefix}discard_emb{j}" for j in range(dim)]
    names += ["turn", "my_turn", "stadium_present"]
    names += [f"stadium_emb{j}" for j in range(dim)]
    return tuple(names)


FEATURE_NAMES_V2 = feature_names_v2()

_ABSENT = object()  # empty slot marker (distinct from facedown None)


def _slot_features(pk, cards: CardIndex, emb: CardEmbeddings) -> list:
    if pk is _ABSENT:
        return [0.0] * len(_SLOT_FEATURES) + [0.0] * emb.dim
    if pk is None:  # facedown: identity unknown -> default embedding
        row = [0.0] * len(_SLOT_FEATURES)
        row[0] = row[1] = 1.0
        return row + list(emb.default)
    card_id = _get(pk, "id")
    card = cards.card(card_id)
    hp = float(_get(pk, "hp", 0))
    max_hp = float(_get(pk, "maxHp", 0))
    return [
        1.0,
        0.0,
        hp / 100.0,
        max_hp / 100.0,
        max(0.0, max_hp - hp) / 100.0,
        float(len(_get(pk, "energies", ()) or ())),
        float(card.retreat_cost),
        float(card.prize_value),
        float(card.max_attack_damage) / 100.0,
        float(card.basic),
        float(card.stage1),
        float(card.stage2),
        float(card.tera),
        float(card.has_ability),
        float(bool(_get(pk, "appearThisTurn", False))),
    ] + list(emb.vector(card_id))


def _card_ids(cards_list) -> list:
    return [_get(c, "id") for c in (cards_list or ()) if c is not None]


def _side_features_v2(side, cards: CardIndex, emb: CardEmbeddings) -> list:
    out = _side_features(side, cards)
    active = list(_get(side, "active", ()) or ())[:1]
    bench = list(_get(side, "bench", ()) or ())[:BENCH_SLOTS]
    slots = active + [_ABSENT] * (1 - len(active))
    slots += bench + [_ABSENT] * (BENCH_SLOTS - len(bench))
    for pk in slots:
        out += _slot_features(pk, cards, emb)
    hand = _get(side, "hand", None)
    out.append(float(hand is not None))
    out += emb.mean(_card_ids(hand)) if hand is not None else [0.0] * emb.dim
    discard = list(_get(side, "discard", ()) or ())
    out.append(len(discard) / 10.0)
    out += emb.mean(_card_ids(discard))
    return out


def featurize_v2(obs, root_player: int, card_index: CardIndex | None = None,
                 embeddings: CardEmbeddings | None = None) -> list:
    """Extended board vector (v2) from `root_player`'s perspective.

    Layout matches feature_names_v2(embeddings.dim): per side the v1 scalars,
    then Active + 5 Bench slots (low-level attributes + card embedding), the
    hand embedding mean (visible for self only), the discard embedding mean;
    globally turn, my_turn and the stadium embedding. Unknown cards use
    CardIndex defaults and the default embedding — never crash.
    """
    cards = card_index if card_index is not None else _EMPTY_INDEX
    emb = embeddings if embeddings is not None else _EMPTY_EMBEDDINGS
    current = _get(obs, "current", {}) or {}
    players = list(_get(current, "players", ()) or ())
    while len(players) < 2:
        players.append({})
    me = players[root_player] if root_player < len(players) else {}
    opp = players[1 - root_player] if 1 - root_player < len(players) else {}
    turn = float(_get(current, "turn", 0))
    actor = _get(current, "yourIndex", 0)
    stadium = list(_get(current, "stadium", ()) or ())
    stadium_id = _get(stadium[0], "id") if stadium else None
    return (_side_features_v2(me, cards, emb)
            + _side_features_v2(opp, cards, emb)
            + [min(turn, TURN_SCALE) / TURN_SCALE,
               float(actor == root_player),
               float(bool(stadium))]
            + (list(emb.vector(stadium_id)) if stadium
               else [0.0] * emb.dim))


def make_featurizer(feature_set: str | None,
                    embeddings: CardEmbeddings | None = None) -> tuple:
    """Resolve a feature-set name -> (feature_names, featurize_fn).

    "v1" (or None) is the original 32-feature extractor; "v2" the extended
    embedding vector. For v2, `embeddings` defaults to the shared
    train/card_embeddings.json table; the returned callable keeps the
    3-argument v1 signature `(obs, root_player, card_index)` so callers
    (LearnedEvaluator, train/gen_selfplay.py) switch sets transparently.
    """
    if feature_set in (None, "v1"):
        return FEATURE_NAMES, featurize
    if feature_set == "v2":
        emb = embeddings if embeddings is not None else shared_embeddings()

        def featurize_with_embeddings(obs, root_player, card_index=None):
            return featurize_v2(obs, root_player, card_index, emb)

        return feature_names_v2(emb.dim), featurize_with_embeddings
    raise ValueError(f"unknown feature set: {feature_set!r}")
