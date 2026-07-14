"""Card-attribute feature index (SOT-1671) + learned card embeddings (SOT-1676).

Evaluation features are derived ONLY from card attributes exposed by the
engine's card master (`cg.api.all_card_data()` / `all_attack()`): HP, damage,
energy requirements, prize impact (ex/megaEx), stages, retreat cost, etc.
Per-card weight tables keyed by card ID or card name are forbidden
(enforced by scripts/lint_hardcoded_cards.py). `CardEmbeddings` is the one
sanctioned per-card table: its vectors are LEARNED from the master's
attribute/effect text by train/train_embeddings.py (arXiv:1808.04794 §IV-A),
not hand-authored, and are keyed by ID only.

Unknown card / attack IDs fall back to neutral default features — never crash
(the enum/card pool may grow during the competition, cg/api.py:118).
"""
import json
import os
from dataclasses import dataclass


# Neutral defaults for cards/attacks that are missing from the master data.
_DEFAULT_HP = 60
_DEFAULT_RETREAT = 1
_DEFAULT_PRIZE_VALUE = 1


@dataclass(frozen=True)
class CardFeatures:
    card_id: int
    known: bool
    card_type: int          # cg.api.CardType value; -1 if unknown
    hp: int
    retreat_cost: int
    weakness: int | None    # cg.api.EnergyType value of the defender's weakness
    resistance: int | None
    energy_type: int        # attacker's type used for weakness/resistance checks
    basic: bool
    stage1: bool
    stage2: bool
    ex: bool
    mega_ex: bool
    tera: bool
    ace_spec: bool
    has_ability: bool       # card has at least one skill (ability/effect)
    attack_ids: tuple
    max_attack_damage: int
    prize_value: int        # prizes the opponent takes when this is Knocked Out


@dataclass(frozen=True)
class AttackFeatures:
    attack_id: int
    known: bool
    damage: int
    energy_cost: int
    energy_types: tuple


def _default_card(card_id: int) -> CardFeatures:
    return CardFeatures(
        card_id=card_id, known=False, card_type=-1,
        hp=_DEFAULT_HP, retreat_cost=_DEFAULT_RETREAT,
        weakness=None, resistance=None, energy_type=-1,
        basic=False, stage1=False, stage2=False,
        ex=False, mega_ex=False, tera=False, ace_spec=False,
        has_ability=False, attack_ids=(), max_attack_damage=0,
        prize_value=_DEFAULT_PRIZE_VALUE,
    )


def _default_attack(attack_id: int) -> AttackFeatures:
    return AttackFeatures(attack_id=attack_id, known=False, damage=0,
                          energy_cost=0, energy_types=())


def _get(obj, name, default=None):
    """Attribute access tolerant of missing fields (future master columns)."""
    return getattr(obj, name, default)


class CardIndex:
    """Lookup from card/attack ID to attribute-derived features.

    Built from iterables of objects shaped like `cg.api.CardData` / `Attack`
    (duck-typed so tests can inject synthetic masters without the engine).
    """

    def __init__(self, card_data=(), attack_data=()):
        self._attacks = {}
        for a in attack_data:
            aid = _get(a, "attackId")
            if aid is None:
                continue
            energies = tuple(_get(a, "energies", ()) or ())
            self._attacks[int(aid)] = AttackFeatures(
                attack_id=int(aid), known=True,
                damage=int(_get(a, "damage", 0) or 0),
                energy_cost=len(energies), energy_types=energies,
            )
        self._cards = {}
        for c in card_data:
            cid = _get(c, "cardId")
            if cid is None:
                continue
            attack_ids = tuple(int(a) for a in (_get(c, "attacks", ()) or ()))
            mega_ex = bool(_get(c, "megaEx", False))
            ex = bool(_get(c, "ex", False))
            prize_value = 3 if mega_ex else (2 if ex else 1)
            self._cards[int(cid)] = CardFeatures(
                card_id=int(cid), known=True,
                card_type=int(_get(c, "cardType", -1) if _get(c, "cardType") is not None else -1),
                hp=int(_get(c, "hp", 0) or 0),
                retreat_cost=int(_get(c, "retreatCost", 0) or 0),
                weakness=_get(c, "weakness"),
                resistance=_get(c, "resistance"),
                energy_type=int(_get(c, "energyType", -1) if _get(c, "energyType") is not None else -1),
                basic=bool(_get(c, "basic", False)),
                stage1=bool(_get(c, "stage1", False)),
                stage2=bool(_get(c, "stage2", False)),
                ex=ex, mega_ex=mega_ex,
                tera=bool(_get(c, "tera", False)),
                ace_spec=bool(_get(c, "aceSpec", False)),
                has_ability=bool(_get(c, "skills", ()) or ()),
                attack_ids=attack_ids,
                max_attack_damage=max(
                    (self.attack(a).damage for a in attack_ids), default=0),
                prize_value=prize_value,
            )

    @classmethod
    def from_engine(cls) -> "CardIndex":
        """Load the real card master via the cabt engine bindings."""
        from cg.api import all_card_data, all_attack
        return cls(all_card_data(), all_attack())

    def card(self, card_id) -> CardFeatures:
        """Features for a card ID; neutral defaults when unknown/None."""
        if card_id is None:
            return _default_card(-1)
        return self._cards.get(int(card_id)) or _default_card(int(card_id))

    def attack(self, attack_id) -> AttackFeatures:
        """Features for an attack ID; neutral defaults when unknown/None."""
        if attack_id is None:
            return _default_attack(-1)
        return self._attacks.get(int(attack_id)) or _default_attack(int(attack_id))

    def __len__(self) -> int:
        return len(self._cards)


_shared_index = None


def shared_index() -> CardIndex:
    """Process-wide CardIndex loaded from the engine (lazy singleton)."""
    global _shared_index
    if _shared_index is None:
        _shared_index = CardIndex.from_engine()
    return _shared_index


DEFAULT_EMBED_DIM = 10  # arXiv:1808.04794 §IV-A trains 10-dim card vectors
DEFAULT_EMBEDDINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "train", "card_embeddings.json")


class CardEmbeddings:
    """Card ID -> learned embedding vector (SOT-1676).

    Loaded from train/train_embeddings.py output. Unknown / None / facedown
    card IDs fall back to the `default` vector (the mean of all card vectors
    at training time) — never crash, matching CardIndex's contract.
    """

    def __init__(self, dim: int = DEFAULT_EMBED_DIM, vectors=None,
                 default=None):
        self.dim = int(dim)
        self._vectors = {}
        for card_id, vec in (vectors or {}).items():
            vec = [float(x) for x in vec]
            if len(vec) != self.dim:
                raise ValueError(
                    f"embedding for card {card_id} has {len(vec)} dims, "
                    f"expected {self.dim}")
            self._vectors[int(card_id)] = vec
        if default is not None:
            self.default = [float(x) for x in default]
            if len(self.default) != self.dim:
                raise ValueError("default embedding dim mismatch")
        else:
            self.default = [0.0] * self.dim

    @classmethod
    def empty(cls, dim: int = DEFAULT_EMBED_DIM) -> "CardEmbeddings":
        """No-vector fallback: every lookup returns the zero default."""
        return cls(dim=dim)

    @classmethod
    def load(cls, path: str | None = None) -> "CardEmbeddings":
        with open(path or DEFAULT_EMBEDDINGS_PATH) as f:
            data = json.load(f)
        return cls(dim=data["dim"], vectors=data.get("cards", {}),
                   default=data.get("default"))

    def vector(self, card_id) -> list:
        """Embedding for a card ID; the default vector when unknown/None."""
        if card_id is None:
            return self.default
        return self._vectors.get(int(card_id), self.default)

    def mean(self, card_ids) -> list:
        """Mean embedding of a card-ID collection; zeros when empty."""
        total = [0.0] * self.dim
        n = 0
        for card_id in card_ids:
            vec = self.vector(card_id)
            for j in range(self.dim):
                total[j] += vec[j]
            n += 1
        if n == 0:
            return total
        return [v / n for v in total]

    def __len__(self) -> int:
        return len(self._vectors)


_shared_embeddings = None


def shared_embeddings() -> CardEmbeddings:
    """Process-wide CardEmbeddings from train/card_embeddings.json (lazy).

    Falls back to an empty (all-zero) table when the file is absent or
    unreadable, so feature extraction degrades instead of crashing.
    """
    global _shared_embeddings
    if _shared_embeddings is None:
        try:
            _shared_embeddings = CardEmbeddings.load()
        except (OSError, ValueError, KeyError):
            _shared_embeddings = CardEmbeddings.empty()
    return _shared_embeddings
