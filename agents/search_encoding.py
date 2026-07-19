"""Deterministic model inputs for cabt Search API observations and moves.

The engine's option order is retained as the action identity.  A policy may
score the returned rows, then :meth:`EncodedActions.decode` converts those
scores back to the exact indices expected by ``search_step``.
"""
from dataclasses import dataclass
import math

from .features import FEATURE_NAMES_V2, featurize_v2


OPTION_TYPE_COUNT = 17
_OPTION_SCALARS = (
    "number", "playerIndex", "area", "index", "inPlayArea",
    "attackId", "skillId", "energy", "specialCondition",
)
ACTION_FEATURE_NAMES = tuple(
    [f"option_type_{i}" for i in range(OPTION_TYPE_COUNT)]
    + ["option_type_unknown", "select_type", "select_context",
       "min_count", "max_count", "remain_damage_counter",
       "remain_energy_cost"]
    + [f"option_{name}" for name in _OPTION_SCALARS]
)


def _get(value, name, default=None):
    if isinstance(value, dict):
        result = value.get(name, default)
    else:
        result = getattr(value, name, default)
    return default if result is None else result


def _number(value, default=0.0) -> float:
    """Finite numeric conversion; unknown enum/object values become zero."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def encode_state(observation, root_player: int, card_index=None,
                 embeddings=None) -> tuple:
    """Encode cards, Pokémon, players and public state to a fixed vector.

    This is the shared v2 feature contract used by training and inference.
    Empty zones are zero padded and unknown cards use neutral card attributes
    and the default embedding supplied by ``features.featurize_v2``.
    """
    return tuple(featurize_v2(observation, root_player, card_index, embeddings))


@dataclass(frozen=True)
class EncodedActions:
    """Legal option rows plus the selection-count contract."""
    features: tuple
    option_indices: tuple
    min_count: int
    max_count: int

    @property
    def mask(self) -> tuple:
        return tuple(True for _ in self.option_indices)

    def decode(self, scores) -> list:
        """Map model scores back to a deterministic, engine-legal selection.

        The highest finite scores are selected.  Ties retain Search API option
        order; the returned indices are sorted because selection order has no
        meaning in the engine contract.
        """
        values = list(scores)
        if len(values) != len(self.option_indices):
            raise ValueError("one model score is required per legal option")
        ranked = sorted(
            range(len(values)),
            key=lambda i: (-_number(values[i], float("-inf")), i),
        )
        count = self.max_count
        chosen = sorted(self.option_indices[i] for i in ranked[:count])
        if not self.min_count <= len(chosen) <= self.max_count:
            raise ValueError("decoded selection violates count bounds")
        return chosen


def encode_actions(select) -> EncodedActions:
    """Encode ``SelectContext`` and every engine-provided legal option."""
    options = list(_get(select, "option", ()) or ())
    minimum = max(0, int(_number(_get(select, "minCount", 0))))
    maximum = max(0, int(_number(_get(select, "maxCount", 0))))
    maximum = min(maximum, len(options))
    minimum = min(minimum, maximum)
    shared = (
        _number(_get(select, "type", -1), -1.0),
        _number(_get(select, "context", -1), -1.0),
        float(minimum), float(maximum),
        _number(_get(select, "remainDamageCounter", 0)),
        _number(_get(select, "remainEnergyCost", 0)),
    )
    rows = []
    for option in options:
        option_type = int(_number(_get(option, "type", -1), -1.0))
        one_hot = [0.0] * OPTION_TYPE_COUNT
        known = 0 <= option_type < OPTION_TYPE_COUNT
        if known:
            one_hot[option_type] = 1.0
        rows.append(tuple(
            one_hot + [float(not known)] + list(shared)
            + [_number(_get(option, name, 0)) for name in _OPTION_SCALARS]
        ))
    return EncodedActions(tuple(rows), tuple(range(len(options))),
                          minimum, maximum)


class SearchSession:
    """Exception-safe lifetime wrapper for an ``EngineBackend``-like object."""

    def __init__(self, backend):
        self.backend = backend

    def __enter__(self):
        return self

    def begin(self, raw_obs, fills, manual_coin=True):
        return self.backend.begin(raw_obs, fills, manual_coin=manual_coin)

    def step(self, search_id, action):
        return self.backend.step(search_id, action)

    def release(self, search_id):
        return self.backend.release(search_id)

    def __exit__(self, exc_type, exc_value, traceback):
        self.backend.end()
        return False


assert len(ACTION_FEATURE_NAMES) == OPTION_TYPE_COUNT + 1 + 6 + len(_OPTION_SCALARS)
assert len(FEATURE_NAMES_V2) > 0
