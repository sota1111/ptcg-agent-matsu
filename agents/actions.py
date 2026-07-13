"""Action Enumerator (SOT-1671) — layer [2] of the 4-layer architecture.

The engine's legal-move API (`obs.select`) is the SINGLE SOURCE OF TRUTH:
an action is a list of indices into `select.option` with
`minCount <= len <= maxCount` and no duplicates (main.py:22-30,
docs/engine-facts.md §1.1). This module never invents moves — it only
enumerates, masks, validates and samples what the engine offers.
"""
from .observation import SelectView
from .rng import Rng


class IllegalActionError(ValueError):
    """An action that violates the engine's selection contract."""


def legal_indices(select: SelectView) -> list:
    """All individually selectable option indices."""
    return list(range(len(select.options)))


def legality_mask(select: SelectView) -> list:
    """Boolean mask over option indices (True = selectable).

    Every option the engine lists is legal as an element; the mask exists as
    the stable interface for upper layers (planner/policy, SOT-1672+).
    """
    return [True] * len(select.options)


def count_bounds(select: SelectView) -> tuple:
    """Clamped (lo, hi) bounds for the number of selected indices.

    The engine promises maxCount <= len(option) (cg/api.py:403); clamp anyway
    so a malformed/unknown observation can never make us emit an illegal
    length.
    """
    n = len(select.options)
    hi = min(max(select.max_count, 0), n)
    lo = min(max(select.min_count, 0), hi)
    return lo, hi


def validate(select: SelectView, action) -> list:
    """Check an action against the selection contract; return it if legal."""
    if not isinstance(action, list) or not all(isinstance(i, int) for i in action):
        raise IllegalActionError(f"action must be list[int], got {action!r}")
    n = len(select.options)
    lo, hi = count_bounds(select)
    if not (lo <= len(action) <= hi):
        raise IllegalActionError(
            f"len(action)={len(action)} outside [{lo}, {hi}] (n={n})")
    if any(i < 0 or i >= n for i in action):
        raise IllegalActionError(f"action index out of range 0..{n - 1}: {action}")
    if len(set(action)) != len(action):
        raise IllegalActionError(f"duplicate indices in action: {action}")
    return action


def random_action(select: SelectView, rng: Rng) -> list:
    """Uniform legal action: uniform count in [lo, hi], then uniform subset."""
    lo, hi = count_bounds(select)
    k = rng.randint(lo, hi)
    return validate(select, sorted(rng.sample(legal_indices(select), k)))
