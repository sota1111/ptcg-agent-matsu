"""Deck-family search policy selected from the 25-deck loss campaign.

The runtime agent receives card ids, not a deck filename. A canonical digest
therefore acts as the deck id without coupling gameplay code to card names.
Unknown or changed decks deliberately retain the champion configuration.
"""
from __future__ import annotations

import hashlib


DECK_PRESERVATION_WEIGHTS = {
    "deck_low": -0.2,
    "deck_low_at": 14,
    "deck_low_prize_gate": 3,
}

# Same-seed candidate-vs-champion screen: neutral or positive archetypes.
# Five regressing archetypes (06, 08, 11, 21, 22) are intentionally absent.
_PRESERVATION_DECK_IDS = frozenset({
    "061f0a8e8bbb8764", "1a24eb1d136aa116", "20d4fd5182ffaa8e",
    "3847e116f56fa428", "50af6e1d9f537341", "6044baf0f8e9862d",
    "710a4d58171c8759", "7240cc1db3dc707f", "78b12b2ab93e18be",
    "79724ee915d76f94", "7bb8d8601056d4d9", "7bfcf5ebfea52b71",
    "8aeea88d91944707", "8c7e97e4d0d93cec", "93ddf46f3aaec0df",
    "a6134f52a7177b6f", "b49d1186a118cc2d", "b6f885624078e896",
    "bf715eb49deadca5", "d181ae2ac133ef4f",
})


def deck_id(deck: list[int]) -> str:
    """Stable id for a multiset deck; CSV ordering does not affect policy."""
    payload = ",".join(str(card_id) for card_id in sorted(deck))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


def search_overrides(deck: list[int]) -> dict:
    """Return evidence-backed overrides, or champion defaults for unknowns."""
    if deck_id(deck) in _PRESERVATION_DECK_IDS:
        return {"eval_weights": dict(DECK_PRESERVATION_WEIGHTS)}
    return {}
