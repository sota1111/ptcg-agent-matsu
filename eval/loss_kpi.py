"""Engine-independent terminal loss classification for A/B artifacts."""


def terminal_loss_cause(current, loser: int) -> str:
    """Classify a completed loss from the terminal board.

    ``board_wipe`` means the losing side has neither an Active nor a Bench
    Pokémon remaining. Other terminal paths (prizes, deck-out, effects) stay
    grouped as ``other``; this deliberately avoids guessing from card names.
    """
    players = (current or {}).get("players") or ()
    if loser not in (0, 1) or len(players) <= loser:
        return "other"
    side = players[loser] or {}
    active = side.get("active") or ()
    bench = side.get("bench") or ()
    return "board_wipe" if not any(active) and not any(bench) else "other"
