"""Shared helpers for the SOT-1671 test suite (synthetic observations,
synthetic card master). Engine-independent."""
from types import SimpleNamespace

from agents.cards import CardIndex


def card(card_id, player_index=0, serial=None):
    return {"id": card_id, "serial": serial or card_id * 10,
            "playerIndex": player_index}


def pokemon(card_id, hp=100, max_hp=100, energies=(), player_index=0):
    return {"id": card_id, "serial": card_id * 10, "hp": hp, "maxHp": max_hp,
            "appearThisTurn": False, "energies": list(energies),
            "energyCards": [], "tools": [], "preEvolution": []}


def player(active=(), bench=(), hand=None, deck_count=40, hand_count=5,
           prize=6, discard=()):
    return {
        "active": list(active), "bench": list(bench), "benchMax": 5,
        "deckCount": deck_count, "discard": list(discard),
        "prize": [None] * prize, "handCount": hand_count, "hand": hand,
        "poisoned": False, "burned": False, "asleep": False,
        "paralyzed": False, "confused": False,
    }


def observation(select, me=None, opp=None, your_index=0, **state):
    players = [me or player(), opp or player(hand=None)]
    if your_index == 1:
        players.reverse()
    current = {
        "turn": 3, "turnActionCount": 0, "yourIndex": your_index,
        "firstPlayer": 0, "supporterPlayed": False, "stadiumPlayed": False,
        "energyAttached": False, "retreated": False, "result": -1,
        "stadium": [], "looking": None, "players": players,
    }
    current.update(state)
    return {"select": select, "logs": [], "current": current,
            "search_begin_input": ""}


def select(options, sel_type=0, context=0, min_count=1, max_count=1,
           deck=None, **extra):
    sel = {
        "type": sel_type, "context": context,
        "minCount": min_count, "maxCount": max_count,
        "remainDamageCounter": 0, "remainEnergyCost": 0,
        "option": list(options), "deck": deck,
        "contextCard": None, "effect": None,
    }
    sel.update(extra)
    return sel


# Synthetic card master (test-only IDs; never used by agents/ code).
# 101: Basic Water Pokémon, attack 201 (damage 50)
# 102: Basic Fire Pokémon (weak to Water), attack 202 (damage 30)
# 103: Supporter
def synthetic_card_index() -> CardIndex:
    cards = [
        SimpleNamespace(cardId=101, cardType=0, retreatCost=1, hp=120,
                        weakness=None, resistance=None, energyType=3,
                        basic=True, stage1=False, stage2=False, ex=False,
                        megaEx=False, tera=False, aceSpec=False,
                        evolvesFrom=None, skills=[], attacks=[201]),
        SimpleNamespace(cardId=102, cardType=0, retreatCost=2, hp=60,
                        weakness=3, resistance=None, energyType=2,
                        basic=True, stage1=False, stage2=False, ex=True,
                        megaEx=False, tera=False, aceSpec=False,
                        evolvesFrom=None, skills=[], attacks=[202]),
        SimpleNamespace(cardId=103, cardType=3, retreatCost=0, hp=0,
                        weakness=None, resistance=None, energyType=0,
                        basic=False, stage1=False, stage2=False, ex=False,
                        megaEx=False, tera=False, aceSpec=False,
                        evolvesFrom=None, skills=[], attacks=[]),
    ]
    attacks = [
        SimpleNamespace(attackId=201, damage=50, energies=[3]),
        SimpleNamespace(attackId=202, damage=30, energies=[2, 0]),
    ]
    return CardIndex(cards, attacks)
