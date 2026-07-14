"""Self-play training-log generator (SOT-1674).

Plays N seeded matches on the local cabt engine and logs one training
example per engine turn: the agents/features.py vector of the state at the
turn's first decision (from the acting player's perspective) plus the final
match result as the label. The heuristic evaluator's prediction for the same
state is logged alongside, so training can report learned-vs-heuristic
predictive quality on identical holdout states.

Agent-side randomness is seeded (--seed); the engine's internal RNG is not
injectable (ASSUMPTIONS.md A-9), so logs vary between runs — reproducibility
here means the generation PROCEDURE (mix, seeds, sampling rule) is fixed,
matching the eval/bench.py convention.

Match mix: mostly Greedy self-play (the MCTS rollout policy is greedy, so
these states match the distribution the evaluator sees at rollout leaves)
plus Greedy-vs-Random games for lopsided-board coverage.

Usage (from the repo root; shard by varying --seed and --out):
    venv/bin/python train/gen_selfplay.py --n 4000 --seed 61674001 \
        --out train/logs/shard_1.jsonl [--mix gg:0.6,gr:0.2,rg:0.2]

Output JSONL fields: m=match index, t=turn, y=label (1 root won / 0 lost),
h=HeuristicEvaluator prediction, x=feature vector.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)  # libcg.so & deck.csv resolve relative to the repo root

from cg import game
from agents import make_agent
from agents.cards import shared_index
from agents.evaluator import HeuristicEvaluator
from agents.features import make_featurizer
from agents.rng import Rng

MAX_DECISIONS = 100000  # engine draws long before this (BattleData.h:66-74)

MIX_AGENTS = {"gg": ("greedy", "greedy"), "gr": ("greedy", "random"),
              "rg": ("random", "greedy"), "rr": ("random", "random")}


def load_deck(path: str) -> list:
    with open(path) as f:
        return [int(x) for x in f.read().split("\n")[:60]]


def _to_namespace(value):
    """Raw-dict subtree -> attribute objects (HeuristicEvaluator's shape)."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def parse_mix(spec: str) -> list:
    """"gg:0.6,gr:0.2,rg:0.2" -> [("gg", 0.6), ("gr", 0.8), ("rg", 1.0)]."""
    pairs = []
    total = 0.0
    for part in spec.split(","):
        name, _, weight = part.partition(":")
        if name not in MIX_AGENTS:
            raise SystemExit(f"unknown mix entry {name!r} (use {sorted(MIX_AGENTS)})")
        total += float(weight or 1.0)
        pairs.append((name, total))
    return [(name, edge / total) for name, edge in pairs]


def play_and_log(agent0, agent1, cards, heuristic, match_id: int,
                 featurize=make_featurizer("v1")[1]):
    """One match; returns (result, records) with one record per engine turn."""
    obs, start = game.battle_start(agent0._deck, agent1._deck)
    if obs is None:
        raise RuntimeError(
            f"battle_start failed: errorPlayer={start.errorPlayer} "
            f"errorType={start.errorType}")
    records = []  # (root_player, turn, heuristic_value, features)
    try:
        decisions = 0
        last_turn = -1
        while decisions < MAX_DECISIONS:
            current = obs.get("current") or {}
            result = current.get("result", -1)
            if result is None:
                result = -1
            if result != -1:
                return result, records
            actor = current.get("yourIndex", 0)
            turn = current.get("turn", 0) or 0
            # One record per engine turn; turn 0 is the setup phase (prizes
            # not dealt yet) and never a rollout leaf, so it is skipped.
            if turn != last_turn:
                last_turn = turn
                if turn > 0:
                    x = featurize(obs, actor, cards)
                    h = heuristic.evaluate(
                        _to_namespace({"current": current}), actor)
                    records.append((actor, turn, h, x))
            agent = agent0 if actor == 0 else agent1
            obs = game.battle_select(agent.act(obs))
            decisions += 1
        return -1, records
    finally:
        game.battle_finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=61674001)
    parser.add_argument("--deck", default="deck.csv")
    parser.add_argument("--mix", default="gg:0.6,gr:0.2,rg:0.2")
    parser.add_argument("--features", default="v1", choices=("v1", "v2"),
                        help="feature extractor: v1 = 32 scalars (SOT-1674), "
                             "v2 = embedding-extended vector (SOT-1676)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    deck = load_deck(args.deck)
    mix = parse_mix(args.mix)
    cards = shared_index()
    _, featurize = make_featurizer(args.features)
    heuristic = HeuristicEvaluator()
    base = Rng(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    stats = {"matches": 0, "examples": 0, "draws": 0, "unfinished": 0}
    with open(args.out, "w") as out:
        for i in range(args.n):
            pick = base.child(f"match{i}.mix").random()
            kind = next(name for name, edge in mix if pick <= edge)
            name0, name1 = MIX_AGENTS[kind]
            agent0 = make_agent(name0, seed=base.child(f"match{i}.a").seed, deck=deck)
            agent1 = make_agent(name1, seed=base.child(f"match{i}.b").seed, deck=deck)
            result, records = play_and_log(agent0, agent1, cards, heuristic,
                                           i, featurize)
            stats["matches"] += 1
            if result not in (0, 1):  # draw/unfinished: no win/loss label
                stats["draws" if result == 2 else "unfinished"] += 1
                continue
            for root, turn, h, x in records:
                out.write(json.dumps(
                    {"m": i, "t": turn, "y": float(result == root),
                     "h": round(h, 6), "x": [round(v, 6) for v in x]},
                    separators=(",", ":")) + "\n")
                stats["examples"] += 1
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{args.n} matches, "
                      f"{stats['examples']} examples", flush=True)

    print(f"DONE seed={args.seed} mix={args.mix} features={args.features} "
          f"{stats} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
