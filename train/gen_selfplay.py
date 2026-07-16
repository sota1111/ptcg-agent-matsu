"""Self-play training-log generator (SOT-1674; cheater mode SOT-1678).

Plays N seeded matches on the local cabt engine and logs training examples
labelled with the final match result.

Two modes:

- Default (SOT-1674): Greedy/Random mix per --mix; one example per engine
  turn from the ACTING player's perspective (fields m, t, y, h, x).
- --cheater (SOT-1678, arXiv:1808.04794 §III-B1/§IV-C): strong MCTS vs
  strong MCTS where each decision's determinization is the TRUE hidden
  state (train/cheater.py; cg.game.visualize_data is the source — local
  generation only, never available to a submitted agent). Engine turns are
  sampled with probability --sample-p (paper: p, default 0.5) and each
  sampled state yields TWO examples — one per player perspective ("両者分"):
  fields m, t, who (perspective player), y (1 if that player won), h, x.
  The perspective vectors are computed from the acting player's observation
  exactly as the value net sees them at inference time inside MCTS
  (featurize(obs, root) with root == actor and root != actor), so the
  training and serving feature distributions match; the true state is used
  ONLY to determinize the players' search, never to build features.

Agent-side randomness and state sampling are seeded (--seed); the engine's
internal RNG is not injectable (ASSUMPTIONS.md A-9), so logs vary between
runs — reproducibility here means the generation PROCEDURE (mode, seeds,
sampling rule) is fixed, matching the eval/bench.py convention. With engine
responses held fixed the generator is a deterministic function of the seed
(tests/test_cheater.py pins this on a scripted engine double).

Usage (from the repo root; shard by varying --seed and --out):
    venv/bin/python train/gen_selfplay.py --n 4000 --seed 61674001 \
        --out train/logs/shard_1.jsonl [--mix gg:0.6,gr:0.2,rg:0.2]
    venv/bin/python train/gen_selfplay.py --cheater --n 200 --seed 61678001 \
        --sample-p 0.5 --out train/logs/cheater_1.jsonl \
        [--mcts-config '{"time_budget_s": 0.1}']

Output JSONL keeps the compact training fields and adds a versioned analysis
contract: actor, legal actions, chosen action, reward/outcome, winner and
termination reason.  Cheater mode adds who=perspective player index.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from agents import make_agent
from agents.cards import shared_index
from agents.evaluator import HeuristicEvaluator
from agents.features import make_featurizer
from agents.rng import Rng

MAX_DECISIONS = 100000  # engine draws long before this (BattleData.h:66-74)

MIX_AGENTS = {"gg": ("greedy", "greedy"), "gr": ("greedy", "random"),
              "rg": ("random", "greedy"), "rr": ("random", "random")}

# Strong-MCTS defaults for cheater self-play: the SOT-1673 champion config
# (docs/ablation.md 最良構成), except n_worlds=1 — with exact true fills
# there is no hidden-information distribution to average over — and a
# smaller per-move budget than the paper's 1 s cheater so a local run
# finishes in minutes, not hours (override via --mcts-config; the measured
# cost per budget is recorded in docs/selfplay-cheater.md).
CHEATER_MCTS_DEFAULTS = {
    "max_root_actions": 6, "max_tree_depth": 1, "rollout_turns": 100,
    "rollout_depth": 200, "n_worlds": 1, "time_budget_s": 0.1,
    "deviate_margin": 0.1,
}


def _load_engine():
    """Import the local engine bindings (repo-root cwd for libcg/deck.csv)."""
    os.chdir(REPO)
    from cg import game
    return game


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


def play_and_log(game, agent0, agent1, cards, heuristic,
                 featurize=make_featurizer("v1")[1]):
    """One match; returns (result, records) with one record per engine turn."""
    obs, start = game.battle_start(agent0._deck, agent1._deck)
    if obs is None:
        raise RuntimeError(
            f"battle_start failed: errorPlayer={start.errorPlayer} "
            f"errorType={start.errorType}")
    records = []
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
                    records.append({"root": actor, "turn": turn, "h": h, "x": x})
            agent = agent0 if actor == 0 else agent1
            action = agent.act(obs)
            if records and records[-1]["turn"] == turn and "action" not in records[-1]:
                records[-1].update({
                    "actor": actor,
                    "legal_actions": (obs.get("select") or {}).get("option", []),
                    "action": action,
                    "turn_action_count": current.get("turnActionCount"),
                })
            obs = game.battle_select(action)
            decisions += 1
        return -1, records
    finally:
        game.battle_finish()


def play_and_log_cheater(game, agent0, agent1, cards, heuristic, featurize,
                         sample_rng, sample_p):
    """One cheater self-play match.

    Before every decision the acting agent receives the true-state fills
    (train/cheater.py) parsed from game.visualize_data(). Each engine turn's
    first decision state is sampled with probability `sample_p`; a sampled
    state yields one (who, turn, h, x) record per player perspective.
    """
    from train.cheater import parse_true_state, true_fills

    obs, start = game.battle_start(agent0._deck, agent1._deck)
    if obs is None:
        raise RuntimeError(
            f"battle_start failed: errorPlayer={start.errorPlayer} "
            f"errorType={start.errorType}")
    records = []  # (perspective_player, turn, heuristic_value, features)
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
            if turn != last_turn:
                last_turn = turn
                if turn > 0 and sample_rng.random() < sample_p:
                    ns = _to_namespace({"current": current})
                    for who in (0, 1):
                        x = featurize(obs, who, cards)
                        h = heuristic.evaluate(ns, who)
                        records.append({"who": who, "turn": turn, "h": h, "x": x})
            agent = agent0 if actor == 0 else agent1
            state = parse_true_state(game.visualize_data())
            agent.set_true_fills(true_fills(current, state))
            action = agent.act(obs)
            for record in records[-2:]:
                if record["turn"] == turn and "action" not in record:
                    record.update({
                        "actor": actor,
                        "legal_actions": (obs.get("select") or {}).get("option", []),
                        "action": action,
                        "turn_action_count": current.get("turnActionCount"),
                    })
            obs = game.battle_select(action)
            decisions += 1
        return -1, records
    finally:
        game.battle_finish()


def generate_cheater(game, agent_factory, n, base, cards, heuristic,
                     featurize, sample_p, out_path) -> dict:
    """Run N cheater matches, write the JSONL, return the stats dict.

    `agent_factory(seed)` -> an agent exposing act()/set_true_fills() (the
    real one is train.cheater.CheaterMctsAgent; tests inject doubles). A
    match that raises is counted under "errors" and generation continues —
    one exotic engine state must not void a long run.
    """
    stats = {"matches": 0, "examples": 0, "draws": 0, "unfinished": 0,
             "errors": 0, "degraded": 0, "fallbacks": 0}
    with open(out_path, "w") as out:
        for i in range(n):
            agent0 = agent_factory(base.child(f"match{i}.a").seed)
            agent1 = agent_factory(base.child(f"match{i}.b").seed)
            sample_rng = base.child(f"match{i}.sample")
            try:
                result, records = play_and_log_cheater(
                    game, agent0, agent1, cards, heuristic, featurize,
                    sample_rng, sample_p)
            except Exception as exc:
                stats["errors"] += 1
                print(f"  match {i} failed: {type(exc).__name__}: {exc}",
                      flush=True)
                continue
            finally:
                for agent in (agent0, agent1):
                    stats["degraded"] += getattr(agent, "degraded_count", 0)
                    stats["fallbacks"] += getattr(agent, "planner_fallbacks", 0)
            stats["matches"] += 1
            if result not in (0, 1):  # draw/unfinished: no win/loss label
                stats["draws" if result == 2 else "unfinished"] += 1
                continue
            for record in records:
                who, turn, h, x = (record["who"], record["turn"],
                                   record["h"], record["x"])
                out.write(json.dumps(
                    {"schema": "matsu-battle-log-v2", "m": i, "t": turn,
                     "who": who, "actor": record.get("actor"),
                     "legal_actions": record.get("legal_actions", []),
                     "action": record.get("action", []),
                     "turn_action_count": record.get("turn_action_count"),
                     "y": float(result == who),
                     "reward": 1.0 if result == who else -1.0,
                     "outcome": "win" if result == who else "loss",
                     "winner": result, "termination_reason": "engine_result",
                     "h": round(h, 6), "x": [round(v, 6) for v in x]},
                    separators=(",", ":")) + "\n")
                stats["examples"] += 1
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{n} matches, "
                      f"{stats['examples']} examples", flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=61674001)
    parser.add_argument("--deck", default="deck.csv")
    parser.add_argument("--mix", default="gg:0.6,gr:0.2,rg:0.2")
    parser.add_argument("--cheater", action="store_true",
                        help="strong-MCTS self-play determinized with the "
                             "TRUE hidden state (SOT-1678); generation-only")
    parser.add_argument("--sample-p", type=float, default=0.5,
                        help="cheater mode: per-turn state sampling "
                             "probability (paper's p, default 0.5)")
    parser.add_argument("--mcts-config", default=None,
                        help="cheater mode: JSON PlannerConfig overrides "
                             f"merged onto {CHEATER_MCTS_DEFAULTS}")
    parser.add_argument("--features", default=None, choices=("v1", "v2"),
                        help="feature extractor: v1 = 32 scalars (SOT-1674), "
                             "v2 = embedding-extended vector (SOT-1676). "
                             "Default: v2 in cheater mode, else v1")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    game = _load_engine()
    deck = load_deck(args.deck)
    cards = shared_index()
    feature_set = args.features or ("v2" if args.cheater else "v1")
    _, featurize = make_featurizer(feature_set)
    heuristic = HeuristicEvaluator()
    base = Rng(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.cheater:
        from train.cheater import CheaterMctsAgent

        mcts_config = dict(CHEATER_MCTS_DEFAULTS)
        mcts_config.update(json.loads(args.mcts_config)
                           if args.mcts_config else {})

        def agent_factory(seed):
            return CheaterMctsAgent(seed=seed, deck=deck, card_index=cards,
                                    **mcts_config)

        stats = generate_cheater(game, agent_factory, args.n, base, cards,
                                 heuristic, featurize, args.sample_p,
                                 args.out)
        print(f"DONE cheater seed={args.seed} sample_p={args.sample_p} "
              f"features={feature_set} mcts={mcts_config} {stats} "
              f"-> {args.out}", flush=True)
        return

    mix = parse_mix(args.mix)
    stats = {"matches": 0, "examples": 0, "draws": 0, "unfinished": 0}
    with open(args.out, "w") as out:
        for i in range(args.n):
            pick = base.child(f"match{i}.mix").random()
            kind = next(name for name, edge in mix if pick <= edge)
            name0, name1 = MIX_AGENTS[kind]
            agent0 = make_agent(name0, seed=base.child(f"match{i}.a").seed, deck=deck)
            agent1 = make_agent(name1, seed=base.child(f"match{i}.b").seed, deck=deck)
            result, records = play_and_log(game, agent0, agent1, cards,
                                           heuristic, featurize)
            stats["matches"] += 1
            if result not in (0, 1):  # draw/unfinished: no win/loss label
                stats["draws" if result == 2 else "unfinished"] += 1
                continue
            for record in records:
                root, turn, h, x = (record["root"], record["turn"],
                                    record["h"], record["x"])
                out.write(json.dumps(
                    {"schema": "matsu-battle-log-v2", "m": i, "t": turn,
                     "actor": record.get("actor", root),
                     "legal_actions": record.get("legal_actions", []),
                     "action": record.get("action", []),
                     "turn_action_count": record.get("turn_action_count"),
                     "y": float(result == root),
                     "reward": 1.0 if result == root else -1.0,
                     "outcome": "win" if result == root else "loss",
                     "winner": result, "termination_reason": "engine_result",
                     "h": round(h, 6), "x": [round(v, 6) for v in x]},
                    separators=(",", ":")) + "\n")
                stats["examples"] += 1
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{args.n} matches, "
                      f"{stats['examples']} examples", flush=True)

    print(f"DONE seed={args.seed} mix={args.mix} features={feature_set} "
          f"{stats} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
