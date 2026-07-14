"""Baseline benchmark runner (SOT-1671).

Plays N side-alternating matches between two agents on the local cabt engine
and reports:
- win/loss/draw for agent A, win rate with a Wilson 95% CI (draws excluded,
  and also reported with draws counted as 0.5),
- engine rejects (illegal actions) and agent exceptions — both must be 0,
- per-match and per-decision wall-clock timing,
- agent fallback counts (BaseAgent degradations, expected 0 on the known pool),
- planner counters when an agent exposes them (SOT-1672 MctsAgent):
  budget violations (時間切れ, must be 0), planner fallbacks, degraded
  decisions, and the per-decision planner time max.

The engine's internal RNG is not externally seedable (ASSUMPTIONS.md A-9), so
matches vary between runs; agent seeds are derived per match from --seed for
agent-side reproducibility.

Usage (from the repo root):
    venv/bin/python eval/bench.py --agent-a greedy --agent-b random \
        --n 1000 --seed 20260713 [--deck deck.csv] [--json out.json]

Agent constructor kwargs (e.g. SOT-1673 ablation points of the MCTS
planner) can be passed as JSON: --config-a '{"n_worlds": 1, "uct_c": 0.7}'.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)  # libcg.so & deck.csv resolve relative to the repo root

from cg import game
from agents import make_agent
from agents.rng import Rng

MAX_DECISIONS = 100000  # engine draws long before this (BattleData.h:66-74)


def load_deck(path: str) -> list:
    with open(path) as f:
        return [int(x) for x in f.read().split("\n")[:60]]


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score 95% interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def play_match(agent0, agent1):
    """One engine match. Returns (result, decisions, reject, exception)."""
    obs, start = game.battle_start(agent0._deck, agent1._deck)
    if obs is None:
        raise RuntimeError(
            f"battle_start failed: errorPlayer={start.errorPlayer} "
            f"errorType={start.errorType}")
    try:
        decisions = 0
        while decisions < MAX_DECISIONS:
            current = obs.get("current") or {}
            result = current.get("result", -1)
            if result != -1:
                return result, decisions, False, False
            agent = agent0 if current.get("yourIndex", 0) == 0 else agent1
            try:
                action = agent.act(obs)
            except Exception:
                return -1, decisions, False, True   # agent exception
            try:
                obs = game.battle_select(action)
            except Exception:
                return -1, decisions, True, False   # engine reject
            decisions += 1
        return -1, decisions, False, False          # should be unreachable
    finally:
        game.battle_finish()


# Planner counters picked up from agents that expose them (MctsAgent).
PLANNER_COUNTERS = ("budget_violations", "planner_fallbacks", "degraded_count",
                    "rollout_cutoffs")


def run_bench(agent_a: str, agent_b: str, n: int, seed: int, deck_path: str,
              config_a=None, config_b=None):
    deck = load_deck(deck_path)
    base = Rng(seed)
    stats = {
        "wins_a": 0, "wins_b": 0, "draws": 0, "unfinished": 0,
        "rejects": 0, "exceptions": 0, "fallbacks_a": 0, "fallbacks_b": 0,
        "decisions": 0,
    }
    for counter in PLANNER_COUNTERS:
        stats[f"{counter}_a"] = 0
        stats[f"{counter}_b"] = 0
    match_times = []
    per_decision = []
    planner_move_max_s = 0.0
    for i in range(n):
        seed_a = base.child(f"match{i}.a").seed
        seed_b = base.child(f"match{i}.b").seed
        a = make_agent(agent_a, seed=seed_a, deck=deck, **(config_a or {}))
        b = make_agent(agent_b, seed=seed_b, deck=deck, **(config_b or {}))
        a_plays_first = (i % 2 == 0)  # alternate sides every match
        p0, p1 = (a, b) if a_plays_first else (b, a)
        t0 = time.perf_counter()
        result, decisions, reject, exception = play_match(p0, p1)
        dt = time.perf_counter() - t0
        match_times.append(dt)
        stats["decisions"] += decisions
        if decisions:
            per_decision.append(dt / decisions)
        stats["rejects"] += int(reject)
        stats["exceptions"] += int(exception)
        stats["fallbacks_a"] += a.fallback_count
        stats["fallbacks_b"] += b.fallback_count
        for agent, suffix in ((a, "a"), (b, "b")):
            for counter in PLANNER_COUNTERS:
                stats[f"{counter}_{suffix}"] += getattr(agent, counter, 0)
            move_times = getattr(agent, "move_times", None)
            if move_times:
                planner_move_max_s = max(planner_move_max_s, max(move_times))
        if result in (0, 1):
            a_won = (result == 0) == a_plays_first
            stats["wins_a" if a_won else "wins_b"] += 1
        elif result == 2:
            stats["draws"] += 1
        else:
            stats["unfinished"] += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n} matches "
                  f"(A {stats['wins_a']} / B {stats['wins_b']} "
                  f"/ draw {stats['draws']})", flush=True)

    decided = stats["wins_a"] + stats["wins_b"]
    ci_lo, ci_hi = wilson_ci(stats["wins_a"], decided)
    report = {
        "agent_a": agent_a, "agent_b": agent_b, "n_matches": n, "seed": seed,
        "deck": deck_path,
        "config_a": config_a or {}, "config_b": config_b or {},
        **stats,
        "planner_move_max_ms": planner_move_max_s * 1000,
        "winrate_a_excl_draws": (stats["wins_a"] / decided) if decided else None,
        "wilson95_excl_draws": [ci_lo, ci_hi],
        "winrate_a_draws_half": (stats["wins_a"] + 0.5 * stats["draws"]) / n,
        "time_per_match_sec": {
            "mean": statistics.fmean(match_times),
            "median": statistics.median(match_times),
            "max": max(match_times),
            "total": sum(match_times),
        },
        "time_per_decision_ms": {
            "mean": statistics.fmean(per_decision) * 1000,
            "max": max(per_decision) * 1000,
        },
        "decisions_per_match_mean": stats["decisions"] / n,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-a", default="greedy")
    parser.add_argument("--agent-b", default="random")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--deck", default="deck.csv")
    parser.add_argument("--json", default=None,
                        help="write the full report to this JSON file")
    parser.add_argument("--config-a", default=None,
                        help="JSON kwargs for agent A's constructor")
    parser.add_argument("--config-b", default=None,
                        help="JSON kwargs for agent B's constructor")
    args = parser.parse_args()

    config_a = json.loads(args.config_a) if args.config_a else None
    config_b = json.loads(args.config_b) if args.config_b else None
    print(f"BENCH: {args.agent_a} (A) vs {args.agent_b} (B), "
          f"n={args.n}, seed={args.seed} "
          f"config_a={config_a} config_b={config_b}", flush=True)
    report = run_bench(args.agent_a, args.agent_b, args.n, args.seed,
                       args.deck, config_a=config_a, config_b=config_b)

    tpm = report["time_per_match_sec"]
    tpd = report["time_per_decision_ms"]
    print(f"""
RESULT: {report['agent_a']} vs {report['agent_b']} (n={report['n_matches']})
  A wins {report['wins_a']}  B wins {report['wins_b']}  draws {report['draws']}  unfinished {report['unfinished']}
  win rate A (excl. draws): {report['winrate_a_excl_draws']:.4f}  Wilson95 [{report['wilson95_excl_draws'][0]:.4f}, {report['wilson95_excl_draws'][1]:.4f}]
  win rate A (draws=0.5)  : {report['winrate_a_draws_half']:.4f}
  engine rejects: {report['rejects']}  agent exceptions: {report['exceptions']}  fallbacks: A={report['fallbacks_a']} B={report['fallbacks_b']}
  budget violations: A={report['budget_violations_a']} B={report['budget_violations_b']}  planner fallbacks: A={report['planner_fallbacks_a']} B={report['planner_fallbacks_b']}  degraded: A={report['degraded_count_a']} B={report['degraded_count_b']}  planner move max: {report['planner_move_max_ms']:.1f} ms
  rollout cutoffs (early cutoff): A={report['rollout_cutoffs_a']} B={report['rollout_cutoffs_b']}
  time/match: mean {tpm['mean'] * 1000:.2f} ms  median {tpm['median'] * 1000:.2f} ms  max {tpm['max'] * 1000:.2f} ms  total {tpm['total']:.1f} s
  time/decision: mean {tpd['mean']:.3f} ms  max {tpd['max']:.3f} ms  decisions/match mean {report['decisions_per_match_mean']:.1f}
""")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
