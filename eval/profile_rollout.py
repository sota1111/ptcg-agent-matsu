"""Rollout hot-path profiling / speedup measurement (SOT-1697).

The champion MCTS greedy rollout used to convert every search-API observation
back to a raw dict with ``dataclasses.asdict()`` just so ``GreedyAgent`` could
score it. Profiling showed that recursive round-trip was ~40% of a champion
decision. SOT-1697 replaced it with ``agents.observation.adapt_engine_obs``,
which reads the engine dataclass directly.

This script measures the change two ways, on a fixed set of real search
positions (so the numbers are comparable run-to-run, unlike a whole match whose
length varies with the engine's non-seedable RNG):

1. **adapter micro-benchmark** — greedy-choice throughput of the OLD asdict path
   vs the NEW ``adapt_engine_obs`` path on identical observations (isolates the
   change; also asserts the two pick the same action every time);
2. **planner iterations/sec** — MCTS iterations completed in a fixed
   ``time_budget_s`` at each position with the old vs new rollout, i.e. how many
   more rollouts the same 0.8s budget now buys end-to-end.

Usage (from the repo root)::

    venv/bin/python eval/profile_rollout.py --positions 12 --budget 0.8 \
        --json eval/results/sot1697/profile.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from cg import api, game  # noqa: E402
from agents.greedy_agent import GreedyAgent  # noqa: E402
from agents.observation import adapt, adapt_engine_obs  # noqa: E402
from agents.planner import MctsPlanner, PlannerConfig, sample_fills  # noqa: E402
from agents.rng import Rng  # noqa: E402
from main import CHAMPION_CONFIG  # noqa: E402


def load_deck(path: str) -> list:
    with open(path) as f:
        return [int(x) for x in f.read().split("\n")[:60]]


def old_choice(greedy: GreedyAgent, obs) -> list:
    """The pre-SOT-1697 rollout greedy path: asdict round-trip then adapt."""
    view = adapt({"select": asdict(obs.select),
                  "current": asdict(obs.current)})
    return greedy.choose(view)


def new_choice(greedy: GreedyAgent, obs) -> list:
    return greedy.choose(adapt_engine_obs(obs))


def collect_positions(deck: list, n: int, seed: int):
    """Play a greedy self-match and capture, at each real decision, both the
    top-level raw observation dict (for the planner bench) and one search-world
    dataclass observation (for the adapter micro-bench). Returns two lists."""
    rng = Rng(seed)
    greedy = GreedyAgent(seed=0)
    raw_obs_list, search_obs_list = [], []
    obs, start = game.battle_start(deck, deck)
    steps = 0
    while len(search_obs_list) < n and steps < 200:
        cur = obs.get("current") or {}
        if cur.get("result", -1) != -1:
            break
        sel = obs.get("select")
        if sel is not None and len(sel.get("option") or ()) > 1:
            raw_obs_list.append(obs)
            try:
                fills = sample_fills(obs, deck, rng.child(f"s{steps}"),
                                     greedy.cards)
                st = api.search_begin(
                    api.to_observation_class(obs), fills.my_deck,
                    fills.my_prize, fills.opp_deck, fills.opp_prize,
                    fills.opp_hand, fills.opp_active, manual_coin=True)
                sobs = st.observation
                if sobs.select is not None and sobs.current is not None \
                        and len(sobs.select.option or ()) > 1:
                    search_obs_list.append(sobs)
                api.search_end()
            except Exception:
                pass
        obs = game.battle_select(greedy.act(obs))
        steps += 1
    game.battle_finish()
    return raw_obs_list, search_obs_list


def bench_adapter(positions, reps: int):
    greedy = GreedyAgent(seed=0)
    # correctness: identical action on every position
    mismatches = sum(1 for o in positions
                     if old_choice(greedy, o) != new_choice(greedy, o))
    t0 = time.perf_counter()
    for _ in range(reps):
        for o in positions:
            old_choice(greedy, o)
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(reps):
        for o in positions:
            new_choice(greedy, o)
    t_new = time.perf_counter() - t0
    calls = reps * len(positions)
    return {
        "positions": len(positions),
        "reps": reps,
        "calls": calls,
        "mismatched_choices": mismatches,
        "old_choices_per_s": round(calls / t_old, 1) if t_old else None,
        "new_choices_per_s": round(calls / t_new, 1) if t_new else None,
        "speedup_x": round(t_old / t_new, 3) if t_new else None,
    }


def _make_old_rollout_action(planner):
    """A drop-in _rollout_action that forces the pre-SOT-1697 asdict path."""
    def _old_rollout_action(obs, rng):
        sel = obs.select
        n = len(sel.option or ())
        if n == 0:
            return []
        hi = min(max(sel.maxCount, 0), n)
        lo = min(max(sel.minCount, 0), hi)
        if getattr(sel, "context", -1) == 46:  # coin (chance)
            return sorted(rng.sample(range(n), max(lo, min(1, hi))))
        try:
            return old_choice(planner._greedy, obs)
        except Exception:
            return sorted(range(lo))
    return _old_rollout_action


def bench_planner(deck, raw_positions, budget: float, seed: int):
    """MCTS iterations completed in a fixed budget at real decisions, old vs new
    rollout path. plan() takes a View built from the top-level raw obs dict."""
    cfg = PlannerConfig(**{**CHAMPION_CONFIG, "time_budget_s": budget})
    result = {}
    for tag in ("old", "new"):
        planner = MctsPlanner(deck, config=cfg)
        if tag == "old":
            planner._rollout_action = _make_old_rollout_action(planner)
        iters, elapsed = [], []
        for raw in raw_positions:
            view = adapt(raw)
            rng = Rng(seed).child("plan")
            try:
                planner.plan(view, rng, budget_s=budget)
                st = planner.last_stats
                if st.get("iterations"):
                    iters.append(st["iterations"])
                    elapsed.append(st.get("elapsed_s", budget))
            except Exception:
                pass
        result[tag] = (iters, elapsed)

    def mean(xs):
        return round(sum(xs) / len(xs), 1) if xs else None
    oi, _ = result["old"]
    ni, _ = result["new"]
    return {
        "budget_s": budget,
        "n_positions": len(raw_positions),
        "old_iters_mean": mean(oi),
        "new_iters_mean": mean(ni),
        "old_n": len(oi),
        "new_n": len(ni),
        "iters_speedup_x": (round(mean(ni) / mean(oi), 3)
                            if oi and ni and mean(oi) else None),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deck", default="deck.csv")
    p.add_argument("--positions", type=int, default=12)
    p.add_argument("--reps", type=int, default=40)
    p.add_argument("--budget", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=20260716)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    deck = load_deck(args.deck)
    raw_positions, search_positions = collect_positions(deck, args.positions,
                                                        args.seed)
    print(f"collected {len(raw_positions)} raw / {len(search_positions)} "
          f"search positions", flush=True)
    adapter = bench_adapter(search_positions, args.reps)
    print("adapter:", json.dumps(adapter), flush=True)
    planner = bench_planner(deck, raw_positions, args.budget, args.seed)
    print("planner:", json.dumps(planner), flush=True)
    report = {"issue": "SOT-1697", "deck": args.deck,
              "adapter_microbench": adapter, "planner_iterations": planner}
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
