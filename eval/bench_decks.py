"""25-deck rotation A/B bench: new main (champion MCTS) vs old main (Greedy).

SOT-1693 generalization gate for the submission switch. Every deck in
``decks/initial`` (the 25 tournament decks, SOT-1684) is played mirror (both
seats pilot the same deck, matching the 25-deck mirror-random environment)
with seat alternation, ``main.SubmissionAgent`` as contestant A vs
``GreedyAgent`` (the previous submission) as contestant B.

The engine RNG is not seedable (ASSUMPTIONS A-9), so results are statistical;
agent seeds are derived per (deck, match) from ``--seed``. Runs are sharded so
long campaigns survive interruption — shard ``s`` plays exactly one match per
deck (25 matches, seat ``s % 2``); rerun a missing shard to resume, then
aggregate:

    venv/bin/python eval/bench_decks.py --match-index 0 \
        --json eval/results/submission/shard_0.json
    ...
    venv/bin/python eval/bench_decks.py --aggregate \
        'eval/results/submission/shard_*.json' \
        --json eval/results/submission/final.json
"""
import argparse
import glob as globmod
import json
import math
import os
import re
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

FAULT_KEYS = ("rejects", "exceptions", "unfinished", "fallbacks_a",
              "fallbacks_b", "budget_violations_a", "planner_fallbacks_a",
              "degraded_a", "emergency_fallbacks_a", "greedy_handoffs_a")


def discover_decks(decks_dir: str) -> list:
    files = [p for p in globmod.glob(os.path.join(decks_dir, "*.csv"))
             if re.match(r"^\d+_", os.path.basename(p))]
    if not files:
        raise SystemExit(f"no NN_*.csv decks found in {decks_dir}")
    files.sort(key=lambda p: int(re.match(r"^(\d+)_", os.path.basename(p)).group(1)))
    return files


def load_deck(path: str) -> list:
    with open(path) as f:
        return [int(x) for x in f.read().split("\n")[:60]]


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def run_shard(args) -> int:
    # Engine + agents only for the play path (aggregate stays engine-free).
    os.chdir(REPO)
    from eval.bench import play_match
    from agents import GreedyAgent
    from agents.rng import Rng
    from main import SubmissionAgent

    decks_dir = args.decks_dir
    if not os.path.isabs(decks_dir):
        decks_dir = os.path.join(REPO, decks_dir)
    deck_files = discover_decks(decks_dir)
    if args.deck_limit is not None:
        deck_files = deck_files[args.deck_offset:
                                args.deck_offset + args.deck_limit]
    base = Rng(args.seed)
    s = args.match_index
    a_first = (s % 2 == 0)
    matches = []
    for deck_path in deck_files:
        deck_name = os.path.basename(deck_path)
        deck = load_deck(deck_path)
        a = SubmissionAgent(seed=base.child(f"{deck_name}.m{s}.a").seed,
                            deck=deck)
        b = GreedyAgent(seed=base.child(f"{deck_name}.m{s}.b").seed, deck=deck)
        p0, p1 = (a, b) if a_first else (b, a)
        t0 = time.perf_counter()
        result, decisions, reject, exception = play_match(p0, p1)
        dt = time.perf_counter() - t0
        rec = {
            "deck": deck_name, "match_index": s, "a_first": a_first,
            "result": result, "decisions": decisions,
            "a_won": result in (0, 1) and ((result == 0) == a_first),
            "b_won": result in (0, 1) and ((result == 0) != a_first),
            "draw": result == 2, "unfinished": result == -1 and not exception,
            "reject": reject, "exception": exception,
            "fallbacks_a": a.fallback_count, "fallbacks_b": b.fallback_count,
            "budget_violations_a": a.budget_violations,
            "planner_fallbacks_a": a.planner_fallbacks,
            "degraded_a": a.degraded_count,
            "emergency_fallbacks_a": a.emergency_fallbacks,
            "greedy_handoffs_a": a.greedy_handoffs,
            "match_time_s": round(dt, 3),
            "a_think_time_s": round(a.think_time_s, 3),
            "a_move_times_ms": [round(t * 1000, 2) for t in a.move_times],
        }
        matches.append(rec)
        print(f"  [{s}] {deck_name}: "
              f"{'A' if rec['a_won'] else 'B' if rec['b_won'] else 'draw/unf'} "
              f"({dt:.1f}s, {decisions} decisions, "
              f"A think {a.think_time_s:.1f}s)", flush=True)
    shard = {"issue": "SOT-1693", "match_index": s, "seed": args.seed,
             "decks_dir": args.decks_dir, "matches": matches}
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(shard, f, indent=1)
        print(f"wrote {args.json}")
    return 0


def aggregate(args) -> int:
    paths = sorted(globmod.glob(args.aggregate))
    if not paths:
        raise SystemExit(f"no shard files match {args.aggregate}")
    matches = []
    for p in paths:
        with open(p) as f:
            matches.extend(json.load(f)["matches"])

    wins_a = sum(m["a_won"] for m in matches)
    wins_b = sum(m["b_won"] for m in matches)
    draws = sum(m["draw"] for m in matches)
    decided = wins_a + wins_b
    ci = wilson_ci(wins_a, decided)
    match_key = {"rejects": "reject", "exceptions": "exception"}
    faults = {k: sum(int(m[match_key.get(k, k)]) for m in matches)
              for k in FAULT_KEYS}

    per_deck = {}
    for m in matches:
        d = per_deck.setdefault(m["deck"], {"n": 0, "a_wins": 0})
        d["n"] += 1
        d["a_wins"] += int(m["a_won"])
    for d in per_deck.values():
        d["a_rate"] = round(d["a_wins"] / d["n"], 3) if d["n"] else None

    move_ms = sorted(t for m in matches for t in m["a_move_times_ms"])
    think = [m["a_think_time_s"] for m in matches]

    def pct(sorted_vals, q):
        if not sorted_vals:
            return None
        i = min(len(sorted_vals) - 1, max(0, math.ceil(q * len(sorted_vals)) - 1))
        return sorted_vals[i]

    report = {
        "issue": "SOT-1693",
        "shards": paths,
        "n_matches": len(matches),
        "wins_a_mcts": wins_a, "wins_b_greedy": wins_b, "draws": draws,
        "winrate_a_excl_draws": (wins_a / decided) if decided else None,
        "wilson95_excl_draws": list(ci),
        "faults": faults,
        "fault_total": sum(faults.values()),
        "per_deck": dict(sorted(per_deck.items())),
        "a_move_time_ms": {
            "n_decisions": len(move_ms),
            "mean": round(statistics.fmean(move_ms), 2) if move_ms else None,
            "p95": pct(move_ms, 0.95),
            "max": move_ms[-1] if move_ms else None,
        },
        "a_think_time_per_match_s": {
            "mean": round(statistics.fmean(think), 2) if think else None,
            "p95": pct(sorted(think), 0.95),
            "max": max(think) if think else None,
            "allowance_s": 600.0,
        },
    }
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("per_deck", "shards")}, indent=1))
    print("\nper-deck A(mcts) win rate:")
    for name, d in report["per_deck"].items():
        print(f"  {name}: {d['a_wins']}/{d['n']} = {d['a_rate']}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
        print(f"\nwrote {args.json}")
    if args.kpi is not None:
        from eval.kpi import append_history, record_from_bench_decks
        path = append_history(record_from_bench_decks(report,
                                                      issue=args.kpi or None))
        print(f"KPI record appended to {path}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decks-dir", default="decks/initial")
    p.add_argument("--match-index", type=int, default=None,
                   help="shard id s: play 1 match per deck, seat A first "
                        "iff s is even")
    p.add_argument("--deck-offset", type=int, default=0,
                   help="skip this many decks (pool-order) in this shard")
    p.add_argument("--deck-limit", type=int, default=None,
                   help="play at most this many decks in this shard")
    p.add_argument("--seed", type=int, default=20260715)
    p.add_argument("--json", default=None, help="output path")
    p.add_argument("--aggregate", default=None,
                   help="glob of shard JSONs to merge instead of playing")
    p.add_argument("--kpi", nargs="?", const="", default=None, metavar="ISSUE",
                   help="with --aggregate: append a KPI record to "
                        "eval/kpi_history.jsonl (SOT-1708)")
    args = p.parse_args(argv)
    if args.aggregate:
        return aggregate(args)
    if args.match_index is None:
        raise SystemExit("--match-index (or --aggregate) is required")
    return run_shard(args)


if __name__ == "__main__":
    sys.exit(main())
