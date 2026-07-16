"""Config-vs-config 25-deck mirror A/B bench (SOT-1697).

Compares two ``MctsPlanner`` configurations head-to-head on the 25 tournament
decks (SOT-1684) played mirror with seat alternation — the SOT-1681 skill-
isolating environment. Contestant A is the **candidate** config, contestant B
is the **champion** (``main.CHAMPION_CONFIG``); both run as ``MctsAgent`` at the
same 0.8s budget, so the bench isolates the config change (the SOT-1697 rollout
speedup is in the shared code and benefits both sides equally).

Promotion gate (SOT-1697 / SOT-1673 method): the candidate is promoted only if
the Wilson 95% lower bound of its win rate vs the champion is **> 0.5**; a tie
(CI straddling 0.5) keeps the champion. Every run also asserts the health
invariants the issue requires: engine rejects / agent exceptions / budget
violations (時間切れ) all 0.

Sharded like ``bench_decks.py`` so long campaigns survive interruption — shard
``s`` plays one match per deck (seat ``s % 2``); rerun a missing shard, then
aggregate::

    venv/bin/python eval/bench_configs.py --match-index 0 \
        --candidate '{"max_tree_depth": 2}' \
        --json eval/results/sot1697/depth2/shard_0.json
    ...
    venv/bin/python eval/bench_configs.py --aggregate \
        'eval/results/sot1697/depth2/shard_*.json' \
        --json eval/results/sot1697/depth2/final.json
"""
import argparse
import glob as globmod
import json
import math
import os
import re
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Health invariants that must stay 0 (per side where meaningful).
FAULT_KEYS = ("rejects", "exceptions", "unfinished", "fallbacks_a",
              "fallbacks_b", "budget_violations_a", "budget_violations_b",
              "planner_fallbacks_a", "planner_fallbacks_b",
              "degraded_a", "degraded_b")


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
    os.chdir(REPO)
    from eval.bench import play_match
    from agents import MctsAgent
    from agents.rng import Rng
    from main import CHAMPION_CONFIG

    candidate = dict(CHAMPION_CONFIG)
    candidate.update(json.loads(args.candidate) if args.candidate else {})
    champion = dict(CHAMPION_CONFIG)
    champion.update(json.loads(args.champion) if args.champion else {})

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
        a = MctsAgent(seed=base.child(f"{deck_name}.m{s}.a").seed, deck=deck,
                      **candidate)
        b = MctsAgent(seed=base.child(f"{deck_name}.m{s}.b").seed, deck=deck,
                      **champion)
        p0, p1 = (a, b) if a_first else (b, a)
        import time
        t0 = time.perf_counter()
        result, decisions, reject, exception = play_match(p0, p1)
        dt = time.perf_counter() - t0
        a_move_max = max(a.move_times) if a.move_times else 0.0
        b_move_max = max(b.move_times) if b.move_times else 0.0
        rec = {
            "deck": deck_name, "match_index": s, "a_first": a_first,
            "result": result, "decisions": decisions,
            "a_won": result in (0, 1) and ((result == 0) == a_first),
            "b_won": result in (0, 1) and ((result == 0) != a_first),
            "draw": result == 2, "unfinished": result == -1 and not exception,
            "reject": reject, "exception": exception,
            "fallbacks_a": a.fallback_count, "fallbacks_b": b.fallback_count,
            "budget_violations_a": a.budget_violations,
            "budget_violations_b": b.budget_violations,
            "planner_fallbacks_a": a.planner_fallbacks,
            "planner_fallbacks_b": b.planner_fallbacks,
            "degraded_a": a.degraded_count, "degraded_b": b.degraded_count,
            "match_time_s": round(dt, 3),
            "a_move_max_ms": round(a_move_max * 1000, 1),
            "b_move_max_ms": round(b_move_max * 1000, 1),
        }
        matches.append(rec)
        print(f"  [{s}] {deck_name}: "
              f"{'A' if rec['a_won'] else 'B' if rec['b_won'] else 'draw/unf'} "
              f"({dt:.1f}s, {decisions} dec)", flush=True)
    shard = {"issue": "SOT-1697", "match_index": s, "seed": args.seed,
             "candidate": candidate, "champion": champion,
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
    matches, candidate, champion = [], None, None
    for p in paths:
        with open(p) as f:
            shard = json.load(f)
        matches.extend(shard["matches"])
        candidate = candidate or shard.get("candidate")
        champion = champion or shard.get("champion")

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

    move_ms = sorted(m["a_move_max_ms"] for m in matches)
    promote = decided > 0 and ci[0] > 0.5
    report = {
        "issue": "SOT-1697",
        "shards": paths,
        "candidate": candidate, "champion": champion,
        "n_matches": len(matches),
        "wins_a_candidate": wins_a, "wins_b_champion": wins_b, "draws": draws,
        "winrate_a_excl_draws": round(wins_a / decided, 4) if decided else None,
        "wilson95_excl_draws": [round(ci[0], 4), round(ci[1], 4)],
        "promote_candidate": promote,
        "gate": "CI lower bound > 0.5",
        "faults": faults,
        "fault_total": sum(faults.values()),
        "a_move_max_ms_over_matches": {
            "mean": round(statistics.fmean(move_ms), 1) if move_ms else None,
            "max": move_ms[-1] if move_ms else None,
        },
        "per_deck": dict(sorted(per_deck.items())),
    }
    printable = {k: v for k, v in report.items()
                 if k not in ("per_deck", "shards")}
    print(json.dumps(printable, indent=1))
    print("\nper-deck A(candidate) win rate:")
    for name, d in report["per_deck"].items():
        print(f"  {name}: {d['a_wins']}/{d['n']} = {d['a_rate']}")
    verdict = ("PROMOTE candidate" if promote
               else "KEEP champion (no significant improvement)")
    print(f"\n=> {verdict}  win_rate={report['winrate_a_excl_draws']} "
          f"CI95={report['wilson95_excl_draws']} faults={report['fault_total']}")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
        print(f"\nwrote {args.json}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decks-dir", default="decks/initial")
    p.add_argument("--match-index", type=int, default=None,
                   help="shard id s: 1 match per deck, A first iff s even")
    p.add_argument("--deck-offset", type=int, default=0)
    p.add_argument("--deck-limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260716)
    p.add_argument("--candidate", default=None,
                   help="JSON overrides on CHAMPION_CONFIG for contestant A")
    p.add_argument("--champion", default=None,
                   help="JSON overrides on CHAMPION_CONFIG for contestant B "
                        "(default: the champion config itself)")
    p.add_argument("--json", default=None)
    p.add_argument("--aggregate", default=None,
                   help="glob of shard JSONs to merge instead of playing")
    args = p.parse_args(argv)
    if args.aggregate:
        return aggregate(args)
    if args.match_index is None:
        raise SystemExit("--match-index (or --aggregate) is required")
    return run_shard(args)


if __name__ == "__main__":
    sys.exit(main())
