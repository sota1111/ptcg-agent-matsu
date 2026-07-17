"""Matsu battle KPI recording (SOT-1708).

Defines the KPI record schema (``matsu-kpi-v1``, see ``docs/KPI.md``), computes
KPI records from measurements or existing bench reports, and appends one line
per measurement to the committed history file ``eval/kpi_history.jsonl`` —
kept separate from the scratch ``eval/results/`` tree so trends survive.

Three ways to produce a record:

1. **Own measurement** (full KPI coverage incl. self-deck-out classification —
   the terminal observation is captured, which ``eval/bench.py`` does not
   expose). Sharded like ``bench_decks.py`` so long runs chunk/resume:

       venv/bin/python eval/kpi.py --measure --match-index 0 \
           --shard-json eval/results/kpi/shard_0.json
       venv/bin/python eval/kpi.py --finalize 'eval/results/kpi/shard_*.json' \
           --issue SOT-1708

2. **From an existing bench report** (win rate / faults / timing KPIs only;
   ``self_deck_out_loss_rate`` is null because those harnesses discard the
   terminal state):

       venv/bin/python eval/kpi.py --from-report eval/results/final.json \
           --report-kind bench_decks --issue SOT-1693

3. **In-process hook** — ``bench.py`` / ``bench_decks.py`` /
   ``aggregate_shards.py`` accept ``--kpi [issue-id]`` and call
   ``record_from_bench`` / ``record_from_bench_decks`` + ``append_history``.

History and comparison display: ``eval/kpi_report.py``.
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

SCHEMA = "matsu-kpi-v1"
HISTORY_PATH = os.path.join(REPO, "eval", "kpi_history.jsonl")

# Improvement direction per KPI: +1 higher is better, -1 lower is better,
# 0 must stay exactly zero (any nonzero value is a regression).
KPI_DIRECTIONS = {
    "mirror_winrate_vs_greedy": 1,
    "self_deck_out_loss_rate": -1,
    "fault_total": 0,
    "decision_time_mean_ms": -1,
}

# Matsu-side health counters that count into fault_total (same population as
# bench_decks.FAULT_KEYS): every one of these must stay 0 on a healthy run.
FAULT_COUNTERS = ("rejects", "exceptions", "unfinished", "fallbacks",
                  "budget_violations", "planner_fallbacks", "degraded",
                  "emergency_fallbacks", "greedy_handoffs")


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _side(current: dict, seat: int) -> dict:
    players = current.get("players") or ({}, {})
    return players[seat] if seat < len(players) else {}


def classify_loss(final_current: dict, seat: int) -> str:
    """Classify a matsu loss from the engine's terminal state.

    Same mechanism taxonomy as ``eval/loss_trace_matsu.classify_loss`` (which
    is engine/subprocess-coupled and cannot be imported engine-free):
    ``no_active`` / ``deck_out`` / ``prize_race`` / ``other``.
    """
    me = _side(final_current, seat)
    opp = _side(final_current, 1 - seat)
    opp_prize = len(opp.get("prize") or ())
    my_deck = me.get("deckCount", 0) or 0
    active = me.get("active") or ()
    has_active = any(x is not None for x in active)
    in_play = has_active or len(me.get("bench") or ()) > 0
    if not in_play:
        return "no_active"
    if my_deck == 0 and opp_prize > 0:
        return "deck_out"
    if opp_prize == 0:
        return "prize_race"
    return "other"


def _base_record(issue: str, source: str) -> dict:
    return {
        "schema": SCHEMA,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "issue": issue or "unknown",
        "source": source,
    }


def build_record(matches: list, issue: str, source: str = "kpi-measure",
                 opponent: str = "greedy", deck_pool: str = "decks/initial",
                 seed=None) -> dict:
    """KPI record from per-match dicts produced by ``run_measure``."""
    wins = sum(m["matsu_won"] for m in matches)
    losses = sum(m["matsu_lost"] for m in matches)
    draws = sum(m["draw"] for m in matches)
    decided = wins + losses
    deck_out = sum(m.get("loss_cause") == "deck_out" for m in matches)
    faults = {k: sum(int(m.get(k, 0)) for m in matches)
              for k in FAULT_COUNTERS}
    move_sum = sum(m.get("move_time_ms_sum", 0.0) for m in matches)
    move_n = sum(m.get("move_count", 0) for m in matches)
    move_max = max((m.get("move_time_ms_max", 0.0) for m in matches),
                   default=0.0)
    rec = _base_record(issue, source)
    rec.update({
        "opponent": opponent,
        "deck_pool": deck_pool,
        "n_decks": len({m["deck"] for m in matches}),
        "n_matches": len(matches),
        "seed": seed,
        "kpis": {
            "mirror_winrate_vs_greedy": {
                "value": round(wins / decided, 4) if decided else None,
                "ci95": [round(x, 4) for x in wilson_ci(wins, decided)],
                "wins": wins, "losses": losses, "draws": draws,
            },
            "self_deck_out_loss_rate": {
                "value": round(deck_out / losses, 4) if losses else None,
                "deck_out_losses": deck_out, "losses": losses,
            },
            "fault_total": {
                "value": sum(faults.values()),
                "breakdown": faults,
            },
            "decision_time_mean_ms": {
                "value": round(move_sum / move_n, 2) if move_n else None,
                "max_ms": round(move_max, 2),
                "n_decisions": move_n,
                "budget_violations": faults["budget_violations"],
            },
        },
    })
    return rec


def _kpis_from_flat(winrate, ci, wins, losses, draws, faults: dict,
                    move_mean, move_max, budget_violations) -> dict:
    """Shared KPI block for converted bench reports (no terminal state)."""
    return {
        "mirror_winrate_vs_greedy": {
            "value": round(winrate, 4) if winrate is not None else None,
            "ci95": ([round(x, 4) for x in ci]
                     if ci and ci[0] is not None else None),
            "wins": wins, "losses": losses, "draws": draws,
        },
        "self_deck_out_loss_rate": {
            "value": None, "deck_out_losses": None, "losses": losses,
            "note": "terminal state not captured by this harness",
        },
        "fault_total": {"value": sum(faults.values()), "breakdown": faults},
        "decision_time_mean_ms": {
            "value": round(move_mean, 2) if move_mean is not None else None,
            "max_ms": round(move_max, 2) if move_max is not None else None,
            "budget_violations": budget_violations,
        },
    }


def record_from_bench_decks(report: dict, issue: str = None) -> dict:
    """KPI record from a ``bench_decks.py --aggregate`` report."""
    faults = dict(report.get("faults") or {})
    mt = report.get("a_move_time_ms") or {}
    rec = _base_record(issue or report.get("issue"), "bench_decks")
    rec.update({
        "opponent": "greedy",
        "deck_pool": report.get("decks_dir", "decks/initial"),
        "n_decks": len(report.get("per_deck") or {}) or None,
        "n_matches": report.get("n_matches"),
        "seed": report.get("seed"),
        "kpis": _kpis_from_flat(
            report.get("winrate_a_excl_draws"),
            report.get("wilson95_excl_draws"),
            report.get("wins_a_mcts", 0), report.get("wins_b_greedy", 0),
            report.get("draws", 0), faults,
            mt.get("mean"), mt.get("max"),
            faults.get("budget_violations_a", 0)),
    })
    return rec


def record_from_bench(report: dict, issue: str = None) -> dict:
    """KPI record from an ``eval/bench.py`` (or ``aggregate_shards.py``)
    report — A-side is treated as matsu."""
    faults = {k: report.get(k, 0) or 0 for k in
              ("rejects", "exceptions", "unfinished", "fallbacks_a",
               "budget_violations_a", "planner_fallbacks_a",
               "degraded_count_a")}
    tpd = report.get("time_per_decision_ms") or {}
    move_max = report.get("planner_move_max_ms", tpd.get("max"))
    rec = _base_record(issue, "bench")
    rec.update({
        "opponent": report.get("agent_b"),
        "deck_pool": report.get("deck"),
        "n_decks": 1,
        "n_matches": report.get("n_matches"),
        "seed": report.get("seed"),
        "kpis": _kpis_from_flat(
            report.get("winrate_a_excl_draws"),
            report.get("wilson95_excl_draws"),
            report.get("wins_a", 0), report.get("wins_b", 0),
            report.get("draws", 0), faults,
            tpd.get("mean"), move_max,
            report.get("budget_violations_a", 0)),
    })
    return rec


def append_history(record: dict, path: str = HISTORY_PATH) -> str:
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def load_history(path: str = HISTORY_PATH) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- measurement

def run_measure(args) -> int:
    """Play one shard: SubmissionAgent (matsu) vs GreedyAgent, mirror per
    deck, capturing the terminal observation for loss classification."""
    os.chdir(REPO)
    from cg import game
    from agents import GreedyAgent
    from agents.rng import Rng
    from eval.bench_decks import discover_decks, load_deck
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
    matsu_seat = s % 2
    matches = []
    for deck_path in deck_files:
        deck_name = os.path.basename(deck_path)
        deck = load_deck(deck_path)
        matsu = SubmissionAgent(seed=base.child(f"{deck_name}.m{s}.a").seed,
                                deck=deck)
        opp = GreedyAgent(seed=base.child(f"{deck_name}.m{s}.b").seed,
                          deck=deck)
        agents = (matsu, opp) if matsu_seat == 0 else (opp, matsu)
        t0 = time.perf_counter()
        obs, start = game.battle_start(agents[0]._deck, agents[1]._deck)
        if obs is None:
            raise RuntimeError(f"battle_start failed: "
                               f"errorType={start.errorType}")
        result, decisions, reject, exception = -1, 0, False, False
        last_current: dict = {}
        try:
            while decisions < 100_000:
                cur = obs.get("current") or {}
                last_current = cur
                result = cur.get("result", -1)
                if result != -1:
                    break
                agent = agents[cur.get("yourIndex", 0)]
                try:
                    action = agent.act(obs)
                except Exception:
                    exception = True
                    result = -1
                    break
                try:
                    obs = game.battle_select(action)
                except Exception:
                    reject = True
                    result = -1
                    break
                decisions += 1
        finally:
            game.battle_finish()
        dt = time.perf_counter() - t0
        matsu_won = result in (0, 1) and result == matsu_seat
        matsu_lost = result in (0, 1) and result != matsu_seat
        move_times = matsu.move_times or []
        rec = {
            "deck": deck_name, "match_index": s, "matsu_seat": matsu_seat,
            "result": result, "decisions": decisions,
            "matsu_won": matsu_won, "matsu_lost": matsu_lost,
            "draw": result == 2,
            "unfinished": result == -1 and not (reject or exception),
            "loss_cause": (classify_loss(last_current, matsu_seat)
                           if matsu_lost else None),
            "rejects": int(reject), "exceptions": int(exception),
            "fallbacks": matsu.fallback_count,
            "budget_violations": matsu.budget_violations,
            "planner_fallbacks": matsu.planner_fallbacks,
            "degraded": matsu.degraded_count,
            "emergency_fallbacks": matsu.emergency_fallbacks,
            "greedy_handoffs": matsu.greedy_handoffs,
            "match_time_s": round(dt, 3),
            "move_time_ms_sum": round(sum(move_times) * 1000, 2),
            "move_count": len(move_times),
            "move_time_ms_max": round(max(move_times, default=0.0) * 1000, 2),
        }
        matches.append(rec)
        print(f"  [{s}] {deck_name}: "
              f"{'W' if matsu_won else 'L' if matsu_lost else 'draw/unf'}"
              f"{' (' + rec['loss_cause'] + ')' if rec['loss_cause'] else ''}"
              f" {dt:.1f}s {decisions} decisions", flush=True)
    shard = {"schema": SCHEMA + "-shard", "match_index": s, "seed": args.seed,
             "decks_dir": args.decks_dir, "matches": matches}
    if args.shard_json:
        os.makedirs(os.path.dirname(args.shard_json) or ".", exist_ok=True)
        with open(args.shard_json, "w") as f:
            json.dump(shard, f, indent=1)
        print(f"wrote {args.shard_json}")
    return 0


def run_finalize(args) -> int:
    paths = sorted(globmod.glob(args.finalize))
    if not paths:
        raise SystemExit(f"no shard files match {args.finalize}")
    matches, seeds, decks_dirs = [], set(), set()
    for p in paths:
        with open(p) as f:
            shard = json.load(f)
        matches.extend(shard["matches"])
        seeds.add(shard.get("seed"))
        decks_dirs.add(shard.get("decks_dir"))
    rec = build_record(matches, issue=args.issue,
                       deck_pool=decks_dirs.pop() if len(decks_dirs) == 1
                       else sorted(decks_dirs),
                       seed=seeds.pop() if len(seeds) == 1
                       else sorted(seeds))
    print(json.dumps(rec, indent=1))
    if args.no_append:
        print("(--no-append: history not written)")
    else:
        print(f"appended to {append_history(rec)}")
    return 0


def run_from_report(args) -> int:
    with open(args.from_report) as f:
        report = json.load(f)
    conv = {"bench_decks": record_from_bench_decks,
            "bench": record_from_bench}[args.report_kind]
    rec = conv(report, issue=args.issue)
    print(json.dumps(rec, indent=1))
    if args.no_append:
        print("(--no-append: history not written)")
    else:
        print(f"appended to {append_history(rec)}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measure", action="store_true",
                   help="play one shard of matsu-vs-greedy mirror matches")
    p.add_argument("--match-index", type=int, default=0,
                   help="shard id s (matsu takes seat s%%2)")
    p.add_argument("--decks-dir", default="decks/initial")
    p.add_argument("--deck-offset", type=int, default=0)
    p.add_argument("--deck-limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260717)
    p.add_argument("--shard-json", default=None)
    p.add_argument("--finalize", default=None,
                   help="glob of --measure shard JSONs -> one history record")
    p.add_argument("--from-report", default=None,
                   help="existing bench report JSON -> one history record")
    p.add_argument("--report-kind", choices=("bench", "bench_decks"),
                   default="bench_decks")
    p.add_argument("--issue", default=None, help="Linear issue id to record")
    p.add_argument("--no-append", action="store_true",
                   help="print the record without touching the history")
    args = p.parse_args(argv)
    if args.measure:
        return run_measure(args)
    if args.finalize:
        return run_finalize(args)
    if args.from_report:
        return run_from_report(args)
    raise SystemExit("one of --measure / --finalize / --from-report required")


if __name__ == "__main__":
    sys.exit(main())
