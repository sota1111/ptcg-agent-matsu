"""Matsu loss-cause tracing vs 竹/梅 (SOT-1697, step 1).

SOT-1681's 25-deck mirror-random rematch put 松 first overall (0.792) but it
still dropped ~24% of games to 梅 and ~18% to 竹. Before touching the search we
find out *why* those games are lost, because the SOT-1682/1694 lesson is that
mining the aggregate is the fastest way to spot the hole worth fixing.

Unlike ``eval/battle_matsu_take_ume.py`` (which runs all three agents as opaque
subprocesses and only tallies wins), this harness hosts **matsu in-process** as a
``main.SubmissionAgent`` — so we get its per-match planner health counters
(degraded / greedy-handoff / emergency-fallback / budget-violation) and the
engine's **terminal observation** — and drives only the opponent (竹 or 梅) as an
``eval/agent_server`` subprocess (their ``agents`` packages collide with matsu's,
so they must stay isolated). Decks are the 25 tournament decks (SOT-1684) played
mirror with seat alternation, exactly the SOT-1681 environment.

Each matsu loss is classified by mechanism from the terminal state:

* ``deck_out``    — matsu's deck ran out (deckCount 0, opponent still had prizes);
* ``no_active``   — matsu had no Pokémon left in play (active gone, bench empty);
* ``prize_race``  — opponent took their last prize (lost the damage race);
* ``other``       — none of the above / unfinished.

Each loss also records the turn it ended (early losses ≈ 立ち上がり事故) and
whether matsu's search was **healthy** (no degraded/fallback decisions) — a
prize-race loss with a healthy search is a genuine strategic/"search miss" loss,
the population the SOT-1697 depth/worlds work targets.

Usage (from the repo root; ../ptcg-agent-take and ../ptcg-agent-ume must exist)::

    venv/bin/python eval/loss_trace_matsu.py --n 24 --opponents take ume \
        --decks-dir decks/initial --seed 20260716 \
        --json eval/results/sot1697/loss_trace.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from eval.battle_matsu_take_ume import (  # noqa: E402
    Contestant, CONTESTANTS, SIBLINGS, build_deck_schedule, discover_decks,
    load_deck, make_sandbox, wilson_ci)
from agents.rng import Rng  # noqa: E402
from main import SubmissionAgent  # noqa: E402

MAX_DECISIONS = 100_000
EARLY_TURN = 6  # loss at/under this turn count is flagged 立ち上がり事故


def _side(current: dict, seat: int) -> dict:
    players = current.get("players") or ({}, {})
    return players[seat] if seat < len(players) else {}


def classify_loss(final_current: dict, matsu_seat: int) -> dict:
    """Classify one matsu loss from the engine's terminal state."""
    me = _side(final_current, matsu_seat)
    opp = _side(final_current, 1 - matsu_seat)
    turn = final_current.get("turn", 0) or 0
    my_prize = len(me.get("prize") or ())
    opp_prize = len(opp.get("prize") or ())
    my_deck = me.get("deckCount", 0) or 0
    active = me.get("active") or ()
    has_active = any(x is not None for x in active)
    bench = me.get("bench") or ()
    in_play = has_active or len(bench) > 0

    if not in_play:
        cause = "no_active"
    elif my_deck == 0 and opp_prize > 0:
        cause = "deck_out"
    elif opp_prize == 0:
        cause = "prize_race"
    else:
        cause = "other"
    return {
        "cause": cause,
        "turn": turn,
        "early": turn <= EARLY_TURN,
        "my_prize_left": my_prize,   # 6 = matsu took none; 1 = matsu almost won
        "opp_prize_left": opp_prize,
    }


def play_match(game, matsu: SubmissionAgent, matsu_seat: int,
               opp: Contestant) -> dict:
    """One match: matsu (in-process) at ``matsu_seat`` vs ``opp`` (subprocess).

    Returns result, steps, the terminal ``current`` dict and a fault flag.
    """
    deck0 = matsu._deck if matsu_seat == 0 else opp.deck
    deck1 = opp.deck if matsu_seat == 0 else matsu._deck
    obs, start = game.battle_start(deck0, deck1)
    if obs is None:
        raise RuntimeError(f"battle_start failed: errorType={start.errorType}")
    steps = 0
    last_current: dict = {}
    fault_seat = None
    try:
        while steps < MAX_DECISIONS:
            cur = obs.get("current") or {}
            last_current = cur
            result = cur.get("result", -1)
            if result != -1:
                return {"result": result, "steps": steps,
                        "current": cur, "fault_seat": fault_seat}
            seat = cur.get("yourIndex", 0)
            actor = matsu if seat == matsu_seat else opp
            try:
                action = actor.act(obs)
            except Exception:
                fault_seat = seat
                return {"result": 1 - seat, "steps": steps,
                        "current": cur, "fault_seat": seat}
            try:
                obs = game.battle_select(action)
            except Exception:
                fault_seat = seat
                return {"result": 1 - seat, "steps": steps,
                        "current": cur, "fault_seat": seat}
            steps += 1
        return {"result": -1, "steps": steps, "current": last_current,
                "fault_seat": None}
    finally:
        game.battle_finish()


def run_opponent(game, opp_label: str, opp_dir: str, deck_pool, deck_cache,
                 schedule, seed: int) -> dict:
    """Play matsu vs one opponent over the mirror schedule; trace matsu losses."""
    repo = os.path.join(SIBLINGS, opp_dir)
    opp = Contestant(label=opp_label, repo=repo, deck=[],
                     sandbox=make_sandbox(repo))
    opp.start()
    base = Rng(seed)
    n = len(schedule)
    wins = losses = draws = unfinished = faults = 0
    loss_records = []
    win_turns = []
    try:
        for i in range(n):
            deck_file = schedule[i][0]  # mirror: both play the same deck
            deck = deck_cache[deck_file]
            opp.deck = deck
            opp.set_deck(deck)
            matsu_seat = 0 if (i % 2 == 0) else 1
            matsu = SubmissionAgent(
                seed=base.child(f"{os.path.basename(deck_file)}.m{i}").seed,
                deck=deck)
            out = play_match(game, matsu, matsu_seat, opp)
            res = out["result"]
            if out["fault_seat"] is not None:
                # A faulting matsu counts as a (search-broken) loss we must see;
                # a faulting opponent we drop (relaunch) — not a matsu result.
                if out["fault_seat"] == matsu_seat:
                    faults += 1
                else:
                    opp.restart()
                    continue
            if res == 2:
                draws += 1
            elif res == -1:
                unfinished += 1
            elif res == matsu_seat:
                wins += 1
                win_turns.append((out["current"].get("turn", 0) or 0))
            else:
                losses += 1
                info = classify_loss(out["current"], matsu_seat)
                info.update({
                    "deck": os.path.basename(deck_file),
                    "matsu_seat": matsu_seat,
                    "steps": out["steps"],
                    "degraded": matsu.degraded_count,
                    "greedy_handoffs": matsu.greedy_handoffs,
                    "emergency_fallbacks": matsu.emergency_fallbacks,
                    "budget_violations": matsu.budget_violations,
                    "planner_fallbacks": matsu.planner_fallbacks,
                    "fault": out["fault_seat"] == matsu_seat,
                    "search_healthy": (matsu.degraded_count == 0
                                       and matsu.greedy_handoffs == 0
                                       and matsu.emergency_fallbacks == 0
                                       and matsu.planner_fallbacks == 0),
                })
                loss_records.append(info)
            if (i + 1) % 10 == 0:
                print(f"  [matsu vs {opp_label}] {i + 1}/{n} "
                      f"(W{wins} L{losses} D{draws})", file=sys.stderr,
                      flush=True)
    finally:
        opp.stop()
        if opp.sandbox:
            import shutil
            shutil.rmtree(opp.sandbox, ignore_errors=True)

    decided = wins + losses
    lo, hi = wilson_ci(wins, decided)
    # Loss-cause breakdown.
    by_cause: dict[str, int] = {}
    for r in loss_records:
        by_cause[r["cause"]] = by_cause.get(r["cause"], 0) + 1
    early = sum(1 for r in loss_records if r["early"])
    unhealthy = sum(1 for r in loss_records if not r["search_healthy"])
    prize_healthy = sum(1 for r in loss_records
                        if r["cause"] == "prize_race" and r["search_healthy"])
    return {
        "opponent": opp_label,
        "n": n, "wins": wins, "losses": losses, "draws": draws,
        "unfinished": unfinished, "faults": faults,
        "matsu_win_rate": round(wins / decided, 4) if decided else None,
        "matsu_win_rate_ci95": [round(lo, 4), round(hi, 4)],
        "loss_causes": dict(sorted(by_cause.items())),
        "loss_early_setup": early,
        "loss_search_unhealthy": unhealthy,
        "loss_prize_race_search_healthy": prize_healthy,
        "loss_turn_mean": (round(sum(r["turn"] for r in loss_records)
                                 / len(loss_records), 1)
                           if loss_records else None),
        "loss_records": loss_records,
    }


def _summarize(opp_label: str, recs: list, wins: int, faults: int,
               draws: int, unfinished: int) -> dict:
    """Recompute the per-opponent breakdown from merged loss records."""
    losses = len(recs)
    decided = wins + losses
    lo, hi = wilson_ci(wins, decided)
    by_cause: dict[str, int] = {}
    for r in recs:
        by_cause[r["cause"]] = by_cause.get(r["cause"], 0) + 1
    early = sum(1 for r in recs if r["early"])
    unhealthy = sum(1 for r in recs if not r["search_healthy"])
    prize_healthy = sum(1 for r in recs
                        if r["cause"] == "prize_race" and r["search_healthy"])
    return {
        "opponent": opp_label,
        "n": decided + draws + unfinished,
        "wins": wins, "losses": losses, "draws": draws,
        "unfinished": unfinished, "faults": faults,
        "matsu_win_rate": round(wins / decided, 4) if decided else None,
        "matsu_win_rate_ci95": [round(lo, 4), round(hi, 4)],
        "loss_causes": dict(sorted(by_cause.items())),
        "loss_early_setup": early,
        "loss_search_unhealthy": unhealthy,
        "loss_prize_race_search_healthy": prize_healthy,
        "loss_turn_mean": (round(sum(r["turn"] for r in recs) / len(recs), 1)
                           if recs else None),
        "loss_records": recs,
    }


def aggregate(paths: list, out_json: str | None) -> int:
    """Merge many shard report JSONs (independent seeds) into one report."""
    import glob as globmod
    files = []
    for pat in paths:
        files.extend(sorted(globmod.glob(pat)))
    if not files:
        raise SystemExit(f"no shard JSONs match {paths}")
    merged: dict[str, dict] = {}
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            rep = json.load(fh)
        for r in rep["results"]:
            m = merged.setdefault(r["opponent"], {
                "recs": [], "wins": 0, "faults": 0, "draws": 0, "unfinished": 0})
            m["recs"].extend(r["loss_records"])
            m["wins"] += r["wins"]
            m["faults"] += r.get("faults", 0)
            m["draws"] += r.get("draws", 0)
            m["unfinished"] += r.get("unfinished", 0)
    results = [_summarize(opp, d["recs"], d["wins"], d["faults"],
                          d["draws"], d["unfinished"])
               for opp, d in sorted(merged.items())]
    report = {"issue": "SOT-1697", "aggregate_of": files, "results": results}
    for r in results:
        print(f"\nmatsu vs {r['opponent']}: W{r['wins']} L{r['losses']} "
              f"D{r['draws']} win_rate={r['matsu_win_rate']} "
              f"CI95={r['matsu_win_rate_ci95']} faults={r['faults']}")
        print(f"  loss causes: {r['loss_causes']}  "
              f"early(setup)={r['loss_early_setup']}  "
              f"search_unhealthy={r['loss_search_unhealthy']}  "
              f"prize_race&healthy={r['loss_prize_race_search_healthy']}  "
              f"loss_turn_mean={r['loss_turn_mean']}")
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"\nwrote {out_json}", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--n", type=int, default=24, help="matches per opponent")
    p.add_argument("--opponents", nargs="+", default=["take", "ume"])
    p.add_argument("--decks-dir", default="decks/initial")
    p.add_argument("--seed", type=int, default=20260716)
    p.add_argument("--json", default=None)
    p.add_argument("--aggregate", nargs="+", default=None,
                   help="glob(s) of shard report JSONs to merge instead of play")
    args = p.parse_args(argv)

    if args.aggregate:
        return aggregate(args.aggregate, args.json)

    from cg import game
    decks_dir = args.decks_dir
    if not os.path.isabs(decks_dir):
        decks_dir = os.path.join(REPO, decks_dir)
    deck_pool = discover_decks(decks_dir)
    deck_cache = {f: load_deck(f) for f in deck_pool}
    dir_by_label = {label: dirname for label, _k, dirname in CONTESTANTS}

    results = []
    for opp_label in args.opponents:
        rng = random.Random(f"{args.seed}:matsu:{opp_label}")
        schedule = build_deck_schedule(args.n, deck_pool, "mirror", rng)
        print(f"== matsu vs {opp_label} (n={args.n}) ==", file=sys.stderr,
              flush=True)
        results.append(run_opponent(game, opp_label,
                                    dir_by_label[opp_label], deck_pool,
                                    deck_cache, schedule, args.seed))

    report = {"issue": "SOT-1697", "n_per_opponent": args.n,
              "seed": args.seed, "decks_dir": args.decks_dir,
              "results": results}
    for r in results:
        print(f"\nmatsu vs {r['opponent']}: W{r['wins']} L{r['losses']} "
              f"D{r['draws']} win_rate={r['matsu_win_rate']} "
              f"CI95={r['matsu_win_rate_ci95']} faults={r['faults']}")
        print(f"  loss causes: {r['loss_causes']}  "
              f"early(setup)={r['loss_early_setup']}  "
              f"search_unhealthy={r['loss_search_unhealthy']}  "
              f"prize_race&healthy={r['loss_prize_race_search_healthy']}  "
              f"loss_turn_mean={r['loss_turn_mean']}")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
