"""Build and run a versioned regression suite from Kaggle episode replays.

The generator accepts replay JSON files downloaded with ``kaggle competitions
replay``.  It keeps losses by the requested submission owner, extracts both
decks from the initial visualization, and selects a deterministic holdout with
mandatory Alakazam and Mega Lucario matchups.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

SCHEMA = "ptcg-replay-matchups-v1"
RESULT_SCHEMA = "ptcg-replay-matchup-results-v1"
REQUIRED = ("Alakazam", "Mega Lucario")


def _initial_visualization(replay):
    for agent in replay.get("steps", [[]])[0]:
        visuals = agent.get("visualize") or []
        if visuals and isinstance(visuals[0], dict):
            return visuals[0]
    raise ValueError("replay has no initial visualization")


def _archetype(card_names):
    names = " ".join(card_names).lower()
    rules = (("mega lucario", "Mega Lucario"),
             ("alakazam", "Alakazam"),
             ("slowking", "Slowking"),
             ("typhlosion", "Ethan's Typhlosion"),
             ("team rocket", "Team Rocket"),
             ("dragapult", "Dragapult"),
             ("mega abomasnow", "Mega Abomasnow"))
    for needle, label in rules:
        if needle in names:
            return label
    pokemon = [n for n in card_names if "energy" not in n.lower()]
    return pokemon[0] if pokemon else "Unknown"


def parse_replay(path, owner):
    with open(path, encoding="utf-8") as source:
        raw = source.read()
    replay = json.loads(raw)
    agents = replay.get("info", {}).get("Agents", [])
    names = [a.get("Name") for a in agents]
    if owner not in names:
        raise ValueError(f"owner {owner!r} not found in replay agents")
    seat = names.index(owner)
    rewards = replay.get("rewards") or []
    statuses = replay.get("statuses") or []
    vis = _initial_visualization(replay)
    decks = vis.get("action")
    players = vis.get("current", {}).get("players") or []
    if not isinstance(decks, list) or len(decks) != 2:
        raise ValueError("initial visualization has no two-deck action")
    opp = 1 - seat
    cards = []
    if len(players) == 2:
        for zone in ("deck", "hand", "active", "bench", "prize"):
            for card in players[opp].get(zone) or []:
                if isinstance(card, dict) and card.get("name"):
                    cards.append(card["name"])
    reward = rewards[seat] if len(rewards) > seat else None
    status = statuses[seat] if len(statuses) > seat else "UNKNOWN"
    failure = "timeout" if status == "TIMEOUT" else (
        "fault" if status not in ("DONE", "ACTIVE") else
        "loss" if reward == -1 else "none")
    return {
        "episode_id": int(replay.get("info", {}).get("EpisodeId")),
        "source_file": os.path.basename(path),
        "source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "submission_owner": owner,
        "submission_seat": seat,
        "reward": reward,
        "status": status,
        "failure_mode": failure,
        "opponent": names[opp],
        "opponent_archetype": _archetype(sorted(set(cards))),
        "submission_deck": decks[seat],
        "opponent_deck": decks[opp],
    }


def build_fixture(paths, owner, submission_ref, version):
    losses = [parse_replay(path, owner) for path in sorted(paths)]
    losses = [r for r in losses if r["reward"] == -1 or r["failure_mode"] in ("fault", "timeout")]
    losses.sort(key=lambda r: (r["opponent_archetype"], r["episode_id"]))
    selected = []
    for required in REQUIRED:
        matches = [r for r in losses if r["opponent_archetype"] == required]
        if not matches:
            raise ValueError(f"required losing matchup missing: {required}")
        selected.append(matches[-1])
    seen = {r["episode_id"] for r in selected}
    selected.extend(r for r in losses if r["episode_id"] not in seen)
    return {
        "schema": SCHEMA, "fixture_version": version,
        "submission_ref": str(submission_ref), "submission_owner": owner,
        "required_archetypes": list(REQUIRED),
        "source_episode_ids": sorted(r["episode_id"] for r in losses),
        "matchups": selected,
    }


def run_fixture(fixture, seed, output):
    os.chdir(REPO)
    from agents import GreedyAgent
    from agents.rng import Rng
    from eval.bench import play_match
    from main import SubmissionAgent
    base = Rng(seed)
    records = []
    for matchup in fixture["matchups"]:
        for submission_seat in (0, 1):
            key = f"e{matchup['episode_id']}.s{submission_seat}"
            ours = SubmissionAgent(seed=base.child(key + ".submission").seed,
                                   deck=matchup["submission_deck"])
            opponent = GreedyAgent(seed=base.child(key + ".opponent").seed,
                                   deck=matchup["opponent_deck"])
            agents = (ours, opponent) if submission_seat == 0 else (opponent, ours)
            result, decisions, reject, exception = play_match(*agents)
            won = result in (0, 1) and result == submission_seat
            records.append({
                "episode_id": matchup["episode_id"],
                "archetype": matchup["opponent_archetype"],
                "submission_seat": submission_seat,
                "result": "win" if won else "loss" if result in (0, 1) else "draw" if result == 2 else "unfinished",
                "winner_seat": result if result in (0, 1) else None,
                "decisions": decisions, "fault": bool(reject or exception),
                "timeout": result == -1 and not reject and not exception,
            })
    payload = {
        "schema": RESULT_SCHEMA, "fixture_version": fixture["fixture_version"],
        "fixture_sha256": hashlib.sha256(json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "seed": seed, "created_at": datetime.now(timezone.utc).isoformat(),
        "matches": records,
        "summary": {"wins": sum(r["result"] == "win" for r in records),
                    "losses": sum(r["result"] == "loss" for r in records),
                    "faults": sum(r["fault"] for r in records),
                    "timeouts": sum(r["timeout"] for r in records)},
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    return payload


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("replays", nargs="+")
    build.add_argument("--owner", default="sota1111")
    build.add_argument("--submission-ref", required=True)
    build.add_argument("--version", required=True)
    build.add_argument("--output", required=True)
    run = sub.add_parser("run")
    run.add_argument("--fixture", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--output", required=True)
    args = p.parse_args(argv)
    if args.command == "build":
        paths = sorted({p for pattern in args.replays for p in glob.glob(pattern)})
        fixture = build_fixture(paths, args.owner, args.submission_ref, args.version)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(fixture, f, indent=2, ensure_ascii=False, sort_keys=True)
            f.write("\n")
        print(f"wrote {args.output}: {len(fixture['matchups'])} losing matchups")
    else:
        with open(args.fixture, encoding="utf-8") as f:
            fixture = json.load(f)
        result = run_fixture(fixture, args.seed, args.output)
        print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
