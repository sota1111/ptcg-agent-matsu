"""Champion/candidate/history league aggregation and CI promotion gate."""
import argparse
import json
import math


def wilson_interval(wins, draws, games, z=1.959963984540054):
    if games <= 0:
        return (0.0, 1.0)
    p = (wins + 0.5 * draws) / games
    d = 1.0 + z * z / games
    center = (p + z * z / (2 * games)) / d
    margin = z * math.sqrt((p * (1-p) + z*z/(4*games))/games) / d
    return max(0.0, center-margin), min(1.0, center+margin)


def aggregate(matches):
    opponents = {}
    totals = {"games": 0, "wins": 0, "draws": 0, "losses": 0,
              "faults": 0, "timeouts": 0, "latency_ms": []}
    for match in matches:
        if match.get("candidate") != "candidate":
            raise ValueError("league rows must be from candidate perspective")
        opponent = match["opponent"]
        if opponent != "champion" and not opponent.startswith("history/"):
            raise ValueError("opponent must be champion or history/<id>")
        bucket = opponents.setdefault(opponent, {"games": 0, "score": 0.0})
        result = match["result"]
        if result not in ("win", "draw", "loss"):
            raise ValueError("invalid match result")
        totals["games"] += 1; bucket["games"] += 1
        totals[result + ("es" if result == "loss" else "s")] += 1
        score = 1.0 if result == "win" else 0.5 if result == "draw" else 0.0
        bucket["score"] += score
        totals["faults"] += int(bool(match.get("fault")))
        totals["timeouts"] += int(bool(match.get("timeout")))
        totals["latency_ms"].append(float(match.get("latency_ms", 0.0)))
    if not totals["games"]:
        raise ValueError("league has no matches")
    low, high = wilson_interval(totals["wins"], totals["draws"], totals["games"])
    latencies = sorted(totals.pop("latency_ms"))
    totals.update({"win_rate": (totals["wins"] + .5*totals["draws"])/totals["games"],
                   "ci95": [low, high], "latency_mean_ms": sum(latencies)/len(latencies),
                   "latency_p95_ms": latencies[min(len(latencies)-1, math.ceil(.95*len(latencies))-1)],
                   "opponents": opponents})
    return totals


def promotion_decision(candidate, champion, thresholds=None):
    limits = {"min_ci_lower": 0.5, "max_faults": 0, "max_timeouts": 0,
              "max_latency_ratio": 1.1, "required_opponents": ["champion"]}
    limits.update(thresholds or {})
    reasons = []
    if candidate["ci95"][0] <= limits["min_ci_lower"]:
        reasons.append("win-rate CI lower bound is not above threshold")
    if candidate["faults"] > limits["max_faults"]:
        reasons.append("candidate fault limit exceeded")
    if candidate["timeouts"] > limits["max_timeouts"]:
        reasons.append("candidate timeout limit exceeded")
    if candidate["latency_mean_ms"] > champion["latency_mean_ms"] * limits["max_latency_ratio"]:
        reasons.append("candidate latency ratio exceeded")
    missing = [x for x in limits["required_opponents"] if x not in candidate["opponents"]]
    if missing:
        reasons.append("required league opponents missing: " + ", ".join(missing))
    return {"schema": "champion-promotion/v1", "promote": not reasons,
            "reasons": reasons, "thresholds": limits,
            "candidate": candidate, "champion": champion}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matches")
    parser.add_argument("--champion-metrics", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.matches, encoding="utf-8") as handle:
        candidate = aggregate(json.loads(line) for line in handle if line.strip())
    with open(args.champion_metrics, encoding="utf-8") as handle:
        champion = json.load(handle)
    decision = promotion_decision(candidate, champion)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(decision, handle, sort_keys=True, indent=2)
    print("PROMOTE" if decision["promote"] else "REJECT")
    raise SystemExit(0 if decision["promote"] else 1)


if __name__ == "__main__":
    main()
