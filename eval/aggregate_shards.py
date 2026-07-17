"""Pool sharded eval/bench.py reports into one aggregate report (SOT-1672).

Long benches are run as independent shards (same agents/config, distinct
seeds) in parallel; this pools their counts and recomputes the Wilson 95%
interval on the pooled win rate. Counters that must be zero across the whole
run (rejects, exceptions, budget violations, fallbacks) are summed so the
aggregate is what the acceptance criteria are checked against.

Usage:
    venv/bin/python eval/aggregate_shards.py out.json shard1.json shard2.json ...
"""
import json
import math
import sys

SUMMED = (
    "n_matches", "wins_a", "wins_b", "draws", "unfinished",
    "rejects", "exceptions", "fallbacks_a", "fallbacks_b", "decisions",
    "budget_violations_a", "budget_violations_b",
    "planner_fallbacks_a", "planner_fallbacks_b",
    "degraded_count_a", "degraded_count_b",
    "rollout_cutoffs_a", "rollout_cutoffs_b",
)


def wilson95(wins: int, n: int) -> list:
    if n == 0:
        return [None, None]
    z = 1.959963984540054
    p = wins / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - half, center + half]


def aggregate(shard_paths: list) -> dict:
    shards = [json.load(open(p)) for p in shard_paths]
    first = shards[0]
    for key in ("agent_a", "agent_b", "deck", "config_a", "config_b"):
        values = {json.dumps(s.get(key), sort_keys=True) for s in shards}
        if len(values) != 1:
            raise SystemExit(f"shards disagree on {key}: {values}")
    out = {key: first.get(key) for key in
           ("agent_a", "agent_b", "deck", "config_a", "config_b")}
    out["shards"] = [{"seed": s["seed"], "n_matches": s["n_matches"],
                      "wins_a": s["wins_a"], "wins_b": s["wins_b"],
                      "draws": s["draws"]} for s in shards]
    for key in SUMMED:
        out[key] = sum(s.get(key, 0) for s in shards)
    out["planner_move_max_ms"] = max(
        s.get("planner_move_max_ms", 0) for s in shards)
    decided = out["wins_a"] + out["wins_b"]
    out["winrate_a_excl_draws"] = (out["wins_a"] / decided) if decided else None
    out["wilson95_excl_draws"] = wilson95(out["wins_a"], decided)
    n = out["n_matches"]
    out["winrate_a_draws_half"] = ((out["wins_a"] + 0.5 * out["draws"]) / n
                                   if n else None)
    return out


def main():
    # SOT-1708: an optional leading --kpi[=ISSUE] appends a KPI record.
    argv = sys.argv[1:]
    kpi = next((a for a in argv if a.startswith("--kpi")), None)
    argv = [a for a in argv if not a.startswith("--kpi")]
    if len(argv) < 2:
        raise SystemExit(__doc__)
    out_path, shard_paths = argv[0], argv[1:]
    report = aggregate(shard_paths)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    lo, hi = report["wilson95_excl_draws"]
    print(f"POOLED: {report['agent_a']} vs {report['agent_b']} "
          f"n={report['n_matches']} ({len(shard_paths)} shards)\n"
          f"  A {report['wins_a']} - B {report['wins_b']} "
          f"draws {report['draws']} unfinished {report['unfinished']}\n"
          f"  win rate A (excl. draws): {report['winrate_a_excl_draws']:.4f}"
          f"  Wilson95 [{lo:.4f}, {hi:.4f}]\n"
          f"  rejects: {report['rejects']}  exceptions: {report['exceptions']}"
          f"  budget violations: A={report['budget_violations_a']}"
          f" B={report['budget_violations_b']}"
          f"  planner fallbacks: A={report['planner_fallbacks_a']}"
          f"  degraded: A={report['degraded_count_a']}\n"
          f"  planner move max: {report['planner_move_max_ms']:.1f} ms\n"
          f"wrote {out_path}")
    if kpi is not None:
        from eval.kpi import append_history, record_from_bench
        issue = kpi.split("=", 1)[1] if "=" in kpi else None
        print("KPI record appended to "
              f"{append_history(record_from_bench(report, issue=issue))}")


if __name__ == "__main__":
    main()
