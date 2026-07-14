"""Search-parameter ablation driver (SOT-1673).

Quantifies the win-rate impact of each MCTS design parameter around the
SOT-1672 adopted configuration (docs/mcts-design.md §7), one factor at a
time:

- world-sample count N   (n_worlds=1 is PIMC-style single-world search),
- PUCT exploration constant (uct_c),
- rollout policy         (greedy | heuristic tier | random),
- leaf-evaluator feature weights (eval_weights -> HeuristicEvaluator).

Every variant is measured against MULTIPLE opponent axes — Random / Greedy /
MCTS(baseline config) — so a variant that merely exploits one opponent does
not look universally better (ByteRL, arXiv:2404.16689). Each (config,
opponent) cell runs `--shards` independent eval/bench.py shards in parallel
(side-alternating, per-shard seeds) and pools them with
eval/aggregate_shards.aggregate into one report with a Wilson 95% CI.

Seeds derive from --seed plus a CRC32 of the cell name, so a cell reruns
with the SAME seeds whether it is run alone (--only-config/--only-opponent)
or as part of the full sweep. Reproducibility is agent-side only: the
engine's internal RNG is not injectable (ASSUMPTIONS.md A-9), so win counts
vary slightly between runs even with identical seeds.

Full sweep (from the repo root; ~45-60 min on 24 cores):
    venv/bin/python eval/ablation.py --n 100 --shards 10 --seed 94000

One cell, reproducibly:
    venv/bin/python eval/ablation.py --n 100 --shards 10 --seed 94000 \
        --only-config n_worlds=1 --only-opponent greedy --force

Markdown results table (after the runs; used by docs/ablation.md):
    venv/bin/python eval/ablation.py --report
"""
import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
import zlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aggregate_shards import aggregate  # noqa: E402

# SOT-1672 adopted configuration (docs/mcts-design.md §7) — the ablation
# reference point. Unlisted fields keep their PlannerConfig defaults
# (uct_c=1.4, rollout="greedy", default evaluator weights).
BASELINE = {
    "max_root_actions": 6,
    "max_tree_depth": 1,
    "rollout_turns": 100,
    "rollout_depth": 200,
    "n_worlds": 4,
    "time_budget_s": 0.8,
    "deviate_margin": 0.1,
}

# (axis, cell name, PlannerConfig/eval_weights overrides on BASELINE).
# One factor at a time; "baseline" is the shared reference row.
CONFIGS = [
    ("baseline", "baseline", {}),
    ("n_worlds", "n_worlds=1", {"n_worlds": 1}),   # PIMC-style single world
    ("n_worlds", "n_worlds=2", {"n_worlds": 2}),
    ("n_worlds", "n_worlds=8", {"n_worlds": 8}),
    ("uct_c", "uct_c=0.35", {"uct_c": 0.35}),
    ("uct_c", "uct_c=0.7", {"uct_c": 0.7}),
    ("uct_c", "uct_c=2.8", {"uct_c": 2.8}),
    ("rollout", "rollout=heuristic", {"rollout": "heuristic"}),
    ("rollout", "rollout=random", {"rollout": "random"}),
    # prize-only value: every non-prize feature (incl. deck-out) zeroed
    ("eval_weights", "eval=prize_only",
     {"eval_weights": {"pokemon": 0.0, "energy": 0.0, "hp": 0.0,
                       "hand": 0.0, "deck_empty": 0.0}}),
    # material-heavy value: non-prize feature weights doubled
    ("eval_weights", "eval=material2x",
     {"eval_weights": {"pokemon": 0.6, "energy": 0.4, "hp": 0.008,
                       "hand": 0.12}}),
]

# Opponent axes. mcts_base plays the BASELINE config on the B side.
OPPONENTS = [
    ("random", None),
    ("greedy", None),
    ("mcts_base", BASELINE),
]

OUT_DIR = os.path.join(REPO, "eval", "results", "ablation")


def cell_slug(config_name: str, opponent: str) -> str:
    return re.sub(r"[^\w.=-]+", "_", f"{config_name}--vs-{opponent}")


def cell_seed(base_seed: int, config_name: str, opponent: str) -> int:
    """Deterministic per-cell seed, independent of sweep composition."""
    crc = zlib.crc32(f"{config_name}|{opponent}".encode())
    return base_seed + (crc % 100003) * 100


def submit_cell(config_name, overrides, opp_name, opp_config, args, pool):
    """Queue one cell's shards on the global pool. Returns (slug, futures)
    or None when the cell's aggregate already exists."""
    slug = cell_slug(config_name, opp_name)
    agg_path = os.path.join(OUT_DIR, f"{slug}.json")
    if os.path.exists(agg_path) and not args.force:
        print(f"SKIP {slug} (aggregate exists; --force to rerun)", flush=True)
        return None
    shard_dir = os.path.join(OUT_DIR, slug)
    os.makedirs(shard_dir, exist_ok=True)
    config_a = {**BASELINE, **overrides}
    agent_b = "mcts" if opp_name == "mcts_base" else opp_name
    per_shard, rem = divmod(args.n, args.shards)
    base = cell_seed(args.seed, config_name, opp_name)

    def one_shard(k):
        n_k = per_shard + (1 if k < rem else 0)
        if n_k == 0:
            return None
        out = os.path.join(shard_dir, f"shard_{k}.json")
        cmd = [sys.executable, os.path.join(REPO, "eval", "bench.py"),
               "--agent-a", "mcts", "--agent-b", agent_b,
               "--n", str(n_k), "--seed", str(base + k),
               "--config-a", json.dumps(config_a), "--json", out]
        if opp_config is not None:
            cmd += ["--config-b", json.dumps(opp_config)]
        log = subprocess.run(cmd, capture_output=True, text=True)
        if log.returncode != 0:
            raise RuntimeError(
                f"{slug} shard {k} failed:\n{log.stdout}\n{log.stderr}")
        return out

    return slug, [pool.submit(one_shard, k) for k in range(args.shards)]


def finalize_cell(config_name, overrides, opp_name, slug, futures, args):
    """Wait for one cell's shards and write the pooled aggregate."""
    shard_paths = [p for p in (f.result() for f in futures) if p]
    report = aggregate(shard_paths)
    report["cell"] = {
        "axis": next(a for a, name, _ in CONFIGS if name == config_name),
        "config": config_name, "overrides": overrides,
        "opponent": opp_name,
        "base_seed": cell_seed(args.seed, config_name, opp_name),
        "repro": (f"venv/bin/python eval/ablation.py --n {args.n} "
                  f"--shards {args.shards} --seed {args.seed} "
                  f"--only-config '{config_name}' "
                  f"--only-opponent {opp_name} --force"),
    }
    with open(os.path.join(OUT_DIR, f"{slug}.json"), "w") as f:
        json.dump(report, f, indent=1)
    lo, hi = report["wilson95_excl_draws"]
    print(f"DONE {slug}: n={report['n_matches']} "
          f"winrate={report['winrate_a_excl_draws']:.3f} "
          f"CI95=[{lo:.3f},{hi:.3f}]", flush=True)


def render_report() -> str:
    """Markdown table over all aggregates in eval/results/ablation/."""
    rows = []
    for _, name, _ in CONFIGS:
        for opp_name, _ in OPPONENTS:
            path = os.path.join(OUT_DIR, f"{cell_slug(name, opp_name)}.json")
            if os.path.exists(path):
                rows.append(json.load(open(path)))
    lines = [
        "| 軸 | 構成 | 対戦相手 | 試合数 | 勝率(引分除く) | 95%CI | "
        "reject/違反/fallback |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        cell = r["cell"]
        lo, hi = r["wilson95_excl_draws"]
        bad = (r["rejects"] + r["exceptions"] + r["budget_violations_a"]
               + r["planner_fallbacks_a"] + r["fallbacks_a"])
        lines.append(
            f"| {cell['axis']} | `{cell['config']}` | {cell['opponent']} "
            f"| {r['n_matches']} | {r['winrate_a_excl_draws']:.3f} "
            f"| [{lo:.3f}, {hi:.3f}] | {bad} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100,
                        help="matches per (config, opponent) cell")
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--seed", type=int, default=94000)
    parser.add_argument("--workers", type=int, default=16,
                        help="max concurrent bench.py shard processes")
    parser.add_argument("--only-config", default=None,
                        help="run only this config cell (e.g. n_worlds=1)")
    parser.add_argument("--only-opponent", default=None,
                        help="run only this opponent axis (e.g. greedy)")
    parser.add_argument("--force", action="store_true",
                        help="rerun cells whose aggregate already exists")
    parser.add_argument("--report", action="store_true",
                        help="print the pooled markdown table and exit")
    args = parser.parse_args()

    if args.report:
        print(render_report())
        return

    known = {name for _, name, _ in CONFIGS}
    if args.only_config and args.only_config not in known:
        raise SystemExit(f"unknown --only-config; choose from {sorted(known)}")
    os.makedirs(OUT_DIR, exist_ok=True)
    cells = [(a, name, ov, opp, opp_cfg)
             for a, name, ov in CONFIGS
             for opp, opp_cfg in OPPONENTS
             if (not args.only_config or name == args.only_config)
             and (not args.only_opponent or opp == args.only_opponent)]
    print(f"ABLATION: {len(cells)} cells x {args.n} matches "
          f"({args.shards} shards, {args.workers} workers, "
          f"seed {args.seed})", flush=True)
    # All shards of all cells go onto ONE global pool up front, so slow
    # cells (vs mcts_base) never leave workers idle at a cell boundary.
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        pending = []
        for _, name, overrides, opp_name, opp_config in cells:
            submitted = submit_cell(name, overrides, opp_name, opp_config,
                                    args, pool)
            if submitted:
                pending.append((name, overrides, opp_name) + submitted)
        for name, overrides, opp_name, slug, futures in pending:
            finalize_cell(name, overrides, opp_name, slug, futures, args)
    print("ABLATION SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
