"""Pure-python logistic-regression value trainer (SOT-1674).

Fits sigmoid(w · standardize(x) + b) on train/gen_selfplay.py logs to
predict the match winner from a mid-game state. No third-party deps (the
repo is pip-free); plain SGD over ~30 features is fast enough.

The split is BY MATCH (crc32 of file:match-id), never by example — states
from one match are correlated, so an example-level split would leak. The
holdout report includes the heuristic evaluator's log-loss/accuracy on the
SAME states (logged as `h` at generation time), which is the like-for-like
predictive comparison quoted in docs/learned-value.md.

Usage (from the repo root):
    venv/bin/python train/train_value.py train/logs/*.jsonl \
        --out train/value_model.json [--epochs 5] [--seed 61674]
"""
import argparse
import json
import math
import os
import sys
import zlib
from array import array

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from agents.features import make_featurizer
from agents.rng import Rng

HOLDOUT_BUCKETS = (8, 9)  # crc32 % 10: 80/20 match-level split


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def logloss_acc(pairs) -> tuple:
    """[(y, p)] -> (mean log-loss, accuracy)."""
    eps = 1e-9
    loss = correct = 0.0
    for y, p in pairs:
        p = min(1.0 - eps, max(eps, p))
        loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        correct += float((p >= 0.5) == (y >= 0.5))
    n = max(1, len(pairs))
    return loss / n, correct / n


def load_examples(paths, max_examples: int, seed: int):
    train, holdout = [], []
    for path in paths:
        tag = os.path.basename(path)
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                bucket = zlib.crc32(f"{tag}:{rec['m']}".encode()) % 10
                row = (rec["y"], rec["h"], rec["t"], array("d", rec["x"]))
                (holdout if bucket in HOLDOUT_BUCKETS else train).append(row)
    if len(train) > max_examples:  # deterministic thinning, keeps match mix
        Rng(seed).child("thin").shuffle(train)
        train = train[:max_examples]
    return train, holdout


def standardizer(train):
    k = len(train[0][3])
    mean = [0.0] * k
    for _, _, _, x in train:
        for j in range(k):
            mean[j] += x[j]
    mean = [m / len(train) for m in mean]
    var = [0.0] * k
    for _, _, _, x in train:
        for j in range(k):
            d = x[j] - mean[j]
            var[j] += d * d
    std = [math.sqrt(v / len(train)) or 1.0 for v in var]
    return mean, std


def fit(train, mean, std, epochs: int, lr0: float, l2: float, seed: int):
    k = len(mean)
    for _, _, _, x in train:  # standardize once, in place
        for j in range(k):
            x[j] = (x[j] - mean[j]) / std[j]
    w = [0.0] * k
    base_rate = sum(y for y, _, _, _ in train) / len(train)
    b = math.log(max(1e-9, base_rate) / max(1e-9, 1 - base_rate))
    rng = Rng(seed).child("sgd")
    step = 0
    for epoch in range(epochs):
        rng.shuffle(train)
        for y, _, _, x in train:
            step += 1
            lr = lr0 / (1.0 + step / (4.0 * len(train)))
            g = sigmoid(b + sum(wj * xj for wj, xj in zip(w, x))) - y
            b -= lr * g
            for j in range(k):
                w[j] -= lr * (g * x[j] + l2 * w[j])
        pairs = [(y, sigmoid(b + sum(wj * xj for wj, xj in zip(w, x))))
                 for y, _, _, x in train]
        loss, acc = logloss_acc(pairs)
        print(f"  epoch {epoch + 1}/{epochs}: train logloss {loss:.4f} "
              f"acc {acc:.4f}", flush=True)
    return w, b


def evaluate_holdout(holdout, mean, std, w, b) -> dict:
    learned, heuristic, by_bucket = [], [], {}
    for y, h, t, x in holdout:
        z = b + sum(wj * (xj - mj) / sj
                    for wj, xj, mj, sj in zip(w, x, mean, std))
        p = sigmoid(z)
        learned.append((y, p))
        heuristic.append((y, h))
        bucket = min(int(t) // 5 * 5, 20)
        by_bucket.setdefault(bucket, []).append((y, p, h))
    ll_l, acc_l = logloss_acc(learned)
    ll_h, acc_h = logloss_acc(heuristic)
    base = sum(y for y, _ in learned) / max(1, len(learned))
    turn_table = {}
    for bucket in sorted(by_bucket):
        rows = by_bucket[bucket]
        _, acc_lb = logloss_acc([(y, p) for y, p, _ in rows])
        _, acc_hb = logloss_acc([(y, h) for y, _, h in rows])
        turn_table[f"turn_{bucket:02d}+"] = {
            "n": len(rows), "acc_learned": round(acc_lb, 4),
            "acc_heuristic": round(acc_hb, 4)}
    return {
        "n_holdout": len(holdout), "holdout_base_rate": round(base, 4),
        "logloss_learned": round(ll_l, 4), "acc_learned": round(acc_l, 4),
        "logloss_heuristic": round(ll_h, 4), "acc_heuristic": round(acc_h, 4),
        "acc_by_turn_bucket": turn_table,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="gen_selfplay.py JSONL files")
    parser.add_argument("--out", default=os.path.join(REPO, "train", "value_model.json"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--max-examples", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=61674)
    parser.add_argument("--features", default="v1", choices=("v1", "v2"),
                        help="feature set the logs were generated with "
                             "(gen_selfplay.py --features)")
    args = parser.parse_args()

    feature_names, _ = make_featurizer(args.features)
    train, holdout = load_examples(args.logs, args.max_examples, args.seed)
    if not train or not holdout:
        raise SystemExit(f"not enough data (train={len(train)}, "
                         f"holdout={len(holdout)})")
    print(f"loaded train={len(train)} holdout={len(holdout)} "
          f"features={len(feature_names)} ({args.features})", flush=True)
    if len(train[0][3]) != len(feature_names):
        raise SystemExit("log feature length does not match agents/features.py "
                         f"feature set {args.features!r} — regenerate the logs")

    mean, std = standardizer(train)
    w, b = fit(train, mean, std, args.epochs, args.lr, args.l2, args.seed)
    metrics = evaluate_holdout(holdout, mean, std, w, b)

    model = {
        "feature_set": args.features,
        "feature_names": list(feature_names),
        "weights": [round(x, 6) for x in w],
        "bias": round(b, 6),
        "mean": [round(x, 6) for x in mean],
        "std": [round(x, 6) for x in std],
        "meta": {
            "issue": "SOT-1674",
            "n_train": len(train),
            "seed": args.seed, "epochs": args.epochs,
            "lr": args.lr, "l2": args.l2,
            "logs": [os.path.basename(p) for p in args.logs],
            **metrics,
        },
    }
    with open(args.out, "w") as f:
        json.dump(model, f, indent=1)
    print(json.dumps(metrics, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
