"""Pure-python value trainer — logistic regression (SOT-1674) and MLP value
networks (SOT-1679, arXiv:1808.04794 §IV-B).

Fits a win-probability predictor on train/gen_selfplay.py logs. No
third-party deps (the repo is pip-free; numpy was considered for the MLP and
rejected — see docs/value-net.md): --arch selects

- "linear" (default): sigmoid(w · standardize(x) + b), plain SGD (SOT-1674
  behavior, unchanged);
- hyphen-separated hidden sizes, e.g. "256-128-64" (the paper's net) or
  "64-32": an MLP with tanh hidden layers and a sigmoid output, trained with
  minibatch ADAM + BCE. Training is deterministic given (--seed, data): all
  shuffles come from per-epoch Rng children, so a checkpointed run resumes
  to the same model a single run produces (--checkpoint, saved per epoch —
  long pure-python runs survive interruption).

The split is BY MATCH (crc32 of file:match-id), never by example — states
from one match are correlated, so an example-level split would leak. The
holdout report includes the heuristic evaluator's log-loss/accuracy on the
SAME states (logged as `h` at generation time), which is the like-for-like
predictive comparison quoted in docs/learned-value.md.

Usage (from the repo root):
    venv/bin/python train/train_value.py train/logs/*.jsonl \
        --out train/value_model.json [--epochs 5] [--seed 61674]
    venv/bin/python train/train_value.py train/logs/cheater_shard_*.jsonl \
        --features v2 --arch 64-32 --out train/value_net.json \
        [--checkpoint train/logs/value_net.ckpt.json]
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


# --- MLP value network (SOT-1679) -------------------------------------------

def parse_arch(spec: str):
    """"linear" -> None; "64-32" -> [64, 32] hidden sizes."""
    if spec == "linear":
        return None
    try:
        hidden = [int(part) for part in spec.split("-")]
    except ValueError:
        hidden = []
    if not hidden or any(h < 1 for h in hidden):
        raise SystemExit(f"bad --arch {spec!r} (use 'linear' or e.g. '64-32')")
    return hidden


def init_layers(k: int, hidden, seed: int):
    """Xavier-uniform tanh MLP init: [(w rows, b), ...], last layer 1 logit."""
    rng = Rng(seed).child("init")
    layers = []
    fan_in = k
    for li, fan_out in enumerate(list(hidden) + [1]):
        s = math.sqrt(6.0 / (fan_in + fan_out))
        w = [[(rng.random() * 2 - 1) * s for _ in range(fan_in)]
             for _ in range(fan_out)]
        layers.append((w, [0.0] * fan_out))
        fan_in = fan_out
    return layers


def mlp_z(layers, x) -> float:
    """Forward pass -> output logit (x already standardized)."""
    h = x
    for w, b in layers[:-1]:
        h = [math.tanh(bj + sum(wj * hj for wj, hj in zip(row, h)))
             for row, bj in zip(w, b)]
    w, b = layers[-1]
    return b[0] + sum(wj * hj for wj, hj in zip(w[0], h))


def _zeros_like(layers):
    return [([[0.0] * len(row) for row in w], [0.0] * len(b))
            for w, b in layers]


class _Adam:
    """Standard ADAM over the layer structure (paper: ADAM + BCE)."""

    def __init__(self, layers, lr, l2, state=None):
        self.lr, self.l2 = lr, l2
        self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8
        self.t = state["t"] if state else 0
        self.m = state["m"] if state else _zeros_like(layers)
        self.v = state["v"] if state else _zeros_like(layers)

    def step(self, layers, grads, batch_n):
        self.t += 1
        b1, b2, eps, lr, l2 = self.b1, self.b2, self.eps, self.lr, self.l2
        c1 = 1 - b1 ** self.t
        c2 = 1 - b2 ** self.t
        inv_n = 1.0 / batch_n
        for (w, b), (gw, gb), (mw, mb), (vw, vb) in zip(
                layers, grads, self.m, self.v):
            for row, grow, mrow, vrow in zip(w, gw, mw, vw):
                for i in range(len(row)):
                    g = grow[i] * inv_n + l2 * row[i]
                    mrow[i] = m = b1 * mrow[i] + (1 - b1) * g
                    vrow[i] = v = b2 * vrow[i] + (1 - b2) * g * g
                    row[i] -= lr * (m / c1) / (math.sqrt(v / c2) + eps)
            for i in range(len(b)):
                g = gb[i] * inv_n
                mb[i] = m = b1 * mb[i] + (1 - b1) * g
                vb[i] = v = b2 * vb[i] + (1 - b2) * g * g
                b[i] -= lr * (m / c1) / (math.sqrt(v / c2) + eps)

    def state(self) -> dict:
        return {"t": self.t, "m": self.m, "v": self.v}


def _ckpt_config(arch, seed, lr, l2, batch, n_train, k) -> dict:
    return {"arch": list(arch), "seed": seed, "lr": lr, "l2": l2,
            "batch": batch, "n_train": n_train, "k": k}


def fit_mlp(train, mean, std, arch, epochs, lr, l2, seed, batch=64,
            checkpoint=None):
    """Minibatch-ADAM tanh MLP on standardized examples.

    Deterministic in (seed, data): epoch e shuffles with
    Rng(seed).child(f"epoch{e}"), so resuming from an epoch-boundary
    checkpoint reproduces exactly what an uninterrupted run produces.
    """
    k = len(mean)
    for _, _, _, x in train:  # standardize once, in place
        for j in range(k):
            x[j] = (x[j] - mean[j]) / std[j]

    config = _ckpt_config(arch, seed, lr, l2, batch, len(train), k)
    start_epoch = 0
    adam_state = None
    layers = None
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint) as f:
            ck = json.load(f)
        if ck.get("config") != config:
            raise SystemExit(f"checkpoint {checkpoint} was written by a "
                             f"different run config; delete it or change "
                             f"--checkpoint")
        start_epoch = ck["epoch"]
        layers = [(w, b) for w, b in ck["layers"]]
        adam_state = ck["adam"]
        print(f"resumed epoch {start_epoch}/{epochs} from {checkpoint}",
              flush=True)
    if layers is None:
        layers = init_layers(k, arch, seed)
    adam = _Adam(layers, lr, l2, state=adam_state)

    n_hidden = len(layers) - 1
    for epoch in range(start_epoch, epochs):
        # Epoch order is a fresh permutation of the ORIGINAL example order
        # (never of the previous epoch's order), so a resumed run sees the
        # exact minibatch sequence an uninterrupted run would.
        order = list(range(len(train)))
        Rng(seed).child(f"epoch{epoch}").shuffle(order)
        loss_sum = correct = 0.0
        eps = 1e-9
        for lo in range(0, len(train), batch):
            chunk = [train[i] for i in order[lo:lo + batch]]
            grads = _zeros_like(layers)
            for y, _, _, x in chunk:
                # forward, keeping activations for backprop
                acts = [x]
                h = x
                for w, b in layers[:-1]:
                    h = [math.tanh(bj + sum(wj * hj
                                            for wj, hj in zip(row, h)))
                         for row, bj in zip(w, b)]
                    acts.append(h)
                w, b = layers[-1]
                z = b[0] + sum(wj * hj for wj, hj in zip(w[0], h))
                p = sigmoid(z)
                pc = min(1.0 - eps, max(eps, p))
                loss_sum += -(y * math.log(pc) + (1 - y) * math.log(1 - pc))
                correct += float((p >= 0.5) == (y >= 0.5))
                # backward: delta at the output logit is (p - y) for BCE
                delta = [p - y]
                for li in range(n_hidden, -1, -1):
                    gw, gb = grads[li]
                    h_prev = acts[li]
                    for j, d in enumerate(delta):
                        if d == 0.0:
                            continue
                        gb[j] += d
                        grow = gw[j]
                        for i, hi in enumerate(h_prev):
                            grow[i] += d * hi
                    if li == 0:
                        break
                    w_l = layers[li][0]
                    h_l = acts[li]
                    delta = [(1.0 - hj * hj)
                             * sum(delta[j] * w_l[j][i]
                                   for j in range(len(delta)))
                             for i, hj in enumerate(h_l)]
            adam.step(layers, grads, len(chunk))
        print(f"  epoch {epoch + 1}/{epochs}: train logloss "
              f"{loss_sum / len(train):.4f} acc {correct / len(train):.4f}",
              flush=True)
        if checkpoint:
            tmp = checkpoint + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"config": config, "epoch": epoch + 1,
                           "layers": layers, "adam": adam.state()}, f)
            os.replace(tmp, checkpoint)
    return layers


def evaluate_holdout(holdout, mean, std, w=None, b=None, layers=None) -> dict:
    learned, heuristic, by_bucket = [], [], {}
    for y, h, t, x in holdout:
        if layers is not None:
            z = mlp_z(layers, [(xj - mj) / sj
                               for xj, mj, sj in zip(x, mean, std)])
        else:
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
    parser.add_argument("--out", default=None,
                        help="model JSON path (default: train/value_model.json"
                             " for --arch linear, train/value_net.json for an"
                             " MLP)")
    parser.add_argument("--arch", default="linear",
                        help="'linear' (SOT-1674) or hyphen-separated hidden "
                             "sizes, e.g. '256-128-64' / '64-32' (SOT-1679)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=None,
                        help="default: 0.05 (linear SGD) / 0.001 (MLP ADAM)")
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--batch", type=int, default=64,
                        help="MLP minibatch size")
    parser.add_argument("--checkpoint", default=None,
                        help="MLP: save/resume per-epoch checkpoint here")
    parser.add_argument("--max-examples", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=61674)
    parser.add_argument("--features", default="v1", choices=("v1", "v2"),
                        help="feature set the logs were generated with "
                             "(gen_selfplay.py --features)")
    args = parser.parse_args()

    arch = parse_arch(args.arch)
    lr = args.lr if args.lr is not None else (0.05 if arch is None else 0.001)
    out = args.out or os.path.join(
        REPO, "train", "value_model.json" if arch is None else "value_net.json")

    feature_names, _ = make_featurizer(args.features)
    train, holdout = load_examples(args.logs, args.max_examples, args.seed)
    if not train or not holdout:
        raise SystemExit(f"not enough data (train={len(train)}, "
                         f"holdout={len(holdout)})")
    print(f"loaded train={len(train)} holdout={len(holdout)} "
          f"features={len(feature_names)} ({args.features}) arch={args.arch}",
          flush=True)
    if len(train[0][3]) != len(feature_names):
        raise SystemExit("log feature length does not match agents/features.py "
                         f"feature set {args.features!r} — regenerate the logs")

    mean, std = standardizer(train)
    if arch is None:
        w, b = fit(train, mean, std, args.epochs, lr, args.l2, args.seed)
        metrics = evaluate_holdout(holdout, mean, std, w, b)
        params = {"weights": [round(x, 6) for x in w], "bias": round(b, 6)}
        issue = "SOT-1674"
    else:
        layers = fit_mlp(train, mean, std, arch, args.epochs, lr, args.l2,
                         args.seed, batch=args.batch,
                         checkpoint=args.checkpoint)
        metrics = evaluate_holdout(holdout, mean, std, layers=layers)
        params = {"arch": list(arch),
                  "layers": [{"w": [[round(v, 6) for v in row] for row in w],
                              "b": [round(v, 6) for v in b]}
                             for w, b in layers]}
        issue = "SOT-1679"

    model = {
        "feature_set": args.features,
        "feature_names": list(feature_names),
        **params,
        "mean": [round(x, 6) for x in mean],
        "std": [round(x, 6) for x in std],
        "meta": {
            "issue": issue,
            "n_train": len(train),
            "seed": args.seed, "epochs": args.epochs,
            "lr": lr, "l2": args.l2,
            **({"arch": args.arch, "batch": args.batch}
               if arch is not None else {}),
            "logs": [os.path.basename(p) for p in args.logs],
            **metrics,
        },
    }
    with open(out, "w") as f:
        json.dump(model, f, indent=1)
    print(json.dumps(metrics, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
