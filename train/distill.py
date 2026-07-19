"""Teacher trajectory dataset and resumable policy/value distillation.

The format is deliberately JSONL/JSON and dependency-free so training artifacts
remain inspectable and usable in the offline competition environment.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from agents.policy_value import PolicyValueTransformer, _softmax
from agents.rng import Rng

DATASET_VERSION = "mcts-teacher/v1"
CHECKPOINT_VERSION = "distillation-checkpoint/v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_revision():
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass(frozen=True)
class TeacherExample:
    state: tuple
    actions: tuple
    visits: tuple
    value: float
    trajectory_id: str
    step: int

    def record(self):
        if not self.actions or len(self.actions) != len(self.visits):
            raise ValueError("actions and visits must be non-empty and aligned")
        if any(v < 0 for v in self.visits) or sum(self.visits) <= 0:
            raise ValueError("visits must be non-negative with positive total")
        if not -1.0 <= self.value <= 1.0:
            raise ValueError("value target must be in [-1, 1]")
        return {"schema": DATASET_VERSION, "trajectory": self.trajectory_id,
                "step": self.step, "state": list(self.state),
                "actions": [list(row) for row in self.actions],
                "visits": list(self.visits), "value": self.value}


class TeacherDatasetWriter:
    """Append complete trajectories; invalid/faulted games never become targets."""

    def __init__(self, path, provenance):
        self.path = path
        self.provenance = dict(provenance)

    def append_trajectory(self, examples, valid=True):
        if not valid:
            return 0
        rows = [example.record() for example in examples]
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True,
                                        separators=(",", ":")) + "\n")
        return len(rows)

    def write_manifest(self, path):
        payload = {"schema": DATASET_VERSION, "dataset": os.path.basename(self.path),
                   "dataset_sha256": sha256_file(self.path),
                   "code_revision": code_revision(), **self.provenance}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
        return payload


def load_dataset(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("schema") != DATASET_VERSION:
                raise ValueError(f"unsupported dataset schema at line {number}")
            example = TeacherExample(tuple(row["state"]),
                                     tuple(tuple(x) for x in row["actions"]),
                                     tuple(row["visits"]), float(row["value"]),
                                     str(row["trajectory"]), int(row["step"]))
            example.record()
            rows.append(example)
    if not rows:
        raise ValueError("teacher dataset is empty")
    return rows


def _loss(model, rows):
    policy_loss = value_loss = 0.0
    for row in rows:
        value, logits = model.predict(row.state, row.actions)
        predicted = _softmax(logits)
        total = sum(row.visits)
        target = [v / total for v in row.visits]
        policy_loss -= sum(t * math.log(max(1e-12, p))
                           for t, p in zip(target, predicted))
        value_loss += (value - row.value) ** 2
    n = len(rows)
    return {"policy_cross_entropy": policy_loss / n,
            "value_mse": value_loss / n,
            "total": (policy_loss + value_loss) / n}


def train(dataset, out, checkpoint, epochs=3, learning_rate=0.05,
          hidden_size=16, seed=20260719, config=None):
    rows = load_dataset(dataset)
    state_size, action_size = len(rows[0].state), len(rows[0].actions[0])
    if any(len(x.state) != state_size or
           any(len(a) != action_size for a in x.actions) for x in rows):
        raise ValueError("dataset contains inconsistent feature shapes")
    config = dict(config or {})
    expected = {"dataset_sha256": sha256_file(dataset), "state_size": state_size,
                "action_size": action_size, "hidden_size": hidden_size,
                "seed": seed, "learning_rate": learning_rate, "config": config}
    start = 0
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint, encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved.get("version") != CHECKPOINT_VERSION or any(
                saved.get(k) != v for k, v in expected.items()):
            raise ValueError("checkpoint provenance/config does not match this run")
        model = PolicyValueTransformer(state_size, action_size, hidden_size,
                                       weights=saved["weights"])
        start = int(saved["epoch"])
    else:
        model = PolicyValueTransformer(state_size, action_size, hidden_size,
                                       seed=seed)
    before = _loss(model, rows)
    for epoch in range(start, epochs):
        ordered = list(rows)
        Rng(seed).child(f"distill-epoch-{epoch}").shuffle(ordered)
        # Train the direct legal-action head and value bias. This dependency-free
        # baseline preserves the Transformer inference format and learns both
        # teacher visit policy and terminal value targets.
        for row in ordered:
            value, logits = model.predict(row.state, row.actions)
            probs = _softmax(logits)
            target = [v / sum(row.visits) for v in row.visits]
            model.weights["value_bias"] -= learning_rate * (value-row.value) * (1-value*value)
            for j in range(action_size):
                gradient = sum((p-t) * action[j] for p, t, action in
                               zip(probs, target, row.actions))
                model.weights["policy_feature_head"][j] -= learning_rate * gradient
        if checkpoint:
            payload = {"version": CHECKPOINT_VERSION, **expected,
                       "epoch": epoch + 1, "weights": model.weights,
                       "code_revision": code_revision()}
            with open(checkpoint + ".tmp", "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            os.replace(checkpoint + ".tmp", checkpoint)
    model.save(out)
    manifest = {"schema": "distillation-run/v1", **expected,
                "code_revision": code_revision(), "epochs": epochs,
                "model_sha256": sha256_file(out), "loss_before": before,
                "loss_after": _loss(model, rows)}
    with open(out + ".manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.out, args.checkpoint,
                           args.epochs, args.learning_rate,
                           args.hidden_size, args.seed), sort_keys=True))


if __name__ == "__main__":
    main()
