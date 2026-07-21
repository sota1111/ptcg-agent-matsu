"""Small dependency-free Transformer policy/value model for Search API MCTS.

The model is intentionally an inference contract, not a trainer: weights are
seeded deterministically for smoke tests and can be replaced by trained JSON
weights without changing the agent or the engine-facing action mapping.
"""
from dataclasses import dataclass
import json
import math

from .base import BaseAgent
from .planner import EngineBackend, sample_fills
from .rng import Rng
from .search_encoding import encode_actions, encode_state


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _softmax(values):
    if not values:
        return []
    top = max(values)
    exps = [math.exp(max(-60.0, min(60.0, x - top))) for x in values]
    total = sum(exps) or 1.0
    return [x / total for x in exps]


class PolicyValueTransformer:
    """Single self-attention block with state value and legal-action heads."""

    VERSION = "policy-value-transformer/v1"

    def __init__(self, state_size, action_size, hidden_size=16, seed=0,
                 weights=None):
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.hidden_size = int(hidden_size)
        if min(self.state_size, self.action_size, self.hidden_size) <= 0:
            raise ValueError("model dimensions must be positive")
        self.weights = weights or self._initialize(seed)
        self._validate()

    def _initialize(self, seed):
        rng = Rng(seed).child("policy-value-transformer")
        def matrix(rows, cols, scale=None):
            scale = scale or 1.0 / math.sqrt(max(1, cols))
            return [[(rng.random() * 2.0 - 1.0) * scale
                     for _ in range(cols)] for _ in range(rows)]
        h = self.hidden_size
        return {
            "state": matrix(h, self.state_size),
            "action": matrix(h, self.action_size),
            "query": matrix(h, h), "key": matrix(h, h),
            "value": matrix(h, h), "output": matrix(h, h),
            "policy": matrix(h, h),
            "policy_feature_head": [0.0] * self.action_size,
            "value_head": matrix(1, h)[0],
            "value_bias": 0.0, "policy_bias": 0.0,
        }

    def _validate(self):
        expected = {
            "state": (self.hidden_size, self.state_size),
            "action": (self.hidden_size, self.action_size),
            "query": (self.hidden_size, self.hidden_size),
            "key": (self.hidden_size, self.hidden_size),
            "value": (self.hidden_size, self.hidden_size),
            "output": (self.hidden_size, self.hidden_size),
            "policy": (self.hidden_size, self.hidden_size),
        }
        for name, (rows, cols) in expected.items():
            value = self.weights.get(name)
            if not isinstance(value, list) or len(value) != rows or any(
                    not isinstance(row, list) or len(row) != cols
                    for row in value):
                raise ValueError(f"invalid {name} weight shape")
        if len(self.weights.get("value_head", ())) != self.hidden_size:
            raise ValueError("invalid value_head weight shape")
        # Older exported v1 models predate distillation. A zero head preserves
        # their output exactly while making legal-action features trainable.
        self.weights.setdefault("policy_feature_head", [0.0] * self.action_size)
        if len(self.weights["policy_feature_head"]) != self.action_size:
            raise ValueError("invalid policy_feature_head weight shape")

    @staticmethod
    def _project(matrix, vector):
        return [math.tanh(_dot(row, vector)) for row in matrix]

    def predict(self, state, legal_actions):
        state = list(state)
        actions = [list(row) for row in legal_actions]
        if len(state) != self.state_size:
            raise ValueError("state vector shape does not match model")
        if any(len(row) != self.action_size for row in actions):
            raise ValueError("action vector shape does not match model")
        tokens = [self._project(self.weights["state"], state)]
        tokens += [self._project(self.weights["action"], row)
                   for row in actions]
        queries = [self._project(self.weights["query"], x) for x in tokens]
        keys = [self._project(self.weights["key"], x) for x in tokens]
        values = [self._project(self.weights["value"], x) for x in tokens]
        attended = []
        scale = math.sqrt(self.hidden_size)
        for query in queries:
            alpha = _softmax([_dot(query, key) / scale for key in keys])
            mixed = [sum(a * value[j] for a, value in zip(alpha, values))
                     for j in range(self.hidden_size)]
            attended.append(self._project(self.weights["output"], mixed))
        state_token = attended[0]
        value = math.tanh(_dot(self.weights["value_head"], state_token)
                          + float(self.weights.get("value_bias", 0.0)))
        policy_state = self._project(self.weights["policy"], state_token)
        logits = [_dot(policy_state, token) + _dot(
                      self.weights["policy_feature_head"], action)
                  + float(self.weights.get("policy_bias", 0.0))
                  for token, action in zip(attended[1:], actions)]
        return value, logits

    def save(self, path):
        payload = {"version": self.VERSION, "state_size": self.state_size,
                   "action_size": self.action_size,
                   "hidden_size": self.hidden_size,
                   "weights": self.weights}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("version") != cls.VERSION:
            raise ValueError("unsupported policy/value model version")
        return cls(payload["state_size"], payload["action_size"],
                   payload["hidden_size"], weights=payload["weights"])


@dataclass
class PolicyMctsConfig:
    simulations: int = 32
    exploration: float = 1.4
    seed: int = 0


class PolicyMctsSearch:
    """Root PUCT search that creates every rollout through begin/step/end."""

    def __init__(self, model, backend=None, config=None):
        self.model = model
        self.backend = backend or EngineBackend()
        self.config = config or PolicyMctsConfig()
        self.last_visits = ()
        self.last_totals = ()

    @staticmethod
    def _terminal_value(obs, root_player):
        current = getattr(obs, "current", None)
        result = getattr(current, "result", -1)
        if result == -1:
            return None
        if result == root_player:
            return 1.0
        if result == 1 - root_player:
            return -1.0
        return 0.0

    def choose(self, raw_obs, fills):
        encoded = encode_actions(raw_obs.get("select") or {})
        if not encoded.option_indices:
            return []
        state = encode_state(raw_obs,
                             (raw_obs.get("current") or {}).get("yourIndex", 0))
        _, logits = self.model.predict(state, encoded.features)
        priors = _softmax(logits)
        visits = [0] * len(priors)
        totals = [0.0] * len(priors)
        root_player = (raw_obs.get("current") or {}).get("yourIndex", 0)
        rng = Rng(self.config.seed).child("policy-mcts")
        simulations = max(1, int(self.config.simulations))
        try:
            for _ in range(simulations):
                total = sum(visits)
                scores = [((totals[i] / visits[i]) if visits[i] else 0.0)
                          + self.config.exploration * priors[i]
                          * math.sqrt(total + 1.0) / (visits[i] + 1.0)
                          + rng.random() * 1e-12
                          for i in range(len(priors))]
                choice = max(range(len(scores)), key=scores.__getitem__)
                sid = None
                try:
                    sid, _ = self.backend.begin(raw_obs, fills,
                                                manual_coin=True)
                    sid, leaf = self.backend.step(
                        sid, encoded.decode([1.0 if i == choice else 0.0
                                             for i in range(len(priors))]))
                    reward = self._terminal_value(leaf, root_player)
                    if reward is None:
                        leaf_actions = encode_actions(
                            getattr(leaf, "select", {}) or {})
                        leaf_state = encode_state(leaf, root_player)
                        reward, _ = self.model.predict(leaf_state,
                                                       leaf_actions.features)
                        current = getattr(leaf, "current", None)
                        if getattr(current, "yourIndex", root_player) != root_player:
                            reward = -reward
                    visits[choice] += 1
                    totals[choice] += reward
                finally:
                    if sid is not None:
                        self.backend.release(sid)
        finally:
            self.backend.end()
        self.last_visits = tuple(visits)
        self.last_totals = tuple(totals)
        best = max(range(len(visits)), key=lambda i: (visits[i], totals[i], -i))
        return encoded.decode([1.0 if i == best else 0.0
                               for i in range(len(visits))])


class PolicyMctsAgent(BaseAgent):
    """Submission agent combining deterministic fills, model and MCTS."""

    def __init__(self, seed, deck=None, card_index=None, model=None,
                 backend=None, simulations=32, exploration=1.4):
        super().__init__(seed, deck)
        from .cards import shared_index
        from .search_encoding import ACTION_FEATURE_NAMES
        from .features import FEATURE_NAMES_V2
        self.card_index = card_index if card_index is not None else shared_index()
        self.model = model or PolicyValueTransformer(
            len(FEATURE_NAMES_V2), len(ACTION_FEATURE_NAMES), seed=seed)
        self.search = PolicyMctsSearch(
            self.model, backend,
            PolicyMctsConfig(simulations, exploration, seed))

    def choose(self, view):
        rng = self.rng.child(f"fills{self.decision_count}")
        fills = sample_fills(view.raw, self._deck, rng, self.card_index)
        return self.search.choose(view.raw, fills)
