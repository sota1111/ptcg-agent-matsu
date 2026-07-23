"""Leaf evaluator (SOT-1672) — layer [4] of the 4-layer architecture.

`Evaluator.evaluate(obs, root_player)` maps a battle observation to an
estimated win probability in [0, 1] for `root_player`. The planner
(agents/planner.py) calls it on non-terminal rollout leaves; a learned value
function can replace `HeuristicEvaluator` by implementing the same interface
(SOT-1674).

Observations are duck-typed: any object shaped like `cg.api.Observation`
(the engine's search states) works, as do test doubles built from
SimpleNamespace. Every feature derives from state/card ATTRIBUTES visible in
the observation (prizes, HP, energy counts, hand/deck sizes) — per-card
weight tables keyed by card ID/name are forbidden
(scripts/lint_hardcoded_cards.py).

SOT-1674 adds `LearnedEvaluator`: a logistic value model over
agents/features.py vectors, trained on self-play logs
(train/gen_selfplay.py + train/train_value.py) and loaded from a JSON
weight file. Select it per agent with the string spec
`evaluator="learned"` (bench: --config-a '{"evaluator": "learned"}').

SOT-1679 extends the same class to MLP value networks (arXiv:1808.04794
§IV-B: tanh hidden layers + a sigmoid win-probability output, pure-Python
forward pass): a model file with a "layers" list is an MLP, one with a flat
"weights" list stays the SOT-1674 logistic model. The spec
`evaluator="value_net"` loads the cheater-selfplay-trained network from
train/value_net.json (paired with the planner's rollout_cutoff, this is the
paper's mctsV variant).
"""
import json
import math
import os

# Feature weights (externally overridable for SOT-1673 ablation). Scores are
# per-side; the value is a logistic squash of (score_me - score_opp).
DEFAULT_WEIGHTS = {
    "prize_taken": 2.0,   # per prize card this side has taken (dominant term)
    "pokemon": 0.3,       # per Pokémon this side has in play
    "energy": 0.2,        # per Energy attached on this side
    "hp": 0.004,          # per HP point this side has in play
    "hand": 0.06,         # per card in hand
    "deck_empty": -3.0,   # this side loses at its next turn start (deck-out)
    # Deck-preservation gradient (SOT-1697): the loss-trace analysis found
    # deck-out is matsu's dominant defeat (53% of losses vs 竹, 91% vs 梅) — the
    # champion only penalised the deck_empty CLIFF at deckCount==0 while `hand`
    # positively rewards drawing, so the search happily mills itself. `deck_low`
    # (<=0) applies a smooth penalty for each card the own deck sits below
    # `deck_low_at`, steering the search away from self-deck-out lines *before*
    # the terminal cliff. Default 0.0 => champion behaviour is unchanged; a
    # candidate turns it on via eval_weights and is CI-gated vs champion.
    "deck_low": 0.0,      # penalty per deck card below the threshold (<=0)
    "deck_low_at": 0,     # threshold deck size; 0 disables the gradient
    # Apply preservation only while this many prizes remain. Near a win,
    # drawing for the finisher remains correctly valued.
    "deck_low_prize_gate": 0,
    # Next-turn board survival (SOT-1878).  For each side, estimate the
    # opponent's highest currently energy-payable attack from Active or
    # Bench (a conservative switch/retreat response).  If that damage can
    # knock out every visible Pokémon, apply this penalty.  Default 0 keeps
    # the promoted champion byte-for-byte equivalent; candidates enable it
    # through eval_weights and must pass the league promotion gate.
    "board_wipe": 0.0,
    # Smooth alternative to the binary wipe cliff (SOT-1883).  The value is
    # the fraction of visible Pokémon that survive the opponent's strongest
    # currently payable attack, with a small normalized HP-margin bonus.
    # Bench Pokémon therefore act as explicit switch/replacement routes.
    "board_survival": 0.0,
    "scale": 0.6,         # logistic scale on the score difference
}

PRIZE_START = 6  # PRIZE_SIZE (ptcgProgram 22/Core.h:14)


class Evaluator:
    """Value interface: estimated win probability for `root_player`."""

    def evaluate(self, obs, root_player: int) -> float:
        raise NotImplementedError


class HeuristicEvaluator(Evaluator):
    """Card-attribute heuristic value; terminal results are exact."""

    def __init__(self, weights: dict | None = None, card_index=None):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self._cards = card_index

    def evaluate(self, obs, root_player: int) -> float:
        current = getattr(obs, "current", None)
        if current is None:
            return 0.5
        result = getattr(current, "result", -1)
        if result is not None and result != -1:
            if result == root_player:
                return 1.0
            if result == 1 - root_player:
                return 0.0
            return 0.5  # draw (result == 2) or unknown future value
        players = getattr(current, "players", None) or ()
        if len(players) < 2:
            return 0.5
        diff = (self._side_score(players[root_player],
                                 players[1 - root_player])
                - self._side_score(players[1 - root_player],
                                   players[root_player]))
        return 1.0 / (1.0 + math.exp(-self.weights["scale"] * diff))

    def _reachable_damage(self, attacker) -> float:
        """Highest damage the opponent can reach next turn from its board.

        Active and Bench are both considered because a legal switch/retreat
        can expose a charged Bench attacker.  Energy cost and damage come
        solely from the card master; unknown/facedown cards contribute zero.
        """
        if self._cards is None:
            return 0.0
        best = 0.0
        in_play = list(getattr(attacker, "active", None) or ())
        in_play += list(getattr(attacker, "bench", None) or ())
        for pk in in_play:
            if pk is None:
                continue
            energy = len(getattr(pk, "energies", None) or ())
            card = self._cards.card(getattr(pk, "id", None))
            for attack_id in card.attack_ids:
                attack = self._cards.attack(attack_id)
                if attack.energy_cost <= energy:
                    best = max(best, float(attack.damage))
        return best

    def board_wipe_risk(self, defender, attacker) -> float:
        """Return 1 when all visible defenders are exposed next turn."""
        damage = self._reachable_damage(attacker)
        return self._board_wipe_risk_at_damage(defender, damage)

    @staticmethod
    def _board_wipe_risk_at_damage(defender, damage: float) -> float:
        if damage <= 0:
            return 0.0
        in_play = list(getattr(defender, "active", None) or ())
        in_play += list(getattr(defender, "bench", None) or ())
        known_hp = [float(getattr(pk, "hp", 0) or 0)
                    for pk in in_play if pk is not None]
        # Facedown defenders are an unknown escape route, so do not claim a
        # full wipe. Empty boards are handled by the engine's terminal state.
        if not known_hp or any(pk is None for pk in in_play):
            return 0.0
        return float(all(0 < hp <= damage for hp in known_hp))

    @staticmethod
    def _board_survival_at_damage(defender, damage: float) -> float:
        """Continuous [0, 1] board survival estimate after one attack.

        Unknown/facedown Pokémon are treated as viable escape routes instead
        of inventing HP.  For known Pokémon the dominant term is the fraction
        whose remaining HP exceeds reachable damage; a bounded margin term
        breaks ties between survivors without introducing a costly rollout.
        """
        in_play = list(getattr(defender, "active", None) or ())
        in_play += list(getattr(defender, "bench", None) or ())
        if not in_play:
            return 0.0
        if damage <= 0:
            return 1.0
        survived = 0.0
        margin = 0.0
        for pk in in_play:
            if pk is None:
                survived += 1.0
                continue
            hp = float(getattr(pk, "hp", 0) or 0)
            if hp > damage:
                survived += 1.0
                margin += min(1.0, (hp - damage) / max(hp, 1.0))
        count = len(in_play)
        return min(1.0, (survived + 0.25 * margin) / (1.25 * count))

    def board_survival(self, defender, attacker) -> float:
        return self._board_survival_at_damage(
            defender, self._reachable_damage(attacker))

    def _side_score(self, p, opponent=None) -> float:
        w = self.weights
        prize = getattr(p, "prize", None) or ()
        score = w["prize_taken"] * max(0, PRIZE_START - len(prize))
        hp_total = 0
        pokemon = 0
        energy = 0
        in_play = list(getattr(p, "active", None) or ())
        in_play += list(getattr(p, "bench", None) or ())
        for pk in in_play:
            if pk is None:  # facedown Pokémon: presence known, stats hidden
                pokemon += 1
                continue
            pokemon += 1
            hp_total += getattr(pk, "hp", 0) or 0
            energy += len(getattr(pk, "energies", None) or ())
        score += w["pokemon"] * pokemon
        score += w["energy"] * energy
        score += w["hp"] * hp_total
        score += w["hand"] * (getattr(p, "handCount", 0) or 0)
        deck = getattr(p, "deckCount", 0) or 0
        if deck == 0:
            score += w["deck_empty"]
        else:
            thr = w.get("deck_low_at", 0) or 0
            if thr and deck < thr:
                gate = w.get("deck_low_prize_gate", 0) or 0
                if not gate or len(prize) >= gate:
                    score += w.get("deck_low", 0.0) * (thr - deck)
        if opponent is not None and (
                w.get("board_wipe", 0.0) or w.get("board_survival", 0.0)):
            damage = self._reachable_damage(opponent)
            if w.get("board_wipe", 0.0):
                score += w["board_wipe"] * self._board_wipe_risk_at_damage(
                    p, damage)
            if w.get("board_survival", 0.0):
                score += w["board_survival"] * self._board_survival_at_damage(
                    p, damage)
        return score


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "train", "value_model.json")
DEFAULT_NET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "train", "value_net.json")


class LearnedEvaluator(Evaluator):
    """Self-play-trained logistic value model (SOT-1674).

    v = sigmoid(w · standardize(featurize(obs, root_player)) + b), with
    exact values on terminal states (same contract as HeuristicEvaluator).
    The model file records the feature names it was trained on; a mismatch
    with the running agents/features.py raises at load time rather than
    silently mis-scoring.
    """

    def __init__(self, model_path: str | None = None, model: dict | None = None,
                 card_index=None, embeddings=None):
        from .features import make_featurizer
        self._card_index = card_index
        if model is None:
            path = model_path or DEFAULT_MODEL_PATH
            with open(path) as f:
                model = json.load(f)
        # feature_set selects the extractor (SOT-1676): "v1" (default,
        # backward compatible) or "v2" (embedding-extended board vector).
        feature_set = model.get("feature_set", "v1")
        expected_names, self._featurize = make_featurizer(
            feature_set, embeddings=embeddings)
        names = tuple(model.get("feature_names", ()))
        if names != tuple(expected_names):
            raise ValueError(
                f"value model feature set {feature_set!r} does not match "
                f"agents/features.py (model has {len(names)}, code has "
                f"{len(expected_names)}); re-run train/train_value.py")
        self.mean = [float(x) for x in model.get("mean", [0.0] * len(names))]
        self.std = [float(x) or 1.0
                    for x in model.get("std", [1.0] * len(names))]
        # "layers" -> MLP (SOT-1679): [{"w": [[fan_in floats] per neuron],
        # "b": [floats]}, ...]; tanh hidden activations, the last layer is a
        # single logit squashed by sigmoid. Absent -> SOT-1674 logistic model.
        self.layers = None
        if "layers" in model:
            self.layers = [([[float(v) for v in row] for row in layer["w"]],
                            [float(v) for v in layer["b"]])
                           for layer in model["layers"]]
            fan_in = len(names)
            for w, b in self.layers:
                if len(w) != len(b) or any(len(row) != fan_in for row in w):
                    raise ValueError("value net layer shapes are inconsistent "
                                     "with the feature set; re-run "
                                     "train/train_value.py")
                fan_in = len(w)
            if len(self.layers[-1][0]) != 1:
                raise ValueError("value net must end in a single output logit")
        else:
            self.weights = [float(x) for x in model["weights"]]
            self.bias = float(model.get("bias", 0.0))

    def evaluate(self, obs, root_player: int) -> float:
        current = getattr(obs, "current", None)
        if current is None:
            return 0.5
        result = getattr(current, "result", -1)
        if result is not None and result != -1:
            if result == root_player:
                return 1.0
            if result == 1 - root_player:
                return 0.0
            return 0.5  # draw (result == 2) or unknown future value
        x = self._featurize(obs, root_player, self._card_index)
        if self.layers is not None:
            h = [(xi - mi) / si for xi, mi, si in zip(x, self.mean, self.std)]
            for w, b in self.layers[:-1]:
                h = [math.tanh(bj + sum(wj * hj for wj, hj in zip(row, h)))
                     for row, bj in zip(w, b)]
            w, b = self.layers[-1]
            z = b[0] + sum(wj * hj for wj, hj in zip(w[0], h))
        else:
            z = self.bias
            for xi, mi, si, wi in zip(x, self.mean, self.std, self.weights):
                z += wi * (xi - mi) / si
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)


def make_evaluator(spec, card_index=None) -> Evaluator:
    """Resolve an evaluator spec: an Evaluator instance passes through;
    "heuristic" / "learned" / "value_net" build the corresponding
    implementation (this is what lets eval/bench.py switch value functions
    from --config JSON)."""
    if isinstance(spec, Evaluator):
        return spec
    if spec in (None, "heuristic"):
        return HeuristicEvaluator()
    if spec == "learned":
        return LearnedEvaluator(card_index=card_index)
    if spec == "value_net":
        return LearnedEvaluator(model_path=DEFAULT_NET_PATH,
                                card_index=card_index)
    raise ValueError(f"unknown evaluator spec: {spec!r}")
