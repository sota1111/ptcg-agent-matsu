"""Card-embedding trainer (SOT-1676) — pure-Python word2vec skip-gram.

Reproduces Świechowski et al., "Improving Hearthstone AI by Combining MCTS
and Supervised Learning Algorithms" (arXiv:1808.04794) §IV-A: card
representations are learned from card TEXT with word2vec (skip-gram,
low-dimensional; the paper uses dimension 10 and window 10), and a card's
embedding is the mean of its tokens' word vectors.

The corpus comes from the engine card master (`cg.api.all_card_data()` /
`all_attack()`): one document per card, made of attribute tokens (type, HP
bucket, stage, retreat, weakness/resistance, ex/tera flags), name tokens,
and the effect text of its abilities and attacks. No third-party deps; all
randomness flows through agents.rng.Rng, so a fixed --seed reproduces the
output JSON byte-for-byte (same corpus -> same file).

The output JSON keys embeddings by card ID only (no card names — the master
text is license-restricted) and carries a `default` vector (mean of all card
vectors) used as the unknown-card / facedown fallback by
agents.cards.CardEmbeddings.

Usage (from the repo root):
    venv/bin/python train/train_embeddings.py \
        --out train/card_embeddings.json [--dim 10] [--window 10] \
        [--epochs 3] [--negative 5] [--seed 61676]
"""
import argparse
import bisect
import json
import math
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from agents.rng import Rng

TOKEN_RE = re.compile(r"[a-z0-9']+")

# Bucket sizes keep numeric attributes as a small closed token set.
HP_BUCKET = 30
DAMAGE_BUCKET = 30


def _get(obj, name, default=None):
    """Attribute access tolerant of missing fields (future master columns)."""
    value = getattr(obj, name, default)
    return default if value is None else value


def _tokens(text) -> list:
    return TOKEN_RE.findall(str(text or "").lower())


def card_document(card, attacks_by_id: dict) -> list:
    """One card -> its token document (attributes + name + effect texts)."""
    doc = []
    doc += _tokens(_get(card, "name", ""))
    doc.append(f"type:{int(_get(card, 'cardType', -1))}")
    doc.append(f"energy:{int(_get(card, 'energyType', -1))}")
    doc.append(f"hp:{int(_get(card, 'hp', 0)) // HP_BUCKET}")
    doc.append(f"retreat:{int(_get(card, 'retreatCost', 0))}")
    for flag in ("basic", "stage1", "stage2", "ex", "megaEx", "tera",
                 "aceSpec"):
        if _get(card, flag, False):
            doc.append(f"is:{flag.lower()}")
    weakness = getattr(card, "weakness", None)
    if weakness is not None:
        doc.append(f"weak:{int(weakness)}")
    resistance = getattr(card, "resistance", None)
    if resistance is not None:
        doc.append(f"resist:{int(resistance)}")
    evolves_from = _get(card, "evolvesFrom")
    if evolves_from:
        doc.append("evolves")
        doc += _tokens(evolves_from)
    for skill in _get(card, "skills", ()) or ():
        doc.append("ability")
        doc += _tokens(_get(skill, "name", ""))
        doc += _tokens(_get(skill, "text", ""))
    for attack_id in _get(card, "attacks", ()) or ():
        attack = attacks_by_id.get(int(attack_id))
        if attack is None:
            continue
        doc.append("attack")
        doc += _tokens(_get(attack, "name", ""))
        doc += _tokens(_get(attack, "text", ""))
        doc.append(f"dmg:{int(_get(attack, 'damage', 0)) // DAMAGE_BUCKET}")
        energies = _get(attack, "energies", ()) or ()
        doc.append(f"cost:{len(energies)}")
        for energy in energies:
            doc.append(f"cost_type:{int(energy)}")
    return doc


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def train_word2vec(docs, dim: int, window: int, epochs: int, negative: int,
                   seed: int, lr0: float = 0.025) -> dict:
    """Skip-gram with negative sampling over token documents.

    Returns {token: vector}. Deterministic for a fixed (docs, params, seed):
    vocabulary order, init, window sampling, and negative sampling all derive
    from sorted structures and the seeded Rng.
    """
    counts = {}
    for doc in docs:
        for token in doc:
            counts[token] = counts.get(token, 0) + 1
    vocab = sorted(counts, key=lambda t: (-counts[t], t))
    if not vocab:
        return {}
    index = {t: i for i, t in enumerate(vocab)}

    # Negative-sampling table: cumulative unigram^0.75 (word2vec convention).
    cumulative = []
    total = 0.0
    for token in vocab:
        total += counts[token] ** 0.75
        cumulative.append(total)

    rng = Rng(seed).child("w2v")
    w_in = [[(rng.random() - 0.5) / dim for _ in range(dim)] for _ in vocab]
    w_out = [[0.0] * dim for _ in vocab]

    id_docs = [[index[t] for t in doc] for doc in docs]
    n_tokens = sum(len(doc) for doc in id_docs)
    total_steps = max(1, epochs * n_tokens)
    step = 0
    for _epoch in range(epochs):
        for doc in id_docs:
            for pos, center in enumerate(doc):
                step += 1
                lr = max(lr0 * 1e-4, lr0 * (1.0 - step / total_steps))
                span = rng.randint(1, window)  # dynamic window (word2vec)
                v_in = w_in[center]
                for offset in range(-span, span + 1):
                    ctx_pos = pos + offset
                    if offset == 0 or ctx_pos < 0 or ctx_pos >= len(doc):
                        continue
                    context = doc[ctx_pos]
                    grad_in = [0.0] * dim
                    for k in range(negative + 1):
                        if k == 0:
                            target, label = context, 1.0
                        else:
                            pick = rng.random() * total
                            target = bisect.bisect_left(cumulative, pick)
                            target = min(target, len(vocab) - 1)
                            if target == context:
                                continue
                            label = 0.0
                        v_out = w_out[target]
                        f = sum(a * b for a, b in zip(v_in, v_out))
                        g = (label - _sigmoid(f)) * lr
                        for j in range(dim):
                            grad_in[j] += g * v_out[j]
                            v_out[j] += g * v_in[j]
                    for j in range(dim):
                        v_in[j] += grad_in[j]
    return {token: w_in[i] for token, i in index.items()}


def build_embeddings(card_data, attack_data, dim: int = 10, window: int = 10,
                     epochs: int = 3, negative: int = 5,
                     seed: int = 61676) -> dict:
    """Card master -> embeddings payload (the card_embeddings.json content).

    A card's embedding is the mean of its document tokens' word vectors
    (arXiv:1808.04794 §IV-A); `default` is the mean of all card vectors and
    serves as the unknown-card fallback.
    """
    attacks_by_id = {}
    for attack in attack_data:
        attack_id = _get(attack, "attackId")
        if attack_id is not None:
            attacks_by_id[int(attack_id)] = attack
    cards = []
    for card in card_data:
        card_id = _get(card, "cardId")
        if card_id is not None:
            cards.append((int(card_id), card))
    cards.sort(key=lambda pair: pair[0])

    docs = [card_document(card, attacks_by_id) for _, card in cards]
    word_vectors = train_word2vec(docs, dim=dim, window=window, epochs=epochs,
                                  negative=negative, seed=seed)

    vectors = {}
    for (card_id, _), doc in zip(cards, docs):
        if doc:
            mean = [0.0] * dim
            for token in doc:
                vec = word_vectors[token]
                for j in range(dim):
                    mean[j] += vec[j]
            vectors[str(card_id)] = [round(v / len(doc), 6) for v in mean]
        else:
            vectors[str(card_id)] = [0.0] * dim
    if vectors:
        default = [0.0] * dim
        for vec in vectors.values():
            for j in range(dim):
                default[j] += vec[j]
        default = [round(v / len(vectors), 6) for v in default]
    else:
        default = [0.0] * dim
    return {
        "dim": dim,
        "cards": vectors,
        "default": default,
        "meta": {
            "issue": "SOT-1676",
            "method": "word2vec skip-gram + negative sampling, card vector = "
                      "mean of document token vectors (arXiv:1808.04794 IV-A)",
            "source": "cg.api.all_card_data() + all_attack()",
            "n_cards": len(vectors),
            "vocab_size": len(word_vectors),
            "seed": seed, "window": window, "epochs": epochs,
            "negative": negative,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(
        REPO, "train", "card_embeddings.json"))
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--negative", type=int, default=5)
    parser.add_argument("--seed", type=int, default=61676)
    args = parser.parse_args()

    os.chdir(REPO)  # libcg.so resolves relative to the repo root
    from cg.api import all_attack, all_card_data
    payload = build_embeddings(all_card_data(), all_attack(), dim=args.dim,
                               window=args.window, epochs=args.epochs,
                               negative=args.negative, seed=args.seed)
    with open(args.out, "w") as f:
        json.dump(payload, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    meta = payload["meta"]
    print(f"wrote {args.out}: {meta['n_cards']} cards, dim {payload['dim']}, "
          f"vocab {meta['vocab_size']}, seed {meta['seed']}")


if __name__ == "__main__":
    main()
