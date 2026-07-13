"""Seeded RNG for agents (SOT-1671).

All agent randomness must flow through a single `Rng` whose seed is injected
from outside. Global `random.*` / `np.random` use is forbidden in agents/
(enforced by scripts/lint_hardcoded_cards.py). Engine-internal randomness
(shuffle, coins) is NOT injectable (ASSUMPTIONS.md A-9), so reproducibility
guarantees apply to agent decisions only: same seed + same observation
sequence -> same actions.
"""
import hashlib
import random


class Rng:
    """A seeded random stream that can derive independent child streams."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self._r = random.Random(self.seed)

    def child(self, name: str) -> "Rng":
        """Derive an independent, deterministic child stream from this seed."""
        digest = hashlib.sha256(f"{self.seed}:{name}".encode("utf-8")).digest()
        return Rng(int.from_bytes(digest[:8], "big"))

    def randint(self, a: int, b: int) -> int:
        return self._r.randint(a, b)

    def sample(self, population, k: int) -> list:
        return self._r.sample(population, k)

    def choice(self, seq):
        return self._r.choice(seq)

    def shuffle(self, seq) -> None:
        self._r.shuffle(seq)

    def random(self) -> float:
        return self._r.random()
