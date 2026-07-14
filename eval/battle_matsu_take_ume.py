"""松竹梅 cross-repo round-robin battle (SOT-1681).

Plays the three sibling projects' **Kaggle submission agents** against each
other on the shared cabt engine and reports who wins:

* 松 (matsu) — this repo's ``main.agent``
* 竹 (take)  — ``../ptcg-agent-take/main.agent``
* 梅 (ume)   — ``../ptcg-agent-ume/main.agent``

Each contestant is its project's *actual* submission entry point (``main.agent``,
``obs_dict -> list[int]``) playing its project's own ``deck.csv`` — an
agent+deck package, exactly what that project ships. Every contestant runs in an
isolated subprocess (``eval/agent_server.py``) in its own repo/venv, because the
three ``agents`` packages have colliding module names and cannot co-exist in one
interpreter. This host process owns only the engine (this repo's ``cg.game``,
a process-global single battle) and the orchestration.

Fairness (先後入替). Turn order is a real advantage, so every pairing is played
in seat-alternating pairs: on even matches contestant A takes engine seat 0
(先手), on odd matches contestant B does. Decks move with their agent (each
always plays its own deck), so this is an exact 先後 swap, not a deck swap.

Robustness. An agent that raises, emits an illegal action (rejected by the
engine — the sole legal-move authority), or whose subprocess dies is charged a
**fault** and loses that match; the opponent wins and the batch continues (the
faulting server is relaunched for the next match). Fault counts are reported —
0 faults across the run is the "all implementations complete / functional"
signal this issue asks to confirm.

Outputs win rate + **Wilson 95% CI** per pairing (and each pairing's 先手/後手
win rate), plus an overall standings table, as JSON (``--json``) and a printed
summary. The engine has **no seed API** (its shuffles are non-deterministic), so
results are statistical, not bit-reproducible — run enough matches that the CIs
separate.

Usage (from this repo root; the take/ume checkouts must exist as siblings)::

    venv/bin/python eval/battle_matsu_take_ume.py --n 200 --json /tmp/sot1681.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(REPO, "eval", "agent_server.py")
SIBLINGS = os.path.dirname(REPO)

# Contestant registry: label (kanji/rōmaji) -> repo directory name.
CONTESTANTS = [
    ("matsu", "松", "ptcg-agent-matsu"),
    ("take", "竹", "ptcg-agent-take"),
    ("ume", "梅", "ptcg-agent-ume"),
]

DECK_SIZE = 60
MAX_DECISIONS = 100_000  # engine draws/decks-out long before this


# --------------------------------------------------------------------------- #
# Pure helpers (stdlib only — unit-testable without the engine or the siblings)
# --------------------------------------------------------------------------- #
def load_deck(path: str) -> list[int]:
    """Read a deck CSV: the first ``DECK_SIZE`` non-blank lines as card ids."""
    with open(path, encoding="utf-8") as fh:
        return [int(line.strip()) for line in fh if line.strip()][:DECK_SIZE]


def wilson_ci(wins: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% interval for ``wins/n`` (clamped to [0, 1]).

    ``(0.0, 1.0)`` when ``n == 0``. Known value: ``wilson_ci(50, 100)`` ≈
    ``(0.4038, 0.5962)``.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class PairResult:
    """Aggregated outcome of one A-vs-B pairing (A/B are contestant labels)."""

    a: str
    b: str
    n: int = 0
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    unfinished: int = 0
    a_faults: int = 0
    b_faults: int = 0
    # 先後 (turn order): wins of the seat-0 (first) player, over decided matches
    # with a known first player.
    first_decided: int = 0
    first_wins: int = 0
    steps_total: int = 0

    def record(self, *, a_first: bool, result: int, steps: int,
               first_player: Optional[int], fault_seat: Optional[int]) -> None:
        """Fold one finished match into the tally.

        ``result``: engine winner seat (0/1), 2=draw, -1=unfinished. ``a_first``
        means contestant A sat in seat 0 this match. ``fault_seat`` is the engine
        seat charged with a fault (its occupant loses), else ``None``.
        """
        self.n += 1
        self.steps_total += steps
        if result == 2:
            self.draws += 1
            return
        if result not in (0, 1):
            self.unfinished += 1
            return
        # winner seat -> contestant
        a_seat = 0 if a_first else 1
        if result == a_seat:
            self.a_wins += 1
        else:
            self.b_wins += 1
        if fault_seat is not None:
            if fault_seat == a_seat:
                self.a_faults += 1
            else:
                self.b_faults += 1
        if first_player in (0, 1):
            self.first_decided += 1
            if result == first_player:
                self.first_wins += 1

    @property
    def decided(self) -> int:
        return self.a_wins + self.b_wins

    def to_dict(self) -> dict:
        lo, hi = wilson_ci(self.a_wins, self.decided)
        return {
            "a": self.a,
            "b": self.b,
            "n": self.n,
            "decided": self.decided,
            "a_wins": self.a_wins,
            "b_wins": self.b_wins,
            "draws": self.draws,
            "unfinished": self.unfinished,
            "a_win_rate": round(self.a_wins / self.decided, 4) if self.decided else None,
            "a_win_rate_ci95": [round(lo, 4), round(hi, 4)],
            "faults": {self.a: self.a_faults, self.b: self.b_faults},
            "first_player_win_rate": (
                round(self.first_wins / self.first_decided, 4)
                if self.first_decided else None
            ),
            "mean_steps": round(self.steps_total / self.n, 1) if self.n else None,
        }


def standings(pairs: list[PairResult], labels: list[str]) -> list[dict]:
    """Round-robin standings: per contestant total wins / losses / win rate.

    Draws and unfinished matches count for neither side (excluded from the
    win-rate denominator), matching the per-pairing ``a_win_rate``.
    """
    wins = {lb: 0 for lb in labels}
    losses = {lb: 0 for lb in labels}
    for pr in pairs:
        wins[pr.a] += pr.a_wins
        losses[pr.a] += pr.b_wins
        wins[pr.b] += pr.b_wins
        losses[pr.b] += pr.a_wins
    table = []
    for lb in labels:
        decided = wins[lb] + losses[lb]
        lo, hi = wilson_ci(wins[lb], decided)
        table.append({
            "contestant": lb,
            "wins": wins[lb],
            "losses": losses[lb],
            "decided": decided,
            "win_rate": round(wins[lb] / decided, 4) if decided else None,
            "win_rate_ci95": [round(lo, 4), round(hi, 4)],
        })
    table.sort(key=lambda r: (r["win_rate"] is not None, r["win_rate"] or 0.0),
               reverse=True)
    return table


# --------------------------------------------------------------------------- #
# Subprocess-isolated contestant
# --------------------------------------------------------------------------- #
@dataclass
class Contestant:
    """One project's submission agent, driven over a subprocess (see agent_server)."""

    label: str
    repo: str
    deck: list[int]
    proc: Optional[subprocess.Popen] = field(default=None, repr=False)

    @property
    def python(self) -> str:
        return os.path.join(self.repo, "venv", "bin", "python")

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [self.python, SERVER],
            cwd=self.repo,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Wait for the READY handshake so import errors surface immediately.
        line = self.proc.stderr.readline()
        if line.strip() != "READY":
            err = self.proc.stderr.read()
            raise RuntimeError(f"{self.label} agent failed to start: {line}{err}")

    def act(self, obs: dict) -> list[int]:
        """Ask the agent for an action; raises on a dead server or agent error."""
        assert self.proc is not None and self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(obs))
        self.proc.stdin.write("\n")
        self.proc.stdin.flush()
        reply = self.proc.stdout.readline()
        if reply == "":  # server died
            raise RuntimeError(f"{self.label} agent server exited unexpectedly")
        action = json.loads(reply)
        if isinstance(action, dict) and "__error__" in action:
            raise RuntimeError(f"{self.label} agent error: {action['__error__']}")
        return action

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - best-effort teardown
            self.proc.kill()
        self.proc = None

    def restart(self) -> None:
        self.stop()
        self.start()


# --------------------------------------------------------------------------- #
# Engine-bound match loop
# --------------------------------------------------------------------------- #
def play_match(game, seat0: Contestant, seat1: Contestant) -> dict:
    """One engine match between the two seated contestants.

    Routes each decision to ``current.yourIndex``'s occupant. Returns
    ``{"result", "steps", "first_player", "fault_seat"}`` where ``result`` is the
    engine winner seat (0/1), 2=draw, -1=unfinished, and ``fault_seat`` names the
    seat charged with a fault (illegal action / agent error / dead server), whose
    occupant is treated as the loser.
    """
    obs, start = game.battle_start(seat0.deck, seat1.deck)
    if obs is None:
        raise RuntimeError(
            f"battle_start failed: errorPlayer={start.errorPlayer} "
            f"errorType={start.errorType}")
    first_player: Optional[int] = None
    steps = 0
    try:
        while steps < MAX_DECISIONS:
            cur = obs.get("current") or {}
            fp = cur.get("firstPlayer", -1)
            if first_player is None and fp in (0, 1):
                first_player = fp
            result = cur.get("result", -1)
            if result != -1:
                return {"result": result, "steps": steps,
                        "first_player": first_player, "fault_seat": None}
            seat = cur.get("yourIndex", 0)
            agent = seat0 if seat == 0 else seat1
            try:
                action = agent.act(obs)
            except Exception:  # noqa: BLE001 - agent error => that seat's loss
                return {"result": 1 - seat, "steps": steps,
                        "first_player": first_player, "fault_seat": seat}
            try:
                obs = game.battle_select(action)
            except Exception:  # noqa: BLE001 - engine reject => illegal move
                return {"result": 1 - seat, "steps": steps,
                        "first_player": first_player, "fault_seat": seat}
            steps += 1
        return {"result": -1, "steps": steps,
                "first_player": first_player, "fault_seat": None}
    finally:
        game.battle_finish()


def run_pairing(game, a: Contestant, b: Contestant, n: int,
                progress: bool = True) -> PairResult:
    """Play ``n`` seat-alternating matches between contestants ``a`` and ``b``."""
    pr = PairResult(a=a.label, b=b.label)
    for i in range(n):
        a_first = (i % 2 == 0)
        seat0, seat1 = (a, b) if a_first else (b, a)
        out = play_match(game, seat0, seat1)
        pr.record(a_first=a_first, result=out["result"], steps=out["steps"],
                  first_player=out["first_player"], fault_seat=out["fault_seat"])
        if out["fault_seat"] is not None:
            # Relaunch the faulting server so a one-off crash can't cascade.
            (seat0 if out["fault_seat"] == 0 else seat1).restart()
        if progress and (i + 1) % 50 == 0:
            print(f"  [{a.label} vs {b.label}] {i + 1}/{n} "
                  f"(A {pr.a_wins} / B {pr.b_wins} / draw {pr.draws})",
                  file=sys.stderr, flush=True)
    return pr


def build_contestants() -> list[Contestant]:
    """Instantiate the three contestants, each with its own repo deck.csv."""
    out = []
    for label, _kanji, dirname in CONTESTANTS:
        repo = os.path.join(SIBLINGS, dirname)
        if not os.path.isfile(os.path.join(repo, "main.py")):
            raise SystemExit(f"contestant repo not found: {repo}")
        deck = load_deck(os.path.join(repo, "deck.csv"))
        out.append(Contestant(label=label, repo=repo, deck=deck))
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--n", type=int, default=200,
                   help="matches per pairing (seat-alternating)")
    p.add_argument("--json", default=None, help="write the full report here")
    args = p.parse_args(argv)

    # Host the engine from this repo (cg.game is a process-global single battle).
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    from cg import game

    contestants = build_contestants()
    labels = [c.label for c in contestants]
    by_label = {c.label: c for c in contestants}
    for c in contestants:
        c.start()

    pairs: list[PairResult] = []
    try:
        for i, a in enumerate(contestants):
            for b in contestants[i + 1:]:
                print(f"== {a.label} vs {b.label} (n={args.n}) ==",
                      file=sys.stderr, flush=True)
                pairs.append(run_pairing(game, a, b, args.n))
    finally:
        for c in by_label.values():
            c.stop()

    table = standings(pairs, labels)
    report = {
        "issue": "SOT-1681",
        "n_per_pairing": args.n,
        "contestants": [
            {"label": lb, "kanji": kanji, "repo": dirname}
            for lb, kanji, dirname in CONTESTANTS
        ],
        "pairings": [pr.to_dict() for pr in pairs],
        "standings": table,
        "note": ("engine has no seed API; results are statistical (Wilson CI), "
                 "not bit-reproducible"),
    }

    # Human-readable summary.
    print("\n=== 松竹梅 BATTLE RESULTS (SOT-1681) ===")
    for d in report["pairings"]:
        ci = d["a_win_rate_ci95"]
        print(f"{d['a']} vs {d['b']}: {d['a']} {d['a_wins']} / {d['b']} "
              f"{d['b_wins']} / draw {d['draws']}  "
              f"{d['a']}_win_rate={d['a_win_rate']} CI95={ci}  "
              f"first_player_wr={d['first_player_win_rate']}  "
              f"faults={d['faults']}")
    print("\n-- standings --")
    for r in table:
        print(f"  {r['contestant']}: {r['wins']}-{r['losses']} "
              f"win_rate={r['win_rate']} CI95={r['win_rate_ci95']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"\nreport written to {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
