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

By default each contestant plays its own repo ``deck.csv``. Pass ``--decks-dir``
to instead draw randomly from a shared deck pool (the 25 tournament decks in
``decks/initial``, SOT-1684) so the ranking reflects general piloting skill
across a diverse metagame rather than each agent's hand-picked champion deck.

Usage (from this repo root; the take/ume checkouts must exist as siblings)::

    venv/bin/python eval/battle_matsu_take_ume.py --n 200 --json /tmp/sot1681.json
    venv/bin/python eval/battle_matsu_take_ume.py --n 400 \
        --decks-dir decks/initial --deck-mode mirror --seed 20260715 \
        --json /tmp/sot1681_random.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(REPO, "eval", "agent_server.py")
SIBLINGS = os.environ.get("PTCG_SIBLINGS_ROOT", os.path.dirname(REPO))
ENGINE_REPO = os.environ.get("PTCG_ENGINE_REPO", REPO)

# Contestant registry: label (kanji/rōmaji) -> repo directory name.
CONTESTANTS = [
    ("matsu", "松", "ptcg-agent-matsu"),
    ("take", "竹", "ptcg-agent-take"),
    ("ume", "梅", "ptcg-agent-ume"),
]

DECK_SIZE = 60
MAX_DECISIONS = 100_000  # engine draws/decks-out long before this


# --------------------------------------------------------------------------- #
# Per-contestant sandbox (SOT-1681 deck-sync). In the 25-deck random modes the
# host must make the deck the ENGINE deals equal the deck the agent's PLANNER
# reasons about (松 MCTS determinizes from deck.csv; 梅 harness/MCTS reads
# deck.csv; 竹 reads deck.csv for its initial-deck return). agent_server rewrites
# ``<cwd>/deck.csv`` and reloads ``main`` on a ``__set_deck__`` message, so each
# server is launched with ``cwd`` set to a *sandbox*: a throwaway dir that
# symlinks every repo entry EXCEPT deck.csv, whose deck.csv is a real writable
# COPY. That copy — never a symlink — guarantees deck rewrites stay in the
# sandbox and can never clobber a sibling repo's committed deck.csv.
# --------------------------------------------------------------------------- #
def make_sandbox(repo: str, root: Optional[str] = None) -> str:
    """Build a per-contestant sandbox cwd (see the section comment).

    Symlinks every top-level entry of ``repo`` into a fresh temp dir except
    ``deck.csv``, which is copied as a real file so ``__set_deck__`` writes never
    reach the repo. Returns the sandbox path. ``.git`` is skipped (never needed by
    the agent and large). Raises if the deck.csv copy ended up a symlink.
    """
    sb = tempfile.mkdtemp(prefix="sot1681_sb_", dir=root)
    for name in sorted(os.listdir(repo)):
        if name in ("deck.csv", ".git"):
            continue
        os.symlink(os.path.join(repo, name), os.path.join(sb, name))
    src_deck = os.path.join(repo, "deck.csv")
    dst_deck = os.path.join(sb, "deck.csv")
    if os.path.isfile(src_deck):
        shutil.copyfile(src_deck, dst_deck)
    if os.path.islink(dst_deck):  # safety: must be a real file, never the repo's
        raise RuntimeError(f"sandbox deck.csv is a symlink: {dst_deck}")
    return sb


# --------------------------------------------------------------------------- #
# Pure helpers (stdlib only — unit-testable without the engine or the siblings)
# --------------------------------------------------------------------------- #
def load_deck(path: str) -> list[int]:
    """Read a deck CSV: the first ``DECK_SIZE`` non-blank lines as card ids."""
    with open(path, encoding="utf-8") as fh:
        return [int(line.strip()) for line in fh if line.strip()][:DECK_SIZE]


def discover_decks(decks_dir: str) -> list[str]:
    """Return the sorted ``NN_<archetype>.csv`` deck files in ``decks_dir``.

    The 25 tournament decks (SOT-1684) are named ``01_*.csv`` .. ``25_*.csv``;
    non-deck files (``manifest.json``, ``README.md``) are ignored. Sorting is by
    the numeric prefix so the pool order is stable and human-readable.
    """
    files = [
        p for p in glob.glob(os.path.join(decks_dir, "*.csv"))
        if re.match(r"^\d+_", os.path.basename(p))
    ]
    if not files:
        raise SystemExit(f"no NN_*.csv decks found in {decks_dir}")
    files.sort(key=lambda p: int(re.match(r"^(\d+)_", os.path.basename(p)).group(1)))
    return files


def build_deck_schedule(n: int, deck_files: list[str], mode: str,
                        rng: random.Random) -> list[tuple[str, str]]:
    """Assign a ``(deck_a, deck_b)`` pair to each of ``n`` matches.

    ``deck_a``/``deck_b`` are the deck files contestant A / B play that match
    (regardless of which engine seat they take — the caller maps deck→seat).

    * ``mirror`` — both contestants pilot the **same** randomly-drawn deck, and
      each seat-alternating pair (matches ``2k``/``2k+1``) reuses one deck so the
      先後 swap exactly cancels deck strength. This isolates agent skill from
      deck luck (the fair ranking design).
    * ``independent`` — every match draws an independent deck for each
      contestant (tournament-like random pairings; higher variance).
    """
    if mode not in ("mirror", "independent"):
        raise ValueError(f"unknown deck mode: {mode}")
    sched: list[tuple[str, str]] = []
    i = 0
    while i < n:
        if mode == "mirror":
            d = rng.choice(deck_files)
            sched.append((d, d))
            if i + 1 < n:
                sched.append((d, d))
            i += 2
        else:
            sched.append((rng.choice(deck_files), rng.choice(deck_files)))
            i += 1
    return sched


def deck_usage(schedule: list[tuple[str, str]]) -> dict[str, int]:
    """Histogram of how many contestant-slots each deck file filled."""
    counts: dict[str, int] = {}
    for a, b in schedule:
        for f in (a, b):
            name = os.path.basename(f)
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


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
            # Raw counters so sharded runs can be summed exactly (see --aggregate).
            "first_decided": self.first_decided,
            "first_wins": self.first_wins,
            "steps_total": self.steps_total,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PairResult":
        """Rebuild a tally from a ``to_dict`` payload (for shard aggregation)."""
        faults = d.get("faults", {})
        return cls(
            a=d["a"], b=d["b"], n=d.get("n", 0),
            a_wins=d.get("a_wins", 0), b_wins=d.get("b_wins", 0),
            draws=d.get("draws", 0), unfinished=d.get("unfinished", 0),
            a_faults=faults.get(d["a"], 0), b_faults=faults.get(d["b"], 0),
            first_decided=d.get("first_decided", 0),
            first_wins=d.get("first_wins", 0),
            steps_total=d.get("steps_total", 0),
        )

    def merge(self, other: "PairResult") -> None:
        """Fold another same-pairing tally into this one (shard aggregation)."""
        if (self.a, self.b) != (other.a, other.b):
            raise ValueError(f"cannot merge {other.a}v{other.b} into {self.a}v{self.b}")
        self.n += other.n
        self.a_wins += other.a_wins
        self.b_wins += other.b_wins
        self.draws += other.draws
        self.unfinished += other.unfinished
        self.a_faults += other.a_faults
        self.b_faults += other.b_faults
        self.first_decided += other.first_decided
        self.first_wins += other.first_wins
        self.steps_total += other.steps_total


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


def aggregate_reports(reports: list[dict]) -> dict:
    """Combine several shard reports (same contestants/deck-mode) into one.

    Each shard is a report dict written by a prior run (independent ``--seed``);
    same-pairing tallies are summed so the merged CIs tighten exactly as if the
    matches had run in one invocation. Used to spread a long random-deck run over
    several time-bounded shards and re-aggregate — the engine has no seed API, so
    more matches (not a fixed seed) is how the CIs are made decisive.
    """
    if not reports:
        raise ValueError("no reports to aggregate")
    labels = [c["label"] for c in reports[0]["contestants"]]
    merged: dict[tuple[str, str], PairResult] = {}
    total_n = 0
    for rep in reports:
        total_n += rep.get("n_per_pairing", 0)
        for pd in rep["pairings"]:
            pr = PairResult.from_dict(pd)
            key = (pr.a, pr.b)
            if key in merged:
                merged[key].merge(pr)
            else:
                merged[key] = pr
    pairs = list(merged.values())
    base = reports[0].get("deck_selection", {})
    return {
        "issue": "SOT-1681",
        "aggregated_from": len(reports),
        "n_per_pairing": total_n,
        "deck_selection": {
            "mode": base.get("mode"),
            "random": base.get("random"),
            "pool_size": base.get("pool_size"),
            "pool": base.get("pool"),
            "seed": [r.get("deck_selection", {}).get("seed") for r in reports],
        },
        "contestants": reports[0]["contestants"],
        "pairings": [pr.to_dict() for pr in pairs],
        "standings": standings(pairs, labels),
        "note": ("aggregated over independent seeded shards; engine has no seed "
                 "API so results are statistical (Wilson CI)"),
    }


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
    # Sandbox cwd for the server (deck-sync modes); ``None`` runs in the repo root
    # (own-deck mode, where the deck is never rewritten).
    sandbox: Optional[str] = None
    # Deck the agent's planner was last synced to, so we only reload on a change.
    _planner_deck: Optional[list[int]] = field(default=None, repr=False)
    deck_source: Optional[str] = None

    @property
    def python(self) -> str:
        return os.path.join(self.repo, "venv", "bin", "python")

    @property
    def cwd(self) -> str:
        return self.sandbox or self.repo

    def start(self) -> None:
        command = [self.python, SERVER]
        if self.deck_source:
            command.extend(["--deck", self.deck_source])
        self.proc = subprocess.Popen(
            command,
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Wait for the READY handshake so import errors surface immediately.
        line = self.proc.stderr.readline()
        if not line.startswith("READY"):
            err = self.proc.stderr.read()
            raise RuntimeError(f"{self.label} agent failed to start: {line}{err}")
        # A relaunched server has a fresh module state, so re-sync its planner deck.
        self._planner_deck = None

    def set_deck(self, deck: list[int]) -> None:
        """Sync the agent's PLANNER to ``deck`` via ``__set_deck__`` (idempotent).

        Requires a sandbox cwd (so the deck rewrite can't touch the repo). Only
        sends when ``deck`` differs from the last synced deck, to avoid needless
        agent rebuilds. Raises on a dead server or a server-side reload error.
        """
        assert self.sandbox is not None, "set_deck needs a sandbox cwd"
        assert self.proc is not None and self.proc.stdin and self.proc.stdout
        if self._planner_deck == deck:
            return
        self.proc.stdin.write(json.dumps({"__set_deck__": deck}))
        self.proc.stdin.write("\n")
        self.proc.stdin.flush()
        reply = self.proc.stdout.readline()
        if reply == "":
            raise RuntimeError(f"{self.label} agent server exited during set_deck")
        payload = json.loads(reply)
        if not (isinstance(payload, dict) and payload.get("__ok__")):
            raise RuntimeError(f"{self.label} set_deck failed: {payload}")
        self._planner_deck = list(deck)

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
                progress: bool = True,
                schedule: Optional[list[tuple[str, str]]] = None,
                deck_cache: Optional[dict[str, list[int]]] = None) -> PairResult:
    """Play ``n`` seat-alternating matches between contestants ``a`` and ``b``.

    When ``schedule`` is given (from ``build_deck_schedule``), match ``i`` assigns
    contestant A ``schedule[i][0]`` and B ``schedule[i][1]`` (decks loaded via
    ``deck_cache``); otherwise each contestant keeps its own fixed ``.deck``.
    """
    pr = PairResult(a=a.label, b=b.label)
    for i in range(n):
        a_first = (i % 2 == 0)
        if schedule is not None:
            deck_a_file, deck_b_file = schedule[i]
            a.deck = deck_cache[deck_a_file]
            b.deck = deck_cache[deck_b_file]
            # Deck-sync: make each agent's planner reason about the deck the
            # engine actually deals it this match (no-op if unchanged, so mirror
            # mode reloads only once per 先後 pair). Skipped when a contestant
            # has no sandbox (own-deck mode never rewrites decks).
            if a.sandbox is not None:
                a.set_deck(a.deck)
            if b.sandbox is not None:
                b.set_deck(b.deck)
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


def build_contestants(load_own_deck: bool = True) -> list[Contestant]:
    """Instantiate the three contestants.

    When ``load_own_deck`` (legacy ``own`` mode) each contestant is preloaded with
    its repo ``deck.csv``; in random-deck mode decks are assigned per match from
    the shared pool, so the fixed deck is left empty here.
    """
    out = []
    for label, _kanji, dirname in CONTESTANTS:
        repo = os.path.join(SIBLINGS, dirname)
        if not os.path.isfile(os.path.join(repo, "main.py")):
            raise SystemExit(f"contestant repo not found: {repo}")
        deck = load_deck(os.path.join(repo, "deck.csv")) if load_own_deck else []
        out.append(Contestant(label=label, repo=repo, deck=deck))
    return out


def resolve_deck(decks_dir: str, deck_id: str) -> str:
    """Resolve a contestant deck id (``01``) to one unambiguous pool CSV."""
    matches = glob.glob(os.path.join(decks_dir, f"{deck_id}_*.csv"))
    if len(matches) != 1:
        raise SystemExit(
            f"deck id {deck_id!r} resolved to {len(matches)} files in {decks_dir}")
    return os.path.abspath(matches[0])


def build_seat_contestant(spec: str, decks_dir: str, seat: int) -> Contestant:
    """Build an isolated seat from ``tactic:deckId`` (same tactic may be used twice)."""
    try:
        tactic, deck_id = spec.split(":", 1)
    except ValueError as exc:
        raise SystemExit(f"--seat{seat} must be tactic:deckId, got {spec!r}") from exc
    registry = {label: dirname for label, _kanji, dirname in CONTESTANTS}
    if tactic not in registry or not deck_id:
        raise SystemExit(f"invalid --seat{seat} contestant: {spec!r}")
    repo = os.path.join(SIBLINGS, registry[tactic])
    deck_source = resolve_deck(decks_dir, deck_id)
    return Contestant(
        label=spec, repo=repo, deck=load_deck(deck_source),
        sandbox=make_sandbox(repo), deck_source=deck_source,
    )


def print_summary(report: dict) -> None:
    """Print the human-readable standings for a live or aggregated report."""
    ds = report.get("deck_selection", {})
    if ds.get("mode") == "explicit_seats":
        deck_desc = "explicit per-seat decks"
    elif ds.get("random"):
        deck_desc = f"random {ds.get('mode')} from {ds.get('pool_size')} decks (seed={ds.get('seed')})"
    else:
        deck_desc = "each repo's own deck.csv"
    agg = f" [aggregated {report['aggregated_from']} shards]" if report.get("aggregated_from") else ""
    print(f"\n=== 松竹梅 BATTLE RESULTS (SOT-1681){agg} === decks: {deck_desc}")
    for d in report["pairings"]:
        ci = d["a_win_rate_ci95"]
        print(f"{d['a']} vs {d['b']}: {d['a']} {d['a_wins']} / {d['b']} "
              f"{d['b_wins']} / draw {d['draws']}  "
              f"{d['a']}_win_rate={d['a_win_rate']} CI95={ci}  "
              f"first_player_wr={d['first_player_win_rate']}  "
              f"faults={d['faults']}")
    print("\n-- standings --")
    for r in report["standings"]:
        print(f"  {r['contestant']}: {r['wins']}-{r['losses']} "
              f"win_rate={r['win_rate']} CI95={r['win_rate_ci95']}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--n", type=int, default=200,
                   help="matches per pairing (seat-alternating)")
    p.add_argument("--json", default=None, help="write the full report here")
    p.add_argument("--aggregate", nargs="+", default=None, metavar="SHARD.json",
                   help="combine these shard report JSONs into one and exit "
                        "(no matches are played)")
    p.add_argument("--decks-dir", default=None,
                   help="directory of NN_*.csv decks to draw from randomly "
                        "(e.g. decks/initial); omit to use each repo's own deck.csv")
    p.add_argument("--deck-mode", choices=("mirror", "independent"),
                   default="mirror",
                   help="mirror: both play the same random deck per 先後 pair "
                        "(isolates agent skill); independent: each draws its own")
    p.add_argument("--seed", type=int, default=None,
                   help="seed the deck-selection RNG (engine shuffles stay random)")
    p.add_argument("--seat0", help="explicit seat contestant as tactic:deckId")
    p.add_argument("--seat1", help="explicit seat contestant as tactic:deckId")
    args = p.parse_args(argv)

    if args.aggregate:
        reports = []
        for path in args.aggregate:
            with open(path, encoding="utf-8") as fh:
                reports.append(json.load(fh))
        report = aggregate_reports(reports)
        print_summary(report)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)
            print(f"\naggregated report written to {args.json}", file=sys.stderr)
        return 0

    # Host the engine from this repo (cg.game is a process-global single battle).
    sys.path.insert(0, ENGINE_REPO)
    os.chdir(ENGINE_REPO)
    from cg import game

    if bool(args.seat0) != bool(args.seat1):
        p.error("--seat0 and --seat1 must be specified together")
    explicit_seats = args.seat0 is not None
    random_decks = args.decks_dir is not None and not explicit_seats
    deck_pool: list[str] = []
    deck_cache: dict[str, list[int]] = {}
    if random_decks:
        decks_dir = args.decks_dir
        if not os.path.isabs(decks_dir):
            decks_dir = os.path.join(REPO, decks_dir)
        deck_pool = discover_decks(decks_dir)
        deck_cache = {f: load_deck(f) for f in deck_pool}
        print(f"random-deck mode: {len(deck_pool)} decks from {decks_dir} "
              f"(deck-mode={args.deck_mode}, seed={args.seed})",
              file=sys.stderr, flush=True)

    if explicit_seats:
        decks_dir = args.decks_dir or os.path.join(REPO, "decks", "initial")
        if not os.path.isabs(decks_dir):
            decks_dir = os.path.join(REPO, decks_dir)
        contestants = [
            build_seat_contestant(args.seat0, decks_dir, 0),
            build_seat_contestant(args.seat1, decks_dir, 1),
        ]
    else:
        contestants = build_contestants(load_own_deck=not random_decks)
    labels = [c.label for c in contestants]
    by_label = {c.label: c for c in contestants}
    # In random-deck mode each server runs in a sandbox cwd so its deck can be
    # rewritten (deck-sync) without touching the sibling repo's committed deck.csv.
    if random_decks:
        for c in contestants:
            c.sandbox = make_sandbox(c.repo)
    for c in contestants:
        c.start()

    schedules: dict[tuple[str, str], list[tuple[str, str]]] = {}
    pairs: list[PairResult] = []
    try:
        for i, a in enumerate(contestants):
            for b in contestants[i + 1:]:
                print(f"== {a.label} vs {b.label} (n={args.n}) ==",
                      file=sys.stderr, flush=True)
                schedule = None
                if random_decks:
                    # One RNG stream per pairing, seeded deterministically off the
                    # base seed + pairing labels, so each pairing samples the pool
                    # independently yet the whole run is reproducible from --seed.
                    # Use a *string* seed: random.Random hashes it deterministically
                    # (Python's builtin hash() is per-process randomized, which would
                    # break the documented --seed reproducibility across shard runs).
                    seed = None if args.seed is None else f"{args.seed}:{a.label}:{b.label}"
                    rng = random.Random(seed)
                    schedule = build_deck_schedule(args.n, deck_pool,
                                                   args.deck_mode, rng)
                    schedules[(a.label, b.label)] = schedule
                pairs.append(run_pairing(game, a, b, args.n, schedule=schedule,
                                         deck_cache=deck_cache))
    finally:
        for c in by_label.values():
            c.stop()
            if c.sandbox is not None:
                shutil.rmtree(c.sandbox, ignore_errors=True)

    table = standings(pairs, labels)
    report = {
        "issue": "SOT-1681",
        "n_per_pairing": args.n,
        "deck_selection": {
            "mode": "explicit_seats" if explicit_seats else (args.deck_mode if random_decks else "own_deck_csv"),
            "random": random_decks,
            "pool_size": len(deck_pool),
            "pool": [os.path.basename(f) for f in deck_pool],
            "seed": args.seed,
            # Planner deck-sync (engine-dealt deck == agent-planned deck) is on for
            # all random-deck runs, so MCTS agents reason about the deck they pilot.
            "planner_deck_sync": random_decks or explicit_seats,
        },
        "contestants": ([{"label": c.label, "repo": os.path.basename(c.repo),
                           "deck": os.path.basename(c.deck_source or "deck.csv")}
                          for c in contestants] if explicit_seats else [
            {"label": lb, "kanji": kanji, "repo": dirname}
            for lb, kanji, dirname in CONTESTANTS
        ]),
        "pairings": [pr.to_dict() for pr in pairs],
        "standings": table,
        "note": ("engine has no seed API; results are statistical (Wilson CI), "
                 "not bit-reproducible"),
    }
    if random_decks:
        report["deck_usage"] = {
            f"{a} vs {b}": deck_usage(sched)
            for (a, b), sched in schedules.items()
        }

    # Human-readable summary, except JSON-to-stdout mode used by adapters.
    if args.json != "-":
        print_summary(report)

    if args.json:
        if args.json == "-":
            json.dump(report, sys.stdout, ensure_ascii=False)
            sys.stdout.write("\n")
        else:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)
            print(f"\nreport written to {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
