"""Subprocess agent server for the cross-repo 松竹梅 battle (SOT-1681).

Runs one project's Kaggle submission agent (``main.agent``) in its OWN process,
working directory and virtualenv, and exposes it over a trivial line-delimited
JSON protocol so a host process (``eval/battle_matsu_take_ume.py``) can drive it
without importing that repo's ``agents`` / ``cg`` packages.

Why a subprocess instead of an in-process import: the three sibling repos
(``ptcg-agent-matsu`` / ``-take`` / ``-ume``) each ship a top-level ``agents``
package whose module names collide (``base``, ``random_agent``, ``search_agent``
exist in more than one), so they cannot all be imported side-by-side in one
interpreter. Isolating each agent in its own process side-steps the collision
entirely and lets each ``main.agent`` resolve its own ``deck.csv`` / native
engine relative to its repo root.

Protocol (one JSON value per line, both directions):

* stdin  ← ``obs_dict`` (the raw engine observation, exactly what the Kaggle
  harness passes to ``agent(obs_dict)``).
* stdout → the action, a ``list[int]`` of option indices; or, if the agent
  raised, ``{"__error__": "<ExceptionType>: <message>"}`` so the host can
  attribute the fault to this agent (a loss) instead of crashing the batch.

The server prints a single ``READY`` line to stderr once ``main.agent`` is
importable, then serves requests until stdin is closed. It is launched as::

    <repo>/venv/bin/python <this file>   # with cwd=<repo>

``cwd`` (the repo root) is prepended to ``sys.path`` so ``import main`` / the
repo's ``cg`` resolve locally.
"""
import json
import os
import sys


def main() -> int:
    sys.path.insert(0, os.getcwd())  # repo root: resolve `main` / `cg` locally
    from main import agent  # the project's Kaggle submission entry point

    sys.stderr.write("READY\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        obs = json.loads(line)
        try:
            action = agent(obs)
        except Exception as exc:  # noqa: BLE001 - report, never crash the server
            payload = {"__error__": f"{type(exc).__name__}: {exc}"}
        else:
            payload = action
        sys.stdout.write(json.dumps(payload))
        sys.stdout.write("\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
