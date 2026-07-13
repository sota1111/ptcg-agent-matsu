#!/usr/bin/env python3
"""Forbidden-term linter (SOT-1671): detect hardcoded card names / card IDs.

Grep-based rules over our agent/eval sources:

1. Card-ID hardcoding — comparisons or lookup tables keyed by card/attack
   IDs (`cardId == 123`, `CARD_IDS = {...}`, per-card weight tables). The
   evaluation must derive from card ATTRIBUTES, never from specific cards.
2. Card-NAME hardcoding — quoted string literals that exactly match a card
   name from the card master CSVs (data/, license-restricted, present only
   locally). Skipped with a notice when data/ is absent (e.g. CI).
3. Global RNG use in agent code — `random.<fn>(...)` / `np.random` /
   `numpy.random`. All agent randomness must flow through agents.rng.Rng
   (`random.Random(seed)` instances are allowed inside agents/rng.py only).

Usage: python3 scripts/lint_hardcoded_cards.py  (from the repo root)
Exit code 0 = clean, 1 = violations found.
"""
import csv
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files/dirs to scan (our own code only; cg/ is engine vendor code).
TARGET_DIRS = ["agents", "eval"]
TARGET_FILES = ["main.py"]
# eval/run_match.py predates SOT-1671 (SOT-1670 probe script, kept verbatim
# as it is line-referenced from docs/engine-facts.md).
EXCLUDE = {os.path.join("eval", "run_match.py")}

ID_PATTERNS = [
    (re.compile(r"\b(?:card_?[iI]d|attack_?[iI]d|cardId|attackId)\s*(?:==|!=|<=|>=)\s*\d"),
     "card/attack ID compared to a literal"),
    (re.compile(r"\b(?:card_?[iI]d|attack_?[iI]d|cardId|attackId)\s+in\s*[\[\{\(]"),
     "card/attack ID membership test against a literal collection"),
    (re.compile(r"\bserial\s*==\s*\d"), "card serial compared to a literal"),
    (re.compile(r"\b[A-Z_]*(?:CARD|ATTACK)_IDS?\b\s*="),
     "lookup table keyed by card/attack IDs"),
    (re.compile(r"\.id\s*==\s*\d"), ".id compared to a literal"),
]

RNG_PATTERNS = [
    (re.compile(r"\brandom\.(?:random|randint|choice|choices|sample|shuffle|"
                r"seed|uniform|randrange|getrandbits|betavariate|gauss)\s*\("),
     "global random.* call (use agents.rng.Rng)"),
    (re.compile(r"\b(?:np|numpy)\.random\b"),
     "numpy global RNG (use agents.rng.Rng)"),
]
# Rng itself wraps random.Random with an injected seed.
RNG_EXEMPT = {os.path.join("agents", "rng.py")}

STRING_LITERAL = re.compile(r"""(['"])((?:(?!\1).)+)\1""")


def load_card_names() -> set:
    names = set()
    for csv_name in ("EN_Card_Data.csv", "JP_Card_Data.csv"):
        path = os.path.join(REPO, "data", csv_name)
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header or len(header) < 2:
                continue
            for row in reader:
                if len(row) >= 2 and row[1].strip():
                    names.add(row[1].strip())
    return names


def target_files() -> list:
    files = []
    for d in TARGET_DIRS:
        for root, _dirs, fnames in os.walk(os.path.join(REPO, d)):
            for fname in sorted(fnames):
                if fname.endswith(".py"):
                    files.append(os.path.join(root, fname))
    for fname in TARGET_FILES:
        path = os.path.join(REPO, fname)
        if os.path.exists(path):
            files.append(path)
    return [f for f in files
            if os.path.relpath(f, REPO) not in EXCLUDE]


def lint() -> int:
    card_names = load_card_names()
    if not card_names:
        print("NOTE: data/ card CSVs not found - card-NAME check skipped "
              "(ID and RNG checks still run).")
    violations = []
    for path in target_files():
        rel = os.path.relpath(path, REPO)
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                stripped = line.split("#", 1)[0]  # ignore comments
                for pattern, why in ID_PATTERNS:
                    if pattern.search(stripped):
                        violations.append((rel, lineno, why, line.strip()))
                if rel not in RNG_EXEMPT:
                    for pattern, why in RNG_PATTERNS:
                        if pattern.search(stripped):
                            violations.append((rel, lineno, why, line.strip()))
                if card_names:
                    for match in STRING_LITERAL.finditer(stripped):
                        if match.group(2).strip() in card_names:
                            violations.append(
                                (rel, lineno,
                                 f"hardcoded card name {match.group(2)!r}",
                                 line.strip()))
    if violations:
        print(f"FORBIDDEN-TERM LINT: {len(violations)} violation(s)")
        for rel, lineno, why, text in violations:
            print(f"  {rel}:{lineno}: {why}\n    {text}")
        return 1
    print(f"FORBIDDEN-TERM LINT: clean "
          f"({len(target_files())} files, {len(card_names)} card names)")
    return 0


if __name__ == "__main__":
    sys.exit(lint())
