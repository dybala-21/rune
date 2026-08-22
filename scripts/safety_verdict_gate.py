"""Fail when a change lets through a command the base branch stopped.

    python scripts/safety_verdict_gate.py base.json head.json

Both files come from safety_verdict_diff.py. A verdict that got stricter is
usually the point of the change and passes silently. A verdict that got
softer fails, unless the command is written down in
safety_verdict_allowlist.txt — which is how a deliberate loosening gets
reviewed instead of merged in silence.
"""
from __future__ import annotations

import json
import os
import sys

ORDER = {"allow": 0, "ask": 1, "deny": 2}
_HERE = os.path.dirname(os.path.abspath(__file__))
_ALLOWLIST = os.path.join(_HERE, "safety_verdict_allowlist.txt")


def allowed() -> set[str]:
    if not os.path.exists(_ALLOWLIST):
        return set()
    out = set()
    with open(_ALLOWLIST, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                out.add(json.loads(line))
    return out


def softer(base: dict, head: dict) -> list[str]:
    """Commands the head tree treats more leniently than the base tree.

    Every field counts. The three defects this check exists for were each
    found by comparing one more field than the time before: a decision, then
    a risk level, then an approval flag that turned a run into a hang.
    """
    return [
        c for c in base if c in head
        and (ORDER[head[c][0]] < ORDER[base[c][0]]   # runs where it would not have
             or head[c][1] < base[c][1]              # reads as less dangerous
             or head[c][2] < base[c][2]              # lost an approval it needed
             or head[c][3] > base[c][3])             # allowed where it was blocked
    ]


def main() -> int:
    base = json.load(open(sys.argv[1], encoding="utf-8"))
    head = json.load(open(sys.argv[2], encoding="utf-8"))
    known = allowed()
    found = softer(base, head)
    undeclared = [c for c in found if c not in known]
    print(f"{len(base)} commands compared, {len(found)} more permissive, "
          f"{len(found) - len(undeclared)} of them declared")
    for c in undeclared:
        print(f"UNDECLARED LOOSENING {base[c]} -> {head[c]}  {c!r}", file=sys.stderr)
    if undeclared:
        print(f"\n{len(undeclared)} command(s) got easier to run without being "
              f"written down. Add them to scripts/safety_verdict_allowlist.txt "
              f"with the reason, or fix the change.", file=sys.stderr)
    return 1 if undeclared else 0


if __name__ == "__main__":
    sys.exit(main())
