"""State the checkable part of a request up front, then check it.

A finished run gets judged on its own account, and prose criteria do not
survive that: graders asked to rate whether a summary "addresses the
request" agree with people barely more often than not, and are weakest
exactly where the answer depends on what happened to the filesystem.

So only the mechanical part is written down. From the files a request
names, and which of them it treats as already existing versus asked for,
two conditions follow that need no judgement at all:

  - a file the request relies on is still there at the end
  - a file the request asks for exists and has something in it

That is a small set deliberately. Everything else — whether the numbers
are right, whether the summary is fair — is left to the checks that can
actually establish it, rather than dressed up as a rule here.

Conditions are derived once, before the work starts, so they cannot be
quietly relaxed to match whatever the run ended up doing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rune.utils.logger import get_logger

log = get_logger(__name__)

_ENV_FLAG = "RUNE_POSTCONDITIONS"


def postconditions_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "1") != "0"


@dataclass(frozen=True)
class Postcondition:
    name: str
    kind: str  # "present" (input survives) | "produced" (output exists)

    def unmet(self, workspace: Path) -> str | None:
        """Why this condition is not satisfied, or None when it is."""
        hits = [p for p in _candidates(workspace, self.name) if p.is_file()]
        if self.kind == "present":
            if not hits:
                return f"{self.name} was an input to this task and is gone"
            return None
        if not hits:
            return f"{self.name} was asked for and does not exist"
        if all(p.stat().st_size == 0 for p in hits):
            return f"{self.name} was created but is empty"
        return None


def _candidates(workspace: Path, name: str) -> list[Path]:
    """Where a bare file name might have landed in the workspace."""
    direct = workspace / name
    if direct.is_file():
        return [direct]
    try:
        return [p for p in workspace.rglob(name)
                if p.is_file() and ".rune-trash" not in p.parts][:5]
    except OSError:
        return []


def derive(roles: dict[str, str], workspace: str | Path) -> list[Postcondition]:
    """Conditions implied by the request, fixed before any work happens.

    An input only earns a condition if it is actually there to begin with —
    a named file that never existed is the missing-input case, which the
    artifact ledger already reports, and duplicating it here would just
    produce a second complaint about the same thing.
    """
    if not postconditions_enabled():
        return []
    ws = Path(workspace).expanduser().resolve()
    out: list[Postcondition] = []
    for name, role in sorted(roles.items()):
        if role == "input":
            if _candidates(ws, name):
                out.append(Postcondition(name, "present"))
        elif role == "output":
            out.append(Postcondition(name, "produced"))
    if out:
        log.info("postconditions_derived",
                 present=[c.name for c in out if c.kind == "present"],
                 produced=[c.name for c in out if c.kind == "produced"])
    return out


def check(conditions: list[Postcondition], workspace: str | Path) -> list[str]:
    """The conditions that are not satisfied, in plain words."""
    ws = Path(workspace).expanduser().resolve()
    return [msg for c in conditions if (msg := c.unmet(ws))]


def unmet_note(problems: list[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        "Before this is described as done, these do not hold:\n"
        f"{listed}\n"
        "Either put that right or say plainly which parts were not completed."
    )
