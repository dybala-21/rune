"""State the checkable part of a request up front, then check it.

A finished run gets judged on its own account, and prose criteria do not
survive that: graders asked to rate whether a summary "addresses the
request" agree with people barely more often than not, and are weakest
exactly where the answer depends on what happened to the filesystem.

So only the mechanical part is written down, and it comes to one thing:
a file the request relies on is still there at the end. Nothing has to be
judged to settle that — the file is present or it is not.

Demanding that a named output exist looked like the obvious companion
rule, and it was wrong. Whether a request is owed its output depends on
whether the data behind it was there, which is not visible on the
filesystem and is usually only discovered part way through the work. Asked
for a discount report from a table with no discount column, the correct
answer is to say the figures cannot be produced — and a rule insisting the
file appear pushes toward inventing one, on exactly the tasks where that
is the failure being guarded against. Across 36 runs it fired four times,
three of them on work that was right, and never once changed how a run
came out. Everything else — whether the numbers are right, whether the
summary is fair — is left to the checks that can actually establish it.

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
    kind: str  # "present" — an input the task relies on survives

    def unmet(self, workspace: Path) -> str | None:
        """Why this condition is not satisfied, or None when it is."""
        if not _candidates(workspace, self.name):
            return f"{self.name} was an input to this task and is gone"
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

    Outputs earn nothing. What a request is owed depends on what the data
    turns out to support, and that is not a filesystem question.
    """
    if not postconditions_enabled():
        return []
    ws = Path(workspace).expanduser().resolve()
    out = [Postcondition(name, "present")
           for name, role in sorted(roles.items())
           if role == "input" and _candidates(ws, name)]
    if out:
        log.info("postconditions_derived", present=[c.name for c in out])
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
