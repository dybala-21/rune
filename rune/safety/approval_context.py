"""Records that the caller's approval gate has already cleared this tool call.

Two gates see the same command: the tool adapter asks Guardian and prompts the
user, then the bash capability runs the execution policy again. Without this the
second gate cannot tell that the first already asked, and refuses what the user
just approved.

A context variable rather than a parameter — approval is generic to every
capability, so a bash-only argument would belong in the wrong place. Async tasks
each get their own copy, so one run's approval cannot leak into another's.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar

_APPROVED: ContextVar[bool] = ContextVar("rune_call_approved", default=False)


@contextlib.contextmanager
def approval_granted() -> Iterator[None]:
    """Mark the current call as one the approval gate has already cleared."""
    token = _APPROVED.set(True)
    try:
        yield
    finally:
        _APPROVED.reset(token)


def was_approved() -> bool:
    """Whether an approval gate cleared the call currently being executed."""
    return _APPROVED.get()
