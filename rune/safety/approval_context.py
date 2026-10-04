"""Bind approval to one capability call; nested calls need their own grant."""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

_APPROVED: ContextVar[bool] = ContextVar("rune_call_approved", default=False)


@dataclass(slots=True)
class _Grant:
    fingerprint: str
    consumed: bool = False
    revisions: dict = field(default_factory=dict)


_GRANT: ContextVar[_Grant | None] = ContextVar("rune_action_grant", default=None)


def _fingerprint(capability: str, params: dict[str, Any]) -> str:
    payload = json.dumps([capability, params], sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@contextlib.contextmanager
def approval_granted(capability: str = "", params: dict[str, Any] | None = None, *, revisions: dict | None = None) -> Iterator[None]:
    """Mark the current call as one the approval gate has already cleared."""
    from copy import deepcopy
    scoped = _Grant(_fingerprint(capability, params or {}), revisions=deepcopy(revisions or {})) if capability else None
    token = _APPROVED.set(True)
    grant = _GRANT.set(scoped)
    try:
        yield
    finally:
        _GRANT.reset(grant)
        _APPROVED.reset(token)


def consume_approval(capability: str, params: dict[str, Any]) -> bool:
    grant = _GRANT.get()
    if grant is None or grant.consumed or grant.fingerprint != _fingerprint(capability, params):
        return False
    grant.consumed = True
    return True


def was_approved() -> bool:
    """Whether an approval gate cleared the call currently being executed."""
    return _APPROVED.get()


def approval_revisions() -> dict:
    from copy import deepcopy
    grant = _GRANT.get()
    return deepcopy(grant.revisions) if grant is not None else {}


@contextlib.contextmanager
def approval_required() -> Iterator[None]:
    """Start a separate action without inheriting the caller's approval."""
    token = _APPROVED.set(False)
    grant = _GRANT.set(None)
    try:
        yield
    finally:
        _GRANT.reset(grant)
        _APPROVED.reset(token)
