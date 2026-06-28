"""Deterministic worker acceptance (G2 verification, invariant I6, Tier 1).

The orchestrator decides whether an isolated worker achieved its goal from a
deterministic check over its change-set — not from the worker's self-reported
``ok`` (LLM self-judgment is unreliable: arXiv:2310.01798, 2406.01297, 2404.17140).
This closes the "silent no-op" hole (``ok=True`` while producing nothing).

Tier 1 is local: decidable from one worker's own change-set. Compile/test checks
that need the merged tree are Tier 2 (post-merge), not here — a per-worker
compile would false-reject a worker whose siblings haven't merged yet. See
``docs/design/worktree-subagent-verification.md`` §4-A.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field


@dataclass(slots=True)
class Acceptance:
    """Tier-1 (local) acceptance criteria for one worker.

    All checks are decidable from the worker's own change-set + result. Leave at
    defaults for the conservative floor ("must change at least one file").
    """
    expect_changes: bool = True
    # repo-root-relative POSIX globs; each must match at least one changed path.
    expect_paths: list[str] = field(default_factory=list)
    # weak fallback for text-only tasks; trivially gameable, never use alone.
    require_nonempty_answer: bool = False


# Conservative default used when a WorkerSpec carries no explicit acceptance:
# "a worker that changed nothing did not do its job." This single rule catches
# every silent no-op observed in practice.
DEFAULT_ACCEPTANCE = Acceptance()


@dataclass(slots=True)
class AcceptanceVerdict:
    ok: bool
    reason: str = ""


def evaluate_local(
    acc: Acceptance | None,
    changed_paths: set[str],
    answer: str,
) -> AcceptanceVerdict:
    """Deterministic Tier-1 verdict. ``changed_paths`` = the worker's change-set
    paths (repo-relative POSIX); ``answer`` = the worker's final text."""
    acc = acc or DEFAULT_ACCEPTANCE

    if acc.expect_changes and not changed_paths:
        return AcceptanceVerdict(False, "no file changes produced")

    for pat in acc.expect_paths:
        if not any(fnmatch.fnmatch(p, pat) for p in changed_paths):
            return AcceptanceVerdict(False, f"expected path not produced: {pat}")

    if acc.require_nonempty_answer and not (answer or "").strip():
        return AcceptanceVerdict(False, "expected a non-empty answer")

    return AcceptanceVerdict(True, "")
