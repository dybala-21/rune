"""Gated Skill Learning — replay corpus capture (T1-1).

To A/B a skill offline we need *reproducible* tasks: the pre-task workspace
state plus a deterministic check. This module snapshots a git workspace at a
given commit into a persistent corpus directory and records a
:class:`ReplayTask` row.

Capture must happen at task *start* (the pre-mutation state) so the replayed
agent actually redoes the work; a post-run snapshot would already contain the
result and the check would pass trivially. Callers (a run-start hook or the
daemon) invoke :func:`capture_replay_snapshot` before the task mutates the
tree. Gated by ``skills.capture_replay`` (default off — snapshots cost disk).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from rune.utils.logger import get_logger

log = get_logger(__name__)


def _corpus_root() -> Path:
    from rune.utils.paths import rune_home
    root = rune_home() / "replay_corpus"
    root.mkdir(parents=True, exist_ok=True)
    return root


def capture_head_ref(cwd: str | None = None) -> str | None:
    """Resolve the current git HEAD commit, or None if *cwd* is not a repo.

    Call this at task *start* and pass the result to
    :func:`capture_replay_snapshot` at task end — so the snapshot is the
    pre-task commit even if the agent committed during the run.
    """
    cwd = cwd or os.getcwd()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
            text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _snapshot_ref(cwd: str, ref: str, dest: Path) -> bool:
    """Archive a git *ref* tree (committed state only) into *dest*."""
    try:
        dest.mkdir(parents=True, exist_ok=True)
        # `git archive <ref> | tar -x -C dest` — committed tree at ref, no .git,
        # no uncommitted edits. Reproducible pre-task state.
        archive = subprocess.run(
            ["git", "archive", ref], cwd=cwd, capture_output=True, timeout=30,
        )
        if archive.returncode != 0:
            return False
        extract = subprocess.run(
            ["tar", "-x", "-C", str(dest)], input=archive.stdout, timeout=30,
        )
        return extract.returncode == 0
    except Exception as exc:
        log.debug("replay_snapshot_failed", error=str(exc)[:120])
        return False


def capture_replay_snapshot(
    goal: str,
    skill_name: str,
    *,
    store: object,
    cwd: str | None = None,
    head_ref: str | None = None,
) -> bool:
    """Snapshot a git workspace at *head_ref* and record a replay task.

    Pass ``head_ref`` from :func:`capture_head_ref` taken at task start so the
    snapshot is the pre-task commit. Falls back to current HEAD when omitted.
    Returns True if a task was captured. Best-effort and gated; never raises.
    """
    try:
        from rune.config import get_config
        if not getattr(get_config().skills, "capture_replay", False):
            return False
    except Exception:
        return False

    cwd = cwd or os.getcwd()
    ref = head_ref or capture_head_ref(cwd)
    if ref is None:
        log.debug("replay_capture_skipped_not_git", cwd=cwd)
        return False

    # Deterministic check: the project's own test/verify command.
    try:
        from rune.agent.auto_verify import detect_test_command, detect_verify_command
        cmd = detect_test_command(cwd) or detect_verify_command(cwd)
    except Exception:
        cmd = None
    if not cmd:
        log.debug("replay_capture_skipped_no_check", cwd=cwd)
        return False
    check_cmd = " ".join(cmd)

    dest = _corpus_root() / f"{skill_name}-{ref[:12]}"
    if not dest.exists() and not _snapshot_ref(cwd, ref, dest):
        return False

    try:
        store.add_replay_task(
            skill_name=skill_name, goal=goal,
            workspace_ref=str(dest), check_cmd=check_cmd,
        )
        log.info("replay_task_captured", skill=skill_name, dest=str(dest))
        return True
    except Exception as exc:
        log.debug("replay_capture_record_failed", error=str(exc)[:120])
        return False
