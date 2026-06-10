"""Best-of-K (verifier-guided rejection sampling) for the CLI one-shot path.

``rune --message "..." --best-of K`` runs K *independent* fresh-context attempts,
each in an isolated tempdir subprocess, then uses RUNE's Evidence Gate as the
selector (see :mod:`rune.agent.rejection_sampler`) to keep the first attempt that
passes a mechanically-extracted success check. The selected attempt's artifacts
are copied back into the real working directory.

best-of-K lifts a weak model by turning model nondeterminism into a *selection*
signal: if a single attempt passes with probability p, sampling K and keeping the
first that verifies succeeds with probability 1-(1-p)^K.

Gated behind the flag: ``K == 1`` is the unchanged single-attempt path with zero
behavior change.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass

from rune.agent.rejection_sampler import (
    make_evidence_gate_verifier,
    sample_parallel,
)
from rune.utils.env import env_int
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Set in every attempt subprocess so a nested invocation can never re-enter
# best-of (defense-in-depth; the child command line also omits ``--best-of``).
RECURSION_GUARD_ENV = "RUNE_IN_BEST_OF"

# Per-attempt wall-clock cap: a single hung attempt (model stall) must not block
# the whole gather forever. Overridable for long tasks.
_ATTEMPT_TIMEOUT_MS_ENV = "RUNE_BESTOF_ATTEMPT_TIMEOUT_MS"
_DEFAULT_ATTEMPT_TIMEOUT_MS = 600_000  # 10 min
_TIMEOUT_RETURNCODE = 124  # mirror coreutils `timeout`


@dataclass
class AttemptArtifact:
    """One best-of-K attempt: its isolated workdir and captured output.

    ``produced`` is the snapshot of top-level entries the attempt itself created,
    taken BEFORE the verifier runs — so verifier side-effects (e.g. a
    ``__pycache__`` from importing the candidate) are never restored into the
    real working directory.
    """

    index: int
    workdir: str
    stdout: str
    returncode: int
    produced: list[str]


async def _run_attempt_subprocess(
    index: int,
    message: str,
    model: str | None,
    provider: str | None,
    seed_from: str | None = None,
) -> AttemptArtifact:
    """Run one fresh-context attempt in an isolated tempdir subprocess.

    Each attempt is a separate ``python -m rune.cli.main --message ...`` process
    with its own working directory and a copied env carrying the recursion guard.
    The child command intentionally OMITS ``--best-of`` so it takes the plain
    single-attempt path; the env flag is a second guard in case it ever leaks.

    ``seed_from`` (set in --include-cwd mode) copies that dir into the workdir
    first so the agent can edit existing files; ``produced`` then becomes the set
    of files CHANGED vs the seed, not every top-level entry.
    """
    workdir = tempfile.mkdtemp(prefix=f"rune_bestof_{index}_")

    seed_manifest: dict[str, tuple[float, int]] | None = None
    if seed_from:
        try:
            _seed_workdir(seed_from, workdir)
            seed_manifest = _tree_manifest(workdir)
        except Exception as exc:
            log.warning("bestof_seed_failed", index=index, error=str(exc)[:120])

    def _produced() -> list[str]:
        if seed_manifest is not None:
            return _changed_vs_seed(workdir, seed_manifest)
        return _snapshot_produced(workdir)

    env = dict(os.environ)
    env[RECURSION_GUARD_ENV] = "1"  # recursion guard

    cmd = [sys.executable, "-m", "rune.cli.main", "--message", message]
    if model:
        cmd += ["--model", model]
    if provider:
        cmd += ["--provider", provider]

    timeout_s = max(
        1.0, env_int(_ATTEMPT_TIMEOUT_MS_ENV, _DEFAULT_ATTEMPT_TIMEOUT_MS) / 1000.0
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=workdir,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except Exception as exc:  # spawn failure — treat as a failed attempt
        log.warning("bestof_attempt_spawn_error", index=index, error=str(exc)[:120])
        return AttemptArtifact(
            index=index, workdir=workdir, stdout="", returncode=1, produced=[]
        )

    try:
        stdout_b, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        returncode = proc.returncode or 0
    except TimeoutError:
        # A stalled attempt must not hang the whole best-of gather. Kill it and
        # record a failed attempt (its partial workdir is left for the verifier,
        # which will almost certainly reject it).
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        log.warning("bestof_attempt_timeout", index=index, timeout_s=timeout_s)
        return AttemptArtifact(
            index=index,
            workdir=workdir,
            stdout="",
            returncode=_TIMEOUT_RETURNCODE,
            produced=_produced(),
        )

    stdout = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
    # Snapshot what the attempt produced BEFORE verification runs, so verifier
    # side-effects (a __pycache__ from importing the candidate) aren't restored.
    produced = _produced()
    log.info(
        "bestof_attempt_done",
        index=index,
        returncode=returncode,
        workdir=workdir,
        produced=produced,
    )
    return AttemptArtifact(
        index=index,
        workdir=workdir,
        stdout=stdout,
        returncode=returncode,
        produced=produced,
    )


# Verification/runtime byproducts that should never be restored even if a tool
# created them inside the attempt's workdir.
_RESTORE_DENYLIST = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache"})


def _snapshot_produced(workdir: str) -> list[str]:
    """Top-level names the attempt created, minus known build/cache byproducts."""
    return sorted(
        name for name in os.listdir(workdir) if name not in _RESTORE_DENYLIST
    )


# --- seeded mode (--include-cwd): copy the working tree into each attempt so the
# agent can EDIT existing files, then restore only what it changed (diff vs seed).

_SEED_IGNORE_PATTERNS = (
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
    "*.pyc", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".rune-bestof-*",
)
_SEED_IGNORE = shutil.ignore_patterns(*_SEED_IGNORE_PATTERNS)

# Refuse to seed a cwd larger than this (× K copies would otherwise exhaust
# disk). Overridable for genuinely large repos.
_SEED_MAX_MB_ENV = "RUNE_BESTOF_SEED_MAX_MB"
_DEFAULT_SEED_MAX_MB = 200
_SEED_MAX_FILES = 20_000


def _seed_footprint(src: str) -> tuple[int, int]:
    """Count (files, total_bytes) that seeding would copy (ignores applied)."""
    import fnmatch

    def ignored(name: str) -> bool:
        return any(fnmatch.fnmatch(name, p) for p in _SEED_IGNORE_PATTERNS)

    files = 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if not ignored(d)]
        for fn in filenames:
            if ignored(fn):
                continue
            files += 1
            try:
                total += os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                pass
    return files, total


def _check_seed_size(src: str) -> str | None:
    """Return an error message if seeding ``src`` would be too large, else None."""
    max_mb = env_int(_SEED_MAX_MB_ENV, _DEFAULT_SEED_MAX_MB)
    files, total = _seed_footprint(src)
    if total > max_mb * 1024 * 1024 or files > _SEED_MAX_FILES:
        return (
            f"--include-cwd would copy {files} files / {total / 1024 / 1024:.0f} MB "
            f"into EACH attempt (× K copies). That exceeds the limit "
            f"({max_mb} MB / {_SEED_MAX_FILES} files). Run from a smaller dir, add "
            f"large paths to a .gitignore-style layout, or raise {_SEED_MAX_MB_ENV}."
        )
    return None


def _seed_workdir(src: str, workdir: str) -> None:
    """Copy the cwd tree into an attempt's workdir (minus VCS/build/cache cruft)."""
    shutil.copytree(src, workdir, ignore=_SEED_IGNORE, dirs_exist_ok=True, symlinks=False)


def _tree_manifest(root: str) -> dict[str, tuple[float, int]]:
    """Map each file's relpath -> (mtime, size). Used to diff seed vs final."""
    manifest: dict[str, tuple[float, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _RESTORE_DENYLIST]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            try:
                st = os.stat(full)
            except OSError:
                continue
            manifest[os.path.relpath(full, root)] = (st.st_mtime, st.st_size)
    return manifest


def _changed_vs_seed(root: str, seed: dict[str, tuple[float, int]]) -> list[str]:
    """Relpaths that are new or modified (mtime/size) vs the seed manifest.

    Deletions are intentionally NOT reported — best-of never deletes user files.
    """
    changed: list[str] = []
    for rel, (mtime, size) in _tree_manifest(root).items():
        prev = seed.get(rel)
        if prev is None or prev != (mtime, size):
            changed.append(rel)
    # If EVERY seeded file looks changed, the mtime/size diff is likely broken
    # (e.g. mtime not preserved on copy) rather than the agent having rewritten
    # the whole tree. Restore still backs up originals, so this is recoverable —
    # but warn loudly so a whole-tree overwrite is visible.
    if seed and len(changed) >= len(seed) and all(r in changed for r in seed):
        log.warning("bestof_seed_diff_suspicious", changed=len(changed), seeded=len(seed))
    return sorted(changed)


def _restore_changed(
    workdir: str, dest: str, relpaths: list[str]
) -> tuple[list[str], str | None]:
    """Copy changed ``relpaths`` from a seeded workdir back into ``dest``.

    Overwriting IS intended here (the agent edited a copy of the user's tree),
    but it's destructive, so every pre-existing target is first backed up into a
    fresh ``.rune-bestof-backup-*`` dir for undo. Returns ``(restored, backup_dir)``.
    """
    restored: list[str] = []
    backup_dir: str | None = None
    for rel in relpaths:
        src = os.path.join(workdir, rel)
        if not os.path.exists(src):
            continue
        dst = os.path.join(dest, rel)
        if os.path.exists(dst):
            if backup_dir is None:
                backup_dir = tempfile.mkdtemp(prefix=".rune-bestof-backup-", dir=dest)
            bdst = os.path.join(backup_dir, rel)
            os.makedirs(os.path.dirname(bdst), exist_ok=True)
            shutil.copy2(dst, bdst)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy2(src, dst)
        restored.append(rel)
    return restored, backup_dir


def _restore_artifacts(
    workdir: str, dest: str, names: list[str]
) -> tuple[list[str], list[str]]:
    """Copy the attempt's produced ``names`` from ``workdir`` into ``dest``.

    ``names`` is the pre-verification snapshot (see ``AttemptArtifact.produced``),
    so only what the attempt itself created is restored — never verifier
    byproducts.

    SAFETY: an attempt runs in an isolated temp dir and we copy its output into
    the real cwd, which may already contain user files. We must NOT silently
    clobber them — a name that already exists in ``dest`` is SKIPPED and returned
    as a conflict so the caller can warn. Returns ``(copied, skipped)``.
    """
    copied: list[str] = []
    skipped: list[str] = []
    for name in names:
        src = os.path.join(workdir, name)
        if not os.path.exists(src):  # vanished/never-created — skip defensively
            continue
        dst = os.path.join(dest, name)
        if os.path.exists(dst):  # never overwrite an existing user path
            skipped.append(name)
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        copied.append(name)
    return copied, skipped


def _preserve_skipped(workdir: str, dest: str, skipped: list[str]) -> str | None:
    """Save winner files that couldn't be restored (name collisions) so the K
    attempts aren't wasted.

    Restore never overwrites existing cwd files, but the selected attempt's work
    must not be silently discarded. Copy the skipped (colliding) artifacts into a
    fresh ``.rune-bestof-*`` dir inside ``dest`` (dotfile → ignored by the
    non-empty-cwd warning) and return its path so the caller can point the user
    at it to diff/adopt. Returns ``None`` if nothing was preserved.
    """
    if not skipped:
        return None
    preserve = tempfile.mkdtemp(prefix=".rune-bestof-", dir=dest)
    saved = False
    for name in skipped:
        src = os.path.join(workdir, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(preserve, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        saved = True
    if not saved:
        shutil.rmtree(preserve, ignore_errors=True)
        return None
    return preserve


def _cleanup(artifacts: list[AttemptArtifact]) -> None:
    for a in artifacts:
        shutil.rmtree(a.workdir, ignore_errors=True)


async def _record_winner(
    message: str, answer: str, changed_files: list[str] | None = None
) -> bool:
    """Record the selected winner as one success episode.

    Attempt subprocesses are ephemeral (no learning), so without this a
    successful best-of solve teaches the self-improving loop nothing. Record
    exactly one episode — for the verifier-confirmed winner — in the parent
    (non-ephemeral) process. Success here is verifier-gated rather than the
    agent's self-report, so it won't record a wrong "success". Returns True if
    recorded.
    """
    try:
        from rune.agent.agent_context import (
            PostProcessInput,
            PrepareContextOptions,
            post_process_agent_result,
            prepare_agent_context,
        )

        ctx = await prepare_agent_context(
            PrepareContextOptions(goal=message, channel="cli")
        )
        goal_type: str | None = None
        try:
            from rune.agent.goal_classifier import classify_goal
            goal_type = (await classify_goal(message)).goal_type
        except Exception:
            pass

        await post_process_agent_result(
            PostProcessInput(
                context=ctx,
                success=True,  # verifier-confirmed
                answer=answer,
                reason="completed",
                evidence_gate=None,
                classification_hint=goal_type,
                changed_files=list(changed_files or []),
            )
        )
        from rune.memory.manager import get_memory_manager
        await get_memory_manager().promote_memories()
        log.info("bestof_winner_recorded")
        return True
    except Exception as exc:  # best-effort: never fail the run over learning
        log.warning("bestof_winner_record_failed", error=str(exc)[:120])
        return False


# Cap LLM rule-gen calls per best-of run (distinct failures learned from).
_MAX_FAILURE_RULES = 3


async def _learn_from_failures(message: str, evidence: list[str]) -> list[str]:
    """Learn correctness rules from the failed attempts' verifier evidence.

    best-of already computed the Evidence Gate verdict for every attempt, so the
    failed candidates' mismatch evidence is a free crisp-failure signal — the
    detection the default path lacks. Recording prevention rules here (in the
    non-ephemeral parent) lets a future task avoid the same mistakes (the rules
    inject via semantic retrieval).

    Learn from all distinct failure evidences (capped), not just the first: the K
    attempts can fail for different reasons — some structurally (missing/broken
    file → "verify files exist") and some on the actual logic (wrong value →
    "division rounding"). Learning from each lets the semantic retriever later
    pick whichever rule is relevant to the future task; crisp-failure signature
    dedup collapses near-duplicates, so this stays bounded. Best-effort; never
    breaks the run.
    """
    seen: set[str] = set()
    distinct: list[str] = []
    for e in evidence:
        e = (e or "").strip()
        if e and e not in seen:
            seen.add(e)
            distinct.append(e)
    if not distinct:
        return []
    try:
        domain = "code_modify"
        try:
            from rune.agent.goal_classifier import classify_goal
            domain = (await classify_goal(message)).goal_type or domain
        except Exception:
            pass
        from rune.memory.rule_learner import learn_from_crisp_failure
        learned: list[str] = []
        for ev in distinct[:_MAX_FAILURE_RULES]:
            try:
                key = await learn_from_crisp_failure("best_of_verifier", ev, domain)
            except Exception as exc:
                log.warning("bestof_learn_one_failed", error=str(exc)[:120])
                continue
            if key:
                learned.append(key)
        if learned:
            log.info("bestof_learned_from_failures", keys=learned)
        return learned
    except Exception as exc:
        log.warning("bestof_learn_failed", error=str(exc)[:120])
        return []


# Reporter: (stdout, solved, selected_index, pass_count, k, copied) -> None
Reporter = Callable[..., None]


async def _best_of_async(
    message: str,
    k: int,
    model: str | None,
    provider: str | None,
    *,
    report: Reporter,
    seed_cwd: bool = False,
) -> int:
    """Core best-of-K flow. Returns a process exit code (0 solved, 1 unsolved).

    ``seed_cwd`` (--include-cwd) copies the working tree into each attempt so the
    agent can edit existing files; restore then writes back only the changed
    files (overwriting, with a backup), instead of the greenfield new-files copy.
    """
    verify_cwd = await make_evidence_gate_verifier(message)
    has_check = bool(getattr(verify_cwd, "has_check", True))

    dest = os.getcwd()
    seed_from = dest if seed_cwd else None

    # Cap concurrent attempt subprocesses: each is a full agent run, so a large
    # K must not spawn K heavyweight processes at once. Mirrors the workflow
    # engine's min(cores-2, ...) policy.
    cap = max(1, min(k, (os.cpu_count() or 4) - 2))
    sem = asyncio.Semaphore(cap)

    async def run_attempt(i: int) -> AttemptArtifact:
        async with sem:
            return await _run_attempt_subprocess(
                i, message, model, provider, seed_from=seed_from
            )

    async def verify(artifact: AttemptArtifact) -> bool:
        # Cap verifier subprocesses too: sample_parallel gathers all K verifies
        # at once, each an Evidence-Gate check subprocess.
        async with sem:
            return await verify_cwd(artifact.workdir)

    res = await sample_parallel(run_attempt, verify, k)
    artifacts: list[AttemptArtifact] = [a.candidate for a in res.attempts]

    # Learn a correctness rule from any failed attempts' verifier evidence
    # (fires whether or not a winner was found — every failed candidate is a
    # detected mistake the default path would have missed).
    ev_map = getattr(verify_cwd, "evidence_by_cwd", {}) or {}
    failed_ev = [
        ev_map.get(a.candidate.workdir, "") for a in res.attempts if not a.passed
    ]
    await _learn_from_failures(message, failed_ev)

    try:
        if res.solved and res.selected is not None:
            selected: AttemptArtifact = res.selected
            if seed_cwd:
                # Seeded mode: write back the agent's edits (overwrite intended),
                # backing up originals for undo.
                copied, backup_dir = _restore_changed(
                    selected.workdir, dest, selected.produced
                )
                skipped, preserved_dir = [], None
            else:
                copied, skipped = _restore_artifacts(
                    selected.workdir, dest, selected.produced
                )
                # Don't discard the winner on collision — save it for the user.
                preserved_dir = _preserve_skipped(selected.workdir, dest, skipped)
                backup_dir = None
            # Learn from the verifier-confirmed winner (1 episode).
            await _record_winner(message, selected.stdout, selected.produced)
            report(
                selected.stdout,
                solved=True,
                selected_index=res.selected_index,
                pass_count=res.pass_count,
                k=k,
                copied=copied,
                skipped=skipped,
                preserved_dir=preserved_dir,
                backup_dir=backup_dir,
                has_check=has_check,
                no_artifact=0,
            )
            return 0

        # No attempt passed. Break the 0/K down so the user can tell WHY:
        #  - no mechanical check could be built  → best-of-K cannot select at all
        #  - attempts produced no files          → generator didn't write artifacts
        #  - attempts wrote files but failed      → generator produced wrong output
        # Surface attempt #0's output as a best-effort but DO NOT restore it
        # (unverified); never silently drop the K-1 candidates.
        best = artifacts[0] if artifacts else None
        no_artifact = sum(1 for a in artifacts if not a.produced)
        report(
            best.stdout if best else "",
            solved=False,
            selected_index=None,
            pass_count=0,
            k=k,
            copied=[],
            skipped=[],
            has_check=has_check,
            no_artifact=no_artifact,
        )
        return 1
    finally:
        _cleanup(artifacts)


def run_best_of(
    message: str,
    k: int,
    model: str | None = None,
    provider: str | None = None,
    seed_cwd: bool = False,
) -> None:
    """Synchronous entry point for the CLI: run best-of-K and print the outcome.

    Raises ``typer.Exit(1)`` when no attempt passes the verifier so the one-shot
    command exits non-zero (mirrors a failed single run). ``seed_cwd`` enables
    --include-cwd mode (edit existing files instead of greenfield new files).
    """
    from rich.console import Console

    console = Console(stderr=True)

    if seed_cwd:
        # Seeding copies the cwd into each of K attempts and all K workdirs
        # persist until cleanup, so a large cwd can exhaust disk. Refuse before
        # doing any work.
        _err = _check_seed_size(os.getcwd())
        if _err:
            console.print(f"[red]{_err}[/red]")
            import typer

            raise typer.Exit(2)
        # Seeded mode edits a copy of the working tree and writes changes back
        # (overwriting, with a backup). Tell the user it's destructive-by-design.
        console.print(
            "[dim]best-of --include-cwd: each attempt edits a copy of the working "
            "tree; the winner's changes are written back (originals backed up).[/dim]"
        )
    else:
        # B-warn: greenfield attempts run in isolated EMPTY temp dirs — they do
        # NOT see the working tree. Warn when run from a non-empty dir so this
        # isn't a silent failure; suggest --include-cwd for edit tasks.
        try:
            if any(not n.startswith(".") for n in os.listdir(os.getcwd())):
                console.print(
                    "[dim]best-of: each attempt runs in an isolated empty temp dir "
                    "(working tree NOT copied in). Suited to new-file tasks; use "
                    "--include-cwd to edit existing files.[/dim]"
                )
        except OSError:
            pass

    def report(
        stdout: str,
        *,
        solved: bool,
        selected_index: int | None,
        pass_count: int,
        k: int,
        copied: list[str],
        skipped: list[str] | None = None,
        preserved_dir: str | None = None,
        backup_dir: str | None = None,
        has_check: bool = True,
        no_artifact: int = 0,
    ) -> None:
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n", flush=True)
        if solved:
            names = ", ".join(copied) if copied else "—"
            console.print(
                f"[dim]best-of-{k}: selected attempt #{selected_index} "
                f"({pass_count}/{k} passed verifier); "
                f"restored {len(copied)} item(s): {names}[/dim]"
            )
            if backup_dir:
                console.print(
                    f"[dim]best-of: originals backed up to "
                    f"{os.path.relpath(backup_dir)}/ before overwrite.[/dim]"
                )
            if skipped:
                where = (
                    f" Winner saved to {os.path.relpath(preserved_dir)}/ — "
                    f"diff against your version."
                    if preserved_dir
                    else ""
                )
                console.print(
                    f"[yellow]best-of: NOT overwritten (already exist in cwd): "
                    f"{', '.join(skipped)}.{where}[/yellow]"
                )
        elif not has_check:
            console.print(
                f"[yellow]best-of-{k}: no mechanical success check could be built "
                f"for this task, so the verifier cannot select a candidate "
                f"(best-of-K only helps verifiable tasks). Showing attempt #0 "
                f"unverified, nothing restored.[/yellow]"
            )
        else:
            wrote = k - no_artifact
            console.print(
                f"[yellow]best-of-{k}: no attempt passed the verifier (0/{k}); "
                f"{no_artifact}/{k} produced no files (generator didn't write "
                f"artifacts), {wrote}/{k} wrote files but failed the check. "
                f"Showing attempt #0 unverified, nothing restored.[/yellow]"
            )

    exit_code = asyncio.run(
        _best_of_async(message, k, model, provider, report=report, seed_cwd=seed_cwd)
    )
    if exit_code != 0:
        import typer

        raise typer.Exit(exit_code)
