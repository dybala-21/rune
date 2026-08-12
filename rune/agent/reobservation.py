"""Look at what a bulk command left behind, before calling the work done.

A sweep reports nothing about what survived it: `rm` over a glob prints no
output, and the summary that follows is written from what the run meant to
do. Asked to clear forty object files and thirty logs, runs cleared the logs,
never touched `build/`, and called both done — some inventing a `rm
build/*.o` they never issued.

The missing piece is observation, not judgment: shown a fresh listing of the
directories it swept, the same model resumes the cleanup. So the listing is
handed over without comment. It cannot be wrong and it argues for nothing,
which matters because the neighbouring failure is a run that deletes past
its brief, and "there are still files here" would push straight into it.
Measured on gbench2 H6, 36 pairs: 6/36 -> 20/36, and no fired run ever lost
a file it was meant to keep.

The trigger counts what a directory absorbed, whichever tool did it. Scoping
it to shell globs would have measured nothing: the approval gate refuses
`find | xargs rm` when nobody is there to approve it, so the sweep arrives
as a run of single-file deletions instead. Those need a threshold — a
handful of removals is not a sweep, and each one reported its own result.
"""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Iterable, Mapping
from pathlib import Path

from rune.utils.logger import get_logger

log = get_logger(__name__)

_ENV_FLAG = "RUNE_REOBSERVE"
_MAX_DIRS = 4
_MAX_NAMES = 20
# Single-file removals in one directory before it counts as a sweep. Below
# this a run is picking off named files, and each removal reported itself.
_REPEAT_THRESHOLD = 5
_GLOB_CHARS = ("*", "?", "[")
# Commands that remove or relocate what they are pointed at. `cp` is absent
# on purpose: it adds files, and the additions are named in the command.
_DELETERS = frozenset({"rm", "unlink", "rmdir", "trash", "shred"})
_CHAIN_SPLIT = re.compile(r"&&|\|\||;|\n")
_BRACE = re.compile(r"\{([^{}]*)\}")
_EXEC_FLAGS = frozenset({"-exec", "-execdir", "-ok", "-okdir"})


def reobservation_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "1") != "0"


def _argv(segment: str) -> list[str]:
    """Tokens of one pipeline segment; quotes stripped, globs left intact."""
    try:
        return shlex.split(segment)
    except ValueError:            # unbalanced quote — take the crude split
        return segment.split()


def _many(token: str) -> bool:
    """Does one token stand for more than one file?

    Globs are the obvious form; brace expansion is the one that gets missed,
    and `rm logs/app_{1..30}.log` takes thirty files without a `*` anywhere.
    The bare `{}` of `find -exec rm {} +` is a placeholder for one path at a
    time, so a brace counts only when something separates its ends.
    """
    if any(c in token for c in _GLOB_CHARS):
        return True
    m = _BRACE.search(token)
    return bool(m and ("," in m.group(1) or ".." in m.group(1)))


def _has_glob(tokens: Iterable[str]) -> bool:
    return any(_many(t) for t in tokens)


def _recursive(argv: list[str]) -> bool:
    return any(t == "--recursive" or (t.startswith("-") and not t.startswith("--")
                                      and ("r" in t or "R" in t))
               for t in argv[1:])


def _is_bulk(argv: list[str]) -> bool:
    """Does this segment change many files at once?"""
    if not argv:
        return False
    name = Path(argv[0]).name
    if name == "find":
        if "-delete" in argv:
            return True
        # What -exec runs is the token straight after it; anything looser
        # reads `find . -exec grep rm {} +` as a deletion.
        return any(Path(b).name in _DELETERS
                   for a, b in zip(argv, argv[1:], strict=False)
                   if a in _EXEC_FLAGS)
    if name in _DELETERS:
        return _has_glob(argv[1:]) or _recursive(argv)
    if name == "mv":
        return _has_glob(argv[1:])
    if name == "xargs":
        return any(Path(t).name in _DELETERS for t in argv[1:])
    return False


def _operands(argv: list[str]) -> list[str]:
    """The paths a segment names, without its flags.

    `find` puts its starting points first and everything after the first
    predicate belongs to the predicate, so it stops there. `xargs` names no
    paths of its own — its input comes from the segment feeding it, which is
    read separately.
    """
    name = Path(argv[0]).name
    if name == "xargs":
        return []
    if name == "find":
        out = []
        for t in argv[1:]:
            if t.startswith("-"):
                break
            out.append(t)
        return out
    return [t for t in argv[1:] if not t.startswith("-")]


def _target_dir(operand: str, base: Path) -> Path | None:
    """The directory an operand lived in, as it exists now.

    A pattern stands for the directory holding it. A path that is gone — the
    usual outcome of `rm -r` — resolves to the nearest parent still there,
    which is where its absence shows.
    """
    raw = str(Path(operand).parent) if _many(operand) else operand
    try:
        p = (base / Path(raw).expanduser()).resolve()
    except OSError:
        return None
    while not p.is_dir() and p != p.parent:
        p = p.parent
    if not p.is_dir():
        return None
    try:                          # stay inside the workspace
        p.relative_to(base)
    except ValueError:
        return None
    return p


def bulk_targets(command: str, cwd: str | Path) -> set[str]:
    """Directories a shell command changed wholesale, if it changed any.

    A pipeline is read as one thing: `find build -name '*.o' | xargs rm`
    deletes, and the directory it deletes from is named in the half that
    does not.
    """
    if not reobservation_enabled() or not command.strip():
        return set()
    try:
        base = Path(cwd).expanduser().resolve()
    except OSError:
        return set()
    found: set[str] = set()
    for chain in _CHAIN_SPLIT.split(command):
        segments = [_argv(s) for s in chain.split("|")]
        if not any(_is_bulk(a) for a in segments):
            continue
        for argv in segments:
            if not argv:
                continue
            for op in _operands(argv):
                d = _target_dir(op, base)
                if d is not None:
                    found.add(str(d))
    return found


def mutation_dir(path: str, cwd: str | Path) -> str | None:
    """The directory one file-level removal touched, or None if out of scope.

    Kept separate from the shell parser because the caller has the path
    already — there is nothing to parse, only the same workspace bound and
    the same walk up to whatever still exists.
    """
    if not reobservation_enabled() or not str(path).strip():
        return None
    try:
        base = Path(cwd).expanduser().resolve()
    except OSError:
        return None
    d = _target_dir(str(path), base)
    return str(d) if d is not None else None


def repeated_mutation_dirs(counts: Mapping[str, int]) -> set[str]:
    """Directories changed one file at a time, often enough to be a sweep."""
    if not reobservation_enabled():
        return set()
    return {d for d, n in counts.items() if n >= _REPEAT_THRESHOLD}


def _entries(d: Path) -> list[str] | None:
    """Names directly in *d*, or None if it cannot be read.

    Dot entries stay out: the trash a guarded delete writes to lives in
    `.rune-trash`, and listing what was rescued alongside what survived
    reads as though nothing had been removed at all.
    """
    try:
        return sorted(p.name + ("/" if p.is_dir() else "")
                      for p in d.iterdir() if not p.name.startswith("."))
    except OSError:
        return None


def observation_note(dirs: Iterable[str], cwd: str | Path) -> str:
    """The listing to hand over, or "" when there is nothing to show."""
    if not reobservation_enabled():
        return ""
    try:
        base = Path(cwd).expanduser().resolve()
    except OSError:
        return ""
    blocks: list[str] = []
    total = 0
    for d in sorted(dirs)[:_MAX_DIRS]:
        path = Path(d)
        names = _entries(path)
        if names is None:
            continue
        try:
            label = str(path.relative_to(base)) or "."
        except ValueError:
            label = str(path)
        total += len(names)
        if not names:
            blocks.append(f"`{label}/` — empty")
            continue
        shown = names[:_MAX_NAMES]
        tail = (f"… ({len(names)} entries, {len(shown)} shown)"
                if len(names) > len(shown) else f"({len(names)} entries)")
        listed = "\n".join(shown)
        blocks.append(f"`{label}/`\n{listed}\n{tail}")
    if not blocks:
        return ""
    log.info("reobservation_listing", dirs=len(blocks), entries=total)
    return (
        "## The directories this run changed, as they are now\n"
        "Read from disk just now, after the commands above ran.\n\n"
        + "\n\n".join(blocks)
    )
