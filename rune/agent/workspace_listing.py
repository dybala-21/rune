"""Say what is in the working directory when nothing else will.

The repository map covers code: it is built only for code-category goals,
and it renders symbols from files tree-sitter can parse. A directory of
spreadsheets and notes produces nothing from either gate, so those runs
begin knowing only the path they are standing in.

That is what it costs. Asked for a June revenue summary "for the management
meeting", the run read the one CSV the request named, picked the column
called `revenue`, and reported it — while the data dictionary sitting beside
it said the reporting figure is always `net_revenue`, gross being inclusive
of refunds, per an audit finding. Three runs in three, and the dictionary
was never opened. Nothing was hidden; it was simply never mentioned, and
the run had no reason to suspect the request was ambiguous at all.

So when there is no map, list the directory. Names only — contents stay
where they are, to be read if they turn out to matter. Bounded, because a
listing that runs to thousands of lines buys attention it does not deserve.
"""

from __future__ import annotations

import os
from pathlib import Path

from rune.utils.logger import get_logger

log = get_logger(__name__)

_ENV_FLAG = "RUNE_WORKSPACE_LISTING"
_MAX_ENTRIES = 80
_MAX_DEPTH = 2

_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "node_modules", ".venv", "venv", "env", ".tox", "dist",
    "build", ".next", ".cache", ".idea", ".vscode", "target",
    ".rune-trash", ".rune",
})


def listing_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "1") != "0"


def build_listing(root: str | Path, max_entries: int = _MAX_ENTRIES) -> str:
    """Relative paths under *root*, shallow first, or "" if there is nothing.

    Directories are walked breadth-first so the entries nearest the top —
    the ones a request is most likely to mean — survive the cap.
    """
    if not listing_enabled():
        return ""
    base = Path(root).expanduser()
    if not base.is_dir():
        return ""

    out: list[str] = []
    frontier = [(base, 0)]
    truncated = False
    while frontier and not truncated:
        nxt: list[tuple[Path, int]] = []
        for d, depth in frontier:
            try:
                entries = sorted(d.iterdir(), key=lambda p: (p.is_dir(), p.name))
            except OSError:
                continue
            for p in entries:
                if p.name.startswith(".") or p.name in _SKIP_DIRS:
                    continue
                rel = os.path.relpath(p, base)
                if p.is_dir():
                    out.append(rel + "/")
                    if depth + 1 < _MAX_DEPTH:
                        nxt.append((p, depth + 1))
                else:
                    out.append(rel)
                if len(out) >= max_entries:
                    truncated = True
                    break
            if truncated:
                break
        frontier = nxt

    if not out:
        return ""
    if truncated:
        out.append("… (truncated)")
    return "\n".join(out)


def listing_section(root: str | Path) -> str:
    """The prompt section, or "" when there is nothing worth saying."""
    body = build_listing(root)
    if not body:
        return ""
    log.info("workspace_listing", entries=body.count("\n") + 1)
    return (
        "\n## Files in the working directory\n"
        "Everything below is present and readable. A file you were not asked "
        "about may still define what the request means — units, which column "
        "counts, what a term refers to — so check before assuming.\n"
        f"```\n{body}\n```"
    )
