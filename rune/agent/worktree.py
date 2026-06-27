"""Give each parallel worker its own workspace so it can't clobber the others.

Two backends:
- git worktree (normal repos): base is a snapshot commit of the *current
  working tree* (so workers see uncommitted edits, not HEAD); changes collected
  via `git diff`.
- copy (non-git / no-HEAD / bare / submodule / LFS): copytree + a path->hash
  manifest as the base.

Gitignored dep dirs (.venv, node_modules, ...) are symlinked in so builds work.
create/collect/cleanup only — merging lives in merge.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

from rune.utils.logger import get_logger

log = get_logger(__name__)

_WORKTREE_PREFIX = "rune-iso-"
# Serialise git metadata mutations (worktree add/remove touch .git/worktrees +
# index.lock); the worker run itself is unaffected.
_git_lock = asyncio.Lock()


@dataclass(slots=True)
class IsolatedWorkspace:
    path: str                       # the worker's cwd
    mode: str                       # "worktree" | "copy"
    repo: str                       # the main repo/workspace root
    base_ref: str = ""              # snapshot commit SHA (worktree mode)
    base_manifest: dict[str, str] = field(default_factory=dict)  # copy mode


@dataclass(slots=True)
class FileChange:
    op: str                         # "modified" | "added" | "deleted"
    content: bytes | None = None    # None for deleted
    mode: int = 0o644               # POSIX mode bits
    is_symlink: bool = False
    symlink_target: str = ""


@dataclass(slots=True)
class ChangeSet:
    worker_id: str
    changes: dict[str, FileChange] = field(default_factory=dict)  # relpath -> change

    @property
    def paths(self) -> set[str]:
        return set(self.changes)


# --- git helpers -------------------------------------------------------------

def _git(repo: str, *args: str, env: dict[str, str] | None = None,
         check: bool = True) -> subprocess.CompletedProcess[str]:
    full_env = {**os.environ, **(env or {})}
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True,
                       text=True, env=full_env, timeout=60)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r


def is_git_repo(path: str) -> bool:
    try:
        r = _git(path, "rev-parse", "--is-inside-work-tree", check=False)
        return r.returncode == 0 and r.stdout.strip() == "true"
    except Exception:
        return False


def _has_head(repo: str) -> bool:
    return _git(repo, "rev-parse", "--verify", "-q", "HEAD", check=False).returncode == 0


def _is_special_repo(repo: str) -> bool:
    """True for repos worktree can't cleanly serve: bare, submodules, LFS."""
    try:
        if _git(repo, "rev-parse", "--is-bare-repository", check=False).stdout.strip() == "true":
            return True
        if os.path.exists(os.path.join(repo, ".gitmodules")):
            return True
        # git-LFS: attributes declare filter=lfs → worktree gets pointer files.
        ga = os.path.join(repo, ".gitattributes")
        if os.path.exists(ga):
            with open(ga, encoding="utf-8", errors="ignore") as fh:
                if "filter=lfs" in fh.read():
                    return True
    except Exception:
        return True
    return False


def _choose_mode(repo: str, requested: str) -> str:
    if requested in ("worktree", "copy", "none"):
        if requested == "worktree" and not (is_git_repo(repo) and _has_head(repo)
                                            and not _is_special_repo(repo)):
            return "copy"
        return requested
    # auto
    if is_git_repo(repo) and _has_head(repo) and not _is_special_repo(repo):
        return "worktree"
    return "copy"


def _snapshot_working_tree(repo: str) -> str:
    """Capture the current working tree (tracked+untracked, no gitignored) as a
    dangling commit; return its SHA. Uses a temp index so the main index/HEAD
    are untouched."""
    tmp_index = tempfile.mktemp(prefix="rune-iso-index-")
    env = {
        "GIT_INDEX_FILE": tmp_index,
        "GIT_AUTHOR_NAME": "rune", "GIT_AUTHOR_EMAIL": "rune@localhost",
        "GIT_COMMITTER_NAME": "rune", "GIT_COMMITTER_EMAIL": "rune@localhost",
    }
    try:
        # Seed temp index from HEAD, then stage all current changes (respects
        # .gitignore, so deps stay out).
        _git(repo, "read-tree", "HEAD", env=env)
        _git(repo, "add", "-A", env=env)
        tree = _git(repo, "write-tree", env=env).stdout.strip()
        head = _git(repo, "rev-parse", "HEAD").stdout.strip()
        commit = _git(repo, "commit-tree", tree, "-p", head,
                      "-m", "rune-isolation-base", env=env).stdout.strip()
        return commit
    finally:
        if os.path.exists(tmp_index):
            os.remove(tmp_index)


# --- gitignored dependency sharing (dynamic, no hardcoded ecosystem list) -----

def _gitignored_dirs(repo: str) -> list[str]:
    """Top-level gitignored directories present in the working tree."""
    try:
        r = _git(repo, "ls-files", "--others", "--ignored",
                 "--exclude-standard", "--directory", check=False)
    except Exception:
        return []
    out = []
    for line in r.stdout.splitlines():
        d = line.strip().rstrip("/")
        if d and os.sep not in d and os.path.isdir(os.path.join(repo, d)):
            out.append(d)
    return out


def _link_deps(repo: str, dest: str, exclude: list[str] | None = None) -> None:
    """Symlink top-level gitignored dirs into *dest* so builds/tests work."""
    exclude = set(exclude or [])
    for d in _gitignored_dirs(repo):
        if d in exclude:
            continue
        src = os.path.join(repo, d)
        link = os.path.join(dest, d)
        if os.path.exists(link):
            continue
        try:
            os.symlink(src, link)  # Windows: falls back handled by caller policy
        except OSError as exc:
            log.debug("dep_symlink_failed", dir=d, error=str(exc)[:80])


# --- manifest (copy mode base) -----------------------------------------------

def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest(root: str) -> dict[str, str]:
    """path(rel)->sha for all regular files under *root* (skips .git)."""
    man: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        if ".git" in dirnames:
            dirnames.remove(".git")
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            if os.path.islink(fp):
                man[os.path.relpath(fp, root)] = "symlink:" + os.readlink(fp)
                continue
            try:
                man[os.path.relpath(fp, root)] = _sha(fp)
            except OSError:
                pass
    return man


# --- public API --------------------------------------------------------------

async def create(repo: str, worker_id: str, *,
                 isolation: str = "auto",
                 share_deps: bool = True,
                 dep_exclude: list[str] | None = None) -> IsolatedWorkspace:
    """Materialise an isolated workspace for *worker_id*. Serialised git ops."""
    mode = _choose_mode(repo, isolation)
    if mode == "none":
        return IsolatedWorkspace(path=repo, mode="none", repo=repo)

    if mode == "worktree":
        async with _git_lock:
            base = _snapshot_working_tree(repo)
            dest = tempfile.mkdtemp(prefix=_WORKTREE_PREFIX)
            shutil.rmtree(dest, ignore_errors=True)
            _git(repo, "worktree", "add", "--detach", "-q", dest, base)
        if share_deps:
            _link_deps(repo, dest, dep_exclude)
        log.info("isolation_created", mode="worktree", worker=worker_id, path=dest)
        return IsolatedWorkspace(path=dest, mode="worktree", repo=repo, base_ref=base)

    # copy mode
    dest = tempfile.mkdtemp(prefix=_WORKTREE_PREFIX)
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(repo, dest, symlinks=True,
                    ignore=shutil.ignore_patterns(".git"))
    manifest = _manifest(dest)
    if share_deps and is_git_repo(repo):
        _link_deps(repo, dest, dep_exclude)
    log.info("isolation_created", mode="copy", worker=worker_id, path=dest)
    return IsolatedWorkspace(path=dest, mode="copy", repo=repo,
                             base_manifest=manifest)


def collect(ws: IsolatedWorkspace, worker_id: str) -> ChangeSet:
    """Collect the worker's file changes vs the base (added/modified/deleted)."""
    cs = ChangeSet(worker_id=worker_id)
    if ws.mode == "none":
        return cs
    if ws.mode == "worktree":
        _collect_git(ws, cs)
    else:
        _collect_copy(ws, cs)
    return cs


def _read_change(abspath: str) -> FileChange:
    if os.path.islink(abspath):
        return FileChange(op="modified", is_symlink=True,
                          symlink_target=os.readlink(abspath))
    with open(abspath, "rb") as fh:
        content = fh.read()
    return FileChange(op="modified", content=content,
                      mode=os.stat(abspath).st_mode & 0o777)


def _collect_git(ws: IsolatedWorkspace, cs: ChangeSet) -> None:
    r = _git(ws.path, "diff", "--name-status", "-z", ws.base_ref, check=False)
    # -z: records separated by NUL; status<TAB>?path entries.
    tokens = r.stdout.split("\0")
    i = 0
    while i < len(tokens):
        status = tokens[i].strip()
        if not status:
            i += 1
            continue
        code = status[0]
        if code == "R":  # rename: status, old, new
            old, new = tokens[i + 1], tokens[i + 2]
            cs.changes[old] = FileChange(op="deleted")
            cs.changes[new] = _read_change(os.path.join(ws.path, new))
            i += 3
            continue
        path = tokens[i + 1]
        i += 2
        if code == "D":
            cs.changes[path] = FileChange(op="deleted")
        else:  # A or M
            ap = os.path.join(ws.path, path)
            ch = _read_change(ap)
            ch.op = "added" if code == "A" else "modified"
            cs.changes[path] = ch

    # `git diff` only reports tracked changes; a worker that creates a NEW file
    # leaves it untracked. Capture those (respecting .gitignore so deps stay out).
    ur = _git(ws.path, "ls-files", "--others", "--exclude-standard", "-z",
              check=False)
    for path in ur.stdout.split("\0"):
        path = path.strip()
        if not path or path in cs.changes:
            continue
        ch = _read_change(os.path.join(ws.path, path))
        ch.op = "added"
        cs.changes[path] = ch


def _collect_copy(ws: IsolatedWorkspace, cs: ChangeSet) -> None:
    cur = _manifest(ws.path)
    base = ws.base_manifest
    for rel, h in cur.items():
        if rel not in base:
            ch = _read_change(os.path.join(ws.path, rel))
            ch.op = "added"
            cs.changes[rel] = ch
        elif base[rel] != h:
            ch = _read_change(os.path.join(ws.path, rel))
            ch.op = "modified"
            cs.changes[rel] = ch
    for rel in base:
        if rel not in cur:
            cs.changes[rel] = FileChange(op="deleted")


async def cleanup(ws: IsolatedWorkspace) -> None:
    """Remove the isolated workspace. Serialised for worktree mode."""
    if ws.mode == "none":
        return
    if ws.mode == "worktree":
        async with _git_lock:
            _git(ws.repo, "worktree", "remove", "--force", ws.path, check=False)
        # In case git left it (e.g. dep symlinks), ensure removal.
        shutil.rmtree(ws.path, ignore_errors=True)
    else:
        shutil.rmtree(ws.path, ignore_errors=True)
    log.debug("isolation_cleaned", path=ws.path)


def reap_orphans(repo: str) -> int:
    """Prune orphaned worktrees + leftover temp dirs. Returns count removed."""
    removed = 0
    try:
        if is_git_repo(repo):
            _git(repo, "worktree", "prune", check=False)
    except Exception:
        pass
    tmp = tempfile.gettempdir()
    try:
        for name in os.listdir(tmp):
            if name.startswith(_WORKTREE_PREFIX):
                shutil.rmtree(os.path.join(tmp, name), ignore_errors=True)
                removed += 1
    except OSError:
        pass
    return removed
