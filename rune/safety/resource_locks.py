"""Cooperative file locks shared by Rune processes on this machine."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path


class ResourceBusy(RuntimeError):
    pass


@dataclass
class _Held:
    task: object
    root: Path
    write: bool
    active: bool = True


_held: ContextVar[tuple[_Held, ...]] = ContextVar("resource_locks", default=())


@contextmanager
def path_access(path: str, *, write: bool):
    from rune.utils.paths import rune_home

    target = Path(path).expanduser().resolve()
    task = asyncio.current_task()
    if any(item.active and item.task is task and (os.name == "nt" or
           target.is_relative_to(item.root) and (item.write or not write)) for item in _held.get()):
        yield
        return
    directory = rune_home() / "resource-locks"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "nt":
        # The Windows fallback serializes file access; it has no shared flock.
        from filelock import FileLock, Timeout

        lock = FileLock(directory / "workspace.lock", timeout=0)
        try:
            lock.acquire()
        except Timeout as exc:
            raise ResourceBusy("Another operation is using the workspace. Wait for it to finish.") from exc
        held = _Held(task, target, True)
        token = _held.set((*_held.get(), held))
        try:
            yield
        finally:
            held.active = False
            _held.reset(token)
            lock.release()
        return
    import fcntl

    descriptors = []
    held = _Held(task, target, write)
    token = None
    try:
        # Ancestor locks guard workspace/file conflicts without blocking unrelated files or readers.
        for node in (*reversed(target.parents), target):
            name = hashlib.sha256(os.fsencode(node)).hexdigest()
            fd = os.open(directory / name, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
            descriptors.append(fd)
            mode = fcntl.LOCK_EX if node == target and write else fcntl.LOCK_SH
            try:
                fcntl.flock(fd, mode | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise ResourceBusy(f"Another operation is using {target}. Wait for it to finish, then read the current state.") from exc
                raise
        token = _held.set((*_held.get(), held))
        yield
    finally:
        held.active = False
        if token is not None:
            _held.reset(token)
        for fd in reversed(descriptors):
            os.close(fd)


@contextmanager
def capability_access(name: str, params: dict):
    from contextlib import ExitStack

    reads = {"file_read", "file_list", "file_search", "document_read", "document_preview", "document_bundle_inspect",
             "code_analyze", "code_find_def", "code_find_refs", "code_impact", "project_map"}
    writes = {"file_write", "file_edit", "file_delete", "document_create", "document_bundle", "document_bundle_update"}
    paths = []
    if name in reads | writes:
        for field in ("path", "file_path", "directory", "source_path", "output_dir", "bundle_path"):
            if params.get(field):
                paths.append((str(params[field]), name in writes and field != "source_path"))
    elif name == "browser_screenshot" and params.get("path"):
        paths.append((str(params["path"]), True))
    with ExitStack() as stack:
        for path, write in sorted(paths):
            stack.enter_context(path_access(path, write=write))
        yield


@contextmanager
def command_access(command: str, cwd: str):
    from contextlib import ExitStack

    from rune.safety.shell_writes import shell_write_targets

    root = Path(cwd).expanduser().resolve()
    targets = shell_write_targets(command, str(root)).paths
    with ExitStack() as stack:
        stack.enter_context(path_access(str(root), write=True))
        for path in sorted(targets):
            if not Path(path).is_relative_to(root):
                stack.enter_context(path_access(path, write=True))
        yield
