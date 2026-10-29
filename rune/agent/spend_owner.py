"""Use process-held locks to identify abandoned spending reservations."""

import atexit
import fcntl
import os
import threading
import time
from itertools import islice
from pathlib import Path
from uuid import UUID, uuid4

_owners: dict[Path, tuple[str, int]] = {}
_lock = threading.Lock()


def owner_id(database: Path) -> str:
    directory = database.resolve().with_suffix(".owners")
    with _lock:
        if directory not in _owners:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            name = uuid4().hex
            fd = os.open(directory / name, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                os.close(fd)
                raise
            _owners[directory] = (name, fd)
        return _owners[directory][0]


def exited(database: Path, owner: str) -> bool:
    if UUID(owner).hex != owner:
        raise ValueError("Invalid spending owner")
    path = database.resolve().with_suffix(".owners") / owner
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True
    finally:
        os.close(fd)


def prune(database: Path, retained: set[str]) -> None:
    directory = database.resolve().with_suffix(".owners")
    if not directory.exists():
        return
    for path in islice(directory.iterdir(), 2000):
        if path.name in retained:
            continue
        try:
            valid = UUID(path.name).hex == path.name
        except ValueError:
            continue
        try:
            if not valid or path.is_symlink() or path.stat().st_mtime > time.time() - 86400:
                continue
        except FileNotFoundError:
            continue
        if exited(database, path.name):
            path.unlink(missing_ok=True)


def _close() -> None:
    for _, fd in _owners.values():
        os.close(fd)
    _owners.clear()


def _after_fork() -> None:
    global _lock
    # Close inherited descriptors without releasing the parent's lock.
    _close()
    _lock = threading.Lock()


atexit.register(_close)
os.register_at_fork(after_in_child=_after_fork)
