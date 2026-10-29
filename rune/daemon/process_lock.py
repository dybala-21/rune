"""Allow one daemon per Rune home using an advisory file lock."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
from pathlib import Path

from rune.utils.logger import get_logger
from rune.utils.paths import rune_home

log = get_logger(__name__)


def is_process_running(pid: int) -> bool:
    """Check process existence without sending a signal."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError as exc:
        # EPERM still means the process exists.
        return exc.errno == errno.EPERM


def acquire_lock(
    lock_path: Path | str | None = None,
    *,
    pid: int | None = None,
) -> Path:
    """Lock the file before replacing its recorded owner."""
    path = Path(lock_path) if lock_path else rune_home() / "daemon.pid"
    owner_pid = pid if pid is not None else os.getpid()

    path.parent.mkdir(parents=True, exist_ok=True)

    # Lock a stable inode before changing its contents.
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(fd, 0)
    except BaseException as exc:
        os.close(fd)
        if isinstance(exc, OSError) and exc.errno in (errno.EAGAIN, errno.EACCES):
            raise RuntimeError(f"Process lock already held (lock file: {path})") from exc
        raise

    try:
        os.write(fd, f"{owner_pid}\n".encode())
    except BaseException:
        os.close(fd)
        raise
    # The descriptor holds the lock until shutdown.
    _held_locks[str(path)] = fd

    log.info("lock_acquired", lock_path=str(path), pid=owner_pid)
    return path


def release_lock(lock_path: Path | str | None = None) -> None:
    """Release a held lock without unlinking the shared lock file."""
    path = Path(lock_path) if lock_path else rune_home() / "daemon.pid"
    key = str(path)

    fd = _held_locks.pop(key, None)
    if fd is not None:
        with contextlib.suppress(OSError):
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(fd)

    log.debug("lock_released", lock_path=str(path))


def is_locked(lock_path: Path | str | None = None) -> bool:
    """Check the advisory lock rather than trusting stale file contents."""
    return get_lock_owner(lock_path) is not None


def get_lock_owner(lock_path: Path | str | None = None) -> int | None:
    """Read the owner only while another descriptor holds the lock."""
    path = Path(lock_path) if lock_path else rune_home() / "daemon.pid"
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                pid = int(os.read(fd, 64).strip())
                return pid if is_process_running(pid) else None
            except ValueError:
                return None
        return None
    finally:
        os.close(fd)


# Keep descriptors open for the lifetime of each lock.
_held_locks: dict[str, int] = {}
