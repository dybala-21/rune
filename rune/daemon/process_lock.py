"""PID file-based process locking for RUNE daemon.

Ported from src/daemon/process-lock.ts - ensures only one daemon
instance runs at a time using PID files and fcntl advisory locks.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
from pathlib import Path

from rune.utils.logger import get_logger
from rune.utils.paths import rune_home

log = get_logger(__name__)

_DEFAULT_LOCK_PATH = rune_home() / "daemon.pid"


def is_process_running(pid: int) -> bool:
    """Check whether a process with the given *pid* is alive.

    Uses ``kill(pid, 0)`` which does not send a real signal but checks
    for the existence of the process.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError as exc:
        # EPERM means the process exists but we lack permission to signal it
        return exc.errno == errno.EPERM


def _read_lock_pid(lock_path: Path) -> int | None:
    """Read the PID stored in *lock_path*, or ``None`` if unreadable."""
    try:
        content = lock_path.read_text(encoding="utf-8").strip()
        if not content:
            return None
        pid = int(content)
        return pid if pid > 0 else None
    except (OSError, ValueError):
        return None


def acquire_lock(
    lock_path: Path | str | None = None,
    *,
    pid: int | None = None,
) -> Path:
    """Acquire a PID-based process lock.

    Creates (or overwrites a stale) lock file at *lock_path* containing the
    current PID, then places an ``fcntl.flock`` advisory lock on it.

    Parameters
    ----------
    lock_path:
        Path to the lock file. Defaults to ``~/.rune/daemon.pid``.
    pid:
        PID to write. Defaults to the current process PID.

    Returns
    -------
    Path
        The resolved lock file path (useful when the default was used).

    Raises
    ------
    RuntimeError
        If another live process already holds the lock.
    OSError
        If the lock file cannot be created or locked.
    """
    path = Path(lock_path) if lock_path else _DEFAULT_LOCK_PATH
    owner_pid = pid if pid is not None else os.getpid()

    path.parent.mkdir(parents=True, exist_ok=True)

    # If the file already exists, check for a stale lock
    if path.exists():
        existing_pid = _read_lock_pid(path)
        if existing_pid is not None and existing_pid != owner_pid:
            if is_process_running(existing_pid):
                raise RuntimeError(
                    f"Process lock already held by pid {existing_pid} "
                    f"(lock file: {path})"
                )
            # Stale lock - remove it
            log.info(
                "removing_stale_lock",
                lock_path=str(path),
                stale_pid=existing_pid,
            )
            with contextlib.suppress(FileNotFoundError):
                path.unlink()

    # Write PID and acquire flock
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EAGAIN, errno.EACCES):
            raise RuntimeError(
                f"Process lock already held (lock file: {path})"
            ) from exc
        raise

    os.write(fd, f"{owner_pid}\n".encode())
    # Keep the fd open so the advisory lock is held for the lifetime of
    # the process.  Store it on the module so release_lock can close it.
    _held_locks[str(path)] = fd

    log.info("lock_acquired", lock_path=str(path), pid=owner_pid)
    return path


def release_lock(lock_path: Path | str | None = None) -> None:
    """Release the process lock and remove the lock file.

    Safe to call even if the lock was never acquired.
    """
    path = Path(lock_path) if lock_path else _DEFAULT_LOCK_PATH
    key = str(path)

    fd = _held_locks.pop(key, None)
    if fd is not None:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(fd)

    with contextlib.suppress(FileNotFoundError):
        path.unlink()

    log.debug("lock_released", lock_path=str(path))


def is_locked(lock_path: Path | str | None = None) -> bool:
    """Return ``True`` if the lock file exists and a live process holds it."""
    path = Path(lock_path) if lock_path else _DEFAULT_LOCK_PATH
    if not path.exists():
        return False
    pid = _read_lock_pid(path)
    if pid is None:
        return False
    return is_process_running(pid)


def get_lock_owner(lock_path: Path | str | None = None) -> int | None:
    """Return the PID of the process holding the lock, or ``None``.

    Returns ``None`` if the lock file does not exist, is unreadable, or
    contains a PID for a process that is no longer running.
    """
    path = Path(lock_path) if lock_path else _DEFAULT_LOCK_PATH
    pid = _read_lock_pid(path)
    if pid is None:
        return None
    return pid if is_process_running(pid) else None


# Internal mapping of lock-path -> held file descriptor.  Keeping the fd
# open ensures the fcntl advisory lock persists.
_held_locks: dict[str, int] = {}
