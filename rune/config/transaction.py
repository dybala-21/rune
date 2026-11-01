"""Stage a settings update before replacing files; restore originals on write failure."""

import os
import tempfile
from pathlib import Path

from rune.utils.logger import get_logger

log = get_logger(__name__)


class SettingsRecoveryError(OSError):
    """A filesystem failure also prevented restoring a prior file."""


def _stage(path: Path, content: bytes, mode: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".rune-settings-")
    staged = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        staged.chmod(mode)
        return staged
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def replace_settings(files: dict[Path, tuple[str, int]]) -> None:
    """Caller holds config and env write locks. No runtime values change here.

    Individual replacements use os.replace; a multi-file update is not crash-atomic.
    Backups remain at mode 0600 if an I/O failure also prevents rollback.
    https://docs.python.org/3/library/os.html#os.replace
    """
    staged: dict[Path, Path] = {}
    backups: dict[Path, tuple[Path | None, int]] = {}
    replaced: list[Path] = []
    retain: set[Path] = set()
    try:
        for path, (content, mode) in files.items():
            if path.exists():
                backups[path] = (_stage(path, path.read_bytes(), 0o600), path.stat().st_mode & 0o777)
            else:
                backups[path] = (None, mode)
            staged[path] = _stage(path, content.encode(), mode)
        for path, temporary in staged.items():
            os.replace(temporary, path)
            replaced.append(path)
    except BaseException as original:
        failed = False
        for path in reversed(replaced):
            backup, mode = backups[path]
            try:
                if backup is None:
                    path.unlink(missing_ok=True)
                else:
                    os.replace(backup, path)
                    path.chmod(mode)
            except OSError:
                failed = True
                if backup and backup.exists():
                    retain.add(backup)
                log.error("settings_rollback_failed", path=str(path))
        if failed:
            raise SettingsRecoveryError("Could not restore all settings files; protected backups were retained.") from original
        raise
    finally:
        for temporary in [*staged.values(), *(p for p, _ in backups.values() if p)]:
            if temporary not in retain:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as exc:
                    log.warning("settings_temporary_cleanup_failed", error=type(exc).__name__)
