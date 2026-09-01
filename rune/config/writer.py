"""Writes config changes back to config.yaml.

The loader resolves ``${VAR}`` placeholders and drops keys the schema does not
declare, so the in-memory :class:`RuneConfig` is a lossy view of the file. Round-
tripping it would write resolved API keys in plaintext and delete every key the
schema does not know about. So this patches the raw YAML instead: read the file
as-is, set only the paths asked for, write it back.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from rune.utils.logger import get_logger
from rune.utils.paths import rune_home

log = get_logger(__name__)

# Read-modify-write is not atomic, and patches arrive from request handlers that
# may overlap. Writes are small and rare, so one lock costs nothing. This
# serialises threads in one process only — two RUNE processes writing the same
# file can still lose an update, which needs file locking if that ever happens
# in practice.
_write_lock = threading.Lock()


def config_file_path() -> Path:
    """The config file to write to.

    Prefers whichever file the loader would read, so a project config stays the
    one place settings live. Falls back to the user-level path, which is where a
    machine with no config file yet gets one.
    """
    from rune.config.loader import _find_config_file

    existing = _find_config_file()
    if existing is not None:
        return existing
    return rune_home() / "config.yaml"


def _assign(root: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``root["a"]["b"] = value`` for a dotted path, creating parents.

    A ``None`` value removes the key, which is how a setting reverts to its
    default rather than being pinned to null.
    """
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        log.warning("config_save_bad_path", path=dotted)
        return

    node: Any = root
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            if value is None:
                return  # nothing to remove
            child = {}
            node[part] = child
        node = child

    leaf = parts[-1]
    if value is None:
        node.pop(leaf, None)
    else:
        node[leaf] = value


def save_config_values(updates: dict[str, Any]) -> Path | None:
    """Persist ``{"llm.activeModel": "gpt-5.4"}``-style updates to disk.

    Keys are dotted paths in the file's own camelCase spelling. Returns the file
    written, or ``None`` if the write failed — callers keep their in-memory
    change either way, so a failure costs persistence, not the setting.
    """
    if not updates:
        return None

    path = config_file_path()

    with _write_lock:
        try:
            from ruamel.yaml import YAML

            yaml = YAML()
            yaml.preserve_quotes = True

            data: dict[str, Any] = {}
            if path.is_file():
                data = yaml.load(path) or {}

            for dotted, value in updates.items():
                _assign(data, dotted, value)

            # Replace what the path points at, not the path itself: a config
            # symlinked into a dotfiles repo would otherwise be swapped for a
            # regular file and the link lost.
            target = path.resolve() if path.is_symlink() else path
            target.parent.mkdir(parents=True, exist_ok=True)

            # Keep the file's own permissions. mkstemp creates at 0600, and
            # carrying that over would silently tighten a 0644 config on
            # every save.
            mode = target.stat().st_mode & 0o777 if target.exists() else None

            # Write to a sibling temp file and rename, so an interrupted write
            # cannot leave a half-serialized config behind.
            fd, tmp_name = tempfile.mkstemp(
                dir=str(target.parent), prefix=".config-", suffix=".yaml.tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    yaml.dump(data, f)
                    f.flush()
                    os.fsync(f.fileno())
                if mode is not None:
                    os.chmod(tmp_name, mode)
                os.replace(tmp_name, target)
            except BaseException:
                Path(tmp_name).unlink(missing_ok=True)
                raise

        except Exception as exc:
            log.warning("config_save_failed", path=str(path), error=str(exc))
            return None

    # The cached mtime is deliberately left stale: the next load_config() then
    # re-reads and sees what was just written. Refreshing it here made the file
    # and the object disagree for readers that had not also been updated in
    # memory. get_config() still returns the cached object, so a session-only
    # override (a temporary /escalate switch) survives until something asks for
    # a reload.
    log.info("config_saved", path=str(path), keys=sorted(updates))
    return path
