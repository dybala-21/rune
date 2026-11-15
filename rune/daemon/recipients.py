"""Who proactive notifications go to on each channel, kept across restarts.

The gateway learns a channel's recipient from the first message that arrives on it. Kept only in
memory, a restart (a hosted machine's upgrade, a crash) forgot it, and scheduled results were
dropped until the owner happened to write again.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from rune.utils.logger import get_logger

log = get_logger(__name__)


def load(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.warning("channel_recipients_unreadable", path=str(path), error=type(exc).__name__)
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str) and v}


def save(path: Path, recipients: dict[str, str]) -> None:
    """Replace the file whole, so a crash leaves the old list or the new one."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(recipients, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    except OSError as exc:
        log.warning("channel_recipients_not_saved", path=str(path), error=type(exc).__name__)
