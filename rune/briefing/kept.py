"""Small JSON files the briefing keeps between runs, under RUNE_HOME/data/briefing."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)


def _path(name: str) -> Path:
    from rune.utils.paths import rune_data

    return rune_data() / "briefing" / f"{name}.json"


def read(name: str) -> dict[str, Any]:
    """The kept object, or an empty one when there is none or it can't be read."""
    try:
        data = json.loads(_path(name).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.warning("briefing_kept_unreadable", name=name, error=type(exc).__name__)
        return {}
    return data if isinstance(data, dict) else {}


def write(name: str, data: dict[str, Any]) -> None:
    """Replace the kept object whole, owner-only."""
    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A temp file of its own per write: two writers never share one half-written file.
    fd, partial = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".partial")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False)
        os.chmod(partial, 0o600)
        os.replace(partial, path)
    except BaseException:
        Path(partial).unlink(missing_ok=True)
        raise
