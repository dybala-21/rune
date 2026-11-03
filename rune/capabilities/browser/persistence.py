"""Opt-in: keep the managed browser's logins across relaunches and machine rebuilds.

With ``RUNE_BROWSER_STATE=1`` the managed browser's cookies and local storage are saved to
``RUNE_HOME/browser/storage-state.json`` (owner-only) when the browser closes, and loaded
into the next browser. A hosted VM that persists only RUNE_HOME then keeps the user's
logins when its root disk is replaced. Off by default: a local user's own Chrome profile
already holds their sessions, and saving cookies is a decision for whoever deploys RUNE.
The attached native browser is never saved.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)


def state_path() -> Path | None:
    if os.environ.get("RUNE_BROWSER_STATE", "").strip() != "1":
        return None
    from rune.utils.paths import rune_home

    return rune_home() / "browser" / "storage-state.json"


def _set_aside(path: Path, reason: str) -> None:
    # Keep the unusable file for inspection instead of deleting the user's sessions.
    aside = path.with_name(path.name + ".unreadable")
    try:
        path.replace(aside)
    except OSError as move_error:
        log.warning("browser_state_set_aside_failed", error=str(move_error))
    log.warning("browser_state_unusable", path=str(path), moved_to=str(aside), reason=reason)


def context_options() -> dict[str, Any]:
    """``new_context`` options that restore the saved state, if there is a usable one."""
    path = state_path()
    if path is None or not path.is_file():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or not isinstance(state.get("cookies", []), list):
            raise ValueError("not a Playwright storage state")
    except (OSError, ValueError) as exc:
        _set_aside(path, type(exc).__name__)
        return {}
    return {"storage_state": state}


async def new_context(browser: Any, **options: Any) -> Any:
    """Open a context with the saved state; if Playwright rejects that state, start clean."""
    restored = context_options()
    if not restored:
        return await browser.new_context(**options)
    try:
        return await browser.new_context(**options, **restored)
    except Exception as exc:
        path = state_path()
        if path is not None and path.is_file():
            _set_aside(path, type(exc).__name__)
        else:
            log.warning("browser_state_restore_failed", error=type(exc).__name__)
        return await browser.new_context(**options)


async def save(context: Any) -> bool:
    """Write the context's cookies and storage atomically, owner-only. Returns True when saved."""
    path = state_path()
    if path is None or context is None:
        return False
    try:
        state = await context.storage_state()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # A unique temp file per save: two processes never interleave writes into one file.
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".partial")
        partial = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(state, stream)
            os.chmod(partial, 0o600)
            partial.replace(path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return True
    except Exception as exc:
        # A failed save keeps the previous file; the next login simply has to be repeated.
        log.warning("browser_state_save_failed", error=type(exc).__name__)
        return False
