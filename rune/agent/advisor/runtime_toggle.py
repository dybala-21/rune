"""Runtime on/off toggle for the advisor layer.

Priority: RUNE_ADVISOR_ENABLED env > ~/.rune/data/advisor_enabled file > default (off).
Re-read on every consult() for mid-episode flips. Phase A native tool stays
attached if toggled off mid-episode — next episode is clean.
"""

from __future__ import annotations

import os

from rune.utils.logger import get_logger

log = get_logger(__name__)

_SETTING_FILE = "advisor_enabled"
_ENV_VAR = "RUNE_ADVISOR_ENABLED"
_DEFAULT = False

_TRUTHY = ("1", "true", "on", "yes")
_FALSY = ("0", "false", "off", "no")


def parse_env_bool(var_name: str, default: bool = False) -> bool:
    """Parse a boolean env var. Returns *default* when unset or unrecognised."""
    val = os.environ.get(var_name, "").strip().lower()
    if val in _TRUTHY:
        return True
    if val in _FALSY:
        return False
    return default


def _setting_path() -> str:
    from rune.utils.paths import rune_data
    return str(rune_data() / _SETTING_FILE)


def is_advisor_enabled() -> bool:
    """Return the current runtime toggle state.

    Never raises. Falls back to the default on any I/O failure so a
    corrupted home dir can't block the main loop.
    """
    env = os.environ.get(_ENV_VAR, "").strip().lower()
    if env in _TRUTHY:
        return True
    if env in _FALSY:
        return False

    # No env override — check file

    try:
        path = _setting_path()
        if os.path.exists(path):
            with open(path) as f:
                val = f.read().strip().lower()
            if val in _TRUTHY:
                return True
            if val in _FALSY:
                return False
    except Exception as exc:
        log.warning("advisor_toggle_read_failed", error=str(exc)[:100])

    return _DEFAULT


def set_advisor_enabled(enabled: bool) -> None:
    """Persist the advisor on/off setting to the toggle file."""
    try:
        path = _setting_path()
        with open(path, "w") as f:
            f.write("on" if enabled else "off")
    except Exception as exc:
        log.warning("advisor_toggle_write_failed", error=str(exc)[:100])
