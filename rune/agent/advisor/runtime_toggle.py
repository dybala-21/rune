"""Runtime on/off toggle for the advisor layer.

The advisor is opt-in at the env level (``RUNE_ADVISOR_MODEL``), but
users also need a way to flip it off mid-session — e.g. from the TUI
with ``/advisor off`` or from the web UI — without restarting the
daemon or unsetting env vars.

Resolution order (highest priority first):
1. ``RUNE_ADVISOR_ENABLED`` env var — hard override for scripts/CI
2. Persisted toggle file at ``~/.rune/data/advisor_enabled``
3. Default: ``False`` — advisor is fully off on a fresh install. The
   user must opt in once (``/advisor on`` in the TUI, or the web
   sidebar toggle) even when ``RUNE_ADVISOR_MODEL`` is set. This keeps
   the UI honest: a fresh install truthfully shows "Advisor: off".

Off semantics: ``AdvisorConfig.from_env`` checks this first and returns
a disabled config regardless of ``RUNE_ADVISOR_MODEL``, so no advisor
calls fire and no tokens are billed. ``AdvisorService.consult`` also
re-reads the toggle on every call so a mid-episode flip is honored
without waiting for the next ``for_episode`` rebuild.

Known limitation — Phase A native path:
    When ``RUNE_ADVISOR_NATIVE=1`` is also opted in and the executor is
    an Anthropic pair, the ``advisor_20260301`` tool is attached to the
    tool list once at episode start (see ``rune.agent.loop``). Flipping
    the toggle off mid-episode does NOT remove that already-attached
    tool, so the server may still run advisor sub-inferences for the
    remainder of the current episode. The next episode is clean.
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
