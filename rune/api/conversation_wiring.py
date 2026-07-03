"""Multi-turn conversation wiring for the API/web server execution paths.

The web UI (``rune web`` → POST /api/message, WebSocket messages) and the
HTTP execute endpoints run agents through ``rune/api/server.py``. Before this
module existed those paths never touched the conversation layer: no turns were
recorded and ``loop.run`` received no ``message_history``, so every web message
was a stateless single-turn run (measured: a turn-2 recall question failed and
``conversations.db`` stayed empty).

This mirrors the proven daemon-gateway pattern (rune/daemon/gateway.py):
load-or-create conversation → record user turn → prepare context with the
conversation manager → run with history → record assistant turn → save.

Kept separate from ``server.py`` (already far over the repo's module-size cap)
so both the server and future callers share one copy instead of a third
hand-rolled variant.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

# user_id namespaces. "web:local" is the server-side sticky conversation for
# the browser UI live chat (the web client historically sent no session key);
# "web:session" marks conversations pinned by an explicit client sessionId.
WEB_STICKY_USER_ID = "web:local"
WEB_SESSION_USER_ID = "web:session"

_conv_manager: Any | None = None
_conv_manager_failed = False
_sticky_conv_id: str = ""
_resolve_lock: asyncio.Lock | None = None


def _reset_for_tests() -> None:
    """Reset module singletons (tests re-point RUNE_HOME between cases)."""
    global _conv_manager, _conv_manager_failed, _sticky_conv_id, _resolve_lock
    _conv_manager = None
    _conv_manager_failed = False
    _sticky_conv_id = ""
    _resolve_lock = None


def get_conv_manager() -> Any | None:
    """Lazily build the shared ConversationManager.

    Returns None (and remembers the failure) when the store is unavailable so
    agent runs degrade to stateless execution instead of erroring.
    """
    global _conv_manager, _conv_manager_failed
    if _conv_manager is not None or _conv_manager_failed:
        return _conv_manager
    try:
        from rune.conversation.manager import ConversationManager
        from rune.conversation.store import ConversationStore
        from rune.utils.paths import conversations_db_path

        _conv_manager = ConversationManager(ConversationStore(conversations_db_path()))
    except Exception as exc:
        _conv_manager_failed = True
        log.warning("api_conv_manager_init_failed", error=str(exc)[:200])
    return _conv_manager


def _is_fresh(updated_at: datetime | None, idle_minutes: int) -> bool:
    if updated_at is None:
        return False
    try:
        return datetime.now() - updated_at < timedelta(minutes=idle_minutes)
    except TypeError:
        return False


async def resolve_conversation(
    conv_manager: Any,
    session_id: str | None,
    *,
    sticky: bool,
) -> str | None:
    """Resolve the conversation id for one run; ensure it is in ``_active``.

    ``session_id`` pins an explicit conversation (client-managed key). With no
    session_id and ``sticky=True`` (web UI live chat), a server-side sticky
    conversation for ``web:local`` is reused until it has been idle for
    CONVERSATION_IDLE_MINUTES, then rotated. ``sticky=False`` (headless
    execute endpoints without a session) stays stateless — those callers must
    not pollute the human's live web chat.

    Note: web-UI sessionIds are conversation-store ids. SessionManager ids
    (the TUI session files listed by rpc sessions.*) are a different id space;
    an unknown id here simply starts an empty conversation under that id.
    """
    from rune.conversation.manager import CONVERSATION_IDLE_MINUTES
    from rune.conversation.types import Conversation

    if session_id:
        # Guard like the gateway: never clobber an in-memory conversation with
        # the DB copy — a concurrent run's just-added turns live only there.
        if session_id not in conv_manager._active:
            existing = await conv_manager._store.load(session_id)
            conv_manager._active[session_id] = existing or Conversation(
                id=session_id, user_id=WEB_SESSION_USER_ID,
            )
        return session_id

    if not sticky:
        return None

    global _sticky_conv_id, _resolve_lock
    if _resolve_lock is None:
        _resolve_lock = asyncio.Lock()
    async with _resolve_lock:
        # In-memory first: start_conversation does not persist, so a DB-only
        # lookup would miss a conversation whose first run is still in flight
        # and split consecutive fast messages across two conversations.
        if _sticky_conv_id:
            conv = conv_manager._active.get(_sticky_conv_id)
            if conv is not None and _is_fresh(
                conv.updated_at, CONVERSATION_IDLE_MINUTES
            ):
                return _sticky_conv_id

        # Cold start (fresh process): most recent active web conversation.
        try:
            row = await conv_manager._store.find_active_conversation(
                WEB_STICKY_USER_ID
            )
        except Exception as exc:
            log.debug("api_conv_find_active_failed", error=str(exc)[:100])
            row = None
        if row is not None and _is_fresh(row.updated_at, CONVERSATION_IDLE_MINUTES):
            if row.id not in conv_manager._active:
                # find_active_conversation returns metadata only — load turns.
                loaded = await conv_manager._store.load(row.id)
                conv_manager._active[row.id] = loaded or row
            _sticky_conv_id = row.id
            return _sticky_conv_id

        conv = conv_manager.start_conversation(WEB_STICKY_USER_ID)
        _sticky_conv_id = conv.id
        return _sticky_conv_id


def record_user_turn(conv_manager: Any, conversation_id: str, text: str) -> None:
    """Record the user turn (before prepare_agent_context, which drops the
    trailing user message from loaded history — the goal is passed to the
    loop separately)."""
    try:
        conv_manager.add_turn(conversation_id, "user", text)
    except Exception as exc:
        log.debug("api_conv_user_turn_failed", error=str(exc)[:100])


async def record_assistant_turn(
    conv_manager: Any,
    conversation_id: str,
    loop: Any,
    streamed_text: str,
) -> None:
    """Record the assistant turn and persist the conversation.

    Uses the loop's final answer when available; falls back to streamed text.
    Persisting here (not on the user turn) matches the gateway: an aborted run
    leaves no half-written conversation on disk.
    """
    try:
        from rune.agent.agent_context import resolve_assistant_answer

        answer = resolve_assistant_answer(
            getattr(loop, "_last_answer_text", ""), streamed_text,
        )
        if not answer:
            return
        conv_manager.add_turn(
            conversation_id, "assistant", answer,
            goal_type=getattr(loop, "_last_goal_type", ""),
        )
        conv = conv_manager._active.get(conversation_id)
        if conv is not None:
            await conv_manager._store.save(conv)
    except Exception as exc:
        log.debug("api_conv_assistant_turn_failed", error=str(exc)[:100])
