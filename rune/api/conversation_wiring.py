"""Multi-turn conversation wiring shared by the API server, REPL and CLI.

Same flow as the daemon gateway (rune/daemon/gateway.py): load-or-create
conversation → record user turn → prepare context with the conversation
manager → run with history → record assistant turn → save.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

# "web:local" = server-side sticky conversation for the browser live chat;
# "web:session" = conversations pinned by an explicit client sessionId.
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
    user_id: str = WEB_SESSION_USER_ID,
) -> str | None:
    """Resolve the conversation id for one run and make sure it is in ``_active``.

    ``session_id`` pins an explicit conversation. Without one, ``sticky=True``
    (browser live chat) reuses the server-side ``web:local`` conversation until
    it has been idle for CONVERSATION_IDLE_MINUTES, then rotates. Headless
    callers without a session stay stateless (``sticky=False``) so scripts
    don't leak into the live chat.
    """
    from rune.conversation.manager import CONVERSATION_IDLE_MINUTES
    from rune.conversation.types import Conversation

    if session_id:
        # Don't replace an in-memory conversation with the DB copy — a
        # concurrent run's unsaved turns live on the in-memory object.
        if session_id not in conv_manager._active:
            existing = await conv_manager._store.load(session_id)
            conv_manager._active[session_id] = existing or Conversation(
                id=session_id, user_id=user_id,
            )
        return session_id

    if not sticky:
        return None

    global _sticky_conv_id, _resolve_lock
    if _resolve_lock is None:
        _resolve_lock = asyncio.Lock()
    async with _resolve_lock:
        # Check memory before the DB: start_conversation doesn't persist, so a
        # conversation whose first run is still in flight only exists here.
        if _sticky_conv_id:
            conv = conv_manager._active.get(_sticky_conv_id)
            if conv is not None and _is_fresh(
                conv.updated_at, CONVERSATION_IDLE_MINUTES
            ):
                return _sticky_conv_id

        # Cold start: most recent active web conversation from disk.
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


# Local chat surfaces whose conversations show in /sessions and the sidebar.
_LOCAL_SURFACE_USER_IDS = (
    WEB_SESSION_USER_ID,
    WEB_STICKY_USER_ID,
    "repl:local",
    "cli:session",
)


async def list_web_conversations(limit: int = 20) -> list[Any]:
    """All local-surface conversations (web, REPL, --session), newest first.

    Single source for the sessions sidebar, /sessions and /load so every
    surface shares one id space (the canonical conversation store — not the
    legacy memory store, whose ids the store here cannot resolve)."""
    manager = get_conv_manager()
    if manager is None:
        return []
    convs: list[Any] = []
    for user_id in _LOCAL_SURFACE_USER_IDS:
        try:
            convs.extend(await manager._store.list(user_id, limit=limit))
        except Exception as exc:
            log.debug("list_web_conversations_failed", error=str(exc)[:100])
    convs.sort(key=lambda c: c.updated_at, reverse=True)
    return convs[:limit]


async def get_workspace(conversation_id: str) -> str | None:
    """Workspace directory pinned to a conversation, if any."""
    import json

    manager = get_conv_manager()
    if manager is None or not conversation_id:
        return None
    conv = manager._active.get(conversation_id)
    if conv is None:
        conv = await manager._store.load(conversation_id)
    if conv is None or not conv.execution_context:
        return None
    try:
        path = json.loads(conv.execution_context).get("cwd", "")
    except (ValueError, AttributeError):
        return None
    return path if path and Path(path).is_dir() else None


async def set_workspace(conversation_id: str, path: str) -> str:
    """Pin a workspace directory to a conversation. Returns the resolved path.

    Raises ValueError when the path is not an existing directory.
    """
    import json

    if not conversation_id:
        raise ValueError("No conversation to pin a workspace to")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"Not a directory: {path}")

    manager = get_conv_manager()
    if manager is None:
        raise ValueError("Conversation store unavailable")
    conv_id = await resolve_conversation(manager, conversation_id, sticky=False)
    conv = manager._active[conv_id]
    try:
        ec = json.loads(conv.execution_context) if conv.execution_context else {}
    except ValueError:
        ec = {}
    ec["cwd"] = str(resolved)
    conv.execution_context = json.dumps(ec)
    await manager._store.save(conv)
    return str(resolved)


async def recent_workspaces(limit: int = 8) -> list[str]:
    """Distinct workspace dirs pinned on recent conversations, newest first."""
    import json

    seen: list[str] = []
    for conv in await list_web_conversations(limit=40):
        if not conv.execution_context:
            continue
        try:
            cwd = json.loads(conv.execution_context).get("cwd", "")
        except (ValueError, AttributeError):
            continue
        if cwd and cwd not in seen and Path(cwd).is_dir():
            seen.append(cwd)
        if len(seen) >= limit:
            break
    return seen


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
    reason: str = "",
) -> None:
    """Record the assistant turn and persist the conversation.

    Uses the loop's final answer when available; falls back to streamed text.
    An empty answer still records a short placeholder: a user turn with no
    assistant reply makes the next turn's model re-answer the previous
    question instead of treating it as done.
    Persisting here (not on the user turn) matches the gateway: an aborted run
    leaves no half-written conversation on disk.
    """
    try:
        from rune.agent.agent_context import resolve_assistant_answer

        answer = resolve_assistant_answer(
            getattr(loop, "_last_answer_text", ""), streamed_text,
        )
        if not answer:
            answer = f"(the run ended without a textual answer: {reason or 'unknown'})"
        conv_manager.add_turn(
            conversation_id, "assistant", answer,
            goal_type=getattr(loop, "_last_goal_type", ""),
        )
        conv = conv_manager._active.get(conversation_id)
        if conv is not None:
            await conv_manager._store.save(conv)
    except Exception as exc:
        log.debug("api_conv_assistant_turn_failed", error=str(exc)[:100])
