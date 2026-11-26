"""What RUNE says first goes into the owner's conversation, not only out to the chat app.

A briefing or a scheduled result sent to Telegram used to leave no turn behind. The owner's reply
("what's the second mail?") then reached a conversation that had never seen it. Each message sent
on RUNE's own initiative is kept as an assistant turn of the conversation that chat belongs to,
and that conversation is listed in the app beside the web chats, so both show one thread.
"""

from __future__ import annotations

from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

# Marks the turns RUNE started, for the app and for the next run's history.
SOURCE = "proactive"


def thread_id(recipient_id: str) -> str:
    """The conversation a channel recipient's messages go to (the gateway's sender key).

    The recipient is a private chat (the relay links only the owner's own chat), where the chat
    id and the sender id are the same."""
    return f"{recipient_id}:{recipient_id}"


# Where a message goes when no chat app took it (an owner who only uses the app): a conversation
# of the app's own, listed with its chats.
INBOX = "rune:inbox"
INBOX_TITLE = "From RUNE"


def where(channel: str) -> str:
    """Where RUNE's scheduled messages for ``channel`` go, as the gateway would route one now. Said
    when a briefing or a watch is set, so the owner is never told of a chat app that won't get them."""
    from rune.daemon.gateway import get_gateway

    kept = f'kept in the app, in the conversation "{INBOX_TITLE}"'
    gateway = get_gateway()
    if gateway is None:
        # Set from outside the daemon: where a message goes is settled when it is sent.
        return f"Messages go to {channel} if it reaches the owner when one is sent; if not, they are {kept}."
    if gateway.reaches(channel):
        return f"Messages go to {channel}."
    return f"Messages are {kept}: {channel} does not reach the owner from here now. Once it does, they go there."


async def remember(manager: Any, recipient_id: str, text: str) -> None:
    """Keep ``text`` as an assistant turn of the recipient's conversation. Never raises."""
    await _keep(manager, thread_id(recipient_id), recipient_id, text)


async def remember_in_app(manager: Any, text: str) -> None:
    """Keep ``text`` in the app's RUNE conversation, for a message no chat app delivered."""
    from rune.api.conversation_wiring import WEB_SESSION_USER_ID

    await _keep(manager, INBOX, WEB_SESSION_USER_ID, text, title=INBOX_TITLE)


async def _keep(manager: Any, conv_id: str, user_id: str, text: str, title: str = "") -> None:
    if manager is None or not text.strip():
        return
    try:
        from rune.conversation.types import Conversation

        if conv_id not in manager._active:
            manager._active[conv_id] = (await manager._store.load(conv_id)
                                        or Conversation(id=conv_id, user_id=user_id, title=title))
        manager.add_turn(conv_id, "assistant", text, goal_type=SOURCE)
        await manager._store.save(manager._active[conv_id], embed=False)
    except Exception as exc:
        log.warning("owner_thread_not_kept", error=type(exc).__name__, exc_info=True)


def listed_owners() -> tuple[str, ...]:
    """The user ids whose channel conversations the app lists: the owner's notification chats."""
    from rune.daemon.gateway import get_gateway

    gateway = get_gateway()
    if gateway is None:
        return ()
    return tuple(sorted({r for r in gateway._default_recipients.values() if r}))
