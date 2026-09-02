"""Channels handler - GET /channels, POST /channels/{id}/send.

Ported from src/api/handlers/channels.ts - query registered channel
adapters and send messages through them.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from rune.api.auth import TokenAuthDependency
from rune.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/channels", tags=["channels"])
auth = TokenAuthDependency()


# Models


class ChannelInfoResponse(BaseModel):
    name: str
    status: str
    type: str
    session_count: int = Field(0, alias="sessionCount")

    model_config = ConfigDict(populate_by_name=True)


class ChannelListResponse(BaseModel):
    channels: list[ChannelInfoResponse]


class ChannelSendRequest(BaseModel):
    message: str
    # Where on that channel to send: a chat id, room or user id. Channels are
    # multi-recipient, so there is no meaningful default.
    recipient: str = ""


class ChannelSendResponse(BaseModel):
    sent: bool
    channel_id: str = Field(alias="channelId")

    model_config = ConfigDict(populate_by_name=True)


# Routes


@router.get("", response_model=ChannelListResponse, dependencies=[Depends(auth)])
async def list_channels() -> ChannelListResponse:
    """List registered channel adapters and their statuses.

    Reads the live registry the daemon fills in this same process, so a
    configured Telegram or Slack connection shows up here.
    """
    from rune.channels.registry import get_channel_registry

    channels = [
        # The web UI itself, always present.
        ChannelInfoResponse(name="api", status="connected", type="api-client", sessionCount=0)
    ]

    try:
        registry = get_channel_registry()
        for name in registry.list():
            adapter = registry.get(name)
            if adapter is None:
                continue
            channels.append(
                ChannelInfoResponse(
                    name=name,
                    status="connected" if _is_running(adapter) else "disconnected",
                    type=type(adapter).__name__,
                    sessionCount=0,
                )
            )
    except Exception as exc:
        log.warning("channel_list_failed", error=str(exc))

    return ChannelListResponse(channels=channels)


def _is_running(adapter: object) -> bool:
    """Whether an adapter reports itself started, if it says at all."""
    for attr in ("is_running", "_running", "_started"):
        value = getattr(adapter, attr, None)
        if isinstance(value, bool):
            return value
    return True


@router.post("/{channel_id}/send", response_model=ChannelSendResponse, dependencies=[Depends(auth)])
async def send_to_channel(channel_id: str, req: ChannelSendRequest) -> ChannelSendResponse:
    """Send a message to a specific channel.

    The channel adapter must be registered and connected.
    """
    from rune.channels.registry import get_channel_registry

    adapter = get_channel_registry().get(channel_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"No such channel: {channel_id}")

    recipient = req.recipient or ""
    if not recipient:
        raise HTTPException(
            status_code=400, detail="recipient is required to send on a channel"
        )

    try:
        await adapter.send_notification(recipient, req.message)
    except Exception as exc:
        log.warning("channel_send_failed", channel_id=channel_id, error=str(exc))
        raise HTTPException(status_code=502, detail=f"Send failed: {exc}") from exc

    log.info("channel_send", channel_id=channel_id, message_length=len(req.message))
    return ChannelSendResponse(channelId=channel_id, sent=True)
