"""Channels handler - GET /channels, POST /channels/{id}/send.

Ported from src/api/handlers/channels.ts - query registered channel
adapters and send messages through them.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
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


class ChannelSendResponse(BaseModel):
    sent: bool
    channel_id: str = Field(alias="channelId")

    model_config = ConfigDict(populate_by_name=True)


# Routes


@router.get("", response_model=ChannelListResponse, dependencies=[Depends(auth)])
async def list_channels() -> ChannelListResponse:
    """List all registered channel adapters and their statuses."""
    # In a full implementation, this would query the Gateway for channel info.
    # For now, return the API channel as a default.
    channels = [
        ChannelInfoResponse(
            name="api",
            status="connected",
            type="api-client",
            sessionCount=0,
        )
    ]
    return ChannelListResponse(channels=channels)


@router.post("/{channel_id}/send", response_model=ChannelSendResponse, dependencies=[Depends(auth)])
async def send_to_channel(channel_id: str, req: ChannelSendRequest) -> ChannelSendResponse:
    """Send a message to a specific channel.

    The channel adapter must be registered and connected.
    """
    # Placeholder - real implementation routes through the Gateway.
    log.info("channel_send", channel_id=channel_id, message_length=len(req.message))
    return ChannelSendResponse(channelId=channel_id, sent=True)
