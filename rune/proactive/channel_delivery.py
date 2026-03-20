"""Smart channel selection and batch delivery for RUNE.

Selects the best delivery channel for a suggestion based on its
priority, batches eligible suggestions, and delivers them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rune.proactive.types import Suggestion
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Priority map per suggestion type
_TYPE_PRIORITY: dict[str, str] = {
    "warning": "high",
    "reminder": "normal",
    "followup": "low",
    "optimization": "low",
    "insight": "low",
}

# Suggestions of these types can be batched together
_BATCH_ELIGIBLE_TYPES = {"optimization", "insight", "followup"}


@dataclass(slots=True)
class DeliveryStrategy:
    """Describes how a suggestion should be delivered."""

    channel: str
    priority: str = "normal"
    batch_eligible: bool = False


class ChannelDeliveryManager:
    """Manages smart channel selection and batched delivery.

    Selects the most appropriate channel for each suggestion type,
    groups batch-eligible suggestions, and delivers them.
    """

    __slots__ = ("_channel_handlers", "_batch_buffer")

    def __init__(self) -> None:
        # channel_id → async handler callable
        self._channel_handlers: dict[str, Any] = {}
        self._batch_buffer: list[Suggestion] = []

    def register_channel(self, channel_id: str, handler: Any) -> None:
        """Register a delivery handler for a channel."""
        self._channel_handlers[channel_id] = handler

    def select_channel(
        self,
        suggestion: Suggestion,
        available_channels: list[str],
    ) -> str:
        """Select the best channel for delivering a suggestion.

        Parameters:
            suggestion: The suggestion to deliver.
            available_channels: List of available channel IDs.

        Returns:
            The selected channel ID.
        """
        if not available_channels:
            return "default"

        priority = _TYPE_PRIORITY.get(suggestion.type, "normal")

        # High-priority suggestions go to the first available channel
        # (assumed to be the primary / most visible channel)
        if priority == "high":
            return available_channels[0]

        # Normal priority: prefer the primary channel
        if priority == "normal":
            return available_channels[0]

        # Low priority: use the last channel (least intrusive) if multiple exist
        if len(available_channels) > 1:
            return available_channels[-1]

        return available_channels[0]

    def batch_suggestions(
        self,
        suggestions: list[Suggestion],
    ) -> list[list[Suggestion]]:
        """Group suggestions into batches.

        Batch-eligible suggestions (optimization, insight, followup) are
        grouped together. Non-batch-eligible suggestions each form their
        own single-item batch.

        Parameters:
            suggestions: List of suggestions to batch.

        Returns:
            A list of batches (each batch is a list of suggestions).
        """
        batches: list[list[Suggestion]] = []
        batch_group: list[Suggestion] = []

        for suggestion in suggestions:
            if suggestion.type in _BATCH_ELIGIBLE_TYPES:
                batch_group.append(suggestion)
            else:
                # Non-batchable: deliver individually
                batches.append([suggestion])

        if batch_group:
            batches.append(batch_group)

        return batches

    async def deliver(
        self,
        channel_id: str,
        suggestions: list[Suggestion],
    ) -> None:
        """Deliver a batch of suggestions to a specific channel.

        Parameters:
            channel_id: The target channel.
            suggestions: The suggestions to deliver.
        """
        handler = self._channel_handlers.get(channel_id)
        if handler is None:
            log.warning(
                "delivery_no_handler",
                channel=channel_id,
                count=len(suggestions),
            )
            return

        try:
            import asyncio

            result = handler(suggestions)
            if asyncio.iscoroutine(result):
                await result

            log.debug(
                "delivery_complete",
                channel=channel_id,
                count=len(suggestions),
            )
        except Exception as exc:
            log.error(
                "delivery_failed",
                channel=channel_id,
                error=str(exc),
            )
