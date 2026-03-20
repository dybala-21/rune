"""LLM failover and profile management for RUNE.

Ported from src/agent/failover.ts (368 lines) - multi-provider failover
with retry, profile switching, thinking reduction, and context compaction.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Literal

from rune.utils.logger import get_logger

log = get_logger(__name__)

# Types

ThinkingLevel = Literal["none", "basic", "extended"]

FailoverReason = Literal[
    "auth",
    "rate_limit",
    "billing",
    "timeout",
    "context_overflow",
    "format",
    "unknown",
]


@dataclass(slots=True)
class LLMProfile:
    """Configuration for a single LLM provider/model."""
    name: str
    provider: str
    model: str
    api_key: str = ""
    endpoint: str = ""
    max_tokens: int = 16_384
    temperature: float = 0.0
    thinking_level: ThinkingLevel = "none"
    priority: int = 0


# Default profiles

_FALLBACK_PROFILES: list[LLMProfile] = [
    LLMProfile(
        name="openai:gpt-5.4",
        provider="openai",
        model="gpt-5.4",
        max_tokens=32_768,
        temperature=0.0,
        thinking_level="basic",
        priority=10,
    ),
    LLMProfile(
        name="anthropic:claude-sonnet-4.5",
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        max_tokens=16_384,
        temperature=0.0,
        thinking_level="extended",
        priority=11,
    ),
    LLMProfile(
        name="openai:gpt-5-mini",
        provider="openai",
        model="gpt-5-mini",
        max_tokens=16_384,
        temperature=0.0,
        thinking_level="none",
        priority=12,
    ),
]


def build_profiles_from_config() -> list[LLMProfile]:
    """Build failover profile list from user config, with hardcoded fallbacks.

    The user's configured provider/model becomes priority 0 (primary).
    Remaining fallback profiles are appended with lower priority, skipping
    any that duplicate the primary.
    """
    try:
        from rune.config import get_config
        cfg = get_config()
        llm = cfg.llm

        # active_provider/active_model (set by /model command) take priority
        active_provider = getattr(llm, "active_provider", None)
        active_model = (getattr(llm, "active_model", None) or "").strip()

        if active_provider and active_model:
            provider = active_provider
            model_name = active_model
        else:
            provider = llm.default_provider or "openai"
            models = getattr(llm.models, provider, None)
            model_name = models.best if models else "gpt-5.2"

        primary = LLMProfile(
            name=f"{provider}:{model_name}",
            provider=provider,
            model=model_name,
            max_tokens=32_768,
            temperature=0.0,
            thinking_level="basic" if provider == "openai" else "extended",
            priority=0,
        )

        # Append fallbacks that don't duplicate the primary
        profiles = [primary]
        for fb in _FALLBACK_PROFILES:
            if fb.provider != primary.provider or fb.model != primary.model:
                profiles.append(fb)

        return profiles

    except Exception as exc:
        log.debug("failover_config_load_failed", error=str(exc)[:100])
        return list(_FALLBACK_PROFILES)


# Strategy

@dataclass(slots=True)
class FailoverStrategy:
    """Describes what action to take on failure."""
    action: Literal["retry", "switch_profile", "reduce_thinking", "compact", "abort"]
    delay: float = 0.0  # seconds
    new_profile: LLMProfile | None = None
    compact_messages: bool = False


@dataclass(slots=True)
class FailoverResult:
    """Result after failover handling is complete."""
    success: bool
    reason: FailoverReason | None = None
    retries_used: int = 0
    final_profile: str = ""


# Error classification

def classify_error(error: Exception | str) -> FailoverReason:
    """Classify an LLM error into a failover reason."""
    msg = str(error).lower()

    if any(k in msg for k in ("401", "unauthorized", "invalid api key", "authentication")):
        return "auth"

    if any(k in msg for k in ("429", "rate limit", "too many requests", "quota")):
        return "rate_limit"

    if any(k in msg for k in ("402", "billing", "insufficient", "payment")):
        return "billing"

    if any(k in msg for k in ("timeout", "timed out", "deadline", "504", "408")):
        return "timeout"

    if any(k in msg for k in (
        "context", "token limit", "max_tokens", "too long",
        "context_length_exceeded", "maximum context",
    )):
        return "context_overflow"

    if any(k in msg for k in ("format", "parse", "json", "invalid response", "schema")):
        return "format"

    return "unknown"


# Strategy determination

def determine_strategy(
    reason: FailoverReason,
    current_profile: LLMProfile,
    retries_left: int,
    profiles: list[LLMProfile],
) -> FailoverStrategy:
    """Determine the failover strategy based on reason and current state."""
    match reason:
        case "auth" | "billing":
            # Non-recoverable with retry - switch profile immediately
            return _find_next_profile(current_profile, profiles)

        case "rate_limit":
            if retries_left > 0:
                # Exponential backoff retry
                delay = min(2.0 ** (3 - retries_left), 30.0)
                return FailoverStrategy(action="retry", delay=delay)
            return _find_next_profile(current_profile, profiles)

        case "timeout":
            if retries_left > 1:
                return FailoverStrategy(action="retry", delay=1.0)
            if retries_left > 0:
                # Try reducing thinking level before giving up
                if current_profile.thinking_level != "none":
                    return FailoverStrategy(action="reduce_thinking")
                return _find_next_profile(current_profile, profiles)
            return _find_next_profile(current_profile, profiles)

        case "context_overflow":
            # Try compaction first, then reduce thinking, then switch
            if current_profile.thinking_level == "extended":
                return FailoverStrategy(action="reduce_thinking")
            return FailoverStrategy(action="compact", compact_messages=True)

        case "format":
            if retries_left > 0:
                return FailoverStrategy(action="retry", delay=0.5)
            return _find_next_profile(current_profile, profiles)

        case "unknown":
            if retries_left > 0:
                return FailoverStrategy(action="retry", delay=1.0)
            return _find_next_profile(current_profile, profiles)

        case _:
            return FailoverStrategy(action="abort")


def _find_next_profile(
    current: LLMProfile,
    profiles: list[LLMProfile],
) -> FailoverStrategy:
    """Find the next profile to switch to, sorted by priority."""
    sorted_profiles = sorted(profiles, key=lambda p: p.priority)

    # Find profiles after the current one in priority order
    found_current = False
    for profile in sorted_profiles:
        if profile.name == current.name:
            found_current = True
            continue
        if found_current:
            return FailoverStrategy(
                action="switch_profile",
                new_profile=profile,
            )

    # If current is the last or not found, try the first different one
    for profile in sorted_profiles:
        if profile.name != current.name:
            return FailoverStrategy(
                action="switch_profile",
                new_profile=profile,
            )

    # No alternatives available
    return FailoverStrategy(action="abort")


# FailoverManager

class FailoverManager:
    """Manages LLM failover with retry, profile switching, and degradation."""

    def __init__(
        self,
        profiles: list[LLMProfile] | None = None,
        max_retries: int = 3,
    ) -> None:
        self.profiles = profiles or build_profiles_from_config()
        self.max_retries = max_retries
        self._retries_left = max_retries
        self._current_idx = 0
        self.error_signature_counts: dict[str, int] = {}

    @property
    def current_profile(self) -> LLMProfile:
        """Get the currently active LLM profile."""
        if not self.profiles:
            return LLMProfile(name="none", provider="none", model="none")
        idx = min(self._current_idx, len(self.profiles) - 1)
        return self.profiles[idx]

    @property
    def retries_left(self) -> int:
        return self._retries_left

    def get_current_profile(self) -> LLMProfile:
        """Get the currently active LLM profile."""
        return self.current_profile

    _CIRCUIT_BREAKER_THRESHOLD = 3

    async def handle_error(self, error: Exception | str) -> FailoverResult:
        """Handle an LLM error and execute the failover strategy.

        Returns a :class:`FailoverResult` indicating whether recovery succeeded.
        """
        reason = classify_error(error)

        # Circuit breaker: if the same error signature repeats N times, abort
        # immediately to prevent infinite error loops.
        # Exclude transient errors (timeout, rate_limit) - those benefit from retry.
        signature = f"{reason}:{str(error)[:100]}"
        if reason not in ("timeout", "rate_limit"):
            self.error_signature_counts[signature] = (
                self.error_signature_counts.get(signature, 0) + 1
            )
        if self.error_signature_counts.get(signature, 0) >= self._CIRCUIT_BREAKER_THRESHOLD:
            log.error(
                "circuit_breaker_tripped",
                signature=signature[:120],
                count=self.error_signature_counts[signature],
            )
            return FailoverResult(
                success=False,
                reason=f"circuit_breaker: {reason}",
                final_profile=self.current_profile.name,
            )

        log.warning(
            "llm_failover",
            reason=reason,
            profile=self.current_profile.name,
            retries_left=self._retries_left,
            error=str(error)[:200],
        )

        strategy = determine_strategy(
            reason=reason,
            current_profile=self.current_profile,
            retries_left=self._retries_left,
            profiles=self.profiles,
        )

        match strategy.action:
            case "retry":
                self._retries_left -= 1
                if strategy.delay > 0:
                    await asyncio.sleep(strategy.delay)
                return FailoverResult(
                    success=True,
                    reason=reason,
                    retries_used=self.max_retries - self._retries_left,
                    final_profile=self.current_profile.name,
                )

            case "switch_profile":
                if strategy.new_profile is not None:
                    old_name = self.current_profile.name
                    # Find the index of the new profile
                    for i, p in enumerate(self.profiles):
                        if p.name == strategy.new_profile.name:
                            self._current_idx = i
                            break
                    self._retries_left = self.max_retries  # Reset retries
                    log.info(
                        "llm_profile_switched",
                        from_profile=old_name,
                        to_profile=self.current_profile.name,
                    )
                    return FailoverResult(
                        success=True,
                        reason=reason,
                        retries_used=self.max_retries - self._retries_left,
                        final_profile=self.current_profile.name,
                    )
                return FailoverResult(
                    success=False,
                    reason=reason,
                    final_profile=self.current_profile.name,
                )

            case "reduce_thinking":
                new_profile = self._reduce_thinking_level(self.current_profile)
                # Update in place
                idx = min(self._current_idx, len(self.profiles) - 1)
                self.profiles[idx] = new_profile
                log.info(
                    "llm_thinking_reduced",
                    profile=new_profile.name,
                    new_level=new_profile.thinking_level,
                )
                return FailoverResult(
                    success=True,
                    reason=reason,
                    retries_used=self.max_retries - self._retries_left,
                    final_profile=new_profile.name,
                )

            case "compact":
                # Signal that compaction is needed - the caller handles it
                return FailoverResult(
                    success=True,
                    reason=reason,
                    retries_used=self.max_retries - self._retries_left,
                    final_profile=self.current_profile.name,
                )

            case "abort":
                return FailoverResult(
                    success=False,
                    reason=reason,
                    retries_used=self.max_retries - self._retries_left,
                    final_profile=self.current_profile.name,
                )

            case _:
                return FailoverResult(
                    success=False,
                    reason=reason,
                    final_profile=self.current_profile.name,
                )

    def _reduce_thinking_level(self, profile: LLMProfile) -> LLMProfile:
        """Create a copy of *profile* with a reduced thinking level."""
        match profile.thinking_level:
            case "extended":
                new_level: ThinkingLevel = "basic"
            case "basic":
                new_level = "none"
            case _:
                new_level = "none"

        return LLMProfile(
            name=profile.name,
            provider=profile.provider,
            model=profile.model,
            api_key=profile.api_key,
            endpoint=profile.endpoint,
            max_tokens=profile.max_tokens,
            temperature=profile.temperature,
            thinking_level=new_level,
            priority=profile.priority,
        )

    def reset(self) -> None:
        """Reset the failover manager to initial state."""
        self._retries_left = self.max_retries
        self._current_idx = 0


# Message compaction

async def compact_messages(
    messages: list[dict[str, Any]],
    summarizer: Callable[[str], Coroutine[Any, Any, str]],
    keep_last: int = 10,
) -> list[dict[str, Any]]:
    """Compact a message history by summarizing older messages.

    Keeps the last *keep_last* messages intact and summarizes everything
    before that into a single system message.

    Args:
        messages: The full message list.
        summarizer: An async function that takes text and returns a summary.
        keep_last: Number of recent messages to preserve verbatim.

    Returns:
        Compacted message list.
    """
    if len(messages) <= keep_last:
        return list(messages)

    # Split into old and recent
    old_messages = messages[:-keep_last]
    recent_messages = messages[-keep_last:]

    # Build text from old messages for summarization
    old_text_parts: list[str] = []
    for msg in old_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            old_text_parts.append(f"[{role}] {content[:500]}")

    if not old_text_parts:
        return list(messages)

    old_text = "\n".join(old_text_parts)

    try:
        summary = await summarizer(old_text)
    except Exception as exc:
        log.warning("message_compaction_failed", error=str(exc))
        # Fall back to simple truncation
        summary = f"(Summarized {len(old_messages)} earlier messages)"

    # Build compacted list
    compacted: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                f"## Conversation Summary (compacted from {len(old_messages)} messages)\n\n"
                f"{summary}"
            ),
        },
    ]
    compacted.extend(recent_messages)

    log.info(
        "messages_compacted",
        original_count=len(messages),
        compacted_count=len(compacted),
        summarized=len(old_messages),
    )

    return compacted
