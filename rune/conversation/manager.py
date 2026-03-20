"""Conversation manager for RUNE.

Manages the lifecycle of conversations: creation, turn addition,
context window fitting within token budgets, LLM-based conversation
compression, and digest generation.

Ported from src/conversation/manager.ts.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from rune.conversation.context_budget import (
    GoalClassification,
    ResolveContextBudgetOptions,
    resolve_context_budget,
)
from rune.conversation.store import ConversationStore
from rune.conversation.types import Conversation, ConversationTurn
from rune.utils.logger import get_logger
from rune.utils.tokenizer import count_tokens, truncate_to_last_tokens, truncate_to_tokens

log = get_logger(__name__)

# Constants

CONVERSATION_IDLE_MINUTES = 30

# S2: assistant message smart truncation (token-based, head + tail preservation)
MAX_ASSISTANT_TOKENS = 2000
TAIL_PRESERVE_TOKENS = 500

# Low-signal title pattern -- matched titles get refreshed on next message
_LOW_SIGNAL_TITLE_PATTERN = re.compile(
    r"^(안녕|하이|hello|hi|hey|ㅎㅇ|ㅇㅇ|응|네|yes|ok)$",
    re.IGNORECASE,
)

# LLM Summarizer Protocol


@runtime_checkable
class LLMSummarizer(Protocol):
    """Protocol for pluggable LLM-based summarization.

    Implementations should call an LLM and return the generated text.
    """

    async def summarize(self, prompt: str, *, max_tokens: int = 800) -> str:
        """Send *prompt* to an LLM and return the summary text."""
        ...


# Event / Snapshot Types


@dataclass(slots=True)
class ConversationCompactionEvent:
    """Emitted when a conversation is compacted."""

    conversation_id: str
    total_tokens: int
    budget_tokens: int
    turns_compacted: int
    turns_preserved: int
    summary_tokens: int


@dataclass(slots=True)
class ExecutionContextSnapshot:
    """Captured execution environment info."""

    cwd: str = ""
    git_branch: str | None = None
    git_dirty: bool | None = None


@dataclass(slots=True)
class BudgetedContext:
    """Result of build_budgeted_context."""

    conversation_messages: list[dict[str, str]] = field(default_factory=list)
    memory_context: str = ""


# Manager

CompactionCallback = Callable[[ConversationCompactionEvent], Any]


class ConversationManager:
    """High-level conversation management with token budget awareness.

    Supports:
    - Basic sliding-window context fitting (``get_context_window``)
    - Dynamic token-budgeted context building (``build_budgeted_context``)
    - LLM-based conversation compression (``compact_conversation``)
    - Stale conversation archiving (``archive_stale_conversations``)
    - Execution context capture (``capture_execution_context``)
    """

    __slots__ = (
        "_store",
        "_token_budget",
        "_active",
        "_max_active",
        "_llm",
        "_classification_cache",
        "_background_tasks",
    )

    def __init__(
        self,
        store: ConversationStore,
        token_budget: int = 100_000,
        llm: LLMSummarizer | None = None,
    ) -> None:
        self._store = store
        self._token_budget = token_budget
        self._active: dict[str, Conversation] = {}
        self._max_active: int = 50
        self._llm = llm
        self._background_tasks: set[asyncio.Task[None]] = set()
        # Goal text -> GoalClassification cache (populated externally)
        self._classification_cache: dict[str, GoalClassification] = {}

    # Classification hint cache

    def set_classification_hint(
        self, goal: str, classification: GoalClassification
    ) -> None:
        """Cache a classification result for *goal*."""
        self._classification_cache[goal] = classification

    def get_cached_classification(
        self, goal: str
    ) -> GoalClassification | None:
        """Return a cached classification for *goal*, or ``None``."""
        return self._classification_cache.get(goal)

    # Conversation lifecycle

    def start_conversation(self, user_id: str) -> Conversation:
        """Create and return a new conversation."""
        # Evict oldest conversation if at capacity (LRU by updated_at)
        if len(self._active) >= self._max_active:
            oldest_id = min(
                self._active,
                key=lambda cid: self._active[cid].updated_at,
            )
            evicted = self._active.pop(oldest_id)
            log.info(
                "conversation_evicted",
                id=oldest_id,
                reason="max_active_reached",
                turns=len(evicted.turns),
            )

        conv = Conversation(user_id=user_id)
        self._active[conv.id] = conv
        log.info("conversation_started", id=conv.id, user_id=user_id)
        return conv

    def add_turn(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> ConversationTurn:
        """Add a turn to an active conversation.

        Raises KeyError if the conversation is not active.
        """
        conv = self._active.get(conversation_id)
        if conv is None:
            raise KeyError(f"Conversation {conversation_id} is not active")

        # S2: assistant content smart truncation (token-based, head + tail)
        effective_content = content
        if role == "assistant":
            tokens = count_tokens(content)
            if tokens > MAX_ASSISTANT_TOKENS:
                head_tokens = MAX_ASSISTANT_TOKENS - TAIL_PRESERVE_TOKENS - 20
                head = truncate_to_tokens(content, head_tokens)
                tail = truncate_to_last_tokens(content, TAIL_PRESERVE_TOKENS)
                effective_content = (
                    head + "\n\n...(truncated: details omitted)...\n\n" + tail
                )

        turn = ConversationTurn(
            role=role,
            content=effective_content,
            tool_calls=tool_calls or [],
        )
        conv.turns.append(turn)
        conv.updated_at = datetime.now()

        # Auto-generate title from first user message
        if not conv.title and role == "user":
            conv.title = content[:80]

        log.debug(
            "turn_added",
            conversation_id=conversation_id,
            role=role,
            content_len=len(content),
        )
        return turn

    # Context window (backward compat)

    def get_context_window(
        self,
        conversation_id: str,
    ) -> list[ConversationTurn]:
        """Return turns that fit within the token budget.

        Keeps the system message (if any) and as many recent turns as possible.
        Uses a sliding window from the end.
        """
        conv = self._active.get(conversation_id)
        if conv is None:
            raise KeyError(f"Conversation {conversation_id} is not active")

        if not conv.turns:
            return []

        # Separate system turns (always included) from the rest
        system_turns: list[ConversationTurn] = []
        other_turns: list[ConversationTurn] = []

        for turn in conv.turns:
            if turn.role == "system":
                system_turns.append(turn)
            else:
                other_turns.append(turn)

        # Count tokens for system turns first
        budget_remaining = self._token_budget
        for turn in system_turns:
            budget_remaining -= self._turn_tokens(turn)

        if budget_remaining <= 0:
            return system_turns[-1:] if system_turns else []

        # Fill from the end with other turns
        selected: list[ConversationTurn] = []
        for turn in reversed(other_turns):
            cost = self._turn_tokens(turn)
            if cost > budget_remaining:
                break
            selected.append(turn)
            budget_remaining -= cost

        selected.reverse()
        return system_turns + selected

    # S1: build_budgeted_context

    async def build_budgeted_context(
        self,
        goal: str,
        conversation_id: str,
        *,
        conversation_budget_tokens: int | None = None,
        memory_budget_tokens: int | None = None,
        model_context_window_tokens: int | None = None,
        reserved_context_tokens: int | None = None,
        on_compaction: CompactionCallback | None = None,
    ) -> BudgetedContext:
        """Build token-budgeted context for the agent loop.

        Budget defaults by goal type (via ``resolve_context_budget``):
        - casual: 12K, general: 24K, web: 20K, complex: 36K

        An implicit continuation floor of 20K conversation tokens is
        applied when the conversation already has turns (multi-turn).

        Background compaction is triggered (non-blocking) when token
        usage exceeds 65% of the budget.
        """
        conv = self._active.get(conversation_id)

        # Resolve classification hint (cached, no LLM call)
        classification_hint = self.get_cached_classification(goal)

        opts = ResolveContextBudgetOptions(
            conversation_budget_tokens=conversation_budget_tokens,
            memory_budget_tokens=memory_budget_tokens,
            model_context_window_tokens=model_context_window_tokens,
            reserved_context_tokens=reserved_context_tokens,
            classification_hint=classification_hint,
        )
        budget = resolve_context_budget(goal, opts)
        conv_budget = budget.conversation_budget_tokens

        # Implicit continuation floor: if conversation already has turns,
        # guarantee a minimum budget so multi-turn context isn't starved.
        if conversation_budget_tokens is None and memory_budget_tokens is None:
            if conv and len(conv.turns) > 0:
                conv_budget = max(conv_budget, 20_000)

        # Background compaction (non-blocking)
        task = asyncio.create_task(
            self._safe_compact(conversation_id, conv_budget, on_compaction)
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        # Build conversation messages within budget
        messages = self._build_messages_within_budget(
            conversation_id, conv_budget
        )

        return BudgetedContext(
            conversation_messages=messages,
            memory_context="",  # Memory bridge integration is external
        )

    async def _safe_compact(
        self,
        conversation_id: str,
        budget_tokens: int,
        on_compaction: CompactionCallback | None,
    ) -> None:
        """Run compaction swallowing errors so it never crashes the caller."""
        try:
            await self.compact_conversation(
                conversation_id, budget_tokens, on_compaction
            )
        except Exception:
            log.debug("background_compaction_error", exc_info=True)

    def _build_messages_within_budget(
        self,
        conversation_id: str,
        budget_tokens: int,
    ) -> list[dict[str, str]]:
        """Select turns from the active conversation within *budget_tokens*.

        System turns are always included first, then recent turns are
        added from the end until the budget is exhausted.
        """
        conv = self._active.get(conversation_id)
        if conv is None or not conv.turns:
            return []

        system_msgs: list[dict[str, str]] = []
        other_turns: list[ConversationTurn] = []

        for turn in conv.turns:
            if turn.role == "system":
                system_msgs.append({"role": turn.role, "content": turn.content})
            else:
                other_turns.append(turn)

        remaining = budget_tokens
        for msg in system_msgs:
            remaining -= count_tokens(msg["content"]) + 4

        if remaining <= 0:
            return system_msgs[-1:] if system_msgs else []

        selected: list[dict[str, str]] = []
        for turn in reversed(other_turns):
            cost = self._turn_tokens(turn)
            if cost > remaining:
                break
            selected.append({"role": turn.role, "content": turn.content})
            remaining -= cost

        selected.reverse()
        return system_msgs + selected

    # Conversation compaction

    async def compact_conversation(
        self,
        conversation_id: str,
        budget_tokens: int = 16_000,
        on_compaction: CompactionCallback | None = None,
    ) -> None:
        """Compact a conversation when tokens exceed 65% of *budget_tokens*.

        The latest 5 turns are preserved. The older portion is summarized
        via the LLM (with a 10s timeout) or an extractive fallback.
        """
        conv = self._active.get(conversation_id)
        if conv is None or not conv.turns:
            return

        total_tokens = sum(self._turn_tokens(t) for t in conv.turns)

        # 65% threshold -- no compaction needed below this
        if total_tokens < budget_tokens * 0.65:
            return

        # Preserve latest turns (up to 5, but at least half)
        preserve_count = min(5, max(1, len(conv.turns) // 2 + 1))
        to_compact = conv.turns[: len(conv.turns) - preserve_count]
        if len(to_compact) < 2:
            return

        log.info(
            "compacting_conversation",
            conversation_id=conversation_id,
            total_tokens=total_tokens,
            turns_to_compact=len(to_compact),
            turns_preserved=preserve_count,
        )

        summary = await self.summarize_turns(to_compact)

        # Replace the compacted turns with a single summary turn
        preserved = conv.turns[len(conv.turns) - preserve_count :]
        summary_turn = ConversationTurn(
            role="system",
            content=summary,
        )
        conv.turns = [summary_turn] + preserved
        conv.updated_at = datetime.now()

        event = ConversationCompactionEvent(
            conversation_id=conversation_id,
            total_tokens=total_tokens,
            budget_tokens=budget_tokens,
            turns_compacted=len(to_compact),
            turns_preserved=preserve_count,
            summary_tokens=count_tokens(summary),
        )

        log.info(
            "conversation_compacted",
            conversation_id=conversation_id,
            summary_tokens=event.summary_tokens,
        )

        if on_compaction is not None:
            on_compaction(event)

    # summarize_turns

    async def summarize_turns(
        self, turns: list[ConversationTurn]
    ) -> str:
        """Summarize *turns* into a condensed text block.

        Uses LLM-based summarization when an ``LLMSummarizer`` is
        available, with a 10-second timeout. Falls back to extractive
        compaction on failure or when no LLM is configured.
        """
        transcript = "\n---\n".join(
            f"[{t.role}] {t.content}" for t in turns
        )
        # Truncate input to ~4000 tokens (rough char estimate: 4 chars/token)
        truncated = truncate_to_tokens(transcript, 4000)

        if self._llm is None:
            return self._extractive_compact(turns)

        prompt = (
            "Summarize the following conversation concisely. "
            "Preserve: project paths, key decisions, error causes, "
            "file names, and current task state. "
            "Output in the same language as input. Max 500 words.\n\n"
            f"{truncated}"
        )

        try:
            result = await asyncio.wait_for(
                self._llm.summarize(prompt, max_tokens=800),
                timeout=10.0,
            )
            return f"[Conversation Summary]\n{result}"
        except Exception as exc:
            log.debug(
                "llm_summarization_failed_using_extractive_fallback",
                error=str(exc),
            )
            return self._extractive_compact(turns)

    # Extractive fallback

    @staticmethod
    def _extractive_compact(turns: list[ConversationTurn]) -> str:
        """Fallback summary: first + last sentence from each turn."""
        parts: list[str] = ["[Conversation Summary]"]
        for t in turns:
            sentences = [
                s
                for s in re.split(r"[.!?。]\s+", t.content)
                if len(s) > 10
            ]
            if sentences:
                first = sentences[0][:100]
                last = (
                    sentences[-1][:100] if len(sentences) > 1 else ""
                )
                entry = f"[{t.role}] {first}"
                if last:
                    entry += f" ... {last}"
                parts.append(entry)
        return "\n".join(parts)

    # Archive stale conversations

    async def archive_stale_conversations(
        self, idle_minutes: int = CONVERSATION_IDLE_MINUTES
    ) -> int:
        """Mark conversations idle for >*idle_minutes* as archived.

        Generates an extractive 1-liner session digest for each.
        Returns the number of conversations archived.
        """
        cutoff = datetime.now() - timedelta(minutes=idle_minutes)
        archived = 0

        stale_ids: list[str] = []
        for conv_id, conv in self._active.items():
            if conv.updated_at < cutoff:
                stale_ids.append(conv_id)

        for conv_id in stale_ids:
            conv = self._active.pop(conv_id)
            conv.digest = self._generate_session_digest(conv)
            conv.updated_at = datetime.now()
            try:
                await self._store.save(conv)
            except Exception:
                log.debug("archive_save_error", conversation_id=conv_id, exc_info=True)
            archived += 1
            log.info(
                "conversation_archived",
                conversation_id=conv_id,
                idle_minutes=idle_minutes,
            )

        return archived

    @staticmethod
    def _generate_session_digest(conv: Conversation) -> str:
        """Extractive 1-liner digest for an archived conversation."""
        if not conv.turns:
            return ""

        user_turns = [t for t in conv.turns if t.role == "user"]
        assistant_turns = [t for t in conv.turns if t.role == "assistant"]
        if not user_turns:
            return ""

        first_goal = user_turns[0].content[:80].replace("\n", " ")
        last_goal = (
            user_turns[-1].content[:60].replace("\n", " ")
            if len(user_turns) > 1
            else ""
        )
        last_result = (
            assistant_turns[-1].content[:60].replace("\n", " ")
            if assistant_turns
            else ""
        )

        parts = [first_goal]
        if last_goal and last_goal != first_goal:
            parts.append(f"-> {last_goal}")
        if last_result:
            parts.append(f"| {last_result}")

        return f"{' '.join(parts)} ({len(conv.turns)} turns)"

    # S6: capture execution context

    @staticmethod
    async def capture_execution_context() -> ExecutionContextSnapshot:
        """Capture current git branch and dirty state.

        Each subprocess has a 3-second timeout. Non-git directories
        are handled gracefully.
        """
        import os

        cwd = os.getcwd()
        git_branch: str | None = None
        git_dirty: bool | None = None

        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "rev-parse", "--abbrev-ref", "HEAD",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3)
            if proc.returncode == 0:
                git_branch = stdout.decode().strip()
        except (TimeoutError, FileNotFoundError, OSError):
            pass

        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "status", "--porcelain",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3)
            if proc.returncode == 0:
                git_dirty = len(stdout.decode().strip()) > 0
        except (TimeoutError, FileNotFoundError, OSError):
            pass

        return ExecutionContextSnapshot(
            cwd=cwd,
            git_branch=git_branch,
            git_dirty=git_dirty,
        )

    # End conversation

    async def end_conversation(self, conversation_id: str) -> None:
        """Generate a digest, save, and remove from active cache."""
        conv = self._active.get(conversation_id)
        if conv is None:
            log.warning("end_unknown_conversation", id=conversation_id)
            return

        conv.digest = self._store._generate_digest(conv)
        conv.updated_at = datetime.now()

        await self._store.save(conv)
        del self._active[conversation_id]

        log.info(
            "conversation_ended",
            id=conversation_id,
            turns=len(conv.turns),
        )

    # Helpers

    @staticmethod
    def _turn_tokens(turn: ConversationTurn) -> int:
        """Estimate token cost of a turn."""
        text = turn.content
        if turn.tool_calls:
            from rune.utils.fast_serde import json_encode

            text += json_encode(turn.tool_calls)
        # Add role overhead (~4 tokens)
        return count_tokens(text) + 4
