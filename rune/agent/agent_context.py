"""Agent context - shared pre/post-processing layer.

Ported from src/agent/agent-context.ts (179 lines) - unified context
preparation for TUI, CLI exec, and Gateway channels.

prepareAgentContext(): @-reference expansion, identity, conversation management,
    token budget, workspace CWD resolution.
postProcessAgentResult(): assistant turn save + episodic memory persistence.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)

# Constants

ASSISTANT_TURN_SAVE_MAX_ATTEMPTS = 3
ASSISTANT_TURN_SAVE_RETRY_DELAY_S = 0.08


# Types

@dataclass(slots=True)
class PrepareContextOptions:
    """Options for preparing agent context."""

    goal: str
    channel: str = "tui"  # 'tui' | 'cli' | custom
    cwd: str = ""
    path_intent_model: Any = None
    sender_id: str = ""
    sender_name: str = ""
    conversation_id: str = ""
    pinned_cwd: str | None = None
    attachments: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class AgentContext:
    """Runtime state for a single agent invocation.

    Holds all pre-processed inputs needed by the agent loop:
    - Sanitized goal text (with @-references expanded)
    - Resolved workspace CWD
    - Conversation history
    - Identity context
    - Token budget parameters
    """

    goal: str = ""
    original_goal: str = ""
    channel: str = "tui"
    workspace_root: str = ""
    conversation_id: str = ""
    sender_id: str = ""
    sender_name: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    identity_context: str = ""
    token_budget: int = 500_000
    pinned_cwd: str | None = None
    at_references: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PostProcessInput:
    """Input for post-processing agent results."""

    context: AgentContext
    success: bool
    answer: str = ""
    error: str | None = None
    changed_files: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


# Helpers

def _sanitize_goal_input(raw: str) -> str:
    """Strip invisible control chars that break short-intent matching."""
    import unicodedata

    # Remove C0/C1 control chars except newline/tab
    cleaned = "".join(
        ch for ch in raw
        if ch in ("\n", "\t", "\r")
        or not unicodedata.category(ch).startswith("C")
    )
    return cleaned.strip()


def _is_same_or_child(base_dir: str, target_path: str) -> bool:
    """Check if target is same as or under base_dir."""
    rel = os.path.relpath(target_path, base_dir)
    return rel == "" or (not rel.startswith("..") and not os.path.isabs(rel))


def _as_existing_workspace(candidate: str) -> str | None:
    """Resolve a candidate to an existing workspace directory."""
    resolved = os.path.abspath(candidate)
    try:
        if os.path.isdir(resolved):
            return resolved
        if os.path.isfile(resolved):
            return os.path.dirname(resolved)
    except OSError:
        return None
    return None


# Workspace CWD resolution

async def _resolve_workspace_cwd_for_turn(
    goal: str,
    requested_cwd: str,
    pinned_cwd: str | None = None,
    path_intent_model: Any = None,
) -> str:
    """Resolve the effective workspace CWD for this turn.

    Priority:
    1. @-path directive (deterministic, no LLM)
    2. LLM path intent oracle (if model available)
    3. Pinned CWD from session
    4. Requested CWD fallback
    """
    workspace_root = os.path.abspath(requested_cwd or os.getcwd())

    # @-path directive - highest priority, deterministic
    try:
        from rune.agent.path_intent_oracle import (
            ResolvePathIntentOptions,
            resolve_intentional_path_allowlist,
        )

        explicit_candidates_raw = await resolve_intentional_path_allowlist(
            goal, workspace_root,
            ResolvePathIntentOptions(model=path_intent_model),
        )
        explicit_candidates = [
            c for c in
            (_as_existing_workspace(c) for c in explicit_candidates_raw)
            if c is not None
        ]

        if len(explicit_candidates) == 1:
            return explicit_candidates[0]

        pinned = _as_existing_workspace(pinned_cwd) if pinned_cwd else None

        if len(explicit_candidates) > 1:
            if pinned:
                preferred = next(
                    (c for c in explicit_candidates
                     if _is_same_or_child(c, pinned) or _is_same_or_child(pinned, c)),
                    None,
                )
                return preferred or pinned
            return workspace_root

        if pinned:
            return pinned

    except Exception as exc:
        log.debug("workspace_cwd_resolution_failed", error=str(exc)[:200])

    return workspace_root


# @-reference expansion

def _expand_at_references(goal: str, cwd: str) -> tuple[str, list[str]]:
    """Expand ``@path`` references in goal text by resolving their content.

    Returns (expanded_goal, list_of_resolved_paths).
    For found files/directories, appends their content as a context block.
    References that don't exist on disk are left as-is for the agent to handle.
    """
    try:
        from rune.utils.at_reference import parse_and_resolve
    except ImportError:
        return goal, []

    resolved = parse_and_resolve(goal, cwd)
    if not resolved:
        return goal, []

    ref_paths: list[str] = []
    context_blocks: list[str] = []

    for ref in resolved:
        ref_paths.append(ref.full_path)
        if ref.error:
            # Path not found or unreadable - skip silently, let agent discover
            continue
        if not ref.content:
            continue

        label = ref.path
        if ref.type == "directory":
            context_blocks.append(
                f"<at-reference path=\"{label}\" type=\"directory\">\n"
                f"{ref.content}\n"
                f"</at-reference>"
            )
        elif ref.type == "file":
            truncated_note = " (truncated)" if ref.truncated else ""
            context_blocks.append(
                f"<at-reference path=\"{label}\" type=\"file\"{truncated_note}>\n"
                f"{ref.content}\n"
                f"</at-reference>"
            )

    if not context_blocks:
        return goal, ref_paths

    expanded = goal + "\n\n" + "\n\n".join(context_blocks)
    return expanded, ref_paths


# Public API

async def prepare_agent_context(
    options: PrepareContextOptions,
    conversation_manager: Any | None = None,
) -> AgentContext:
    """Prepare a full AgentContext for an agent invocation.

    Steps:
    1. Sanitize goal input
    2. Resolve workspace CWD
    3. Expand @-references (inject file/directory content)
    4. Load conversation history from ConversationManager
    5. Resolve identity context

    If *conversation_manager* is provided along with a valid
    ``conversation_id``, previous turns are loaded into the context
    so the agent loop has multi-turn awareness.
    """
    sanitized_goal = _sanitize_goal_input(options.goal)

    workspace_root = await _resolve_workspace_cwd_for_turn(
        sanitized_goal,
        options.cwd,
        options.pinned_cwd,
        options.path_intent_model,
    )

    # Expand @-references: resolve file/directory content and append to goal
    expanded_goal, at_refs = _expand_at_references(sanitized_goal, workspace_root)

    # Load conversation history if a manager and conversation_id are available.
    # We exclude the current (latest) user turn since the goal is passed
    # separately to the agent loop - including it would cause duplication.
    conversation_messages: list[dict[str, Any]] = []
    if conversation_manager is not None and options.conversation_id:
        try:
            budgeted = await conversation_manager.build_budgeted_context(
                sanitized_goal,
                options.conversation_id,
            )
            msgs = budgeted.conversation_messages or []
            # Drop the last message if it's the current user turn
            if msgs and msgs[-1].get("role") == "user":
                msgs = msgs[:-1]
            conversation_messages = msgs
        except Exception as exc:
            log.debug("conversation_history_load_failed", error=str(exc)[:200])

    # Pass attachments through metadata so the agent loop can include
    # them as vision/document content alongside the goal text.
    metadata: dict[str, Any] = {}
    if options.attachments:
        metadata["attachments"] = options.attachments

    return AgentContext(
        goal=expanded_goal,
        original_goal=options.goal,
        channel=options.channel,
        workspace_root=workspace_root,
        conversation_id=options.conversation_id,
        sender_id=options.sender_id,
        sender_name=options.sender_name,
        pinned_cwd=options.pinned_cwd,
        at_references=at_refs,
        messages=conversation_messages,
        metadata=metadata,
    )


async def post_process_agent_result(inp: PostProcessInput) -> None:
    """Post-process agent result: save assistant turn + episodic memory.

    Retries assistant turn save up to ASSISTANT_TURN_SAVE_MAX_ATTEMPTS times.
    """
    if not inp.answer:
        return

    # Save assistant turn with retry
    for attempt in range(1, ASSISTANT_TURN_SAVE_MAX_ATTEMPTS + 1):
        try:
            # Attempt to save via memory bridge
            from rune.agent.memory_bridge import save_agent_result_to_memory
            from rune.memory.manager import get_memory_manager

            manager = get_memory_manager()
            await save_agent_result_to_memory(
                goal=inp.context.goal or inp.context.original_goal,
                result={"output": inp.answer, "success": inp.success},
                memory_manager=manager,
                conversation_id=inp.context.conversation_id,
            )
            if attempt > 1:
                log.warning(
                    "assistant_turn_save_recovered",
                    conversation_id=inp.context.conversation_id,
                    attempt=attempt,
                )
            return
        except Exception as exc:
            log.warning(
                "assistant_turn_save_failed",
                conversation_id=inp.context.conversation_id,
                attempt=attempt,
                max_attempts=ASSISTANT_TURN_SAVE_MAX_ATTEMPTS,
                error=str(exc)[:200],
            )
            if attempt < ASSISTANT_TURN_SAVE_MAX_ATTEMPTS:
                await asyncio.sleep(ASSISTANT_TURN_SAVE_RETRY_DELAY_S * attempt)

    log.error(
        "assistant_turn_save_exhausted",
        conversation_id=inp.context.conversation_id,
    )
