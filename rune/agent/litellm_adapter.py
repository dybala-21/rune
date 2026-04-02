"""LiteLLM streaming adapter, drop-in replacement for PydanticAI Agent.

Provides the same interface as ``pydantic_ai.Agent`` (run_stream,
stream_text, usage, all_messages, get_output) but calls LiteLLM directly.
This removes the PydanticAI dependency and enables all LiteLLM providers
(OpenAI, Anthropic, Gemini, Azure, Ollama, etc.) in a single code path.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import litellm

from rune.utils.logger import get_logger

log = get_logger(__name__)


# Usage tracking

@dataclass(slots=True)
class StreamUsage:
    """Token usage from a streaming completion."""
    input_tokens: int = 0
    output_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(slots=True)
class UsageLimits:
    """Drop-in replacement for pydantic_ai.usage.UsageLimits."""
    request_tokens_limit: int = 1_000_000
    response_tokens_limit: int = 16_384


# Tool schema conversion

def tools_to_openai_schema(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert PydanticAI Tool objects or raw callables to OpenAI tool format.

    Handles three cases:
    1. PydanticAI Tool with .definition attribute
    2. PydanticAI Tool with ._schema / .json_schema attribute
    3. Raw async functions (from fallback build_tool_set when PydanticAI absent)
    """
    result: list[dict[str, Any]] = []
    for tool in tools:
        name = ""
        description = ""
        parameters: dict[str, Any] = {"type": "object", "properties": {}}

        # PydanticAI Tool object - extract schema
        if hasattr(tool, "name"):
            name = tool.name
        elif hasattr(tool, "__name__"):
            name = tool.__name__

        if hasattr(tool, "description"):
            description = tool.description or ""
        elif hasattr(tool, "__doc__"):
            description = (tool.__doc__ or "").strip()

        # Try known PydanticAI Tool schema access patterns
        if hasattr(tool, "_schema"):
            schema = tool._schema
            if hasattr(schema, "parameters_json_schema"):
                parameters = schema.parameters_json_schema
            elif isinstance(schema, dict):
                parameters = schema.get("parameters", parameters)
        elif hasattr(tool, "json_schema"):
            parameters = tool.json_schema
        elif hasattr(tool, "definition"):
            defn = tool.definition
            if hasattr(defn, "parameters_json_schema"):
                parameters = defn.parameters_json_schema

        if not name:
            continue

        # OpenAI requires tool names to match ^[a-zA-Z0-9_-]+$
        # MCP tools use dots (mcp.github.list_issues) — convert to double underscore
        api_name = name.replace(".", "__")

        # TAFC: add optional 'think' parameter for reasoning before tool call
        # Improves accuracy on weaker models (arXiv:2601.18282)
        if "properties" in parameters:
            parameters["properties"]["think"] = {
                "type": "string",
                "description": "Brief reasoning for this call (removed before execution)",
            }

        result.append({
            "type": "function",
            "function": {
                "name": api_name,
                "description": description,
                "parameters": parameters,
            },
        })

    return result


def _build_tool_lookup(tools: list[Any]) -> dict[str, Any]:
    """Build name-to-callable lookup from PydanticAI Tool objects or raw functions."""
    lookup: dict[str, Any] = {}
    for tool in tools:
        name = getattr(tool, "name", None) or getattr(tool, "__name__", "")
        if not name:
            continue
        # Store both original and API-safe name for reverse lookup
        api_name = name.replace(".", "__")
        # PydanticAI Tool wraps the function
        fn = tool.function if hasattr(tool, "function") else (tool if callable(tool) else None)
        if fn:
            lookup[name] = fn
            if api_name != name:
                lookup[api_name] = fn
    return lookup


# Provider prefix resolution

_PROVIDER_PREFIX: dict[str, str] = {
    "anthropic": "anthropic/",
    "gemini": "gemini/",
    "azure": "azure/",
    "ollama": "ollama/",
}


def _resolve_litellm_model(model_str: str) -> str:
    """Convert 'provider:model' to LiteLLM format (e.g. 'anthropic/claude-...')."""
    if "/" in model_str:
        return model_str  # Already in LiteLLM format
    if ":" in model_str:
        provider, model_name = model_str.split(":", 1)
        prefix = _PROVIDER_PREFIX.get(provider, "")
        return f"{prefix}{model_name}"
    return model_str  # OpenAI models need no prefix


# StreamResult - mirrors PydanticAI's StreamedRunResult interface

class StreamResult:
    """Drop-in for PydanticAI's ``StreamedRunResult``.

    Provides: stream_text(), usage(), get_output(), all_messages().
    """

    def __init__(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        tool_lookup: dict[str, Any],
        max_tokens: int,
        temperature: float,
        request_tokens_limit: int,
        response_tokens_limit: int,
        max_tool_rounds: int = 10,
        tool_call_policy: Any = None,
    ) -> None:
        self._model = model
        self._messages = list(messages)
        self._tool_schemas = tool_schemas
        self._tool_lookup = tool_lookup
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._request_tokens_limit = request_tokens_limit
        self._response_tokens_limit = response_tokens_limit
        self._max_tool_rounds = max_tool_rounds
        self._collected_text = ""
        self._usage = StreamUsage()
        self._stream: Any = None
        # Tool call policy for weak-model guardrails
        if tool_call_policy is None:
            from rune.agent.tool_call_policy import ToolCallPolicy
            tool_call_policy = ToolCallPolicy()
        self._policy = tool_call_policy

    async def stream_text(self, *, delta: bool = True) -> AsyncIterator[str]:
        """Yield text deltas, auto-executing tool calls when encountered."""
        _max_tool_rounds = self._max_tool_rounds
        _tool_round = 0
        _force_tool = False  # tool_choice="required" flag for retry
        _output_recovery_count = 0  # max output tokens recovery attempts
        _MAX_OUTPUT_RECOVERY = 2
        self._policy.reset()

        while True:
            # Early stop: already have a substantial answer after 2+ tool
            # rounds. Don't make another LLM call that would regenerate
            # the same content and waste tokens.
            if (_tool_round >= 2
                    and self._collected_text
                    and len(self._collected_text.strip()) > 300):
                log.info(
                    "stream_text_early_stop",
                    tool_rounds=_tool_round,
                    text_len=len(self._collected_text),
                )
                break

            # Suppress text yield for turns after we already have a
            # substantial answer.  This prevents the TUI from showing
            # duplicate response boxes when the LLM rewrites its earlier
            # analysis after additional tool calls.
            _suppress_yield = len(self._collected_text.strip()) > 300

            # Build extra params from policy
            extra: dict[str, Any] = self._policy.get_extra_params()
            if _force_tool:
                extra["tool_choice"] = "required"
                _force_tool = False  # one-shot

            self._stream = await litellm.acompletion(
                model=self._model,
                messages=self._messages,
                tools=self._tool_schemas or None,
                stream=True,
                temperature=self._temperature,
                max_tokens=min(self._max_tokens, self._response_tokens_limit),
                stream_options={"include_usage": True},
                **extra,
            )

            text_this_turn = ""
            tool_calls_by_index: dict[int, dict[str, Any]] = {}
            _finish_reason: str | None = None

            async for chunk in self._stream:
                if not chunk.choices:
                    # Usage-only chunk (final)
                    if hasattr(chunk, "usage") and chunk.usage:
                        self._update_usage(chunk.usage)
                    continue

                choice = chunk.choices[0]

                # Track finish_reason from final chunk
                if choice.finish_reason:
                    _finish_reason = choice.finish_reason

                # Text delta
                if choice.delta and choice.delta.content:
                    text_this_turn += choice.delta.content
                    if delta and not _suppress_yield:
                        yield choice.delta.content

                # Tool call deltas - accumulate across chunks
                if choice.delta and choice.delta.tool_calls:
                    for tc in choice.delta.tool_calls:
                        idx = tc.index if hasattr(tc, "index") else 0
                        if idx not in tool_calls_by_index:
                            tool_calls_by_index[idx] = {
                                "id": getattr(tc, "id", None) or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        entry = tool_calls_by_index[idx]
                        if tc.id:
                            entry["id"] = tc.id
                        if hasattr(tc, "function") and tc.function:
                            if tc.function.name:
                                entry["function"]["name"] = tc.function.name
                            if tc.function.arguments:
                                entry["function"]["arguments"] += tc.function.arguments

                # Usage from final chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    self._update_usage(chunk.usage)

            # Only accumulate text that was yielded to the user.
            # Suppressed turns are not added to _collected_text so that
            # get_output() returns only what the user actually saw.
            if not _suppress_yield:
                self._collected_text += text_this_turn

            # No tool calls — check if we should force a retry
            if not tool_calls_by_index:
                if (self._tool_schemas
                        and self._policy.should_force_tool(
                            has_tool_calls=False, has_text=bool(text_this_turn.strip()))):
                    log.info("policy_force_tool_retry")
                    _force_tool = True
                    # Reset collected text so the forced tool round's
                    # subsequent answer is not suppressed.
                    self._collected_text = ""
                    continue  # retry with tool_choice="required"

                # Output truncation recovery: if the response was cut
                # short by max_tokens, inject a "resume" message and
                # escalate the token limit.
                if (
                    _finish_reason == "length"
                    and text_this_turn
                    and _output_recovery_count < _MAX_OUTPUT_RECOVERY
                ):
                    _output_recovery_count += 1
                    # First attempt: double max_tokens
                    if _output_recovery_count == 1:
                        self._max_tokens = min(self._max_tokens * 2, 64_000)
                        log.info(
                            "output_truncation_escalate",
                            new_max=self._max_tokens,
                            attempt=_output_recovery_count,
                        )
                    # Append partial text and inject resume directive
                    self._messages.append({
                        "role": "assistant",
                        "content": text_this_turn,
                    })
                    self._messages.append({
                        "role": "user",
                        "content": (
                            "Output was cut short. Resume directly — "
                            "no recap of what you already wrote. "
                            "Continue from where you stopped."
                        ),
                    })
                    # Don't let the suppress gate block the continuation
                    # text. The user needs to see the resumed output.
                    self._collected_text = ""
                    continue  # retry with higher limit

                # Truly done — append final text
                if text_this_turn:
                    self._messages.append({
                        "role": "assistant",
                        "content": text_this_turn,
                    })
                break

            # Safety: if we already collected a substantial text answer
            # and the LLM is still generating tool calls, execute the
            # pending tool calls (to keep message history valid) but
            # mark this as the last round so no further LLM call is made.
            _force_last_round = False
            if self._collected_text and len(self._collected_text.strip()) > 200 and text_this_turn and len(text_this_turn.strip()) > 100:
                log.info("stream_text_stop_after_answer", text_len=len(self._collected_text))
                _force_last_round = True

            # Max tool rounds guard
            _tool_round += 1
            _is_last_round = (
                _force_last_round
                or _tool_round > _max_tool_rounds
            )
            if _tool_round > _max_tool_rounds:
                log.warning("stream_text_max_tool_rounds", rounds=_tool_round)

            # Always append assistant message WITH tool_calls and execute
            # them. This keeps the message history valid for both the
            # current provider (Anthropic requires tool_result immediately
            # after tool_use) and OpenAI (requires tool response for each
            # tool_call_id).
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if text_this_turn:
                assistant_msg["content"] = text_this_turn
            assistant_msg["tool_calls"] = list(tool_calls_by_index.values())
            self._messages.append(assistant_msg)

            # Execute tool calls. Read-only tools run concurrently,
            # write/execute tools run serially.
            tc_list = list(tool_calls_by_index.values())
            await self._execute_tool_batch(tc_list)

            if _is_last_round:
                break

            # Loop back to make another LLM call with tool results

    _READ_ONLY_TOOLS: frozenset[str] = frozenset({
        "file_read", "file_list", "file_search", "grep", "glob",
        "code_analyze", "code_search", "code_symbols", "project_map",
        "web_search", "web_fetch", "memory_search", "think",
    })
    _MAX_CONCURRENT_TOOLS: int = 5

    async def _execute_tool_batch(
        self, tc_list: list[dict[str, Any]],
    ) -> None:
        """Execute tool calls with read-only batching.

        Consecutive read-only tools run concurrently (up to 5).
        Write/execute tools run one at a time.
        """
        import asyncio as _aio

        # Partition into batches
        batches: list[tuple[bool, list[dict[str, Any]]]] = []
        for tc in tc_list:
            name = tc["function"]["name"]
            is_ro = name in self._READ_ONLY_TOOLS
            if batches and batches[-1][0] == is_ro and is_ro:
                batches[-1][1].append(tc)
            else:
                batches.append((is_ro, [tc]))

        for is_concurrent, batch in batches:
            # Collect nudge messages to append AFTER all tool results
            # (Anthropic requires all tool_result blocks before any
            # other message type).
            deferred_nudges: list[str] = []

            if is_concurrent and len(batch) > 1:
                # Run read-only tools concurrently
                async def _run_one(tc_data: dict[str, Any]) -> tuple[str, str, str]:
                    fn = tc_data["function"]["name"]
                    args_str = tc_data["function"]["arguments"]
                    tc_id = tc_data["id"]
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    if self._policy.should_block_tool(fn):
                        res = f"ERROR: {fn} blocked — called too many times consecutively."
                    else:
                        res = await self._execute_tool(fn, args)
                    return tc_id, fn, res

                _sem = _aio.Semaphore(self._MAX_CONCURRENT_TOOLS)

                async def _limited(tc_data: dict[str, Any], sem: _aio.Semaphore = _sem) -> tuple[str, str, str]:
                    async with sem:
                        return await _run_one(tc_data)

                results = await _aio.gather(*[_limited(tc) for tc in batch])
                for tc_id, fn, res in results:
                    self._messages.append({
                        "role": "tool", "tool_call_id": tc_id, "content": res,
                    })
                    nudge = self._policy.record_tool_call(fn)
                    if nudge:
                        deferred_nudges.append(nudge)
            else:
                # Run serially (write tools or single read)
                for tc_data in batch:
                    fn = tc_data["function"]["name"]
                    args_str = tc_data["function"]["arguments"]
                    tc_id = tc_data["id"]
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    if self._policy.should_block_tool(fn):
                        res = f"ERROR: {fn} blocked — called too many times consecutively."
                        log.warning("policy_tool_blocked", tool=fn)
                    else:
                        res = await self._execute_tool(fn, args)
                    self._messages.append({
                        "role": "tool", "tool_call_id": tc_id, "content": res,
                    })
                    nudge = self._policy.record_tool_call(fn)
                    if nudge:
                        deferred_nudges.append(nudge)

            # Append nudges after ALL tool results for this batch
            for nudge_text in deferred_nudges:
                self._messages.append({"role": "user", "content": nudge_text})
                log.info("policy_tool_loop_nudge")

    async def _execute_tool(self, name: str, params: dict[str, Any]) -> str:
        """Execute a tool by name and return string result."""
        # TAFC: strip 'think' reasoning parameter before execution
        params.pop("think", None)
        func = self._tool_lookup.get(name)
        if func is None:
            return f"Error: unknown tool '{name}'"
        try:
            result = await func(**params)
            return str(result) if result is not None else ""
        except Exception as exc:
            return f"Error executing {name}: {exc}"

    def _update_usage(self, usage: Any) -> None:
        """Extract token counts from a usage object."""
        self._usage.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self._usage.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        self._usage.prompt_tokens = self._usage.input_tokens
        self._usage.completion_tokens = self._usage.output_tokens

    def usage(self) -> StreamUsage:
        """Return accumulated token usage."""
        return self._usage

    async def get_output(self) -> str:
        """Return the full collected text output."""
        return self._collected_text

    def all_messages(self) -> list[dict[str, Any]]:
        """Return the full message history (system + conversation)."""
        return list(self._messages)


# LiteLLMAgent - mirrors PydanticAI Agent interface

class LiteLLMAgent:
    """Drop-in replacement for ``pydantic_ai.Agent``.

    Usage identical to PydanticAI::

        agent = LiteLLMAgent(model="openai:gpt-5.2", system_prompt="...", tools=[...])
        async with agent.run_stream(goal, message_history=msgs, usage_limits=limits) as stream:
            async for delta in stream.stream_text(delta=True):
                print(delta)
            messages = stream.all_messages()
    """

    def __init__(
        self,
        model: str,
        *,
        system_prompt: str = "",
        tools: list[Any] | None = None,
        max_tokens: int = 16_384,
        temperature: float = 0.0,
        max_tool_rounds: int = 10,
        tool_call_policy: Any = None,
    ) -> None:
        self._model = _resolve_litellm_model(model)
        self._system_prompt = system_prompt
        self._tools = tools or []
        self._tool_schemas = tools_to_openai_schema(self._tools)
        self._tool_lookup = _build_tool_lookup(self._tools)
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_tool_rounds = max_tool_rounds
        if tool_call_policy is None:
            from rune.agent.tool_call_policy import ToolCallPolicy
            tool_call_policy = ToolCallPolicy()
        self._policy = tool_call_policy

    @asynccontextmanager
    async def run_stream(
        self,
        goal: str,
        *,
        message_history: list[Any] | None = None,
        usage_limits: Any = None,
    ) -> AsyncIterator[StreamResult]:
        """Start a streaming run. Mirrors ``Agent.run_stream()``."""
        # Build messages list
        messages: list[dict[str, Any]] = []

        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        # Append history (convert PydanticAI messages to dicts if needed)
        # Skip system messages from history; we already prepended our own.
        if message_history:
            for msg in message_history:
                if isinstance(msg, dict):
                    if msg.get("role") == "system":
                        continue  # Avoid duplicating system prompt
                    messages.append(msg)
                else:
                    # PydanticAI message object - extract role + content
                    role = getattr(msg, "role", "user")
                    if role == "system":
                        continue
                    content = getattr(msg, "content", str(msg))
                    messages.append({"role": role, "content": content})

        # Append goal as user message, but only if it's not already
        # the last user message in history (prevents duplication when
        # the outer loop calls run_stream() on every step).
        _dominated_by_history = False
        if message_history:
            # Check if the conversation already contains the goal
            for msg in reversed(messages):
                role = msg.get("role", "") if isinstance(msg, dict) else ""
                if role == "user":
                    content = msg.get("content", "") if isinstance(msg, dict) else ""
                    if content == goal:
                        _dominated_by_history = True
                    break  # Only check the most recent user message

        if not _dominated_by_history:
            messages.append({"role": "user", "content": goal})

        # Extract limits
        request_limit = 1_000_000
        response_limit = self._max_tokens
        if usage_limits is not None:
            request_limit = getattr(usage_limits, "request_tokens_limit", request_limit) or request_limit
            response_limit = getattr(usage_limits, "response_tokens_limit", response_limit) or response_limit

        stream_result = StreamResult(
            model=self._model,
            messages=messages,
            tool_schemas=self._tool_schemas,
            tool_lookup=self._tool_lookup,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            request_tokens_limit=request_limit,
            response_tokens_limit=response_limit,
            max_tool_rounds=self._max_tool_rounds,
            tool_call_policy=self._policy,
        )

        yield stream_result

    def update_model(self, model: str) -> None:
        """Switch the underlying model (used by failover)."""
        self._model = _resolve_litellm_model(model)

    def update_tools(self, tools: list[Any]) -> None:
        """Replace the tool set (used when tools change mid-loop)."""
        self._tools = tools
        self._tool_schemas = tools_to_openai_schema(tools)
        self._tool_lookup = _build_tool_lookup(tools)
