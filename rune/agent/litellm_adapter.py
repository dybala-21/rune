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

# Suppress cost-calculation warnings for models not in LiteLLM's price DB
# (e.g., local ollama models routed via openai/ prefix).
litellm.suppress_debug_info = True

from rune.agent.message_utils import validate_tool_pairs
from rune.agent.obs_cap import mask_stale_tool_messages
from rune.capabilities.output_prefixes import looks_like_failure_output
from rune.utils.env import env_flag as _env_flag
from rune.utils.env import env_int as _env_int
from rune.utils.logger import get_logger

log = get_logger(__name__)

_BENCH_STOP_BATCH_ON_TOOL_FAILURE_ENV = "RUNE_BENCH_STOP_BATCH_ON_TOOL_FAILURE"
# Optional per-assistant-turn cap on serial write/execute tool calls. Off by
# default (unset/<=0). When set to N>0, only the first N write/execute calls in
# a turn run; the rest are skipped so the model must read their results before
# scheduling more. Counters the v6/v7 single-turn tool explosion where one
# response queued many alternative artifact writes before any feedback.
_BENCH_MAX_WRITE_EXEC_PER_TURN_ENV = "RUNE_BENCH_MAX_WRITE_EXEC_PER_TURN"
_STOP_BATCH_FAILURE_TOOLS = frozenset({
    "bash_execute",
    "file_delete",
    "file_edit",
    "file_write",
})
# Tools subject to the per-turn cap — same mutating/executing set as the
# failure-stop list.
_WRITE_EXEC_TOOLS = _STOP_BATCH_FAILURE_TOOLS


def _looks_like_tool_failure(result: str) -> bool:
    return looks_like_failure_output(result)


def _redirect_edit_to_write(fn: str, args: dict[str, Any]) -> str:
    """Map a misused file_edit call onto file_write when the model supplied the
    whole new file as `content` instead of a search/replace pair.

    Weak local models conflate the two tools: they pass `content` to file_edit,
    which fails schema validation, and then retry the identical call. When an
    edit carries `content` but no `search`, the intent is a full-file write, so
    rewrite the call. Edit-specific keys are dropped so file_write validates.
    """
    if fn == "file_edit" and args.get("content") and not args.get("search"):
        args.pop("replace", None)
        args.pop("all", None)
        return "file_write"
    return fn


def _skipped_after_batch_failure_message(tool_name: str) -> str:
    return (
        "[BLOCKED] Skipped this tool because an earlier write/execute tool in the same "
        "assistant turn failed. Read the failed tool result first, then issue a new "
        f"{tool_name} call only after choosing the next fix."
    )


def _skipped_after_write_exec_cap_message(tool_name: str, cap: int) -> str:
    return (
        f"[BLOCKED] Skipped this tool: the per-turn write/execute limit ({cap}) was "
        "reached. Read the results of the writes/commands already issued in this turn, "
        f"then decide the next {tool_name} call in your following response."
    )


# Usage tracking

@dataclass(slots=True)
class StreamUsage:
    """Token usage from a streaming completion."""
    input_tokens: int = 0
    output_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(slots=True)
class UsageLimits:
    """Drop-in replacement for pydantic_ai.usage.UsageLimits."""
    request_tokens_limit: int = 1_000_000
    response_tokens_limit: int = 16_384


# Tool schema conversion

def tools_to_openai_schema(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert tool objects to OpenAI tool format. Advisor-type schemas
    (``advisor_*``) pass through unwrapped."""
    from rune.agent.advisor.native_tool import is_native_schema

    result: list[dict[str, Any]] = []
    for tool in tools:
        # Passthrough for Anthropic native server-side tools.
        if hasattr(tool, "json_schema") and is_native_schema(tool.json_schema):
            result.append(dict(tool.json_schema))
            continue

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


def _iter_json_objects(text: str) -> list[str]:
    """Yield top-level {...} JSON substrings via a balanced-brace scan.

    Handles prose around the JSON and nested braces (unlike a regex), and skips
    braces inside strings. Used to recover tool calls emitted as text.
    """
    out: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start : i + 1])
                    start = -1
    return out


# Common argument-key aliases weak models emit for RUNE's tools. Every file tool
# uses "path" and bash uses "command", so these renames are unambiguous. Applied
# only to recovered text tool-calls (where args aren't schema-constrained).
_ARG_ALIASES: dict[str, str] = {
    "filename": "path", "file": "path", "filepath": "path", "file_path": "path",
    "cmd": "command", "code": "content",
}


def _normalize_arg_keys(args: dict[str, Any]) -> dict[str, Any]:
    """Rename common alias keys to RUNE's canonical tool params (e.g. filename
    to path), only when the canonical key isn't already present."""
    out = dict(args)
    for alias, canonical in _ARG_ALIASES.items():
        if alias in out and canonical not in out:
            out[canonical] = out.pop(alias)
    return out


def _extract_text_tool_calls(
    text: str, known_names: set[str]
) -> list[dict[str, Any]]:
    """Recover tool calls a model emitted as text JSON instead of native tool_calls.

    Small local models (qwen2.5-coder, gemma) routed through ollama sometimes emit
    ``{"name": "file_write", "arguments": {...}}`` in the message content rather
    than as native ``tool_calls``, so the loop would treat it as a plain text
    answer and run nothing. Parse such JSON, require the name to be in the known
    tool set (so real text answers aren't misread), and synthesize tool_call
    entries for the normal execution path.
    """
    import json as _json

    if not text or "{" not in text or not known_names:
        return []
    calls: list[dict[str, Any]] = []
    for blob in _iter_json_objects(text):
        try:
            obj = _json.loads(blob)
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        # Accept common shapes: {name,arguments} / {tool,args} / {function,parameters}
        name = obj.get("name") or obj.get("tool") or obj.get("function")
        if isinstance(name, dict):  # {"function": {"name": ..., "arguments": ...}}
            inner = name
            name = inner.get("name")
            args = inner.get("arguments", inner.get("parameters", {}))
        else:
            args = obj.get("arguments")
            if args is None:
                args = obj.get("parameters", obj.get("args", obj.get("input", {})))
        if not isinstance(name, str) or name not in known_names:
            continue
        if isinstance(args, dict):
            args = _normalize_arg_keys(args)
        if isinstance(args, str):
            args_str = args  # already a JSON string
        else:
            try:
                args_str = _json.dumps(args if isinstance(args, dict) else {})
            except (ValueError, TypeError):
                args_str = "{}"
        calls.append({
            "id": f"call_text_{len(calls)}",
            "type": "function",
            "function": {"name": name, "arguments": args_str},
        })
    return calls


def _guided_tools_enabled() -> bool:
    """Whether to schema-constrain tool calls for local models (default off).

    Local models (ollama) often fail to emit native tool_calls and instead write
    them as text, so the loop runs nothing. With guided decoding the output is
    grammar-forced to a tool-call schema, so the model produces a valid action.
    Opt-in via RUNE_GUIDED_TOOLS.
    """
    import os

    return os.environ.get("RUNE_GUIDED_TOOLS", "").strip().lower() in (
        "1", "true", "on", "yes",
    )


def _guided_for_model(model: str, has_tools: bool, provider_extra: object) -> bool:
    """Whether guided decoding applies: enabled, has tools, an ollama endpoint,
    and not a '-cloud' model (cloud models use native tool calls)."""
    return (
        _guided_tools_enabled()
        and has_tools
        and "11434" in str(provider_extra)
        and "-cloud" not in str(model).lower()
    )


def _build_action_schema(tool_schemas: list[dict[str, Any]]) -> dict[str, Any]:
    """A JSON schema the model's output must satisfy each turn: either a tool
    call (``{tool, arguments}``) or a final answer (``{final}``).

    The tool name is constrained to the real tool set (an enum), so the model
    can't invent a tool; ``arguments`` is a free object. Per-tool argument
    schemas are intentionally NOT inlined: an anyOf over dozens of rich tool
    param schemas is rejected by ollama's grammar engine as "invalid JSON
    schema", whereas a small enum plus generic args scales to the full tool set
    and is accepted (argument-key correctness is left to the tools' own
    validation). This guarantees a parseable call naming a real tool, which weak
    local models otherwise fail to emit.
    """
    names = [
        (s.get("function") or {}).get("name")
        for s in tool_schemas
    ]
    names = [n for n in names if n]
    return {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": names},
                    "arguments": {"type": "object"},
                },
                "required": ["tool", "arguments"],
            },
            {
                "type": "object",
                "properties": {"final": {"type": "string"}},
                "required": ["final"],
            },
        ],
    }


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


def _usage_value(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _coerce_usage_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _nested_usage_int(usage: Any, path: tuple[str, ...]) -> int:
    value = usage
    for name in path:
        value = _usage_value(value, name)
        if value is None:
            return 0
    return _coerce_usage_int(value)


def _usage_int(
    usage: Any,
    *names: str,
    nested: tuple[tuple[str, ...], ...] = (),
) -> int:
    for name in names:
        value = _coerce_usage_int(_usage_value(usage, name))
        if value:
            return value
    for path in nested:
        value = _nested_usage_int(usage, path)
        if value:
            return value
    return 0


# Provider prefix resolution

_PROVIDER_PREFIX: dict[str, str] = {
    "anthropic": "anthropic/",
    "gemini": "gemini/",
    "azure": "azure/",
    # Ollama uses openai/ prefix to route through the OpenAI-compatible
    # /v1 endpoint. LiteLLM's native ollama/ adapter has broken tool_calls
    # parsing in streaming mode (BerriAI/litellm#24091).
    "ollama": "openai/",
}

# Extra kwargs injected per provider (e.g., api_base for ollama).
_PROVIDER_EXTRA: dict[str, dict[str, str]] = {
    "ollama": {
        "api_base": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
}


def _is_o_series(model: str) -> bool:
    """O-series reasoning models (o1/o3/o4) only support temperature=1."""
    import re as _re
    base = model.rsplit("/", 1)[-1]
    return bool(_re.match(r"^o[134](-|$)", base))


def _rejects_temperature(model: str) -> bool:
    """True for reasoning models that reject temperature != 1: o1/o3/o4 and the
    gpt-5 family (but not gpt-5.x point releases like gpt-5.1, which allow it).
    """
    if _is_o_series(model):
        return True
    base = model.rsplit("/", 1)[-1].lower()
    return "gpt-5" in base and "gpt-5." not in base


def _vertex_active() -> bool:
    """True when GOOGLE_APPLICATION_CREDENTIALS points to an existing file."""
    import os as _os
    path = _os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not (path and _os.path.isfile(path)):
        return False
    global _VERTEX_SDK_WARNED
    try:
        import google.auth  # noqa: F401
    except ImportError:
        if not _VERTEX_SDK_WARNED:
            log.warning(
                "vertex_sdk_missing",
                hint="install with: uv pip install 'rune-ai[vertex]'",
            )
            _VERTEX_SDK_WARNED = True
    return True


_VERTEX_SDK_WARNED: bool = False


def _vertex_extra() -> dict[str, str]:
    """Return vertex_project / vertex_location kwargs for LiteLLM."""
    import os as _os
    extra: dict[str, str] = {}
    proj = _os.environ.get("VERTEX_PROJECT") or _os.environ.get("VERTEXAI_PROJECT")
    loc = (
        _os.environ.get("VERTEX_LOCATION")
        or _os.environ.get("VERTEXAI_LOCATION")
        or "us-central1"
    )
    if proj:
        extra["vertex_project"] = proj
    extra["vertex_location"] = loc
    return extra


_MODEL_OUTPUT_CAP_CACHE: dict[str, int] = {}


def _model_output_cap(model: str) -> int | None:
    """Return the model's max output-token limit (cached). None if unknown."""
    cached = _MODEL_OUTPUT_CAP_CACHE.get(model)
    if cached is not None:
        return cached if cached > 0 else None
    try:
        import litellm as _litellm
        info = _litellm.get_model_info(model) or {}
        limit = info.get("max_output_tokens") or info.get("max_tokens")
        if isinstance(limit, int) and limit > 0:
            _MODEL_OUTPUT_CAP_CACHE[model] = limit
            return limit
    except Exception:
        pass
    _MODEL_OUTPUT_CAP_CACHE[model] = 0  # sentinel: unknown
    return None


def _clamp_max_tokens(model: str, max_tokens: int) -> int:
    """Clamp a desired max_tokens to the model's hard output cap."""
    cap = _model_output_cap(model)
    if cap is None:
        return max_tokens
    return min(max_tokens, cap)


def _resolve_litellm_model(model_str: str) -> tuple[str, dict[str, str]]:
    """Convert 'provider:model' to LiteLLM format.

    Returns (model_id, extra_kwargs) where extra_kwargs contains
    provider-specific parameters like api_base for ollama or
    vertex_project/vertex_location for Vertex AI.
    """
    # Already in LiteLLM format (e.g. "ollama/gemma4:26b" from failover)
    if "/" in model_str:
        # Reroute ollama/ to openai/ for working tool calling
        if model_str.startswith("ollama/"):
            model_name = model_str[len("ollama/"):]
            extra = dict(_PROVIDER_EXTRA.get("ollama", {}))
            return f"openai/{model_name}", extra
        # Reroute gemini/ to vertex_ai/ when service-account creds are active
        if model_str.startswith("gemini/") and _vertex_active():
            model_name = model_str[len("gemini/"):]
            return f"vertex_ai/{model_name}", _vertex_extra()
        if model_str.startswith("vertex_ai/"):
            return model_str, _vertex_extra()
        return model_str, {}
    # RUNE format: "provider:model"
    if ":" in model_str:
        provider, model_name = model_str.split(":", 1)
        if provider == "gemini" and _vertex_active():
            return f"vertex_ai/{model_name}", _vertex_extra()
        if provider == "vertex_ai":
            return f"vertex_ai/{model_name}", _vertex_extra()
        prefix = _PROVIDER_PREFIX.get(provider, "")
        extra = dict(_PROVIDER_EXTRA.get(provider, {}))
        return f"{prefix}{model_name}", extra
    return model_str, {}  # OpenAI models need no prefix


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
        provider_extra: dict[str, str] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._model = model
        self._messages = list(messages)
        self._tool_schemas = tool_schemas
        self._tool_lookup = tool_lookup
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._request_tokens_limit = request_tokens_limit
        self._response_tokens_limit = response_tokens_limit
        self._provider_extra = provider_extra or {}
        self._extra_headers = extra_headers or {}
        self._max_tool_rounds = max_tool_rounds
        # Guided decoding: schema-constrain tool calls for local (ollama) models,
        # detected via the ollama api_base. When on, the model must emit a tool
        # call or a final answer as schema-valid JSON each turn.
        self._guided = _guided_for_model(
            self._model, bool(tool_schemas), self._provider_extra
        )
        # Guided-mode {final} guard: weak models bail with a final answer after a
        # failed edit without ever writing anything. Block {final} until a
        # write/exec tool has actually succeeded (capped, so genuine no-op finals
        # still terminate).
        self._action_ok = False
        self._final_blocks = 0
        if self._guided:
            self._messages.append({
                "role": "system",
                "content": (
                    "Respond ONLY as JSON matching the schema. To use a tool, "
                    'emit {"tool": "<tool_name>", "arguments": {...}}. When the '
                    'task is fully done, emit {"final": "<answer>"}. One object '
                    "per turn. To CREATE a new file use file_write with "
                    '{"path", "content"}; file_edit only modifies an existing '
                    "file. Do not emit a final answer until the work is actually "
                    "done (files written, tests run). When the code involves a "
                    "calculation or multi-step rule, first write the computation "
                    "as explicit ordered steps in comments, defining each "
                    "intermediate value (compute the base result, then apply each "
                    "further adjustment to that result in sequence), then "
                    "implement to follow those steps."
                ),
            })
        self._collected_text = ""
        self._usage = StreamUsage()
        self._stream: Any = None
        self._tool_result_cache: dict[str, str] = {}
        self._tool_fail_streak: dict[str, int] = {}
        self._blocked_groups: set[str] = set()
        self._native_advisor_events: list[dict[str, Any]] = []
        # Tool call policy for weak-model guardrails
        if tool_call_policy is None:
            from rune.agent.tool_call_policy import ToolCallPolicy
            tool_call_policy = ToolCallPolicy()
        self._policy = tool_call_policy

    # Cross-step failure state injection/export

    def inject_failure_state(
        self, streak: dict[str, int], blocked: set[str],
    ) -> None:
        """Seed failure counters from a previous step's state."""
        self._tool_fail_streak.update(streak)
        self._blocked_groups.update(blocked)

    def get_failure_state(self) -> tuple[dict[str, int], set[str]]:
        """Export current failure state for cross-step persistence."""
        return dict(self._tool_fail_streak), set(self._blocked_groups)

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

            _tools = self._tool_schemas or None

            _effective_max = _clamp_max_tokens(
                self._model,
                min(self._max_tokens, self._response_tokens_limit),
            )
            _acompletion_kwargs: dict[str, Any] = {
                "model": self._model,
                # Mask stale tool outputs for the wire only; self._messages
                # stays full so history/rollover are unaffected.
                "messages": mask_stale_tool_messages(validate_tool_pairs(self._messages)),
                "tools": _tools,
                "stream": True,
                "max_tokens": _effective_max,
                "stream_options": {"include_usage": True},
            }
            if not _rejects_temperature(self._model):
                _acompletion_kwargs["temperature"] = self._temperature
            if self._extra_headers:
                _acompletion_kwargs["extra_headers"] = dict(self._extra_headers)
            _acompletion_kwargs.update(self._provider_extra)
            _acompletion_kwargs.update(extra)

            # Guided decoding for local models: replace the native tools param
            # with a schema the output must satisfy (tool call or final answer),
            # so a weak model that can't emit native tool_calls still produces a
            # valid, parseable action. ollama-only (detected via its api_base);
            # the content-JSON is parsed by the recovery path below.
            if self._guided:
                _acompletion_kwargs.pop("tools", None)
                _acompletion_kwargs.pop("tool_choice", None)
                _gschema = _build_action_schema(self._tool_schemas)
                log.info("guided_decoding_active",
                         tools=len(self._tool_schemas),
                         branches=len(_gschema.get("anyOf", [])))
                _acompletion_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "rune_action", "schema": _gschema},
                }

            self._stream = await litellm.acompletion(**_acompletion_kwargs)

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

                # Text delta — buffer only, do NOT yield yet.
                # We must wait until the stream ends to know whether
                # tool_calls are present.  Yielding text before that
                # check causes hallucinated text to reach the UI when
                # the LLM generates text + tool_calls in the same turn.
                if choice.delta and choice.delta.content:
                    text_this_turn += choice.delta.content

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

            # Local-model fallback: if there were no native tool_calls but the
            # text is a JSON tool call naming a known tool, recover it. Small
            # ollama models degrade to text tool-calls under the full tool set +
            # streaming; models that emit native tool_calls never reach here.
            if not tool_calls_by_index and text_this_turn.strip():
                # Guided mode emits the final answer as {"final": "..."}; unwrap
                # it to clean text so the loop finalizes with the answer, not raw
                # JSON. A {"tool", "arguments"} object falls through to recovery.
                if self._guided:
                    import json as _json
                    try:
                        _obj = _json.loads(text_this_turn.strip())
                    except (ValueError, TypeError):
                        _obj = None
                    if isinstance(_obj, dict) and isinstance(_obj.get("final"), str):
                        if not self._action_ok and self._final_blocks < 2:
                            # Bailing without doing the work; force another round.
                            self._final_blocks += 1
                            log.info("guided_final_blocked", n=self._final_blocks)
                            self._messages.append(
                                {"role": "assistant", "content": text_this_turn})
                            self._messages.append({"role": "user", "content": (
                                "You returned a final answer but have not "
                                "successfully written any file or run any command. "
                                "Do the actual work now: use file_write to create "
                                "files and bash_execute to run the tests, then "
                                "finalize."
                            )})
                            self._collected_text = ""
                            continue  # re-prompt instead of finalizing
                        text_this_turn = _obj["final"]

                _known = {
                    s.get("function", {}).get("name") or s.get("name")
                    for s in (self._tool_schemas or [])
                }
                _known.discard(None)
                _recovered = _extract_text_tool_calls(text_this_turn, _known)
                if _recovered:
                    log.info("text_tool_call_recovered", count=len(_recovered),
                             names=[c["function"]["name"] for c in _recovered])
                    for _i, _c in enumerate(_recovered):
                        tool_calls_by_index[_i] = _c
                    text_this_turn = ""  # the text WAS the call, not an answer

            # Post-stream decision: yield text or discard
            # We check ALL continuation paths (tool_calls, force_tool,
            # truncation recovery) BEFORE yielding.  Any path that
            # loops back makes the current text intermediate/speculative.
            _has_tools = bool(tool_calls_by_index) or _finish_reason == "tool_calls"
            _discard = _has_tools and bool(text_this_turn)

            if _discard:
                log.info(
                    "speculative_text_discarded",
                    text_len=len(text_this_turn),
                    tool_count=len(tool_calls_by_index),
                )

            # No tool calls - check continuation paths before yielding
            if not tool_calls_by_index:
                # force_tool_on_empty: only retry when the model produced
                # NO text at all (empty response).  If text was generated,
                # the model decided tools weren't needed — respect that.
                # Change A already handles text+tools (hallucination).
                if (not text_this_turn.strip()
                        and self._tool_schemas
                        and self._policy.should_force_tool(
                            has_tool_calls=False, has_text=False)):
                    log.info("policy_force_tool_retry")
                    _force_tool = True
                    _discard = True
                    self._collected_text = ""
                    continue  # retry with tool_choice="required"

                # Output truncation recovery
                if (
                    _finish_reason == "length"
                    and text_this_turn
                    and _output_recovery_count < _MAX_OUTPUT_RECOVERY
                ):
                    _output_recovery_count += 1
                    if _output_recovery_count == 1:
                        self._max_tokens = _clamp_max_tokens(
                            self._model, min(self._max_tokens * 2, 64_000),
                        )
                        log.info(
                            "output_truncation_escalate",
                            new_max=self._max_tokens,
                            attempt=_output_recovery_count,
                        )
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
                    self._collected_text = ""
                    continue  # retry with higher limit

            # Yield (only reached if no continuation path triggered)
            if not _discard and text_this_turn:
                if delta and not _suppress_yield:
                    yield text_this_turn
                    self._collected_text += text_this_turn
                elif not _suppress_yield:
                    self._collected_text += text_this_turn

            # Pure text, no continuations — done
            if not tool_calls_by_index:
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

            # Replace previous browser_observe results with 1-line
            # summary.  Only the latest snapshot is useful; older ones
            # waste tokens as context accumulates across rounds.
            _latest_observe_idx = -1
            for _i, _m in enumerate(self._messages):
                if (_m.get("role") == "tool"
                        and "Interactive Elements" in _m.get("content", "")):
                    _latest_observe_idx = _i
            if _latest_observe_idx > 0:
                for _i, _m in enumerate(self._messages[:_latest_observe_idx]):
                    if (_m.get("role") == "tool"
                            and "Interactive Elements" in _m.get("content", "")):
                        _m["content"] = "[Previous page snapshot — superseded by latest observe]"

            if _is_last_round:
                break

            # Loop back to make another LLM call with tool results

    # Tool group mapping for failure-based group blocking.
    # When browser_act fails 3 times, all browser_* tools are blocked.
    _TOOL_GROUPS: dict[str, str] = {
        "browser_act": "browser", "browser_navigate": "browser",
        "browser_observe": "browser", "browser_find": "browser",
        "browser_extract": "browser", "browser_batch": "browser",
        "browser_screenshot": "browser", "browser_open": "browser",
        "browser_discover_apis": "browser",
    }

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

        # Collect nudge messages and append them only after all tool results
        # for this assistant turn. OpenAI requires every tool_call_id to be
        # answered by contiguous tool messages before any user/system message.
        deferred_nudges: list[str] = []
        stop_after_failure = _env_flag(_BENCH_STOP_BATCH_ON_TOOL_FAILURE_ENV)
        batch_failure_seen = False
        # Per-turn write/execute cap (off by default; None when unset/<=0).
        write_exec_cap = _env_int(_BENCH_MAX_WRITE_EXEC_PER_TURN_ENV)
        write_exec_count = 0

        for is_concurrent, batch in batches:
            if batch_failure_seen:
                for tc_data in batch:
                    fn = tc_data["function"]["name"]
                    tc_id = tc_data["id"]
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": _skipped_after_batch_failure_message(fn),
                    })
                continue

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
                    if batch_failure_seen:
                        self._messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": _skipped_after_batch_failure_message(fn),
                        })
                        continue
                    is_write_exec = fn in _WRITE_EXEC_TOOLS
                    if (
                        write_exec_cap is not None
                        and is_write_exec
                        and write_exec_count >= write_exec_cap
                    ):
                        self._messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": _skipped_after_write_exec_cap_message(fn, write_exec_cap),
                        })
                        log.info("bench_write_exec_cap_skip", tool=fn, cap=write_exec_cap)
                        continue
                    if is_write_exec:
                        write_exec_count += 1
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    if self._guided:
                        _redirected = _redirect_edit_to_write(fn, args)
                        if _redirected != fn:
                            fn = _redirected
                            log.info("guided_edit_to_write_redirect")
                    if self._policy.should_block_tool(fn):
                        res = f"ERROR: {fn} blocked — called too many times consecutively."
                        log.warning("policy_tool_blocked", tool=fn)
                    else:
                        res = await self._execute_tool(fn, args)
                    self._messages.append({
                        "role": "tool", "tool_call_id": tc_id, "content": res,
                    })
                    if is_write_exec and not _looks_like_tool_failure(res):
                        self._action_ok = True  # a real action succeeded
                    nudge = self._policy.record_tool_call(fn)
                    if nudge:
                        deferred_nudges.append(nudge)
                    if (
                        stop_after_failure
                        and fn in _STOP_BATCH_FAILURE_TOOLS
                        and _looks_like_tool_failure(res)
                    ):
                        batch_failure_seen = True

        for nudge_text in deferred_nudges:
            self._messages.append({"role": "user", "content": nudge_text})
            log.info("policy_tool_loop_nudge")

    async def _execute_tool(self, name: str, params: dict[str, Any]) -> str:
        """Execute a tool by name and return string result.

        Identical (name, params) calls within the same stream_text()
        session return a cached result to prevent wasteful repetition
        (e.g. same URL fetched 26 times).
        """
        # TAFC: strip 'think' reasoning parameter before execution
        params.pop("think", None)

        # Dedup: return cached result for identical calls
        import hashlib as _hl
        _cache_key = _hl.md5(
            f"{name}:{json.dumps(params, sort_keys=True)}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:16]
        if _cache_key in self._tool_result_cache:
            return (
                "[CACHED — identical call already executed. "
                "Use the result above or try a different approach.]\n"
                + self._tool_result_cache[_cache_key][:500]
            )

        # Block tool after 3 consecutive failures.  When a tool like
        # browser_act fails 3 times, block the ENTIRE tool group
        # (all browser_* tools) since navigate+observe without
        # working act is just burning tokens.
        _group = self._TOOL_GROUPS.get(name, "")

        _MAX_FAILS = 3

        if _group and _group in self._blocked_groups:
            return (
                f"[BLOCKED — {_group} tools disabled after repeated failures. "
                f"Use web_search or web_fetch instead.]"
            )

        if self._tool_fail_streak.get(name, 0) >= _MAX_FAILS:
            if _group:
                self._blocked_groups.add(_group)
                return (
                    f"[BLOCKED — {name} failed {_MAX_FAILS} times. "
                    f"All {_group} tools disabled. Use web_search instead.]"
                )
            return (
                f"[BLOCKED — {name} failed {_MAX_FAILS} times. "
                f"Try a different approach.]"
            )

        func = self._tool_lookup.get(name)
        if func is None:
            return f"Error: unknown tool '{name}'"
        try:
            result = await func(**params)
            result_str = str(result) if result is not None else ""
            self._tool_result_cache[_cache_key] = result_str
            # Reset fail streak on success.
            # Also treat "NO CHANGES DETECTED" as failure — the action
            # technically executed but had no effect (phantom click).
            is_failure = _looks_like_tool_failure(result_str)
            if is_failure:
                streak = self._tool_fail_streak.get(name, 0) + 1
                self._tool_fail_streak[name] = streak
                # At 2 failures: hint to try URL construction before
                # the group gets blocked at 3.
                if streak == 2 and name == "browser_act":
                    result_str += (
                        "\n\n[HINT] browser_act failed twice. Before trying again, "
                        "construct the target URL directly with browser_navigate. "
                        "Example: browser_navigate(url='https://site.com/search?q=keyword&sort=review')"
                    )
                # Block entire group immediately when threshold reached
                if streak >= _MAX_FAILS and _group:
                    self._blocked_groups.add(_group)
            else:
                self._tool_fail_streak[name] = 0
            return result_str
        except Exception as exc:
            streak = self._tool_fail_streak.get(name, 0) + 1
            self._tool_fail_streak[name] = streak
            if streak >= _MAX_FAILS and _group:
                self._blocked_groups.add(_group)
            return f"Error executing {name}: {exc}"

    def _update_usage(self, usage: Any) -> None:
        """Extract token counts; parse native advisor events when beta is active."""
        self._usage.input_tokens += _usage_int(usage, "prompt_tokens", "input_tokens")
        self._usage.output_tokens += _usage_int(usage, "completion_tokens", "output_tokens")
        self._usage.prompt_tokens = self._usage.input_tokens
        self._usage.completion_tokens = self._usage.output_tokens
        self._usage.cached_input_tokens += _usage_int(
            usage,
            "cached_tokens",
            "cache_read_input_tokens",
            "cache_read_tokens",
            nested=(
                ("prompt_tokens_details", "cached_tokens"),
                ("prompt_tokens_details", "cache_read_input_tokens"),
                ("input_token_details", "cache_read"),
                ("input_token_details", "cached_tokens"),
            ),
        )
        self._usage.cache_write_tokens += _usage_int(
            usage,
            "cache_creation_input_tokens",
            "cache_write_input_tokens",
            "cache_write_tokens",
            nested=(
                ("prompt_tokens_details", "cache_creation_tokens"),
                ("prompt_tokens_details", "cache_write_tokens"),
                ("input_token_details", "cache_creation"),
                ("input_token_details", "cache_write"),
            ),
        )
        self._usage.reasoning_tokens += _usage_int(
            usage,
            "reasoning_tokens",
            nested=(
                ("completion_tokens_details", "reasoning_tokens"),
                ("output_token_details", "reasoning_tokens"),
            ),
        )
        if self._extra_headers.get("anthropic-beta"):
            from rune.agent.advisor.native_tool import (
                extract_synthetic_events_from_usage,
            )
            synthetic = extract_synthetic_events_from_usage(usage)
            if synthetic:
                self._native_advisor_events.extend(synthetic)

    def native_advisor_events(self) -> list[dict[str, Any]]:
        return list(self._native_advisor_events)

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
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._model, self._provider_extra = _resolve_litellm_model(model)
        self._system_prompt = system_prompt
        self._tools = tools or []
        self._tool_schemas = tools_to_openai_schema(self._tools)
        self._tool_lookup = _build_tool_lookup(self._tools)
        self._requested_max_tokens = max_tokens
        self._max_tokens = _clamp_max_tokens(self._model, max_tokens)
        self._temperature = temperature
        self._max_tool_rounds = max_tool_rounds
        self._extra_headers: dict[str, str] = dict(extra_headers or {})
        self._last_stream_result: StreamResult | None = None
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
            provider_extra=self._provider_extra,
            extra_headers=self._extra_headers,
        )
        self._last_stream_result = stream_result

        yield stream_result

    def native_advisor_events(self) -> list[dict[str, Any]]:
        if self._last_stream_result is None:
            return []
        return self._last_stream_result.native_advisor_events()

    def update_model(self, model: str) -> None:
        """Switch the underlying model (used by failover)."""
        self._model, self._provider_extra = _resolve_litellm_model(model)
        self._max_tokens = _clamp_max_tokens(
            self._model, self._requested_max_tokens,
        )

    def update_tools(self, tools: list[Any]) -> None:
        """Replace the tool set (used when tools change mid-loop)."""
        self._tools = tools
        self._tool_schemas = tools_to_openai_schema(tools)
        self._tool_lookup = _build_tool_lookup(tools)
