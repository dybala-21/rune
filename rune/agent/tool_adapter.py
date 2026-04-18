"""Tool adapter for RUNE. Bridges capabilities to LLM tool format.

Ported from src/agent/tool-adapter.ts. Covers stall limits, tool set construction,
cognitive caching integration, and Guardian validation wrappers.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import re
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rune.agent.cognitive_cache import SessionToolCache
from rune.capabilities.output_prefixes import (
    BASH_CMD_PREFIX,
    BASH_EXIT_PREFIX,
    FILE_READ_PATH_PREFIX,
)
from rune.capabilities.registry import CapabilityRegistry, get_capability_registry
from rune.capabilities.types import CapabilityDefinition
from rune.types import CapabilityResult
from rune.utils.logger import get_logger

log = get_logger(__name__)


# ToolWrapper - lightweight replacement for PydanticAI Tool

@dataclass(slots=True)
class ToolWrapper:
    """Lightweight tool descriptor compatible with ``tools_to_openai_schema()``."""
    name: str
    description: str
    json_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    function: Any = None  # async callable(**kwargs) -> str


# Stall limits - frozen constant thresholds for stall detection

STALL_LIMITS: dict[str, Any] = {
    "fileRead": {
        "warning": 20,
        "hardStop": 30,
        "sameFile": 2,
    },
    "fileWrite": {
        "sameFile": 6,
    },
    "bash": {
        "consecutiveFailures": 8,
        "intentRepeat": 8,
        "errorSignature": 5,
        "intentFailure": 6,
    },
    "cycle": {
        "normal": 3,
        "devWorkflow": 6,
        "browser": 5,
    },
    "web": {
        "fetchWarning": 10,
        "fetchHardStop": 15,
        "searchWarning": 8,
        "searchHardStop": 12,
        "sameUrlRepeat": 2,
    },
    "browserFind": {
        "noMatchWarning": 3,
        "callWarning": 5,
        "noMatchHardStop": 6,
    },
    "expensive": {
        "count": 6,
        "slowCount": 3,
        "slowThresholdMs": 30_000,
    },
    "maxNudges": 5,
    "recentCallWindow": 8,
    "callHistoryWindow": 30,
}

# Multiplier applied when "extended" stall mode is active
EXTENDED_MULTIPLIER: float = 1.5

# Multimodal output - image/media MIME mapping and size limits

# Supported image extensions to MIME types for multimodal tool output
_IMAGE_EXTENSIONS: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}

# Max image file size we'll inline as base64 (~10 MB)
_MAX_IMAGE_INLINE_BYTES: int = 10 * 1024 * 1024

# Regex to detect image file paths in tool output text
_IMAGE_PATH_RE = re.compile(
    r"""(?:^|[\s"'=])(/[^\s"']+\.(?:png|jpe?g|gif|webp|svg))""",
    re.IGNORECASE,
)


def _mime_for_image(path: str) -> str | None:
    """Return MIME type if *path* is a supported image extension, else None."""
    ext = Path(path).suffix.lower()
    return _IMAGE_EXTENSIONS.get(ext)


def _build_multimodal_output(
    text: str,
    image_base64: str | None = None,
    image_mime_type: str = "image/jpeg",
) -> str:
    """Embed image data inline when present, otherwise return plain text.

    For SVG, appends the SVG source as text.
    For raster images, appends a base64 data-URI reference so the LLM
    can receive the image via the OpenAI content_parts format if the
    caller decides to post-process it.
    """
    if not image_base64:
        return text

    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception:
        log.warning("multimodal_decode_failed", mime=image_mime_type)
        return text

    if image_mime_type == "image/svg+xml":
        svg_text = image_bytes.decode("utf-8", errors="replace")
        return f"{text}\n\n[SVG image]\n{svg_text}"

    return f"{text}\n\n[image: data:{image_mime_type};base64,{image_base64[:80]}... ({len(image_bytes)} bytes)]"


def _extract_image_from_file(file_path: str) -> tuple[str, str] | None:
    """Read an image file and return ``(base64_data, mime_type)`` or None.

    Only reads files that exist, are within size limits, and have a
    supported image extension.
    """
    mime = _mime_for_image(file_path)
    if mime is None:
        return None
    try:
        size = os.path.getsize(file_path)
    except OSError:
        return None
    if size > _MAX_IMAGE_INLINE_BYTES or size == 0:
        return None
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("ascii"), mime
    except OSError:
        return None


# StallState - unified type lives in loop.py (#15)
# To avoid circular imports (loop.py imports from tool_adapter.py), we use
# a Protocol for duck-typed stall state. The canonical dataclass is
# ``rune.agent.loop.StallState``.

@runtime_checkable
class StallStateProtocol(Protocol):
    """Duck-typed stall state interface used by the tool adapter."""
    bash_stalled: bool
    bash_stalled_reason: str
    bash_stalled_intent: str
    file_read_exhausted: bool
    intent_repeat_count: int
    error_signature_counts: dict[str, int]
    web_fetch_count: int
    web_search_count: int
    browser_no_match_count: int
    web_fetch_urls: dict[str, int]
    recent_tool_calls: list[str]
    cycle_detected: bool

    def record_error(self, signature: str) -> None: ...


@dataclass(slots=True)
class StallState:
    """Concrete stall state for use outside the agent loop.

    The canonical stall state lives in ``rune.agent.loop.StallState`` with
    additional fields.  This lightweight version satisfies
    :class:`StallStateProtocol` and is safe to instantiate in tests and the
    tool adapter when the full loop state is not available.
    """
    bash_stalled: bool = False
    bash_stalled_reason: str = ""
    bash_stalled_intent: str = ""
    file_read_exhausted: bool = False
    intent_repeat_count: int = 0
    error_signature_counts: dict[str, int] = field(default_factory=dict)
    web_fetch_count: int = 0
    web_search_count: int = 0
    browser_no_match_count: int = 0
    web_fetch_urls: dict[str, int] = field(default_factory=dict)
    recent_tool_calls: list[str] = field(default_factory=list)
    cycle_detected: bool = False

    def record_error(self, signature: str) -> None:
        self.error_signature_counts[signature] = self.error_signature_counts.get(signature, 0) + 1


# Effective limits with optional extension

def get_effective_stall_limits(extended: bool = False) -> dict[str, Any]:
    """Return stall limits, optionally scaled by EXTENDED_MULTIPLIER.

    Numeric values are multiplied and rounded up; nested dicts are
    recursively scaled.
    """
    if not extended:
        return dict(STALL_LIMITS)

    def _scale(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _scale(v) for k, v in value.items()}
        if isinstance(value, (int, float)):
            return math.ceil(value * EXTENDED_MULTIPLIER)
        return value

    return {k: _scale(v) for k, v in STALL_LIMITS.items()}


# Tool wrapper options

@dataclass(slots=True)
class ToolAdapterOptions:
    """Options passed to :func:`build_tool_set`."""
    profile_name: str = ""
    enable_guardian: bool = True
    sandbox_policy: str = "balanced"
    cognitive_cache: SessionToolCache | None = None
    stall_state: Any = None  # StallState (loop.py) or StallStateProtocol
    step_counter: Callable[[], int] | None = None
    allowed_tools: list[str] | None = None
    on_tool_start: Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]] | None = None
    on_tool_end: Callable[[str, CapabilityResult], Coroutine[Any, Any, None]] | None = None
    approval_callback: Callable[[str, str], Awaitable[bool]] | None = None  # (capability, reason) -> approved
    budget_percent: float = 0.0  # current budget consumption ratio (0.0-1.0)


# File-mutating capability names

_FILE_MUTATING_CAPABILITIES = frozenset({
    "file_write", "file_edit", "file_delete",
})

_BASH_CAPABILITY = "bash_execute"

# Default pattern for detecting MCP write operations
_MCP_WRITE_PATTERN = re.compile(
    r"create|update|delete|remove|send|post|put|patch|write|insert|modify|edit|add|move|archive",
    re.IGNORECASE,
)

# Smart file expansion threshold (lines)
_SMART_EXPAND_MAX_LINES = 500

# Maximum tool output size (~30KB ≈ 7500 tokens)
MAX_TOOL_OUTPUT_CHARS = 30_000


def is_mcp_write_operation(cap_name: str) -> bool:
    """Detect whether an MCP capability name represents a write operation.

    Write operations include create, update, delete, write, send, post, put,
    patch, and similar verbs found in the tool name portion of the capability.
    """
    if not cap_name.startswith("mcp."):
        return False
    parts = cap_name.split(".")
    if len(parts) < 3:
        return False
    tool_name = ".".join(parts[2:])
    return bool(_MCP_WRITE_PATTERN.search(tool_name))


# build_tool_set

def build_tool_set(
    options: ToolAdapterOptions | None = None,
    registry: CapabilityRegistry | None = None,
) -> dict[str, Any]:
    """Build a dict of ToolWrapper objects from the capability registry.

    Each tool wrapper:
    1. Checks the cognitive cache for a hit.
    2. Validates via Guardian if enabled.
    3. Executes the capability.
    4. Stores the result in the cognitive cache.
    5. Invalidates cache entries on file mutations.
    6. Tracks stall state.

    Returns ``ToolWrapper`` objects with proper JSON schemas derived
    from each capability's ``parameters_model`` (Pydantic BaseModel).
    """

    opts = options or ToolAdapterOptions()
    reg = registry or get_capability_registry()
    cache = opts.cognitive_cache

    # Use provided stall state or create one from loop.StallState (#15)
    stall = opts.stall_state
    if stall is None:
        try:
            from rune.agent.loop import StallState as _LoopStallState
            stall = _LoopStallState()
        except ImportError:
            # Minimal fallback if loop.py cannot be imported
            from dataclasses import dataclass as _dc
            from dataclasses import field as _f
            @_dc
            class _FallbackStall:
                bash_stalled: bool = False
                bash_stalled_reason: str = ""
                bash_stalled_intent: str = ""
                file_read_exhausted: bool = False
                intent_repeat_count: int = 0
                error_signature_counts: dict[str, int] = _f(default_factory=dict)
                web_fetch_count: int = 0
                web_search_count: int = 0
                browser_no_match_count: int = 0
                def record_error(self, sig: str) -> None:
                    self.error_signature_counts[sig] = self.error_signature_counts.get(sig, 0) + 1
            stall = _FallbackStall()

    tools: dict[str, Any] = {}

    for cap in reg.list_all():
        if opts.allowed_tools is not None and cap.name not in opts.allowed_tools:
            continue
        if not reg.is_allowed(cap.name):
            continue

        # Build a typed wrapper using the capability's parameters_model
        tool_obj = _build_typed_tool(
            cap_def=cap,
            opts=opts,
            reg=reg,
            cache=cache,
            stall=stall,
        )
        tools[cap.name] = tool_obj

    return tools


def _build_typed_tool(
    *,
    cap_def: CapabilityDefinition,
    opts: ToolAdapterOptions,
    reg: CapabilityRegistry,
    cache: SessionToolCache | None,
    stall: Any,  # StallState or StallStateProtocol (duck-typed, #15)
) -> Any:
    """Create a ToolWrapper with proper parameter JSON schema.

    Returns a lightweight ``ToolWrapper`` that carries *name*, *description*,
    *json_schema*, and *function*, compatible with ``tools_to_openai_schema()``
    in ``litellm_adapter.py``.
    """
    cap_name = cap_def.name

    # Feature 1: Approval denial escalation - consecutive denial counter
    # Mutable container so the inner async closure can read/write it.
    _consecutive_denials: list[int] = [0]

    async def _execute(params: dict[str, Any]) -> str | Any:
        current_step = opts.step_counter() if opts.step_counter else 0

        if opts.on_tool_start is not None:
            await opts.on_tool_start(cap_name, params)

        # -- Feature 4: Smart file expansion ---------------------
        effective_params = dict(params)
        if (
            cap_name == "file_read"
            and (effective_params.get("offset") or effective_params.get("limit"))
            and cache is not None
        ):
            file_path = str(effective_params.get("file_path") or effective_params.get("path", ""))
            if file_path:
                file_info = cache.get_file_info(file_path)
                if file_info is not None:
                    total_lines = file_info.get("total_lines") or file_info.get("estimated_tokens", 0)
                    # Use char_count heuristic: ~80 chars per line
                    if "total_lines" not in file_info and file_info.get("char_count"):
                        total_lines = file_info["char_count"] // 80
                    if total_lines <= _SMART_EXPAND_MAX_LINES:
                        effective_params.pop("offset", None)
                        effective_params.pop("limit", None)
                        log.debug(
                            "smart_expand",
                            path=file_path,
                            total_lines=total_lines,
                        )

        # -- Feature 2: Budget-aware web.fetch maxLength scaling --
        if cap_name == "web_fetch" and opts.budget_percent > 0.4:
            current_max = int(effective_params.get("maxLength") or effective_params.get("max_length") or 20000)
            bp = opts.budget_percent
            if bp >= 0.85:
                scale = 0.25
            elif bp >= 0.75:
                scale = 0.4
            elif bp >= 0.6:
                scale = 0.6
            else:
                scale = 1.0
            adaptive_max = max(3000, round(current_max * scale))
            if adaptive_max < current_max:
                effective_params["maxLength"] = adaptive_max
                effective_params["max_length"] = adaptive_max

        # 1. Cognitive cache check
        if cache is not None:
            cache_key = cache.generate_key(cap_name, effective_params)
            if cache_key is not None:
                hit = cache.get(cache_key, cap_name, effective_params)
                if hit is not None:
                    log.debug("cache_hit", capability=cap_name, key=cache_key)
                    return hit.output

        # 2. Guardian validation
        if opts.enable_guardian:
            guard_result = _validate_with_guardian(cap_name, effective_params)
            if guard_result.blocked:
                err = CapabilityResult(success=False, error=guard_result.reason)
                if opts.on_tool_end is not None:
                    await opts.on_tool_end(cap_name, err)
                return f"[BLOCKED] {guard_result.reason}"
            if guard_result.requires_approval:
                if opts.approval_callback is not None:
                    approved = await opts.approval_callback(cap_name, guard_result.reason)
                    if not approved:
                        # Feature 1: Denial escalation
                        _consecutive_denials[0] += 1
                        if _consecutive_denials[0] >= 2:
                            deny_hint = (
                                "\n\n[SYSTEM STOP] This command has been denied multiple times. "
                                "Do NOT retry similar commands. Either use ask_user to explain "
                                "why you need this command, or find a completely different "
                                "approach that does not require this operation."
                            )
                        else:
                            deny_hint = (
                                "\n\n[Recovery Guidance]\n"
                                "This command was denied by the user. Do NOT retry the same or similar command.\n"
                                "  1. Try a fundamentally different approach\n"
                                "  2. Use ask_user to explain why you need this operation\n"
                                "  3. If this is a prerequisite for the task, report the blocker to the user"
                            )
                        err = CapabilityResult(success=False, error=guard_result.reason)
                        if opts.on_tool_end is not None:
                            await opts.on_tool_end(cap_name, err)
                        return f"[DENIED] {guard_result.reason}{deny_hint}"
                    # Approved, reset denial counter
                    _consecutive_denials[0] = 0
                else:
                    err = CapabilityResult(success=False, error=guard_result.reason)
                    if opts.on_tool_end is not None:
                        await opts.on_tool_end(cap_name, err)
                    return f"[BLOCKED] Guardian requires approval: {guard_result.reason}"

        # -- Feature 3: MCP write operation approval guard -------
        if cap_name.startswith("mcp.") and opts.approval_callback is not None:
            if is_mcp_write_operation(cap_name):
                parts = cap_name.split(".")
                service_name = parts[1] if len(parts) > 1 else "unknown"
                tool_name = ".".join(parts[2:]) if len(parts) > 2 else cap_name
                params_preview = json.dumps(effective_params, indent=2)[:300]
                reason = f"External service write operation: {cap_name}\n{params_preview}"
                approved = await opts.approval_callback(
                    f"[{service_name}] {tool_name}",
                    reason,
                )
                if not approved:
                    err = CapabilityResult(
                        success=False,
                        error="User declined the service operation.",
                    )
                    if opts.on_tool_end is not None:
                        await opts.on_tool_end(cap_name, err)
                    return "[DENIED] User declined the service operation."

        # 2.5 Edit-loop circuit breaker — blocks at tool dispatch level
        if cap_name in ("file_edit", "file_write") and stall is not None:
            if hasattr(stall, "file_edit_counts"):
                _fp = effective_params.get("file_path") or effective_params.get("path", "")
                if _fp:
                    _count = stall.file_edit_counts.get(_fp, 0)
                    _limit = STALL_LIMITS.get("fileWrite", {}).get("sameFile", 3)
                    if _count >= _limit:
                        _block_msg = (
                            f"[BLOCKED] File '{_fp}' has been edited {_count} times. "
                            "You are stuck in an edit loop. STOP editing this file. "
                            "Step back and reconsider your entire approach."
                        )
                        err = CapabilityResult(success=False, error=_block_msg)
                        if opts.on_tool_end is not None:
                            await opts.on_tool_end(cap_name, err)
                        return _block_msg

        # 3. Execute
        start_time = time.monotonic()
        try:
            result = await reg.execute(cap_name, effective_params)
        except Exception as exc:
            result = CapabilityResult(success=False, error=f"Execution error: {exc}")
        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Feature 1: Reset denials on successful bash execution
        if cap_name == _BASH_CAPABILITY and result.success:
            _consecutive_denials[0] = 0

        # 4. Cache store
        if cache is not None and result.success:
            cache_key = cache.generate_key(cap_name, effective_params)
            if cache_key is not None:
                cache.set(cache_key, cap_name, effective_params, result, current_step)

        # 5. Cache invalidation
        if cache is not None:
            if cap_name in _FILE_MUTATING_CAPABILITIES and result.success:
                fp = effective_params.get("file_path") or effective_params.get("path", "")
                if fp:
                    cache.invalidate_file(fp)
            if cap_name == _BASH_CAPABILITY:
                cache.invalidate_from_bash(effective_params.get("command", ""), result.success)

        # 6. Stall tracking
        _update_stall_state(stall, cap_name, effective_params, result, elapsed_ms)

        if opts.on_tool_end is not None:
            await opts.on_tool_end(cap_name, result)

        # -- Feature 5: Wire output prefixes --------------------
        output = _format_tool_output(cap_name, effective_params, result)

        # -- Feature 6: Multimodal output (images/media) --------
        # Mirrors TS toModelOutput: when result.metadata contains
        # image_base64/image_mime_type (e.g. from browser screenshots),
        # return a ToolReturn with BinaryContent so the LLM sees the image.
        image_b64: str | None = None
        image_mime: str = "image/jpeg"

        if result.metadata and isinstance(result.metadata, dict):
            # Source 1: capability explicitly provides base64 image data
            image_b64 = result.metadata.get("image_base64") or result.metadata.get("imageBase64")
            image_mime = (
                result.metadata.get("image_mime_type")
                or result.metadata.get("imageMimeType")
                or "image/jpeg"
            )

        # Source 2: detect image file paths in the output text and inline them
        if not image_b64 and result.success:
            matches = _IMAGE_PATH_RE.findall(output)
            for match_path in matches:
                extracted = _extract_image_from_file(match_path)
                if extracted is not None:
                    image_b64, image_mime = extracted
                    break  # Only inline the first detected image

        return _build_multimodal_output(output, image_b64, image_mime)

    # --- Wrapper that receives **kwargs from LiteLLMAgent's tool executor ---
    async def _wrapper(**kwargs: Any) -> str | Any:
        return await _execute(kwargs)

    _wrapper.__name__ = cap_name
    _wrapper.__doc__ = cap_def.description

    param_model = cap_def.parameters_model
    schema: dict[str, Any] = {"type": "object", "properties": {}}

    if param_model is not None:
        try:
            json_schema = param_model.model_json_schema()
            schema = {
                "type": "object",
                "properties": json_schema.get("properties", {}),
            }
            if "required" in json_schema:
                schema["required"] = json_schema["required"]
            # Add field descriptions from model_fields
            for fname, finfo in param_model.model_fields.items():
                if fname in schema["properties"]:
                    if finfo.description and "description" not in schema["properties"][fname]:
                        schema["properties"][fname]["description"] = finfo.description
        except Exception as exc:
            log.debug("tool_schema_fallback", tool=cap_name, error=str(exc)[:100])
    elif cap_def.raw_json_schema is not None:
        # MCP tools provide raw JSON schema without Pydantic model
        raw = cap_def.raw_json_schema
        schema = {
            "type": "object",
            "properties": raw.get("properties", {}),
        }
        if "required" in raw:
            schema["required"] = raw["required"]

    return ToolWrapper(
        name=cap_name,
        description=cap_def.description,
        json_schema=schema,
        function=_wrapper,
    )


# Output formatting helper (Feature 5)

def _format_tool_output(
    cap_name: str,
    params: dict[str, Any],
    result: CapabilityResult,
) -> str:
    """Format a capability result with appropriate output prefixes.

    Mirrors ``formatToolResult`` in tool-adapter.ts. Prepends structured
    prefixes so the loop summary can deterministically parse tool outputs.
    """
    parts: list[str] = []

    if result.success:
        # file_read: path prefix + content + end anchor
        if cap_name == "file_read":
            path = str(params.get("file_path") or params.get("path", ""))
            if path.strip():
                parts.append(f"{FILE_READ_PATH_PREFIX}{path}")
        # bash: structured cmd/exit prefix
        if cap_name == "bash_execute":
            command = params.get("command", "")
            if isinstance(command, str):
                parts.append(f"{BASH_CMD_PREFIX}{command}{BASH_EXIT_PREFIX}0]")
        parts.append(result.output or "(success, no output)")
        # file_read: end anchor to prevent lost-in-the-middle
        if cap_name == "file_read":
            path = str(params.get("file_path") or params.get("path", ""))
            if path.strip():
                parts.append(f"[END: {path}]")
    else:
        # bash error: include cmd/exit prefix
        if cap_name == "bash_execute":
            command = params.get("command", "")
            exit_code = 1
            if result.metadata and isinstance(result.metadata, dict):
                exit_code = result.metadata.get("exitCode", 1)
            if isinstance(command, str):
                parts.append(f"{BASH_CMD_PREFIX}{command}{BASH_EXIT_PREFIX}{exit_code}]")
        parts.append(f"[ERROR] {result.error or 'Unknown error'}")
        if result.output:
            parts.append(result.output)

    if result.suggestions:
        parts.append("Suggestions:\n" + "\n".join(f"  - {s}" for s in result.suggestions))

    text = "\n".join(parts)

    # Output size cap: smart head+tail truncation (~30KB, ~7500 tokens)
    if len(text) > MAX_TOOL_OUTPUT_CHARS:
        head_size = int(MAX_TOOL_OUTPUT_CHARS * 0.7)
        tail_size = int(MAX_TOOL_OUTPUT_CHARS * 0.25)
        head = text[:head_size]
        tail = text[-tail_size:]
        total_kb = len(text) // 1024
        omitted_lines = text[head_size:len(text) - tail_size].count("\n")
        text = (
            f"{head}\n\n"
            f"... [{omitted_lines} lines omitted — showing first+last of {total_kb}KB total] ...\n\n"
            f"{tail}"
        )

    # Error enrichment: inject recovery hints for common error patterns
    if not result.success and result.error:
        text = enrich_error_message(cap_name, text, params)

    return text


# Error enrichment (ported from tool-adapter.ts:1911-2152)

def enrich_error_message(
    cap_name: str,
    error: str,
    params: dict[str, Any],
    ws_markers: list[str] | None = None,
) -> str:
    """도구 에러 메시지에 복구 가이드를 덧붙인다.

    긴 if/elif 체인을 패턴(조건)과 렌더링(힌트 생성)을 분리해
    유지보수성을 높인다.

    Ported from ``enrichErrorMessage`` in tool-adapter.ts.
    """

    error_lower = error.lower()

    def _append_common_file_discovery_hints(*, attempted_path: str, max_depth: int) -> list[str]:
        hints: list[str] = []
        hints.append("Recovery options:")
        hints.append("  1. Use file_list on the parent directory to check correct path")
        hints.append(f'  2. Use bash: find ~ -maxdepth {max_depth} -type d -name "<dirname>"')
        hints.append("  3. Use ask_user to confirm the absolute/correct path")
        if attempted_path:
            hints.append(f"  Attempted path: {attempted_path}")
        return hints

    def _project_type_hints(markers: set[str], *, prefix: str) -> list[str]:
        hints: list[str] = [prefix]
        if "go.mod" in markers:
            hints.append("  → Go project: use go test or go run with net/http")
        if "package.json" in markers:
            hints.append("  → Node project: use node -e with http module")
        if "pyproject.toml" in markers or "pytest.ini" in markers:
            hints.append("  → Python project: use python3 -c with urllib")
        if "Cargo.toml" in markers:
            hints.append("  → Rust project: use cargo test or std::net")
        return hints

    def _build_hints_project_map_no_sources() -> list[str]:
        attempted = str(params.get("path", ""))
        hints: list[str] = []
        hints.append("Recovery options:")
        hints.append("  1. The path may be wrong. Use file_list on the parent directory to verify")
        hints.append('  2. Use bash: find ~ -maxdepth 4 -type d -name "<dirname>" to locate the directory')
        hints.append("  3. Try with a different or absolute path")
        hints.append("  4. Use ask_user to confirm the correct project path")
        if attempted:
            hints.append(f"  Attempted path: {attempted}")
        return hints

    def _build_hints_help_intent_mismatch() -> list[str]:
        return [
            "Intent mismatch detected. Recovery options:",
            "  1. If you need help text, run only --help/-h command that exits immediately",
            "  2. If you need runtime validation, use explicit server start + readiness probe",
            "  3. Separate phases: help check -> start -> verify -> cleanup",
        ]

    def _build_hints_tool_env_broken() -> list[str]:
        markers = set(ws_markers or [])
        hints = _project_type_hints(
            markers,
            prefix="Probe tool environment issue (infra limitation). Use project-native tools:",
        )
        hints.append(
            '  This is an infrastructure limitation, NOT a task failure. Classify as "verification unavailable".'
        )
        return hints

    def _build_hints_command_not_found() -> list[str]:
        cmd = str(params.get("command", "")).split()[0] if params.get("command") else ""
        return [
            "Recovery options:",
            (f"  1. Check if installed: bash which {cmd}" if cmd else "  1. Check if the command is installed"),
            "  2. Try alternative command (e.g., python3 instead of python)",
            "  3. Use a built-in capability instead of bash",
        ]

    def _build_hints_file_edit_target_not_found() -> list[str]:
        return [
            "Recovery options:",
            "  1. Use file_read to see the current file content",
            "  2. The file may have changed — use the exact current text",
            "  3. Try a smaller, more unique match string",
        ]

    def _build_hints_directory_not_found() -> list[str]:
        attempted = str(params.get("path", ""))
        # slightly different maxdepth guidance than generic ENOENT branch
        return [
            "Recovery options:",
            "  1. Use file_list with parent path to find available directories",
            '  2. Use bash: find ~ -maxdepth 3 -type d -name "<dirname>"',
            "  3. Use ask_user to confirm the correct path",
            *( [f"  Attempted path: {attempted}"] if attempted else [] ),
        ]

    def _build_hints_generic_path_not_found() -> list[str]:
        file_path = str(params.get("path", "") or params.get("file_path", "") or params.get("directory", ""))
        hints = _append_common_file_discovery_hints(attempted_path=file_path, max_depth=3)
        # 원래 문구를 최대한 유지
        hints[1] = "  1. Use file_list on the parent directory to check correct path"
        hints[2] = '  2. Use bash: find ~ -maxdepth 3 -name "<dirname>" -type d'
        hints[3] = "  3. Use ask_user to confirm the absolute path"
        return hints

    def _build_hints_curl_tls_broken() -> list[str]:
        markers = set(ws_markers or [])
        hints: list[str] = [
            "curl environment failure (infra issue, not task failure). Use project-native HTTP probe:",
        ]
        if "go.mod" in markers:
            hints.append(
                "  → Go: bash go run -e 'package main; import (\"fmt\";\"net/http\"); "
                "func main() { r,e:=http.Get(\"http://127.0.0.1:PORT/healthz\"); "
                "if e!=nil{fmt.Println(e);return}; fmt.Println(r.StatusCode) }'"
            )
        if "package.json" in markers:
            hints.append(
                "  → Node: bash node -e \"require('http').get('http://127.0.0.1:PORT/healthz', "
                "r => { let d=''; r.on('data',c=>d+=c); r.on('end',()=>console.log(r.statusCode,d)) })\""
            )
        if "pyproject.toml" in markers or "pytest.ini" in markers:
            hints.append(
                '  → Python: bash python3 -c "import urllib.request; '
                "print(urllib.request.urlopen('http://127.0.0.1:PORT/healthz').read().decode())\""
            )
        if "Cargo.toml" in markers:
            hints.append("  → Rust: use reqwest in test or std::net::TcpStream for raw probe")
        if not markers:
            hints.append(
                "  → Detect project type first (go.mod/package.json/pyproject.toml) and use its native HTTP client"
            )
        hints.append(
            "  IMPORTANT: Do NOT retry curl or use wget. This is an infrastructure limitation, not a task failure."
        )
        return hints

    def _build_hints_eperm() -> list[str]:
        return [
            "[Sandbox EPERM] This operation was blocked by the host sandbox.",
            "This is an intentional security restriction — do NOT attempt to bypass it.",
            "Recovery options:",
            "  1. Work within the allowed workspace directory",
            "  2. Use file_write/file_edit/file_read for file operations within workspace",
            "  3. If the target path is outside the workspace, inform the user and suggest "
            "they run the command directly in their terminal",
        ]

    def _build_hints_permission_denied() -> list[str]:
        return [
            "Recovery options:",
            "  1. Check file permissions: bash ls -la <path>",
            "  2. Try a different path with write access",
        ]

    def _build_hints_timeout() -> list[str]:
        cmd = str(params.get("command", ""))
        is_long_running = bool(
            re.search(
                r"docker|docker-compose|npm\s+(install|ci|build)|yarn\s+(install|build)"
                r"|cargo\s+(build|test)|pip\s+install|apt\s+install|brew\s+install"
                r"|python|pytest|vitest|jest|node\s+\S|go\s+test|make\b",
                cmd,
                re.IGNORECASE,
            )
        )
        if is_long_running:
            return [
                "This command may need more time. Recovery options:",
                '  1. Retry with a longer timeout: { command: "...", timeout: 300000 }',
                "  2. Check if the service/daemon is running first",
                "  3. For docker: ensure Docker daemon is running (docker info)",
            ]
        return [
            "Command timed out. Recovery options:",
            "  1. Retry with a longer timeout parameter",
            "  2. Check if the target service is responding",
        ]

    def _build_hints_connection_refused() -> list[str]:
        return [
            "Connection refused/reset. Recovery options:",
            "  1. Check if the target service is running",
            "  2. Verify the port number is correct",
            '  3. For Docker: check "docker info" or "docker ps" first',
            "  4. If the service is not running, report the blocker to the user via ask_user",
        ]

    def _build_hints_mcp_auth_error() -> list[str]:
        service_name = cap_name.split(".")[1] if len(cap_name.split(".")) > 1 else "unknown"
        return [
            "[MANDATORY ACTION] This MCP service needs re-authentication.",
            f'  → Call service_connect with service name "{service_name}" to fix this automatically.',
            "  → Do NOT ask the user about configuration, key files, or MCP server setup.",
            "  → Do NOT try to restart MCP servers via bash commands.",
            "  → service_connect handles credentials, OAuth browser flow, and reconnection.",
        ]

    def _build_hints_api_auth_error() -> list[str]:
        return [
            "Recovery options:",
            "  1. Try an alternative approach that does not require this API",
            "  2. Use browser_navigate to access the resource directly",
        ]

    def _build_hints_build_error() -> list[str]:
        return [
            "Compilation/build error detected. Recovery options:",
            "  1. Read the dependency file (Cargo.toml, package.json, requirements.txt, go.mod) "
            "to check library versions",
            "  2. Use web_search to find the correct API usage for the CURRENT library version",
            "  3. The API may have changed between versions — do NOT guess, search official documentation",
            "  4. If type mismatch: the function signature may differ in this version — check docs",
            "  5. STOP repeating the same fix — try a fundamentally different approach",
        ]

    def _is_build_error() -> bool:
        return bool(
            cap_name == "bash_execute"
            and (
                re.search(r"error\[E\d+\]", error)
                or re.search(r"TS\d{4}:", error)
                or re.search(r"failed to compile|build failed|compilation failed", error, re.IGNORECASE)
                or re.search(r"cannot find (type|crate|module|package)", error, re.IGNORECASE)
                or re.search(r"mismatched types|type mismatch", error, re.IGNORECASE)
                or re.search(r"unresolved import|unresolved reference", error, re.IGNORECASE)
                or re.search(r"no method named|method not found", error, re.IGNORECASE)
            )
        )

    def _is_curl_tls_broken() -> bool:
        return bool(
            cap_name == "bash_execute"
            and "curl" in str(params.get("command", "")).lower()
            and "auto configuration failed" in error_lower
            and "openssl.cnf" in error_lower
        )

    conditions: dict[str, tuple[Callable[[], bool], Callable[[], list[str]]]] = {
        "project_map_no_source_files": (
            lambda: cap_name == "project_map" and "no source files found" in error_lower,
            _build_hints_project_map_no_sources,
        ),
        "bash_intent_mismatch": (
            lambda: cap_name == "bash_execute" and ("e_intent_mismatch" in error_lower or "intent mismatch" in error_lower),
            _build_hints_help_intent_mismatch,
        ),
        "bash_tool_env_broken": (
            lambda: cap_name == "bash_execute" and (
                "e_tool_env_broken" in error_lower or "e_runtime_probe_unavailable" in error_lower
            ),
            _build_hints_tool_env_broken,
        ),
        "command_not_found": (
            lambda: ("command not found" in error_lower or "exit code 127" in error_lower),
            _build_hints_command_not_found,
        ),
        "file_edit_target_not_found": (
            lambda: cap_name == "file_edit" and (
                "not found in file" in error_lower or "search string not found" in error_lower
            ),
            _build_hints_file_edit_target_not_found,
        ),
        "directory_not_found": (
            lambda: ("directory not found" in error_lower or "not a directory" in error_lower),
            _build_hints_directory_not_found,
        ),
        "mcp_auth": (
            lambda: cap_name.startswith("mcp.")
            and any(
                kw in error_lower
                for kw in ("token", "auth", "credential", "re-authenticate", "expired", "invalid", "unauthorized", "not found")
            ),
            _build_hints_mcp_auth_error,
        ),
        "generic_path_not_found": (
            lambda: ("enoent" in error_lower or "not found" in error_lower or "no such file" in error_lower),
            _build_hints_generic_path_not_found,
        ),
        "curl_tls_init_failure": (
            _is_curl_tls_broken,
            _build_hints_curl_tls_broken,
        ),
        "sandbox_eperm": (
            lambda: ("operation not permitted" in error_lower or "eperm" in error_lower),
            _build_hints_eperm,
        ),
        "permission_denied": (
            lambda: ("eacces" in error_lower or "permission denied" in error_lower),
            _build_hints_permission_denied,
        ),
        "timeout": (
            lambda: ("timeout" in error_lower or "etimedout" in error_lower),
            _build_hints_timeout,
        ),
        "connection_refused": (
            lambda: ("econnrefused" in error_lower or "econnreset" in error_lower),
            _build_hints_connection_refused,
        ),
        "api_auth": (
            lambda: any(kw in error_lower for kw in ("401", "403", "unauthorized", "api key")),
            _build_hints_api_auth_error,
        ),
        "build_error": (
            _is_build_error,
            _build_hints_build_error,
        ),
    }

    for _name, (matches, build_hints) in conditions.items():
        if matches():
            hints = build_hints()
            if hints:
                return f"{error}\n\n[Recovery Guidance]\n" + "\n".join(hints)
            return error

    return error


# Bash preflight probe (ported from tool-adapter.ts:1700-1806)

@dataclass(slots=True)
class BashPreflightSnapshot:
    """Cached snapshot of available bash tool environment."""

    curl_available: bool = False
    curl_healthy: bool = False
    wget_available: bool = False
    node_available: bool = False
    go_available: bool = False
    reason_code: str = ""
    timestamp: float = 0.0


# TTL for cached preflight results (seconds)
_BASH_PREFLIGHT_TTL_S = 300.0

_preflight_cache: BashPreflightSnapshot | None = None


async def resolve_bash_preflight(force: bool = False) -> BashPreflightSnapshot:
    """Probe the bash environment for tool availability.

    Checks: curl, wget, node, go.
    Caches the result for ``_BASH_PREFLIGHT_TTL_S`` seconds.
    """
    global _preflight_cache

    if not force and _preflight_cache is not None:
        if (time.monotonic() - _preflight_cache.timestamp) < _BASH_PREFLIGHT_TTL_S:
            return _preflight_cache

    snapshot = BashPreflightSnapshot(timestamp=time.monotonic())

    import shutil as _shutil

    def _check(cmd: str) -> bool:
        return _shutil.which(cmd) is not None

    curl_avail = _check("curl")
    wget_avail = _check("wget")
    node_avail = _check("node")
    go_avail = _check("go")

    snapshot.curl_available = curl_avail
    snapshot.wget_available = wget_avail
    snapshot.node_available = node_avail
    snapshot.go_available = go_avail

    # Health-check curl if available
    if curl_avail:
        try:
            proc = await asyncio.create_subprocess_shell(
                "curl -sS --max-time 2 http://127.0.0.1:9 2>&1 || true",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            output = stdout.decode(errors="replace").lower()
            if "auto configuration failed" in output and "openssl.cnf" in output:
                snapshot.curl_healthy = False
                snapshot.reason_code = "curl_tls_broken"
            else:
                snapshot.curl_healthy = True
        except Exception:
            snapshot.curl_healthy = False
            snapshot.reason_code = "curl_health_check_failed"

    _preflight_cache = snapshot
    return snapshot


# Guardian validation helper

@dataclass(slots=True)
class _GuardianResult:
    """Internal result from Guardian validation."""
    blocked: bool = False
    requires_approval: bool = False
    reason: str = ""


def _validate_with_guardian(cap_name: str, params: dict[str, Any]) -> _GuardianResult:
    """Validate a tool call with Guardian.

    Returns a :class:`_GuardianResult` indicating whether the call is
    blocked, requires approval, or is allowed (both fields False).
    """
    try:
        from rune.safety.guardian import get_guardian

        guardian = get_guardian()

        if cap_name == "bash_execute":
            command = params.get("command", "")
            result = guardian.validate(command)
            if not result.allowed:
                return _GuardianResult(blocked=True, reason=f"Guardian blocked bash: {result.reason}")
            if result.requires_approval:
                return _GuardianResult(requires_approval=True, reason=result.reason)

        elif cap_name in ("file_write", "file_edit", "file_delete"):
            file_path = params.get("file_path") or params.get("path", "")
            result = guardian.validate_file_path(file_path)
            if not result.allowed:
                return _GuardianResult(blocked=True, reason=f"Guardian blocked file write: {result.reason}")

        elif cap_name == "file_read":
            file_path = params.get("file_path") or params.get("path", "")
            result = guardian.validate_file_read_path(file_path)
            if not result.allowed:
                return _GuardianResult(blocked=True, reason=f"Guardian blocked file read: {result.reason}")

    except Exception as exc:
        log.error("guardian_validation_error", error=str(exc))
        # Fail closed: block if Guardian itself errors
        return _GuardianResult(blocked=True, reason=f"Guardian validation error (fail-closed): {exc}")

    return _GuardianResult()


# Stall tracking helper

def _detect_cycle_pattern(history: list[str], max_cycle_len: int = 4) -> int:
    """Detect repeating patterns of length 2-4 in recent history. Returns cycle length or 0."""
    if len(history) < 4:
        return 0
    for cycle_len in range(2, min(max_cycle_len + 1, len(history) // 2 + 1)):
        recent = history[-cycle_len:]
        prev = history[-2 * cycle_len:-cycle_len]
        if recent == prev:
            return cycle_len
    return 0


def _update_stall_state(
    stall: Any,
    cap_name: str,
    params: dict[str, Any],
    result: CapabilityResult,
    elapsed_ms: float,
) -> None:
    """Update stall state after a tool call.

    Uses duck typing (#15). Stall may be loop.StallState or any compatible object.
    """
    if cap_name == "bash_execute" and not result.success:
        # Track consecutive bash failures via the stall flag
        error_text = result.error or ""
        if "permission denied" in error_text.lower():
            stall.bash_stalled = True
            stall.bash_stalled_reason = "permission_denied"
            stall.bash_stalled_intent = params.get("command", "")[:100]
        elif "command not found" in error_text.lower():
            stall.bash_stalled = True
            stall.bash_stalled_reason = "command_not_found"
            stall.bash_stalled_intent = params.get("command", "")[:100]
        # Record error signature (#15)
        if hasattr(stall, "record_error") and error_text:
            stall.record_error(error_text[:80])

    elif cap_name == "bash_execute" and result.success:
        # Successful bash resets stall
        stall.bash_stalled = False
        stall.bash_stalled_reason = ""
        stall.bash_stalled_intent = ""

    if cap_name == "file_read" and not result.success:
        error_text = result.error or ""
        if "not found" in error_text.lower() or "no such file" in error_text.lower():
            stall.file_read_exhausted = True
    elif cap_name == "file_read" and result.success:
        stall.file_read_exhausted = False

    # Web stall counters (#15)
    if cap_name == "web_fetch":
        if hasattr(stall, "web_fetch_count"):
            stall.web_fetch_count += 1
        # Per-URL fetch tracking (H3)
        if hasattr(stall, "web_fetch_urls"):
            url = params.get("url", "")
            if url:
                stall.web_fetch_urls[url] = stall.web_fetch_urls.get(url, 0) + 1
                same_url_limit = STALL_LIMITS.get("web", {}).get("sameUrlRepeat", 2)
                if stall.web_fetch_urls[url] >= same_url_limit:
                    stall.stall_warning_issued = True if hasattr(stall, "stall_warning_issued") else None
    if cap_name == "web_search" and hasattr(stall, "web_search_count"):
        stall.web_search_count += 1

    # Browser find stall (#15)
    if cap_name == "browser_find" and not result.success:
        if hasattr(stall, "browser_no_match_count"):
            stall.browser_no_match_count += 1
    elif cap_name == "browser_find" and result.success:
        if hasattr(stall, "browser_no_match_count"):
            stall.browser_no_match_count = 0

    # File edit loop tracking (mirrors web_fetch_urls pattern)
    if cap_name in ("file_edit", "file_write") and hasattr(stall, "file_edit_counts"):
        fp = params.get("file_path") or params.get("path", "")
        if fp:
            stall.file_edit_counts[fp] = stall.file_edit_counts.get(fp, 0) + 1
            same_file_limit = STALL_LIMITS.get("fileWrite", {}).get("sameFile", 3)
            if stall.file_edit_counts[fp] >= same_file_limit:
                stall.stall_warning_issued = True

    # General error signature recording (#15)
    if not result.success and result.error and cap_name != "bash_execute":
        if hasattr(stall, "record_error"):
            stall.record_error(result.error[:80])

    # Cycle detection (H4): track recent tool calls and detect repeating patterns
    if hasattr(stall, "recent_tool_calls"):
        stall.recent_tool_calls.append(cap_name)
        # Keep last 30 entries
        if len(stall.recent_tool_calls) > 30:
            stall.recent_tool_calls[:] = stall.recent_tool_calls[-30:]
        cycle_len = _detect_cycle_pattern(stall.recent_tool_calls)
        if hasattr(stall, "cycle_detected"):
            stall.cycle_detected = cycle_len > 0


# Shell prefix helpers (ported from tool-adapter.ts:1505-1510)

def strip_shell_prefixes(command: str) -> str:
    """Strip leading env assignments and ``sudo`` from a command string."""
    normalized = command.strip()
    normalized = re.sub(r"^(\w+=\S+\s+)+", "", normalized)
    normalized = re.sub(r"^sudo\s+", "", normalized)
    return normalized


def is_help_only_bash_command(command: str) -> bool:
    """Return True if the command appears to be a help-only invocation."""
    normalized = strip_shell_prefixes(command)
    return bool(re.search(r"(^|\s)(--help|-h)(\s|$)", normalized))


def _has_help_markers(text: str) -> bool:
    """Return True if output text contains typical help/usage markers."""
    lower = text.lower()
    return bool(
        re.search(r"\busage:", lower)
        or re.search(r"\boptions:", lower)
        or re.search(r"\bflags:", lower)
        or re.search(r"\bshow this help\b", lower)
        or re.search(r"\bhelp\b", lower)
    )


def _has_service_startup_markers(text: str) -> bool:
    """Return True if output text contains service startup markers."""
    lower = text.lower()
    return bool(
        re.search(r"\blistening on\b", lower)
        or re.search(r"\bgateway listening\b", lower)
        or re.search(r"\badmin listening\b", lower)
        or re.search(r"\bserver started\b", lower)
        or re.search(r"\bready to accept\b", lower)
    )


# Managed service helpers (ported from tool-adapter.ts:1560-1679)

def build_managed_readiness_command(port: int, timeout: int = 30) -> str:
    """Generate a bash script that polls ``localhost:port`` until responsive.

    Tries ``nc -z`` first, then falls back to ``curl``.  Retries every
    second up to *timeout* seconds.
    """
    return (
        f"for i in $(seq 1 {timeout}); do "
        f"if command -v nc >/dev/null 2>&1 && nc -z 127.0.0.1 {port} >/dev/null 2>&1; then exit 0; fi; "
        f"if command -v curl >/dev/null 2>&1 && curl -sf --max-time 1 http://127.0.0.1:{port}/ >/dev/null 2>&1; then exit 0; fi; "
        f"sleep 1; "
        f"done; "
        f"exit 1"
    )


def build_managed_smoke_command(port: int, paths: list[str]) -> str:
    """Generate a bash script that tests each *path* via ``curl -sf``."""
    checks: list[str] = []
    for p in paths:
        path_suffix = p if p.startswith("/") else f"/{p}"
        url = f"http://127.0.0.1:{port}{path_suffix}"
        checks.append(
            f"if command -v curl >/dev/null 2>&1 && "
            f"curl -sf --max-time 2 {_shell_quote(url)} >/dev/null 2>&1; "
            f"then exit 0; fi"
        )
    checks.append("exit 1")
    return "; ".join(checks)


def build_managed_teardown_command(pid: int | None = None) -> str:
    """Generate a bash script to cleanly kill a process (group).

    If *pid* is provided, kills that specific PID (group).  Otherwise
    generates a generic no-op exit.
    """
    if pid is not None:
        return (
            f"kill -TERM -{pid} >/dev/null 2>&1 || "
            f"kill -TERM {pid} >/dev/null 2>&1 || true; "
            f"exit 0"
        )
    return "exit 0"


def should_auto_enable_managed_service_mode(command: str) -> bool:
    """Heuristic: returns True if *command* looks like a long-running service.

    Ported from tool-adapter.ts ``shouldAutoEnableManagedServiceMode``.
    """
    normalized = strip_shell_prefixes(command).lower()
    if not normalized or is_help_only_bash_command(normalized):
        return False
    # Compound commands are not single long-running services.
    if re.search(r"[;&|]\s*|&&|\|\|", normalized):
        return False

    if re.search(r"\bgo\s+run\b", normalized):
        return True
    if re.search(r"\bnpm\s+run\s+(dev|start|serve|preview)\b", normalized):
        return True
    if re.search(r"\bpnpm\s+(dev|start|serve|preview)\b", normalized):
        return True
    if re.search(r"\byarn\s+(dev|start|serve|preview)\b", normalized):
        return True
    if (re.search(r"\bdocker\s+compose\s+up\b", normalized) or re.search(
        r"\bdocker-compose\s+up\b", normalized
    )) and not re.search(r"\s(-d|--detach)(\s|$)", normalized):
        return True
    if re.search(r"\bpython\d*(\.\d+)?\s+-m\s+http\.server\b", normalized):
        return True
    if re.match(r"^(uvicorn|gunicorn)\b", normalized):
        return True
    if re.search(r"\bflask\s+run\b", normalized):
        return True
    return bool(re.search(r"\brails\s+server\b", normalized))


def _shell_quote(s: str) -> str:
    """Simple shell quoting. Wraps in single quotes."""
    return "'" + s.replace("'", "'\\''") + "'"


# Bash intent contract (ported from tool-adapter.ts:1838-1891)

def apply_bash_intent_contract(
    command: str,
    exit_code: int,
    output: str,
    duration_ms: float,
    *,
    stderr: str = "",
    success: bool = True,
    suggestions: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Detect intent mismatch for help-intent bash commands.

    If the command was a help invocation (``--help``, ``-h``) but the
    output looks like a service startup (e.g. "listening on ...") rather
    than help text, return a dict with:

    - ``error_code``: ``"E_INTENT_MISMATCH"``
    - ``message``: human-readable explanation
    - ``suggestions``: recovery guidance
    - ``metadata``: enriched metadata dict

    Returns ``None`` if no mismatch is detected.
    """
    if not success:
        return None
    if not is_help_only_bash_command(command):
        return None

    # Combine stdout and stderr for analysis.
    parts = [p for p in (output, stderr) if p and p.strip()]
    combined = "\n".join(parts).strip() or output

    if not _has_service_startup_markers(combined) or _has_help_markers(combined):
        return None

    base_suggestions = list(suggestions or [])
    extra = [
        "If intent is help text, run only a command that exits immediately",
        "If intent is runtime check, use explicit server start + readiness probe",
        "Split phases: help check -> start -> verify -> cleanup",
    ]
    merged = list(dict.fromkeys(base_suggestions + extra))  # dedupe, order-preserving

    return {
        "error_code": "E_INTENT_MISMATCH",
        "message": (
            "[E_INTENT_MISMATCH] Help-intent command appears to have "
            "started a long-running service"
        ),
        "suggestions": merged,
        "metadata": {
            **(metadata or {}),
            "reasonCode": "E_INTENT_MISMATCH",
            "intent": "help_only",
            "observed": "service_startup",
        },
    }
