"""Agent loop - the core execution engine for RUNE.

Ported from src/agent/loop.ts (7000+ LOC) - PydanticAI-based multi-step
tool calling with streaming, stall detection, token budgeting, and
observation masking.

This is the most complex module in the system. It implements:
- Multi-step tool calling via PydanticAI agent.run()
- Streaming text deltas via run_stream()
- Stall detection (consecutive + cumulative)
- Token budget 4-phase management
- 4-phase context rollover
- Active tools reduction after step 6
- Cognitive caching
- Observation masking
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from rune.agent.advisor import AdvisorService
from rune.agent.advisor.compliance import (
    PendingAdvice,
    build_pending,
    check_compliance,
)
from rune.agent.advisor.loop_integration import (
    build_advisor_request,
    build_policy_input,
    maybe_consult,
)
from rune.agent.checkpoint import CheckpointData, CheckpointManager
from rune.agent.cognitive_cache import SessionToolCache
from rune.agent.completion_gate import (
    CompletionGateInput,
    ExecutionEvidenceSnapshot,
    evaluate_completion_gate,
)
from rune.agent.failover import FailoverManager, classify_error
from rune.agent.goal_classifier import ClassificationResult, classify_goal
from rune.agent.intent_engine import resolve_intent_contract
from rune.agent.litellm_adapter import LiteLLMAgent, UsageLimits
from rune.agent.output_integrity import (
    build_nudge,
    fabricated_citations,
    output_integrity_enabled,
)
from rune.agent.prompts import build_system_prompt
from rune.agent.requirement_gate import RequirementGate, requirement_gate_enabled
from rune.agent.tool_adapter import STALL_LIMITS, ToolAdapterOptions, build_tool_set
from rune.config.defaults import (
    ACTIVE_TOOLS_REDUCTION_STEP,
    COGNITIVE_CACHE_MAX,
    DEFAULT_AGENT_TIMEOUT,
    DEFAULT_MAX_ITERATIONS,
    FULL_WINDOW_MAX,
    ROLLOVER_THRESHOLDS,
    TOKEN_BUDGET_PHASES,
    TRUNCATE_WINDOW_MAX,
)
from rune.types import (
    AgentConfig,
    AgentStatus,
    CapabilityResult,
    CompletionTrace,
)
from rune.utils.env import env_flag as _env_flag
from rune.utils.env import env_int as _env_int
from rune.utils.events import EventEmitter
from rune.utils.logger import get_logger

_HAS_PYDANTIC_AI = True  # Always True - LiteLLMAgent replaces PydanticAI

log = get_logger(__name__)

T = TypeVar("T")


def _auto_skill_enabled() -> bool:
    """Whether skill-reuse is on (distil on verified completion, inject a
    matching skill). Env ``RUNE_AUTO_SKILL`` overrides config ``skills.auto_skill``;
    default off. See SkillsConfig."""
    import os

    env = os.environ.get("RUNE_AUTO_SKILL", "").strip().lower()
    if env in ("1", "true", "on", "yes"):
        return True
    if env in ("0", "false", "off", "no"):
        return False
    try:
        from rune.config import get_config

        return bool(getattr(get_config().skills, "auto_skill", False))
    except Exception:
        return False


class _FastSkillRefiner:
    """Distil a skill body via the FAST tier — guidance, not correctness, so the
    cheapest tier fits; failures fall back to the step transcript."""

    async def refine(self, prompt: str, max_tokens: int = 600) -> str:
        from rune.llm.client import get_llm_client
        from rune.types import ModelTier

        resp = await get_llm_client().completion(
            messages=[{"role": "user", "content": prompt}],
            tier=ModelTier.FAST,
            max_tokens=max_tokens,
        )
        if isinstance(resp, dict):
            choices = resp.get("choices") or []
            if choices:
                return choices[0].get("message", {}).get("content", "") or ""
            return ""
        try:
            return resp.choices[0].message.content or ""
        except (AttributeError, IndexError):
            return ""


# Turn-atomic grouping — shared implementation in message_utils
from rune.agent.message_utils import (
    group_into_turns as _group_into_turns,
)
from rune.agent.message_utils import (
    msg_role as _msg_role,
)
from rune.agent.message_utils import (
    validate_tool_pairs as _validate_tool_pairs,
)

# Token budget scaling by intent (#24)

_BUDGET_BY_INTENT: dict[str, int] = {
    "chat": 50_000,
    "quick_fix": 100_000,
    "code_modify": 200_000,
    "research": 300_000,
    "deep_research": 500_000,
    "complex_coding": 1_000_000,
    "multi_task": 800_000,
}

_MAX_OUTPUT_TOKENS_BY_INTENT: dict[str, int] = {
    "chat": 4_096,
    "quick_fix": 4_096,
    "code_modify": 8_192,
    "research": 8_192,
    "deep_research": 16_384,
    "complex_coding": 8_192,
    "multi_task": 8_192,
}

_FAILED_TOOL_OUTPUT_TAIL_BYTES = 4_000
_FAILED_TOOL_NUDGE_TAIL_BYTES = 1_200


def _tail_text(text: str, max_bytes: int) -> str:
    data = text.encode("utf-8", errors="replace")
    if len(data) <= max_bytes:
        return text
    return data[-max_bytes:].decode("utf-8", errors="ignore")


def _tool_result_event_payload(cap_name: str, result: CapabilityResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": cap_name,
        "success": result.success,
        "output_length": len(result.output or ""),
        # Clipped output so UI surfaces can show what a command/tool actually
        # returned (the workbench activity feed); full output stays loop-side.
        "output_head": (result.output or "")[:2000],
    }
    if result.error:
        payload["error_length"] = len(result.error)
        payload["error_head"] = result.error[:500]

    if not result.success and _env_flag("RUNE_BENCH_CAPTURE_FAILED_TOOL_OUTPUT"):
        tail_bytes = _env_int(
            "RUNE_BENCH_FAILED_TOOL_OUTPUT_TAIL_BYTES",
            _FAILED_TOOL_OUTPUT_TAIL_BYTES,
        )
        if result.output:
            payload["output_tail"] = _tail_text(result.output, tail_bytes)
            payload["output_tail_truncated"] = (
                len(result.output.encode("utf-8", errors="replace")) > tail_bytes
            )
        if result.error:
            payload["error_tail"] = _tail_text(result.error, min(tail_bytes, 2_000))
            payload["error_tail_truncated"] = len(
                result.error.encode("utf-8", errors="replace")
            ) > min(tail_bytes, 2_000)
    return payload


def _failed_tool_nudge(cap_name: str, result: CapabilityResult) -> str:
    parts: list[str] = []
    if result.error:
        parts.append(f"error: {_tail_text(result.error, _FAILED_TOOL_NUDGE_TAIL_BYTES)}")
    if result.output:
        parts.append(f"output tail: {_tail_text(result.output, _FAILED_TOOL_NUDGE_TAIL_BYTES)}")
    if not parts:
        parts.append("no output captured")
    return (
        "[SYSTEM] Most recent failed tool result for "
        f"{cap_name}. Use this concrete failure before retrying or finalizing:\n" + "\n".join(parts)
    )


def _should_clear_failed_tool_nudge(cap_name: str, result: CapabilityResult) -> bool:
    return result.success and cap_name == "bash_execute"


def _effective_max_steps(budget: int) -> int:
    """Scale max steps based on token budget size."""
    if budget >= 800_000:
        return 200
    if budget >= 500_000:
        return 100
    if budget >= 200_000:
        return 50
    if budget >= 100_000:
        return 30
    return 20


def _compute_tool_rounds(
    classification: Any,
    executor_model: str,
    advisor_enabled: bool,
    fast_lane: bool = False,
) -> int:
    """Tool-round budget by task complexity and executor tier."""
    if fast_lane:
        from rune.agent.fast_lane import FAST_LANE_TOOL_ROUNDS

        return FAST_LANE_TOOL_ROUNDS

    from rune.agent.advisor.tiers import extract_provider_and_model, resolve_tier

    provider, model_name = extract_provider_and_model(executor_model)
    tier = resolve_tier(provider, model_name)

    is_complex = getattr(classification, "is_complex_coding", False)
    # Per-turn tool-round budget. Token budget and max_iterations bound runaway.
    base = 24 if is_complex else 12

    # Modest tier bonus. Larger bonuses burn tokens on weak models
    # that keep retrying the same failing tool.
    if tier < 50:
        tier_bonus = 2
    elif tier < 70:
        tier_bonus = 1
    else:
        tier_bonus = 0

    advisor_discount = -2 if advisor_enabled else 0
    return max(base + tier_bonus + advisor_discount, base)


# Phase-adaptive observation windows (#12)

_PHASE_WINDOWS: dict[str, tuple[int, int]] = {
    # (full_window, truncate_window)
    "exploration": (2, 4),
    "implementation": (4, 8),
    "verification": (3, 6),
    "research": (6, 10),
}


# Vision Cache (#28)


class VisionCache:
    """Cache image summaries to avoid re-processing identical images."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}  # hash -> summary

    def get_or_none(self, image_hash: str) -> str | None:
        return self._cache.get(image_hash)

    def store(self, image_hash: str, summary: str) -> None:
        self._cache[image_hash] = summary

    @staticmethod
    def hash_image(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()[:16]


# Wind-down state type (#14)

WindDownPhase = Literal["none", "wrapping", "stopping", "final", "hard_stop"]


# Stall State (ported from TS StallState) - unified, single source (#15)


@dataclass(slots=True)
class StallState:
    """Unified stall tracker - combines loop-level + tool-level tracking.

    This is the single source of truth; tool_adapter.py imports from here.
    """

    # Loop-level (consecutive/cumulative progress)
    consecutive_no_progress: int = 0
    cumulative_no_progress: int = 0
    last_activity_time: float = field(default_factory=time.monotonic)
    last_tool_call: str = ""
    stall_warning_issued: bool = False
    # Tool-level (written by tool_adapter._update_stall_state via duck typing)
    bash_stalled: bool = False
    bash_stalled_reason: str = ""
    bash_stalled_intent: str = ""
    file_read_exhausted: bool = False
    # Extended stall fields (#15)
    intent_repeat_count: int = 0
    error_signature_counts: dict[str, int] = field(default_factory=dict)
    web_fetch_count: int = 0
    web_search_count: int = 0
    browser_no_match_count: int = 0
    web_fetch_urls: dict[str, int] = field(default_factory=dict)
    recent_tool_calls: list[str] = field(default_factory=list)
    cycle_detected: bool = False
    file_edit_counts: dict[str, int] = field(default_factory=dict)

    def mark_activity(self, tool_name: str = "") -> None:
        self.consecutive_no_progress = 0
        self.last_activity_time = time.monotonic()
        if tool_name == self.last_tool_call:
            self.intent_repeat_count += 1
        else:
            self.intent_repeat_count = 0
        self.last_tool_call = tool_name

    def mark_no_progress(self) -> None:
        self.consecutive_no_progress += 1
        self.cumulative_no_progress += 1

    def record_error(self, signature: str) -> None:
        """Track a unique error signature."""
        self.error_signature_counts[signature] = self.error_signature_counts.get(signature, 0) + 1

    @property
    def is_stalled(self) -> bool:
        return (
            self.consecutive_no_progress >= 3
            or self.cumulative_no_progress >= 8
            or self.bash_stalled
            or self.file_read_exhausted
            or self.intent_repeat_count >= 8
        )

    @property
    def time_since_activity(self) -> float:
        return time.monotonic() - self.last_activity_time


# Token Budget Manager


@dataclass(slots=True)
class TokenBudget:
    total: int = 500_000
    used: int = 0

    @property
    def fraction(self) -> float:
        return self.used / self.total if self.total > 0 else 0.0

    @property
    def phase(self) -> int:
        """Current budget phase (1-4) based on usage fraction."""
        f = self.fraction
        if f >= TOKEN_BUDGET_PHASES["phase_4"]:
            return 4
        if f >= TOKEN_BUDGET_PHASES["phase_3"]:
            return 3
        if f >= TOKEN_BUDGET_PHASES["phase_2"]:
            return 2
        return 1

    @property
    def needs_rollover(self) -> bool:
        f = self.fraction
        return f >= ROLLOVER_THRESHOLDS["phase_1"]

    @property
    def rollover_phase(self) -> int:
        f = self.fraction
        if f >= ROLLOVER_THRESHOLDS["phase_4"]:
            return 4
        if f >= ROLLOVER_THRESHOLDS["phase_3"]:
            return 3
        if f >= ROLLOVER_THRESHOLDS["phase_2"]:
            return 2
        if f >= ROLLOVER_THRESHOLDS["phase_1"]:
            return 1
        return 0


# Agent Loop


class NativeAgentLoop(EventEmitter):
    """Core agent execution loop using PydanticAI.

    Replaces the TS NativeAgentLoop that used AI SDK's generateText().
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__()
        self._config = config or AgentConfig()
        self._status = AgentStatus.IDLE
        self._step = 0
        self._stall = StallState()
        self._token_budget = TokenBudget()
        self._running = False
        self._cancel_event = asyncio.Event()
        # Wind-down 5-stage state machine (#14)
        self._wind_down_phase: WindDownPhase = "none"
        self._wind_down_write_forced: bool = False
        # Step watchdog (#29)
        self._step_start_time: float = 0.0
        self._STEP_WARN_MS: int = 45_000
        self._STEP_ABORT_MS: int = 120_000
        # Streaming activity (#30)
        self._last_activity: float = 0.0
        # Vision cache (#28)
        self._vision_cache = VisionCache()
        # Execution nudges (#27) - counters
        self._consecutive_reads_without_write: int = 0
        self._pending_verification_nudge: bool = False
        # Skill-reuse (auto-skill, default off): tool trace for this run,
        # captured only when the flag is on. Distilled into a skill on verified
        # completion.
        self._tool_trace: list[Any] = []
        self._auto_skill: bool = False
        # Gated Skill Learning (T1-1): name+match-score of the skill injected
        # into this run (if any), so its outcome can be logged at run end.
        self._injected_skill: tuple[str, float] | None = None
        # Replay-corpus capture (T1-1): pre-task git ref recorded at
        # run start when capture_replay is on, snapshotted at run end if a skill
        # is distilled. None when capture is off / cwd isn't a git repo.
        self._replay_capture: dict[str, str] | None = None
        # Completion gate enhancements (#16) - tracked files + hard failures
        self._files_written: set[str] = set()
        # Requirement-Adherence Gate (opt-in via RUNE_REQUIREMENT_GATE); None when off.
        self._requirement_gate_obj: RequirementGate | None = None
        self._last_answer_text: str = ""  # latest final-answer text, for the gate's artifact
        self._files_read: set[str] = set()
        self._hard_failure_signatures: set[str] = set()
        self._hard_failures: list[str] = []
        # R19: tool-call sequence counter (not step) to catch intra-step patterns
        self._last_code_write_step: int = 0
        self._last_verify_step: int = 0
        # Did the last test/verify command fail, and how many times we've held
        # back a finish because the code wasn't verified yet.
        self._last_verify_failed: bool = False
        self._unverified_completion_blocks: int = 0
        self._tool_call_seq: int = 0
        # Activity phase for adaptive observation windows
        self._activity_phase: str = "exploration"
        self._prev_activity_phase: str = "exploration"
        # Output token scaling by intent (#H5)
        self._max_output_tokens: int = 8_192
        # Per-run rollover bookkeeping
        self._rollover_phase_done: set[int] = set()
        self._session_id: str | None = None
        # Cross-step tool failure persistence (#P4) — streak only, not blocked groups.
        # Blocked groups are re-derived each step from the streak.
        self._persistent_fail_streak: dict[str, int] = {}
        self._recent_failed_tool_nudge: str = ""
        # Injected callbacks (set from CLI / controller)
        self._approval_callback: Callable[[str, str], Awaitable[bool]] | None = None
        self._ask_user_callback: Any = None  # AskUserCallback
        # Last classification result (for domain change detection across turns)
        self._last_goal_type: str = ""
        # Whether this run's task is verified by execution (tests/commands). When
        # true, execution verification owns correctness and the LLM requirement
        # gate is skipped (redundant, and risks false-blocking on a snapshot).
        self._requires_execution: bool = False
        # Whether this run produces substantive non-executable output that gets
        # the depth scaffold and depth critic.
        # Rehydration subsystem (initialized per run in _execute_loop)
        self._rehydration_recorder: Any = None
        self._rehydration_trigger: Any = None
        self._gate_blocked_count: int = 0
        self._output_integrity_fired: int = 0
        self._evidence_gate: Any = None

    @property
    def status(self) -> AgentStatus:
        return self._status

    @property
    def step(self) -> int:
        return self._step

    @property
    def tokens_used(self) -> int:
        return self._token_budget.used

    def set_approval_callback(self, cb: Callable[[str, str], Awaitable[bool]] | None) -> None:
        """Set the approval callback: (capability, reason) -> approved."""
        self._approval_callback = cb

    def set_ask_user_callback(self, cb: Any) -> None:
        """Set the ask_user callback (AskUserCallback type)."""
        self._ask_user_callback = cb

    def _reset_run_state(self) -> None:
        """Reset mutable per-run state before starting a fresh invocation."""
        self._step = 0
        self._stall = StallState()
        self._token_budget = TokenBudget()
        self._wind_down_phase = "none"
        self._wind_down_write_forced = False
        self._step_start_time = 0.0
        self._last_activity = 0.0
        self._consecutive_reads_without_write = 0
        self._pending_verification_nudge = False
        self._files_written.clear()
        self._files_read.clear()
        self._tool_trace = []
        self._auto_skill = _auto_skill_enabled()
        self._hard_failure_signatures.clear()
        self._hard_failures.clear()
        self._last_code_write_step = 0
        self._last_verify_step = 0
        self._last_verify_failed = False
        self._unverified_completion_blocks = 0
        self._tool_call_seq = 0
        self._activity_phase = "exploration"
        self._prev_activity_phase = "exploration"
        self._max_output_tokens = 8_192
        self._rollover_phase_done = set()
        self._session_id = None
        self._cognitive_cache = None
        # Reset cross-step failure state (#P4)
        self._persistent_fail_streak.clear()
        self._recent_failed_tool_nudge = ""
        # Rehydration
        self._rehydration_recorder = None
        self._rehydration_trigger = None
        self._gate_blocked_count = 0
        self._output_integrity_fired = 0
        # Evidence Gate (benchmark output-correctness verification; opt-in)
        self._evidence_gate = None

    @property
    def files_written(self) -> list[str]:
        """Files the agent actually wrote this run (from its file tools)."""
        return sorted(self._files_written)

    async def _benchmark_completion_blocker(self) -> str | None:
        # 1. Recent unresolved tool failure (behavioral evidence).
        if _env_flag("RUNE_BENCH_CAPTURE_FAILED_TOOL_OUTPUT") and self._recent_failed_tool_nudge:
            return (
                "[Completion Gate] A recent tool failed and no later successful "
                "command cleared it. Fix or explicitly verify the failure before "
                "finalizing.\n" + self._recent_failed_tool_nudge
            )
        # 2. Evidence Gate: re-verify the produced artifact against the task's
        #    own success criteria before allowing finalization (opt-in).
        _state, message = await self._evidence_verdict()
        if _state == "fail" and message:
            log.info("evidence_gate_block", step=self._step)
            return message
        return None

    async def _evidence_verdict(self) -> tuple[str, str | None]:
        """Return the Evidence Gate verdict: ``("pass"|"fail"|"skip", message)``.

        ``"skip"`` when the gate is disabled or inconclusive. Never raises —
        verification failure is treated as ``"skip"`` so it cannot turn a real
        success into a false block.
        """
        if self._evidence_gate is None:
            return "skip", None
        try:
            state, message = await self._evidence_gate.verdict()
            return str(state), message
        except Exception as exc:  # never let verification crash finalize
            log.warning("evidence_gate_check_error", error=str(exc)[:120])
            return "skip", None

    async def _auto_verify(self) -> tuple[str, str]:
        """Run the project's check after edits, preferring correctness over lint.

        Prefer the correctness test command (the repo's pytest/npm test) over the
        fast lint/typecheck, since a logic error passes lint but fails the tests.
        Falls back to the lint verifier when no test runner is evident. Returns
        ``("pass"|"fail"|"skip", evidence)``; ``"skip"`` when neither is detected.
        Never raises.
        """
        import os as _os

        from rune.agent.auto_verify import (
            detect_test_command,
            detect_verify_command,
            run_verify,
        )

        cwd = _os.getcwd()
        cmd = detect_test_command(cwd) or detect_verify_command(cwd)
        if not cmd:
            return "skip", ""
        return await run_verify(cmd, cwd)

    async def _auto_verify_gate(
        self, messages: list[Any], blocked_count: int
    ) -> tuple[bool, list[Any], int]:
        """Shared post-edit auto-verify gate for every code-producing completion
        path. Returns ``(ok, messages, blocked_count)``:

        - ``ok=True``: pass / skip / disabled — the caller may finalize.
        - ``ok=False``: the check failed; the problem is injected into
          *messages* and *blocked_count* is incremented. The caller should
          ``continue`` so the agent self-fixes, or finalize as
          ``max_gate_blocked`` when the returned count hits the cap.
        """
        if not _env_flag("RUNE_AUTO_VERIFY"):
            return True, messages, blocked_count
        state, msg = await self._auto_verify()
        if state == "fail" and msg:
            log.info("auto_verify_block", step=self._step)
            self._auto_verify_failure = msg
            messages = self._inject_system_message(
                messages,
                "[Auto-Verify] The project's check failed after your edits. "
                f"Fix this before finishing:\n{msg}",
            )
            blocked_count += 1
            self._gate_blocked_count = blocked_count
            return False, messages, blocked_count
        return True, messages, blocked_count

    def _gather_artifact(self) -> str:
        """Bounded snapshot of the files written plus the final answer text, for
        the requirement gate to check against the user's requirements. The
        per-source cap is generous: judging subject adherence on a report needs
        enough context to see what the document is dominantly about. A 2000-char
        cut can leave only an intro that name-drops the requested subject while
        the body is about something else, which reads as satisfied. The
        requirement gate re-caps the total (``_MAX_ARTIFACT_CHARS``)."""
        parts: list[str] = []
        for fp in self.files_written[:20]:
            try:
                with open(fp, encoding="utf-8", errors="replace") as fh:
                    parts.append(f"=== file: {fp} ===\n{fh.read()[:8000]}")
            except OSError:
                continue
        if self._last_answer_text.strip():
            parts.append(f"=== final answer ===\n{self._last_answer_text[:8000]}")
        return "\n\n".join(parts)

    def _gather_citation_text(self) -> str:
        """Full text of written files plus the final answer, untruncated. Citations
        in a report sit at the end (a ## Sources section), so the requirement
        gate's 2000-char snapshot would drop them. The deterministic URL scan is
        cheap, so the integrity gate reads full content."""
        parts: list[str] = []
        for fp in self.files_written[:20]:
            try:
                with open(fp, encoding="utf-8", errors="replace") as fh:
                    parts.append(fh.read())
            except OSError:
                continue
        if self._last_answer_text.strip():
            parts.append(self._last_answer_text)
        return "\n\n".join(parts)

    async def _requirement_gate(
        self, messages: list[Any], blocked_count: int
    ) -> tuple[bool, list[Any], int]:
        """Requirement-adherence gate (opt-in, default off). Same contract as
        :meth:`_auto_verify_gate`: checks the produced output against the user's
        requirements and injects the unmet list on a miss; pass / skip / disabled
        return ``ok=True``."""
        if self._requirement_gate_obj is None:
            return True, messages, blocked_count
        artifact = self._gather_artifact()
        state, msg = await self._requirement_gate_obj.verdict(artifact)
        if state == "fail" and msg:
            log.info("requirement_gate_block", step=self._step)
            messages = self._inject_system_message(messages, msg)
            blocked_count += 1
            self._gate_blocked_count = blocked_count
            return False, messages, blocked_count
        return True, messages, blocked_count

    def _output_integrity_gate(
        self, messages: list[Any], blocked_count: int
    ) -> tuple[bool, list[Any], int]:
        """Deterministic citation-integrity check (opt-in, model-free). Flags URLs
        cited in the output that were never retrieved. Skips when disabled or when
        retrieval cannot be determined."""
        if not output_integrity_enabled():
            return True, messages, blocked_count
        # Bounded: after a couple of blocks the model is not fixing the citation
        # (it may be a true hallucination it cannot ground, or a borderline case).
        # Pass through with a warning rather than consuming the whole gate-block
        # budget and failing the run while a usable artifact already exists.
        if self._output_integrity_fired >= 2:
            log.warning("output_integrity_budget_spent", step=self._step)
            return True, messages, blocked_count
        bad = fabricated_citations(self._gather_citation_text(), messages)
        if bad:
            self._output_integrity_fired += 1
            log.info("output_integrity_block", step=self._step, n=len(bad))
            messages = self._inject_system_message(messages, build_nudge(bad))
            blocked_count += 1
            self._gate_blocked_count = blocked_count
            return False, messages, blocked_count
        return True, messages, blocked_count

    def _maybe_force_wind_down_write(self, messages: list[Any]) -> list[Any]:
        """In the budget wind-down 'final' phase, inject a one-shot directive to
        write the best-effort artifact now. The final phase only trims tools;
        without this a task can keep reading/thinking until the budget hard-stops
        and produce nothing. No-op if not in the final phase or already injected."""
        if self._wind_down_phase != "final" or self._wind_down_write_forced:
            return messages
        self._wind_down_write_forced = True
        log.info("wind_down_force_write", step=self._step)
        return self._inject_system_message(
            messages,
            "The token budget is almost spent. Stop researching and write your "
            "best answer NOW with what you already have: if the task expects a "
            "file, call file_write with the complete content this turn; otherwise "
            "give the final answer. Do not promise to continue; produce the "
            "artifact now.",
        )

    def _max_gate_reason(self) -> str:
        """Reason to record when the gate-block budget is exhausted. For a
        non-executable task the blocking gates are the soft quality gates
        (requirement / depth / integrity) and a usable artifact already exists, so
        report a warning rather than a hard failure. Executable tasks keep
        failing: do not ship code whose tests/verification never passed."""
        return "max_gate_blocked" if self._requires_execution else "completed_gate_warnings"

    async def _finalize_gates(
        self, messages: list[Any], blocked_count: int
    ) -> tuple[bool, list[Any], int]:
        """Pre-finalize checks: deterministic output-integrity (model-agnostic),
        the requirement gate, then the depth critic. All opt-in; no-op when
        disabled.

        The requirement gate and depth critic (LLM-judged) are skipped for
        execution-verified tasks: running the code/tests is the authoritative
        check there, so an LLM judgment is redundant and risks false-blocking.
        Output-integrity is deterministic and harmless, so it always runs when
        enabled."""
        ok, messages, blocked_count = self._output_integrity_gate(messages, blocked_count)
        if not ok:
            return False, messages, blocked_count
        if self._requires_execution:
            log.info("requirement_gate_skip_execution_task")
            return True, messages, blocked_count
        return await self._requirement_gate(messages, blocked_count)

    async def run(
        self,
        goal: str,
        *,
        context: dict[str, Any] | None = None,
        max_steps: int | None = None,
        resume_session_id: str | None = None,
        message_history: list[dict[str, str]] | None = None,
        extra_system_context: str | None = None,
    ) -> CompletionTrace:
        """Execute the agent loop for a given goal.

        This is the main entry point. It:
        1. Classifies the goal
        2. Selects appropriate tool subset
        3. Runs PydanticAI agent with streaming
        4. Handles stall detection and token budgeting

        If *resume_session_id* is provided, attempts to restore from the
        latest checkpoint for that session (step counter, token usage, goal).

        If *message_history* is provided (list of {role, content} dicts),
        the conversation history is prepended to messages so the LLM has
        multi-turn context from prior turns.
        """
        if self._running:
            raise RuntimeError("Agent loop is already running")

        self._running = True
        self._cancel_event.clear()
        self._reset_run_state()
        self._replay_capture = None
        # T1-1: record the pre-task git ref now (before the agent
        # mutates the tree) so a distilled skill can later be A/B'd by replay.
        # Gated by skills.capture_replay; best-effort, never blocks the run.
        if self._auto_skill:
            try:
                from rune.config import get_config

                if getattr(get_config().skills, "capture_replay", False):
                    from rune.skills.capture import capture_head_ref

                    _cwd = os.getcwd()
                    _ref = capture_head_ref(_cwd)
                    if _ref:
                        self._replay_capture = {"goal": goal, "cwd": _cwd, "head": _ref}
            except Exception as exc:
                log.debug("replay_capture_start_failed", error=str(exc)[:120])
        max_iterations = max_steps or self._config.max_iterations
        self._requirement_gate_obj = RequirementGate(goal) if requirement_gate_enabled() else None

        # Checkpoint restoration
        if resume_session_id:
            try:
                ckpt_mgr = CheckpointManager()
                ckpt_data = ckpt_mgr.load(resume_session_id)
                if ckpt_data is not None:
                    self._step = ckpt_data.step
                    self._token_budget.used = ckpt_data.token_usage
                    self._session_id = resume_session_id
                    goal = ckpt_data.goal or goal
                    # Restore file tracking from checkpoint metadata
                    for entry in ckpt_data.tool_results:
                        if "changed_files" in entry:
                            self._files_written.update(entry["changed_files"])
                    log.info(
                        "checkpoint_restored",
                        session_id=resume_session_id,
                        step=self._step,
                        token_usage=self._token_budget.used,
                    )
            except Exception as exc:
                log.warning("checkpoint_restore_failed", error=str(exc)[:100])

        try:
            # Step 1: Classify goal
            self._status = AgentStatus.THINKING
            await self.emit("status_change", self._status)

            from rune.config.defaults import TOKEN_OPTIMIZATION_ENABLED

            # Extract previous goal info from message_history for domain change detection
            _prev_goal = ""
            _prev_goal_type = ""
            if message_history:
                # goal_type is stored on assistant turns (set via add_turn in cli/api/ui)
                for msg in reversed(message_history):
                    if msg.get("goal_type"):
                        _prev_goal_type = msg["goal_type"]
                        break
                for msg in reversed(message_history):
                    if msg.get("role") == "user" and msg.get("content"):
                        _prev_goal = msg["content"]
                        break
            classification = await classify_goal(
                goal,
                previous_goal=_prev_goal,
                previous_goal_type=_prev_goal_type,
            )

            # Evidence Gate: re-verify the artifact against the task's own
            # success criteria before finalizing (benchmark-only, opt-in). Built
            # once here so the LLM check extraction is cached for the run.
            from rune.agent.evidence_gate import EvidenceGate, evidence_gate_enabled

            if evidence_gate_enabled():
                import os as _os

                self._evidence_gate = EvidenceGate(goal, _os.getcwd())

            # Mark as continuation only when history exists AND domain hasn't changed
            if message_history:
                if classification.is_domain_change:
                    classification.is_continuation = False
                    log.info(
                        "domain_change_detected",
                        previous_type=_prev_goal_type,
                        current_type=classification.goal_type,
                    )
                else:
                    classification.is_continuation = True

            # Tier 2 LLM fallback for low-confidence regex results (#17)
            if classification.confidence < 0.7 and classification.tier == 1:
                llm_intent = await self._classify_intent_llm(goal)
                if llm_intent is not None:
                    # Map LLM intent names to GoalType
                    _intent_to_goal: dict[str, str] = {
                        "chat": "chat",
                        "quick_fix": "code_modify",
                        "code_modify": "code_modify",
                        "research": "research",
                        "deep_research": "research",
                    }
                    mapped = _intent_to_goal.get(llm_intent, "full")
                    classification = ClassificationResult(
                        goal_type=mapped,  # type: ignore[arg-type]
                        confidence=0.75,
                        tier=2,
                        reason=f"LLM intent fallback: {llm_intent}",
                        # Tier-2 LLM intent classifier doesn't surface
                        # email/document detection; default to including
                        # both prompt sections so multilingual users
                        # don't lose guidance on this fallback path.
                        intent_categories=frozenset({"email", "document"}),
                    )

            log.info(
                "goal_classified",
                type=classification.goal_type,
                confidence=classification.confidence,
                tier=classification.tier,
                is_domain_change=classification.is_domain_change,
            )
            await self.emit("goal_classified", classification)
            self._last_goal_type = classification.goal_type
            self._requires_execution = bool(getattr(classification, "requires_execution", False))

            # Token budget scaling by intent (#24)
            intent_key = classification.goal_type
            # Map GoalType to budget intent names
            _goal_to_budget: dict[str, str] = {
                "chat": "chat",
                "web": "research",
                "research": "research",
                "code_modify": "code_modify",
                "execution": "code_modify",
                "browser": "research",
                "full": "deep_research",
            }
            budget_intent = _goal_to_budget.get(intent_key, "research")
            self._token_budget.total = _BUDGET_BY_INTENT.get(budget_intent, 500_000)
            # Optional explicit override (used by /goal for heavy tasks).
            _budget_override = getattr(self._config, "token_budget_override", None)
            if _budget_override:
                self._token_budget.total = int(_budget_override)
            # Output token scaling by intent (#H5)
            self._max_output_tokens = _MAX_OUTPUT_TOKENS_BY_INTENT.get(budget_intent, 8_192)
            log.info(
                "token_budget_set",
                intent=budget_intent,
                budget=self._token_budget.total,
                max_output_tokens=self._max_output_tokens,
                token_opt=TOKEN_OPTIMIZATION_ENABLED,
            )

            # Scale max iterations based on budget (#6)
            budget_max = _effective_max_steps(self._token_budget.total)
            if max_iterations < budget_max and not max_steps:
                max_iterations = budget_max

            # Step 2: Select tools based on classification
            tools = self._select_tools(classification)

            # Step 3: Build system prompt + provider supplement
            system_prompt = self._build_system_prompt(goal, classification, context)
            # Caller-supplied supplement (used by paired skill replay to inject a
            # specific skill into the "with" arm deterministically). Neutral when
            # unset.
            if extra_system_context:
                system_prompt = f"{system_prompt}\n\n{extra_system_context}"
            try:
                from rune.agent.provider_capabilities import get_prompt_supplement

                _model_id = getattr(self._config, "model", "") or ""
                supplement = get_prompt_supplement(_model_id)
                if supplement:
                    system_prompt += supplement
            except Exception:
                pass

            # Step 4: Run agent loop
            self._status = AgentStatus.ACTING
            await self.emit("status_change", self._status)

            trace = await self._execute_loop(
                goal=goal,
                system_prompt=system_prompt,
                tools=tools,
                max_iterations=max_iterations,
                classification=classification,
                context=context,
                message_history=message_history,
            )

            self._status = AgentStatus.IDLE
            await self.emit("status_change", self._status)

            # A/B comparison logging
            log.info(
                "agent_run_complete",
                token_opt=TOKEN_OPTIMIZATION_ENABLED,
                goal_type=classification.goal_type,
                budget_intent=budget_intent,
                budget_total=self._token_budget.total,
                tokens_used=self._token_budget.used,
                steps=trace.final_step,
                reason=trace.reason,
                system_prompt_len=len(system_prompt),
                tools_count=len(tools),
            )
            if self._evidence_gate is not None:
                try:
                    trace.evidence_gate = self._evidence_gate.summary()
                except Exception:  # observability must never break the run
                    trace.evidence_gate = None
            await self.emit("completed", trace)

            # On a verified-successful run, distil the tool trace into a skill
            # (flag-gated). maybe_generate_skill's quality gate requires reason
            # completed/verified, so a failed run produces nothing.
            if self._auto_skill and self._tool_trace:
                _distilled = await self._maybe_distill_skill(goal, trace)
                # T1-1: capture this run as a replayable corpus task
                # for the freshly distilled skill (gated; closes the loop).
                self._maybe_capture_replay(_distilled)

            # T1-1: record whether an injected skill coincided with a verified
            # run (observational baseline). Logging only — never affects the run.
            self._log_skill_outcome(trace)

            return trace

        except asyncio.CancelledError:
            self._status = AgentStatus.IDLE
            return CompletionTrace(reason="cancelled")
        except Exception as exc:
            self._status = AgentStatus.IDLE
            log.error("agent_loop_error", error=str(exc))
            await self.emit("error", exc)
            return CompletionTrace(reason=f"error: {exc}")
        finally:
            self._running = False

    async def cancel(self) -> None:
        """Request cancellation of the running loop."""
        self._cancel_event.set()

    def _select_tools(self, classification: ClassificationResult) -> list[str]:
        """Select tool subset based on goal classification."""
        from rune.config.defaults import TOOLS_CHAT, TOOLS_RESEARCH, TOOLS_WEB

        match classification.goal_type:
            case "chat":
                return TOOLS_CHAT
            case "web":
                return TOOLS_WEB
            case "research":
                return TOOLS_RESEARCH
            case _:
                # Full toolset for code_modify, execution, browser, full
                from rune.capabilities.registry import get_capability_registry

                return get_capability_registry().list_names()

    def _get_disabled_tools(self) -> set[str]:
        """Return tool names that should be disabled due to stall limits."""
        disabled: set[str] = set()
        limits = STALL_LIMITS

        if self._stall.web_search_count >= limits["web"]["searchHardStop"]:
            disabled.add("web_search")
        if self._stall.web_fetch_count >= limits["web"]["fetchHardStop"]:
            disabled.add("web_fetch")
        if self._stall.browser_no_match_count >= limits["browserFind"]["noMatchHardStop"]:
            disabled.add("browser_find")
        if self._stall.file_read_exhausted:
            file_read_limit = limits["fileRead"]["hardStop"]
            if getattr(self._stall, "file_read_count", 0) >= file_read_limit:
                disabled.add("file_read")

        return disabled

    async def _maybe_distill_skill(self, goal: str, trace: Any) -> str | None:
        """Distil this run's tool trace into a skill (best-effort).

        Gated upstream by ``self._auto_skill``. ``maybe_generate_skill`` applies
        the quality gate (success + evidence), so only good runs register a
        skill. Never raises into the run.
        """
        try:
            from rune.agent.memory_bridge import maybe_generate_skill

            skill = await maybe_generate_skill(
                goal=goal,
                result=trace,
                intent=None,
                trace=self._tool_trace,
                refiner=_FastSkillRefiner(),
            )
            if skill:
                log.info(
                    "auto_skill_distilled",
                    name=skill.get("name"),
                    steps=len(skill.get("steps", [])),
                    quality=skill.get("quality_score"),
                )
                return skill.get("name")
        except Exception as exc:
            log.debug("auto_skill_distill_failed", error=str(exc)[:120])
        return None

    def _maybe_capture_replay(self, skill_name: str | None) -> None:
        """Snapshot the pre-task workspace for a distilled skill (T1-1 1b).

        Closes the gated-learning loop: a skill distilled this run becomes a
        replayable corpus task (pre-task git ref + project check) so it can be
        A/B'd offline before being trusted. Best-effort; gated upstream by
        capture_replay (self._replay_capture is None when off).
        """
        if not skill_name or self._replay_capture is None:
            return
        try:
            from rune.memory.store import get_memory_store
            from rune.skills.capture import capture_replay_snapshot

            cap = self._replay_capture
            capture_replay_snapshot(
                cap["goal"],
                skill_name,
                store=get_memory_store(),
                cwd=cap["cwd"],
                head_ref=cap["head"],
            )
        except Exception as exc:
            log.debug("replay_capture_end_failed", error=str(exc)[:120])

    def _log_skill_outcome(self, trace: Any) -> None:
        """Record an injected skill's run outcome (T1-1, observational).

        Pure measurement: writes one ``arm="with"`` row tying the injected skill
        to whether the run verified, building the base success rate the
        evaluator compares a control arm against. Best-effort; never raises into
        the run.
        """
        if self._injected_skill is None:
            return
        name, score = self._injected_skill
        verified = getattr(trace, "reason", "") in ("completed", "verified")
        try:
            from rune.memory.store import get_memory_store

            get_memory_store().log_skill_eval(
                name,
                verified=verified,
                session_id=self._session_id or "",
                arm="with",
                match_score=score,
                duration_ms=getattr(trace, "duration_ms", 0) or 0,
                tokens=self._token_budget.used,
            )
            log.info("skill_outcome_logged", skill=name, verified=verified)
        except Exception as exc:
            log.debug("skill_outcome_log_failed", error=str(exc)[:120])

    def _build_skill_context(self, goal: str, goal_category: str) -> str | None:
        """Retrieve a matching learned skill to inject (flag-gated, best-effort)."""
        if not self._auto_skill or goal_category not in ("code", "full"):
            return None
        try:
            from rune.skills.executor import build_skill_context_for_goal

            gated = False
            try:
                from rune.config import get_config

                gated = bool(getattr(get_config().skills, "gated_learning", False))
            except Exception:
                gated = False
            ctx = build_skill_context_for_goal(goal, gated=gated)
            if ctx is None:
                return None
            # T1-1: remember which skill was injected so its outcome can be
            # logged when the run finishes (observational arm).
            self._injected_skill = (ctx.active_skill_name, 0.0)
            return ctx.formatted_context
        except Exception:
            return None

    def _build_system_prompt(
        self,
        goal: str,
        classification: ClassificationResult,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Build the system prompt for the agent using the canonical prompt builder."""
        # Map GoalType → goal_category for prompt assembly
        from rune.config.defaults import TOKEN_OPTIMIZATION_ENABLED

        if TOKEN_OPTIMIZATION_ENABLED:
            _CATEGORY_MAP: dict[str, str] = {
                "code_modify": "code",
                "research": "code",
                "execution": "code",
                "web": "web",
                "browser": "browser",
                "full": "full",
                "chat": "chat",  # optimized: lightweight chat prompt
            }
        else:
            _CATEGORY_MAP: dict[str, str] = {
                "code_modify": "code",
                "research": "code",
                "execution": "code",
                "web": "web",
                "browser": "browser",
                "full": "full",
                "chat": "full",  # original: full prompt for chat
            }
        goal_category = _CATEGORY_MAP.get(classification.goal_type, "full")

        # Enable deep research prompts (synthesis verification, source quality)
        # for research/web/full tasks, but NOT for simple chat
        goal_type = getattr(classification, "goal_type", "")
        is_deep = goal_type in ("research", "web", "full") and goal_type != "chat"

        # Build repo map for code tasks (auto-selects most relevant symbols)
        repo_map_text: str | None = None
        if goal_category in ("code", "full"):
            try:
                from rune.intelligence.repo_map import build_repo_map_sync

                repo_map_text = build_repo_map_sync(os.getcwd(), max_tokens=2048)
            except Exception:
                pass

        # Detect connected MCP servers for prompt guide
        from rune.capabilities.registry import get_capability_registry

        _reg = get_capability_registry()
        _mcp_caps = [c for c in _reg.list_all() if c.name.startswith("mcp.")]
        _mcp_servers: dict[str, int] = {}
        for c in _mcp_caps:
            server = c.name.split(".")[1] if len(c.name.split(".")) > 1 else "unknown"
            _mcp_servers[server] = _mcp_servers.get(server, 0) + 1

        return build_system_prompt(
            goal=goal,
            classification=classification,
            memory_context=context,
            goal_category=goal_category,
            channel=getattr(self._config, "channel", None),
            environment={
                "cwd": os.getcwd(),
                "home": os.path.expanduser("~"),
            },
            repo_map=repo_map_text,
            is_deep_research=is_deep,
            has_mcp_services=bool(_mcp_servers),
            mcp_server_names=_mcp_servers,
            skill_context=self._build_skill_context(goal, goal_category),
        )

    async def _execute_loop(
        self,
        goal: str,
        system_prompt: str,
        tools: list[str],
        max_iterations: int,
        classification: ClassificationResult,
        context: dict[str, Any] | None = None,
        message_history: list[dict[str, str]] | None = None,
    ) -> CompletionTrace:
        """The inner execution loop using PydanticAI.

        Runs the agent in a multi-step loop with:
        - Cognitive caching (avoids redundant tool calls)
        - Completion gate evaluation after each step
        - Failover with retry / profile switching / compaction
        - Observation masking for large tool outputs (>4000 chars)
        - Token budget tracking from PydanticAI usage stats
        """
        trace = CompletionTrace()
        failover = FailoverManager()
        cache = SessionToolCache(max_entries=COGNITIVE_CACHE_MAX)
        self._cognitive_cache = cache
        evidence = ExecutionEvidenceSnapshot()

        # Initialize rehydration subsystem
        try:
            from rune.agent.rehydration import CompactionRecorder, RehydrationTrigger

            session_id = self._session_id or uuid.uuid4().hex[:12]
            self._session_id = session_id
            self._rehydration_recorder = CompactionRecorder(session_id)
            self._rehydration_trigger = RehydrationTrigger()
        except Exception as exc:
            log.debug("rehydration_init_skipped", error=str(exc)[:100])

        # Observation masking constant
        _OBSERVATION_TRUNCATE_LIMIT = 4000

        _last_tool_params: dict[str, Any] = {}

        async def _on_tool_start(cap_name: str, params: dict[str, Any]) -> None:
            nonlocal _last_tool_params
            _last_tool_params = params
            self._last_activity = time.monotonic()  # (#30) activity on tool call
            self._stall.mark_activity(cap_name)
            _step_tool_calls.append(cap_name)
            # Track written file paths (#16)
            if cap_name in ("file_write", "file_edit", "file_delete"):
                fp = params.get("file_path") or params.get("path", "")
                if fp:
                    self._files_written.add(fp)
            # Detect activity phase for adaptive observation windows
            self._prev_activity_phase = self._activity_phase
            if cap_name in ("file_write", "file_edit"):
                self._activity_phase = "implementation"
            elif cap_name in ("bash_execute",) and any(
                kw in (params.get("command", ""))
                for kw in ("test", "pytest", "vitest", "jest", "check")
            ):
                self._activity_phase = "verification"
            elif cap_name in ("web_search", "web_fetch"):
                self._activity_phase = "research"
            await self.emit("tool_call", {"name": cap_name, "params": params})

        def _tool_key(name: str, params: dict[str, Any]) -> str:
            """Extract command-level key for behavior prediction.

            "bash_execute" + {"command": "uv run ruff check ."} → "bash:ruff"
            "file_write" + any → "file_write" (unchanged)
            """
            if name == "bash_execute":
                cmd = params.get("command", "")
                _skip = {"uv", "python", "python3", "npx", "run", "exec", "sudo", "-m", "-c"}
                for part in cmd.split():
                    if part not in _skip and not part.startswith("-"):
                        return f"bash:{part}"
            return name

        async def _on_tool_end(cap_name: str, result: CapabilityResult) -> None:
            self._last_activity = time.monotonic()  # (#30) activity on tool result

            # A throwaway best-of-K sample (RUNE_IN_BEST_OF set in the attempt
            # subprocess) must not persist tool-call telemetry: the proactive
            # engine seeds its behavior predictor from tool_call_log, so K
            # discarded attempts would pollute later sessions' predictions. Skip
            # both the persistent log and the in-process predictor record.
            _ephemeral = bool(os.environ.get("RUNE_IN_BEST_OF"))

            # Record tool call to persistent log (with params for command extraction)
            if not _ephemeral:
                try:
                    from rune.memory.store import get_memory_store

                    _store = get_memory_store()
                    _sid = self._session_id or "unknown"
                    _err = getattr(result, "error", "") or "" if not result.success else ""
                    _store.log_tool_call(
                        _sid,
                        cap_name,
                        params=_last_tool_params or None,
                        result_success=result.success,
                        error_message=str(_err)[:500],
                        duration_ms=getattr(result, "duration_ms", 0) or 0,
                    )
                except Exception:
                    pass  # Tool logging must never break the agent loop

            # Record tool call for proactive behavior prediction
            if not _ephemeral:
                try:
                    from rune.proactive.prediction.engine import get_prediction_engine

                    pred_eng = get_prediction_engine()
                    if result.success:
                        pred_eng.behavior_predictor.record_tool_call(
                            _tool_key(cap_name, _last_tool_params)
                        )
                    # Also record success/failure for frustration detection
                    pred_eng._recent_actions.append(
                        {
                            "type": "tool",
                            "tool": cap_name,
                            "success": result.success,
                        }
                    )
                    # Keep bounded
                    if len(pred_eng._recent_actions) > 50:
                        pred_eng._recent_actions = pred_eng._recent_actions[-50:]
                except Exception:
                    pass  # Prediction recording must never break the agent loop

            # Update evidence counters
            if cap_name == "file_read":
                evidence.file_reads += 1
                evidence.reads += 1
                self._consecutive_reads_without_write += 1  # (#27)
                # Track unique file reads for R15 analysis depth
                fp = _last_tool_params.get("file_path") or _last_tool_params.get("path", "")
                if fp and fp not in self._files_read:
                    self._files_read.add(fp)
                evidence.unique_file_reads = len(self._files_read)
            elif cap_name in ("file_write", "file_edit", "file_delete"):
                evidence.writes += 1
                self._consecutive_reads_without_write = 0  # (#27) reset
                self._pending_verification_nudge = True  # (#27)
                # Written file paths are tracked in _on_tool_start where
                # params are available - no duplicate tracking needed here.
                # Note code-file writes for the verification gate. Count attempts,
                # not just successes: a rejected write still leaves the task
                # unverified, so it shouldn't finish on the model's word alone.
                if cap_name != "file_delete":
                    fp = _last_tool_params.get("file_path") or _last_tool_params.get("path", "")
                    from rune.intelligence.ast_analyzer import is_code_extension

                    _dot = fp.rfind(".")
                    if fp and _dot >= 0 and is_code_extension(fp[_dot:]):
                        self._tool_call_seq += 1
                        self._last_code_write_step = self._tool_call_seq
            elif cap_name == "bash_execute":
                evidence.executions += 1
                self._consecutive_reads_without_write = 0  # (#27) reset
                # R19: any successful bash counts as verification
                from rune.agent.bash_parsing import is_verification_command

                _bash_cmd = _last_tool_params.get("command", "")
                if result.success:
                    self._tool_call_seq += 1
                    self._last_verify_step = self._tool_call_seq
                    # R06: count formal verification commands separately
                    if _bash_cmd and is_verification_command(_bash_cmd):
                        evidence.verifications += 1
                        self._last_verify_failed = False  # tests just passed
                elif _bash_cmd and is_verification_command(_bash_cmd):
                    self._last_verify_failed = True  # tests just failed
            elif cap_name == "web_search":
                evidence.web_searches += 1
                self._stall.web_search_count += 1  # (#15)
            elif cap_name == "web_fetch":
                evidence.web_fetches += 1
                self._stall.web_fetch_count += 1  # (#15)
            elif cap_name == "browser_extract":
                # browser_extract produces actual data — count as a read
                evidence.reads += 1
                evidence.browser_reads += 1
            elif cap_name.startswith("browser_"):
                evidence.browser_reads += 1

            # Track hard failures with dedup (#16)
            if not result.success and result.error:
                sig = result.error[:80]
                if sig not in self._hard_failure_signatures:
                    self._hard_failure_signatures.add(sig)
                    self._hard_failures.append(sig)
                # Also record in stall state
                self._stall.record_error(sig)

            if _env_flag("RUNE_BENCH_CAPTURE_FAILED_TOOL_OUTPUT"):
                if not result.success:
                    self._recent_failed_tool_nudge = _failed_tool_nudge(cap_name, result)
                elif self._recent_failed_tool_nudge and _should_clear_failed_tool_nudge(
                    cap_name, result
                ):
                    self._recent_failed_tool_nudge = ""

            await self.emit("tool_result", _tool_result_event_payload(cap_name, result))

            # Capture the tool step for skill distillation (flag-gated;
            # ephemeral best-of attempts never persist).
            if self._auto_skill and not _ephemeral:
                try:
                    from rune.agent.memory_bridge import ToolTraceEntry

                    if len(self._tool_trace) < 100:  # bound memory
                        self._tool_trace.append(
                            ToolTraceEntry(
                                tool_name=cap_name,
                                params=dict(_last_tool_params or {}),
                                result_summary=(getattr(result, "output", "") or "")[:200],
                                success=bool(result.success),
                            )
                        )
                except Exception:
                    pass  # skill capture must never break the loop

        # Build PydanticAI agent and tool set

        # Resolve model from failover profile (reads active_provider/active_model from config)
        from rune.llm.client import loop_model_string

        profile = failover.current_profile
        if profile.provider not in ("none", ""):
            model = loop_model_string(profile.provider, profile.model)
        else:
            model = self._config.model

        # Simple-query fast lane: high-confidence chat/web goals run on the
        # provider's fast tier. Must come before advisor pairing, vision trim
        # and tool-round scaling so they all see the lane model.
        from rune.agent.fast_lane import FAST_LANE_UPSHIFT_BLOCKS, decide_fast_lane

        _primary_model = model
        _lane = decide_fast_lane(classification)
        _fast_lane_active = _lane.active
        if _lane.active and _lane.model:
            model = _lane.model
            if model != _primary_model:
                log.info(
                    "fast_lane_downshift",
                    model=model,
                    primary=_primary_model,
                    goal_type=classification.goal_type,
                    confidence=classification.confidence,
                )
            else:
                # Primary already is the fast-tier model (light-model /
                # ollama sessions); only the round cap and grounding change.
                log.info(
                    "fast_lane_active_no_downshift",
                    model=model,
                    goal_type=classification.goal_type,
                    confidence=classification.confidence,
                )

        adapter_opts = ToolAdapterOptions(
            cognitive_cache=cache,
            stall_state=self._stall,
            allowed_tools=tools,
            step_counter=lambda: self._step,
            on_tool_start=_on_tool_start,
            on_tool_end=_on_tool_end,
            approval_callback=self._approval_callback,
            fast_lane_active=lambda: _fast_lane_active,
            workspace_root=(context or {}).get("workspace_root", ""),
        )
        tool_functions = build_tool_set(adapter_opts)

        # Wire ask_user callback if set
        if self._ask_user_callback is not None:
            from rune.capabilities.ask_user import reset_ask_user_count, set_ask_user_callback

            set_ask_user_callback(self._ask_user_callback)
            reset_ask_user_count()

        # Remove vision-only tools if the model doesn't support vision.
        # This prevents non-vision models from wasting tokens on
        # screenshots they cannot interpret.
        from rune.llm.model_capabilities import get_capabilities

        _model_name = model.split("/")[-1] if "/" in model else model
        _caps = get_capabilities(_model_name)
        if not _caps.supports_vision:
            _VISION_TOOLS = {"browser_screenshot"}
            tools = [t for t in tools if t not in _VISION_TOOLS]

        advisor_service = AdvisorService.for_episode(model)

        # Native advisor tool (Anthropic pairs only, inert otherwise)
        from rune.agent.advisor.native_tool import (
            build_native_tool_wrapper,
            resolve_native_config,
        )

        _native_cfg = resolve_native_config(
            executor_model=model,
            advisor_model_full=advisor_service.model_full,
            max_uses=advisor_service.budget.max_calls,
        )
        if _native_cfg.enabled:
            _native_wrapper = build_native_tool_wrapper(_native_cfg)
            if _native_wrapper is not None:
                tool_functions["advisor"] = _native_wrapper
            from rune.agent.prompts import PROMPT_ADVISOR_TIMING

            system_prompt = system_prompt + "\n\n" + PROMPT_ADVISOR_TIMING
            log.info(
                "advisor_native_path_enabled",
                executor=model,
                advisor=advisor_service.model_full,
                max_uses=_native_cfg.max_uses,
            )

        # Scale tool rounds by complexity and executor capability.
        _tool_rounds = _compute_tool_rounds(
            classification,
            model,
            advisor_service.enabled,
            fast_lane=_fast_lane_active,
        )
        agent = LiteLLMAgent(
            model=model,
            system_prompt=system_prompt,
            tools=list(tool_functions.values()),
            max_tool_rounds=_tool_rounds,
            extra_headers=_native_cfg.beta_headers,
        )

        messages: list[Any] = []

        # Seed with conversation history from previous turns so the LLM
        # has multi-turn context (fixes multi-turn follow-up like "정리해").
        if message_history:
            if getattr(classification, "is_domain_change", False):
                # Domain changed: strip tool results to prevent context bleed.
                # Keep only user/assistant messages so the LLM has conversational
                # context without being polluted by unrelated tool outputs.
                for msg in message_history:
                    role = msg.get("role", "")
                    if role == "tool":
                        continue
                    if role == "assistant":
                        # Keep only the final text, drop tool_calls
                        cleaned = {"role": "assistant", "content": msg.get("content", "")}
                        if cleaned["content"]:
                            messages.append(cleaned)
                    else:
                        messages.append(dict(msg))
                log.info("domain_change_history_trimmed", kept=len(messages))
            else:
                for msg in message_history:
                    messages.append(dict(msg))

        # Workspace root for guard checks
        workspace_root = (context or {}).get("workspace_root", "") if context else ""

        # (advisor_service + native path detection happens earlier —
        # see `AdvisorService.for_episode(model)` above, right after
        # `build_tool_set()`. That site is required because the native
        # advisor tool must be in the first LiteLLMAgent construction.)

        import os as _os_for_freshness

        verify_freshness_enabled = _os_for_freshness.environ.get(
            "RUNE_VERIFY_FRESHNESS", ""
        ).strip().lower() in ("1", "true", "yes", "on")

        # Don't let a code task finish while its latest change is unverified —
        # nothing ran since the change, or the last test failed and never passed.
        # On by default (RUNE_REQUIRE_TEST_PASS=0 turns it off), and only for
        # tasks that actually need verification. Getting a green test clears it;
        # if it never clears, the run ends as max_gate_blocked, which points the
        # user at /escalate.
        _require_test_pass = _os_for_freshness.environ.get(
            "RUNE_REQUIRE_TEST_PASS", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
        try:
            _require_test_pass = _require_test_pass and resolve_intent_contract(
                classification, classification.confidence
            ).requires_code_verification
        except Exception:
            _require_test_pass = False

        def _unverified_code() -> bool:
            """True when the latest code change still has no passing test behind it."""
            if not _require_test_pass:
                return False
            if self._last_code_write_step > self._last_verify_step:
                return True
            return self._last_verify_failed

        # main loop
        _prev_evidence_total = 0
        _no_new_evidence_steps = 0
        _gate_blocked_count = 0

        # Last advisor raw text — fed back into reconcile requests
        _last_advisor_raw: str = ""

        # Advisor closures (deferred: request built only after should_call fires)
        def _make_policy_input():
            return build_policy_input(
                classification=classification,
                activity_phase=self._activity_phase,
                reads=evidence.reads,
                writes=evidence.writes,
                web_fetches=evidence.web_fetches,
                files_written=len(self._files_written),
                gate_blocked_count=_gate_blocked_count,
                stall_consecutive=self._stall.consecutive_no_progress,
                no_progress_steps=_no_new_evidence_steps,
                wind_down_phase=self._wind_down_phase,
                hard_failures=len(self._hard_failures),
            )

        def _load_file_contents_for_architect() -> dict[str, str]:
            """Read files_written from disk for architect-mode advisor.

            Only fires when advisor mode is architect. Bounded to the
            5 most recent files, 20KB per file. Read errors are skipped
            so advisor calls never fail on I/O.
            """
            if advisor_service.mode != "architect":
                return {}
            contents: dict[str, str] = {}
            for path in list(self._files_written)[-5:]:
                try:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        data = f.read(20_000)
                    contents[path] = data
                except Exception as exc:
                    log.debug("architect_file_read_failed", path=path, error=str(exc)[:100])
            return contents

        def _make_advisor_request(trigger: str, gate_result=None):
            return build_advisor_request(
                trigger=trigger,
                goal=goal,
                classification=classification,
                activity_phase=self._activity_phase,
                step=self._step,
                token_budget_frac=(self._token_budget.used / max(1, self._token_budget.total)),
                evidence=evidence,
                gate_result=gate_result,
                stall_consecutive=self._stall.consecutive_no_progress,
                stall_cumulative=self._stall.cumulative_no_progress,
                recent_messages=messages,
                files_written=self._files_written,
                hard_failures=list(self._hard_failures),
                last_advisor_note=_last_advisor_raw or None,
                file_contents=_load_file_contents_for_architect(),
            )

        # BCT: deferred compliance tracking for advisor plans
        _pending_advice: PendingAdvice | None = None
        _step_tool_calls: list[str] = []

        for step in range(max_iterations):
            self._step = step + 1
            self._step_start_time = time.monotonic()
            _step_tool_calls.clear()

            # -- pre-flight checks ---
            if self._cancel_event.is_set():
                trace.reason = "cancelled"
                break

            # Fast-lane escape hatch: failover only fires on exceptions and
            # only moves down the chain, so gate pressure is what restores
            # full rigor. Checked here, not in one gate branch, so blocks
            # from every gate path (_finalize_gates, _auto_verify_gate,
            # Evidence Gate) count toward the upshift.
            if _fast_lane_active and _gate_blocked_count >= FAST_LANE_UPSHIFT_BLOCKS:
                _fast_lane_active = False
                log.info(
                    "fast_lane_upshift",
                    step=self._step,
                    gate_blocked=_gate_blocked_count,
                )
                # The restored model gets the full block budget and a clean
                # no-progress window (the lane's empty steps raised both).
                # Bounded: the upshift fires once, so a run absorbs at most
                # the lane's ~3 blocks plus the normal 5.
                _gate_blocked_count = 0
                self._gate_blocked_count = 0
                _no_new_evidence_steps = 0
                # Restore from the live failover profile, not the run-start
                # snapshot — failover may have switched providers mid-lane.
                _up_profile = failover.current_profile
                if _up_profile.provider not in ("none", ""):
                    model = loop_model_string(_up_profile.provider, _up_profile.model)
                else:
                    model = _primary_model
                # Re-resolve the native advisor config for the restored
                # model — the setup one was paired with the lane model.
                _native_cfg = resolve_native_config(
                    executor_model=model,
                    advisor_model_full=advisor_service.model_full,
                    max_uses=advisor_service.budget.max_calls,
                )
                # Rebuild even when the model is unchanged (light-model
                # sessions) — the 3-round cap goes away with the rigor.
                _tool_rounds = _compute_tool_rounds(
                    classification,
                    model,
                    advisor_service.enabled,
                )
                agent = LiteLLMAgent(
                    model=model,
                    system_prompt=system_prompt,
                    tools=list(tool_functions.values()),
                    max_tool_rounds=_tool_rounds,
                    extra_headers=_native_cfg.beta_headers,
                )

            # BCT: check compliance for a pending advisor plan
            if _pending_advice is not None:
                _ev_total = (
                    evidence.reads + evidence.writes + evidence.executions + evidence.web_fetches
                )
                _verdict = check_compliance(
                    _pending_advice,
                    self._step,
                    _ev_total,
                    len(self._hard_failures),
                    _step_tool_calls,
                )
                if _verdict is not None:
                    if _verdict.followed:
                        self._stall.consecutive_no_progress = 0
                        _no_new_evidence_steps = 0
                        log.info(
                            "advisor_compliance_followed", reason=_verdict.reason, step=self._step
                        )
                    else:
                        log.info(
                            "advisor_compliance_ignored", reason=_verdict.reason, step=self._step
                        )
                    # Merge verdict back into call_history so persistence
                    # writes it to advisor_events (Plan B measurement).
                    idx = _pending_advice.call_history_index
                    history = advisor_service.budget.call_history
                    if 0 <= idx < len(history):
                        history[idx]["compliance_verdict"] = _verdict.reason
                        history[idx]["expected_tool"] = _pending_advice.expected_tool or ""
                        history[idx]["observed_tools"] = ",".join(
                            _pending_advice.observed_tools[:20]
                        )
                        history[idx]["step_at_call"] = _pending_advice.injected_at_step
                        history[idx]["pre_call_evidence"] = _pending_advice.baseline_evidence
                        history[idx]["post_call_evidence"] = _ev_total
                        history[idx]["advice_mode"] = _pending_advice.advice_mode
                    _pending_advice = None

            # Site A: advisor hook (skipped when native path active)
            if advisor_service.enabled and not _native_cfg.enabled:
                messages, _adv_dec = await maybe_consult(
                    advisor_service,
                    policy_input=_make_policy_input(),
                    build_request=lambda trg: _make_advisor_request(trg),
                    messages=messages,
                    inject=self._inject_system_message,
                )
                if _adv_dec and _adv_dec.raw_text:
                    _last_advisor_raw = _adv_dec.raw_text[:300]
                if _adv_dec and _adv_dec.action == "abort":
                    log.warning("advisor_recommended_abort", step=self._step)
                    trace.reason = "advisor_abort"
                    trace.final_step = self._step
                    break
                if _adv_dec and _adv_dec.plan_steps and _adv_dec.trigger == "stuck":
                    # Deferred reset: set pending_advice, counters reset on verdict
                    _ev_total = (
                        evidence.reads
                        + evidence.writes
                        + evidence.executions
                        + evidence.web_fetches
                    )
                    _pending_advice = build_pending(
                        _adv_dec,
                        self._step,
                        _ev_total,
                        len(self._hard_failures),
                        call_history_index=len(advisor_service.budget.call_history) - 1,
                        advice_mode=advisor_service.mode,
                    )

            if self._stall.is_stalled:
                # Defer stall when prior tool evidence exists the LLM
                # may still be formulating its answer after a successful
                # tool call.  Other stall conditions (bash_stalled,
                # file_read_exhausted, intent_repeat) are not deferred.
                prior_evidence = (
                    evidence.reads
                    + evidence.writes
                    + evidence.executions
                    + evidence.web_searches
                    + evidence.web_fetches
                )
                if prior_evidence > 0 and self._stall.consecutive_no_progress < 5:
                    log.debug(
                        "stall_deferred",
                        step=self._step,
                        evidence=prior_evidence,
                        consecutive=self._stall.consecutive_no_progress,
                    )
                else:
                    log.warning(
                        "agent_stalled",
                        step=self._step,
                        consecutive=self._stall.consecutive_no_progress,
                    )
                    trace.reason = "stalled"
                    break

            # Wind-down hard_stop check (#14)
            self._update_wind_down_phase()
            if self._wind_down_phase == "hard_stop":
                log.warning("wind_down_hard_stop", used=self._token_budget.used)
                trace.reason = "token_budget_exhausted"
                break

            # Disable non-essential tools in "final" phase (#14)
            if self._wind_down_phase == "final":
                essential = {
                    "think",
                    "file_read",
                    "file_write",
                    "file_edit",
                    "bash_execute",
                    "ask_user",
                }
                if set(tools) != essential and not set(tools).issubset(essential):
                    tools = [t for t in tools if t in essential]
                    adapter_opts.allowed_tools = tools
                    tool_functions = build_tool_set(adapter_opts)
                    agent = LiteLLMAgent(
                        model=model,
                        system_prompt=system_prompt,
                        tools=list(tool_functions.values()),
                        max_tool_rounds=_tool_rounds,
                    )
                    log.info("tools_reduced_final_phase", tools=len(tools))
                messages = self._maybe_force_wind_down_write(messages)

            # Active tools reduction after the configured step
            if self._step >= ACTIVE_TOOLS_REDUCTION_STEP:
                reduced = self._reduce_active_tools(tools)
                if set(reduced) != set(tools):
                    tools = reduced
                    adapter_opts.allowed_tools = tools
                    tool_functions = build_tool_set(adapter_opts)
                    agent = LiteLLMAgent(
                        model=model,
                        system_prompt=system_prompt,
                        tools=list(tool_functions.values()),
                        max_tool_rounds=_tool_rounds,
                    )

            # Stall-limit tool removal (#H3)
            disabled = self._get_disabled_tools()
            if disabled:
                before_count = len(tools)
                tools = [t for t in tools if t not in disabled]
                if len(tools) < before_count:
                    adapter_opts.allowed_tools = tools
                    tool_functions = build_tool_set(adapter_opts)
                    agent = LiteLLMAgent(
                        model=model,
                        system_prompt=system_prompt,
                        tools=list(tool_functions.values()),
                        max_tool_rounds=_tool_rounds,
                    )
                    log.info("tools_disabled_stall", disabled=sorted(disabled))

            log.debug("agent_step", step=self._step, tools=len(tools))
            await self.emit("step", self._step)

            # -- step watchdog: check elapsed time from previous step (#4a) --
            elapsed_ms = (time.monotonic() - self._step_start_time) * 1000
            if elapsed_ms > self._STEP_ABORT_MS:
                log.error("step_abort", step=self._step, elapsed_ms=elapsed_ms)
                raise TimeoutError(f"Step {self._step} exceeded {self._STEP_ABORT_MS}ms timeout")
            elif elapsed_ms > self._STEP_WARN_MS:
                log.warning("step_slow", step=self._step, elapsed_ms=elapsed_ms)

            # -- prepare step hook (#25): masking + nudges + guards --
            if messages:
                messages = self._prepare_step(messages, workspace_root)

            # -- run PydanticAI agent --
            try:
                remaining_tokens = max(0, self._token_budget.total - self._token_budget.used)
                usage_limits = UsageLimits(
                    request_tokens_limit=remaining_tokens,
                    response_tokens_limit=min(
                        self._max_output_tokens,
                        self._config.max_tokens,
                        remaining_tokens,
                    ),
                )

                async with agent.run_stream(
                    goal,
                    message_history=messages or None,
                    usage_limits=usage_limits,
                ) as stream:
                    # Inject persistent fail streak into this step (#P4)
                    if self._persistent_fail_streak:
                        stream.inject_failure_state(
                            self._persistent_fail_streak,
                            set(),
                        )
                    collected_text = ""

                    async for delta in stream.stream_text(delta=True):
                        collected_text += delta
                        self._last_activity = time.monotonic()  # (#30)
                        await self.emit("text_delta", delta)

                    # Mark activity for stream completion (#30)
                    self._last_activity = time.monotonic()
                    result = stream

                # -- export fail streak for cross-step persistence (#P4) --
                try:
                    streak, _ = result.get_failure_state()
                    self._persistent_fail_streak.update(streak)
                except Exception:
                    pass

                # -- update token budget from usage stats --
                try:
                    usage = result.usage()
                    request_tokens = (
                        getattr(usage, "input_tokens", 0)
                        or getattr(usage, "request_tokens", 0)
                        or 0
                    )
                    response_tokens = (
                        getattr(usage, "output_tokens", 0)
                        or getattr(usage, "response_tokens", 0)
                        or 0
                    )
                    cached_input_tokens = getattr(usage, "cached_input_tokens", 0) or 0
                    cache_write_tokens = getattr(usage, "cache_write_tokens", 0) or 0
                    reasoning_tokens = getattr(usage, "reasoning_tokens", 0) or 0
                    step_tokens = request_tokens + response_tokens
                    self._token_budget.used += step_tokens
                    trace.total_tokens_used = self._token_budget.used
                    await self.emit(
                        "step_tokens",
                        self._step,
                        step_tokens,
                        self._token_budget.used,
                        self._token_budget.total,
                    )
                    await self.emit(
                        "model_usage",
                        {
                            "step": self._step,
                            "model": model,
                            "input_tokens": max(0, request_tokens - cached_input_tokens),
                            "raw_input_tokens": request_tokens,
                            "cached_input_tokens": cached_input_tokens,
                            "cache_write_tokens": cache_write_tokens,
                            "reasoning_tokens": reasoning_tokens,
                            "output_tokens": response_tokens,
                            "step_tokens": step_tokens,
                            "total_tokens": self._token_budget.used,
                            "token_budget": self._token_budget.total,
                        },
                    )
                except Exception:
                    pass

                # -- rollover / checkpoint handling --
                rollover_phase = self._token_budget.rollover_phase
                if rollover_phase >= 1 and 1 not in self._rollover_phase_done:
                    # Phase 1 (70%): Save structured checkpoint (#2a)
                    try:
                        ckpt = CheckpointManager()
                        session_id = self._session_id or str(uuid.uuid4())[:8]
                        self._session_id = session_id

                        # Build structured checkpoint with evidence (#2a)
                        evidence_list = list(self._files_written) if self._files_written else []
                        stall_summary = {
                            "consecutive_no_progress": self._stall.consecutive_no_progress,
                            "cumulative_no_progress": self._stall.cumulative_no_progress,
                            "is_stalled": self._stall.is_stalled,
                            "last_tool_call": self._stall.last_tool_call,
                        }
                        try:
                            gate_input_snapshot = gate_input  # type: ignore[possibly-undefined]
                        except NameError:
                            gate_input_snapshot = None
                        completion_progress: dict[str, Any] = {}
                        if gate_input_snapshot:
                            completion_progress = {
                                "intent_resolved": gate_input_snapshot.intent_resolved,
                                "changed_files_count": gate_input_snapshot.changed_files_count,
                                "answer_length": gate_input_snapshot.answer_length,
                            }

                        ckpt_data = CheckpointData(
                            session_id=session_id,
                            step=self._step,
                            goal=goal,
                            token_usage=self._token_budget.used,
                        )
                        # Attach structured metadata as tool_results (#2a)
                        ckpt_data.tool_results = [
                            {"evidence": evidence_list},
                            {"changed_files": list(self._files_written)},
                            {"stall_state": stall_summary},
                            {"completion_progress": completion_progress},
                        ]
                        ckpt.save(ckpt_data)
                        log.info("checkpoint_saved", step=self._step, phase=1)
                    except Exception as exc:
                        log.warning("checkpoint_save_failed", error=str(exc)[:100])
                    self._rollover_phase_done.add(1)

                # -- update message history for next iteration --
                with contextlib.suppress(Exception):
                    messages = result.all_messages()

                if rollover_phase >= 2 and 2 not in self._rollover_phase_done:
                    # Phase 2 (80%): Deterministic rollover - compact old messages
                    # Use dynamic context cap to determine how aggressively to compact (#2b)
                    dynamic_cap = self._get_dynamic_context_cap()
                    keep_count = max(3, int(FULL_WINDOW_MAX * dynamic_cap / 0.70))
                    if messages and len(messages) > keep_count + 2:
                        # Record originals before compaction (rehydration)
                        if self._rehydration_recorder is not None:
                            to_compact = (
                                messages[:-keep_count] if keep_count < len(messages) else []
                            )
                            if to_compact:
                                try:
                                    raw = [
                                        m if isinstance(m, dict) else m.model_dump()
                                        for m in to_compact
                                    ]
                                    await self._rehydration_recorder.record(
                                        raw,
                                        step_range=(max(0, self._step - len(raw)), self._step),
                                        activity_phase=self._activity_phase,
                                        compaction_event="phase_2_rollover",
                                    )
                                except Exception as exc:
                                    log.debug("rehydration_record_p2_failed", error=str(exc)[:100])
                        # Use turn-atomic compaction to keep assistant+tool pairs together
                        messages = self._compact_messages_atomic(messages, keep_last=keep_count)
                        log.info(
                            "rollover_compacted",
                            step=self._step,
                            phase=2,
                            msg_count=len(messages),
                            dynamic_cap=dynamic_cap,
                        )

                    # Inject rollover resume directive with incomplete requirements from completion gate
                    incomplete_reqs: list[str] = []
                    try:
                        gate_input_snapshot = gate_input  # type: ignore[possibly-undefined]
                    except NameError:
                        gate_input_snapshot = None
                    if gate_input_snapshot is not None:
                        rollover_gate_result = evaluate_completion_gate(gate_input_snapshot)
                        for req in rollover_gate_result.requirements:
                            if req.required and req.status != "done":
                                detail = req.failure_reason or req.description
                                incomplete_reqs.append(f"{req.id}: {detail}")

                    if incomplete_reqs:
                        reqs_str = ", ".join(incomplete_reqs)
                        resume_msg = (
                            f"[Context rollover at step {self._step}] Previous context was compacted. "
                            f"Incomplete requirements: [{reqs_str}]. "
                            f"{len(self._files_written)} files modified. "
                            f"Continue from where you left off, focusing on the incomplete requirements."
                        )
                    else:
                        resume_msg = (
                            f"[Context rollover at step {self._step}] Previous context was compacted. "
                            f"{len(self._files_written)} files modified, "
                            f"{self._stall.consecutive_no_progress} stall count. "
                            f"Continue from where you left off."
                        )
                    messages = self._inject_system_message(messages, resume_msg)

                    # Cognitive cache partial clear on rollover (#2c)
                    if cache and hasattr(cache, "clear_older_than"):
                        cache.clear_older_than(keep_count)
                    elif cache and hasattr(cache, "access_order") and hasattr(cache, "entries"):
                        # Manual partial clear: remove entries not accessed recently
                        if len(cache.access_order) > keep_count:
                            stale_keys = cache.access_order[:-keep_count]
                            for key in stale_keys:
                                cache.entries.pop(key, None)
                            cache.access_order = cache.access_order[-keep_count:]
                            log.debug("cognitive_cache_partial_clear", removed=len(stale_keys))

                    self._rollover_phase_done.add(2)

                if rollover_phase >= 3 and 3 not in self._rollover_phase_done:
                    # Phase 3 (90%): Emergency rollover - aggressive compaction
                    if messages and len(messages) > 4:
                        # Record originals before emergency compaction (rehydration)
                        if self._rehydration_recorder is not None:
                            to_compact = messages[:-3] if len(messages) > 3 else []
                            if to_compact:
                                try:
                                    raw = [
                                        m if isinstance(m, dict) else m.model_dump()
                                        for m in to_compact
                                    ]
                                    await self._rehydration_recorder.record(
                                        raw,
                                        step_range=(max(0, self._step - len(raw)), self._step),
                                        activity_phase=self._activity_phase,
                                        compaction_event="phase_3_emergency",
                                    )
                                except Exception as exc:
                                    log.debug("rehydration_record_p3_failed", error=str(exc)[:100])
                        summary = f"[Emergency rollover] Context compacted at step {self._step}. Goal: '{goal[:100]}'"
                        turns = _group_into_turns(messages)
                        kept = turns[-3:] if len(turns) > 3 else turns
                        messages = [{"role": "system", "content": summary}]
                        for t in kept:
                            messages.extend(t.messages)
                        messages = _validate_tool_pairs(messages)
                        log.warning(
                            "emergency_rollover", step=self._step, phase=3, msg_count=len(messages)
                        )
                    self._rollover_phase_done.add(3)

                # Wind-down state machine is handled by _prepare_step (#14)
                # (no more goal-text modification - nudges are system messages)

                # -- stall tracking --
                output_text = ""
                try:
                    out = await result.get_output()
                    output_text = str(out) if out else collected_text
                except Exception:
                    output_text = collected_text

                if output_text:
                    self._last_answer_text = output_text
                    self._stall.mark_activity("agent_step")
                else:
                    self._stall.mark_no_progress()

                # -- Rehydration trigger (per-step, fail-safe) --
                if self._rehydration_trigger is not None:
                    try:
                        from rune.agent.rehydration import format_injection, rehydrate

                        view = self._make_loop_state_view(goal)
                        decision = self._rehydration_trigger.evaluate(view)
                        if decision.fired and decision.reading is not None:
                            results = await rehydrate(
                                view,
                                decision.reading,
                                self._session_id or "",
                                k=3,
                            )
                            if results:
                                injection = format_injection(results, decision.reading)
                                if injection:
                                    messages = self._inject_system_message(messages, injection)
                                    log.info(
                                        "rehydration_injected",
                                        step=self._step,
                                        signal=decision.reading.name,
                                        results=len(results),
                                    )
                    except Exception as exc:
                        log.debug("rehydration_trigger_failed", error=str(exc)[:100])

                # Pre-compute evidence total for reuse below
                total_evidence = (
                    evidence.reads
                    + evidence.writes
                    + evidence.executions
                    + evidence.web_searches
                    + evidence.web_fetches
                    + evidence.browser_reads
                )

                # Progress detection: break if no new *actionable* evidence
                # AND no new text for 3 consecutive steps.
                # browser_reads are excluded because observe/find loops
                # inflate the counter without producing useful output.
                actionable_evidence = (
                    evidence.reads
                    + evidence.writes
                    + evidence.executions
                    + evidence.web_searches
                    + evidence.web_fetches
                )
                new_evidence = actionable_evidence - _prev_evidence_total
                _prev_evidence_total = actionable_evidence
                has_new_output = bool(output_text and output_text.strip())

                if new_evidence == 0 and not has_new_output:
                    _no_new_evidence_steps += 1
                    # If prior steps already produced evidence (tool calls succeeded)
                    # the LLM may just need more turns to
                    # formulate its answer.  Allow extra patience.
                    threshold = 5 if actionable_evidence > 0 else 3
                    if _no_new_evidence_steps >= threshold:
                        log.warning("no_progress_break", step=self._step, evidence=total_evidence)
                        trace.reason = "no_progress"
                        trace.final_step = self._step
                        break
                else:
                    _no_new_evidence_steps = 0

                # final answer detection
                # stream_text() internally handles tool calls and re-calls the LLM.
                # Once it returns, the collected text IS the final answer.
                # For code_modify/execution tasks, require writes or executions
                # (not just reads) before considering the task done.
                needs_action = classification.goal_type in (
                    "code_modify",
                    "execution",
                    "full",
                )
                action_evidence = evidence.writes + evidence.executions
                # When action evidence exists (tools ran successfully),
                # accept shorter answers — the result speaks for itself.
                min_answer_len = 20 if action_evidence > 0 else 50
                if output_text and len(output_text.strip()) > min_answer_len and total_evidence > 0:
                    if needs_action and action_evidence == 0:
                        # LLM read files but didn't write/execute yet — nudge it
                        messages = self._inject_system_message(
                            messages,
                            "[System] You have read the files but not made any changes yet. "
                            "Proceed to edit/write/execute now. Do not just describe what you plan to do.",
                        )
                    elif (
                        verify_freshness_enabled
                        and self._last_code_write_step > self._last_verify_step
                    ):
                        # R19: code modified but not re-run
                        log.info(
                            "verify_freshness_fastpath_block",
                            step=self._step,
                            last_write=self._last_code_write_step,
                            last_verify=self._last_verify_step,
                        )
                        messages = self._inject_system_message(
                            messages,
                            f"[Completion Gate] You modified a code file (tool call "
                            f"#{self._last_code_write_step}) but did not run "
                            f"bash_execute after that (last verify call "
                            f"#{self._last_verify_step}). Re-run the script and "
                            f"show the real output before declaring done.",
                        )
                    elif _unverified_code():
                        # Code changed but nothing verifies it yet. Don't take the
                        # model's word — nudge it to get tests green; if it can't,
                        # stop honestly (max_gate_blocked -> /escalate).
                        self._unverified_completion_blocks += 1
                        log.info(
                            "require_test_pass_block",
                            step=self._step,
                            last_write=self._last_code_write_step,
                            last_verify=self._last_verify_step,
                            verify_failed=self._last_verify_failed,
                            count=self._unverified_completion_blocks,
                        )
                        if self._unverified_completion_blocks >= 5:
                            log.warning("require_test_pass_exhausted", step=self._step)
                            trace.reason = self._max_gate_reason()
                            trace.final_step = self._step
                            break
                        messages = self._inject_system_message(
                            messages,
                            "[Completion Gate] You changed code (tool call "
                            f"#{self._last_code_write_step}) but no test/verification "
                            "command has PASSED since that change"
                            + (
                                " — your last test run FAILED."
                                if self._last_verify_failed
                                else "."
                            )
                            + " Run the tests now and make them pass before declaring"
                            " done. If they genuinely cannot pass, say so explicitly"
                            " and do NOT claim the task is complete.",
                        )
                    else:
                        blocker = await self._benchmark_completion_blocker()
                        if blocker:
                            log.info("benchmark_failed_tool_fastpath_block", step=self._step)
                            messages = self._inject_system_message(messages, blocker)
                        else:
                            # Post-edit auto-verify before fast-path completion.
                            _avok, messages, _gate_blocked_count = await self._auto_verify_gate(
                                messages, _gate_blocked_count
                            )
                            if not _avok:
                                if _gate_blocked_count >= 5:
                                    log.warning(
                                        "max_gate_blocked",
                                        step=self._step,
                                        count=_gate_blocked_count,
                                    )
                                    trace.reason = self._max_gate_reason()
                                    trace.final_step = self._step
                                    break
                                continue
                            # Requirement gate (opt-in, default off).
                            _rqok, messages, _gate_blocked_count = await self._finalize_gates(
                                messages, _gate_blocked_count
                            )
                            if not _rqok:
                                if _gate_blocked_count >= 5:
                                    log.warning(
                                        "max_gate_blocked",
                                        step=self._step,
                                        count=_gate_blocked_count,
                                    )
                                    trace.reason = self._max_gate_reason()
                                    trace.final_step = self._step
                                    break
                                continue
                            log.info(
                                "final_answer_detected",
                                step=self._step,
                                text_len=len(output_text),
                                evidence=total_evidence,
                            )
                            trace.reason = "completed"
                            trace.final_step = self._step
                            break

                # completion gate (full 18-requirement integration)
                intent_contract = resolve_intent_contract(classification, classification.confidence)

                # Determine requirements from IntentContract
                effective_tool_req = intent_contract.tool_requirement
                effective_output_exp = intent_contract.output_expectation
                requires_grounding = intent_contract.grounding_requirement == "required"
                requires_code_verification = intent_contract.requires_code_verification
                requires_code_write = getattr(
                    intent_contract, "requires_code_write_artifact", False
                )

                # If IntentContract says no tools needed and LLM answered
                # without tools, it's a text-only response - mark as complete.
                # No tools used + text answer = done. Don't loop for short answers
                # regardless of tool_requirement — if the LLM chose to answer
                # without tools, respect that decision.
                if total_evidence == 0 and output_text and output_text.strip():
                    blocker = await self._benchmark_completion_blocker()
                    if blocker:
                        log.info("benchmark_failed_tool_text_only_block", step=self._step)
                        messages = self._inject_system_message(messages, blocker)
                    else:
                        # Requirement gate (opt-in, default off).
                        _rqok, messages, _gate_blocked_count = await self._finalize_gates(
                            messages, _gate_blocked_count
                        )
                        if not _rqok:
                            if _gate_blocked_count >= 5:
                                log.warning(
                                    "max_gate_blocked", step=self._step, count=_gate_blocked_count
                                )
                                trace.reason = self._max_gate_reason()
                                trace.final_step = self._step
                                break
                            continue
                        log.info("text_only_complete", step=self._step, text_len=len(output_text))
                        trace.reason = "completed"
                        trace.final_step = self._step
                        break

                if (
                    total_evidence == 0
                    and output_text
                    and intent_contract.tool_requirement == "none"
                ):
                    effective_tool_req = "none"
                    effective_output_exp = "text"
                    requires_grounding = False

                # If this step produced text WITHOUT any tool calls, the LLM
                # has finished its final answer.  Treat as complete to prevent
                # re-generation loops.
                step_had_tool_calls = any(
                    isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
                    for m in (
                        result.all_messages()[-3:]
                        if len(result.all_messages()) > 3
                        else result.all_messages()
                    )
                )
                if output_text and not step_had_tool_calls and total_evidence > 0:
                    effective_tool_req = "none"
                    effective_output_exp = "text"

                # Calculate analysis depth minimums based on goal type
                analysis_min_reads = 0
                min_web_searches = 0
                min_web_fetches = 0
                if classification.goal_type in ("research",):
                    # Complex research needs deeper analysis; simple lookups need less
                    if getattr(classification, "is_multi_task", False) or getattr(
                        classification, "is_complex_coding", False
                    ):
                        analysis_min_reads = 3
                    else:
                        analysis_min_reads = 1
                elif classification.goal_type in ("code_modify", "execution"):
                    analysis_min_reads = 1
                if intent_contract.grounding_requirement == "required":
                    min_web_searches = 1
                    # Fast lane: a fresh search counts as grounding (R14
                    # passes on search or fetch; R18 reads these minimums).
                    min_web_fetches = 0 if _fast_lane_active else 1

                gate_input = CompletionGateInput(
                    intent_resolved=bool(output_text),
                    tool_requirement=effective_tool_req,
                    output_expectation=effective_output_exp,
                    evidence=evidence,
                    changed_files_count=evidence.writes,
                    answer_length=len(output_text),
                    # Code verification
                    requires_code_verification=requires_code_verification,
                    # Code write artifact
                    requires_code_write_artifact=requires_code_write,
                    # Grounding
                    grounding_requirement=requires_grounding,
                    # Analysis depth
                    analysis_depth_min_reads=analysis_min_reads,
                    # Web evidence
                    min_web_searches=min_web_searches,
                    min_web_fetches=min_web_fetches,
                    # R19: verification freshness (opt-in via env var)
                    verify_freshness_enabled=verify_freshness_enabled,
                    last_code_write_step=self._last_code_write_step,
                    last_verify_step=self._last_verify_step,
                    # Hard failures tracked (deduplicated) (#16)
                    hard_failures=list(self._hard_failures),
                )
                gate_result = evaluate_completion_gate(gate_input)

                # Same check here — the fast path isn't the only way to finish, so
                # the full gate must not wave through unverified code either.
                if _unverified_code():
                    self._unverified_completion_blocks += 1
                    log.info(
                        "require_test_pass_block_fullgate",
                        step=self._step,
                        last_write=self._last_code_write_step,
                        last_verify=self._last_verify_step,
                        verify_failed=self._last_verify_failed,
                        count=self._unverified_completion_blocks,
                    )
                    if self._unverified_completion_blocks >= 5:
                        log.warning("require_test_pass_exhausted", step=self._step)
                        trace.reason = self._max_gate_reason()
                        trace.final_step = self._step
                        break
                    messages = self._inject_system_message(
                        messages,
                        "[Completion Gate] No test/verification command has PASSED "
                        f"since you last changed code (tool call #{self._last_code_write_step})"
                        + (
                            " and your last test run FAILED."
                            if self._last_verify_failed
                            else "."
                        )
                        + " Run the tests and make them pass before declaring done.",
                    )
                    continue

                if gate_result.outcome == "verified":
                    blocker = await self._benchmark_completion_blocker()
                    if blocker:
                        log.info("benchmark_failed_tool_verified_block", step=self._step)
                        messages = self._inject_system_message(messages, blocker)
                    else:
                        # Post-edit auto-verify, then Evidence Gate, before final.
                        _avok, messages, _gate_blocked_count = await self._auto_verify_gate(
                            messages, _gate_blocked_count
                        )
                        if not _avok:
                            if _gate_blocked_count >= 5:
                                log.warning(
                                    "max_gate_blocked", step=self._step, count=_gate_blocked_count
                                )
                                trace.reason = self._max_gate_reason()
                                trace.final_step = self._step
                                break
                            continue
                        # Behavior done != output correct. If the Evidence Gate
                        # is active, re-verify before finalizing: a fail injects
                        # evidence and continues; skip/off finalizes as before.
                        _ev_state, _ev_msg = await self._evidence_verdict()
                        if _ev_state == "fail" and _ev_msg:
                            log.info("evidence_gate_block_on_verified", step=self._step)
                            messages = self._inject_system_message(messages, _ev_msg)
                            _gate_blocked_count += 1
                            self._gate_blocked_count = _gate_blocked_count
                            if _gate_blocked_count >= 5:
                                log.warning(
                                    "max_gate_blocked", step=self._step, count=_gate_blocked_count
                                )
                                trace.reason = self._max_gate_reason()
                                trace.final_step = self._step
                                break
                        else:
                            # Requirement gate (opt-in, default off).
                            _rqok, messages, _gate_blocked_count = await self._finalize_gates(
                                messages, _gate_blocked_count
                            )
                            if not _rqok:
                                if _gate_blocked_count >= 5:
                                    log.warning(
                                        "max_gate_blocked",
                                        step=self._step,
                                        count=_gate_blocked_count,
                                    )
                                    trace.reason = self._max_gate_reason()
                                    trace.final_step = self._step
                                    break
                                continue
                            trace.reason = "completed"
                            trace.final_step = self._step
                            trace.evidence_score = 1.0
                            break

                # If blocked, let the loop continue — but limit repeats
                # to prevent infinite loops when requirements can't be met.
                if gate_result.outcome == "blocked":
                    # Evidence Gate override: behavioral requirements are weaker
                    # than a passing outcome check. If the produced artifact
                    # actually satisfies the task's own success criteria, accept
                    # completion even though some behavioral requirement is unmet
                    # (e.g. "not enough reads"). A failing check instead injects
                    # first-mismatch evidence and keeps the loop going.
                    _avok, messages, _gate_blocked_count = await self._auto_verify_gate(
                        messages, _gate_blocked_count
                    )
                    if not _avok:
                        if _gate_blocked_count >= 5:
                            log.warning(
                                "max_gate_blocked", step=self._step, count=_gate_blocked_count
                            )
                            trace.reason = self._max_gate_reason()
                            trace.final_step = self._step
                            break
                        continue
                    _ev_state, _ev_msg = await self._evidence_verdict()
                    if _ev_state == "pass":
                        log.info("evidence_gate_override_blocked", step=self._step)
                        trace.reason = "completed"
                        trace.final_step = self._step
                        trace.evidence_score = 1.0
                        break
                    if _ev_state == "fail" and _ev_msg:
                        log.info("evidence_gate_block", step=self._step)
                        messages = self._inject_system_message(messages, _ev_msg)

                    _gate_blocked_count += 1
                    self._gate_blocked_count = _gate_blocked_count
                    log.debug(
                        "completion_gate_blocked",
                        step=self._step,
                        count=_gate_blocked_count,
                        missing=gate_result.missing_requirement_ids,
                    )
                    # Inject missing requirements so the agent knows
                    # exactly what to do next instead of retrying blindly.
                    _missing = [
                        f"{r.id}: {r.failure_reason or r.description}"
                        for r in gate_result.requirements
                        if r.required and r.status != "done"
                    ]
                    if _missing:
                        messages = self._inject_system_message(
                            messages,
                            "[Completion Gate] Requirements not met: "
                            + ", ".join(_missing)
                            + ". Focus on completing these before finishing.",
                        )
                    # H1: advisor at gate_blocked boundary (skipped for native path)
                    if advisor_service.enabled and not _native_cfg.enabled:
                        messages, _adv_dec = await maybe_consult(
                            advisor_service,
                            policy_input=_make_policy_input(),
                            build_request=lambda trg, _gr=gate_result: _make_advisor_request(
                                trg,
                                gate_result=_gr,
                            ),
                            messages=messages,
                            inject=self._inject_system_message,
                        )
                        if _adv_dec and _adv_dec.raw_text:
                            _last_advisor_raw = _adv_dec.raw_text[:300]
                        if _adv_dec and _adv_dec.action == "abort":
                            log.warning("advisor_recommended_abort", step=self._step)
                            trace.reason = "advisor_abort"
                            trace.final_step = self._step
                            break
                        if _adv_dec and _adv_dec.plan_steps:
                            # Deferred: full reset only after compliance verdict.
                            # Not while the lane is active — the decrement would
                            # hold the count below the upshift threshold.
                            if not _fast_lane_active:
                                _gate_blocked_count = max(0, _gate_blocked_count - 1)
                            _ev_total = (
                                evidence.reads
                                + evidence.writes
                                + evidence.executions
                                + evidence.web_fetches
                            )
                            _pending_advice = build_pending(
                                _adv_dec,
                                self._step,
                                _ev_total,
                                len(self._hard_failures),
                                call_history_index=len(advisor_service.budget.call_history) - 1,
                                advice_mode=advisor_service.mode,
                            )
                    if _gate_blocked_count >= 5:
                        # Last chance before giving up: if the artifact now
                        # actually satisfies the task's success criteria, accept
                        # it. The correct artifact is often written on the final
                        # iteration, after which no further finalize attempt
                        # would re-run the Evidence Gate.
                        # Auto-verify last: a fail at the block cap means broken
                        # code, so give up rather than retry.
                        _avok, _, _ = await self._auto_verify_gate(messages, _gate_blocked_count)
                        if not _avok:
                            trace.reason = self._max_gate_reason()
                            trace.final_step = self._step
                            break
                        _ev_state2, _ = await self._evidence_verdict()
                        if _ev_state2 == "pass":
                            log.info("evidence_gate_override_maxblocked", step=self._step)
                            trace.reason = "completed"
                            trace.final_step = self._step
                            trace.evidence_score = 1.0
                            break
                        log.warning("max_gate_blocked", step=self._step, count=_gate_blocked_count)
                        trace.reason = self._max_gate_reason()
                        trace.final_step = self._step
                        break

                # "partial" with substantial output - treat as completed to
                # prevent indefinite loops when only minor requirements remain.
                # Exception: if code write was required but nothing was written,
                # do NOT accept partial — force the agent to keep working.
                if (
                    gate_result.outcome == "partial"
                    and output_text
                    and len(output_text.strip()) > 50
                ):
                    if requires_code_write and gate_input.structured_write_count == 0:
                        log.debug(
                            "partial_rejected_no_write",
                            step=self._step,
                            writes=gate_input.structured_write_count,
                        )
                        messages = self._inject_system_message(
                            messages,
                            "[Completion Gate] Code write required but no files written yet. "
                            "You must write/edit code files to complete this task.",
                        )
                    else:
                        blocker = await self._benchmark_completion_blocker()
                        if blocker:
                            log.info("benchmark_failed_tool_partial_block", step=self._step)
                            messages = self._inject_system_message(messages, blocker)
                        else:
                            _avok, messages, _gate_blocked_count = await self._auto_verify_gate(
                                messages, _gate_blocked_count
                            )
                            if not _avok:
                                if _gate_blocked_count >= 5:
                                    log.warning(
                                        "max_gate_blocked",
                                        step=self._step,
                                        count=_gate_blocked_count,
                                    )
                                    trace.reason = self._max_gate_reason()
                                    trace.final_step = self._step
                                    break
                                continue
                            log.info("partial_completion_accepted", step=self._step)
                            trace.reason = "completed"
                            trace.final_step = self._step
                            trace.evidence_score = 0.8
                            break

            except Exception as exc:
                reason = classify_error(exc)
                log.warning(
                    "agent_step_error",
                    step=self._step,
                    error=str(exc)[:300],
                    reason=reason,
                )

                failover_result = await failover.handle_error(exc)

                if failover_result.success:
                    # Provider instability outranks the latency optimization:
                    # drop the fast lane so the recovered profile runs with
                    # full gate rigor and uncapped fetches.
                    if _fast_lane_active:
                        _fast_lane_active = False
                        log.info("fast_lane_deactivated_on_failover")
                    # Failover recovered - rebuild agent if profile changed
                    new_profile = failover.current_profile
                    new_model = (
                        loop_model_string(new_profile.provider, new_profile.model)
                        if new_profile.provider != "none"
                        else model
                    )

                    if new_model != model:
                        model = new_model
                        agent = LiteLLMAgent(
                            model=model,
                            system_prompt=system_prompt,
                            tools=list(tool_functions.values()),
                        )
                        log.info("agent_model_switched", model=model)

                    # If compaction was signalled, compact messages
                    if failover_result.reason == "context_overflow" and messages:
                        # Record originals before overflow compaction (rehydration)
                        if self._rehydration_recorder is not None:
                            to_compact = messages[:-10] if len(messages) > 10 else []
                            if to_compact:
                                try:
                                    raw = [
                                        m if isinstance(m, dict) else m.model_dump()
                                        for m in to_compact
                                    ]
                                    await self._rehydration_recorder.record(
                                        raw,
                                        step_range=(max(0, self._step - len(raw)), self._step),
                                        activity_phase=self._activity_phase,
                                        compaction_event="context_overflow",
                                    )
                                except Exception as exc_r:
                                    log.debug(
                                        "rehydration_record_overflow_failed", error=str(exc_r)[:100]
                                    )

                        summary = f"(Summary of prior conversation compacted at step {self._step})"
                        turns = _group_into_turns(messages)
                        kept = turns[-10:] if len(turns) > 10 else turns
                        messages = [{"role": "system", "content": summary}]
                        for t in kept:
                            messages.extend(t.messages)
                        messages = _validate_tool_pairs(messages)
                        log.info("messages_compacted_after_overflow")

                    continue
                else:
                    trace.reason = f"error: {exc}"
                    break

        # If loop exhausted without explicit reason, mark as max_iterations
        if not trace.reason:
            trace.reason = "max_iterations"
            trace.final_step = self._step

        # Flush rehydration recorder on session end
        if self._rehydration_recorder is not None:
            try:
                await self._rehydration_recorder.flush()
            except Exception:
                pass

        # Persist advisor events (merge native path events first)
        try:
            _native_events = agent.native_advisor_events() if agent else []
            if _native_events:
                advisor_service.budget.call_history.extend(_native_events)
                advisor_service.budget.calls_used += len(_native_events)
                advisor_service.budget.tokens_used += sum(
                    int(e.get("output_tokens") or 0) for e in _native_events
                )
        except Exception as exc:
            log.warning(
                "advisor_native_events_merge_failed",
                error=str(exc)[:200],
            )
        try:
            import os as _os_persist

            _persist_env = _os_persist.environ.get("RUNE_ADVISOR_PERSIST", "").strip().lower()
            _persist_enabled = _persist_env not in ("0", "false", "no", "off")
            if _persist_enabled and advisor_service.budget.calls_used > 0 and self._session_id:
                from rune.memory.store import get_memory_store

                _store = get_memory_store()
                for entry in advisor_service.budget.call_history:
                    _store.log_advisor_event(
                        session_id=self._session_id,
                        trigger=entry.get("trigger", "") or "",
                        action=entry.get("action", "") or "",
                        provider=entry.get("provider", "") or "",
                        model=entry.get("model", "") or "",
                        output_tokens=int(entry.get("output_tokens") or 0),
                        latency_ms=int(entry.get("latency_ms") or 0),
                        plan_injected=bool(entry.get("plan_injected")),
                        stuck_reason=entry.get("stuck_reason", "") or "",
                        advice_mode=entry.get("advice_mode", "") or "",
                        compliance_verdict=entry.get("compliance_verdict", "") or "",
                        expected_tool=entry.get("expected_tool", "") or "",
                        observed_tools=entry.get("observed_tools", "") or "",
                        step_at_call=int(entry.get("step_at_call") or 0),
                        pre_call_evidence=int(entry.get("pre_call_evidence") or 0),
                        post_call_evidence=int(entry.get("post_call_evidence") or 0),
                    )
                _store.update_advisor_outcome(
                    session_id=self._session_id,
                    outcome=trace.reason or "",
                )
                log.info(
                    "advisor_events_persisted",
                    session_id=self._session_id,
                    count=advisor_service.budget.calls_used,
                    outcome=trace.reason,
                )
        except Exception as exc:
            log.warning(
                "advisor_events_persist_failed",
                error=str(exc)[:200],
            )

        trace.total_tokens_used = self._token_budget.used
        return trace

    def _compute_pinned_steps(self, messages: list, full_window: int) -> set[int]:
        """Find older steps that are referenced by recent steps (dependency pinning).

        If step 10 reads a file that was written in step 3, step 3 is "pinned"
        and kept at full detail to preserve context the model still depends on.
        """
        pinned: set[int] = set()
        recent_start = max(0, len(messages) - full_window)

        # Collect file paths mentioned in recent messages
        recent_paths: set[str] = set()
        for msg in messages[recent_start:]:
            recent_paths.update(self._extract_paths_from_message(msg))

        # Find older messages that created/modified those paths
        for i, msg in enumerate(messages[:recent_start]):
            if i in pinned:
                continue
            msg_paths = self._extract_paths_from_message(msg)
            if msg_paths & recent_paths:
                pinned.add(i)

        return pinned

    @staticmethod
    def _extract_paths_from_message(msg: Any) -> set[str]:
        """Extract file paths from a message's tool calls."""
        paths: set[str] = set()
        text = str(msg) if not isinstance(msg, dict) else str(msg.get("content", ""))
        for match in re.finditer(r'["\']?(/[\w./\-]+\.\w+)["\']?', text):
            paths.add(match.group(1))
        return paths

    def _is_exploration_only(self, msg: Any, idx: int, messages: list[Any]) -> bool:
        """Detect if a file_read was exploratory (never referenced later).

        A file_read is exploratory if the content it returned was never used
        in any subsequent tool call arguments.
        """
        # Only applies to tool results that look like file reads
        text = str(msg)
        if "file_read" not in text:
            return False

        paths = self._extract_paths_from_message(msg)
        if not paths:
            return False

        # Check if any subsequent message references those paths
        for later_msg in messages[idx + 1 :]:
            later_text = str(later_msg)
            for p in paths:
                if p in later_text:
                    return False
        return True

    def _mask_observations(self, messages: list[Any]) -> list[Any]:
        """3-tier observation masking for context efficiency.

        Uses phase-adaptive windows (#12):
        - Recent (last full_window): Keep full content
        - Middle (next truncate_window): head+tail truncation
        - Older: 1-line tool-specific summary

        Pinned steps (dependency graph) are kept at full detail regardless
        of their position. Exploration-only reads are forced to Tier 3.

        Phases: exploration, implementation, verification, research
        """
        # Phase-adaptive windows (#12)
        phase = getattr(self, "_activity_phase", "exploration")
        full_window, truncate_window = _PHASE_WINDOWS.get(
            phase, (FULL_WINDOW_MAX, TRUNCATE_WINDOW_MAX)
        )

        if len(messages) <= full_window:
            return messages

        # Compute pinned steps (dependency graph)
        pinned = self._compute_pinned_steps(messages, full_window)

        result: list[Any] = []
        total = len(messages)
        full_start = max(0, total - full_window)
        truncate_start = max(0, full_start - truncate_window)

        # Tier 3: Old messages -> 1-line tool-specific summary
        for i, msg in enumerate(messages[:truncate_start]):
            if i in pinned:
                result.append(msg)  # Pinned: keep full detail
            else:
                result.append(self._summarize_message(msg))

        # Tier 2: Middle messages -> head+tail truncation
        for i, msg in enumerate(messages[truncate_start:full_start], start=truncate_start):
            if i in pinned:
                result.append(msg)  # Pinned: keep full detail
            elif self._is_exploration_only(msg, i, messages):
                # Context pollution prevention (#1c): exploratory reads -> Tier 3
                result.append(self._summarize_message(msg))
            else:
                result.append(self._truncate_message(msg))

        # Tier 1: Recent messages -> full content
        for msg in messages[full_start:]:
            result.append(msg)

        # Context pollution prevention (#1c): remove empty messages
        # Preserve tool messages even if empty — removing them breaks assistant+tool pairs
        result = [m for m in result if not self._is_empty_message(m) or _msg_role(m) == "tool"]

        return result

    @staticmethod
    def _is_empty_message(msg: Any) -> bool:
        """Check if a message became empty after summarization/truncation."""
        try:
            if hasattr(msg, "parts"):
                if not msg.parts:
                    return True
                # Check if all parts are empty
                for part in msg.parts:
                    if hasattr(part, "content") and part.content:
                        return False
                    if hasattr(part, "text") and part.text:
                        return False
                    if hasattr(part, "tool_name"):
                        return False  # tool call parts are not empty
                    if hasattr(part, "image") and part.image:
                        return False
                return True
            if isinstance(msg, dict):
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])
                return not content and not tool_calls
            return False
        except Exception:
            return False

    @staticmethod
    def _summarize_message(msg: Any) -> Any:
        """Compress a message to a 1-line tool-specific summary (#1a).

        Handles 12 tool types with specific summary formats:
        - file_read -> "Read {path} ({n} lines)"
        - file_write -> "Wrote {path}"
        - file_edit -> "Edited {path}"
        - file_delete -> "Deleted {path}"
        - file_search -> "Searched for '{query}' -- {n} matches"
        - bash_execute -> "Ran `{cmd[:80]}` -> {exit_code}"
        - web_search -> "Searched: '{query}' -- {n} results"
        - web_fetch -> "Fetched {url[:60]} ({n} chars)"
        - think -> "Thought about: {thought[:60]}"
        - ask_user -> "Asked user: {question[:60]}"
        - browser_navigate -> "Navigated to {url[:60]}"
        - delegate -> "Delegated: {task[:60]}"
        - Generic -> first 100 chars
        """
        try:
            if hasattr(msg, "parts"):
                # PydanticAI message with parts
                text = ""
                tool_name = ""
                tool_args: dict[str, Any] = {}
                for part in msg.parts:
                    if hasattr(part, "tool_name"):
                        tool_name = part.tool_name
                    if hasattr(part, "args") and isinstance(part.args, dict):
                        tool_args = part.args
                    if hasattr(part, "content") and isinstance(part.content, str):
                        text = part.content
                        break

                if not text and not tool_name:
                    return msg

                # Tool-specific summarization (#1a - 12 types)
                summary = ""
                if tool_name == "file_read":
                    path = tool_args.get("file_path") or tool_args.get("path", "")
                    lines = text.count("\n") + 1 if text else 0
                    summary = f"[Summary] Read {path or 'file'} ({lines} lines)"
                elif tool_name == "file_write":
                    path = tool_args.get("file_path") or tool_args.get("path", "")
                    summary = f"[Summary] Wrote {path or 'file'}"
                elif tool_name == "file_edit":
                    path = tool_args.get("file_path") or tool_args.get("path", "")
                    summary = f"[Summary] Edited {path or 'file'}"
                elif tool_name == "file_delete":
                    path = tool_args.get("file_path") or tool_args.get("path", "")
                    summary = f"[Summary] Deleted {path or 'file'}"
                elif tool_name == "file_search":
                    query = tool_args.get("pattern") or tool_args.get("query", "")
                    matches = text.count("\n") + 1 if text else 0
                    summary = f"[Summary] Searched for '{query}' -- {matches} matches"
                elif tool_name == "bash_execute":
                    cmd = tool_args.get("command", "")
                    exit_code = tool_args.get("exit_code", 0)
                    # Try to extract exit code from text if not in args
                    if not exit_code and text:
                        import re as _re

                        m = _re.search(r"exit[_ ]code[:\s]+(\d+)", text.lower())
                        if m:
                            exit_code = int(m.group(1))
                    summary = f"[Summary] Ran `{cmd[:80]}` -> {exit_code}"
                elif tool_name == "web_search":
                    query = tool_args.get("query", "")
                    results = text.count("\n") + 1 if text else 0
                    summary = f"[Summary] Searched: '{query}' -- {results} results"
                elif tool_name == "web_fetch":
                    url = tool_args.get("url", "")
                    chars = len(text) if text else 0
                    summary = f"[Summary] Fetched {url[:60]} ({chars} chars)"
                elif tool_name == "think":
                    thought = tool_args.get("thought") or text or ""
                    summary = f"[Summary] Thought about: {thought[:60]}"
                elif tool_name == "ask_user":
                    question = tool_args.get("question") or text or ""
                    summary = f"[Summary] Asked user: {question[:60]}"
                elif tool_name == "browser_navigate":
                    url = tool_args.get("url", "")
                    summary = f"[Summary] Navigated to {url[:60]}"
                elif tool_name == "delegate":
                    task = tool_args.get("task") or text or ""
                    summary = f"[Summary] Delegated: {task[:60]}"
                else:
                    summary = f"[Summary] {(text or '')[:100]}..."

                from copy import deepcopy

                summarized = deepcopy(msg)
                for part in summarized.parts:
                    if hasattr(part, "content") and isinstance(part.content, str):
                        part.content = summary
                        break
                return summarized
            return msg
        except Exception:
            return msg

    @staticmethod
    def _truncate_message(msg: Any, limit: int = 4000, text_limit: int = 500) -> Any:
        """Truncate message content to head+tail (#1b).

        For middle-tier messages, both tool result content and plain text
        content are truncated if they exceed their respective limits.
        Assistant text content is also truncated if >500 chars.
        """
        try:
            if hasattr(msg, "parts"):
                from copy import deepcopy

                truncated = deepcopy(msg)
                for part in truncated.parts:
                    if hasattr(part, "content") and isinstance(part.content, str):
                        content = part.content
                        # Tool result content uses the larger limit
                        effective_limit = limit if hasattr(part, "tool_name") else text_limit
                        if len(content) > effective_limit:
                            head = content[: effective_limit // 2]
                            tail = content[-(effective_limit // 2) :]
                            part.content = (
                                f"{head}\n\n... [{len(content) - effective_limit} "
                                f"chars truncated] ...\n\n{tail}"
                            )
                    # Also truncate plain text parts (#1b)
                    elif hasattr(part, "text") and isinstance(getattr(part, "text", None), str):
                        text_val = part.text
                        if len(text_val) > text_limit:
                            head = text_val[: text_limit // 2]
                            tail = text_val[-(text_limit // 2) :]
                            part.text = (
                                f"{head}\n\n... [{len(text_val) - text_limit} "
                                f"chars truncated] ...\n\n{tail}"
                            )
                return truncated
            # Handle dict-based messages
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                content = msg["content"]
                role = msg.get("role", "")
                # Assistant text masking (#1b): truncate long assistant text
                if role == "assistant" and len(content) > 500:
                    content = content[:300] + "\n... (truncated) ...\n" + content[-200:]
                    return {**msg, "content": content}
                if len(content) > text_limit:
                    head = content[: text_limit // 2]
                    tail = content[-(text_limit // 2) :]
                    return {
                        **msg,
                        "content": (
                            f"{head}\n\n... [{len(content) - text_limit} "
                            f"chars truncated] ...\n\n{tail}"
                        ),
                    }
            return msg
        except Exception:
            return msg

    def _reduce_active_tools(self, tools: list[str]) -> list[str]:
        """After step 6, reduce to recently used + base tools."""
        base_tools = {
            "think",
            "file_read",
            "file_write",
            "file_edit",
            "bash_execute",
            "ask_user",
            "memory_search",
        }
        # Keep base tools + any recently used
        return [t for t in tools if t in base_tools or t == self._stall.last_tool_call]

    # Vision cache integration (#3)

    def _apply_vision_cache(self, messages: list[Any]) -> None:
        """Replace duplicate image content in dict-based messages with cached text.

        LiteLLM messages are plain dicts.  When a message contains an
        ``image_url`` content part, check the vision cache and substitute
        with text if a hit is found.
        """
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for i, part in enumerate(content):
                if not isinstance(part, dict):
                    continue
                if part.get("type") != "image_url":
                    continue
                url = (part.get("image_url") or {}).get("url", "")
                if not url:
                    continue
                try:
                    img_hash = self._vision_cache.hash_image(url.encode("utf-8"))
                    cached = self._vision_cache.get_or_none(img_hash)
                    if cached:
                        content[i] = {"type": "text", "text": cached}
                except Exception:
                    continue

    # Rehydration state view adapter

    def _make_loop_state_view(self, goal_text: str) -> Any:
        """Create a lightweight read-only view for the rehydration trigger."""
        loop = self

        class _View:
            @property
            def step(self) -> int:
                return loop._step

            @property
            def activity_phase(self) -> str:
                return loop._activity_phase

            @property
            def phase_just_changed(self) -> bool:
                return loop._prev_activity_phase != loop._activity_phase

            @property
            def stall_consecutive(self) -> int:
                return loop._stall.consecutive_no_progress

            @property
            def gate_blocked_count(self) -> int:
                return loop._gate_blocked_count

            @property
            def token_budget_fraction(self) -> float:
                return loop._token_budget.fraction

            def recent_tool_names(self, n: int) -> list[str]:
                return []

            def goal(self) -> str:
                return goal_text

        return _View()

    # Turn-atomic message compaction (#2a)

    @staticmethod
    def _compact_messages_atomic(messages: list, keep_last: int) -> list:
        """Compact keeping last N turn-atomic groups (assistant+tools)."""
        turns = _group_into_turns(messages)
        if len(turns) <= keep_last:
            return list(messages)
        kept = turns[-keep_last:]
        return [msg for turn in kept for msg in turn.messages]

    # Dynamic context cap (#2b)

    # Token caps by activity phase
    _TOKEN_CAPS: dict[str, int] = {
        "research": 80_000,
        "implementation": 60_000,
        "verification": 50_000,
        "exploration": 60_000,
    }

    def _get_dynamic_context_cap(self) -> float:
        """Return the rollover threshold based on current activity phase.

        - research: 0.75 (more conservative, need context)
        - implementation: 0.70 (standard)
        - verification: 0.65 (can afford to forget more)
        """
        phase = getattr(self, "_activity_phase", "exploration")
        if phase == "research":
            return 0.75
        if phase == "verification":
            return 0.65
        # implementation or exploration
        return 0.70

    def _get_budget_proportional_cap(self) -> int:
        """Compute context cap as 25% of remaining budget, clamped to [40_000, 80_000]."""
        remaining = self._token_budget.total - self._token_budget.used
        proportional = int(remaining * 0.25)
        return max(40_000, min(80_000, proportional))

    def _trim_to_token_cap(self, messages: list) -> list:
        """Trim messages to fit within the token cap for current phase.

        Uses turn-atomic grouping so that assistant tool-call + tool result
        pairs are never split. Pins the first message (original goal) and
        keeps the most recent ``keep_recent`` messages, then fills the
        remaining budget from newest to oldest turns.

        Ported from trimMessagesToContextCapWithMeta() in loop.ts.
        """
        phase_cap = self._TOKEN_CAPS.get(self._activity_phase, 60_000)
        budget_cap = self._get_budget_proportional_cap()
        cap = min(phase_cap, budget_cap)
        estimated = sum(self._estimate_tokens(m) for m in messages)

        if estimated <= cap:
            return messages

        keep_recent_turns = 3
        turns = _group_into_turns(messages)

        # If we don't have enough turns to trim, return as-is
        if len(turns) <= keep_recent_turns:
            return messages

        # Split into tail/head by atomic turns so assistant tool_calls are never
        # separated from their tool results.
        tail_turns = turns[-keep_recent_turns:]
        tail = [msg for turn in tail_turns for msg in turn.messages]
        tail_tokens = sum(self._estimate_tokens(m) for m in tail)

        # If tail alone exceeds cap, we can't trim - return as-is
        if tail_tokens > cap:
            return messages

        head_turns = turns[:-keep_recent_turns]

        # Pin the first message (original goal/user request)
        pinned_first = head_turns[0] if head_turns else None
        pinned_tokens = (
            sum(self._estimate_tokens(m) for m in pinned_first.messages) if pinned_first else 0
        )
        budget_for_head = cap - tail_tokens - 50  # 50 = summary placeholder
        budget_after_pin = budget_for_head - pinned_tokens

        rest_turns = head_turns[1:]

        # Compute token estimates for each turn
        for turn in rest_turns:
            turn.token_estimate = sum(self._estimate_tokens(m) for m in turn.messages)

        # Fill from newest turns backward
        head_tokens = 0
        keep_from_turn_idx = len(rest_turns)  # default: drop all turns
        for i in range(len(rest_turns) - 1, -1, -1):
            if head_tokens + rest_turns[i].token_estimate > budget_after_pin:
                break
            head_tokens += rest_turns[i].token_estimate
            keep_from_turn_idx = i

        # Flatten kept turns back into messages
        kept_msgs: list[Any] = []
        for turn in rest_turns[keep_from_turn_idx:]:
            kept_msgs.extend(turn.messages)

        dropped_count = sum(len(t.messages) for t in rest_turns[:keep_from_turn_idx])

        if dropped_count == 0:
            return messages

        # Summary placeholder for dropped messages
        summary_msg = {
            "role": "user",
            "content": (
                f"[CONTEXT TRIMMED] {dropped_count} earlier messages were "
                "omitted due to context size limits. Continue based on "
                "recent conversation."
            ),
        }

        # Reassemble: pinned first + summary + kept head turns + tail
        trimmed: list[Any] = []
        if pinned_first is not None:
            trimmed.extend(pinned_first.messages)
        trimmed.append(summary_msg)
        trimmed.extend(kept_msgs)
        trimmed.extend(tail)
        trimmed = _validate_tool_pairs(trimmed)

        new_estimate = sum(self._estimate_tokens(m) for m in trimmed)
        log.info(
            "context_cap_trimmed",
            before_tokens=estimated,
            after_tokens=new_estimate,
            dropped_count=dropped_count,
        )

        return trimmed

    @staticmethod
    def _estimate_tokens(msg: Any) -> int:
        """Rough token estimate: ~4 chars per token."""
        text = str(msg) if not isinstance(msg, dict) else str(msg.get("content", ""))
        return len(text) // 4

    # Wind-down 5-stage state machine (#14)

    def _update_wind_down_phase(self) -> None:
        """Advance the wind-down state machine based on token budget fraction."""
        f = self._token_budget.fraction
        if f >= 0.97:
            self._wind_down_phase = "hard_stop"
        elif f >= 0.90:
            self._wind_down_phase = "final"
        elif f >= 0.75:
            self._wind_down_phase = "stopping"
        elif f >= 0.60:
            self._wind_down_phase = "wrapping"
        else:
            self._wind_down_phase = "none"

    def _get_wind_down_message(self) -> str | None:
        """Return a system nudge message for the current wind-down phase, or None."""
        match self._wind_down_phase:
            case "wrapping":
                return (
                    "[SYSTEM] You are at 60% of your token budget. "
                    "Consider wrapping up soon and ensuring your work is saved."
                )
            case "stopping":
                return (
                    "[SYSTEM] You are at 75% of your token budget. "
                    "Please complete your current subtask and provide a summary. "
                    "Do not start new investigations."
                )
            case "final":
                return (
                    "[SYSTEM] You are at 90% of your token budget. "
                    "Finish immediately. Only essential tool calls are permitted."
                )
            case "hard_stop":
                return (
                    "[SYSTEM] Token budget nearly exhausted (97%+). "
                    "You must stop now and output your final answer."
                )
            case _:
                return None

    # Prepare step hook (#25)

    def _prepare_step(
        self,
        messages: list[Any],
        workspace_root: str = "",
    ) -> list[Any]:
        """Pre-step hook: mask observations, inject nudges, check guards.

        Called before each agent iteration. Returns the prepared message list.
        """
        # 1. Observation masking (with phase-adaptive windows)
        messages = self._mask_observations(messages)

        # 2. Wind-down nudge as a system message
        self._update_wind_down_phase()
        nudge_msg = self._get_wind_down_message()
        if nudge_msg:
            messages = self._inject_system_message(messages, nudge_msg)

        # 3. Benchmark failure context
        if _env_flag("RUNE_BENCH_CAPTURE_FAILED_TOOL_OUTPUT") and self._recent_failed_tool_nudge:
            messages = self._inject_system_message(messages, self._recent_failed_tool_nudge)

        # 4. Stall guidance
        stall_guidance = self._get_stall_guidance()
        if stall_guidance:
            messages = self._inject_system_message(messages, stall_guidance)

        # 5. Workspace guard
        if workspace_root:
            # Check last tool call if it was bash
            if self._stall.last_tool_call == "bash_execute":
                warning = self._check_workspace_scope(
                    self._stall.bash_stalled_intent, workspace_root
                )
                if warning:
                    messages = self._inject_system_message(messages, warning)

        # 6. Execution nudges
        nudges = self._get_execution_nudges()
        for n in nudges:
            messages = self._inject_system_message(messages, n)

        # 7. Knowledge inventory at milestone steps (#5)
        inventory = self._maybe_inject_knowledge_inventory()
        if inventory:
            messages = self._inject_system_message(messages, inventory)

        # 8. Vision cache integration (#3): replace duplicate images with text
        self._apply_vision_cache(messages)

        # 9. Token-based trimming (#H4): trim messages to fit within phase cap
        messages = self._trim_to_token_cap(messages)

        return messages

    @staticmethod
    def _inject_system_message(messages: list[Any], text: str) -> list[Any]:
        """Append a system-level nudge as a user message.

        Previously used role="system", but litellm_adapter skips system messages
        from history (to avoid duplicating the main system prompt). Using a user
        message with <system-reminder> tags ensures the nudge actually reaches
        the model while being recognized as system-level guidance.

        Nudges include: wind-down budget warnings, stall guidance, workspace
        warnings, knowledge inventory, failed-tool hints.
        """
        messages = list(messages)  # shallow copy
        messages.append({
            "role": "user",
            "content": f"<system-reminder>\n{text}\n</system-reminder>",
        })
        return messages

    # Cognitive cache knowledge inventory (#5)

    _INVENTORY_STEPS = {5, 15, 30, 50, 80}

    def _maybe_inject_knowledge_inventory(self) -> str | None:
        """At specific step milestones, generate a summary of cached knowledge."""
        if self._step not in self._INVENTORY_STEPS:
            return None
        cache = getattr(self, "_cognitive_cache", None)
        if not cache:
            return None
        stats = cache.stats() if hasattr(cache, "stats") else None
        if not stats:
            return None
        return (
            f"[Knowledge inventory at step {self._step}]: "
            f"{stats.get('entries', 0)} cached results, "
            f"{stats.get('hits', 0)} cache hits"
        )

    # Stall guidance (#25)

    def _get_stall_guidance(self) -> str | None:
        """Generate guidance text when stall indicators are active."""
        parts: list[str] = []
        if self._stall.bash_stalled:
            parts.append(
                f"[STALL] Bash is stalled ({self._stall.bash_stalled_reason}). "
                "Try a different approach or command."
            )
        if self._stall.file_read_exhausted:
            parts.append(
                "[STALL] File reads are failing. Check that the path exists "
                "and try alternative files or approaches."
            )
        if self._stall.intent_repeat_count >= 5:
            parts.append(
                f"[STALL] You have repeated the same tool ({self._stall.last_tool_call}) "
                f"{self._stall.intent_repeat_count} times. Try a different approach."
            )
        if self._stall.consecutive_no_progress >= 2:
            parts.append(
                "[STALL] No progress detected for multiple steps. Reconsider your strategy."
            )
        if self._stall.cycle_detected:
            parts.append(
                "[STALL] Repeating tool-call pattern detected. "
                "Break the cycle — try a fundamentally different approach."
            )
        if self._stall.file_edit_counts:
            same_file_limit = 3
            for fp, count in self._stall.file_edit_counts.items():
                if count >= same_file_limit:
                    parts.append(
                        f"[STALL] File '{fp}' edited {count} times. "
                        "Stop editing and reconsider your approach from scratch."
                    )
                    break
        if self._stall.error_signature_counts:
            total_errors = sum(self._stall.error_signature_counts.values())
            if total_errors >= 10:
                parts.append(
                    f"[STALL] {total_errors} errors across "
                    f"{len(self._stall.error_signature_counts)} different patterns. "
                    "You may be stuck. Try a completely different approach."
                )
        if not parts:
            return None
        return "\n".join(parts)

    # Execution nudges (#27)

    def _get_execution_nudges(self) -> list[str]:
        """Generate execution nudges based on tool-call patterns."""
        nudges: list[str] = []

        # Read-only nudge: 5+ reads without any write
        if self._consecutive_reads_without_write >= 5:
            nudges.append(
                "[NUDGE] You have read multiple files without making changes. "
                "Consider making edits based on your findings."
            )
            self._consecutive_reads_without_write = 0  # reset after nudge

        # Verification nudge: after code modifications
        if self._pending_verification_nudge:
            nudges.append(
                "[NUDGE] You made code modifications. Consider running tests "
                "or verifying your changes."
            )
            self._pending_verification_nudge = False

        # Research depth nudge
        if self._stall.web_search_count >= 3 and self._stall.web_fetch_count == 0:
            nudges.append(
                "[NUDGE] You have searched the web multiple times without "
                "fetching any pages. Consider fetching results for deeper research."
            )

        return nudges

    # Workspace guard (#26)

    @staticmethod
    def _check_workspace_scope(command: str, workspace_root: str) -> str | None:
        """Check if a bash command writes outside the workspace root (#26).

        Delegates to the full workspace_guard module for thorough path analysis.
        Returns a warning message if a violation is detected, None otherwise.
        """
        if not command or not workspace_root:
            return None

        try:
            from rune.agent.workspace_guard import find_bash_scope_violations

            violations = find_bash_scope_violations(
                command=command,
                workspace_root=workspace_root,
            )
            if violations:
                details = ", ".join(f"{v.source}: {v.requested}" for v in violations)
                return (
                    f"[WORKSPACE GUARD] Command may access paths outside workspace "
                    f"({workspace_root}): {details}. "
                    "Please keep all modifications within the workspace."
                )
        except ImportError:
            # Fallback: basic regex checks
            import os

            workspace_root = os.path.realpath(workspace_root)
            violation_parts: list[str] = []

            for match in re.findall(r"\bcd\s+([^\s;&|]+)", command):
                target = os.path.expanduser(match)
                if os.path.isabs(target):
                    real = os.path.realpath(target)
                    if not real.startswith(workspace_root):
                        violation_parts.append(f"cd to {target}")

            for match in re.findall(r">{1,2}\s*([^\s;&|]+)", command):
                target = os.path.expanduser(match)
                if os.path.isabs(target):
                    real = os.path.realpath(target)
                    if not real.startswith(workspace_root):
                        violation_parts.append(f"redirect to {target}")

            for match in re.findall(r"-C\s+([^\s;&|]+)", command):
                target = os.path.expanduser(match)
                if os.path.isabs(target):
                    real = os.path.realpath(target)
                    if not real.startswith(workspace_root):
                        violation_parts.append(f"-C flag to {target}")

            if violation_parts:
                return (
                    f"[WORKSPACE GUARD] Command may write outside workspace "
                    f"({workspace_root}): {', '.join(violation_parts)}. "
                    "Please keep all modifications within the workspace."
                )

        return None

    # Step watchdog / retry (#29)

    async def _with_retry(self, fn: Callable[[], Awaitable[T]], max_retries: int = 3) -> T:
        """Execute a callable returning an awaitable, with exponential backoff on rate limits.

        Unlike the previous version that accepted a coroutine (which can only
        be awaited once), this accepts a callable so the operation can actually
        be retried on failure (#4b).
        """
        for attempt in range(max_retries + 1):
            try:
                return await fn()
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < max_retries:
                    delay = min(1.0 * (2**attempt), 8.0)
                    log.warning(
                        "rate_limit_retry",
                        attempt=attempt + 1,
                        delay=delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

    # LLM intent classification Tier 2 fallback (#17)

    async def _classify_intent_llm(self, goal: str) -> str | None:
        """Tier 2: LLM-based intent classification fallback."""
        try:
            from rune.llm.client import get_llm_client

            client = get_llm_client()
            response = await client.completion(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Classify this task intent into one of: "
                            "chat, quick_fix, code_modify, research, deep_research\n\n"
                            f"Task: {goal}\n\nIntent:"
                        ),
                    }
                ],
                max_tokens=20,
                timeout=5.0,
            )
            # Parse response
            text = ""
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
            else:
                try:
                    text = response.choices[0].message.content  # type: ignore[union-attr]
                except (AttributeError, IndexError):
                    pass

            text = text.strip().lower()
            for intent in ("chat", "quick_fix", "code_modify", "research", "deep_research"):
                if intent in text:
                    return intent
        except Exception:
            pass
        return None

    # Fallback execution loop (no PydanticAI) (#Task1)

    async def _execute_fallback_loop(
        self,
        goal: str,
        system_prompt: str,
        tools: list[str],
        max_iterations: int,
        classification: ClassificationResult,
    ) -> CompletionTrace:
        """Minimal agent loop when PydanticAI is not installed.

        Attempts to:
        1. Use the LLM client (LiteLLM) to generate a tool-calling plan
        2. Execute tool calls through the capability registry
        3. Track all steps in a CompletionTrace

        Falls back to a structured diagnostic trace if no LLM backend is
        available either.
        """
        from rune.capabilities.registry import get_capability_registry

        trace = CompletionTrace()
        registry = get_capability_registry()
        available_caps = [t for t in tools if registry.get(t) is not None]

        # Try LLM-based plan generation
        llm_available = False
        plan_text = ""
        try:
            from rune.llm.client import get_llm_client

            client = get_llm_client()
            await client.initialize()

            planning_prompt = (
                f"{system_prompt}\n\n"
                f"You are operating in fallback mode (no PydanticAI). "
                f"Available tools: {', '.join(available_caps)}\n\n"
                f"For each step, output a JSON line: "
                f'{{"tool": "<name>", "params": {{...}}}}\n\n'
                f"Goal: {goal}\n\nPlan:"
            )
            response = await client.completion(
                messages=[{"role": "user", "content": planning_prompt}],
                max_tokens=self._max_output_tokens,
                timeout=min(60.0, DEFAULT_AGENT_TIMEOUT),
            )

            # Extract text from LiteLLM response
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    plan_text = choices[0].get("message", {}).get("content", "")
            else:
                try:
                    plan_text = response.choices[0].message.content  # type: ignore[union-attr]
                except (AttributeError, IndexError):
                    pass

            if plan_text:
                llm_available = True
                log.info("fallback_plan_generated", length=len(plan_text))
        except Exception as exc:
            log.warning("fallback_llm_unavailable", error=str(exc))

        if llm_available and plan_text:
            # Parse and execute tool calls from the LLM plan
            from rune.utils.fast_serde import json_decode

            step = 0
            for line in plan_text.splitlines():
                line = line.strip()
                if not line:
                    continue

                # Try to extract JSON from the line (may be wrapped in markdown)
                json_match = re.search(r"\{.*\}", line)
                if not json_match:
                    continue

                try:
                    call = json_decode(json_match.group())
                except Exception:
                    continue

                tool_name = call.get("tool", "")
                params = call.get("params", {})

                if tool_name not in available_caps:
                    log.debug("fallback_skip_tool", tool=tool_name, reason="not available")
                    continue

                if step >= max_iterations:
                    log.info("fallback_max_iterations", step=step)
                    trace.reason = "max_iterations"
                    break

                step += 1
                self._step = step
                log.info("fallback_tool_call", step=step, tool=tool_name)
                await self.emit("tool_call", {"name": tool_name, "params": params})

                try:
                    result = await registry.execute(tool_name, params)
                    await self.emit(
                        "tool_result",
                        {
                            "name": tool_name,
                            "success": result.success,
                            "output": str(result.output)[:200] if result.output else "",
                        },
                    )
                except Exception as exc:
                    log.warning("fallback_tool_error", tool=tool_name, error=str(exc))

            if not trace.reason:
                trace.reason = "no_pydantic_ai"
            trace.final_step = step
            return trace

        # No LLM available either - produce a structured diagnostic trace
        log.warning(
            "fallback_no_llm",
            msg="Neither PydanticAI nor LLM client available; returning diagnostic trace",
        )
        trace.reason = "no_pydantic_ai"
        trace.final_step = 0
        trace.evidence_score = 0.0
        log.info(
            "fallback_diagnostic",
            goal=goal[:200],
            goal_type=classification.goal_type,
            confidence=classification.confidence,
            available_tools=available_caps[:10],
            tool_count=len(available_caps),
            msg=(
                "To enable full execution, install pydantic-ai "
                "(uv add pydantic-ai) or configure an LLM provider "
                "(set OPENAI_API_KEY or ANTHROPIC_API_KEY)."
            ),
        )
        return trace


# Factory: create_agent_loop (#Task2)


def create_agent_loop(
    role: str = "executor",
    max_iterations: int | None = None,
    timeout_seconds: float | None = None,
) -> NativeAgentLoop:
    """Create a configured :class:`NativeAgentLoop` for the given role.

    Parameters
    ----------
    role:
        One of ``researcher``, ``planner``, ``executor``, ``communicator``.
        Determines default iteration/timeout limits and tool subset.
    max_iterations:
        Override the role's default ``max_iterations``.
    timeout_seconds:
        Override the role's default ``timeout_seconds``.

    Returns
    -------
    NativeAgentLoop
        A ready-to-use loop instance.  Call ``.run(goal)`` to execute.
    """
    # Resolve role defaults
    try:
        from rune.agent.roles import get_role

        role_def = get_role(role)  # type: ignore[arg-type]
        effective_iterations = max_iterations or role_def.max_iterations
        effective_timeout = int(timeout_seconds or role_def.timeout_seconds)
    except (KeyError, ImportError):
        log.warning("create_agent_loop_unknown_role", role=role)
        effective_iterations = max_iterations or DEFAULT_MAX_ITERATIONS
        effective_timeout = int(timeout_seconds or DEFAULT_AGENT_TIMEOUT)

    config = AgentConfig(
        max_iterations=effective_iterations,
        timeout_seconds=effective_timeout,
    )

    loop = NativeAgentLoop(config=config)
    log.info(
        "agent_loop_created",
        role=role,
        max_iterations=effective_iterations,
        timeout_seconds=effective_timeout,
    )
    return loop
