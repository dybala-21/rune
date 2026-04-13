"""Strict verb parser for advisor responses.

Contract:

    NEXT: <verb>[:<target>]
    1. <step>
    2. <step>
    ...

``verb`` is ASCII-fixed (``continue``, ``retry_tool``, ``switch_approach``,
``abort``, ``need_reconcile``) so the parser is language-independent — the
advisor may write the numbered steps in any language and it still works.
When parsing fails, the result degrades to ``action="continue"`` with
``plan_steps=[]``; the executor simply carries on.

Hard cap: at most 5 plan steps and ~100 words of combined step text. This
defends against verbose local models that ignore the conciseness prompt.
"""

from __future__ import annotations

import re
from typing import get_args

from rune.agent.advisor.protocol import AdvisorAction, AdvisorDecision, AdvisorTrigger

_VALID_ACTIONS = set(get_args(AdvisorAction))

_VERB_LINE_RE = re.compile(
    r"^\s*NEXT\s*:\s*([a-zA-Z_]+)(?:\s*:\s*([^\s\n]+))?",
    re.IGNORECASE,
)
_NUMBERED_LINE_RE = re.compile(r"^\s*(?:\d+\s*[.)]|[-*])\s*(.+)$")

# Architect mode: FILE: <path> marker + fenced code block
_FILE_LINE_RE = re.compile(r"^\s*FILE\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```(?:[\w.+-]*)?\n(.*?)```", re.DOTALL)

MAX_PLAN_STEPS = 5
MAX_PLAN_WORDS = 100
MAX_PATCH_BYTES = 200_000  # defend against runaway output


def parse(
    normalized_text: str,
    *,
    trigger: AdvisorTrigger | None = None,
    provider: str = "",
    model: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: int = 0,
) -> AdvisorDecision:
    """Parse normalized advisor text into an ``AdvisorDecision``.

    Never raises. On any parsing ambiguity, returns a benign
    ``action="continue"`` decision with ``raw_text`` preserved.
    """
    raw = normalized_text or ""
    if not raw.strip():
        return AdvisorDecision(
            action="continue",
            plan_steps=[],
            raw_text=raw,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            trigger=trigger,
            error_code="empty_response",
        )

    lines = raw.splitlines()
    action: AdvisorAction = "continue"
    target_tool: str | None = None

    # Scan for the first verb line anywhere in the first 5 lines.
    verb_line_idx: int | None = None
    for i, line in enumerate(lines[:5]):
        match = _VERB_LINE_RE.match(line)
        if match:
            verb_raw = (match.group(1) or "").lower()
            target_raw = match.group(2)
            if verb_raw in _VALID_ACTIONS:
                action = verb_raw  # type: ignore[assignment]
                target_tool = target_raw.strip() if target_raw else None
                verb_line_idx = i
                break

    # Collect numbered/bulleted lines AFTER the verb line (or from start
    # if no verb line was found — degraded parse still captures plan).
    start = (verb_line_idx + 1) if verb_line_idx is not None else 0
    plan_steps: list[str] = []
    total_words = 0
    for line in lines[start:]:
        m = _NUMBERED_LINE_RE.match(line)
        if not m:
            continue
        step = m.group(1).strip()
        if not step:
            continue
        step_words = len(step.split())
        if total_words + step_words > MAX_PLAN_WORDS:
            break  # drop whole step if it would overflow the word cap
        plan_steps.append(step)
        total_words += step_words
        if len(plan_steps) >= MAX_PLAN_STEPS:
            break

    # Architect mode: extract FILE: path + fenced code block.
    # If either is missing or the block is too large, fall back gracefully
    # to an advice-style continue decision so the loop never gets a
    # half-formed patch.
    patch: str | None = None
    patch_target_file: str | None = None
    if action == "apply_patch":
        file_match = _FILE_LINE_RE.search(raw)
        code_match = _CODE_BLOCK_RE.search(raw)
        if file_match and code_match:
            candidate = code_match.group(1)
            if len(candidate.encode("utf-8", errors="ignore")) <= MAX_PATCH_BYTES:
                patch = candidate
                patch_target_file = file_match.group(1).strip()
        if patch is None or not patch_target_file:
            # Degrade to plain advice — loop will treat as informational
            action = "continue"

    return AdvisorDecision(
        action=action,
        plan_steps=plan_steps,
        target_tool=target_tool,
        raw_text=raw,
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        trigger=trigger,
        patch=patch,
        patch_target_file=patch_target_file,
    )


def format_injection(decision: AdvisorDecision) -> str:
    """Render an AdvisorDecision with graduated strength.

    Strongest → weakest:
      APPLY_PATCH  : architect mode — advisor provides full file content.
      STUCK/RECONCILE: directive — executor must follow.
      EARLY/PRE_DONE: informational hint.
    """
    # Architect mode — strongest injection
    if decision.action == "apply_patch" and decision.patch and decision.patch_target_file:
        parts = [
            "[ADVISOR PATCH — MANDATORY]",
            f"The advisor has written corrected code for {decision.patch_target_file}.",
            "Your NEXT action MUST be:",
            f"  file_write(file_path='{decision.patch_target_file}', content=<PATCH below>)",
            "Do NOT modify the patch. Do NOT write a different file.",
            "After writing, run bash_execute to verify.",
            "",
            "<PATCH>",
            decision.patch,
            "</PATCH>",
        ]
        return "\n".join(parts)

    if not decision.plan_steps and decision.action == "continue":
        return ""

    steps = [f"{i + 1}. {step}" for i, step in enumerate(decision.plan_steps)]

    if decision.trigger in ("stuck", "reconcile"):
        header = "[Advisor — DIRECTIVE]"
        if decision.target_tool:
            header += f" USE {decision.target_tool}"
        prefix = (
            "Your progress has stalled. Follow these steps IN ORDER:"
            if decision.trigger == "stuck"
            else "Previous advice was not followed. You MUST take a different approach:"
        )
        suffix = "Execute step 1 IMMEDIATELY in your next action."
        parts = [header, prefix, ""] + steps + ["", suffix]
        return "\n".join(parts)

    # EARLY / PRE_DONE: informational
    header = f"[Advisor] {decision.action.upper()}"
    if decision.target_tool:
        header += f" → {decision.target_tool}"
    if not steps:
        return header
    return header + "\n" + "\n".join(steps)
