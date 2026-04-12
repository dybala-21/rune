"""Context window aware payload construction for advisor calls.

The advisor may be a model with a smaller context window than the
executor (e.g. executor=Claude 1M, advisor=ollama/qwen2.5:32b with 32k).
This module builds a bounded user-content string that never exceeds the
target budget, dropping lower-priority sections first.

Priority ladder (P0 never dropped, P3 dropped first):

    P0: goal, trigger, classification_summary, missing_requirements,
        evidence_snapshot (numeric), files_written (paths only)
    P1: recent_messages tail-N (each truncated)
    P2: stall_state numeric summary
    P3: last_advisor_note reconciliation context

Token estimation is intentionally crude — we use a characters/3.5
approximation which is within 20% for English and acceptable for
budget-fitting. We never rely on exact provider-specific tokenizers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rune.agent.advisor.protocol import AdvisorRequest

DEFAULT_TARGET_TOKENS = 7_500
CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: str) -> int:
    """Rough token estimate, provider-independent."""
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _fmt_kv(data: dict) -> str:
    if not data:
        return "{}"
    parts = []
    for k, v in sorted(data.items()):
        parts.append(f"{k}={v}")
    return "{" + ", ".join(parts) + "}"


def _render_p0(req: AdvisorRequest) -> list[str]:
    lines: list[str] = []
    lines.append(f"TRIGGER: {req.trigger}")
    lines.append(f"GOAL: {req.goal}")
    lines.append(f"CLASSIFICATION: {req.classification_summary}")
    lines.append(f"PHASE: {req.activity_phase}  STEP: {req.step}")
    lines.append(f"TOKEN_BUDGET_USED_FRAC: {req.token_budget_frac:.2f}")
    lines.append(f"EVIDENCE: {_fmt_kv(req.evidence_snapshot)}")
    if req.gate_state:
        gs = req.gate_state
        missing = gs.get("missing_requirement_ids") or []
        outcome = gs.get("outcome", "?")
        hard_failures = gs.get("hard_failures") or []
        lines.append(
            f"GATE: outcome={outcome} missing={list(missing)[:10]} "
            f"hard_failures={list(hard_failures)[:5]}"
        )
    if req.files_written:
        lines.append(f"FILES_WRITTEN: {req.files_written[-20:]}")
    return lines


def _render_p1(req: AdvisorRequest, budget_chars: int) -> list[str]:
    if not req.recent_messages:
        return []
    lines = ["RECENT_MESSAGES:"]
    per_msg_budget = max(80, budget_chars // max(1, len(req.recent_messages)))
    for m in req.recent_messages[-5:]:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                str(p.get("text", p)) if isinstance(p, dict) else str(p)
                for p in content
            )
        content = str(content)
        if len(content) > per_msg_budget:
            content = content[: per_msg_budget - 3] + "..."
        lines.append(f"  [{role}] {content}")
    return lines


def _render_p2(req: AdvisorRequest) -> list[str]:
    if not req.stall_state:
        return []
    return [f"STALL: {_fmt_kv(req.stall_state)}"]


def _render_p3(req: AdvisorRequest) -> list[str]:
    if not req.last_advisor_note:
        return []
    note = req.last_advisor_note
    if len(note) > 500:
        note = note[:497] + "..."
    return [f"LAST_ADVISOR_NOTE: {note}"]


def build_payload(
    req: AdvisorRequest,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
) -> str:
    """Render a budget-bounded string payload for the advisor.

    The output is plain text with section headers — any provider can
    consume it as user content. P0 is always included; lower tiers are
    dropped if the budget is tight.
    """
    p0 = _render_p0(req)
    p0_text = "\n".join(p0)
    p0_tokens = estimate_tokens(p0_text)
    if p0_tokens >= target_tokens:
        return p0_text

    remaining_tokens = target_tokens - p0_tokens
    remaining_chars = int(remaining_tokens * CHARS_PER_TOKEN)

    sections: list[tuple[int, list[str]]] = []
    sections.append((2, _render_p2(req)))
    sections.append((3, _render_p3(req)))
    p1 = _render_p1(req, budget_chars=max(200, remaining_chars // 2))
    sections.insert(0, (1, p1))

    out = list(p0)
    used_chars = 0
    for _prio, lines in sections:
        if not lines:
            continue
        block = "\n".join(lines)
        block_chars = len(block)
        if used_chars + block_chars > remaining_chars:
            break
        out.append("")
        out.extend(lines)
        used_chars += block_chars + 1
    return "\n".join(out)
