"""Morning report for ``rune overnight``.

Turns a finished goal-loop run into a short, honest summary: what it shipped
(only verified work), or what it could not do and why. Pure and testable; no
model or loop dependency.
"""

from __future__ import annotations

_STUCK_WHY = {
    "max_iterations": "ran out of attempts before the checks passed",
    "stagnation": "stopped making progress (repeated the same outcome)",
    "budget": "hit the token budget before the checks passed",
    "advisor_abort": "the advisor reviewed it and recommended stopping",
    "cancelled": "the run was cancelled",
    "error": "the run hit an error",
}


def format_overnight_report(
    *,
    goal: str,
    success: bool,
    stop_cause: str,
    iterations: int,
    validation: list[str],
    changed_files: list[str],
    escalation_hint: str | None = None,
) -> str:
    """Build the morning report. ``success`` means the loop's objective checks
    passed; anything else is reported as not-done with the reason, never as a
    success that cannot be backed up."""
    checks = ", ".join(f"`{c}`" for c in validation) if validation else "the inner gate"
    lines: list[str] = [f'overnight: "{goal[:70]}"']

    if success:
        lines.append(f"  DONE in {iterations} iteration(s), verified by {checks}.")
        if changed_files:
            shown = ", ".join(changed_files[:8])
            more = f" (+{len(changed_files) - 8} more)" if len(changed_files) > 8 else ""
            lines.append(f"  changed: {shown}{more}")
        else:
            lines.append("  no files changed.")
        lines.append("  shipped only what the checks confirmed.")
        return "\n".join(lines)

    why = _STUCK_WHY.get(stop_cause, stop_cause)
    lines.append(f"  NOT done after {iterations} iteration(s): {why}.")
    lines.append("  nothing was reported as passing that did not (no fabricated success).")
    if escalation_hint:
        lines.append(f"  {escalation_hint}")
    return "\n".join(lines)
