"""Context window usage calculation for RUNE TUI.

Provides a dataclass for context usage metrics and helpers
to calculate and format token budget consumption.
"""

from __future__ import annotations

from dataclasses import dataclass

from rune.ui.theme import COLORS, format_tokens


@dataclass(slots=True)
class ContextUsage:
    """Snapshot of context window usage."""

    used_tokens: int
    total_tokens: int
    fraction: float
    phase: str  # "low", "medium", "high", "critical"


def calculate_context_usage(
    used_tokens: int,
    total_tokens: int,
) -> ContextUsage:
    """Calculate context usage metrics.

    Parameters
    ----------
    used_tokens:
        Number of tokens consumed so far.
    total_tokens:
        Maximum token budget for the context window.

    Returns
    -------
    ContextUsage with fraction and phase classification.
    """
    total = max(1, total_tokens)
    fraction = used_tokens / total

    if fraction < 0.50:
        phase = "low"
    elif fraction < 0.75:
        phase = "medium"
    elif fraction < 0.90:
        phase = "high"
    else:
        phase = "critical"

    return ContextUsage(
        used_tokens=used_tokens,
        total_tokens=total_tokens,
        fraction=fraction,
        phase=phase,
    )


_PHASE_COLORS: dict[str, str] = {
    "low": COLORS["success"],
    "medium": COLORS["secondary"],
    "high": COLORS["warning"],
    "critical": COLORS["error"],
}


def format_context_usage(usage: ContextUsage) -> str:
    """Format context usage as a human-readable string.

    Example: "12.3K / 500K (2.5%)"
    """
    used = format_tokens(usage.used_tokens)
    total = format_tokens(usage.total_tokens)
    pct = usage.fraction * 100
    color = _PHASE_COLORS.get(usage.phase, COLORS["muted"])
    return f"[{color}]{used}[/{color}] / {total} ({pct:.1f}%)"
