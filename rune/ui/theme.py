"""Theme system for RUNE TUI.

Provides three theme variants (dark, light, minimal) with helpers to
get/set the active theme.

Also provides backward-compatible exports: COLORS, role_color, risk_color,
format_duration, format_tokens, truncate_text.
"""

from __future__ import annotations

# Theme definitions

THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "primary": "#D4A017",
        "secondary": "#8BADC8",
        "accent": "#C07830",
        "user": "#A0D2F0",
        "assistant": "#D4A860",
        "system": "#E8A040",
        "thinking": "#B48EAD",
        "tool": "#D4A017",
        "success": "#5E9E5E",
        "error": "#D45050",
        "warning": "#D4A017",
        "info": "#7EB8DA",
        "border": "#505050",
        "border_active": "#D4A017",
        "muted": "#6B6B6B",
        "text": "#E8E8E8",
        "bg": "#1A1A1A",
        "diff_add": "#5E9E5E",
        "diff_remove": "#D45050",
    },
    "light": {
        "primary": "#B8860B",
        "secondary": "#5F8BA8",
        "accent": "#A05520",
        "user": "#2563eb",
        "assistant": "#B8860B",
        "system": "#C07830",
        "thinking": "#8B6FA3",
        "tool": "#B8860B",
        "success": "#4A8A4A",
        "error": "#C04040",
        "warning": "#B8860B",
        "info": "#5A8CB8",
        "border": "#C0C0C0",
        "border_active": "#B8860B",
        "muted": "#808080",
        "text": "#1A1A1A",
        "bg": "#F5F0E8",
        "diff_add": "#4A8A4A",
        "diff_remove": "#C04040",
    },
    "minimal": {
        "primary": "#A89060",
        "secondary": "#A89060",
        "accent": "#C0A878",
        "user": "#D0C8B8",
        "assistant": "#C0A878",
        "system": "#807060",
        "thinking": "#A89060",
        "tool": "#C0A878",
        "success": "#A89060",
        "error": "#C0A878",
        "warning": "#C0A878",
        "info": "#A89060",
        "border": "#404040",
        "border_active": "#A89060",
        "muted": "#585048",
        "text": "#E0D8C8",
        "bg": "#1A1A18",
        "diff_add": "#A89060",
        "diff_remove": "#807060",
    },
}


# Active theme state

_active_theme: str = "dark"


def get_theme(name: str) -> dict[str, str]:
    """Return the theme dict for *name*, falling back to ``dark``."""
    return THEMES.get(name, THEMES["dark"])


def set_theme(name: str) -> None:
    """Set the active theme by name.  Raises ``KeyError`` if unknown."""
    global _active_theme  # noqa: PLW0603
    if name not in THEMES:
        raise KeyError(f"Unknown theme: {name!r}. Choose from {sorted(THEMES)}")
    _active_theme = name


def current_theme() -> dict[str, str]:
    """Return the currently-active theme dict."""
    return THEMES[_active_theme]


def current_theme_name() -> str:
    """Return the name of the currently-active theme."""
    return _active_theme


# Backward-compatible COLORS dict (derived from active theme)

COLORS: dict[str, str] = THEMES["dark"]

_RISK_COLORS: dict[str, str] = {
    "low": "#5E9E5E",
    "medium": "#D4A017",
    "high": "#D45050",
    "critical": "#D45050",
}


def role_color(role: str) -> str:
    """Return the hex color for a message role."""
    t = current_theme()
    return t.get(role, t.get("muted", "#6b7280"))


def risk_color(risk_level: str) -> str:
    """Return the hex color for a risk level."""
    return _RISK_COLORS.get(risk_level, COLORS.get("muted", "#6b7280"))


# Formatting helpers

def format_duration(ms: float) -> str:
    """Format a duration in milliseconds to a human-readable string."""
    if ms < 1000:
        return f"{int(ms)}ms"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining = int(seconds % 60)
    if remaining == 0:
        return f"{minutes}m"
    return f"{minutes}m {remaining}s"


def format_tokens(count: int) -> str:
    """Format a token count to a human-readable string."""
    if count < 1_000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1_000:.1f}k"
    return f"{count / 1_000_000:.1f}M"


def truncate_text(text: str, max_length: int = 80) -> str:
    """Truncate text to *max_length* characters, adding ellipsis if needed."""
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]
    return text[: max_length - 3] + "..."

