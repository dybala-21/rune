"""Profile Loader for RUNE.

Ported from src/config/profile.ts -- loads and parses USER.md and
PERSONA.md files for per-user settings and response persona configuration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rune.utils.logger import get_logger
from rune.utils.paths import expand_path

log = get_logger(__name__)

# Default config directory (can be overridden).
_DEFAULT_CONFIG_DIR = Path.home() / ".rune"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class WorkHours:
    start: str = ""
    end: str = ""


@dataclass
class RecentInterest:
    date: str = ""
    topic: str = ""


@dataclass
class UserProfile:
    """Parsed representation of ``USER.md``."""

    preferred_language: str | None = None
    work_hours: WorkHours | None = None
    do_nots: list[str] = field(default_factory=list)
    organization_preferences: list[str] = field(default_factory=list)
    recent_interests: list[RecentInterest] = field(default_factory=list)
    raw_content: str = ""
    sections: dict[str, str] = field(default_factory=dict)


@dataclass
class PersonaConfig:
    """Parsed representation of ``PERSONA.md``."""

    response_style: list[str] = field(default_factory=list)
    tone: str | None = None
    use_emoji: bool | None = None
    default_language: str | None = None
    additional_instructions: list[str] = field(default_factory=list)
    raw_content: str = ""
    sections: dict[str, str] = field(default_factory=dict)


# ============================================================================
# Markdown Helpers
# ============================================================================


def _parse_markdown_sections(content: str) -> dict[str, str]:
    """Split markdown into ``## Section`` -> body mappings."""
    sections: dict[str, str] = {}
    current_section = ""
    current_lines: list[str] = []

    for line in content.split("\n"):
        header_match = re.match(r"^##\s+(.+)", line)
        if header_match:
            if current_section:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = header_match.group(1).strip()
            current_lines = []
        elif current_section:
            current_lines.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_lines).strip()

    return sections


def _parse_list_items(content: str) -> list[str]:
    """Extract ``- item`` or ``* item`` list entries from *content*."""
    items: list[str] = []
    for line in content.split("\n"):
        m = re.match(r"^[-*]\s+(.+)", line)
        if m:
            items.append(m.group(1).strip())
    return items


# ============================================================================
# Profile Parsing
# ============================================================================


def _parse_user_profile(content: str) -> UserProfile:
    sections = _parse_markdown_sections(content)
    profile = UserProfile(raw_content=content, sections=sections)

    # Basic Info
    basic_info = sections.get("Basic Info") or sections.get("\uae30\ubcf8 \uc815\ubcf4") or ""
    if basic_info:
        lang_match = re.search(r"\uc120\ud638 \uc5b8\uc5b4[:\s]*(.+)", basic_info)
        if lang_match:
            profile.preferred_language = lang_match.group(1).strip()

        wh_match = re.search(
            r"\uc791\uc5c5 \uc2dc\uac04\ub300[:\s]*(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})",
            basic_info,
        )
        if wh_match:
            profile.work_hours = WorkHours(start=wh_match.group(1), end=wh_match.group(2))

    # Do Not
    do_not_section = sections.get("Do Not") or sections.get("\uae08\uae30 \uc0ac\ud56d") or ""
    if do_not_section:
        profile.do_nots = _parse_list_items(do_not_section)

    # Preferences
    pref_section = sections.get("Preferences") or sections.get("\uc120\ud638 \uc0ac\ud56d") or ""
    if pref_section:
        profile.organization_preferences = _parse_list_items(pref_section)

    # Recent Interests
    interest_section = (
        sections.get("Recent Interests") or sections.get("\ucd5c\uadfc \uad00\uc2ec\uc0ac") or ""
    )
    if interest_section:
        items = _parse_list_items(interest_section)
        for item in items:
            date_match = re.match(r"^(\d{4}-\d{2}-\d{2})[:\s]*(.+)", item)
            if date_match:
                profile.recent_interests.append(
                    RecentInterest(date=date_match.group(1), topic=date_match.group(2))
                )
            else:
                profile.recent_interests.append(RecentInterest(topic=item))

    return profile


def _parse_persona_config(content: str) -> PersonaConfig:
    sections = _parse_markdown_sections(content)
    persona = PersonaConfig(raw_content=content, sections=sections)

    # Response Style
    style_section = (
        sections.get("Response Style") or sections.get("\uc751\ub2f5 \uc2a4\ud0c0\uc77c") or ""
    )
    if style_section:
        persona.response_style = _parse_list_items(style_section)
        persona.use_emoji = (
            "\uc774\ubaa8\uc9c0 \uc0ac\uc6a9 \uc548\ud568" not in style_section
            and "No emoji" not in style_section
        )

    # Language
    lang_section = sections.get("Language") or sections.get("\uc5b8\uc5b4") or ""
    if lang_section:
        lang_match = re.search(r"\uae30\ubcf8[:\s]*(.+)", lang_section)
        if lang_match:
            persona.default_language = lang_match.group(1).strip()

    # Additional Instructions
    add_section = (
        sections.get("Additional Instructions") or sections.get("\ucd94\uac00 \uc9c0\uc2dc") or ""
    )
    if add_section:
        persona.additional_instructions = _parse_list_items(add_section)

    return persona


# ============================================================================
# Profile Loader
# ============================================================================


class ProfileLoader:
    """Loads USER.md and PERSONA.md from the RUNE config directory.

    Usage::

        loader = ProfileLoader()
        await loader.initialize()
        profile = loader.get_user_profile()
        persona = loader.get_persona_config()
        llm_ctx = loader.build_llm_context()
    """

    def __init__(self, config_dir: Path | None = None) -> None:
        self._config_dir = config_dir or _DEFAULT_CONFIG_DIR
        self._user_profile: UserProfile | None = None
        self._persona_config: PersonaConfig | None = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self.load_user_profile()
        await self.load_persona_config()
        self._initialized = True
        log.debug("profile_loader_initialized")

    async def load_user_profile(self) -> UserProfile | None:
        user_md_path = self._config_dir / "USER.md"
        try:
            expanded = expand_path(str(user_md_path))
            content = expanded.read_text(encoding="utf-8")
            self._user_profile = _parse_user_profile(content)
            log.debug("user_md_loaded")
            return self._user_profile
        except (OSError, UnicodeDecodeError):
            log.debug("user_md_not_found_using_defaults")
            self._user_profile = UserProfile()
            return self._user_profile

    async def load_persona_config(self) -> PersonaConfig | None:
        persona_md_path = self._config_dir / "PERSONA.md"
        try:
            expanded = expand_path(str(persona_md_path))
            content = expanded.read_text(encoding="utf-8")
            self._persona_config = _parse_persona_config(content)
            log.debug("persona_md_loaded")
            return self._persona_config
        except (OSError, UnicodeDecodeError):
            log.debug("persona_md_not_found_using_defaults")
            self._persona_config = PersonaConfig()
            return self._persona_config

    def get_user_profile(self) -> UserProfile | None:
        return self._user_profile

    def get_persona_config(self) -> PersonaConfig | None:
        return self._persona_config

    def build_llm_context(self) -> str:
        """Build a context string suitable for an LLM system prompt."""
        parts: list[str] = []

        if self._user_profile and self._user_profile.raw_content:
            parts.append("## User Profile\n")

            if self._user_profile.preferred_language:
                parts.append(f"Preferred language: {self._user_profile.preferred_language}")
            if self._user_profile.work_hours:
                parts.append(
                    f"Work hours: {self._user_profile.work_hours.start}"
                    f" - {self._user_profile.work_hours.end}"
                )
            if self._user_profile.do_nots:
                parts.append("\nDo NOT:")
                for item in self._user_profile.do_nots:
                    parts.append(f"- {item}")
            if self._user_profile.organization_preferences:
                parts.append("\nUser preferences:")
                for pref in self._user_profile.organization_preferences:
                    parts.append(f"- {pref}")

        if self._persona_config and self._persona_config.raw_content:
            parts.append("\n## Response Guidelines\n")

            if self._persona_config.response_style:
                for style in self._persona_config.response_style:
                    parts.append(f"- {style}")
            if self._persona_config.use_emoji is False:
                parts.append("- Do NOT use emojis in responses")
            if self._persona_config.default_language:
                parts.append(
                    f"- Default response language: {self._persona_config.default_language}"
                )
            if self._persona_config.additional_instructions:
                parts.append("\nAdditional instructions:")
                for instr in self._persona_config.additional_instructions:
                    parts.append(f"- {instr}")

        return "\n".join(parts)

    async def reload(self) -> None:
        self._initialized = False
        await self.initialize()


# ============================================================================
# Singleton
# ============================================================================

_profile_loader: ProfileLoader | None = None


def get_profile_loader() -> ProfileLoader:
    global _profile_loader
    if _profile_loader is None:
        _profile_loader = ProfileLoader()
    return _profile_loader
