"""User model manager - structured user profiles and preferences.

Ported from src/memory/user-model.ts - manages persistent user profile
data including work patterns, autonomy preferences, and communication style.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from rune.utils.fast_serde import json_decode
from rune.utils.logger import get_logger
from rune.utils.paths import rune_home

log = get_logger(__name__)


# Data types

@dataclass(slots=True)
class CommunicationPreferences:
    """User communication preference settings."""

    language: str = "auto"
    verbosity: Literal["concise", "normal", "verbose"] = "concise"
    tone: Literal["professional", "casual", "technical"] = "technical"
    code_examples: bool = True
    observed_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WorkProfile:
    """Tracks the user's development environment and work patterns."""

    preferred_languages: list[str] = field(default_factory=list)
    preferred_tools: dict[str, Any] = field(default_factory=dict)
    active_hours: dict[int, int] = field(default_factory=dict)  # hour -> activity count
    workspaces: dict[str, int] = field(default_factory=dict)  # path -> usage count
    language_stats: dict[str, Any] = field(default_factory=dict)  # lang -> {files, lines, ...}
    coding_style: dict[str, Any] = field(default_factory=dict)  # indent, naming convention, etc.


@dataclass
class UserModel:
    """Complete user profile for personalising agent behaviour."""

    user_id: str = ""
    work_profile: WorkProfile = field(default_factory=WorkProfile)
    autonomy_preferences: dict[str, Any] = field(default_factory=dict)
    communication_prefs: CommunicationPreferences = field(default_factory=CommunicationPreferences)
    goals: list[dict[str, Any]] = field(default_factory=list)  # subtasks with optional deadlines
    relationships: dict[str, str] = field(default_factory=dict)  # e.g. {"teammate": "Alice"}
    response_language: str = ""  # preferred response language

    # Backward-compatible property: older code / tests use communication_style
    @property
    def communication_style(self) -> str:
        """Backward-compatible accessor that maps to communication_prefs.verbosity."""
        return self.communication_prefs.verbosity

    @communication_style.setter
    def communication_style(self, value: str) -> None:
        if value in ("concise", "normal", "verbose"):
            self.communication_prefs.verbosity = value  # type: ignore[assignment]


# Serialisation helpers

def _model_to_dict(model: UserModel) -> dict[str, Any]:
    """Convert a UserModel to a JSON-safe dict."""
    return {
        "user_id": model.user_id,
        "work_profile": asdict(model.work_profile),
        "autonomy_preferences": model.autonomy_preferences,
        "communication_prefs": asdict(model.communication_prefs),
        # Keep communication_style for file-format backward compat
        "communication_style": model.communication_style,
        "goals": model.goals,
        "relationships": model.relationships,
        "response_language": model.response_language,
    }


def _dict_to_model(data: dict[str, Any]) -> UserModel:
    """Reconstruct a UserModel from a dict (lenient: missing keys get defaults)."""
    wp_data = data.get("work_profile", {})
    # active_hours keys are ints in memory but strings after JSON round-trip
    raw_hours = wp_data.get("active_hours", {})
    active_hours = {int(k): v for k, v in raw_hours.items()}

    raw_workspaces = wp_data.get("workspaces", {})
    workspaces = {str(k): v for k, v in raw_workspaces.items()}

    work_profile = WorkProfile(
        preferred_languages=wp_data.get("preferred_languages", []),
        preferred_tools=wp_data.get("preferred_tools", {}),
        active_hours=active_hours,
        workspaces=workspaces,
        language_stats=wp_data.get("language_stats", {}),
        coding_style=wp_data.get("coding_style", {}),
    )

    # Backwards compatibility: goals was previously list[str], now list[dict]
    raw_goals = data.get("goals", [])
    goals: list[dict[str, Any]] = []
    for g in raw_goals:
        if isinstance(g, str):
            goals.append({"description": g})
        elif isinstance(g, dict):
            goals.append(g)

    # Communication preferences: try new format first, fall back to old field
    cp_data = data.get("communication_prefs", {})
    if cp_data and isinstance(cp_data, dict):
        comm_prefs = CommunicationPreferences(
            language=cp_data.get("language", "auto"),
            verbosity=cp_data.get("verbosity", "normal"),
            tone=cp_data.get("tone", "technical"),
            code_examples=cp_data.get("code_examples", True),
            observed_notes=cp_data.get("observed_notes", []),
        )
    else:
        # Legacy: map old communication_style string to verbosity
        old_style = data.get("communication_style", "concise")
        verbosity = old_style if old_style in ("concise", "normal", "verbose") else "concise"
        comm_prefs = CommunicationPreferences(verbosity=verbosity)  # type: ignore[arg-type]

    return UserModel(
        user_id=data.get("user_id", ""),
        work_profile=work_profile,
        autonomy_preferences=data.get("autonomy_preferences", {}),
        communication_prefs=comm_prefs,
        goals=goals,
        relationships=data.get("relationships", {}),
        response_language=data.get("response_language", ""),
    )


# UserModelManager

class UserModelManager:
    """Manages a persistent user model stored as JSON."""

    def __init__(self, store_path: str | Path | None = None) -> None:
        if store_path is None:
            store_path = rune_home() / "user-model.json"
        self._path = Path(store_path)
        self._lock = threading.RLock()

    def load(self) -> UserModel:
        """Load the user model from disk, returning defaults if absent."""
        if not self._path.exists():
            log.debug("user_model_not_found", path=str(self._path))
            return UserModel()

        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json_decode(raw)
            model = _dict_to_model(data)
            log.debug("user_model_loaded", user_id=model.user_id)
            return model
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("user_model_load_failed", error=str(exc))
            return UserModel()

    def save(self, model: UserModel) -> None:
        """Persist the user model to disk (atomic via temp-rename)."""
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            try:
                tmp.write_text(
                    json.dumps(_model_to_dict(model), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                tmp.replace(self._path)
                log.debug("user_model_saved", user_id=model.user_id)
            except OSError as exc:
                log.error("user_model_save_failed", error=str(exc))
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
                raise

    # Activity tracking

    def record_activity_hour(self, hour: int) -> None:
        """Track when user is active (hour 0-23)."""
        with self._lock:
            model = self.load()
            model.work_profile.active_hours[hour] = (
                model.work_profile.active_hours.get(hour, 0) + 1
            )
            self.save(model)

    def record_workspace(self, path: str) -> None:
        """Track workspace paths and their usage frequency."""
        with self._lock:
            model = self.load()
            model.work_profile.workspaces[path] = (
                model.work_profile.workspaces.get(path, 0) + 1
            )
            self.save(model)

    def record_language_usage(self, language: str) -> None:
        """Track programming language usage (most-recently-used ordering)."""
        with self._lock:
            model = self.load()
            langs = model.work_profile.preferred_languages
            if language in langs:
                langs.remove(language)
            langs.insert(0, language)
            # Keep at most 10
            model.work_profile.preferred_languages = langs[:10]
            self.save(model)

    def record_coding_style_observation(self, key: str, value: str) -> None:
        """Record a coding style observation, e.g. ('indent', '4 spaces')."""
        with self._lock:
            model = self.load()
            model.work_profile.coding_style[key] = value
            self.save(model)

    def record_language_task_result(self, language: str, success: bool) -> None:
        """Track per-language task success/failure rates."""
        with self._lock:
            model = self.load()
            stats = model.work_profile.language_stats.get(language, {
                "task_count": 0,
                "success_count": 0,
                "last_used": "",
            })
            stats["task_count"] = stats.get("task_count", 0) + 1
            if success:
                stats["success_count"] = stats.get("success_count", 0) + 1
            stats["last_used"] = datetime.now(UTC).isoformat()
            model.work_profile.language_stats[language] = stats
            self.save(model)

    # Autonomy preferences

    def set_domain_autonomy_level(self, domain: str, level: str) -> None:
        """Set the autonomy level for a specific task domain."""
        with self._lock:
            model = self.load()
            domain_levels = model.autonomy_preferences.setdefault("domain_levels", {})
            domain_levels[domain] = level
            self.save(model)
            log.info("autonomy_domain_set", domain=domain, level=level)

    def set_autonomy_enabled(self, enabled: bool) -> None:
        """Enable or disable autonomous execution."""
        with self._lock:
            model = self.load()
            model.autonomy_preferences["enabled"] = enabled
            self.save(model)
            log.info("autonomy_enabled_set", enabled=enabled)

    def set_global_max_level(self, level: str) -> None:
        """Set the global maximum autonomy level."""
        with self._lock:
            model = self.load()
            model.autonomy_preferences["global_max_level"] = level
            self.save(model)

    # Goal tracking

    def get_active_goals_summary(self) -> str:
        """Formatted string of active goals for daily briefing."""
        model = self.load()
        active = [g for g in model.goals if g.get("status", "active") == "active"]
        if not active:
            return "No active goals."

        now_ts = datetime.now(UTC).timestamp()
        lines: list[str] = []
        for g in active:
            desc = g.get("description", str(g))
            progress = g.get("progress", 0)
            parts = f"- {desc}: {progress}%"

            sub_tasks = g.get("sub_tasks", [])
            if sub_tasks:
                completed = sum(1 for t in sub_tasks if t.get("completed", False))
                parts += f" [{completed}/{len(sub_tasks)} tasks]"

            deadline = g.get("deadline", "")
            if deadline:
                try:
                    dl_dt = datetime.fromisoformat(deadline)
                    days_left = math.ceil((dl_dt.timestamp() - now_ts) / 86_400)
                    if days_left <= 0:
                        parts += " [OVERDUE]"
                    elif days_left <= 3:
                        parts += f" [{days_left}d left!]"
                    elif days_left <= 7:
                        parts += f" [{days_left}d left]"
                    else:
                        parts += f" (due: {deadline})"
                except (ValueError, TypeError):
                    parts += f" (due: {deadline})"

            lines.append(parts)

        return "\n".join(lines)

    # Context building (for agent prompt injection)

    def build_user_context(self) -> str:
        """Format the user model as a context string for agent prompts."""
        model = self.load()
        parts: list[str] = []

        # Communication preferences
        cp = model.communication_prefs
        comm_parts: list[str] = []
        if cp.language != "auto":
            comm_parts.append(f"lang={cp.language}")
        if cp.verbosity != "normal":
            comm_parts.append(f"verbosity={cp.verbosity}")
        if cp.tone != "technical":
            comm_parts.append(f"tone={cp.tone}")
        if not cp.code_examples:
            comm_parts.append("no-code-examples")
        if comm_parts:
            parts.append(f"Communication: {', '.join(comm_parts)}")
        if cp.observed_notes:
            parts.append(f"Communication notes: {'; '.join(cp.observed_notes[-3:])}")

        if model.response_language:
            parts.append(f"Response language: {model.response_language}")

        # Active goals with urgency signals
        active_goals = [g for g in model.goals if g.get("status", "active") == "active"]
        if active_goals:
            parts.append("Active goals:")
            now_ts = datetime.now(UTC).timestamp()
            for g in active_goals[:5]:
                desc = g.get("description", str(g))
                progress = g.get("progress", 0)
                line = f"  - {desc}: {progress}%"

                sub_tasks = g.get("sub_tasks", [])
                if sub_tasks:
                    completed = sum(1 for t in sub_tasks if t.get("completed", False))
                    line += f" [{completed}/{len(sub_tasks)} tasks]"

                deadline = g.get("deadline", "")
                if deadline:
                    try:
                        dl_dt = datetime.fromisoformat(deadline)
                        days_left = math.ceil((dl_dt.timestamp() - now_ts) / 86_400)
                        if days_left <= 0:
                            line += " [OVERDUE]"
                        elif days_left <= 3:
                            line += f" [{days_left}d left!]"
                        elif days_left <= 7:
                            line += f" [{days_left}d left]"
                    except (ValueError, TypeError):
                        pass

                parts.append(line)

        if model.relationships:
            parts.append("Relationships:")
            for role, name in model.relationships.items():
                parts.append(f"  - {role}: {name}")

        wp = model.work_profile

        # Language stats (prefer detailed stats over simple list)
        lang_stats_entries = [
            (lang, s) for lang, s in wp.language_stats.items()
            if isinstance(s, dict) and s.get("task_count", 0) >= 3
        ]
        lang_stats_entries.sort(key=lambda x: -x[1].get("task_count", 0))
        if lang_stats_entries:
            formatted = []
            for lang, s in lang_stats_entries[:5]:
                tc = s.get("task_count", 0)
                sc = s.get("success_count", 0)
                rate = round((sc / tc) * 100) if tc > 0 else 0
                formatted.append(f"{lang}({tc} tasks, {rate}% success)")
            parts.append(f"Language experience: {', '.join(formatted)}")
        elif wp.preferred_languages:
            parts.append(f"Preferred languages: {', '.join(wp.preferred_languages[:5])}")

        if wp.preferred_tools:
            tools_str = ", ".join(f"{k}={v}" for k, v in wp.preferred_tools.items())
            parts.append(f"Preferred tools: {tools_str}")

        # Coding style
        if wp.coding_style:
            style_items = [f"{k}={v}" for k, v in wp.coding_style.items()]
            parts.append(f"Coding style: {', '.join(style_items)}")

        if model.autonomy_preferences:
            prefs = ", ".join(
                f"{k}={v}" for k, v in model.autonomy_preferences.items()
            )
            parts.append(f"Autonomy preferences: {prefs}")

        if wp.active_hours:
            sorted_hours = sorted(wp.active_hours.items(), key=lambda x: -x[1])[:5]
            hours_str = ", ".join(f"{h}:00({c})" for h, c in sorted_hours)
            parts.append(f"Most active hours: {hours_str}")

        if wp.workspaces:
            sorted_ws = sorted(wp.workspaces.items(), key=lambda x: -x[1])[:3]
            ws_str = ", ".join(f"{p}({c})" for p, c in sorted_ws)
            parts.append(f"Frequent workspaces: {ws_str}")

        return "\n".join(parts) if parts else "No user profile data available."

    # Batch / preference helpers

    def batch_update_work_profile(self, updater: Callable[[WorkProfile], None]) -> None:
        """Atomic load-mutate-save cycle for the work profile."""
        with self._lock:
            model = self.load()
            updater(model.work_profile)
            self.save(model)

    def update_preference(self, key: str, value: Any) -> None:
        """Set a single autonomy/preference value."""
        with self._lock:
            model = self.load()
            model.autonomy_preferences[key] = value
            self.save(model)

    def get_preference(self, key: str) -> Any:
        """Retrieve a single preference value (None if missing)."""
        model = self.load()
        return model.autonomy_preferences.get(key)


# Module singleton

_manager: UserModelManager | None = None


def get_user_model_manager() -> UserModelManager:
    global _manager
    if _manager is None:
        _manager = UserModelManager()
    return _manager
