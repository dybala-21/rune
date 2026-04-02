"""Memory bridge for RUNE - connects agent to memory subsystem.

Ported from src/agent/memory-bridge.ts (1106 lines) - builds agent memory
context from 8 layers, saves results, extracts artifacts, detects languages
and tools, and manages auto-skill generation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from rune.memory.store import Episode
from rune.skills.registry import get_skill_registry
from rune.skills.types import Skill
from rune.types import Domain, Intent
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Constants

PATH_PATTERN = re.compile(
    r"""(?:^|[\s"'`(=;|&<>])"""
    r"""((?:/[\w.\-]+)+|(?:\.\.?/[\w.\-]+(?:/[\w.\-]+)*))"""
)

PROJECT_MEMORY_MAX_LINES = 200
PROJECT_MEMORY_MAX_CHARS = 3000

# Auto-skill refinement limits
_REFINEMENT_MAX_TOKENS = 600
_REFINEMENT_MAX_STEPS = 20
_REFINEMENT_MIN_STEPS = 1


# LLM Refiner Protocol


@runtime_checkable
class LLMRefiner(Protocol):
    """Protocol for pluggable LLM-based skill refinement.

    Implementations should call an LLM and return the generated text.
    If no LLM is available, callers fall back to the pattern-extracted
    version of the skill steps.
    """

    async def refine(self, prompt: str, max_tokens: int = 600) -> str:
        """Send *prompt* to an LLM and return the refined text."""
        ...

# Data classes


@dataclass(slots=True)
class AgentMemoryContext:
    """Memory context assembled for the agent system prompt."""
    relevant_history: list[str] = field(default_factory=list)
    preferences: dict[str, str] = field(default_factory=dict)
    recent_commands: list[str] = field(default_factory=list)
    safety_rules: list[dict[str, str]] = field(default_factory=list)
    formatted: str = ""


@dataclass(slots=True)
class ExecutionBlueprint:
    """Extracted artifacts from agent execution history."""
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    commands_run: list[str] = field(default_factory=list)
    errors_encountered: list[str] = field(default_factory=list)
    languages_used: list[str] = field(default_factory=list)
    tools_used: dict[str, str] = field(default_factory=dict)


# Time formatting (Korean relative time)

def format_relative_time(timestamp: str | float | datetime) -> str:
    """Format a timestamp as Korean relative time.

    Examples: "방금", "3분 전", "2시간 전", "1일 전", "3개월 전"
    """
    if isinstance(timestamp, (int, float)):
        ts = datetime.fromtimestamp(timestamp, tz=UTC)
    elif isinstance(timestamp, str):
        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            return timestamp
    elif isinstance(timestamp, datetime):
        ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=UTC)
    else:
        return str(timestamp)

    now = datetime.now(UTC)
    delta = now - ts
    total_seconds = int(delta.total_seconds())

    if total_seconds < 60:
        return "방금"

    minutes = total_seconds // 60
    if minutes < 60:
        return f"{minutes}분 전"

    hours = minutes // 60
    if hours < 24:
        return f"{hours}시간 전"

    days = hours // 24
    if days < 30:
        return f"{days}일 전"

    months = days // 30
    return f"{months}개월 전"


# Intent extraction

def extract_intent_from_goal(
    goal: str,
    hint: str | None = None,
) -> Intent:
    """Extract a structured Intent from a natural language goal.

    Uses keyword matching for fast extraction. The optional *hint*
    (e.g. from goal classifier) biases the result.
    """
    goal_lower = goal.lower()

    # Domain detection
    domain = Domain.GENERAL
    if any(k in goal_lower for k in ("file", "read", "write", "edit", "create", "delete", "folder", "directory")):
        domain = Domain.FILE
    elif any(k in goal_lower for k in ("browse", "click", "navigate", "page", "website", "url")):
        domain = Domain.BROWSER
    elif any(k in goal_lower for k in ("run", "execute", "command", "bash", "shell", "terminal", "test", "build")):
        domain = Domain.PROCESS
    elif any(k in goal_lower for k in ("search", "web", "google", "fetch", "download", "http")):
        domain = Domain.NETWORK
    elif any(k in goal_lower for k in ("git", "commit", "push", "pull", "branch", "merge")):
        domain = Domain.GIT
    elif any(k in goal_lower for k in ("remember", "memory", "recall", "forget", "save")):
        domain = Domain.MEMORY
    elif any(k in goal_lower for k in ("schedule", "cron", "timer", "remind", "alarm")):
        domain = Domain.SCHEDULE

    # Apply hint override
    if hint:
        hint_lower = hint.lower()
        if hint_lower in ("web", "research"):
            domain = Domain.NETWORK
        elif hint_lower in ("code_modify",):
            domain = Domain.FILE
        elif hint_lower in ("execution",):
            domain = Domain.PROCESS
        elif hint_lower in ("browser",):
            domain = Domain.BROWSER

    # Action detection
    action = "unknown"
    if any(k in goal_lower for k in ("create", "make", "build", "implement", "add", "write")):
        action = "create"
    elif any(k in goal_lower for k in ("edit", "modify", "change", "update", "fix", "refactor")):
        action = "modify"
    elif any(k in goal_lower for k in ("delete", "remove", "drop")):
        action = "delete"
    elif any(k in goal_lower for k in ("read", "show", "display", "list", "analyze", "explain")):
        action = "read"
    elif any(k in goal_lower for k in ("run", "execute", "test", "launch", "start")):
        action = "execute"
    elif any(k in goal_lower for k in ("search", "find", "look", "google")):
        action = "search"

    # Target extraction - find paths or quoted strings
    target = ""
    path_match = PATH_PATTERN.search(goal)
    if path_match:
        target = path_match.group(1)
    else:
        # Try quoted strings
        quoted = re.search(r'["\']([^"\']+)["\']', goal)
        if quoted:
            target = quoted.group(1)
        else:
            # Use last significant word
            words = [w for w in goal.split() if len(w) > 3 and not w.startswith("-")]
            if words:
                target = words[-1]

    return Intent(
        domain=domain,
        action=action,
        target=target,
        confidence=0.7,
    )


# Build agent memory context (8 layers)

async def build_agent_memory_context(
    goal: str,
    memory_manager: Any,
    options: dict[str, Any] | None = None,
) -> AgentMemoryContext:
    """Build the full agent memory context from 8 layers.

    Layers:
    1. UserModel - user preferences and facts
    2. User Profile - identity information
    3. Project Context - project-specific memory
    4. Relevant Past Work - semantically similar episodes
    5. Recent Commands - command history
    6. Safety Rules - learned safety constraints
    7. Reflexion Lessons - past mistakes and learnings
    8. Temporal Context - session digests, time-awareness
    """
    ctx = AgentMemoryContext()
    parts: list[str] = []

    try:
        # Ensure memory is initialized
        if hasattr(memory_manager, "initialize"):
            await memory_manager.initialize()

        working = getattr(memory_manager, "working", None)

        # Layer 1 & 2: User Model / Profile
        if working and working.facts:
            ctx.preferences = dict(working.facts)
            pref_lines: list[str] = []
            for key, value in list(working.facts.items())[:20]:
                pref_lines.append(f"  - {key}: {value}")
            if pref_lines:
                parts.append("### User Preferences\n" + "\n".join(pref_lines))

        # Layer 3: Project Context (DB facts + MEMORY.md file)
        try:
            project_lines: list[str] = []

            # 3a: DB-stored project facts
            project_facts = await _search_facts(memory_manager, "project")
            for fact in (project_facts or [])[:10]:
                project_lines.append(f"  - {fact.key}: {fact.value}")

            # 3b: Project MEMORY.md file (user-editable, Claude Code style)
            try:
                import os

                from rune.memory.project_memory import read_project_memory_head
                md_content = read_project_memory_head(
                    os.getcwd(),
                    {"max_lines": PROJECT_MEMORY_MAX_LINES, "max_chars": PROJECT_MEMORY_MAX_CHARS},
                )
                if md_content.strip():
                    project_lines.append(f"\n{md_content.strip()}")
            except Exception:
                pass

            if project_lines:
                text = "\n".join(project_lines)
                if len(text) > PROJECT_MEMORY_MAX_CHARS:
                    text = text[:PROJECT_MEMORY_MAX_CHARS] + "..."
                parts.append("### Project Context\n" + text)
        except Exception:
            pass

        # Layer 4: Relevant Past Work
        try:
            search_results = await memory_manager.search(goal, k=5)
            if search_results:
                history_lines: list[str] = []
                for sr in search_results:
                    summary = getattr(sr, "summary", "") or getattr(sr, "text", str(sr))
                    ts = getattr(sr, "timestamp", "")
                    relative = format_relative_time(ts) if ts else ""
                    line = f"  - {summary}"
                    if relative:
                        line += f" ({relative})"
                    history_lines.append(line)
                    ctx.relevant_history.append(summary)
                if history_lines:
                    parts.append("### Relevant Past Work\n" + "\n".join(history_lines))
        except Exception:
            pass

        # Layer 4b: Open Commitments (Phase 1: Episode Memory)
        try:
            store = getattr(memory_manager, "store", None)
            if store is not None and hasattr(store, "get_open_commitments"):
                open_commitments = store.get_open_commitments(limit=5)
                if open_commitments:
                    commit_lines: list[str] = []
                    for c in open_commitments:
                        line = f"  - {c['text']}"
                        if c.get("deadline"):
                            line += f" (deadline: {c['deadline']})"
                        line += f" — from: {c['task_summary'][:60]}"
                        commit_lines.append(line)
                    parts.append("### Open Commitments\n" + "\n".join(commit_lines))
        except Exception:
            pass

        # Layer 5: Recent Commands
        if working and working.recent_commands:
            recent = working.recent_commands[:10]
            ctx.recent_commands = list(recent)
            cmd_lines = [f"  - `{cmd}`" for cmd in recent]
            parts.append("### Recent Commands\n" + "\n".join(cmd_lines))

        # Layer 6: Safety Rules
        if working and working.safety_rules:
            ctx.safety_rules = list(working.safety_rules)
            rule_lines: list[str] = []
            for rule in working.safety_rules[:10]:
                rule_type = rule.get("type", "")
                pattern = rule.get("pattern", "")
                reason = rule.get("reason", "")
                rule_lines.append(f"  - [{rule_type}] {pattern}: {reason}")
            if rule_lines:
                parts.append("### Safety Rules\n" + "\n".join(rule_lines))

        # Layer 7: Reflexion Lessons
        try:
            lesson_facts = await _search_facts(memory_manager, "lesson")
            if lesson_facts:
                lesson_lines: list[str] = []
                for fact in lesson_facts[:5]:
                    lesson_lines.append(f"  - {fact.key}: {fact.value}")
                if lesson_lines:
                    parts.append("### Lessons Learned\n" + "\n".join(lesson_lines))
        except Exception:
            pass

        # Layer 8: Temporal Context
        now = datetime.now(UTC)
        temporal = (
            f"### Temporal Context\n"
            f"  - Current time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"  - Day of week: {now.strftime('%A')}"
        )
        parts.append(temporal)

    except Exception as exc:
        log.warning("memory_context_build_failed", error=str(exc))

    ctx.formatted = "\n\n".join(parts) if parts else ""
    return ctx


async def _search_facts(memory_manager: Any, category: str) -> list[Any]:
    """Search facts by category from markdown (MEMORY.md + learned.md)."""
    try:
        from rune.memory.markdown_store import parse_learned_md, parse_memory_md
        from rune.memory.types import Fact

        results: list[Fact] = []

        # From MEMORY.md
        sections = parse_memory_md()
        for section, lines in sections.items():
            if category and section.lower() != category:
                continue
            for line in lines:
                if ":" in line:
                    key, _, value = line.partition(":")
                    results.append(Fact(category=section.lower(), key=key.strip(), value=value.strip()))

        # From learned.md
        for fact in parse_learned_md():
            if not category or fact["category"] == category:
                results.append(Fact(
                    category=fact["category"], key=fact["key"],
                    value=fact["value"], confidence=fact["confidence"],
                ))

        return results
    except Exception:
        return []


# Save agent result to memory

def _extract_files_from_text(text: str) -> list[str]:
    """Extract file paths mentioned in text."""
    paths: list[str] = []
    for match in PATH_PATTERN.finditer(text):
        path = match.group(1)
        if len(path) > 4 and not path.startswith("http"):
            paths.append(path)
    return list(dict.fromkeys(paths))[:20]  # dedup, limit 20


async def save_agent_result_to_memory(
    goal: str,
    result: Any,
    memory_manager: Any,
    conversation_id: str = "",
    classification_hint: str | None = None,
) -> None:
    """Save an agent execution result to episodic memory.

    Creates an Episode with task summary, intent, result text, and files.
    Structured extraction (commitments, lessons, entities) is handled by
    background consolidation via LLM - no regex patterns.
    """
    try:
        result_text = ""
        success = False
        lessons = ""
        duration_ms = 0.0

        if hasattr(result, "reason"):
            result_text = str(result.reason)
            success = result.reason in ("completed", "verified")
            duration_ms = getattr(result, "duration_ms", 0.0) or 0.0
        elif isinstance(result, dict):
            result_text = str(result.get("reason", result.get("output", "")))
            success = result.get("success", False)
            duration_ms = result.get("duration_ms", 0.0) or 0.0
        else:
            result_text = str(result)

        # Extract intent
        intent = extract_intent_from_goal(goal, classification_hint)

        # Extract file paths (deterministic, no LLM needed)
        import json as _json
        combined_text = f"{goal}\n{result_text}"
        files = _extract_files_from_text(combined_text)

        # Generate lessons from both success and failure
        if not success:
            lessons = f"Task failed: {result_text[:200]}. Consider alternative approaches."
        else:
            # Success lessons: capture what worked for future reference
            lesson_parts: list[str] = []
            if intent.domain:
                lesson_parts.append(f"domain={intent.domain}")
            if intent.action:
                lesson_parts.append(f"action={intent.action}")
            if duration_ms > 0:
                lesson_parts.append(f"took {duration_ms / 1000:.1f}s")
            if files:
                lesson_parts.append(f"files={','.join(files[:3])}")
            if lesson_parts:
                lessons = "Success: " + "; ".join(lesson_parts)

        # Utility: +1 (golden), -1 (warning).
        # Simple rule: trust the agent's completion status.
        # Edge cases (Guardian refusal counted as success) are acceptable —
        # they're +1 among many +1s and don't skew the pattern.
        _utility = 1 if success else -1

        episode = Episode(
            task_summary=goal[:500],
            intent=f"{intent.domain}:{intent.action}:{intent.target}",
            result=result_text[:1000],
            lessons=lessons,
            conversation_id=conversation_id,
            importance=0.7,
            files_touched=_json.dumps(files) if files else "",
            duration_ms=duration_ms,
            utility=_utility,
        )

        await memory_manager.save_episode(episode)

        # Save learned pattern (time-slot activity tracking)
        try:
            from datetime import datetime, timezone
            from rune.memory.store import get_memory_store

            now = datetime.now(timezone.utc)
            hour = now.hour
            if hour < 6:
                slot = "night"
            elif hour < 12:
                slot = "morning"
            elif hour < 18:
                slot = "afternoon"
            else:
                slot = "evening"
            day_type = "weekday" if now.weekday() < 5 else "weekend"
            activity = f"{intent.domain}:{intent.action}" if intent.domain else goal[:30]

            mem_store = get_memory_store()
            mem_store.save_learned_pattern(
                time_slot=slot,
                activity=activity,
                day_type=day_type,
                avg_duration_minutes=duration_ms / 60_000 if duration_ms else 0,
            )
        except Exception:
            pass  # Pattern tracking must never block episode saving

        # Rule outcome feedback: update confidence of active rules
        domain = intent.domain or "code_modify"
        try:
            from rune.memory.rule_learner import update_rules_from_outcome
            update_rules_from_outcome(domain, success, goal=goal, error_message=result_text[:300])
        except Exception:
            pass  # Rule feedback must never block episode saving

        # Rule Learner: trigger on failure
        if not success:
            try:
                from rune.memory.rule_learner import learn_from_failures
                from rune.memory.store import get_memory_store
                await learn_from_failures(get_memory_store(), domain)
            except Exception:
                pass  # Rule learning must never block episode saving

        # Write daily log entry (markdown)
        try:
            from rune.memory.markdown_store import append_daily_entry

            actions: list[str] = []
            if files:
                for fp in files[:3]:
                    actions.append(f"touched {fp}")
            if not actions and result_text:
                actions.append(result_text[:100])

            append_daily_entry(
                title=goal[:100],
                actions=actions,
                lessons=[lessons] if lessons else [],
            )
        except Exception:
            pass  # Daily log failure must never block episode saving

        # Trigger background consolidation (LLM-based extraction)
        try:
            import asyncio

            from rune.memory.consolidation import consolidate_episode
            asyncio.create_task(consolidate_episode(episode.id))
        except Exception:
            pass  # Consolidation failure must never block episode saving

        log.debug(
            "agent_result_saved",
            goal=goal[:80],
            success=success,
            files=len(files),
        )

    except Exception as exc:
        log.warning("save_agent_result_failed", error=str(exc))


# Artifact extraction from execution history

def extract_artifacts_from_history(
    history: list[dict[str, Any]],
) -> ExecutionBlueprint:
    """Extract an execution blueprint from a tool call history.

    Scans through the history for file operations, commands, and errors.
    """
    blueprint = ExecutionBlueprint()

    for entry in history:
        tool = entry.get("tool", "")
        params = entry.get("params", {})
        result = entry.get("result", {})
        success = result.get("success", True) if isinstance(result, dict) else True

        match tool:
            case "file_write":
                path = params.get("file_path") or params.get("path", "")
                if path:
                    blueprint.files_created.append(path)

            case "file_edit":
                path = params.get("file_path") or params.get("path", "")
                if path:
                    blueprint.files_modified.append(path)

            case "file_delete":
                path = params.get("file_path") or params.get("path", "")
                if path:
                    blueprint.files_deleted.append(path)

            case "bash_execute":
                cmd = params.get("command", "")
                if cmd:
                    blueprint.commands_run.append(cmd[:200])

            case _:
                if tool:
                    blueprint.tools_used[tool] = blueprint.tools_used.get(tool, "used")

        if not success:
            error = (
                result.get("error", "")
                if isinstance(result, dict) else str(result)
            )
            if error:
                blueprint.errors_encountered.append(f"[{tool}] {error[:150]}")

    # Detect languages from file extensions
    all_files = blueprint.files_created + blueprint.files_modified
    blueprint.languages_used = detect_languages_from_files(all_files)

    return blueprint


def format_artifact_lessons(artifacts: ExecutionBlueprint) -> list[str]:
    """Format an execution blueprint into human-readable lesson strings."""
    lessons: list[str] = []

    if artifacts.files_created:
        lessons.append(f"Created files: {', '.join(artifacts.files_created[:10])}")

    if artifacts.files_modified:
        lessons.append(f"Modified files: {', '.join(artifacts.files_modified[:10])}")

    if artifacts.files_deleted:
        lessons.append(f"Deleted files: {', '.join(artifacts.files_deleted[:5])}")

    if artifacts.commands_run:
        lessons.append(f"Commands run: {len(artifacts.commands_run)}")

    if artifacts.errors_encountered:
        lessons.append(f"Errors encountered: {len(artifacts.errors_encountered)}")
        for err in artifacts.errors_encountered[:3]:
            lessons.append(f"  - {err}")

    if artifacts.languages_used:
        lessons.append(f"Languages: {', '.join(artifacts.languages_used)}")

    return lessons


# Language & tool detection

_LANGUAGE_EXTENSIONS: dict[str, str] = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript (React)",
    ".jsx": "JavaScript (React)",
    ".rs": "Rust",
    ".go": "Go",
    ".java": "Java",
    ".kt": "Kotlin",
    ".swift": "Swift",
    ".c": "C",
    ".cpp": "C++",
    ".h": "C/C++ Header",
    ".cs": "C#",
    ".rb": "Ruby",
    ".php": "PHP",
    ".sh": "Shell",
    ".bash": "Bash",
    ".zsh": "Zsh",
    ".sql": "SQL",
    ".html": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".vue": "Vue",
    ".svelte": "Svelte",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".json": "JSON",
    ".toml": "TOML",
    ".md": "Markdown",
    ".dockerfile": "Docker",
}

_LANGUAGE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bpython[3]?\b|\bpip\b|\bpytest\b|\bdjango\b|\bflask\b", re.I), "Python"),
    (re.compile(r"\bnode\b|\bnpm\b|\byarn\b|\bpnpm\b|\breact\b|\bnext\.js\b", re.I), "JavaScript/TypeScript"),
    (re.compile(r"\bcargo\b|\brustc\b|\brust\b", re.I), "Rust"),
    (re.compile(r"\bgo\s+(build|run|test|get)\b", re.I), "Go"),
    (re.compile(r"\bjava\b|\bmaven\b|\bgradle\b|\bspring\b", re.I), "Java"),
    (re.compile(r"\bdocker\b|\bdocker-compose\b|\bcontainerfile\b", re.I), "Docker"),
    (re.compile(r"\bsql\b|\bpostgres\b|\bmysql\b|\bsqlite\b", re.I), "SQL"),
]

_TOOL_PATTERNS: dict[str, re.Pattern[str]] = {
    "git": re.compile(r"\bgit\s+(add|commit|push|pull|clone|branch|checkout|merge|rebase|stash)\b", re.I),
    "docker": re.compile(r"\bdocker\s+(build|run|compose|push|pull|stop|rm)\b", re.I),
    "npm": re.compile(r"\bnpm\s+(install|run|test|build|publish)\b", re.I),
    "pip": re.compile(r"\bpip[3]?\s+(install|uninstall|freeze)\b", re.I),
    "pytest": re.compile(r"\bpytest\b", re.I),
    "vitest": re.compile(r"\bvitest\b", re.I),
    "jest": re.compile(r"\bjest\b", re.I),
    "cargo": re.compile(r"\bcargo\s+(build|test|run|add|bench)\b", re.I),
    "make": re.compile(r"\bmake\b", re.I),
}


def detect_languages(text: str) -> list[str]:
    """Detect programming languages mentioned in text."""
    found: list[str] = []
    for pattern, lang in _LANGUAGE_PATTERNS:
        if pattern.search(text) and lang not in found:
            found.append(lang)
    return found


def detect_languages_from_files(file_paths: list[str]) -> list[str]:
    """Detect programming languages from file extensions."""
    import os
    found: list[str] = []
    for path in file_paths:
        _, ext = os.path.splitext(path.lower())
        lang = _LANGUAGE_EXTENSIONS.get(ext)
        if lang and lang not in found:
            found.append(lang)
    return found


def detect_tools(text: str) -> dict[str, str]:
    """Detect development tools mentioned in text.

    Returns a mapping of tool name to a sample match.
    """
    found: dict[str, str] = {}
    for tool_name, pattern in _TOOL_PATTERNS.items():
        match = pattern.search(text)
        if match:
            found[tool_name] = match.group(0)
    return found


# Auto-skill quality scoring

def compute_auto_skill_quality_score(result: Any) -> float:
    """Compute a quality score (0.0-1.0) for an agent result.

    Higher scores indicate the result is a good candidate for
    auto-skill extraction.
    """
    score = 0.0

    # Success is the primary factor
    success = False
    if hasattr(result, "reason"):
        success = result.reason in ("completed", "verified")
    elif isinstance(result, dict):
        success = result.get("success", False)

    if not success:
        return 0.0

    score += 0.4  # Base score for success

    # Evidence score (from completion trace)
    evidence_score = 0.0
    if hasattr(result, "evidence_score"):
        evidence_score = float(result.evidence_score)
    elif isinstance(result, dict):
        evidence_score = float(result.get("evidence_score", 0))

    if evidence_score > 0.8:
        score += 0.3
    elif evidence_score > 0.5:
        score += 0.2
    elif evidence_score > 0:
        score += 0.1

    # Steps used (fewer steps = more efficient)
    final_step = 0
    if hasattr(result, "final_step"):
        final_step = int(result.final_step)
    elif isinstance(result, dict):
        final_step = int(result.get("final_step", 0))

    if 0 < final_step <= 5:
        score += 0.2
    elif final_step <= 10:
        score += 0.1

    # Token efficiency
    total_tokens = 0
    if hasattr(result, "total_tokens_used"):
        total_tokens = int(result.total_tokens_used)
    elif isinstance(result, dict):
        total_tokens = int(result.get("total_tokens_used", 0))

    if 0 < total_tokens < 10_000:
        score += 0.1

    return min(1.0, score)


# Auto-skill generation (pattern-based, no LLM required)

@dataclass(slots=True)
class ToolTraceEntry:
    """A single tool invocation within an execution trace."""

    tool_name: str
    params: dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    success: bool = True


def _parameterise_value(value: Any) -> Any:
    """Replace concrete values with template placeholders where appropriate.

    Strings that look like file paths, URLs, or long free-text get replaced
    with a ``{{placeholder}}`` so the skill template stays reusable.
    """
    if not isinstance(value, str):
        return value
    # Detect file paths
    if re.match(r"^[./~].*[/\\]", value):
        return "{{file_path}}"
    # Detect URLs
    if re.match(r"^https?://", value):
        return "{{url}}"
    # Detect long free-text (e.g. search queries, goals)
    if len(value) > 80:
        return "{{text}}"
    return value


def _build_param_template(params: dict[str, Any]) -> dict[str, Any]:
    """Convert concrete parameters into a reusable template."""
    return {k: _parameterise_value(v) for k, v in params.items()}


def _deduplicate_steps(
    steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove consecutive duplicate tool calls (same tool + same template)."""
    deduped: list[dict[str, Any]] = []
    for step in steps:
        if deduped and deduped[-1] == step:
            continue
        deduped.append(step)
    return deduped


def _detect_pattern_name(steps: list[dict[str, Any]]) -> str:
    """Return a short pattern label based on the tool sequence."""
    tool_names = [s["tool"] for s in steps]
    unique = list(dict.fromkeys(tool_names))  # preserve order, dedupe

    if len(unique) == 1:
        return f"single_{unique[0]}"

    # Read-then-write patterns
    read_tools = {"file_read", "code_analyze", "code_find_def", "code_find_refs",
                  "file_search", "file_list", "project_map"}
    write_tools = {"file_write", "file_edit", "bash_execute"}
    has_read = any(t in read_tools for t in unique)
    has_write = any(t in write_tools for t in unique)

    if has_read and has_write:
        return "read_modify"
    if has_read and not has_write:
        return "research"
    if has_write and not has_read:
        return "generate"

    return "multi_step"


def extract_skill_template(
    goal: str,
    trace: list[ToolTraceEntry],
    *,
    intent: Intent | None = None,
) -> dict[str, Any] | None:
    """Extract a reusable skill template from a sequence of tool calls.

    Uses pattern matching on the tool sequence (no LLM call) so the
    operation is fast and fully local.

    Returns a skill definition dict with ``steps`` (list of tool-call
    templates) or ``None`` if the trace is too short / not generalisable.
    """
    # Filter to successful calls only
    successful = [t for t in trace if t.success]
    if len(successful) < 1:
        return None

    # Build step templates
    raw_steps: list[dict[str, Any]] = []
    for entry in successful:
        raw_steps.append({
            "tool": entry.tool_name,
            "params_template": _build_param_template(entry.params),
        })

    steps = _deduplicate_steps(raw_steps)
    pattern = _detect_pattern_name(steps)

    # Derive a description from the goal and pattern
    description = (
        f"Auto-extracted '{pattern}' skill: {goal[:160]}"
    )

    # Build a deterministic fingerprint so we can detect duplicates
    fp_input = json.dumps(
        [{"tool": s["tool"], "tpl": sorted(s["params_template"].keys())} for s in steps],
        sort_keys=True,
    )
    fingerprint = hashlib.sha256(fp_input.encode()).hexdigest()[:12]

    return {
        "steps": steps,
        "pattern": pattern,
        "description": description,
        "domain": intent.domain if intent else "general",
        "action": intent.action if intent else "unknown",
        "fingerprint": fingerprint,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _build_refinement_prompt(
    goal: str,
    steps: list[dict[str, Any]],
    pattern: str,
) -> str:
    """Build a prompt asking the LLM to clean up auto-extracted skill steps."""
    step_lines: list[str] = []
    for idx, step in enumerate(steps, 1):
        params_str = ", ".join(
            f"{k}={v}" for k, v in step.get("params_template", {}).items()
        )
        step_lines.append(f"  {idx}. tool={step['tool']}  params={params_str}")

    return (
        "You are a skill-template editor. Below is an auto-extracted sequence of "
        "tool-call steps from a successful agent execution. Clean up and improve "
        "these steps so the skill is reusable for similar future goals.\n\n"
        f"Original goal: {goal[:300]}\n"
        f"Pattern type: {pattern}\n\n"
        "Raw steps:\n" + "\n".join(step_lines) + "\n\n"
        "Instructions:\n"
        "- Remove redundant or unnecessary steps.\n"
        "- Ensure template placeholders ({{file_path}}, {{url}}, {{text}}) are "
        "used for values that change between invocations.\n"
        "- Reorder steps if a better sequence exists.\n"
        "- Keep each step as: tool=<name> params=<key1>=<val1>, <key2>=<val2>\n"
        "- Output ONLY the refined steps, one per line, numbered.\n"
        "- Do NOT add commentary outside the step list."
    )


def _parse_refined_steps(
    raw: str,
    original_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parse LLM-refined steps back into structured dicts.

    Falls back to *original_steps* if parsing fails or the result is
    degenerate (too few / too many steps).
    """
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    parsed: list[dict[str, Any]] = []

    for line in lines:
        # Expected format: "1. tool=file_read params=file_path={{file_path}}"
        # or variants like "1. tool=file_read  params=file_path={{file_path}}, command=ls"
        tool_match = re.search(r"tool\s*=\s*(\w+)", line)
        if not tool_match:
            continue
        tool_name = tool_match.group(1)

        params_template: dict[str, Any] = {}
        params_match = re.search(r"params\s*=\s*(.*)", line)
        if params_match:
            params_raw = params_match.group(1).strip()
            # Parse key=value pairs separated by commas
            for pair in re.split(r",\s*", params_raw):
                kv = pair.split("=", 1)
                if len(kv) == 2:
                    params_template[kv[0].strip()] = kv[1].strip()

        parsed.append({"tool": tool_name, "params_template": params_template})

    # Validate: must have reasonable number of steps
    if (
        len(parsed) < _REFINEMENT_MIN_STEPS
        or len(parsed) > _REFINEMENT_MAX_STEPS
    ):
        log.debug(
            "refined_steps_invalid_count",
            parsed=len(parsed),
            falling_back=True,
        )
        return original_steps

    return parsed


async def refine_skill_steps(
    goal: str,
    steps: list[dict[str, Any]],
    pattern: str,
    refiner: LLMRefiner | None,
) -> list[dict[str, Any]]:
    """Refine auto-extracted skill steps via an LLM, falling back to
    the original steps when no refiner is available or on error."""
    if refiner is None:
        return steps

    prompt = _build_refinement_prompt(goal, steps, pattern)
    try:
        raw = await refiner.refine(prompt, max_tokens=_REFINEMENT_MAX_TOKENS)
        refined = _parse_refined_steps(raw, steps)
        log.debug(
            "skill_steps_refined",
            original=len(steps),
            refined=len(refined),
        )
        return refined
    except Exception as exc:
        log.warning("skill_refinement_failed", error=str(exc))
        return steps


async def maybe_generate_skill(
    goal: str,
    result: Any,
    intent: Intent | None = None,
    trace: list[ToolTraceEntry] | None = None,
    refiner: LLMRefiner | None = None,
) -> dict[str, Any] | None:
    """Attempt to generate a reusable skill from a successful execution.

    When a ``trace`` (list of :class:`ToolTraceEntry`) is provided the
    function uses local pattern matching to extract a generalised skill
    template and registers it in the :class:`SkillRegistry`.

    An optional *refiner* (:class:`LLMRefiner`) is used to improve the
    auto-extracted steps via an LLM call.  When ``None`` or when the LLM
    call fails, the pattern-extracted steps are kept as-is.

    Returns a skill definition dict or ``None`` if the result is not
    suitable for skill extraction.
    """
    quality = compute_auto_skill_quality_score(result)
    if quality < 0.6:
        log.debug("skill_generation_skipped", quality=quality)
        return None

    skill_name = generate_skill_name(goal)
    if skill_name is None:
        return None

    # --- Skill template extraction via pattern matching ---
    template: dict[str, Any] | None = None
    if trace:
        template = extract_skill_template(goal, trace, intent=intent)

    if template is not None:
        # --- LLM refinement of steps (optional) ---
        refined_steps = await refine_skill_steps(
            goal,
            template["steps"],
            template["pattern"],
            refiner,
        )
        template["steps"] = refined_steps

        # Build a Skill body that encodes the steps as structured YAML-like
        # markdown so it can be rendered by the skill runner.
        body_lines: list[str] = [f"# Auto-generated skill: {skill_name}", ""]
        for idx, step in enumerate(template["steps"], 1):
            body_lines.append(f"## Step {idx}: {step['tool']}")
            for pkey, pval in step["params_template"].items():
                body_lines.append(f"- {pkey}: {pval}")
            body_lines.append("")
        body = "\n".join(body_lines)

        skill = Skill(
            name=skill_name,
            description=template["description"],
            body=body,
            scope="user",
            author="auto",
            metadata={
                "quality_score": quality,
                "pattern": template["pattern"],
                "fingerprint": template["fingerprint"],
                "domain": template["domain"],
                "action": template["action"],
                "steps": template["steps"],
                "generated_at": template["generated_at"],
            },
        )

        # Register in the global skill registry
        registry = get_skill_registry()
        existing = registry.get(skill_name)
        if existing is None:
            registry.register(skill)
            log.info(
                "skill_auto_registered",
                name=skill_name,
                pattern=template["pattern"],
                steps=len(template["steps"]),
            )

        return {
            "name": skill_name,
            "description": template["description"],
            "quality_score": quality,
            "domain": template["domain"],
            "action": template["action"],
            "pattern": template["pattern"],
            "steps": template["steps"],
            "fingerprint": template["fingerprint"],
            "generated_at": template["generated_at"],
        }

    # Fallback: no trace provided - return a basic skill definition
    return {
        "name": skill_name,
        "description": goal[:200],
        "quality_score": quality,
        "domain": intent.domain if intent else "general",
        "action": intent.action if intent else "unknown",
        "generated_at": datetime.now(UTC).isoformat(),
    }


def generate_skill_name(goal: str) -> str | None:
    """Generate a snake_case skill name from a goal description.

    Returns ``None`` if the goal is too vague for a meaningful name.
    """
    # Remove common stop words and articles
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "do", "does", "did", "will", "would", "could", "should",
        "can", "may", "might", "shall", "to", "of", "in", "on",
        "at", "by", "for", "with", "from", "and", "or", "but",
        "not", "no", "this", "that", "it", "its", "my", "your",
        "his", "her", "our", "me", "i", "you", "he", "she", "we",
        "please", "just", "also", "very", "really", "now", "then",
    }

    # Clean and split
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", goal.lower())
    words = [w for w in cleaned.split() if w not in stop_words and len(w) > 1]

    if len(words) < 2:
        return None

    # Take first 4 significant words
    name_parts = words[:4]
    name = "_".join(name_parts)

    # Validate length
    if len(name) < 5 or len(name) > 60:
        return None

    return name


# Record command to memory

async def record_command_to_memory(
    command: str,
    success: bool,
    memory_manager: Any,
) -> None:
    """Record a command execution to memory for future reference."""
    try:
        if hasattr(memory_manager, "log_command"):
            memory_manager.log_command(command, success)
        else:
            log.debug("memory_manager_no_log_command")
    except Exception as exc:
        log.warning("record_command_failed", error=str(exc))
