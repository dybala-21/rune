"""Markdown-primary storage for RUNE memory (Zone 1 + Zone 2).

Parses and writes:
  - MEMORY.md (Zone 1, human-owned)
  - learned.md (Zone 2, machine-written)
  - daily/*.md (Zone 2, session logs)
  - user-profile.md (Zone 2, mixed ownership)
  - rules.md (Zone 1, project safety rules)
"""

from __future__ import annotations

import fcntl
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rune.utils.logger import get_logger
from rune.utils.paths import rune_home

log = get_logger(__name__)

# learned.md line pattern: - [category] key: value (confidence)
# Category may contain colons (e.g. "rule:code_modify") so we use [^\]]+ instead of \w+
_LEARNED_RE = re.compile(
    r"^- \[(?P<category>[^\]]+)\]\s*(?P<key>[^:]+):\s*(?P<value>.+?)"
    r"(?:\s*\((?P<confidence>[\d.]+)\))?\s*$"
)

_SOFT_CAP = 200


def memory_dir() -> Path:
    d = rune_home() / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _lock_path() -> Path:
    return memory_dir() / ".memory.lock"


def _atomic_write(target: Path, content: str) -> None:
    """Write content to target atomically with advisory lock."""
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = _lock_path()
    fd_lock = os.open(str(lock), os.O_WRONLY | os.O_CREAT)
    try:
        fcntl.flock(fd_lock, fcntl.LOCK_EX)
        fd, tmp_str = tempfile.mkstemp(
            dir=str(target.parent), prefix=".rune_", suffix=".tmp",
        )
        tmp = Path(tmp_str)
        try:
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(target)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        finally:
            os.close(fd)
    finally:
        fcntl.flock(fd_lock, fcntl.LOCK_UN)
        os.close(fd_lock)


def _atomic_append(target: Path, content: str) -> None:
    """Append content to target with advisory lock. Creates file if missing."""
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = _lock_path()
    fd_lock = os.open(str(lock), os.O_WRONLY | os.O_CREAT)
    try:
        fcntl.flock(fd_lock, fcntl.LOCK_EX)
        with open(target, "a", encoding="utf-8") as f:
            f.write(content)
    finally:
        fcntl.flock(fd_lock, fcntl.LOCK_UN)
        os.close(fd_lock)


def _backup(target: Path) -> None:
    """Copy target to .state/<name>.bak before modifying."""
    if not target.exists():
        return
    bak_dir = memory_dir() / ".state"
    bak_dir.mkdir(parents=True, exist_ok=True)
    bak = bak_dir / f"{target.name}.bak"
    try:
        bak.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    except OSError:
        pass


# MEMORY.md parsing

def parse_memory_md(path: Path | None = None) -> dict[str, list[str]]:
    """Parse MEMORY.md into {section: [lines]}.

    Returns a dict keyed by H1 heading (without #), values are bullet lines.
    """
    if path is None:
        path = memory_dir() / "MEMORY.md"
    if not path.exists():
        return {}

    sections: dict[str, list[str]] = {}
    current_section = ""

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            current_section = stripped[2:].strip()
            sections.setdefault(current_section, [])
        elif stripped.startswith("- ") and current_section:
            sections[current_section].append(stripped[2:].strip())

    return sections


def memory_md_has_key(key: str, path: Path | None = None) -> bool:
    """Check if a key exists in any section of MEMORY.md."""
    sections = parse_memory_md(path)
    key_lower = key.lower()
    for lines in sections.values():
        for line in lines:
            if ":" in line and line.split(":")[0].strip().lower() == key_lower:
                return True
    return False


def append_to_memory_md(section: str, line: str, path: Path | None = None) -> None:
    """Append a bullet line under an H1 section in MEMORY.md."""
    if path is None:
        path = memory_dir() / "MEMORY.md"

    _backup(path)

    content = ""
    if path.exists():
        content = path.read_text(encoding="utf-8")

    heading = f"# {section}"
    bullet = f"- {line}"

    if heading in content:
        lines = content.split("\n")
        insert_idx = len(lines)
        in_section = False
        for i, l in enumerate(lines):
            if l.strip() == heading:
                in_section = True
                continue
            if in_section and l.strip().startswith("# "):
                insert_idx = i
                break
        lines.insert(insert_idx, bullet)
        content = "\n".join(lines)
    else:
        if content and not content.endswith("\n"):
            content += "\n"
        content += f"\n{heading}\n\n{bullet}\n"

    _atomic_write(path, content)


# learned.md parsing

def parse_learned_md(path: Path | None = None) -> list[dict[str, Any]]:
    """Parse learned.md into a list of fact dicts.

    Each entry: {category, key, value, confidence, line_num, raw}.
    Lenient: broken lines are parsed as free-text with confidence 0.3.
    """
    if path is None:
        path = memory_dir() / "learned.md"
    if not path.exists():
        return []

    facts: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        m = _LEARNED_RE.match(stripped)
        if m:
            conf_str = m.group("confidence")
            facts.append({
                "category": m.group("category").strip(),
                "key": m.group("key").strip(),
                "value": m.group("value").strip(),
                "confidence": float(conf_str) if conf_str else 0.5,
                "line_num": i,
                "raw": stripped,
            })
        elif stripped.startswith("- "):
            # Broken format fallback
            text = stripped[2:].strip()
            facts.append({
                "category": "general",
                "key": text[:40].replace(" ", "_").lower(),
                "value": text,
                "confidence": 0.3,
                "line_num": i,
                "raw": stripped,
            })

    return facts


def learned_md_has_key(key: str, path: Path | None = None) -> str | None:
    """Return the current value for key in learned.md, or None."""
    for fact in parse_learned_md(path):
        if fact["key"].lower() == key.lower():
            return fact["value"]
    return None


def save_learned_fact(
    category: str,
    key: str,
    value: str,
    confidence: float = 0.5,
    path: Path | None = None,
) -> None:
    """Add or update a fact in learned.md.

    The entire read-modify-write happens under a single advisory lock
    to prevent concurrent writers from clobbering each other.
    """
    if path is None:
        path = memory_dir() / "learned.md"

    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _lock_path()
    fd_lock = os.open(str(lock), os.O_WRONLY | os.O_CREAT)
    try:
        fcntl.flock(fd_lock, fcntl.LOCK_EX)

        _backup(path)

        new_line = f"- [{category}] {key}: {value} ({confidence:.2f})"

        if not path.exists():
            content = (
                "# Auto-learned facts\n"
                "# RUNE extracts these automatically. Delete any line to forget it.\n"
                "# Promote useful entries to MEMORY.md: rune memory promote\n\n"
            )
            content += new_line + "\n"
        else:
            content = path.read_text(encoding="utf-8")
            lines = content.splitlines()
            key_lower = key.lower()
            replaced = False

            for i, line in enumerate(lines):
                m = _LEARNED_RE.match(line.strip())
                if m and m.group("key").strip().lower() == key_lower:
                    lines[i] = new_line
                    replaced = True
                    break

            if not replaced:
                lines.append(new_line)

            content = "\n".join(lines) + "\n"

        # Write atomically (already under lock, use direct temp+rename)
        fd, tmp_str = tempfile.mkstemp(
            dir=str(path.parent), prefix=".rune_", suffix=".tmp",
        )
        tmp = Path(tmp_str)
        try:
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        finally:
            os.close(fd)
    finally:
        fcntl.flock(fd_lock, fcntl.LOCK_UN)
        os.close(fd_lock)


def remove_learned_fact(key: str, path: Path | None = None) -> bool:
    """Remove a fact line from learned.md by key. Returns True if removed."""
    if path is None:
        path = memory_dir() / "learned.md"
    if not path.exists():
        return False

    _backup(path)

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    key_lower = key.lower()
    new_lines = []
    removed = False

    for line in lines:
        m = _LEARNED_RE.match(line.strip())
        if m and m.group("key").strip().lower() == key_lower:
            removed = True
            continue
        new_lines.append(line)

    if removed:
        _atomic_write(path, "\n".join(new_lines) + "\n")

    return removed


def prune_learned_md(cap: int = _SOFT_CAP, path: Path | None = None) -> list[str]:
    """Remove lowest-confidence entries if learned.md exceeds cap.

    Returns list of removed keys.
    """
    facts = parse_learned_md(path)
    if len(facts) <= cap:
        return []

    facts_sorted = sorted(facts, key=lambda f: f["confidence"])
    to_remove = facts_sorted[:len(facts) - cap]
    removed_keys = [f["key"] for f in to_remove]

    for key in removed_keys:
        remove_learned_fact(key, path)

    return removed_keys


# daily/*.md

def append_daily_entry(
    title: str,
    actions: list[str],
    lessons: list[str] | None = None,
    decisions: list[str] | None = None,
    date: str | None = None,
    time_str: str | None = None,
) -> Path:
    """Append a task entry to today's daily log."""
    if date is None:
        date = datetime.now(UTC).strftime("%Y-%m-%d")
    if time_str is None:
        time_str = datetime.now(UTC).strftime("%H:%M")

    daily_dir = memory_dir() / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    path = daily_dir / f"{date}.md"

    lines: list[str] = []
    lines.append(f"## {time_str} | {title}")
    for action in actions[:5]:
        lines.append(f"- {action}")
    for lesson in (lessons or []):
        lines.append(f"> lesson: {lesson}")
    for decision in (decisions or []):
        lines.append(f"> decision: {decision}")

    entry = "\n".join(lines) + "\n"

    if not path.exists():
        content = f"# {date}\n\n{entry}"
    else:
        content = path.read_text(encoding="utf-8")
        if not content.endswith("\n"):
            content += "\n"
        content += f"\n{entry}"

    _atomic_write(path, content)
    return path


def parse_daily_log(path: Path) -> list[dict[str, Any]]:
    """Parse a daily log into a list of task entries.

    Each entry: {time, title, actions, lessons, decisions, raw}.
    """
    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()

        if stripped.startswith("## "):
            if current:
                entries.append(current)
            # Parse "## 14:23 | Fix auth bug in gateway"
            header = stripped[3:].strip()
            parts = header.split("|", 1)
            time_part = parts[0].strip() if parts else ""
            title_part = parts[1].strip() if len(parts) > 1 else header
            current = {
                "time": time_part,
                "title": title_part,
                "actions": [],
                "lessons": [],
                "decisions": [],
                "raw": stripped,
            }
        elif current is not None:
            if stripped.startswith("> lesson:"):
                current["lessons"].append(stripped[9:].strip())
            elif stripped.startswith("> decision:"):
                current["decisions"].append(stripped[11:].strip())
            elif stripped.startswith("- "):
                current["actions"].append(stripped[2:].strip())

    if current:
        entries.append(current)

    return entries


# rules.md parsing

def parse_rules_md(path: Path) -> list[dict[str, str]]:
    """Parse rules.md into a list of rule dicts.

    Each entry: {name, type, pattern, reason}.
    Rules are H2 sections with bullet metadata.
    """
    if not path.exists():
        return []

    rules: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()

        if stripped.startswith("## "):
            if current and current.get("pattern"):
                rules.append(current)
            current = {"name": stripped[3:].strip()}
        elif current is not None and stripped.startswith("- **") and ":**" in stripped:
            # Parse "- **type:** blocklist"
            m = re.match(r"^- \*\*(\w+):\*\*\s*(.+)$", stripped)
            if m:
                current[m.group(1)] = m.group(2).strip()
        elif current is not None and stripped.startswith("- ") and ":" in stripped:
            # Simple "- key: value" format
            k, _, v = stripped[2:].partition(":")
            current[k.strip()] = v.strip()

    if current and current.get("pattern"):
        rules.append(current)

    return rules


# user-profile.md parsing

def parse_user_profile(path: Path | None = None) -> dict[str, list[str]]:
    """Parse user-profile.md into {section: [lines]}.

    Same format as MEMORY.md: H1 sections with bullet lines.
    """
    if path is None:
        path = memory_dir() / "user-profile.md"
    return parse_memory_md(path)


def update_user_profile_section(
    section: str,
    lines: list[str],
    path: Path | None = None,
) -> None:
    """Overwrite a single H1 section in user-profile.md, preserving others."""
    if path is None:
        path = memory_dir() / "user-profile.md"

    _backup(path)

    if not path.exists():
        content = f"# {section}\n\n"
        for line in lines:
            content += f"- {line}\n"
        _atomic_write(path, content)
        return

    file_content = path.read_text(encoding="utf-8")
    file_lines = file_content.splitlines()
    heading = f"# {section}"

    # Find section boundaries
    start_idx = None
    end_idx = len(file_lines)
    for i, fl in enumerate(file_lines):
        if fl.strip() == heading:
            start_idx = i
            continue
        if start_idx is not None and fl.strip().startswith("# ") and i > start_idx:
            end_idx = i
            break

    new_section = [heading, ""]
    for line in lines:
        new_section.append(f"- {line}")
    new_section.append("")

    if start_idx is not None:
        file_lines[start_idx:end_idx] = new_section
    else:
        if file_lines and file_lines[-1].strip():
            file_lines.append("")
        file_lines.extend(new_section)

    _atomic_write(path, "\n".join(file_lines) + "\n")


# First-run scaffolding

def ensure_memory_structure() -> None:
    """Create the initial memory directory and files if they don't exist."""
    d = memory_dir()

    memory_md = d / "MEMORY.md"
    if not memory_md.exists():
        memory_md.write_text("# Preferences\n\n# Environment\n\n# Notes\n", encoding="utf-8")

    learned = d / "learned.md"
    if not learned.exists():
        learned.write_text(
            "# Auto-learned facts\n"
            "# RUNE extracts these automatically. Delete any line to forget it.\n"
            "# Promote useful entries to MEMORY.md: rune memory promote\n",
            encoding="utf-8",
        )

    daily = d / "daily"
    daily.mkdir(exist_ok=True)

    profile = d / "user-profile.md"
    if not profile.exists():
        profile.write_text(
            "# Communication\n\n"
            "- language: auto\n"
            "- verbosity: concise\n"
            "- tone: technical\n"
            "- code_examples: true\n"
            "\n# Goals\n\n"
            "# Work Profile\n\n"
            "# Coding Style\n\n"
            "# Stats\n\n"
            "# Autonomy\n\n"
            "- enabled: true\n"
            "- global_max_level: balanced\n",
            encoding="utf-8",
        )

    state = d / ".state"
    state.mkdir(exist_ok=True)
    for name in ("fact-meta.json", "suppressed.json", "index-state.json"):
        f = state / name
        if not f.exists():
            f.write_text("{}", encoding="utf-8")
    conflicts = state / "conflicts.json"
    if not conflicts.exists():
        conflicts.write_text("[]", encoding="utf-8")
