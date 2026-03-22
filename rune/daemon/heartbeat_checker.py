"""HEARTBEAT.md checker — user-editable proactive monitoring.

Reads ~/.rune/HEARTBEAT.md, parses check items with schedule tags,
and runs backtick-enclosed commands with Guardian validation.

Format:
    ## Every 30 minutes
    - [ ] `git status --short` — uncommitted changes
    - [ ] `df -h / | tail -1` — disk usage

    ## Every 1 hour
    - [ ] `ruff check . 2>&1 | tail -3` — lint errors

    ## Daily at 9:00
    - [ ] `git log --oneline -5` — recent commits

Commands in backticks are executed as-is via bash.
No backtick = item is skipped.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rune.utils.logger import get_logger
from rune.utils.paths import rune_home

log = get_logger(__name__)

_HEARTBEAT_PATH = None  # lazy init

# Parsing patterns
_SCHEDULE_RE = re.compile(
    r"##\s+(?:every|매)\s+(\d+)\s*(min(?:ute)?s?|hour|시간|분)",
    re.IGNORECASE,
)
_DAILY_RE = re.compile(
    r"##\s+(?:daily|매일)\s+(?:at\s+)?(\d{1,2}):?(\d{2})?",
    re.IGNORECASE,
)
_ITEM_RE = re.compile(r"^-\s+\[[ x]\]\s+(.+)$", re.IGNORECASE)
_BACKTICK_RE = re.compile(r"`([^`]+)`")


def _get_heartbeat_path() -> Path:
    global _HEARTBEAT_PATH
    if _HEARTBEAT_PATH is None:
        _HEARTBEAT_PATH = rune_home() / "HEARTBEAT.md"
    return _HEARTBEAT_PATH


@dataclass(slots=True)
class CheckItem:
    """A single heartbeat check item."""
    text: str
    command: str  # extracted from backticks, empty if none
    interval_minutes: int  # 0 = daily at specific time
    daily_hour: int = 0
    daily_minute: int = 0
    last_run: datetime | None = None


def parse_heartbeat_md(path: Path | None = None) -> list[CheckItem]:
    """Parse HEARTBEAT.md into check items with schedules."""
    path = path or _get_heartbeat_path()
    if not path.exists():
        return []

    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return []

    items: list[CheckItem] = []
    current_interval = 30  # default: 30 minutes
    current_daily_hour = -1
    current_daily_minute = 0

    for line in content.splitlines():
        line = line.strip()
        if not line or (line.startswith("#") and not line.startswith("##")):
            continue

        # Check for schedule header
        schedule_match = _SCHEDULE_RE.match(line)
        if schedule_match:
            num = int(schedule_match.group(1))
            unit = schedule_match.group(2).lower()
            if "hour" in unit or "시간" in unit:
                current_interval = num * 60
            else:
                current_interval = num
            current_daily_hour = -1
            continue

        daily_match = _DAILY_RE.match(line)
        if daily_match:
            current_daily_hour = int(daily_match.group(1))
            current_daily_minute = int(daily_match.group(2) or "0")
            current_interval = 0
            continue

        # Check for item
        item_match = _ITEM_RE.match(line)
        if item_match:
            text = item_match.group(1).strip()
            # Extract command from backticks
            cmd_match = _BACKTICK_RE.search(text)
            command = cmd_match.group(1) if cmd_match else ""

            if current_daily_hour >= 0:
                items.append(CheckItem(
                    text=text, command=command,
                    interval_minutes=0,
                    daily_hour=current_daily_hour,
                    daily_minute=current_daily_minute,
                ))
            else:
                items.append(CheckItem(
                    text=text, command=command,
                    interval_minutes=current_interval,
                ))

    return items


def should_run(item: CheckItem, now: datetime) -> bool:
    """Check if an item should run based on its schedule and last run time."""
    if item.interval_minutes == 0:
        if now.hour != item.daily_hour or now.minute != item.daily_minute:
            return False
        if item.last_run and item.last_run.date() == now.date():
            return False
        return True

    if item.last_run is None:
        return True
    elapsed = (now - item.last_run).total_seconds() / 60
    return elapsed >= item.interval_minutes


async def run_heartbeat_checks() -> list[dict[str, Any]]:
    """Run all due heartbeat checks and return results.

    Each result: {item, command, output, exit_code, needs_attention}
    """
    items = parse_heartbeat_md()
    if not items:
        return []

    now = datetime.now()
    results: list[dict[str, Any]] = []

    for item in items:
        if not should_run(item, now):
            continue
        if not item.command:
            continue  # No backtick command — skip

        item.last_run = now
        log.debug("heartbeat_check_running", command=item.command[:80])

        try:
            output, exit_code = await _execute_check(item.command)
            attention = exit_code != 0
            results.append({
                "item": item.text,
                "command": item.command,
                "output": output,
                "exit_code": exit_code,
                "needs_attention": attention,
            })
            if attention:
                log.info("heartbeat_attention", command=item.command[:80], exit_code=exit_code)
        except Exception as exc:
            log.warning("heartbeat_check_failed", command=item.command[:50], error=str(exc)[:200])

    return results


async def _execute_check(command: str) -> tuple[str, int]:
    """Execute a command with Guardian validation. Returns (output, exit_code)."""
    # Guardian validation — block dangerous commands
    try:
        from rune.safety.guardian import get_guardian
        guardian = get_guardian()
        result = guardian.validate(command)
        if not result.allowed:
            log.warning("heartbeat_blocked", command=command[:50], reason=result.reason)
            return f"[BLOCKED] {result.reason}", 1
    except Exception:
        pass  # Guardian unavailable — proceed (local-only file)

    # Execute with timeout
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except TimeoutError:
        proc.kill()
        return "[TIMEOUT] Command exceeded 30s", 1

    output = (stdout or b"").decode(errors="replace")
    if proc.returncode != 0:
        err = (stderr or b"").decode(errors="replace")
        if err.strip():
            output = f"{output}\n{err}".strip()

    return output[:2000], proc.returncode or 0


def create_default_heartbeat() -> Path:
    """Create a default HEARTBEAT.md template if it doesn't exist."""
    path = _get_heartbeat_path()
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# HEARTBEAT.md\n"
        "# RUNE runs commands in backticks periodically.\n"
        "# Exit code != 0 triggers a notification.\n"
        "# Edit freely — add, remove, or comment out items.\n"
        "# Delete this file to disable heartbeat checks.\n\n"
        "## Every 30 minutes\n"
        "- [ ] `git status --short` — uncommitted changes\n\n"
        "## Every 1 hour\n"
        "- [ ] `ruff check . 2>&1 | tail -3` — lint errors\n\n"
        "# ## Daily at 09:00\n"
        "# - [ ] `git log --oneline -5` — recent commits\n",
        encoding="utf-8",
    )
    log.info("heartbeat_md_created", path=str(path))
    return path
