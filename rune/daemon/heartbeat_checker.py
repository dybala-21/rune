"""HEARTBEAT.md checker — user-editable proactive monitoring.

Reads ~/.rune/HEARTBEAT.md, parses check items with schedule tags,
and runs matching items via lightweight agent execution.

Format:
    ## Every 30 minutes
    - [ ] git status — notify if uncommitted changes

    ## Every 1 hour
    - [ ] pytest — notify if tests fail

    ## Daily at 9:00
    - [ ] summarize yesterday's work
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rune.utils.logger import get_logger
from rune.utils.paths import rune_home

log = get_logger(__name__)

_HEARTBEAT_PATH = None  # lazy init


def _get_heartbeat_path() -> Path:
    global _HEARTBEAT_PATH
    if _HEARTBEAT_PATH is None:
        _HEARTBEAT_PATH = rune_home() / "HEARTBEAT.md"
    return _HEARTBEAT_PATH


# Schedule patterns
_SCHEDULE_RE = re.compile(
    r"##\s+(?:every|매)\s+(\d+)\s*(min(?:ute)?s?|hour|시간|분)",
    re.IGNORECASE,
)
_DAILY_RE = re.compile(
    r"##\s+(?:daily|매일)\s+(?:at\s+)?(\d{1,2}):?(\d{2})?",
    re.IGNORECASE,
)
_ITEM_RE = re.compile(r"^-\s+\[[ x]\]\s+(.+)$", re.IGNORECASE)


@dataclass(slots=True)
class CheckItem:
    """A single heartbeat check item."""
    text: str
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
        if not line or line.startswith("#") and not line.startswith("##"):
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
            current_interval = 0  # daily mode
            continue

        # Check for item
        item_match = _ITEM_RE.match(line)
        if item_match:
            text = item_match.group(1).strip()
            if current_daily_hour >= 0:
                items.append(CheckItem(
                    text=text,
                    interval_minutes=0,
                    daily_hour=current_daily_hour,
                    daily_minute=current_daily_minute,
                ))
            else:
                items.append(CheckItem(
                    text=text,
                    interval_minutes=current_interval,
                ))

    return items


def should_run(item: CheckItem, now: datetime) -> bool:
    """Check if an item should run based on its schedule and last run time."""
    if item.interval_minutes == 0:
        # Daily mode: run once per day at the specified time
        if now.hour != item.daily_hour or now.minute != item.daily_minute:
            return False
        if item.last_run and item.last_run.date() == now.date():
            return False
        return True

    # Interval mode
    if item.last_run is None:
        return True
    elapsed = (now - item.last_run).total_seconds() / 60
    return elapsed >= item.interval_minutes


async def run_heartbeat_checks() -> list[dict[str, Any]]:
    """Run all due heartbeat checks and return results.

    Each result: {item: str, output: str, needs_attention: bool}
    """
    items = parse_heartbeat_md()
    if not items:
        return []

    now = datetime.now()
    results: list[dict[str, Any]] = []

    for item in items:
        if not should_run(item, now):
            continue

        item.last_run = now
        log.debug("heartbeat_check_running", item=item.text[:80])

        try:
            output = await _execute_check(item.text)
            needs_attention = _needs_attention(output)
            results.append({
                "item": item.text,
                "output": output,
                "needs_attention": needs_attention,
            })
            if needs_attention:
                log.info("heartbeat_attention", item=item.text[:80])
        except Exception as exc:
            log.warning("heartbeat_check_failed", item=item.text[:50], error=str(exc)[:200])

    return results


async def _execute_check(check_text: str) -> str:
    """Execute a single check item via lightweight agent or bash."""
    import asyncio
    # Simple heuristic: if the check mentions a command, run it directly
    # Otherwise delegate to the agent
    cmd_patterns = [
        "git status", "git diff", "pytest", "ruff", "curl", "df ", "du ",
        "docker", "npm test", "ping",
    ]
    for pattern in cmd_patterns:
        if pattern in check_text.lower():
            # Extract or construct the command
            cmd = _extract_command(check_text, pattern)
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = (stdout or b"").decode(errors="replace")
            if proc.returncode != 0:
                err = (stderr or b"").decode(errors="replace")
                output = f"[EXIT {proc.returncode}] {output}\n{err}".strip()
            return output[:2000]

    # No command pattern found — return the check text for manual review
    return f"[MANUAL CHECK] {check_text}"


def _extract_command(check_text: str, pattern: str) -> str:
    """Extract a runnable command from check text."""
    # If the text contains a known command, try to extract it
    lower = check_text.lower()
    if "git status" in lower:
        return "git status --short"
    if "git diff" in lower:
        return "git diff --stat"
    if "pytest" in lower:
        return "python3 -m pytest --tb=line -q 2>&1 | tail -5"
    if "ruff" in lower:
        return "ruff check . 2>&1 | tail -5"
    if "curl" in lower:
        # Try to extract URL
        import re
        url_match = re.search(r"(https?://\S+)", check_text)
        if url_match:
            return f"curl -sf -o /dev/null -w '%{{http_code}}' {url_match.group(1)}"
    if "df " in lower or "디스크" in lower or "disk" in lower:
        return "df -h / | tail -1"
    if "docker" in lower:
        return "docker ps --format 'table {{.Names}}\t{{.Status}}' 2>&1"
    # Fallback: run the text as-is
    return pattern


def _needs_attention(output: str) -> bool:
    """Determine if the output indicates a problem that needs user attention."""
    if not output:
        return False
    # Exit code errors
    if output.startswith("[EXIT ") and "[EXIT 0]" not in output:
        return True
    # Test failures
    if "failed" in output.lower() or "error" in output.lower():
        return True
    # Git uncommitted changes
    if output.strip() and not output.startswith("[MANUAL"):
        # git status --short returns non-empty if there are changes
        lines = [l for l in output.splitlines() if l.strip()]
        if lines and any(l.startswith(("M ", "A ", "D ", "?? ", " M")) for l in lines):
            return True
    # Disk usage > 80%
    if "%" in output:
        import re
        pcts = re.findall(r"(\d+)%", output)
        if any(int(p) >= 80 for p in pcts):
            return True
    return False


def create_default_heartbeat() -> Path:
    """Create a default HEARTBEAT.md template if it doesn't exist."""
    path = _get_heartbeat_path()
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# HEARTBEAT.md\n"
        "# RUNE checks these items periodically and notifies you if something needs attention.\n"
        "# Edit freely — add, remove, or comment out items.\n"
        "# Delete this file to disable heartbeat checks.\n\n"
        "## Every 30 minutes\n"
        "- [ ] git status — notify if uncommitted changes\n\n"
        "## Every 1 hour\n"
        "- [ ] ruff check — notify if lint errors\n\n"
        "# ## Daily at 09:00\n"
        "# - [ ] summarize yesterday's work\n",
        encoding="utf-8",
    )
    log.info("heartbeat_md_created", path=str(path))
    return path
