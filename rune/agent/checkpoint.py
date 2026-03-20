"""Deterministic checkpoints for agent state.

Ported from src/agent/checkpoint.ts - save/restore agent state
for mid-task resumption.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rune.utils.fast_serde import json_decode
from rune.utils.paths import rune_data


@dataclass(slots=True)
class CheckpointData:
    session_id: str = ""
    step: int = 0
    goal: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    token_usage: int = 0
    created_at: str = ""


class CheckpointManager:
    """Save and restore deterministic agent state snapshots."""

    def __init__(self, checkpoint_dir: Path | None = None) -> None:
        self._dir = checkpoint_dir or (rune_data() / "checkpoints")
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, data: CheckpointData) -> Path:
        """Save a checkpoint. Returns the file path."""
        if not data.created_at:
            data.created_at = datetime.now(UTC).isoformat()

        filename = f"{data.session_id}_step{data.step}.json"
        path = self._dir / filename

        path.write_text(json.dumps(asdict(data), ensure_ascii=False, indent=2))
        return path

    def load(self, session_id: str, step: int | None = None) -> CheckpointData | None:
        """Load a checkpoint. If step is None, loads the latest."""
        if step is not None:
            path = self._dir / f"{session_id}_step{step}.json"
            if path.is_file():
                return self._parse(path)
            return None

        # Find latest
        candidates = sorted(
            self._dir.glob(f"{session_id}_step*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return self._parse(candidates[0])
        return None

    def _parse(self, path: Path) -> CheckpointData | None:
        try:
            data = json_decode(path.read_text())
            return CheckpointData(**data)
        except (json.JSONDecodeError, TypeError):
            return None

    def delete(self, session_id: str) -> int:
        """Delete all checkpoints for a session. Returns count deleted."""
        count = 0
        for path in self._dir.glob(f"{session_id}_step*.json"):
            path.unlink()
            count += 1
        return count
