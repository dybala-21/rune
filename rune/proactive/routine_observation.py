"""Compare explicitly tracked files before spending another agent run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rune.capabilities.document_inspection import read_snapshot
from rune.utils.logger import get_logger

log = get_logger(__name__)


def observe(policy) -> dict:
    root = Path(policy.workspace).expanduser().resolve()
    observation: dict = {"inputs": {}, "outputs": {}, "complete": True,
                         "policy": hashlib.sha256(policy.model_dump_json().encode()).hexdigest()}
    total = 0
    for group, names in (("inputs", policy.input_paths), ("outputs", policy.output_paths)):
        for name in names:
            target = (root / name).resolve()
            try:
                if not target.is_relative_to(root):
                    raise ValueError("Tracked files must stay inside the routine workspace")
                if not target.exists():
                    observation[group][name] = {"state": "missing"}
                    continue
                _, data = read_snapshot(target)
                total += len(data)
                if total > 50_000_000:
                    raise ValueError("Tracked files exceed the 50 MB observation budget")
                observation[group][name] = {"state": "present", "sha256": hashlib.sha256(data).hexdigest()}
            except (OSError, ValueError) as exc:
                log.debug("routine_observation_unavailable", path=str(target), error=str(exc))
                observation["complete"] = False
                observation[group][name] = {"state": "unknown", "error": str(exc)}
    return observation


def reusable(job, current: dict, previous: dict | None) -> bool:
    if not job.policy.input_paths or not previous or previous.get("execution_unknown"):
        return False
    if previous.get("success") is not True or previous.get("status") not in {"completed", "verified", "unchanged"}:
        return False
    before = previous.get("observations")
    if not before or not before.get("complete") or not current.get("complete"):
        return False
    if previous.get("routine_goal") != job.goal or previous.get("routine_command") != job.command:
        return False
    # Missing inputs/outputs cannot establish a reusable completed result.
    if any(item["state"] != "present" for group in ("inputs", "outputs") for item in current[group].values()):
        return False
    return before == current


def result_identity(result: dict) -> str | None:
    observation = result.get("observations")
    if not observation or not observation.get("complete"):
        return None
    actual = observation.get("outputs") or observation.get("inputs")
    if not actual:
        return None
    return json.dumps(actual, sort_keys=True)
