"""Compare explicitly tracked files before spending another agent run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from rune.capabilities.document_inspection import read_snapshot
from rune.utils.logger import get_logger

log = get_logger(__name__)


def observe(policy) -> dict:
    root = Path(policy.workspace).expanduser().resolve()
    observation: dict = {"inputs": {}, "outputs": {}, "complete": True,
                         "policy": hashlib.sha256(json.dumps(policy.model_dump(
                             include={"workspace", "verification", "input_paths", "output_paths"}), sort_keys=True).encode()).hexdigest()}
    total = 0
    inspected = 0
    outputs = [(root / name).resolve() for name in policy.output_paths]

    def inspect(target: Path, group: str, name: str, *, declared: bool = False) -> None:
        nonlocal total, inspected
        inspected += 1
        if inspected > 1000:
            raise ValueError("Tracked folders exceed the 1000 entry observation budget")
        resolved = target.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("Tracked files must stay inside the routine workspace")
        from rune.agent.isolation import enforce
        from rune.safety.guardian import get_guardian

        decision = get_guardian().validate_file_read_path(str(resolved))
        if not decision.allowed or enforce(str(resolved)):
            raise ValueError(decision.reason or "Tracked path is outside the execution workspace")
        if target.is_dir():
            before = target.stat()
            # Output parents may be created by the task; still inspect their other contents.
            if declared or group != "inputs" or not any(output.is_relative_to(resolved) for output in outputs):
                observation[group][name] = {"state": "present", "kind": "directory"}
            with os.scandir(target) as entries:
                for entry in entries:
                    child = Path(entry.path)
                    if group == "inputs" and any(child.resolve().is_relative_to(output) for output in outputs):
                        continue
                    if entry.is_symlink():
                        raise ValueError("Tracked folders cannot contain symbolic links")
                    inspect(child, group, str(child.relative_to(root)))
            after = target.stat()
            if (before.st_mtime_ns, before.st_ctime_ns) != (after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("Folder changed while it was being inspected")
        elif not target.exists():
            observation[group][name] = {"state": "missing"}
        else:
            _, data = read_snapshot(target)
            total += len(data)
            if total > 50_000_000:
                raise ValueError("Tracked files exceed the 50 MB observation budget")
            observation[group][name] = {"state": "present", "sha256": hashlib.sha256(data).hexdigest()}

    for group, names in (("inputs", policy.input_paths), ("outputs", policy.output_paths)):
        for name in names:
            target = root / name
            try:
                inspect(target, group, name, declared=True)
            except (OSError, ValueError, RuntimeError) as exc:
                log.debug("routine_observation_unavailable", path=str(target), error=str(exc))
                observation["complete"] = False
                observation[group][name] = {"state": "unknown", "error": str(exc)}
    return observation


def changes(current: dict | None, previous: dict | None) -> dict | None:
    before = (previous or {}).get("observations")
    if not current or not before or not current.get("complete") or not before.get("complete"):
        return None
    result = {}
    for group in ("inputs", "outputs"):
        old, new = before[group], current[group]
        result[group] = {
            "added": sorted(new.keys() - old.keys()),
            "removed": sorted(old.keys() - new.keys()),
            "modified": sorted(key for key in new.keys() & old.keys() if old[key] != new[key]),
        }
    return result


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
