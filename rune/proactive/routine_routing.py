"""Reuse task routing without reusing file state or approvals."""

import hashlib
import json
from pathlib import Path

from rune.agent.goal_classifier import _TIER2_SYSTEM_PROMPT, VALID_GOAL_TYPES, from_wire, to_wire


def routing_key(goal: str, workspace: str) -> str:
    from rune.config import get_config
    from rune.safety.execution_environment import execution_config

    value = {"version": 1, "goal": goal, "workspace": str(Path(workspace).resolve()),
             "llm": get_config().llm.model_dump(mode="json"),
             "environment": execution_config().model_dump(mode="json"), "rules": _TIER2_SYSTEM_PROMPT}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def snapshot(key: str, classification, goal: str = "") -> dict | None:
    if (not classification.available or classification.requires_desktop_input
            or classification.goal_type in {"browser", "web"} or "desktop" in classification.intent_categories):
        return None
    roles = classification.artifact_roles
    return {"key": key, "classification": to_wire(classification),
            "roles": roles.matching_roles(goal) if roles is not None else {}}


def restore(saved: dict | None, key: str, goal: str = ""):
    if not isinstance(saved, dict) or saved.get("key") != key or not isinstance(saved.get("classification"), str):
        return None
    try:
        classification = from_wire(saved["classification"])
        if classification is None or classification.goal_type not in VALID_GOAL_TYPES or snapshot(key, classification) is None:
            return None
        from rune.agent.provenance import ArtifactRoleHints

        if isinstance(saved.get("roles"), dict):
            classification.artifact_roles = ArtifactRoleHints.for_request(goal, saved["roles"])
    except (AttributeError, TypeError, ValueError):
        return None
    return classification
