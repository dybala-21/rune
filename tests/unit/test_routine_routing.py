"""Routing reuse and invalidation when approved tasks resume."""

import json
from types import SimpleNamespace

import pytest

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.provenance import ArtifactRoleHints
from rune.proactive.routine_routing import restore, routing_key, snapshot


def test_routing_key_invalidates_changed_goal_workspace_model_and_rules(monkeypatch, tmp_path):
    settings = {"provider": "xai", "model": "grok-4.6"}
    environment = {"mode": "local"}
    monkeypatch.setattr("rune.config.get_config", lambda: SimpleNamespace(
        llm=SimpleNamespace(model_dump=lambda **kw: settings)))
    monkeypatch.setattr("rune.safety.execution_environment.execution_config", lambda: SimpleNamespace(
        model_dump=lambda **kw: environment))
    key = routing_key("Prepare draft.txt", str(tmp_path))
    assert key != routing_key("Review draft.txt", str(tmp_path))
    assert key != routing_key("Prepare draft.txt", str(tmp_path / "other"))
    settings["model"] = "grok-4.7"
    assert key != routing_key("Prepare draft.txt", str(tmp_path))
    settings["model"] = "grok-4.6"
    environment["mode"] = "container"
    assert key != routing_key("Prepare draft.txt", str(tmp_path))
    environment["mode"] = "local"
    monkeypatch.setattr("rune.proactive.routine_routing._TIER2_SYSTEM_PROMPT", "updated")
    assert key != routing_key("Prepare draft.txt", str(tmp_path))


def test_same_goal_keeps_file_roles_without_file_observations():
    goal = "Read source.txt and create draft.txt"
    classification = ClassificationResult(goal_type="artifact", confidence=.95, tier=2,
        artifact_roles=ArtifactRoleHints.for_request(goal, {"source.txt": "input", "draft.txt": "output"}))
    saved = snapshot("key", classification, goal)
    resumed = restore(saved, "key", goal)
    assert resumed.goal_type == "artifact"
    assert resumed.artifact_roles.matching_roles(goal) == {"source.txt": "input", "draft.txt": "output"}
    assert restore(saved, "changed", goal) is None
    assert "read_ok" not in json.dumps(saved) and "approval" not in json.dumps(saved)


@pytest.mark.parametrize("changes", [{"goal_type": "browser"}, {"requires_desktop_input": True},
                                     {"available": False}, {"intent_categories": frozenset({"desktop"})}])
def test_live_surface_routing_is_not_reused(changes):
    classification = ClassificationResult(**{"goal_type": "full", "confidence": .9, "tier": 2, **changes})
    assert snapshot("key", classification) is None


@pytest.mark.parametrize("blob", ["[]", "null", "{}", '{"goal_type": []}', "invalid"])
def test_malformed_routing_is_discarded(blob):
    assert restore({"key": "key", "classification": blob}, "key") is None


async def test_background_passes_saved_classification_only_to_the_matching_resume(tmp_path, monkeypatch):
    from rune.agent.background import BackgroundTask, run_background
    from rune.types import CompletionTrace
    from rune.utils.events import EventEmitter

    monkeypatch.setenv("RUNE_WORKSPACE", str(tmp_path))
    journal = SimpleNamespace(previous=None, run_id="first")
    monkeypatch.setattr("rune.agent.execution_journal.active_journal", lambda: journal)
    classifications = []

    class Loop(EventEmitter):
        def __init__(self, **kwargs):
            super().__init__()

        def set_approval_callback(self, callback):
            pass

        async def run(self, goal, **kwargs):
            classification = kwargs.get("classification")
            classifications.append(classification)
            if classification is None:
                classification = ClassificationResult(goal_type="artifact", confidence=.9, tier=2)
            await self.emit("goal_classified", classification)
            return CompletionTrace(reason="completed")

    task = BackgroundTask("Save draft.txt", "test", workspace=str(tmp_path))
    first = await run_background(task, loop_factory=Loop)
    journal.previous = []
    second = await run_background(BackgroundTask(task.goal, "test", workspace=str(tmp_path), routing=first["routing"]),
                                  loop_factory=Loop)
    assert classifications[0] is None and classifications[1].goal_type == "artifact"
    assert first["routing"] == second["routing"]
    await run_background(BackgroundTask("Review source.txt", "test", workspace=str(tmp_path), routing=first["routing"]),
                         loop_factory=Loop)
    assert classifications[2] is None
