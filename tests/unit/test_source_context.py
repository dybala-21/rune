"""Source reuse stays bounded, scoped to one conversation, and explicitly historical."""

import json
from types import SimpleNamespace

import pytest

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.source_context import (
    HISTORY_KEY,
    MAX_AGE,
    MAX_CHARS,
    SourceContext,
    attach_sources,
    remember_sources,
)
from rune.conversation.types import Conversation, ConversationTurn
from rune.utils.tokenizer import count_tokens


def snapshot():
    sources = SourceContext()
    sources.capture("https://example.org/data?key=private", '{"station":"North","value":17.125}', "application/json")
    return sources


def test_context_is_bounded_and_excludes_query_credentials():
    sources = snapshot()
    assert "private" not in sources.pack()
    for i in range(10):
        sources.capture(f"https://example.org/{i}", "x" * 7_000, "text/plain")
    assert len(sources.entries) <= 4 and len(sources.pack()) <= MAX_CHARS
    sources.capture("https://user:password@example.org", "secret", "text/plain")
    assert "secret" not in sources.pack()


def test_expiry_and_integrity_do_not_reset_on_reuse(monkeypatch):
    sources = snapshot()
    now = sources.entries[0]["retrieved_at"]
    monkeypatch.setattr("rune.agent.source_context.time.time", lambda: now + MAX_AGE - 1)
    restored = SourceContext.unpack(sources.pack())
    assert restored.entries[0]["retrieved_at"] == now
    monkeypatch.setattr("rune.agent.source_context.time.time", lambda: now + MAX_AGE + 1)
    assert not SourceContext.unpack(restored.pack()).entries
    monkeypatch.setattr("rune.agent.source_context.time.time", lambda: now)
    data = json.loads(sources.pack())
    data[0]["body"] = '{"station":"North","value":999}'
    assert not SourceContext.unpack(json.dumps(data)).entries


@pytest.mark.parametrize("changes,allowed", [
    ({}, True), ({"needs_current_source": True}, False), ({"is_domain_change": True}, False),
    ({"is_continuation": False}, False), ({"source_output": "none"}, False),
])
def test_reuse_requires_a_related_noncurrent_source_request(changes, allowed):
    classification = ClassificationResult("chat", .99, 2, **({"is_continuation": True, "source_output": "verbatim"} | changes))
    history = [{"role": "assistant", "content": "Read it", HISTORY_KEY: snapshot().pack()}]
    assert bool(SourceContext.from_history(history, classification).entries) is allowed
    assert not SourceContext.from_history([], classification).entries


def test_prompt_budget_does_not_truncate_raw_evidence():
    sources = snapshot()
    message = sources.message(max_tokens=400)
    assert message and count_tokens(message["content"]) <= 400
    assert sources.entries[0]["body"] in message["content"].replace('\\"', '"')
    assert sources.message(max_tokens=1) is None and not sources.entries


def test_conversation_binding_does_not_attach_to_another_answer():
    conv = Conversation(execution_context='{"cwd":"/workspace"}', turns=[ConversationTurn("assistant", "Read it")])
    remember_sources(conv, SimpleNamespace(_source_context=snapshot()))
    assert json.loads(conv.execution_context)["cwd"] == "/workspace"
    wrong = [{"role": "assistant", "content": "Different answer"}]
    attach_sources(wrong, conv.execution_context)
    assert HISTORY_KEY not in wrong[0]
    right = [{"role": "assistant", "content": "Read it"}]
    attach_sources(right, conv.execution_context)
    assert SourceContext.unpack(right[0][HISTORY_KEY]).entries
    remember_sources(conv, SimpleNamespace(_source_context=SourceContext()))
    assert "sourceEvidence" in json.loads(conv.execution_context)


async def test_snapshot_survives_conversation_reload(tmp_path):
    from rune.api.conversation_wiring import record_assistant_turn
    from rune.conversation.manager import ConversationManager
    from rune.conversation.store import ConversationStore

    store = ConversationStore(tmp_path / "conversations.db")
    manager = ConversationManager(store)
    conv = manager.start_conversation("user")
    manager.add_turn(conv.id, "user", "Read the source")
    await record_assistant_turn(manager, conv.id, SimpleNamespace(
        _last_answer_text="17.125", _last_goal_type="web", _source_context=snapshot()), "", embed=False)
    manager._active.clear()
    manager._active[conv.id] = await store.load(conv.id)
    messages = manager._build_messages_within_budget(conv.id, 1000)
    assert SourceContext.unpack(messages[-1][HISTORY_KEY]).entries
    other = manager.start_conversation("other")
    manager.add_turn(other.id, "assistant", "17.125")
    assert HISTORY_KEY not in manager._build_messages_within_budget(other.id, 1000)[-1]
