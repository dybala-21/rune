"""Where remembered things came from, and what a connector's data does on its way back into a run.

A connector call closes a hosted machine's open web (rune-bot's broker). What RUNE remembers from
that call carries the same data into later runs, so it closes the web the same way before it goes
in, or stays out when the broker can't be told. Forgetting a connector removes what came from it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from rune.memory import manager as manager_mod
from rune.memory import markdown_store as md
from rune.memory import provenance
from rune.memory import store as store_mod
from rune.memory.manager import MemoryManager
from rune.memory.state import load_fact_meta, load_suppressed
from rune.memory.store import MemoryStore
from rune.memory.types import Episode
from rune.memory.vector import VectorStore


@pytest.fixture
def memory(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / "memory.db")
    manager = MemoryManager(store=store, vector_store=VectorStore(index_path=tmp_path / "vectors"))
    monkeypatch.setattr(store_mod, "_store", store)
    monkeypatch.setattr(manager_mod, "_manager", manager)
    monkeypatch.setattr("rune.utils.tokenizer._get_encoding", lambda *a, **k: None, raising=False)
    return manager


class Broker:
    """The hosted broker's /private, as the core sees it."""

    def __init__(self, monkeypatch, *, answer: dict[str, Any] | None = None, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.answer = answer if answer is not None else {"private": True, "closed": True}
        self.fail = fail
        monkeypatch.setattr("rune.connectors.client.hosted_approval", lambda: True)
        monkeypatch.setattr("rune.connectors.client.call_broker", self)

    async def __call__(self, operation: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(operation)
        if self.fail:
            raise OSError("broker down")
        return self.answer


def remember_from_connector(key: str, value: str, connector: str = "gmail") -> None:
    md.save_learned_fact("preference", key, value, 0.9)
    provenance.note_facts([key], connectors=[connector], conversation="c1", episode="e1")


# Recording


def test_a_run_records_each_connector_it_calls_and_nested_scopes_share_it():
    with provenance.run_sources() as outer:
        provenance.record_connector("gmail")
        with provenance.run_sources() as inner:
            provenance.record_connector("calendar")
        assert inner is outer
        assert provenance.contacted() == ["calendar", "gmail"]
    assert provenance.contacted() == []


async def test_the_connector_tool_records_its_connector_before_the_call_goes_out(monkeypatch):
    from rune.capabilities.connector import connector_request
    from rune.connectors.models import ConnectorRequest

    seen: list[list[str]] = []

    async def broker(operation: str, params: dict[str, Any]) -> dict[str, Any]:
        seen.append(provenance.contacted())
        raise OSError("lost")  # an uncertain call still counts: its answer may have arrived

    monkeypatch.setattr("rune.capabilities.connector.hosted_approval", lambda: True)
    monkeypatch.setattr("rune.capabilities.connector.call_broker", broker)
    request = ConnectorRequest(connector="gmail", origin="https://gmail.example", revision="a" * 32, path="/v1/x")
    with provenance.run_sources() as contacted:
        await connector_request(request)
    assert seen == [["gmail"]]
    assert contacted == {"gmail"}


def test_an_episode_keeps_its_connectors_across_a_reopen(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    store.save_episode(Episode(id="e1", task_summary="read mail", sources=provenance.encode(["gmail"])))
    store.save_episode(Episode(id="e2", task_summary="local work"))
    store.close()
    again = MemoryStore(tmp_path / "m.db")
    assert [ep.id for ep in again.episodes_with_sources()] == ["e1"]
    assert provenance.decode(again.get_episodes_by_ids(["e1"])[0].sources) == ["gmail"]


def test_a_fact_saved_again_from_another_run_keeps_every_origin():
    provenance.note_facts(["diet"], connectors=["gmail"], conversation="c1", episode="e1")
    provenance.note_facts(["diet"], conversation="c2", episode="e2")
    origin = load_fact_meta()["diet"]["origin"]
    assert origin == {"connectors": ["gmail"], "conversations": ["c1", "c2"], "episodes": ["e1", "e2"]}


def test_unreadable_sources_count_as_a_connectors():
    assert provenance.decode("not json") == ["unknown"]


async def test_a_fact_the_agent_saves_during_a_connector_run_keeps_the_connector(memory):
    from rune.memory.types import Fact

    with provenance.run_sources():
        provenance.record_connector("gmail")
        await memory.save_fact(Fact(key="boss", value="Kim", category="preference", source="agent:user"))
    assert provenance.fact_connectors(load_fact_meta(), "boss") == ["gmail"]


def test_the_daily_log_names_the_connectors_and_forgetting_removes_only_those_entries():
    md.append_daily_entry("read mail", ["summarised 3 mails"], connectors=["gmail"], date="2026-11-14", time_str="09:00")
    md.append_daily_entry("fix tests", ["touched a.py"], date="2026-11-14", time_str="10:00")
    path = md.memory_dir() / "daily" / "2026-11-14.md"
    assert [e["connectors"] for e in md.parse_daily_log(path)] == [["gmail"], []]
    assert md.remove_daily_entries("gmail") == 1
    assert [e["title"] for e in md.parse_daily_log(path)] == ["fix tests"]


# Going back into a run


async def test_locally_a_remembered_connector_answer_goes_in_without_asking_anyone(memory, monkeypatch):
    monkeypatch.setattr("rune.connectors.client.hosted_approval", lambda: False)
    remember_from_connector("diet", "vegetarian")
    await memory.initialize()
    assert "vegetarian" in await memory.build_memory_context("suggest dinner")


async def test_hosted_it_goes_in_only_after_the_broker_closed_the_web(memory, monkeypatch):
    broker = Broker(monkeypatch)
    remember_from_connector("diet", "vegetarian")
    await memory.initialize()
    assert "vegetarian" in await memory.build_memory_context("suggest dinner")
    assert broker.calls == ["private"]


@pytest.mark.parametrize("trouble", [{"fail": True}, {"answer": {"closed": True}}])
async def test_hosted_it_stays_out_when_the_broker_cant_be_told(memory, monkeypatch, trouble):
    Broker(monkeypatch, **trouble)
    remember_from_connector("diet", "vegetarian")
    md.save_learned_fact("preference", "editor", "vim", 0.9)
    await memory.initialize()
    context = await memory.build_memory_context("suggest dinner editor")
    assert "vegetarian" not in context
    assert "vim" in context, "what didn't come from a connector still goes in"


async def test_hosted_nothing_from_a_connector_asks_nothing(memory, monkeypatch):
    broker = Broker(monkeypatch)
    md.save_learned_fact("preference", "editor", "vim", 0.9)
    await memory.initialize()
    assert "vim" in await memory.build_memory_context("editor")
    assert broker.calls == []


@pytest.mark.parametrize("told", [True, False])
async def test_an_episode_from_a_connector_run_is_past_experience_only_once_the_web_closed(memory, monkeypatch, told):
    Broker(monkeypatch, fail=not told)
    memory.store.save_episode(Episode(id="e1", task_summary="summarise invoice mail", result="Invoice 4411 total 90",
                                      utility=1, importance=0.9, sources=provenance.encode(["gmail"])))
    await memory.initialize()
    assert ("summarise invoice mail" in await memory.build_memory_context("summarise invoice mail")) is told


@pytest.mark.parametrize("told", [True, False])
async def test_memory_search_returns_a_connector_fact_only_once_the_web_closed(memory, monkeypatch, told):
    from rune.capabilities.memory_capability import MemorySearchParams, memory_search

    broker = Broker(monkeypatch, fail=not told)
    remember_from_connector("diet", "vegetarian dinner")
    md.save_learned_fact("preference", "dinner_time", "dinner at 7", 0.9)
    await memory.initialize()
    result = await memory_search(MemorySearchParams(query="dinner"))
    assert ("vegetarian" in result.output) is told
    assert "dinner at 7" in result.output
    assert broker.calls == ["private"]


async def test_conversation_search_returns_a_connector_conversation_only_once_the_web_closed(memory, monkeypatch):
    from rune.capabilities.memory_capability import ConversationSearchParams, conversation_search

    Broker(monkeypatch, fail=True)
    memory.store.save_episode(Episode(id="e1", conversation_id="mail-chat", sources=provenance.encode(["gmail"])))

    class Store:
        def __init__(self, *_: Any) -> None: ...

        def fetch_searchable_turns(self) -> list[dict[str, Any]]:
            return [{"conversation_id": cid, "content": "invoice total", "content_hash": cid, "created_order": 0,
                     "title": cid, "timestamp": "2026-11-14"} for cid in ("mail-chat", "local-chat")]

        def get_turn_window(self, cid: str, *_: Any) -> list[dict[str, str]]:
            return [{"role": "user", "content": f"invoice total in {cid}"}]

    monkeypatch.setattr("rune.conversation.store.ConversationStore", Store)
    result = await conversation_search(ConversationSearchParams(query="invoice total"))
    assert "local-chat" in result.output
    assert "mail-chat" not in result.output


async def test_a_connector_run_saves_nothing_into_project_memory(memory, monkeypatch, tmp_path):
    from rune.capabilities.memory_capability import MemorySaveParams, memory_save

    monkeypatch.chdir(tmp_path)
    with provenance.run_sources():
        provenance.record_connector("gmail")
        await memory_save(MemorySaveParams(type="preference", content="likes tea", key="drink"))
    assert not (tmp_path / ".rune").exists()


# Forgetting a connector


async def test_forgetting_a_connector_removes_what_came_from_it_and_nothing_else(memory):
    from rune.memory.forget import connectors_remembered, forget_connector

    remember_from_connector("diet", "vegetarian")
    remember_from_connector("team", "platform", connector="calendar")
    md.save_learned_fact("preference", "editor", "vim", 0.9)
    md.append_to_memory_md("Preferences", "boss: Kim")
    provenance.note_facts(["boss"], connectors=["gmail"])  # promoted from learned.md earlier
    memory.store.save_episode(Episode(id="e1", task_summary="read mail", sources=provenance.encode(["gmail"])))
    memory.store.save_episode(Episode(id="e2", task_summary="local work"))
    md.append_daily_entry("read mail", [], connectors=["gmail"])
    await memory.initialize()
    assert [c["name"] for c in connectors_remembered()] == ["calendar", "gmail"]

    done = await forget_connector("gmail")

    assert done.payload() == {"facts": 2, "episodes": 1, "dailyEntries": 1}
    assert {f["key"] for f in md.parse_learned_md()} == {"team", "editor"}
    assert "boss: Kim" not in md.parse_memory_md().get("Preferences", [])
    assert [ep.id for ep in memory.store.get_recent_episodes(10)] == ["e2"]
    assert load_suppressed()["diet"]["reason"] == "connector_forgotten:gmail", "not drawn out again"
    assert "diet" not in memory.working.facts and "boss" not in memory.working.facts
    assert [c["name"] for c in connectors_remembered()] == ["calendar"]


# The memory screen's RPC


async def test_the_screen_lists_each_fact_with_where_it_came_from(memory):
    from rune.api.handlers.memory import handle

    remember_from_connector("diet", "vegetarian")
    md.save_learned_fact("preference", "editor", "vim", 0.9)
    listed = await handle("memory.list", {})
    by_key = {f["key"]: f for f in listed["learned"]}
    assert by_key["diet"]["origin"]["connectors"] == ["gmail"]
    assert by_key["editor"]["origin"]["connectors"] == []
    assert listed["connectors"] == [{"name": "gmail", "facts": 1, "episodes": 0}]


async def test_editing_forgetting_and_restoring_a_fact(memory):
    from rune.api.handlers.memory import handle

    md.save_learned_fact("preference", "editor", "vim", 0.9)
    await memory.initialize()
    await handle("memory.save", {"zone": "learned", "key": "editor", "value": "helix\nnot two lines"})
    assert md.learned_md_has_key("editor") == "helix not two lines"
    assert memory.working.facts["editor"] == "helix not two lines", "the next run reads the edit"

    listed = await handle("memory.forget", {"zone": "learned", "key": "editor"})
    assert listed["learned"] == [] and listed["suppressed"][0]["key"] == "editor"
    listed = await handle("memory.unsuppress", {"key": "editor"})
    assert listed["suppressed"] == []


async def test_editing_a_memory_md_line_that_changed_meanwhile_is_refused(memory):
    from rune.api.handlers.memory import handle

    md.append_to_memory_md("Notes", "deploy: friday")
    await handle("memory.save", {"zone": "memory", "section": "Notes", "old": "deploy: friday", "value": "deploy: monday"})
    assert md.parse_memory_md()["Notes"] == ["deploy: monday"]
    with pytest.raises(LookupError):
        await handle("memory.save", {"zone": "memory", "section": "Notes", "old": "deploy: friday", "value": "x"})
    with pytest.raises(ValueError):
        await handle("memory.save", {"zone": "memory", "section": "", "value": "x"})


async def test_the_rpc_forgets_a_connector_by_name(memory):
    from rune.api.handlers.memory import handle

    remember_from_connector("diet", "vegetarian")
    listed = await handle("memory.forget_connector", {"connector": "gmail"})
    assert listed["forgotten"]["facts"] == 1 and listed["connectors"] == []
    with pytest.raises(ValueError):
        await handle("memory.forget_connector", {"connector": " "})


def test_sources_are_stored_as_a_sorted_json_list():
    assert json.loads(provenance.encode(["b", "a", "a"])) == ["a", "b"]
    assert provenance.encode([]) == ""


async def test_the_agent_loop_puts_the_connectors_it_called_on_its_trace(monkeypatch):
    from rune.agent.loop import NativeAgentLoop
    from rune.types import CompletionTrace

    async def run(self, goal: str, **kwargs: Any) -> CompletionTrace:
        provenance.record_connector("gmail")
        return CompletionTrace(reason="completed")

    monkeypatch.setattr(NativeAgentLoop, "_run", run)
    loop = NativeAgentLoop.__new__(NativeAgentLoop)
    trace = await loop.run("read my mail")
    assert trace.connectors == ["gmail"]
    assert loop.connectors_contacted == ["gmail"]


async def test_the_saved_episode_and_daily_entry_keep_the_runs_connectors(memory):
    from rune.agent.memory_bridge import save_agent_result_to_memory

    await save_agent_result_to_memory(
        goal="summarise my inbox", memory_manager=memory, conversation_id="c1", classification_hint="research",
        result={"output": "3 mails", "success": True, "connectors": ["gmail"]},
    )
    (episode,) = memory.store.episodes_with_sources()
    assert provenance.decode(episode.sources) == ["gmail"] and episode.conversation_id == "c1"
    (daily,) = (md.memory_dir() / "daily").glob("*.md")
    assert md.parse_daily_log(daily)[-1]["connectors"] == ["gmail"]
