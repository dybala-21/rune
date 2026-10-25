"""Source checks through the streaming adapter, including a mistaken chat route."""

import json

import pytest

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.loop import NativeAgentLoop
from rune.capabilities.registry import CapabilityRegistry
from rune.capabilities.types import CapabilityDefinition
from rune.capabilities.web import WebFetchParams, WebSearchParams
from rune.types import AgentConfig, CapabilityResult
from tests.unit.test_task_blocked import _astream, _delta


@pytest.mark.parametrize("lookup", [True, False])
@pytest.mark.parametrize("rebuild", [False, True])
@pytest.mark.parametrize("provider,model", [("openai", "gpt-6-astra"), ("xai", "grok-4.7"),
                                          ("anthropic", "claude-opus-5"), ("gemini", "gemini-3.1-pro-preview"),
                                          ("azure", "gpt-6-astra")])
async def test_frame_research_preserves_source_body_and_stops_without_an_extra_model_call(monkeypatch, tmp_path, lookup, rebuild, provider, model):
    from types import SimpleNamespace

    import httpx

    from rune.agent.failover import LLMProfile
    from rune.agent.request_budget import input_estimate
    from rune.api.trust import build_trust_payload
    from rune.capabilities.registry import _register_all_capabilities
    from rune.config import get_config

    for name, value in {"RUNE_HOME": str(tmp_path / "home"), "RUNE_IN_BEST_OF": "1", "RUNE_AUTO_SKILL": "0",
                        "RUNE_AUTO_VERIFY": "0", "RUNE_REQUIREMENT_GATE": "0", "RUNE_ADVISOR": "0",
                        "RUNE_BENCH_EVIDENCE_GATE": "0", "LITELLM_LOCAL_MODEL_COST_MAP": "True"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("RUNE_TOOL_ROUNDS", raising=False)
    monkeypatch.setenv("VERTEX_LOCATION", "global")
    monkeypatch.setenv("AZURE_API_BASE", "https://fixture.openai.azure.com")
    monkeypatch.setenv("AZURE_API_KEY", "test-only")
    get_config().llm.route_simple_queries = False
    monkeypatch.setattr("rune.agent.failover.build_profiles_from_config", lambda: [
        LLMProfile(name=model, provider=provider, model=model),
    ])
    monkeypatch.setattr("rune.agent.loop._compute_tool_rounds", lambda *args, **kwargs: 1)
    pages = {
        "/": '<frameset><frame src="/main"></frameset>',
        "/main": '<table><tr><td><iframe src="/news"></iframe></td></tr></table>',
        "/news": '<main>HQ-217: Login reopened on 2026-10-22. Player counts are unpublished.</main>'
                 + ''.join(f'<iframe src="/ads/{i}"></iframe>' for i in range(20)),
    }
    reads, requests = [], []
    checkpoints = []
    monkeypatch.setattr("rune.agent.loop.CheckpointManager.save", lambda *args: checkpoints.append(True))

    async def send(client, req, *args, **kwargs):
        reads.append(req.url.path)
        assert req.url.host == "research.example" and req.url.path in pages
        return httpx.Response(200, text=pages[req.url.path], headers={"content-type": "text/html"}, request=req)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    registry = CapabilityRegistry()
    _register_all_capabilities(registry)
    monkeypatch.setattr("rune.agent.tool_adapter.get_capability_registry", lambda: registry)
    paths = list(pages)
    answer = "HQ-217: Login reopened on 2026-10-22. Player counts are unpublished. https://research.example/news"

    async def completion(**kwargs):
        requests.append(kwargs)
        step = len(requests)
        assert step <= 2
        usage = SimpleNamespace(choices=[], usage={"prompt_tokens": 22_000 if rebuild else input_estimate(kwargs),
                                                  "completion_tokens": 100})
        if step == 1:
            assert {tool["function"]["name"] for tool in kwargs["tools"]} == {
                "web_search", "web_fetch", "task_blocked", "tool_search",
            }
            return _astream([_delta(tool_calls=[{"index": 0, "id": f"read{step}", "name": "web_fetch",
                "arguments": json.dumps({"url": "https://research.example/", "maxLength": 300})}], finish_reason="tool_calls"), usage])
        assert {tool["function"]["name"] for tool in kwargs.get("tools", [])} <= {"task_blocked"}
        assert "HQ-217: Login reopened on 2026-10-22. Player counts are unpublished." in str(kwargs["messages"])
        assert "[SOURCE_URL: https://research.example/news]" in str(kwargs["messages"])
        return _astream([_delta(content=answer, finish_reason="stop"), usage])

    monkeypatch.setattr("rune.agent.litellm_adapter.litellm.acompletion", completion)
    loop = NativeAgentLoop(AgentConfig(model=model, provider=provider, max_iterations=2, max_tokens=1024))
    loop._token_budget.total = 50_000
    if rebuild:
        monkeypatch.setattr(loop, "_get_disabled_tools", lambda: {"browser_find"})
    from rune.agent.prompts import build_system_prompt

    classification = ClassificationResult("web", .99, 2, needs_web_evidence=True, needs_current_source=True,
                                         browser_effect="view", lookup_only=lookup,
                                         complexity="moderate" if rebuild and not lookup else "simple")
    prompt = build_system_prompt("Research recent service notices", classification, goal_category="web")
    assert "read the relevant frame URL before opening a browser" in prompt
    trace = await loop._execute_loop(goal="Research recent service notices", system_prompt=prompt,
        tools=loop._select_tools(classification), max_iterations=2,
        classification=classification,
        context={"workspace_root": str(tmp_path), "ephemeral": True})
    assert trace.reason == "completed" and loop._last_answer_text == answer
    assert trace.source_review["freshSources"] == 1
    assert reads == paths and len(requests) == 2
    request = requests[0]
    effort = (request.get("output_config", {}).get("effort") if provider == "anthropic" else
              request.get("thinkingConfig", {}).get("thinkingLevel") if provider == "gemini" else
              request.get("extra_body", {}).get("reasoning_effort") if provider == "xai" else
              request.get("reasoning_effort", request.get("extra_body", {}).get("reasoning", {}).get("effort")))
    assert effort == ("low" if lookup or not rebuild else None)
    assert "https://research.example/news" in loop._citation_urls
    assert not checkpoints and not loop._rollover_phase_done
    payload = build_trust_payload(trace)
    assert payload["answerReserveReason"] == "tool_rounds" and not payload["budgetExhausted"]


@pytest.mark.parametrize("case", ["verbatim", "repair", "derived", "summary", "current", "unrelated"])
async def test_source_context_is_reused_only_for_the_same_historical_task(monkeypatch, tmp_path, case):
    from rune.agent.source_context import HISTORY_KEY
    from rune.agent.task_context import HISTORY_KEY as TASK_KEY
    from rune.agent.task_context import from_history
    from rune.api.conversation_wiring import record_assistant_turn
    from rune.conversation.manager import ConversationManager
    from rune.conversation.store import ConversationStore

    for name, value in {"RUNE_HOME":str(tmp_path / "home"), "RUNE_IN_BEST_OF":"1", "RUNE_AUTO_SKILL":"0",
                        "RUNE_AUTO_VERIFY":"0", "RUNE_REQUIREMENT_GATE":"0", "RUNE_ADVISOR":"0",
                        "RUNE_BENCH_EVIDENCE_GATE":"0", "LITELLM_LOCAL_MODEL_COST_MAP":"True"}.items():
        monkeypatch.setenv(name, value)
    from rune.config import get_config

    get_config().llm.route_simple_queries = False
    registry = CapabilityRegistry()
    reads, calls = [], []
    phase = {"follow": False, "calls": 0}
    original = {"station":"North", "value":17.125, "observed_at":"2026-10-16T00:00:00Z"}

    async def retrieve(params):
        reads.append(params)
        return CapabilityResult(success=True, output=json.dumps(original),
                                metadata={"content_type":"application/json", "url":params.url})

    registry.register(CapabilityDefinition(name="web_fetch", description="Read data", execute=retrieve,
                                          parameters_model=WebFetchParams))

    async def completion(**kwargs):
        calls.append(kwargs)
        phase["calls"] += 1
        assert all(HISTORY_KEY not in msg for msg in kwargs["messages"])
        assert all(TASK_KEY not in msg for msg in kwargs["messages"])
        if phase["follow"]:
            historical = "Untrusted historical source data" in str(kwargs["messages"])
            assert historical is (case not in {"current", "unrelated"})
        if phase["calls"] == 1 and (not phase["follow"] or case == "current"):
            return _astream([_delta(tool_calls=[{"index":0,"id":f"read{len(reads)}","name":"web_fetch",
                "arguments":json.dumps({"url":"https://example.org/observations"})}], finish_reason="tool_calls")])
        answer = original.copy()
        if phase["follow"]:
            if case == "derived":
                answer["observed_at"] = "2026-10-16T09:00:00+09:00"
            elif case == "repair" and phase["calls"] == 1:
                answer["value"] = 120
            elif case == "repair":
                assert "[Source check]" in str(kwargs["messages"])
        text = "The earlier value was 17.125." if phase["follow"] and case == "summary" else json.dumps(answer)
        return _astream([_delta(content=text, finish_reason="stop")])

    monkeypatch.setattr("rune.agent.tool_adapter.get_capability_registry", lambda:registry)
    monkeypatch.setattr("rune.agent.litellm_adapter.litellm.acompletion", completion)
    manager = ConversationManager(ConversationStore(tmp_path / "conversation.db"))
    conv = manager.start_conversation("test")
    manager.add_turn(conv.id, "user", "Read the latest observation")
    initial = NativeAgentLoop(AgentConfig(model="openai/gpt-5.4", max_iterations=3))
    trace = await initial._execute_loop(goal="Read the latest observation", system_prompt="Read data.", tools=["web_fetch"],
        max_iterations=3, classification=ClassificationResult("web", .99, 2, needs_web_evidence=True,
        needs_current_source=True, lookup_only=True, source_output="verbatim"),
        context={"workspace_root":str(tmp_path), "ephemeral":True})
    assert trace.reason == "completed" and len(reads) == 1
    await record_assistant_turn(manager, conv.id, initial, "", embed=False)
    from types import SimpleNamespace

    manager.add_turn(conv.id, "user", "Thanks")
    await record_assistant_turn(manager, conv.id, SimpleNamespace(_last_answer_text="You're welcome", _last_goal_type="chat"), "", embed=False)
    manager._active.clear()
    manager._active[conv.id] = await manager._store.load(conv.id)
    history = manager._build_messages_within_budget(conv.id, 2000)
    task_id = from_history(history)[0]["id"]
    phase.update(follow=True, calls=0)
    follow = NativeAgentLoop(AgentConfig(model="openai/gpt-5.4", max_iterations=3))
    classification = ClassificationResult("chat", .99, 2, is_continuation=case != "unrelated",
        is_domain_change=case == "unrelated", needs_current_source=case == "current",
        resume_task=task_id if case != "unrelated" else "",
        needs_web_evidence=case == "current", source_output=case if case in {"derived", "summary"} else "verbatim")
    trace = await follow._execute_loop(goal="Continue as requested", system_prompt="Use the requested format.",
        tools=["web_fetch"], max_iterations=3, classification=classification, message_history=history,
        context={"workspace_root":str(tmp_path), "ephemeral":True})
    assert trace.reason == "completed"
    assert len(reads) == (2 if case == "current" else 1)
    assert phase["calls"] == (2 if case in {"current", "repair"} else 1)
    assert trace.source_review["reusedSources"] == int(case not in {"current", "unrelated"})
    assert trace.source_review["status"] == ("supported" if case in {"verbatim", "repair", "current"} else "not_checked")


@pytest.mark.parametrize("source_tool", ["web_fetch", "browser_discover_apis"])
@pytest.mark.parametrize("wrong", [False, True])
async def test_structured_source_repair_runs_only_for_mismatched_records(monkeypatch, tmp_path, source_tool, wrong):
    from rune.capabilities.browser.discover import BrowserDiscoverApisParams

    for name, value in {"RUNE_HOME":str(tmp_path / "home"), "RUNE_IN_BEST_OF":"1", "RUNE_AUTO_SKILL":"0",
                        "RUNE_AUTO_VERIFY":"0", "RUNE_REQUIREMENT_GATE":"0", "RUNE_ADVISOR":"0",
                        "RUNE_BENCH_EVIDENCE_GATE":"0", "LITELLM_LOCAL_MODEL_COST_MAP":"True"}.items():
        monkeypatch.setenv(name, value)
    from rune.config import get_config

    get_config().llm.route_simple_queries = False
    registry = CapabilityRegistry()
    calls, reads = [], []
    answer = {"symbol":"ACME", "value":210, "unit":"USD", "observed_at":"2026-10-16"}

    async def retrieve(params):
        reads.append(params)
        return CapabilityResult(success=True, output=json.dumps({"unit":"USD", "data":[
            {"symbol":"OTHER", "value":120, "observed_at":"2026-10-16"}, answer,
        ]}), metadata={"content_type":"application/json", "captured_body":source_tool == "browser_discover_apis"})

    registry.register(CapabilityDefinition(name=source_tool, description="Read source", execute=retrieve,
                                          parameters_model=WebFetchParams if source_tool == "web_fetch" else BrowserDiscoverApisParams))

    async def completion(**kwargs):
        calls.append(kwargs)
        step = len(calls) - int(source_tool == "browser_discover_apis")
        if step == 0:
            return _astream([_delta(tool_calls=[{"index":0,"id":"load1","name":"tool_search",
                                                "arguments":json.dumps({"query":source_tool})}], finish_reason="tool_calls")])
        if step == 1:
            args = {"url":"https://example.org/quote"} if source_tool == "web_fetch" else {"readBody":"quote"}
            return _astream([_delta(tool_calls=[{"index":0,"id":"read1","name":source_tool,"arguments":json.dumps(args)}],
                                   finish_reason="tool_calls")])
        if step == 3:
            assert "do not match a single retrieved record" in str(kwargs["messages"])
        text = json.dumps(answer | {"value":120}) if wrong and step == 2 else json.dumps(answer)
        return _astream([_delta(content=text, finish_reason="stop")])

    monkeypatch.setattr("rune.agent.tool_adapter.get_capability_registry", lambda:registry)
    monkeypatch.setattr("rune.agent.litellm_adapter.litellm.acompletion", completion)
    classification = ClassificationResult("web", .99, 2, needs_web_evidence=True, needs_current_source=True, lookup_only=True)
    loop = NativeAgentLoop(AgentConfig(model="openai/gpt-5.4", max_iterations=4))
    loop._auto_skill = False
    trace = await loop._execute_loop(goal="Read ACME's latest quote as JSON", system_prompt="Read the source.",
                                     tools=[source_tool], max_iterations=4, classification=classification,
                                     context={"workspace_root":str(tmp_path), "ephemeral":True})
    assert trace.reason == "completed"
    assert len(reads) == 1 and len(calls) == (3 if wrong else 2) + int(source_tool == "browser_discover_apis")


@pytest.mark.parametrize("case", ["chat", "missing", "memory_only", "search", "fetch", "web_lookup", "empty", "failed", "recovered", "misrouted", "reserve", "reserve_empty", "explicit_limit", "continued_lookup", "explicit_research", "current_repaired", "current_refused", "current_excerpt"])
async def test_sources_are_required_without_redundant_model_calls(monkeypatch, tmp_path, case):
    for name, value in {
        "RUNE_HOME": str(tmp_path / "home"), "RUNE_IN_BEST_OF": "1",
        "RUNE_AUTO_SKILL": "0", "RUNE_AUTO_VERIFY": "0", "RUNE_REQUIREMENT_GATE": "0",
        "RUNE_ADVISOR": "0", "RUNE_BENCH_EVIDENCE_GATE": "0",
        "RUNE_BENCH_CAPTURE_FAILED_TOOL_OUTPUT": "0", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }.items():
        monkeypatch.setenv(name, value)
    requests, actions = [], []
    monkeypatch.delenv("RUNE_TOOL_ROUNDS", raising=False)
    from rune.config import get_config

    get_config().llm.route_simple_queries = False
    if case.startswith("reserve") or case == "web_lookup":
        monkeypatch.setattr("rune.agent.loop._compute_tool_rounds", lambda *args, **kwargs: 1)
    if case == "explicit_limit":
        monkeypatch.setenv("RUNE_TOOL_ROUNDS", "3")
    registry = CapabilityRegistry()

    async def retrieve(params):
        actions.append(params)
        if case == "current_excerpt" and isinstance(params, WebFetchParams):
            return CapabilityResult(success=True, output="Quotes unavailable; 20 minute delay")
        if case in {"empty", "reserve_empty"}:
            return CapabilityResult(success=True, output="No search results found.", metadata={"count": 0})
        if case == "failed":
            return CapabilityResult(success=False, error="Source unavailable")
        return CapabilityResult(success=True, output="Observed value: 120 on 2026-10-15", metadata={"count": 1})

    registry.register(CapabilityDefinition(name="web_search", description="Search",
                                          parameters_model=WebSearchParams, execute=retrieve))
    registry.register(CapabilityDefinition(name="web_fetch", description="Read page",
                                          parameters_model=WebFetchParams, execute=retrieve))
    from rune.capabilities.memory_capability import MemorySearchParams

    async def recall(params):
        return CapabilityResult(success=True, output="An old preference, not a current source")

    registry.register(CapabilityDefinition(name="memory_search", description="Recall",
                                          parameters_model=MemorySearchParams, execute=recall))

    async def completion(**kwargs):
        requests.append(kwargs)
        step = len(requests)
        assert step <= 5
        if case == "current_excerpt" and step == 1:
            return _astream([_delta(tool_calls=[
                {"index": 0, "id": "search1", "name": "web_search", "arguments": json.dumps({"query": "current quote"})},
                {"index": 1, "id": "fetch1", "name": "web_fetch", "arguments": json.dumps({"url": "https://example.org/quote"})},
            ], finish_reason="tool_calls")])
        lookup = (step == 1 and case in {"search", "fetch", "web_lookup", "empty", "failed", "misrouted", "memory_only", "reserve", "reserve_empty"}
                  or step == 2 and case == "recovered"
                  or step == 1 and case in {"current_repaired", "current_refused"}
                  or step == 3 and case == "current_repaired")
        lookup = lookup or case == "explicit_limit" and step <= 3 or case in {"continued_lookup", "explicit_research"} and step <= 4
        if lookup:
            if case == "recovered" or case == "current_repaired" and step == 3:
                assert "[Source check]" in str(kwargs["messages"])
            name = "web_fetch" if case in {"fetch", "web_lookup"} or case == "current_repaired" and step == 3 else "memory_search" if case == "memory_only" else "web_search"
            args = {"url": "https://example.org/prices"} if name == "web_fetch" else {"query": f"current conditions {step}"}
            return _astream([_delta(tool_calls=[{"index": 0, "id": f"call{step}", "name": name,
                                                "arguments": json.dumps(args)}], finish_reason="tool_calls")])
        if case.startswith("reserve") or case in {"explicit_limit", "web_lookup"}:
            assert not kwargs.get("tools") and "No more tools" in str(kwargs["messages"])
        if case in {"continued_lookup", "explicit_research"}:
            assert kwargs.get("tools")
        return _astream([_delta(content="Current price: 120" if case == "current_excerpt" else "Here is my response.", finish_reason="stop")])

    monkeypatch.setattr("rune.agent.tool_adapter.get_capability_registry", lambda: registry)
    monkeypatch.setattr("rune.agent.litellm_adapter.litellm.acompletion", completion)
    classification = ClassificationResult(goal_type="web" if case in {"explicit_research", "web_lookup"} else "chat", confidence=.99, tier=2,
                                         needs_web_evidence=case not in {"chat", "misrouted"},
                                         lookup_only=case in {"fetch", "web_lookup"},
                                         needs_current_source=case in {"current_repaired", "current_refused", "current_excerpt"})
    loop = NativeAgentLoop(AgentConfig(model="openai/gpt-5.6" if case in {"fetch", "web_lookup"} else "openai/gpt-5.4", max_iterations=4))
    loop._auto_skill = False
    loop._token_budget.total = 300_000
    trace = await loop._execute_loop(
        goal="Respond to the current concern", system_prompt="Respond helpfully.",
        tools=loop._select_tools(classification), max_iterations=4, classification=classification,
        context={"workspace_root": str(tmp_path), "ephemeral": True},
    )
    should_complete = case in {"chat", "search", "fetch", "web_lookup", "recovered", "misrouted", "reserve", "explicit_limit", "continued_lookup", "explicit_research", "current_repaired", "current_excerpt"}
    assert trace.reason == ("completed" if should_complete else "completed_gate_warnings")
    if not should_complete:
        assert trace.completion_check["name"] == "Source retrieval"
    assert len(actions) == (2 if case in {"current_repaired", "current_excerpt"} else 1 if case == "current_refused" else 3 if case == "explicit_limit" else 4 if case in {"continued_lookup", "explicit_research"} else
                            int(case in {"search", "fetch", "web_lookup", "empty", "failed", "recovered", "misrouted", "reserve", "reserve_empty"}))
    assert len(requests) == (4 if case in {"explicit_limit", "current_repaired"} else 5 if case in {"continued_lookup", "explicit_research"} else
                            1 if case == "chat" else 2 if case in {"missing", "search", "fetch", "web_lookup", "misrouted", "reserve", "reserve_empty", "current_excerpt"} else 3)
    assert all(call.get("stream") for call in requests)
    from rune.api.trust import build_trust_payload

    expected_limit = "tool_rounds" if case in {"reserve", "reserve_empty", "web_lookup", "explicit_limit"} else ""
    payload = build_trust_payload(trace)
    assert trace.answer_reserve_reason == payload["answerReserveReason"] == expected_limit
    assert not payload["budgetExhausted"]
    if case == "current_excerpt":
        assert payload["sourceReview"]["excerptOnlyNumbers"] == ["120"]
        assert not payload["verified"]
    efforts = [call.get("reasoning_effort", call.get("extra_body", {}).get("reasoning", {}).get("effort")) for call in requests]
    assert efforts == [("low" if case in {"fetch", "web_lookup"} else None)] * len(requests)
