"""Decision routing, fallback behavior, settings persistence, and key changes."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from rune.agent import decision_router
from rune.agent.goal_classifier import _TIER2_SYSTEM_PROMPT, classify_goal
from rune.config.schema import RuneConfig
from rune.llm import jev


@pytest.fixture
def setup(monkeypatch):
    import rune.llm.client
    import rune.llm.model_selection

    cfg = RuneConfig()
    cfg.llm.active_provider = "xai"
    cfg.llm.active_model = "grok-4.6"
    monkeypatch.setattr("rune.config.get_config", lambda: cfg)
    monkeypatch.setattr(rune.llm.model_selection, "get_config", lambda: cfg)
    monkeypatch.setattr(rune.llm.client, "get_config", lambda: cfg)
    monkeypatch.setattr(decision_router, "_cooldown", None)
    monkeypatch.setattr(decision_router, "_rejected_key", None)
    monkeypatch.setattr(decision_router, "_verified_key", None)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only-key")
    values = {"goal_type": "chat", "confidence": .95, "reason": "explanation",
              "requires_execution": False, "intent_categories": [], "requires_desktop_input": False,
              "is_related_to_previous": False, "table_output": "none", "calculation_expression": ""}
    client = SimpleNamespace(completion=AsyncMock(return_value={
        "choices": [{"message": {"content": json.dumps(values)}}],
    }))
    monkeypatch.setattr("rune.llm.client.get_llm_client", lambda: client)
    return cfg, client


def payload(questions, **picks):
    defaults = {"goal_type": "chat", "desktop": "none", "calculation": "none", "table_output": "none"}
    answers = {}
    for name, question in questions.items():
        choice = picks.get(name, defaults.get(name, "no"))
        answers[name] = {"type": "choice", "choice": choice, "confidence": .99,
                         "probabilities": {key: float(key == choice) for key in question["criteria"]}}
    return {"model": jev.MODEL, "answers": answers, "usage": {"input_tokens": 2000, "output_tokens": 200}}


@pytest.mark.parametrize("uncertain_routing", [False, True])
async def test_file_roles_survive_independent_routing_abstention(setup, monkeypatch, uncertain_routing):
    cfg, client = setup
    cfg.llm.decision_routing.backend = "jev"
    goal = "Read source.csv and create summary.csv."

    async def request(**kwargs):
        return payload(kwargs["questions"], goal_type="unknown" if uncertain_routing else "code_modify",
                       file_role_0="input", file_role_1="output")

    external = AsyncMock(side_effect=request)
    monkeypatch.setattr(jev, "_request", external)
    result = await classify_goal(goal)
    assert result.available
    assert result.decision_backend == ("connected" if uncertain_routing else "jev")
    assert result.fallback_reason == ("uncertain" if uncertain_routing else "")
    assert result.artifact_roles.matching_roles(goal) == {"source.csv": "input", "summary.csv": "output"}
    assert external.await_count == 1 and client.completion.await_count == int(uncertain_routing)
    assert decision_router.accelerator_status() == "ready"


@pytest.mark.parametrize("choice,confidence", [("unknown", .99), ("output", .89)])
async def test_uncertain_file_does_not_discard_routing_or_other_roles(setup, monkeypatch, choice, confidence):
    cfg, client = setup
    cfg.llm.decision_routing.backend = "jev"
    goal = "Read source.csv and create summary.csv."

    async def request(**kwargs):
        data = payload(kwargs["questions"], goal_type="code_modify", file_role_0="input", file_role_1=choice)
        data["answers"]["file_role_1"]["confidence"] = confidence
        return data

    monkeypatch.setattr(jev, "_request", request)
    result = await classify_goal(goal)
    assert result.decision_backend == "jev"
    assert result.artifact_roles.matching_roles(goal) == {"source.csv": "input"}
    client.completion.assert_not_called()


async def test_invalid_file_answer_rejects_entire_batch_even_when_routing_abstains(setup, monkeypatch):
    cfg, _ = setup
    cfg.llm.decision_routing.backend = "jev"

    async def request(**kwargs):
        data = payload(kwargs["questions"], goal_type="unknown", file_role_0="input", file_role_1="output")
        data["answers"]["file_role_1"]["probabilities"] = None
        return data

    monkeypatch.setattr(jev, "_request", request)
    result = await classify_goal("Read source.csv and create summary.csv.")
    assert result.available and result.fallback_reason == "invalid_probabilities"
    assert result.artifact_roles is None
    assert decision_router.accelerator_status() == "cooldown"


def test_role_hints_cannot_cross_requests_or_serialized_runs():
    from dataclasses import asdict

    from rune.agent.goal_classifier import ClassificationResult, from_wire, to_wire
    from rune.agent.provenance import ArtifactRoleHints

    goal = "Create report.md"
    hints = ArtifactRoleHints.for_request(goal, {"report.md": "output", "unmentioned.md": "input"})
    assert hints.matching_roles(goal) == {"report.md": "output"}
    assert hints.matching_roles("Read report.md") == {}
    result = ClassificationResult("code_modify", .99, 2, artifact_roles=hints)
    assert from_wire(to_wire(result)).artifact_roles is None
    raw = json.loads(to_wire(result))
    raw["artifact_roles"] = asdict(hints)
    assert from_wire(json.dumps(raw)).artifact_roles is None


def test_file_role_batch_excludes_ambiguous_names_and_respects_limits(monkeypatch):
    from rune.agent.provenance import role_hint_names

    assert role_hint_names("Read src/config.json and create dst/config.json and notes.md") == ["notes.md"]
    assert role_hint_names("report.txtを読んでsummary.mdに要約。report.txtは変更しない。") == ["report.txt"]
    assert role_hint_names("Read レポート.md and create 要約.md") == ["レポート.md", "要約.md"]
    assert role_hint_names("https://example.org/notes.md") == []
    assert role_hint_names(" ".join(f"file{i}.txt" for i in range(13))) == []
    monkeypatch.setenv("RUNE_ARTIFACT_PROVENANCE", "0")
    assert role_hint_names("Read source.csv") == []


async def test_key_alone_does_not_enable_external_routing(setup, monkeypatch):
    cfg, client = setup
    external = AsyncMock()
    monkeypatch.setattr(jev, "classify", external)
    result = await classify_goal("Explain TCP sockets")
    assert result.available and result.decision_backend == "connected"
    assert client.completion.await_count == 1
    external.assert_not_called()


@pytest.mark.parametrize("local,has_key,reason", [(True, True, "local"), (False, False, "missing_key")])
async def test_local_and_missing_key_use_current_model(setup, monkeypatch, local, has_key, reason):
    cfg, client = setup
    cfg.llm.decision_routing.backend = "jev"
    if local:
        cfg.llm.active_provider, cfg.llm.active_model = "ollama", "local-model"
    if not has_key:
        monkeypatch.delenv("TYPESAFE_API_KEY")
    external = AsyncMock()
    monkeypatch.setattr(jev, "classify", external)
    result = await classify_goal("Explain this")
    assert result.available and result.fallback_reason == reason
    assert client.completion.call_args.kwargs["provider"] == cfg.llm.active_provider
    external.assert_not_called()


async def test_success_uses_one_request_and_reports_cost(setup, monkeypatch):
    from rune.agent.calculation import calculation_context
    from rune.agent.timing import capture_timing, timing_snapshot

    cfg, client = setup
    cfg.llm.decision_routing.backend = "jev"

    async def request(**kwargs):
        return payload(kwargs["questions"], calculation="expression_0")

    external = AsyncMock(side_effect=request)
    monkeypatch.setattr(jev, "_request", external)
    with capture_timing() as run:
        result = await classify_goal("계산해줘: 0.1 + 0.2")
        timing = timing_snapshot(run)
    assert result.decision_backend == "jev" and result.decision_model == jev.MODEL
    assert '"result": "0.3"' in calculation_context("계산해줘: 0.1 + 0.2", result)
    assert external.await_count == 1
    client.completion.assert_not_called()
    assert timing["usage"]["calls"] == 1
    assert timing["usage"]["cost_usd"] == pytest.approx(.000084)
    assert "test-only-key" not in str(timing) and "계산해줘" not in str(timing)


@pytest.mark.parametrize("choice", ["unknown", "unavailable"])
async def test_abstention_falls_back_without_disabling_other_requests(setup, monkeypatch, choice):
    cfg, client = setup
    cfg.llm.decision_routing.backend = "jev"

    async def request(**kwargs):
        return payload(kwargs["questions"], calculation=choice)

    monkeypatch.setattr(jev, "_request", request)
    result = await classify_goal("Calculate the expression")
    assert result.available and result.decision_backend == "connected"
    assert result.fallback_reason in {"uncertain", "extraction_needed"}
    assert client.completion.await_count == 1
    assert decision_router.accelerator_status() == "ready"


@pytest.mark.parametrize("failure", [jev.JevUnavailable("http_429"), jev.JevUnavailable("invalid_answers"), TimeoutError()])
async def test_service_failure_pauses_calls_and_new_key_can_retry(setup, monkeypatch, failure):
    cfg, client = setup
    cfg.llm.decision_routing.backend = "jev"
    external = AsyncMock(side_effect=failure)
    monkeypatch.setattr(jev, "classify", external)
    for _ in range(2):
        assert (await classify_goal("Explain sockets")).available
    assert external.await_count == 1 and client.completion.await_count == 2
    assert decision_router.accelerator_status() == "cooldown"
    monkeypatch.setenv("TYPESAFE_API_KEY", "replacement-test-key")
    assert decision_router.accelerator_status() == "unverified"


async def test_cancellation_does_not_start_fallback(setup, monkeypatch):
    cfg, client = setup
    cfg.llm.decision_routing.backend = "jev"
    entered = asyncio.Event()

    async def wait(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(jev, "classify", wait)
    task = asyncio.create_task(classify_goal("Explain sockets"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    client.completion.assert_not_called()
    assert decision_router.accelerator_status() == "unverified"


async def test_accelerator_and_fallback_share_one_deadline(setup, monkeypatch):
    cfg, client = setup
    cfg.llm.decision_routing.backend = "jev"
    monkeypatch.setattr("rune.agent.classification_response.ROUTING_TIMEOUT", .03)

    async def wait(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(jev, "classify", wait)
    client.completion.side_effect = wait
    result = await asyncio.wait_for(classify_goal("Explain sockets"), .2)
    assert not result.available
    assert client.completion.await_count <= 1


@pytest.mark.parametrize("field,value", [
    ("confidence", float("nan")), ("confidence", True), ("choice", "not-an-option"),
    ("probabilities", {"chat": 1}), ("probabilities", None),
])
def test_invalid_answers_are_rejected(field, value):
    questions, expressions = jev.build_questions({"request_to_classify": "Explain sockets"})
    data = payload(questions)
    data["answers"]["goal_type"][field] = value
    with pytest.raises(jev.JevUnavailable):
        jev.decode_answers(data, questions, expressions)


@pytest.mark.parametrize("picks", [
    {"desktop": "input", "calculation": "expression_0"},
    {"desktop": "read", "goal_type": "chat"},
    {"calculation": "expression_0", "goal_type": "code_modify"},
    {"table_output": "csv", "goal_type": "chat"},
])
def test_conflicting_decisions_fall_back(picks):
    questions, expressions = jev.build_questions({"request_to_classify": "Calculator: 1 + 2"})
    with pytest.raises(jev.DecisionAbstained, match="inconsistent"):
        jev.decode_answers(payload(questions, **picks), questions, expressions)


@pytest.mark.parametrize("goal,expected", [
    ("계산: 173 × 29 − 417", ["173 × 29 − 417"]),
    ("Evaluate .5 + .25.", [".5 + .25"]),
    ("Compute (2 + 3) * 4", ["(2 + 3) * 4"]),
    ("Compute −2 + 3", ["−2 + 3"]),
    ("Compute 1,000 + 2", []),
    ("Compute 1,000 + 2 * 3", []),
    ("Compute 12 % 5 + 2", []),
    ("variable_123+4", []),
    ("variable_123+4*5", []),
    ("Compute 1 / 0", []),
])
def test_numeric_candidates_do_not_invent_or_truncate_expressions(goal, expected):
    from rune.agent.calculation import expression_candidates

    assert expression_candidates(goal) == expected


async def test_http_failure_never_exposes_response_or_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only-key")
    original = httpx.AsyncClient

    def handler(request):
        assert request.url == jev.ENDPOINT
        assert request.headers["authorization"] == "Bearer test-only-key"
        return httpx.Response(401, json={"private": "request content and credentials"})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    with pytest.raises(jev.JevUnavailable, match="^http_401$") as error:
        await jev.classify(_TIER2_SYSTEM_PROMPT, json.dumps({"request_to_classify": "hi"}), timeout=1)
    assert "credentials" not in str(error.value) and "test-only-key" not in str(error.value)


async def test_settings_require_opt_in_and_save_before_applying(setup, monkeypatch):
    from fastapi import HTTPException

    from rune.api.handlers.config import ConfigPatchRequest, patch_config

    cfg, _ = setup
    monkeypatch.setattr("rune.config.save_config_values", lambda updates: None)
    with pytest.raises(HTTPException) as error:
        await patch_config(ConfigPatchRequest(decisionRouting={"backend": "jev"}))
    assert error.value.status_code == 500 and cfg.llm.decision_routing.backend == "connected"
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(HTTPException) as error:
        await patch_config(ConfigPatchRequest(decisionRouting={"backend": "jev"}))
    assert error.value.status_code == 400


def test_accelerator_confidence_cannot_downshift_the_answer_model(setup, monkeypatch):
    from rune.agent.fast_lane import decide_fast_lane

    cfg, _ = setup
    cfg.llm.active_model = None
    monkeypatch.setattr("rune.config.loader.get_config", lambda: cfg)
    result = decide_fast_lane(SimpleNamespace(goal_type="chat", confidence=.99, decision_backend="jev"))
    assert not result.active


@pytest.mark.parametrize("backend", ["connected", "jev"])
async def test_automatic_model_selection_keeps_its_single_classifier_and_fast_model(setup, monkeypatch, backend):
    from rune.agent.fast_lane import decide_fast_lane
    from rune.types import Provider

    cfg, client = setup
    cfg.llm.active_model = None
    cfg.llm.default_provider = "xai"
    cfg.llm.decision_routing.backend = backend
    monkeypatch.setattr("rune.config.loader.get_config", lambda: cfg)
    client._effective_provider = lambda _: Provider.XAI
    client.resolve_model = lambda *args: "grok-4.3"
    external = AsyncMock()
    monkeypatch.setattr(jev, "classify", external)
    result = await classify_goal("Explain TCP sockets")
    lane = decide_fast_lane(result)
    assert result.available and result.decision_backend == "connected"
    assert lane.active and lane.model == "xai/grok-4.3"
    assert client.completion.await_count == 1
    external.assert_not_called()


async def test_jev_still_runs_when_automatic_model_selection_is_disabled(setup, monkeypatch):
    cfg, client = setup
    cfg.llm.active_model = None
    cfg.llm.route_simple_queries = False
    cfg.llm.decision_routing.backend = "jev"
    values = json.loads(client.completion.return_value["choices"][0]["message"]["content"])
    external = AsyncMock(return_value=jev.DecisionBatch(values))
    monkeypatch.setattr(jev, "classify", external)
    assert decision_router.accelerator_status() == "unverified"
    result = await classify_goal("Explain TCP sockets")
    assert result.decision_backend == "jev"
    assert decision_router.accelerator_status() == "ready"
    client.completion.assert_not_called()


@pytest.mark.parametrize("code", [401, 403])
async def test_auth_rejection_stays_blocked_until_a_different_key_is_used(setup, monkeypatch, code):
    cfg, client = setup
    cfg.llm.decision_routing.backend = "jev"
    clock = SimpleNamespace(monotonic=lambda: 100.0)
    monkeypatch.setattr(decision_router, "time", clock)
    external = AsyncMock(side_effect=jev.JevUnavailable(f"http_{code}"))
    monkeypatch.setattr(jev, "classify", external)
    assert (await classify_goal("Explain sockets")).available
    clock.monotonic = lambda: 86400.0
    assert decision_router.accelerator_status() == "auth_error"
    assert (await classify_goal("Explain sockets")).available
    assert external.await_count == 1 and client.completion.await_count == 2
    monkeypatch.setenv("TYPESAFE_API_KEY", "replacement-test-key")
    assert decision_router.accelerator_status() == "unverified"
    assert (await classify_goal("Explain sockets")).available
    assert external.await_count == 2


async def test_inflight_failure_does_not_reject_a_replacement_key(setup, monkeypatch):
    cfg, _ = setup
    cfg.llm.decision_routing.backend = "jev"

    async def rejected(*args, **kwargs):
        assert kwargs["api_key"] == "test-only-key"
        monkeypatch.setenv("TYPESAFE_API_KEY", "replacement-test-key")
        raise jev.JevUnavailable("http_401")

    monkeypatch.setattr(jev, "classify", rejected)
    assert (await classify_goal("Explain sockets")).available
    assert decision_router.accelerator_status() == "unverified"


async def test_partial_settings_preserve_other_fields_on_disk_and_in_memory(setup, monkeypatch, tmp_path):
    from ruamel.yaml import YAML

    from rune.api.handlers.config import ConfigPatchRequest, patch_config

    cfg, _ = setup
    cfg.llm.decision_routing.backend = "jev"
    cfg.llm.decision_routing.timeout_ms = 3200
    path = tmp_path / "config.yaml"
    path.write_text('llm:\n  decisionRouting:\n    backend: jev\n    timeoutMs: 3200\n')
    monkeypatch.setattr("rune.config.writer.config_file_path", lambda: path)
    await patch_config(ConfigPatchRequest(decisionRouting={"backend": "connected"}))
    assert cfg.llm.decision_routing.timeout_ms == 3200
    await patch_config(ConfigPatchRequest(decisionRouting={"timeoutMs": 2500}))
    assert cfg.llm.decision_routing.backend == "connected"
    persisted = RuneConfig.model_validate(YAML().load(path)).llm.decision_routing
    assert persisted == cfg.llm.decision_routing
    before = path.read_bytes()
    assert not (await patch_config(ConfigPatchRequest(decisionRouting={}))).updated
    assert path.read_bytes() == before


async def test_mixed_config_write_failure_applies_nothing(setup, monkeypatch):
    from fastapi import HTTPException

    from rune.api.handlers.config import ConfigPatchRequest, patch_config

    cfg, _ = setup
    before = cfg.model_dump()
    monkeypatch.setattr("rune.config.save_config_values", lambda updates: None)
    with pytest.raises(HTTPException):
        await patch_config(ConfigPatchRequest(decisionRouting={"backend": "jev"},
                                             activeModel={"provider": "anthropic", "model": "claude-opus-5"}))
    assert cfg.model_dump() == before
