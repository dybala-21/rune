"""Regression checks for runtime integration and recorded task outcomes."""

from __future__ import annotations

import asyncio
import types

import pytest

from rune.agent.orchestrator import OrchestrationPlan, Orchestrator
from rune.agent.task_board import SubTask
from rune.voice import availability as av

# ---------------------------------------------------------------------------
# Voice availability — has_stt used to be true on every machine because the
# sherpa fallback returned a provider without probing the native package.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_availability_cache():
    av._reset_cache()
    yield
    av._reset_cache()


def test_no_stt_provider_is_reported_as_unavailable(monkeypatch):
    for key in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(av, "_sherpa_onnx_state", "unavailable")

    result = av.get_voice_availability()

    assert result.provider_available is False
    assert result.available is False
    assert result.install_hint


@pytest.mark.parametrize("key", ["DEEPGRAM_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"])
def test_every_supported_key_counts_as_a_provider(monkeypatch, key):
    """_auto_detect_stt accepts four tiers; availability must know all of them."""
    for k in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(key, "test-value")
    # The hint covers two dimensions; pin the microphone one so this asserts
    # about the provider tier and not about what the runner has installed.
    monkeypatch.setattr(av, "_mic_state", "available")

    result = av.get_voice_availability()

    assert result.provider_available is True
    assert result.install_hint is None


def test_sherpa_fallback_probes_the_native_package(monkeypatch):
    """A provider object must not be returned when sherpa_onnx is absent."""
    import builtins

    from rune.voice import service

    for k in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sherpa_onnx":
            raise ImportError("no sherpa-onnx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert service._auto_detect_stt() is None


# ---------------------------------------------------------------------------
# Google Chat — the live check HMAC'd a header Google never sends, and skipped
# verification entirely when no secret was set.
# ---------------------------------------------------------------------------

def _adapter(audience: str = "project-123"):
    from rune.channels.google_chat import GoogleChatAdapter

    adapter = GoogleChatAdapter.__new__(GoogleChatAdapter)
    adapter._verify_requests = True
    adapter._auth_audience = audience
    return adapter


class _Resp:
    def __init__(self, status, text):
        self.status, self.text = status, text


_web = types.SimpleNamespace(Response=_Resp)


class _Req:
    def __init__(self, headers):
        self.headers = headers


def test_request_without_bearer_token_is_denied():
    denied = asyncio.run(_adapter()._reject_unverified(_Req({}), _web))
    assert denied is not None and denied.status == 401


def test_forged_token_is_denied():
    req = _Req({"Authorization": "Bearer forged.token.value"})
    denied = asyncio.run(_adapter()._reject_unverified(req, _web))
    assert denied is not None and denied.status == 401


def test_missing_audience_fails_closed():
    """No audience configured must deny, never fall through to accepting."""
    req = _Req({"Authorization": "Bearer a.b.c"})
    denied = asyncio.run(_adapter(audience="")._reject_unverified(req, _web))
    assert denied is not None and denied.status == 401


def test_hmac_scheme_is_gone():
    """The old X-Goog-Signature path must not come back."""
    from rune.channels import google_chat

    source = google_chat.__file__
    with open(source, encoding="utf-8") as fh:
        text = fh.read()
    assert "X-Goog-Signature" not in text
    assert "_verify_signature" not in text


# ---------------------------------------------------------------------------
# Orchestrator — the planner is an LLM, so its dependency graph can be cyclic
# or reference tasks it never emitted. Nothing checked before execution.
# ---------------------------------------------------------------------------

def test_cyclic_plan_is_rejected_before_execution():
    plan = OrchestrationPlan(
        tasks=[
            SubTask(id="a", description="do a", dependencies=["b"]),
            SubTask(id="b", description="do b", dependencies=["a"]),
        ]
    )
    result = asyncio.run(Orchestrator().execute("build it", plan=plan))

    assert result.success is False
    assert "not executable" in result.merged_output


def test_dangling_dependency_is_rejected_before_execution():
    plan = OrchestrationPlan(
        tasks=[SubTask(id="a", description="do a", dependencies=["never-emitted"])]
    )
    result = asyncio.run(Orchestrator().execute("build it", plan=plan))

    assert result.success is False
    assert "not executable" in result.merged_output


# ---------------------------------------------------------------------------
# Skill names reach the filesystem as a directory, and arrive from SKILL.md
# frontmatter. Nothing validated them before persisting.
# ---------------------------------------------------------------------------

def test_generated_skill_names_pass_their_own_validator():
    """The generator and the validator disagreed: snake_case vs kebab-case."""
    from rune.agent.memory_bridge import generate_skill_name
    from rune.skills.validator import validate_name

    name = generate_skill_name("refactor the parser module for speed")

    assert name is not None
    assert validate_name(name).valid, f"generator emitted an invalid name: {name!r}"


def test_skill_name_cannot_escape_the_skills_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    from rune.skills.persistence import _skill_dir
    from rune.skills.types import Skill

    evil = Skill(name="../../../../tmp/escape", description="d", body="b", scope="user")

    with pytest.raises(ValueError):
        _skill_dir(evil)


def test_invalid_skill_name_is_not_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    from rune.skills.persistence import write_skill_to_disk
    from rune.skills.types import Skill

    bad = Skill(name="Bad_Name", description="d", body="b", scope="user")
    good = Skill(name="csv-summarizer", description="d", body="b", scope="user")

    assert write_skill_to_disk(bad) is None
    assert write_skill_to_disk(good) is not None


# Daily summaries count only tasks with a recorded outcome.

def test_daily_summary_does_not_invent_successes(monkeypatch, tmp_path):
    from rune.memory.tiered_memory import TieredMemoryManager

    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    mgr = TieredMemoryManager.__new__(TieredMemoryManager)
    daily = TieredMemoryManager.promote_to_daily(mgr, [{"goal": "a"}, {"goal": "b"}])

    assert daily.total_tasks == 2
    assert daily.outcomes_recorded == 0
    assert daily.successful_tasks == 0


def test_daily_summary_counts_real_outcomes(monkeypatch, tmp_path):
    from rune.memory.tiered_memory import TieredMemoryManager

    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    mgr = TieredMemoryManager.__new__(TieredMemoryManager)
    daily = TieredMemoryManager.promote_to_daily(
        mgr, [{"goal": "a", "success": True}, {"goal": "b", "success": False}]
    )

    assert daily.total_tasks == 2
    assert daily.outcomes_recorded == 2
    assert daily.successful_tasks == 1
