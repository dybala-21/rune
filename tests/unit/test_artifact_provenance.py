"""An artifact the request treated as existing cannot be invented.

The failure: told to fix the bug described in BUGREPORT.md when no such
file exists, the agent wrote BUGREPORT.md itself, invented a bug, edited
unrelated source and reported success. The decision here is made from the
record of which reads succeeded — no model judgement, no phrase matching,
so it behaves the same for every provider and every language.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import rune.agent.litellm_adapter as la
from rune.agent.litellm_adapter import StreamResult
from rune.agent.provenance import ArtifactLedger, referenced_paths


class TestReferencedPaths:
    def test_picks_out_file_names_in_any_language(self):
        assert "BUGREPORT.md" in referenced_paths(
            "BUGREPORT.md에 적힌 버그 고쳐줘")
        assert "orders.csv" in referenced_paths(
            "please summarise orders.csv by category")
        assert referenced_paths("config/settings.yaml 확인해") == {
            "settings.yaml"}

    def test_ignores_prose_and_domains(self):
        assert referenced_paths("just clean up the working directory") == set()
        assert referenced_paths("see example.com for details") == set()

    def test_handles_empty_input(self):
        assert referenced_paths("") == set()
        assert referenced_paths(None) == set()


class TestLedger:
    def _ledger(self):
        return ArtifactLedger.for_request("BUGREPORT.md 에 적힌 버그 고쳐줘")

    def test_failed_read_makes_the_path_a_phantom(self):
        led = self._ledger()
        led.record_read("BUGREPORT.md", ok=False)
        assert led.is_phantom("BUGREPORT.md")
        assert led.unresolved() == ["BUGREPORT.md"]

    def test_a_successful_read_clears_it(self):
        led = self._ledger()
        led.record_read("BUGREPORT.md", ok=False)
        led.record_read("./BUGREPORT.md", ok=True)
        assert not led.is_phantom("BUGREPORT.md")
        assert led.unresolved() == []

    def test_a_successful_read_elsewhere_does_not_clear_a_missing_path(self):
        # A listing of some other directory that happens to mention the
        # same file name proves nothing about this one.
        led = self._ledger()
        led.record_read("BUGREPORT.md", ok=False)
        assert led.is_phantom("BUGREPORT.md")

    def test_files_the_request_never_mentioned_are_free(self):
        led = self._ledger()
        led.record_read("notes.md", ok=False)
        assert not led.is_phantom("notes.md")
        assert led.unresolved() == []

    def test_never_looked_for_is_not_unresolved(self):
        # A named file nobody searched for is an output to create, not a
        # missing input — creating it is the job, so it must stay allowed.
        led = self._ledger()
        assert led.unresolved() == []
        assert not led.is_phantom("BUGREPORT.md")

    def test_a_classified_input_is_a_phantom_without_any_search(self):
        # The run that writes immediately, never looking, is the one the
        # search-based rule misses.
        led = self._ledger()
        led.roles["BUGREPORT.md"] = "input"
        assert led.is_phantom("BUGREPORT.md")

    def test_a_classified_output_is_never_blocked(self):
        led = ArtifactLedger.for_request("정리해서 cleaned.csv 만들어줘")
        led.roles["cleaned.csv"] = "output"
        led.record_read("cleaned.csv", ok=False)  # agent checked first
        assert not led.is_phantom("cleaned.csv")
        assert led.unresolved() == []

    def test_without_a_classification_the_search_rule_still_applies(self):
        led = self._ledger()
        assert not led.is_phantom("BUGREPORT.md")
        led.record_read("BUGREPORT.md", ok=False)
        assert led.is_phantom("BUGREPORT.md")

    def test_a_shell_search_counts_as_looking(self):
        # Agents locate files with find/ls/grep as readily as with a read
        # tool; the ledger must not care which.
        led = self._ledger()
        led.record_lookup('{"command": "find . -name BUGREPORT.md"}')
        assert led.is_phantom("BUGREPORT.md")
        assert led.unresolved() == ["BUGREPORT.md"]


@pytest.fixture
def no_classifier(monkeypatch):
    """Default for the adapter tests: classification unavailable, so the
    search-based fallback is what gets exercised."""
    import rune.agent.provenance as prov

    async def _none(request, names, model, provider):
        return {}
    monkeypatch.setattr(prov, "classify_roles", _none)


def _delta(*, content=None, tool_calls=None, finish_reason=None):
    tcs = None
    if tool_calls:
        tcs = [
            SimpleNamespace(
                index=t["index"], id=t.get("id", f"tc{t['index']}"),
                function=SimpleNamespace(name=t.get("name"),
                                         arguments=t.get("arguments")),
            ) for t in tool_calls
        ]
    return SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=content, tool_calls=tcs),
            finish_reason=finish_reason)],
        usage=None)


async def _astream(chunks):
    for c in chunks:
        yield c


def _fake_completion(streams):
    it = iter(streams)

    async def fake(**kwargs):
        return _astream(next(it))
    return fake


def _result(tmp_path, reads, writes):
    async def file_read(**kw):
        reads.append(kw.get("path"))
        return f"Error: File not found: {kw.get('path')}"

    async def file_write(**kw):
        writes.append(kw.get("path"))
        return f"Written to {kw.get('path')}"

    return StreamResult(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user",
                   "content": "BUGREPORT.md 에 적힌 버그 고쳐줘"}],
        tool_schemas=[{"function": {"name": "file_read"}},
                      {"function": {"name": "file_write"}}],
        tool_lookup={"file_read": file_read, "file_write": file_write},
        max_tokens=4096, temperature=0.0,
        request_tokens_limit=200000, response_tokens_limit=4096,
        max_tool_rounds=8,
    )


_READ_TURN = [
    _delta(tool_calls=[{"index": 0, "name": "file_read",
                        "arguments": '{"path": "BUGREPORT.md"}'}]),
    _delta(finish_reason="tool_calls"),
]
_WRITE_TURN = [
    _delta(tool_calls=[{"index": 0, "name": "file_write",
                        "arguments": '{"path": "BUGREPORT.md", '
                                     '"content": "# Bug\\nInvented."}'}]),
    _delta(finish_reason="tool_calls"),
]
_FINAL = [_delta(content="다 고쳤습니다."), _delta(finish_reason="stop")]


@pytest.mark.asyncio
async def test_writing_the_missing_artifact_is_blocked(monkeypatch, tmp_path, no_classifier):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUNE_ARTIFACT_PROVENANCE", raising=False)
    reads, writes = [], []
    res = _result(tmp_path, reads, writes)
    monkeypatch.setattr(la.litellm, "acompletion",
                        _fake_completion([_READ_TURN, _WRITE_TURN, _FINAL, _FINAL]))

    async for _ in res.stream_text():
        pass

    assert reads == ["BUGREPORT.md"]
    assert writes == []  # the fabricating write never reached the tool
    blocked = [m for m in res.all_messages()
               if m.get("role") == "tool" and "BLOCKED" in str(m.get("content"))]
    assert blocked, "agent was not told why the write was refused"


@pytest.mark.asyncio
async def test_run_cannot_end_silently_with_a_missing_input(monkeypatch, tmp_path, no_classifier):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUNE_ARTIFACT_PROVENANCE", raising=False)
    res = _result(tmp_path, [], [])
    monkeypatch.setattr(la.litellm, "acompletion",
                        _fake_completion([_READ_TURN, _FINAL, _FINAL]))

    async for _ in res.stream_text():
        pass

    notes = [m for m in res.all_messages()
             if m.get("role") == "user" and "BUGREPORT.md" in str(m.get("content"))
             and "missing" in str(m.get("content"))]
    assert len(notes) == 1, "expected exactly one bounded reminder"


@pytest.mark.asyncio
async def test_missing_read_is_judged_by_the_filesystem_not_the_wording(
    monkeypatch, tmp_path, no_classifier
):
    """A read tool that reports a miss in its own words — no "Error" prefix,
    no marker the adapter knows — must still count as "not found". Earlier
    this was inferred from the result string, so a differently-phrased tool
    made the file look present and the fabricating write went through."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUNE_ARTIFACT_PROVENANCE", raising=False)
    writes = []

    async def file_read(**kw):
        return "success=False detail='no such path'"

    async def file_write(**kw):
        writes.append(kw.get("path"))
        return "ok"

    res = StreamResult(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "BUGREPORT.md 에 적힌 버그 고쳐줘"}],
        tool_schemas=[{"function": {"name": "file_read"}},
                      {"function": {"name": "file_write"}}],
        tool_lookup={"file_read": file_read, "file_write": file_write},
        max_tokens=4096, temperature=0.0,
        request_tokens_limit=200000, response_tokens_limit=4096,
        max_tool_rounds=8,
    )
    monkeypatch.setattr(la.litellm, "acompletion",
                        _fake_completion([_READ_TURN, _WRITE_TURN, _FINAL, _FINAL]))

    async for _ in res.stream_text():
        pass

    assert writes == []


@pytest.mark.asyncio
async def test_shell_route_around_the_refusal_is_undone(monkeypatch, tmp_path, no_classifier):
    """Blocking the write tool closes one door; a shell redirect opens
    another. What gets checked is the outcome, not the command text: a
    path we refused that exists afterwards was authored, not found."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUNE_ARTIFACT_PROVENANCE", raising=False)

    async def file_read(**kw):
        return "not found"

    async def file_write(**kw):
        (tmp_path / kw["path"]).write_text(kw.get("content", ""))
        return "written"

    async def _missing_read(**kw):
        return "not found"

    async def bash_execute(**kw):
        (tmp_path / "BUGREPORT.md").write_text("# invented via shell\n")
        return "[cmd: cat > BUGREPORT.md] [exit: 0]"

    res = StreamResult(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "BUGREPORT.md 에 적힌 버그 고쳐줘"}],
        tool_schemas=[{"function": {"name": n}}
                      for n in ("file_read", "file_write", "bash_execute")],
        tool_lookup={"file_read": file_read, "file_write": file_write,
                     "bash_execute": bash_execute},
        max_tokens=4096, temperature=0.0,
        request_tokens_limit=200000, response_tokens_limit=4096,
        max_tool_rounds=8,
    )
    shell_turn = [
        _delta(tool_calls=[{"index": 0, "name": "bash_execute",
                            "arguments": '{"command": "cat > BUGREPORT.md"}'}]),
        _delta(finish_reason="tool_calls"),
    ]
    monkeypatch.setattr(la.litellm, "acompletion", _fake_completion(
        [_READ_TURN, _WRITE_TURN, shell_turn, _FINAL, _FINAL, _FINAL]))

    async for _ in res.stream_text():
        pass

    assert not (tmp_path / "BUGREPORT.md").exists(), "fabricated file survived"
    reverted = [m for m in res.all_messages()
                if "REVERTED" in str(m.get("content"))]
    assert reverted, "agent was not told the file was removed again"


@pytest.mark.asyncio
async def test_opt_out_restores_previous_behaviour(monkeypatch, tmp_path, no_classifier):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUNE_ARTIFACT_PROVENANCE", "0")
    reads, writes = [], []
    res = _result(tmp_path, reads, writes)
    monkeypatch.setattr(la.litellm, "acompletion",
                        _fake_completion([_READ_TURN, _WRITE_TURN, _FINAL]))

    async for _ in res.stream_text():
        pass

    assert writes == ["BUGREPORT.md"]


@pytest.mark.asyncio
async def test_classified_input_blocks_a_write_with_no_search_at_all(
    monkeypatch, tmp_path
):
    """The run that never looks. Only the request itself says whether the
    file was supposed to be there, so a model settles that once and the
    write is refused without any prior read."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUNE_ARTIFACT_PROVENANCE", raising=False)
    import rune.agent.provenance as prov

    seen = {}

    async def _roles(request, names, model, provider):
        seen["names"] = list(names)
        return {"BUGREPORT.md": "input"}
    monkeypatch.setattr(prov, "classify_roles", _roles)

    reads, writes = [], []
    res = _result(tmp_path, reads, writes)
    monkeypatch.setattr(la.litellm, "acompletion",
                        _fake_completion([_WRITE_TURN, _FINAL, _FINAL]))

    async for _ in res.stream_text():
        pass

    assert seen["names"] == ["BUGREPORT.md"]
    assert reads == []          # nothing was ever looked up
    assert writes == []         # and the write was still refused


@pytest.mark.asyncio
async def test_classified_output_is_created_normally(monkeypatch, tmp_path):
    """The mirror case that must not regress: a file the request asks to be
    produced gets produced, even though it does not exist yet."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUNE_ARTIFACT_PROVENANCE", raising=False)
    import rune.agent.provenance as prov

    async def _roles(request, names, model, provider):
        return {"summary.md": "output"}
    monkeypatch.setattr(prov, "classify_roles", _roles)

    writes = []

    async def file_write(**kw):
        writes.append(kw.get("path"))
        return "written"

    res = StreamResult(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "정리해서 summary.md 만들어줘"}],
        tool_schemas=[{"function": {"name": "file_write"}}],
        tool_lookup={"file_write": file_write},
        max_tokens=4096, temperature=0.0,
        request_tokens_limit=200000, response_tokens_limit=4096,
        max_tool_rounds=8,
    )
    turn = [
        _delta(tool_calls=[{"index": 0, "name": "file_write",
                            "arguments": '{"path": "summary.md", '
                                         '"content": "# summary"}'}]),
        _delta(finish_reason="tool_calls"),
    ]
    monkeypatch.setattr(la.litellm, "acompletion",
                        _fake_completion([turn, _FINAL, _FINAL]))

    async for _ in res.stream_text():
        pass

    assert writes == ["summary.md"]


@pytest.mark.asyncio
async def test_a_missing_input_written_by_shell_is_undone(monkeypatch, tmp_path):
    """No file_write call happens at all here — the shell authors the file
    directly. The run had already looked for it and found nothing, so its
    appearance mid-run means it was written, whichever tool did it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUNE_ARTIFACT_PROVENANCE", raising=False)
    import rune.agent.provenance as prov

    async def _roles(request, names, model, provider):
        return {"BUGREPORT.md": "input"}
    monkeypatch.setattr(prov, "classify_roles", _roles)

    async def _missing_read(**kw):
        return "not found"

    async def bash_execute(**kw):
        (tmp_path / "BUGREPORT.md").write_text("# invented\n")
        return "[cmd: cat > BUGREPORT.md] [exit: 0]"

    async def file_write(**kw):
        (tmp_path / kw["path"]).write_text(kw.get("content", ""))
        return "written"

    res = StreamResult(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "BUGREPORT.md 에 적힌 버그 고쳐줘"}],
        tool_schemas=[{"function": {"name": "bash_execute"}},
                      {"function": {"name": "file_write"}},
                      {"function": {"name": "file_read"}}],
        tool_lookup={"bash_execute": bash_execute, "file_write": file_write,
                     "file_read": _missing_read},
        max_tokens=4096, temperature=0.0,
        request_tokens_limit=200000, response_tokens_limit=4096,
        max_tool_rounds=8,
    )
    shell_turn = [
        _delta(tool_calls=[{"index": 0, "name": "bash_execute",
                            "arguments": '{"command": "cat > BUGREPORT.md"}'}]),
        _delta(finish_reason="tool_calls"),
    ]
    # the failed read first: that is what establishes the file was absent
    monkeypatch.setattr(la.litellm, "acompletion",
                        _fake_completion([_READ_TURN, shell_turn, _FINAL, _FINAL]))

    async for _ in res.stream_text():
        pass

    assert not (tmp_path / "BUGREPORT.md").exists()


@pytest.mark.asyncio
async def test_an_input_read_via_shell_is_never_removed(monkeypatch, tmp_path):
    """The regression this exists to prevent: the agent reads a policy file
    with a shell command, so it never reaches the read ledger, and the file
    gets classified as an input. Removing it on that basis destroys real
    user data — exactly the failure the guards are for."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUNE_ARTIFACT_PROVENANCE", raising=False)
    policy = tmp_path / "tier_policy.md"
    policy.write_text("# rules\n")
    import rune.agent.provenance as prov

    async def _roles(request, names, model, provider):
        return {"tier_policy.md": "input", "members.csv": "input"}
    monkeypatch.setattr(prov, "classify_roles", _roles)

    async def bash_execute(**kw):
        return "[cmd: cat tier_policy.md] [exit: 0]\n# rules"

    res = StreamResult(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user",
                   "content": "tier_policy.md 기준으로 members.csv 다시 계산해줘"}],
        tool_schemas=[{"function": {"name": "bash_execute"}}],
        tool_lookup={"bash_execute": bash_execute},
        max_tokens=4096, temperature=0.0,
        request_tokens_limit=200000, response_tokens_limit=4096,
        max_tool_rounds=8,
    )
    turn = [
        _delta(tool_calls=[{"index": 0, "name": "bash_execute",
                            "arguments": '{"command": "cat tier_policy.md"}'}]),
        _delta(finish_reason="tool_calls"),
    ]
    monkeypatch.setattr(la.litellm, "acompletion",
                        _fake_completion([turn, _FINAL, _FINAL]))

    async for _ in res.stream_text():
        pass

    assert policy.exists(), "an existing input file was removed"
    assert policy.read_text() == "# rules\n"

