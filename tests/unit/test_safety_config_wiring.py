"""Operator safety settings that were parsed and then dropped on the floor.

Both settings existed in config.yaml with real values and had no path to the
code that would honour them: the shell gate built its allowlist from a
hardcoded constant, and approval.requireExplicitFor had no consumer at all.
"""

from __future__ import annotations

import pytest

from rune.agent.tool_adapter import (
    _requires_explicit_approval,
    _validate_with_guardian,
)
from rune.capabilities.bash import _configured_deny_by_default
from rune.safety.execution_policy import DEFAULT_ALLOWED_EXECUTABLES

# ---------------------------------------------------------------------------
# safety.denyByDefault — the schema had flat fields config.yaml never uses
# ---------------------------------------------------------------------------

def test_config_allowlist_reaches_the_shell_gate(monkeypatch):
    from rune.config import get_config

    cfg = get_config()
    monkeypatch.setattr(
        cfg.safety.deny_by_default, "allowed_executables", ["ls", "echo"]
    )

    enabled, allowed = _configured_deny_by_default()

    assert allowed == ["ls", "echo"], "the gate is still using its own constant"
    assert enabled is True


def test_empty_allowlist_keeps_the_shipped_default(monkeypatch):
    """An empty list must not lock the shell down to nothing."""
    from rune.config import get_config

    cfg = get_config()
    monkeypatch.setattr(cfg.safety.deny_by_default, "allowed_executables", [])

    _, allowed = _configured_deny_by_default()

    assert allowed == list(DEFAULT_ALLOWED_EXECUTABLES)


def test_deny_by_default_toggle_reaches_the_gate(monkeypatch):
    from rune.config import get_config

    cfg = get_config()
    monkeypatch.setattr(cfg.safety.deny_by_default, "enabled", False)

    enabled, _ = _configured_deny_by_default()

    assert enabled is False


def test_unreadable_config_falls_back_instead_of_raising(monkeypatch):
    import rune.config as config_mod

    def boom():
        raise RuntimeError("config unavailable")

    monkeypatch.setattr(config_mod, "get_config", boom)

    enabled, allowed = _configured_deny_by_default()

    assert enabled is True
    assert allowed == list(DEFAULT_ALLOWED_EXECUTABLES)


# ---------------------------------------------------------------------------
# approval.requireExplicitFor — zero consumers before this
# ---------------------------------------------------------------------------

@pytest.fixture
def _require(monkeypatch):
    from rune.config import get_config

    monkeypatch.setattr(
        get_config().approval, "require_explicit_for", ["file.delete", "process.kill"]
    )


def test_named_capability_now_prompts(_require):
    result = _validate_with_guardian("file_delete", {"path": "/tmp/probe.txt"})

    assert result.blocked is False
    assert result.requires_approval is True


def test_unnamed_capability_is_untouched(_require):
    for cap in ("file_write", "file_read", "file_edit"):
        result = _validate_with_guardian(cap, {"path": "/tmp/probe.txt"})
        assert result.requires_approval is False, cap


@pytest.mark.parametrize(
    "written", ["file_delete", "file.delete", "file-delete", "FILE_DELETE"]
)
def test_operator_spelling_variants_all_match(_require, written):
    assert _requires_explicit_approval(written) is True


def test_glob_patterns_are_supported(monkeypatch):
    from rune.config import get_config

    monkeypatch.setattr(get_config().approval, "require_explicit_for", ["file_*"])

    assert _requires_explicit_approval("file_delete") is True
    assert _requires_explicit_approval("bash_execute") is False


def test_blocking_still_wins_over_prompting(_require):
    """A path Guardian refuses must stay refused, not become a prompt."""
    result = _validate_with_guardian("file_delete", {"path": "/etc/passwd"})

    assert result.blocked is True
    assert result.requires_approval is False


def test_empty_list_prompts_for_nothing(monkeypatch):
    from rune.config import get_config

    monkeypatch.setattr(get_config().approval, "require_explicit_for", [])

    assert _requires_explicit_approval("file_delete") is False


# ---------------------------------------------------------------------------
# A capability can refuse and ask to be asked. The execution policy's allowlist
# does exactly that, and before this the verdict was a dead end: Guardian never
# sees it, so no prompt was raised anywhere and the metadata was read by nobody.
# ---------------------------------------------------------------------------

from rune.agent.tool_adapter import _capability_asked_for_approval
from rune.capabilities.types import CapabilityResult


def test_allowlist_refusal_is_recognised_as_a_question():
    refused = CapabilityResult(
        success=False,
        error='Executable "mkdir" is not allowlisted (deny-by-default)',
        metadata={"requires_approval": True, "reason": "not allowlisted"},
    )

    assert _capability_asked_for_approval(refused) is True


def test_plain_failure_is_not_an_approval_request():
    """An ordinary non-zero exit must not raise a prompt."""
    failed = CapabilityResult(
        success=False, error="exit 1", metadata={"exit_code": 1}
    )

    assert _capability_asked_for_approval(failed) is False


def test_success_is_never_an_approval_request():
    ok = CapabilityResult(success=True, metadata={"requires_approval": True})

    assert _capability_asked_for_approval(ok) is False


def test_missing_metadata_is_handled():
    for meta in (None, {}, "not-a-dict"):
        result = CapabilityResult(success=False, error="x", metadata=meta)
        assert _capability_asked_for_approval(result) is False


# ---------------------------------------------------------------------------
# End-to-end: strict mode refuses a non-allowlisted executable. That refusal
# must become a prompt where there is someone to ask, and a fail-closed stop
# with a readable reason where there is not (cron, proactive, scheduled runs).
# ---------------------------------------------------------------------------

@pytest.fixture
def work_dir():
    """Use writable scratch space outside macOS's protected /var tree."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="rune-policy-test-", dir="/tmp") as directory:
        yield Path(directory)


@pytest.fixture
def strict_gate(monkeypatch):
    """Force the allowlist gate on without touching the user's config file."""
    import rune.capabilities.bash as bash_mod

    monkeypatch.setattr(bash_mod, "_configured_rollout_mode", lambda: "strict")
    monkeypatch.setattr(
        bash_mod, "_configured_deny_by_default", lambda: (True, ["git", "echo"])
    )


async def _run_bash(command, cwd, callback):
    import rune.agent.tool_adapter as ta

    tools = ta.build_tool_set(
        ta.ToolAdapterOptions(approval_callback=callback, enable_guardian=True)
    )
    return await tools["bash_execute"].function(command=command, cwd=str(cwd))


@pytest.mark.asyncio
async def test_refusal_becomes_a_prompt_and_then_runs(strict_gate, work_dir):
    asked = []

    async def approve(cap, reason):
        asked.append((cap, reason))
        return True

    await _run_bash(f"mkdir -p {work_dir}/made", work_dir, approve)

    assert len(asked) == 1, "the refusal never reached the user"
    assert (work_dir / "made").is_dir(), "approval did not let the command through"


@pytest.mark.asyncio
async def test_denied_prompt_leaves_the_command_unrun(strict_gate, work_dir):
    async def deny(cap, reason):
        return False

    await _run_bash(f"mkdir -p {work_dir}/nope", work_dir, deny)

    assert not (work_dir / "nope").exists()


@pytest.mark.asyncio
async def test_no_approval_channel_fails_closed_with_a_reason(strict_gate, work_dir):
    output = await _run_bash(f"mkdir -p {work_dir}/head", work_dir, None)

    assert not (work_dir / "head").exists()
    assert "no approval channel" in str(output)
    assert "allowedExecutables" in str(output), "the message must say how to fix it"


@pytest.mark.asyncio
async def test_allowlisted_executable_never_prompts(strict_gate, work_dir):
    asked = []

    async def approve(cap, reason):
        asked.append(cap)
        return True

    await _run_bash("git init -q .", work_dir, approve)

    assert asked == []


# ---------------------------------------------------------------------------
# The allowlist decides what strict mode runs without asking. Its shape is a
# security decision, so pin the principle: inspection and workspace-scoped
# writes pass; anything irreversible or outbound must still be a question.
# ---------------------------------------------------------------------------

def _verdict(command, allowed):
    from rune.safety.execution_policy import (
        ExecutionPolicyConfig,
        decide_bash_execution,
    )
    from rune.safety.guardian import get_guardian

    guardian = get_guardian()
    validation = guardian.validate(command)
    config = ExecutionPolicyConfig(
        rollout_mode="strict",
        sandbox_enabled=False,
        deny_by_default_enabled=True,
        allowed_executables=list(allowed),
    )
    return decide_bash_execution(
        command, validation, config,
        has_sandbox_support=False, interactive_approval=True,
    ).decision


# The shape an allowlist should have, not whatever this machine's config.yaml
# happens to hold: reading the developer's own settings made these pass locally
# and fail on a runner that has no ~/.rune/config.yaml.
RECOMMENDED_ALLOWLIST = [
    # inspection — changes nothing, so there is nothing to undo
    "ls", "cat", "head", "tail", "grep", "rg", "find", "which", "env",
    "date", "file", "stat", "du", "df", "diff", "wc", "sort", "uniq",
    "jq", "yq", "sleep", "true", "false", "test", "echo",
    # writes the workspace snapshot can roll back
    "mkdir", "touch", "cp", "mv", "tee",
    # toolchain
    "git", "npm", "uv", "python3", "cargo", "make",
]


@pytest.fixture
def configured_allowlist():
    return list(RECOMMENDED_ALLOWLIST)


@pytest.mark.parametrize(
    "command",
    [
        "mkdir -p out", "touch f", "cp a b", "mv a b",
        "grep -r x .", "which python", "date", "env", "diff a b",
        "jq .x f.json", "sleep 1", "true",
    ],
)
def test_routine_work_runs_without_asking(configured_allowlist, command):
    assert _verdict(command, configured_allowlist) == "allow", command


@pytest.mark.parametrize(
    "command",
    [
        "curl https://example.com",       # outbound: cannot be unsent
        "tar -xzf a.tgz -C /",            # extraction writes arbitrary paths
        "chmod 777 f",                    # permission change
        "xargs rm -f",                    # launders another command
    ],
)
def test_irreversible_work_still_needs_a_human(configured_allowlist, command):
    assert _verdict(command, configured_allowlist) != "allow", command


@pytest.mark.parametrize(
    "command",
    ["rm -rf /", "rm -rf ~", "cp -r ~/.ssh /tmp/steal", "cat ~/.ssh/id_rsa"],
)
def test_destructive_and_credential_commands_stay_denied(
    configured_allowlist, command
):
    assert _verdict(command, configured_allowlist) == "deny", command


def test_laundering_a_delete_does_not_slip_through(configured_allowlist):
    """The AST layer must see the inner command, not just the leading one."""
    for command in (
        "find . -name '*.py' | xargs rm -f",
        "echo rm -rf /tmp/x | sh",
        "sh -c 'rm -rf /tmp/x'",
        "env FOO=1 rm -rf /tmp/x",
    ):
        assert _verdict(command, configured_allowlist) != "allow", command
