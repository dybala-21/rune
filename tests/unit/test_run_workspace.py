from unittest.mock import Mock

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.loop import NativeAgentLoop
from rune.types import CompletionTrace


async def test_prompt_and_verifier_follow_each_run_workspace(monkeypatch, tmp_path):
    server = tmp_path / "server"
    server.mkdir()
    monkeypatch.chdir(server)
    monkeypatch.setenv("RUNE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("RUNE_AUTO_SKILL", "0")
    monkeypatch.delenv("RUNE_AUTO_VERIFY_CMD", raising=False)
    repo_map = Mock(return_value="")
    monkeypatch.setattr("rune.intelligence.repo_map.build_repo_map_sync", repo_map)
    loop = NativeAgentLoop()
    observed = []

    async def execute(**kwargs):
        observed.append((kwargs["system_prompt"], await loop._auto_verify()))
        return CompletionTrace(reason="completed")

    monkeypatch.setattr(loop, "_execute_loop", execute)
    monkeypatch.setattr(loop, "_select_tools", lambda _: [])
    classification = ClassificationResult(goal_type="code_modify", confidence=1, tier=1)
    for name in ("office", "coding"):
        workspace = tmp_path / name
        workspace.mkdir()
        (workspace / "marker.txt").write_text(name)
        (workspace / "test_workspace.py").write_text(
            "from pathlib import Path\n"
            "def test_working_directory():\n"
            f"    assert Path('marker.txt').read_text() == {name!r}\n"
        )
        await loop.run("Check this project", context={"workspace_root": str(workspace)},
                       classification=classification)
        prompt, (verdict, _) = observed[-1]
        assert f"Current working directory: {workspace}" in prompt
        assert verdict == "pass"
        assert repo_map.call_args.args[0] == str(workspace)
        assert str(server) not in prompt

    # Reusing a loop without a pinned workspace must not retain the prior project.
    await loop.run("Check this project", classification=classification)
    prompt, (verdict, _) = observed[-1]
    assert f"Current working directory: {server}" in prompt
    assert verdict == "skip"
    assert repo_map.call_args.args[0] == str(server)
