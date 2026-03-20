"""Tests for rune.agent.workspace_guard — path scope enforcement."""


from rune.agent.workspace_guard import (
    ScopeGuardOptions,
    extract_explicit_path_allowlist,
    extract_intentional_path_allowlist,
    find_bash_scope_violations,
    find_write_path_scope_violation,
)

WORKSPACE_ROOT = "/Users/test/workspace/rune"


# ---------------------------------------------------------------------------
# extract_explicit_path_allowlist
# ---------------------------------------------------------------------------

class TestExtractExplicitPathAllowlist:
    def test_extracts_absolute_paths(self):
        goal = "check /Users/test/test-workspace/AetherArc"
        allowlist = extract_explicit_path_allowlist(goal, WORKSPACE_ROOT)
        assert "/Users/test/test-workspace/AetherArc" in allowlist

    def test_extracts_relative_paths(self):
        goal = "switch to ../test-workspace/AetherArc now"
        allowlist = extract_explicit_path_allowlist(goal, WORKSPACE_ROOT)
        assert "/Users/test/workspace/test-workspace/AetherArc" in allowlist

    def test_does_not_treat_urls_as_paths(self):
        goal = "docs at https://example.com/docs/index.html"
        allowlist = extract_explicit_path_allowlist(goal, WORKSPACE_ROOT)
        assert len(allowlist) == 0

    def test_extracts_tilde_paths(self):
        goal = "check ~/workspace/project"
        allowlist = extract_explicit_path_allowlist(goal, WORKSPACE_ROOT)
        assert len(allowlist) >= 1

    def test_empty_goal_returns_empty(self):
        assert extract_explicit_path_allowlist("", WORKSPACE_ROOT) == []


# ---------------------------------------------------------------------------
# extract_intentional_path_allowlist
# ---------------------------------------------------------------------------

class TestExtractIntentionalPathAllowlist:
    def test_extracts_path_with_workspace_context(self):
        goal = "continue at /Users/test/test-workspace/AetherArc"
        allowlist = extract_intentional_path_allowlist(goal, WORKSPACE_ROOT)
        assert "/Users/test/test-workspace/AetherArc" in allowlist

    def test_ignores_paths_without_intent_signal(self):
        goal = "note: /Users/test/test-workspace/AetherArc is nice"
        allowlist = extract_intentional_path_allowlist(goal, WORKSPACE_ROOT)
        assert len(allowlist) == 0

    def test_ignores_paths_in_code_fences(self):
        goal = "```\ncd /Users/test/other/project\n```"
        allowlist = extract_intentional_path_allowlist(goal, WORKSPACE_ROOT)
        assert len(allowlist) == 0


# ---------------------------------------------------------------------------
# find_write_path_scope_violation
# ---------------------------------------------------------------------------

class TestFindWritePathScopeViolation:
    def test_allows_write_inside_workspace(self):
        violation = find_write_path_scope_violation(
            "src/agent/loop.py",
            ScopeGuardOptions(workspace_root=WORKSPACE_ROOT),
        )
        assert violation is None

    def test_blocks_write_outside_workspace(self):
        violation = find_write_path_scope_violation(
            "/Users/test/test-workspace/AetherArc/src/main.rs",
            ScopeGuardOptions(workspace_root=WORKSPACE_ROOT),
        )
        assert violation is not None
        assert violation.source == "file.path"

    def test_allows_outside_write_with_explicit_allowlist(self):
        violation = find_write_path_scope_violation(
            "/Users/test/test-workspace/AetherArc/src/main.rs",
            ScopeGuardOptions(
                workspace_root=WORKSPACE_ROOT,
                explicit_allowlist=["/Users/test/test-workspace/AetherArc"],
            ),
        )
        assert violation is None

    def test_none_path_returns_none(self):
        violation = find_write_path_scope_violation(
            None,
            ScopeGuardOptions(workspace_root=WORKSPACE_ROOT),
        )
        assert violation is None

    def test_empty_path_returns_none(self):
        violation = find_write_path_scope_violation(
            "",
            ScopeGuardOptions(workspace_root=WORKSPACE_ROOT),
        )
        assert violation is None


# ---------------------------------------------------------------------------
# find_bash_scope_violations
# ---------------------------------------------------------------------------

class TestFindBashScopeViolations:
    def test_blocks_cwd_outside_workspace(self):
        violations = find_bash_scope_violations(
            cwd="/Users/test/test-workspace/AetherArc",
            command="cargo test -q",
            workspace_root=WORKSPACE_ROOT,
        )
        assert len(violations) == 1
        assert violations[0].source == "bash.cwd"

    def test_blocks_cd_outside_workspace(self):
        violations = find_bash_scope_violations(
            command="cd /Users/test/test-workspace/AetherArc && cargo test -q",
            workspace_root=WORKSPACE_ROOT,
        )
        assert len(violations) == 1
        assert violations[0].source == "bash.cd"

    def test_allows_cd_outside_with_explicit_allowlist(self):
        violations = find_bash_scope_violations(
            command="cd /Users/test/test-workspace/AetherArc && cargo test -q",
            workspace_root=WORKSPACE_ROOT,
            explicit_allowlist=["/Users/test/test-workspace/AetherArc"],
        )
        assert len(violations) == 0

    def test_allows_relative_cd_inside_workspace(self):
        violations = find_bash_scope_violations(
            command="cd src && npm test",
            workspace_root=WORKSPACE_ROOT,
        )
        assert len(violations) == 0

    def test_blocks_dash_c_path_outside_workspace(self):
        violations = find_bash_scope_violations(
            command="git -C /Users/test/test-workspace/AetherArc status",
            workspace_root=WORKSPACE_ROOT,
        )
        assert len(violations) == 1
        assert violations[0].source == "bash.arg"

    def test_blocks_manifest_path_outside_workspace(self):
        violations = find_bash_scope_violations(
            command="cargo test --manifest-path /Users/test/test-workspace/AetherArc/Cargo.toml",
            workspace_root=WORKSPACE_ROOT,
        )
        assert len(violations) == 1
        assert violations[0].source == "bash.arg"

    def test_blocks_prefix_outside_workspace(self):
        violations = find_bash_scope_violations(
            command="npm --prefix /Users/test/test-workspace/AetherArc test",
            workspace_root=WORKSPACE_ROOT,
        )
        assert len(violations) == 1
        assert violations[0].source == "bash.arg"

    def test_no_violations_for_empty_command(self):
        violations = find_bash_scope_violations(
            command="",
            workspace_root=WORKSPACE_ROOT,
        )
        assert len(violations) == 0
