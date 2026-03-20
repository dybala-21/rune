"""Tests for the tool adapter module."""

from __future__ import annotations

import math

from rune.agent.tool_adapter import (
    EXTENDED_MULTIPLIER,
    STALL_LIMITS,
    StallState,
    ToolAdapterOptions,
    _format_tool_output,
    _update_stall_state,
    get_effective_stall_limits,
    is_mcp_write_operation,
)
from rune.capabilities.output_prefixes import (
    BASH_CMD_PREFIX,
    BASH_EXIT_PREFIX,
    FILE_READ_PATH_PREFIX,
)
from rune.types import CapabilityResult


class TestStallLimits:
    def test_stall_limits_has_required_keys(self):
        assert "fileRead" in STALL_LIMITS
        assert "bash" in STALL_LIMITS
        assert "cycle" in STALL_LIMITS
        assert "web" in STALL_LIMITS
        assert "maxNudges" in STALL_LIMITS

    def test_file_read_thresholds(self):
        fr = STALL_LIMITS["fileRead"]
        assert fr["warning"] == 20
        assert fr["hardStop"] == 30
        assert fr["sameFile"] == 2

    def test_bash_thresholds(self):
        bash = STALL_LIMITS["bash"]
        assert bash["consecutiveFailures"] == 8
        assert bash["intentRepeat"] == 8

    def test_effective_limits_no_extension(self):
        limits = get_effective_stall_limits(extended=False)
        assert limits["maxNudges"] == STALL_LIMITS["maxNudges"]
        assert limits["fileRead"]["warning"] == 20

    def test_effective_limits_extended(self):
        limits = get_effective_stall_limits(extended=True)
        # Numeric values should be multiplied and ceiled
        assert limits["maxNudges"] == math.ceil(5 * EXTENDED_MULTIPLIER)
        assert limits["fileRead"]["warning"] == math.ceil(20 * EXTENDED_MULTIPLIER)
        assert limits["fileRead"]["hardStop"] == math.ceil(30 * EXTENDED_MULTIPLIER)

    def test_extended_multiplier_value(self):
        assert EXTENDED_MULTIPLIER == 1.5


class TestStallState:
    def test_default_state(self):
        state = StallState()
        assert state.bash_stalled is False
        assert state.bash_stalled_reason == ""
        assert state.file_read_exhausted is False

    def test_update_stall_bash_permission_denied(self):
        state = StallState()
        result = CapabilityResult(success=False, error="Permission denied: /etc/shadow")
        _update_stall_state(state, "bash_execute", {"command": "cat /etc/shadow"}, result, 100.0)
        assert state.bash_stalled is True
        assert state.bash_stalled_reason == "permission_denied"

    def test_update_stall_bash_command_not_found(self):
        state = StallState()
        result = CapabilityResult(success=False, error="command not found: foobar")
        _update_stall_state(state, "bash_execute", {"command": "foobar"}, result, 50.0)
        assert state.bash_stalled is True
        assert state.bash_stalled_reason == "command_not_found"

    def test_update_stall_bash_success_resets(self):
        state = StallState()
        state.bash_stalled = True
        state.bash_stalled_reason = "permission_denied"
        result = CapabilityResult(success=True, output="ok")
        _update_stall_state(state, "bash_execute", {"command": "ls"}, result, 20.0)
        assert state.bash_stalled is False
        assert state.bash_stalled_reason == ""

    def test_update_stall_file_read_not_found(self):
        state = StallState()
        result = CapabilityResult(success=False, error="No such file or directory")
        _update_stall_state(state, "file_read", {"file_path": "/tmp/nope.py"}, result, 10.0)
        assert state.file_read_exhausted is True

    def test_update_stall_file_read_success_resets(self):
        state = StallState()
        state.file_read_exhausted = True
        result = CapabilityResult(success=True, output="content")
        _update_stall_state(state, "file_read", {"file_path": "/tmp/ok.py"}, result, 10.0)
        assert state.file_read_exhausted is False


class TestToolAdapterOptions:
    def test_defaults(self):
        opts = ToolAdapterOptions()
        assert opts.enable_guardian is True
        assert opts.sandbox_policy == "balanced"
        assert opts.cognitive_cache is None
        assert opts.allowed_tools is None

    def test_custom_allowed_tools(self):
        opts = ToolAdapterOptions(allowed_tools=["file_read", "bash_execute"])
        assert "file_read" in opts.allowed_tools
        assert len(opts.allowed_tools) == 2

    def test_budget_percent_default(self):
        opts = ToolAdapterOptions()
        assert opts.budget_percent == 0.0

    def test_budget_percent_custom(self):
        opts = ToolAdapterOptions(budget_percent=0.75)
        assert opts.budget_percent == 0.75


class TestMCPWriteDetection:
    """Feature 3: MCP write operation detection."""

    def test_non_mcp_returns_false(self):
        assert is_mcp_write_operation("bash_execute") is False
        assert is_mcp_write_operation("file_read") is False

    def test_mcp_read_returns_false(self):
        assert is_mcp_write_operation("mcp.github.list_repos") is False
        assert is_mcp_write_operation("mcp.slack.get_channels") is False

    def test_mcp_write_operations_detected(self):
        assert is_mcp_write_operation("mcp.github.create_issue") is True
        assert is_mcp_write_operation("mcp.slack.send_message") is True
        assert is_mcp_write_operation("mcp.jira.update_ticket") is True
        assert is_mcp_write_operation("mcp.db.delete_record") is True
        assert is_mcp_write_operation("mcp.api.post_data") is True
        assert is_mcp_write_operation("mcp.api.put_item") is True
        assert is_mcp_write_operation("mcp.api.patch_field") is True
        assert is_mcp_write_operation("mcp.fs.write_file") is True

    def test_mcp_too_few_parts(self):
        assert is_mcp_write_operation("mcp.") is False
        assert is_mcp_write_operation("mcp.github") is False

    def test_mcp_nested_tool_name(self):
        assert is_mcp_write_operation("mcp.service.nested.create_item") is True
        assert is_mcp_write_operation("mcp.service.nested.list_items") is False


class TestFormatToolOutput:
    """Feature 5: Output prefix wiring."""

    def test_file_read_success_has_path_prefix(self):
        result = CapabilityResult(success=True, output="file contents here")
        output = _format_tool_output("file_read", {"file_path": "/tmp/foo.py"}, result)
        assert output.startswith(f"{FILE_READ_PATH_PREFIX}/tmp/foo.py")
        assert "file contents here" in output
        assert "[END: /tmp/foo.py]" in output

    def test_bash_success_has_cmd_prefix(self):
        result = CapabilityResult(success=True, output="hello world")
        output = _format_tool_output("bash_execute", {"command": "echo hello"}, result)
        assert f"{BASH_CMD_PREFIX}echo hello{BASH_EXIT_PREFIX}0]" in output
        assert "hello world" in output

    def test_bash_error_has_cmd_prefix_with_exit_code(self):
        result = CapabilityResult(
            success=False, error="not found", metadata={"exitCode": 127}
        )
        output = _format_tool_output("bash_execute", {"command": "foobar"}, result)
        assert f"{BASH_CMD_PREFIX}foobar{BASH_EXIT_PREFIX}127]" in output
        assert "[ERROR] not found" in output

    def test_generic_success(self):
        result = CapabilityResult(success=True, output="done")
        output = _format_tool_output("web_fetch", {"url": "http://x"}, result)
        assert output == "done"

    def test_generic_error_with_suggestions(self):
        result = CapabilityResult(
            success=False, error="timeout", suggestions=["retry", "check url"]
        )
        output = _format_tool_output("web_fetch", {"url": "http://x"}, result)
        assert "[ERROR] timeout" in output
        assert "  - retry" in output
        assert "  - check url" in output

    def test_file_read_no_path_skips_prefix(self):
        result = CapabilityResult(success=True, output="content")
        output = _format_tool_output("file_read", {}, result)
        assert not output.startswith(FILE_READ_PATH_PREFIX)

    def test_success_no_output(self):
        result = CapabilityResult(success=True, output="")
        output = _format_tool_output("think", {}, result)
        assert "(success, no output)" in output

    def test_output_truncated_when_exceeds_max(self):
        """Large outputs are head+tail truncated at MAX_TOOL_OUTPUT_CHARS."""
        from rune.agent.tool_adapter import MAX_TOOL_OUTPUT_CHARS

        big_output = "x" * (MAX_TOOL_OUTPUT_CHARS + 5000)
        result = CapabilityResult(success=True, output=big_output)
        output = _format_tool_output("web_fetch", {"url": "http://x"}, result)
        assert len(output) < len(big_output)
        assert "lines omitted" in output
        assert "KB total" in output

    def test_truncation_preserves_head_and_tail(self):
        from rune.agent.tool_adapter import MAX_TOOL_OUTPUT_CHARS

        # Build output with recognizable head and tail markers
        head_marker = "HEAD_MARKER_START"
        tail_marker = "TAIL_MARKER_END"
        filler = "a" * (MAX_TOOL_OUTPUT_CHARS + 10000)
        big_output = head_marker + filler + tail_marker
        result = CapabilityResult(success=True, output=big_output)
        output = _format_tool_output("web_fetch", {"url": "http://x"}, result)
        assert head_marker in output
        assert tail_marker in output


class TestBashPreflightSnapshot:
    def test_defaults(self):
        from rune.agent.tool_adapter import BashPreflightSnapshot

        snap = BashPreflightSnapshot()
        assert snap.curl_available is False
        assert snap.curl_healthy is False
        assert snap.wget_available is False
        assert snap.node_available is False
        assert snap.go_available is False
        assert snap.reason_code == ""
        assert snap.timestamp == 0.0

    def test_custom_values(self):
        from rune.agent.tool_adapter import BashPreflightSnapshot

        snap = BashPreflightSnapshot(
            curl_available=True,
            curl_healthy=True,
            wget_available=True,
            node_available=False,
            go_available=True,
            reason_code="ok",
            timestamp=123.456,
        )
        assert snap.curl_available is True
        assert snap.curl_healthy is True
        assert snap.wget_available is True
        assert snap.go_available is True
        assert snap.reason_code == "ok"
        assert snap.timestamp == 123.456


class TestEnrichErrorMessage:
    """Tests for enrich_error_message() — error pattern classification."""

    def _enrich(self, cap_name, error, params=None, ws_markers=None):
        from rune.agent.tool_adapter import enrich_error_message
        return enrich_error_message(cap_name, error, params or {}, ws_markers)

    def test_command_not_found(self):
        result = self._enrich("bash_execute", "command not found: foobar", {"command": "foobar --help"})
        assert "[Recovery Guidance]" in result
        assert "Check if installed" in result
        assert "foobar" in result

    def test_command_not_found_exit_code_127(self):
        result = self._enrich("bash_execute", "exit code 127", {"command": "nonexistent"})
        assert "[Recovery Guidance]" in result
        assert "alternative command" in result

    def test_permission_denied(self):
        result = self._enrich("bash_execute", "EACCES: permission denied /root/secret", {"command": "cat /root/secret"})
        assert "[Recovery Guidance]" in result
        assert "file permissions" in result

    def test_timeout_long_running(self):
        result = self._enrich("bash_execute", "ETIMEDOUT", {"command": "docker build ."})
        assert "[Recovery Guidance]" in result
        assert "more time" in result
        assert "longer timeout" in result

    def test_timeout_generic(self):
        result = self._enrich("bash_execute", "timeout", {"command": "curl http://example.com"})
        assert "[Recovery Guidance]" in result
        assert "timed out" in result

    def test_connection_refused(self):
        result = self._enrich("bash_execute", "ECONNREFUSED", {"command": "curl localhost:8080"})
        assert "[Recovery Guidance]" in result
        assert "service is running" in result
        assert "port number" in result

    def test_file_not_found_enoent(self):
        result = self._enrich("file_read", "ENOENT: no such file", {"file_path": "/tmp/missing.py"})
        assert "[Recovery Guidance]" in result
        assert "file_list" in result
        assert "/tmp/missing.py" in result

    def test_file_not_found_generic(self):
        result = self._enrich("bash_execute", "not found", {"path": "/tmp/gone"})
        assert "[Recovery Guidance]" in result
        assert "/tmp/gone" in result

    def test_build_error_rust(self):
        result = self._enrich("bash_execute", "error[E0308]: mismatched types\nexpected u32, found &str", {"command": "cargo build"})
        assert "[Recovery Guidance]" in result
        assert "Compilation/build error" in result
        assert "library versions" in result

    def test_build_error_typescript(self):
        result = self._enrich("bash_execute", "TS2304: Cannot find name 'foo'", {"command": "npx tsc"})
        assert "[Recovery Guidance]" in result
        assert "Compilation/build error" in result

    def test_build_error_failed_to_compile(self):
        result = self._enrich("bash_execute", "Failed to compile", {"command": "npm run build"})
        assert "[Recovery Guidance]" in result
        assert "Compilation/build error" in result

    def test_build_error_cannot_find_module(self):
        result = self._enrich("bash_execute", "cannot find module 'foo'", {"command": "cargo build"})
        assert "[Recovery Guidance]" in result

    def test_mcp_auth_token(self):
        result = self._enrich("mcp.github.create_issue", "token expired", {})
        assert "[Recovery Guidance]" in result
        assert "re-authentication" in result
        assert "github" in result

    def test_mcp_auth_unauthorized(self):
        result = self._enrich("mcp.slack.send_message", "unauthorized", {})
        assert "[Recovery Guidance]" in result
        assert "service_connect" in result
        assert "slack" in result

    def test_eperm_sandbox(self):
        result = self._enrich("bash_execute", "operation not permitted", {"command": "rm -rf /"})
        assert "[Recovery Guidance]" in result
        assert "Sandbox EPERM" in result
        assert "security restriction" in result

    def test_eperm_keyword(self):
        result = self._enrich("bash_execute", "EPERM", {"command": "chmod 777 /etc"})
        assert "[Recovery Guidance]" in result
        assert "Sandbox EPERM" in result

    def test_intent_mismatch(self):
        result = self._enrich("bash_execute", "E_INTENT_MISMATCH: server started instead of help", {"command": "node server.js --help"})
        assert "[Recovery Guidance]" in result
        assert "Intent mismatch" in result
        assert "--help" in result

    def test_curl_tls_failure_with_markers(self):
        error = "curl: auto configuration failed - openssl.cnf missing"
        result = self._enrich("bash_execute", error, {"command": "curl https://example.com"}, ws_markers=["pyproject.toml"])
        assert "[Recovery Guidance]" in result
        assert "curl environment failure" in result
        assert "Python" in result
        assert "Do NOT retry curl" in result

    def test_curl_tls_failure_no_markers(self):
        # Note: the error must contain "auto configuration failed" and "openssl.cnf"
        # but NOT "not found" (which would match the generic file-not-found pattern first).
        error = "curl: auto configuration failed - openssl.cnf missing"
        result = self._enrich("bash_execute", error, {"command": "curl https://api.com"}, ws_markers=[])
        assert "[Recovery Guidance]" in result
        assert "Detect project type" in result

    def test_no_match_returns_original(self):
        """When no pattern matches, return the original error string."""
        result = self._enrich("web_fetch", "some random error", {})
        assert result == "some random error"
        assert "[Recovery Guidance]" not in result

    def test_file_edit_not_found(self):
        result = self._enrich("file_edit", "search string not found in file", {"file_path": "/tmp/f.py"})
        assert "[Recovery Guidance]" in result
        assert "file_read" in result

    def test_directory_not_found(self):
        result = self._enrich("file_list", "directory not found", {"path": "/tmp/gone_dir"})
        assert "[Recovery Guidance]" in result
        assert "/tmp/gone_dir" in result

    def test_api_auth_401(self):
        result = self._enrich("web_fetch", "HTTP 401 unauthorized", {})
        assert "[Recovery Guidance]" in result
        assert "alternative approach" in result

    def test_project_map_no_source_files(self):
        result = self._enrich("project_map", "no source files found in directory", {"path": "/tmp/proj"})
        assert "[Recovery Guidance]" in result
        assert "/tmp/proj" in result
        assert "file_list" in result
