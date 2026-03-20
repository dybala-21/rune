"""Tests for rune.agent.bash_parsing and managed service helpers in tool_adapter."""

from __future__ import annotations

import pytest

from rune.agent.bash_parsing import (
    is_load_test_command,
    is_service_cleanup_command,
    is_service_runtime_probe_command,
    is_service_start_command,
    is_verification_command,
    parse_command_segments,
    split_shell_segments,
    split_shell_tokens,
    strip_leading_env_assignments,
)
from rune.agent.tool_adapter import (
    apply_bash_intent_contract,
    build_managed_readiness_command,
    build_managed_smoke_command,
    build_managed_teardown_command,
    is_help_only_bash_command,
    should_auto_enable_managed_service_mode,
    strip_shell_prefixes,
)

# ===================================================================
# split_shell_segments
# ===================================================================

class TestSplitShellSegments:
    def test_simple_semicolon(self):
        assert split_shell_segments("echo a; echo b") == ["echo a", "echo b"]

    def test_pipe(self):
        assert split_shell_segments("cat file | grep foo") == ["cat file", "grep foo"]

    def test_double_ampersand(self):
        assert split_shell_segments("make && make test") == ["make", "make test"]

    def test_double_pipe(self):
        assert split_shell_segments("false || echo fallback") == ["false", "echo fallback"]

    def test_quoted_semicolon_preserved(self):
        result = split_shell_segments("echo 'a;b' && echo c")
        assert result == ["echo 'a;b'", "echo c"]

    def test_double_quoted_pipe(self):
        result = split_shell_segments('echo "a|b" | wc')
        assert result == ['echo "a|b"', "wc"]

    def test_empty_string(self):
        assert split_shell_segments("") == []

    def test_single_command(self):
        assert split_shell_segments("ls -la") == ["ls -la"]

    def test_newline_separator(self):
        assert split_shell_segments("echo a\necho b") == ["echo a", "echo b"]


# ===================================================================
# split_shell_tokens
# ===================================================================

class TestSplitShellTokens:
    def test_simple(self):
        assert split_shell_tokens("echo hello world") == ["echo", "hello", "world"]

    def test_single_quotes(self):
        assert split_shell_tokens("echo 'hello world'") == ["echo", "hello world"]

    def test_double_quotes(self):
        assert split_shell_tokens('echo "hello world"') == ["echo", "hello world"]

    def test_empty(self):
        assert split_shell_tokens("") == []

    def test_extra_whitespace(self):
        assert split_shell_tokens("  echo   hello  ") == ["echo", "hello"]


# ===================================================================
# strip_leading_env_assignments
# ===================================================================

class TestStripLeadingEnvAssignments:
    def test_single_assignment(self):
        assert strip_leading_env_assignments(["FOO=bar", "echo", "hi"]) == ["echo", "hi"]

    def test_multiple_assignments(self):
        tokens = ["A=1", "B=2", "cmd", "--flag"]
        assert strip_leading_env_assignments(tokens) == ["cmd", "--flag"]

    def test_no_assignments(self):
        assert strip_leading_env_assignments(["echo", "hi"]) == ["echo", "hi"]

    def test_flag_not_treated_as_assignment(self):
        assert strip_leading_env_assignments(["--foo=bar", "cmd"]) == ["--foo=bar", "cmd"]

    def test_empty(self):
        assert strip_leading_env_assignments([]) == []


# ===================================================================
# parse_command_segments
# ===================================================================

class TestParseCommandSegments:
    def test_simple(self):
        result = parse_command_segments("echo hello")
        assert len(result) == 1
        assert result[0]["executable"] == "echo"
        assert result[0]["tokens"] == ["echo", "hello"]

    def test_with_env_and_pipe(self):
        result = parse_command_segments("FOO=bar python app.py | grep err")
        assert len(result) == 2
        assert result[0]["executable"] == "python"
        assert result[1]["executable"] == "grep"

    def test_tokens_lowercased(self):
        result = parse_command_segments("NPM Run Build")
        assert result[0]["tokens"] == ["npm", "run", "build"]


# ===================================================================
# is_verification_command
# ===================================================================

class TestIsVerificationCommand:
    @pytest.mark.parametrize("cmd", [
        "pytest",
        "pytest tests/ -v",
        "go test ./...",
        "go build .",
        "cargo test",
        "cargo build",
        "cargo check",
        "cargo clippy",
        "npm test",
        "npm build",
        "npm run test",
        "npm run build",
        "pnpm test",
        "yarn build",
        "make test",
        "make build",
        "make check",
        "vitest",
        "jest",
        "mypy .",
        "ruff check .",
        "eslint src/",
    ])
    def test_detected(self, cmd: str):
        assert is_verification_command(cmd) is True

    @pytest.mark.parametrize("cmd", [
        "echo hello",
        "npm run dev",
        "go run main.go",
        "python app.py",
    ])
    def test_not_detected(self, cmd: str):
        assert is_verification_command(cmd) is False


# ===================================================================
# is_load_test_command
# ===================================================================

class TestIsLoadTestCommand:
    @pytest.mark.parametrize("cmd", [
        "k6 run load.js",
        "wrk http://localhost:8080",
        "hey -n 1000 http://localhost:8080",
        "vegeta attack",
        "ab -n 100 http://localhost/",
        "locust -f locustfile.py",
        "jmeter -n -t test.jmx",
        "artillery run scenario.yml",
        "gatling",
        "npm run loadtest",
        "go run ./cmd/loadtest",
    ])
    def test_detected(self, cmd: str):
        assert is_load_test_command(cmd) is True

    @pytest.mark.parametrize("cmd", [
        "pytest tests/",
        "npm test",
        "go test ./...",
    ])
    def test_not_detected(self, cmd: str):
        assert is_load_test_command(cmd) is False


# ===================================================================
# is_service_start_command
# ===================================================================

class TestIsServiceStartCommand:
    @pytest.mark.parametrize("cmd", [
        "go run main.go",
        "npm run dev",
        "npm run start",
        "npm run serve",
        "pnpm dev",
        "yarn start",
        "docker compose up",
        "docker-compose up",
        "docker run nginx",
        "python -m uvicorn app:app",
        "python3 -m http.server",
        "uvicorn app:app",
        "gunicorn app:app",
        "rails server",
        "flask run",
    ])
    def test_detected(self, cmd: str):
        assert is_service_start_command(cmd) is True

    @pytest.mark.parametrize("cmd", [
        "go run --help",
        "docker compose up -d",
        "docker-compose up --detach",
        "pytest tests/",
        "npm test",
        "echo hello",
    ])
    def test_not_detected(self, cmd: str):
        assert is_service_start_command(cmd) is False


# ===================================================================
# is_service_runtime_probe_command
# ===================================================================

class TestIsServiceRuntimeProbeCommand:
    @pytest.mark.parametrize("cmd", [
        "curl http://localhost:8080/health",
        "curl http://127.0.0.1:3000/healthz",
        "wget http://localhost:8080/readyz",
        "curl http://0.0.0.0:8080/",
        "nc -z 127.0.0.1 8080",
    ])
    def test_detected(self, cmd: str):
        assert is_service_runtime_probe_command(cmd) is True

    @pytest.mark.parametrize("cmd", [
        "curl https://example.com",
        "echo hello",
        "wget https://github.com/release.tar.gz",
    ])
    def test_not_detected(self, cmd: str):
        assert is_service_runtime_probe_command(cmd) is False


# ===================================================================
# is_service_cleanup_command
# ===================================================================

class TestIsServiceCleanupCommand:
    @pytest.mark.parametrize("cmd", [
        "kill 1234",
        "pkill node",
        "killall python",
        "docker compose down",
        "docker-compose stop",
        "npm run stop",
        "pnpm stop",
        "yarn stop",
        "lsof -ti :8080",
    ])
    def test_detected(self, cmd: str):
        assert is_service_cleanup_command(cmd) is True

    @pytest.mark.parametrize("cmd", [
        "echo hello",
        "docker compose up",
        "npm run dev",
    ])
    def test_not_detected(self, cmd: str):
        assert is_service_cleanup_command(cmd) is False


# ===================================================================
# Managed service helpers
# ===================================================================

class TestBuildManagedReadinessCommand:
    def test_contains_port(self):
        cmd = build_managed_readiness_command(8080)
        assert "8080" in cmd
        assert "nc -z" in cmd
        assert "curl" in cmd

    def test_custom_timeout(self):
        cmd = build_managed_readiness_command(3000, timeout=10)
        assert "seq 1 10" in cmd

    def test_exit_1_on_timeout(self):
        cmd = build_managed_readiness_command(8080)
        assert cmd.endswith("exit 1")


class TestBuildManagedSmokeCommand:
    def test_single_path(self):
        cmd = build_managed_smoke_command(8080, ["/health"])
        assert "8080/health" in cmd
        assert "curl -sf" in cmd

    def test_multiple_paths(self):
        cmd = build_managed_smoke_command(3000, ["/health", "/ready"])
        assert "3000/health" in cmd
        assert "3000/ready" in cmd

    def test_path_normalization(self):
        cmd = build_managed_smoke_command(8080, ["health"])
        assert "8080/health" in cmd

    def test_ends_with_exit_1(self):
        cmd = build_managed_smoke_command(8080, ["/"])
        assert cmd.strip().endswith("exit 1")


class TestBuildManagedTeardownCommand:
    def test_with_pid(self):
        cmd = build_managed_teardown_command(pid=1234)
        assert "1234" in cmd
        assert "kill -TERM" in cmd
        assert "exit 0" in cmd

    def test_without_pid(self):
        cmd = build_managed_teardown_command()
        assert cmd == "exit 0"


# ===================================================================
# should_auto_enable_managed_service_mode
# ===================================================================

class TestShouldAutoEnableManagedServiceMode:
    @pytest.mark.parametrize("cmd", [
        "go run main.go",
        "npm run dev",
        "npm run start",
        "pnpm dev",
        "yarn serve",
        "docker compose up",
        "python3 -m http.server",
        "uvicorn app:app",
        "gunicorn app:app",
        "flask run",
        "rails server",
    ])
    def test_enabled(self, cmd: str):
        assert should_auto_enable_managed_service_mode(cmd) is True

    @pytest.mark.parametrize("cmd", [
        "go run --help",
        "npm test",
        "echo hello && npm run dev",
        "docker compose up -d",
        "",
    ])
    def test_not_enabled(self, cmd: str):
        assert should_auto_enable_managed_service_mode(cmd) is False


# ===================================================================
# apply_bash_intent_contract
# ===================================================================

class TestApplyBashIntentContract:
    def test_no_mismatch_for_non_help(self):
        result = apply_bash_intent_contract(
            command="npm run dev",
            exit_code=0,
            output="listening on port 3000",
            duration_ms=5000.0,
        )
        assert result is None

    def test_no_mismatch_when_not_success(self):
        result = apply_bash_intent_contract(
            command="npm run --help",
            exit_code=1,
            output="listening on port 3000",
            duration_ms=5000.0,
            success=False,
        )
        assert result is None

    def test_no_mismatch_when_help_output(self):
        result = apply_bash_intent_contract(
            command="npm run --help",
            exit_code=0,
            output="Usage: npm run <script>\n\nOptions:\n  --help  Show this help",
            duration_ms=500.0,
        )
        assert result is None

    def test_mismatch_detected(self):
        result = apply_bash_intent_contract(
            command="go run --help main.go",
            exit_code=0,
            output="server started listening on :8080",
            duration_ms=15000.0,
        )
        assert result is not None
        assert result["error_code"] == "E_INTENT_MISMATCH"
        assert "suggestions" in result
        assert len(result["suggestions"]) >= 3
        assert result["metadata"]["reasonCode"] == "E_INTENT_MISMATCH"
        assert result["metadata"]["intent"] == "help_only"

    def test_mismatch_with_stderr(self):
        result = apply_bash_intent_contract(
            command="flask run --help",
            exit_code=0,
            output="",
            duration_ms=12000.0,
            stderr="gateway listening on 0.0.0.0:5000",
        )
        assert result is not None
        assert result["error_code"] == "E_INTENT_MISMATCH"

    def test_existing_suggestions_preserved(self):
        result = apply_bash_intent_contract(
            command="uvicorn --help app:app",
            exit_code=0,
            output="listening on 0.0.0.0:8000",
            duration_ms=10000.0,
            suggestions=["Try a different approach"],
        )
        assert result is not None
        assert "Try a different approach" in result["suggestions"]


# ===================================================================
# Helper function tests
# ===================================================================

class TestStripShellPrefixes:
    def test_env_prefix(self):
        assert strip_shell_prefixes("FOO=bar echo hi") == "echo hi"

    def test_sudo(self):
        assert strip_shell_prefixes("sudo apt install foo") == "apt install foo"

    def test_both(self):
        assert strip_shell_prefixes("FOO=1 sudo cmd") == "cmd"


class TestIsHelpOnlyBashCommand:
    def test_help_flag(self):
        assert is_help_only_bash_command("npm run --help") is True

    def test_h_flag(self):
        assert is_help_only_bash_command("go run -h") is True

    def test_no_help(self):
        assert is_help_only_bash_command("npm run dev") is False
