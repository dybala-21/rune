"""Real-world AI agent scenario tests based on actual project types.

Each scenario simulates a real developer's week working with an AI agent
on a specific kind of project. The tests verify that the self-improving
rule system correctly learns from realistic task sequences.

Project inspiration: fintech, devops, gateway-go, rust-axum, security,
science, dataeng, AetherArc (WebSocket chat), from test-workspace.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from rune.memory.rule_learner import (
    _INITIAL_CONFIDENCE,
    _INJECTION_THRESHOLD,
    _error_signature,
    find_repeated_failures,
    get_rules_for_domain,
    update_rules_from_outcome,
)
from rune.memory.store import MemoryStore
from rune.memory.types import Episode
from rune.proactive.reflexion import ReflexionLearner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store():
    s = MemoryStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def meta_env(tmp_dir, monkeypatch):
    """Wired meta environment for rule validation tests."""
    state_dir = tmp_dir / "memory" / ".state"
    state_dir.mkdir(parents=True)
    memory_dir = tmp_dir / "memory"

    def _load():
        p = state_dir / "fact-meta.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _save(data):
        (state_dir / "fact-meta.json").write_text(json.dumps(data, indent=2))

    def _update(key, updates):
        meta = _load()
        entry = meta.get(key, {})
        entry.update(updates)
        meta[key] = entry
        _save(meta)

    monkeypatch.setattr("rune.memory.rule_learner.load_fact_meta", _load)
    monkeypatch.setattr("rune.memory.rule_learner.save_fact_meta", _save)
    monkeypatch.setattr("rune.memory.rule_learner.update_fact_meta", _update)
    monkeypatch.setattr("rune.memory.rule_learner.load_suppressed", lambda: {})

    # Wire learned.md
    learned_md = memory_dir / "learned.md"
    learned_md.write_text("# Auto-learned facts\n\n")

    def _save_learned(category, key, value, confidence=0.5, path=None):
        import re as _re
        line = f"- [{category}] {key}: {value} ({confidence:.2f})\n"
        text = learned_md.read_text()
        pattern = rf"- \[{_re.escape(category)}\] {_re.escape(key)}:.*\n?"
        text = _re.sub(pattern, line, text) if _re.search(pattern, text) else text + line
        learned_md.write_text(text)

    def _parse_learned():
        import re as _re
        pat = _re.compile(
            r"^- \[(?P<category>[^\]]+)\]\s*(?P<key>[^:]+):\s*(?P<value>.+?)"
            r"(?:\s*\((?P<confidence>[\d.]+)\))?\s*$"
        )
        facts = []
        for line in learned_md.read_text().splitlines():
            m = pat.match(line.strip())
            if m:
                c = m.group("confidence")
                facts.append({
                    "category": m.group("category").strip(),
                    "key": m.group("key").strip(),
                    "value": m.group("value").strip(),
                    "confidence": float(c) if c else 0.5,
                })
        return facts

    monkeypatch.setattr("rune.memory.rule_learner.save_learned_fact", _save_learned)
    monkeypatch.setattr("rune.memory.markdown_store.parse_learned_md", _parse_learned)

    class Env:
        def __init__(self):
            self.load_meta = _load
            self.save_meta = _save

        def add_rule(self, domain, human_key, value, sig=None):
            sig = sig or _error_signature("bash", human_key)
            meta_key = f"rule:{domain}:{sig}"
            _save_learned(f"rule:{domain}", human_key, value, _INITIAL_CONFIDENCE)
            meta = _load()
            meta[meta_key] = {
                "confidence": _INITIAL_CONFIDENCE,
                "hit_count": 0, "eval_count": 0,
                "source": "rule_learner",
                "human_key": human_key,
                "category": f"rule:{domain}",
                "created_at": "2026-03-20T00:00:00+00:00",
            }
            _save(meta)
            return meta_key

        def run_task(self, domain, success, goal, error=""):
            return update_rules_from_outcome(domain, success, goal=goal, error_message=error)

        def get_confidence(self, key):
            return _load().get(key, {}).get("confidence", 0)

        def get_injected(self, domain):
            return get_rules_for_domain(domain)

    yield Env()


@pytest.fixture
def learner():
    return ReflexionLearner()


# ===========================================================================
# Scenario 1: Fintech — 포트폴리오 분석기 개발
#   사용자가 금융 분석 코드를 작성하면서 숫자 정밀도 에러를 반복
# ===========================================================================


class TestFintechPortfolioScenario:
    """fintech 프로젝트: 포트폴리오 백테스팅 중 float 정밀도 에러 반복."""

    def test_float_precision_pattern_detected(self, store):
        """같은 pytest 에러가 다른 파일/줄에서 반복되면 패턴 감지.

        Note: _error_signature normalizes paths and numbers but NOT
        arbitrary text. So we use errors that differ only in path/line."""
        errors = [
            "FAILED /Users/dev/fintech/test_backtest.py::test_precision - AssertionError: float mismatch",
            "FAILED /Users/dev/fintech/test_portfolio.py::test_precision - AssertionError: float mismatch",
            "FAILED /Users/dev/fintech/test_risk.py::test_precision - AssertionError: float mismatch",
        ]
        for i, err in enumerate(errors):
            store.log_tool_call(f"s{i}", "bash", result_success=False,
                               error_message=err, duration_ms=200.0)

        patterns = find_repeated_failures(store)
        assert len(patterns) >= 1

    def test_financial_rule_progression(self, meta_env):
        """'Decimal 사용' 규칙이 금융 태스크에서 검증됨."""
        key = meta_env.add_rule("code_modify", "use_decimal_for_finance",
                                "use Decimal type for financial calculations")

        # 금융 계산 태스크: 대부분 성공 (Decimal 덕분)
        finance_tasks = [
            ("calculate portfolio returns with decimal precision", True),
            ("compute Sharpe ratio using decimal math", True),
            ("backtest strategy with decimal accuracy", True),
            ("price bond using decimal yield curve", True),
            ("calculate compound interest with decimal", False),  # 다른 이유로 실패
            ("compute option pricing with decimal greeks", True),
            ("generate decimal-precise tax report", True),
            ("validate decimal portfolio weights sum to 1", True),
            ("calculate decimal risk-adjusted returns", True),
            ("compute decimal dividend yield", False),
            ("run backtest with decimal transaction costs", True),
            ("calculate decimal annualized volatility", True),
        ]
        for goal, success in finance_tasks:
            meta_env.run_task("code_modify", success, goal)

        conf = meta_env.get_confidence(key)
        assert conf >= _INJECTION_THRESHOLD, f"Finance rule should be promoted: {conf:.3f}"


# ===========================================================================
# Scenario 2: DevOps — 설정 파일 자동화
#   YAML 설정 오류를 반복하면서 규칙 학습
# ===========================================================================


class TestDevOpsConfigScenario:
    """devops 프로젝트: YAML/JSON 설정 검증 실패 반복."""

    def test_yaml_validation_pattern(self, store):
        """YAML 문법 에러가 다른 파일에서 반복."""
        store.log_tool_call("s1", "bash", result_success=False,
                           error_message="yaml.scanner.ScannerError: mapping values not allowed in /Users/dev/devops/config.yaml line 15")
        store.log_tool_call("s2", "bash", result_success=False,
                           error_message="yaml.scanner.ScannerError: mapping values not allowed in /Users/dev/devops/nginx.yaml line 8")

        patterns = find_repeated_failures(store)
        assert len(patterns) == 1

    def test_config_rules_domain_isolation(self, meta_env, learner):
        """DevOps 태스크의 교훈이 code_modify 도메인에 간섭 안 함."""
        meta_env.add_rule("execution", "validate_yaml_syntax",
                          "validate YAML with yamllint before applying")

        # execution 도메인 태스크
        meta_env.run_task("execution", True, "validate yaml config and deploy")
        meta_env.run_task("execution", True, "validate yaml syntax for nginx")

        # code_modify 도메인에는 영향 없음
        assert meta_env.get_injected("code_modify") == []

        # Reflexion도 별도 도메인으로 추적
        learner.record_task_outcome({"domain": "execution", "success": True,
                                     "goal": "deploy config", "steps_taken": 3})
        learner.record_task_outcome({"domain": "code_modify", "success": False,
                                     "goal": "refactor auth", "error": "timeout",
                                     "steps_taken": 15})

        assert learner.get_domain_success_rate("execution") == 1.0
        assert learner.get_domain_success_rate("code_modify") == 0.0


# ===========================================================================
# Scenario 3: Go API Gateway — 동시 수정 충돌
#   여러 미들웨어 파일을 수정하면서 stale content 에러 반복
# ===========================================================================


class TestGoGatewayScenario:
    """gateway-go 프로젝트: 미들웨어 파일 동시 수정 시 stale content."""

    def test_stale_content_across_middleware_files(self, store):
        """Go 미들웨어 파일들에서 stale content 에러 반복."""
        middleware_errors = [
            "file_edit failed: content mismatch in /Users/dev/gateway-go/middleware/rate_limit.go at line 42",
            "file_edit failed: content mismatch in /Users/dev/gateway-go/middleware/auth.go at line 18",
            "file_edit failed: content mismatch in /Users/dev/gateway-go/middleware/logging.go at line 7",
        ]
        for i, err in enumerate(middleware_errors):
            store.log_tool_call(f"gw-{i}", "file_edit", result_success=False,
                               error_message=err, duration_ms=150.0)

        patterns = find_repeated_failures(store)
        assert len(patterns) == 1
        assert patterns[0]["tool_name"] == "file_edit"
        assert patterns[0]["count"] == 3

    def test_concurrent_edit_rule_validation(self, meta_env):
        """'수정 전 re-read' 규칙이 Go 프로젝트 수정에서 검증."""
        key = meta_env.add_rule("code_modify", "reread_before_edit",
                                "always re-read file before applying edit")

        # Go gateway 수정 작업들
        go_tasks = [
            ("edit rate_limit.go to reread and update config", True),
            ("add JWT auth middleware, reread handler first", True),
            ("edit logging middleware to reread format config", True),
            ("fix circuit breaker, reread state before edit", True),
            ("update proxy headers, reread upstream config", False),  # 네트워크 이슈
            ("edit metrics endpoint to reread counter", True),
            ("refactor router, reread route table first", True),
            ("add health check, reread service config", True),
        ]
        for goal, success in go_tasks:
            meta_env.run_task("code_modify", success, goal)

        assert meta_env.get_confidence(key) > _INITIAL_CONFIDENCE


# ===========================================================================
# Scenario 4: Rust Axum — 소유권/수명 에러 학습
#   Rust 초보자가 ownership/borrow 에러를 반복
# ===========================================================================


class TestRustOwnershipScenario:
    """rust-axum-study: 소유권 에러 패턴 학습."""

    def test_borrow_checker_pattern(self, store):
        """Rust borrow checker 에러가 다른 파일에서 반복.
        Error text differs only in path and line number."""
        store.log_tool_call("r1", "bash", result_success=False,
                           error_message="error: use of moved value in /Users/dev/rust/src/handler.rs line 25")
        store.log_tool_call("r2", "bash", result_success=False,
                           error_message="error: use of moved value in /Users/dev/rust/src/main.rs line 42")

        patterns = find_repeated_failures(store)
        assert len(patterns) >= 1

    def test_ownership_lessons_via_reflexion(self, learner):
        """Rust 소유권 에러에서 구체적 교훈 추출."""
        learner.record_task_outcome({
            "domain": "code_modify",
            "success": False,
            "goal": "Share database pool across Axum handlers",
            "error": "error[E0382]: use of moved value: `pool`",
            "steps_taken": 8,
        })

        lessons = learner.get_domain_lessons("code_modify")
        assert len(lessons) >= 1


# ===========================================================================
# Scenario 5: 데이터 엔지니어링 — ETL 파이프라인 디버깅
#   스키마 불일치와 파티션 에러 반복
# ===========================================================================


class TestDataEngScenario:
    """dataeng 프로젝트: ETL 파이프라인 스키마 검증 실패."""

    def test_schema_mismatch_pattern(self, store):
        """스키마 불일치 에러가 다른 파일에서 반복.
        Text after path/line normalization must be identical."""
        store.log_tool_call("etl-1", "bash", result_success=False,
                           error_message="SchemaValidationError: type mismatch in /Users/dev/dataeng/pipeline.py line 55")
        store.log_tool_call("etl-2", "bash", result_success=False,
                           error_message="SchemaValidationError: type mismatch in /Users/dev/dataeng/transform.py line 82")

        patterns = find_repeated_failures(store)
        assert len(patterns) == 1

    def test_etl_rule_with_mixed_tasks(self, meta_env):
        """스키마 검증 규칙이 ETL 관련 태스크에서만 평가됨."""
        meta_env.add_rule("code_modify", "validate_schema_before_load",
                                "validate schema compatibility before data load")

        # 혼합 태스크: ETL + 비ETL
        tasks = [
            ("validate schema and load user data", True),          # 관련 ✓
            ("fix CSS styling in dashboard", True),                 # 무관
            ("validate schema for transaction table", True),        # 관련 ✓
            ("update README documentation", True),                  # 무관
            ("validate schema migration script", False),            # 관련 ✓
            ("refactor frontend components", True),                 # 무관
            ("run schema validate on staging data", True),          # 관련 ✓
        ]
        for goal, success in tasks:
            meta_env.run_task("code_modify", success, goal)

        meta = meta_env.load_meta()
        entry = [v for v in meta.values() if v.get("human_key") == "validate_schema_before_load"][0]
        # 무관한 태스크에서는 평가 안 됐으므로 eval_count < 7
        assert entry["eval_count"] < len(tasks), "Irrelevant tasks should be filtered"
        assert entry["eval_count"] >= 3, "Related tasks should be counted"


# ===========================================================================
# Scenario 6: 보안 도구 — 크리덴셜 노출 실수 반복
#   .env 파일이나 API 키를 실수로 커밋하려는 패턴
# ===========================================================================


class TestSecurityScenario:
    """security 프로젝트: 민감 정보 노출 방지 규칙."""

    def test_credential_leak_pattern(self, store):
        """Guardian이 민감 정보 쓰기를 차단하는 패턴 감지.
        Text must normalize identically — differ only in path/line."""
        store.log_tool_call("sec-1", "bash", result_success=False,
                           error_message="Guardian DENIED: secret detected in /Users/dev/security/config.py line 3")
        store.log_tool_call("sec-2", "bash", result_success=False,
                           error_message="Guardian DENIED: secret detected in /Users/dev/security/auth.py line 12")

        patterns = find_repeated_failures(store)
        assert len(patterns) >= 1

    def test_security_rule_stays_strict(self, meta_env):
        """보안 규칙은 성공/실패와 무관하게 유지되어야 함.
        보안 관련 태스크에서 실패해도 규칙 자체는 올바를 수 있음."""
        key = meta_env.add_rule("code_modify", "never_hardcode_secrets",
                                "use environment variables for secrets, never hardcode")

        # 보안 태스크: secrets 관련 작업
        tasks = [
            ("refactor auth to use env secrets instead of hardcode", True),
            ("move API secrets to env file", True),
            ("rotate secrets and update config", True),
            ("fix secrets management in CI pipeline", False),  # CI 문제
            ("audit hardcoded secrets in codebase", True),
            ("implement secrets vault integration", True),
        ]
        for goal, success in tasks:
            meta_env.run_task("code_modify", success, goal)

        conf = meta_env.get_confidence(key)
        assert conf > _INITIAL_CONFIDENCE, "Security rule should gain confidence"


# ===========================================================================
# Scenario 7: WebSocket 채팅 서버 — 연결 끊김 디버깅
#   AetherArc 스타일: 타임아웃/연결 관련 에러 반복
# ===========================================================================


class TestWebSocketScenario:
    """AetherArc 프로젝트: WebSocket 연결 관리 에러."""

    def test_connection_timeout_pattern(self, store):
        """WebSocket 타임아웃이 다른 핸들러에서 반복."""
        store.log_tool_call("ws-1", "bash", result_success=False,
                           error_message="ConnectionResetError: WebSocket connection reset in /Users/dev/aether/gateway.rs line 120")
        store.log_tool_call("ws-2", "bash", result_success=False,
                           error_message="ConnectionResetError: WebSocket connection reset in /Users/dev/aether/handler.rs line 85")

        patterns = find_repeated_failures(store)
        assert len(patterns) == 1

    def test_websocket_debugging_session(self, store, learner):
        """WebSocket 디버깅 세션: 다수 시도 후 성공."""
        # 디버깅 과정에서의 tool call 패턴
        debug_steps = [
            ("bash", False, "error: WebSocket handshake failed, timeout after 30s"),
            ("file_read", True, ""),
            ("bash", False, "error: connection refused on port 8080"),
            ("file_edit", True, ""),
            ("bash", True, ""),  # 드디어 성공
        ]
        for tool, success, error in debug_steps:
            store.log_tool_call("ws-debug", tool, result_success=success,
                               error_message=error, duration_ms=500.0)

        # Reflexion: 긴 디버깅 → 교훈
        learner.record_task_outcome({
            "domain": "code_modify",
            "success": True,
            "goal": "Fix WebSocket timeout in chat gateway",
            "steps_taken": 12,
            "duration_ms": 60000,
        })

        lessons = learner.get_domain_lessons("code_modify")
        assert any("direct" in l.lower() or "12 steps" in l for l in lessons)


# ===========================================================================
# Scenario 8: 전체 주간 시뮬레이션 — 다양한 프로젝트 혼합
#   하루에 여러 프로젝트를 오가며 작업하는 현실적 패턴
# ===========================================================================


class TestMixedProjectWeek:
    """여러 프로젝트를 오가며 작업하는 주간 시뮬레이션."""

    def test_multi_project_rule_evolution(self, meta_env):
        """다양한 프로젝트에서 규칙들이 독립적으로 진화."""
        # 규칙 3개 (각각 다른 프로젝트 유형에서 생성)
        rule_edit = meta_env.add_rule("code_modify", "reread_before_edit",
                                      "re-read file content before editing")
        rule_test = meta_env.add_rule("code_modify", "run_related_tests",
                                      "run related tests after code changes")
        rule_schema = meta_env.add_rule("code_modify", "check_schema_compat",
                                        "check schema compatibility before migration")

        # 월요일: fintech (포트폴리오 코드 수정)
        meta_env.run_task("code_modify", True, "reread and edit backtest.py for new strategy")
        meta_env.run_task("code_modify", True, "run related tests for portfolio analyzer")
        meta_env.run_task("code_modify", True, "reread and edit risk_calculator.py")
        meta_env.run_task("code_modify", False, "run related tests for bond pricing",
                          error="TimeoutError: test suite exceeded 120s")

        # 화요일: gateway-go (미들웨어 수정)
        meta_env.run_task("code_modify", True, "reread rate_limit.go and edit config")
        meta_env.run_task("code_modify", True, "reread auth.go before edit")
        meta_env.run_task("code_modify", True, "run related tests for middleware")

        # 수요일: dataeng (ETL 스키마 작업)
        meta_env.run_task("code_modify", True, "check schema for user_events table")
        meta_env.run_task("code_modify", False, "check schema migration failed",
                          error="SchemaValidationError: type mismatch")
        meta_env.run_task("code_modify", True, "check schema compatibility for new columns")

        # 목요일: security (인증 리팩터링)
        meta_env.run_task("code_modify", True, "reread auth module before edit")
        meta_env.run_task("code_modify", True, "run related tests for auth refactor")

        # 금요일: AetherArc (WebSocket 디버깅)
        meta_env.run_task("code_modify", True, "reread gateway.rs before edit")
        meta_env.run_task("code_modify", False, "run related tests, WebSocket timeout")
        meta_env.run_task("code_modify", True, "reread handler.rs and fix connection")

        # 결과 검증
        edit_conf = meta_env.get_confidence(rule_edit)
        test_conf = meta_env.get_confidence(rule_test)
        schema_conf = meta_env.get_confidence(rule_schema)

        # reread_before_edit: 거의 항상 성공 → 높은 신뢰도
        assert edit_conf > test_conf, (
            f"Edit rule ({edit_conf:.3f}) should be higher than test rule ({test_conf:.3f})"
        )

        # run_related_tests: 일부 실패 포함 → 변화 방향은 keyword 매칭에 따라 다름
        # 중요한 건 edit 규칙보다 낮다는 것
        assert test_conf != _INITIAL_CONFIDENCE, (
            f"Test rule should have been evaluated: {test_conf:.3f}"
        )

        # check_schema: 적은 관련 태스크 → 느린 변화
        # schema 규칙이 edit보다 낮아야 함 (관련 태스크가 적으므로)
        assert schema_conf < edit_conf

    def test_reflexion_tracks_cross_project_patterns(self, learner):
        """여러 프로젝트에서 공통 패턴 추출."""
        # 월~금 태스크 결과
        outcomes = [
            {"domain": "code_modify", "success": True,
             "goal": "Fix fintech backtest", "steps_taken": 5},
            {"domain": "code_modify", "success": False,
             "goal": "Deploy Go gateway", "error": "TimeoutError: build exceeded limit",
             "steps_taken": 12},
            {"domain": "research", "success": True,
             "goal": "Research Rust async patterns", "steps_taken": 3},
            {"domain": "code_modify", "success": True,
             "goal": "Fix ETL pipeline", "steps_taken": 1},
            {"domain": "code_modify", "success": False,
             "goal": "Refactor WebSocket handler",
             "error": "PermissionError: /var/run/docker.sock",
             "steps_taken": 8},
        ]
        for o in outcomes:
            learner.record_task_outcome(o)

        # 성공률 = 2/4 = 50%
        rate = learner.get_domain_success_rate("code_modify")
        assert rate == pytest.approx(0.5, abs=0.01)

        # 교훈이 추출되어야 함
        lessons = learner.get_domain_lessons("code_modify")
        assert len(lessons) >= 2  # timeout + permission

        # research는 100%
        assert learner.get_domain_success_rate("research") == 1.0


# ===========================================================================
# Scenario 9: 에피소드 기반 파일 회상 — 프로젝트 간 이동
# ===========================================================================


class TestCrossProjectFileRecall:
    """여러 프로젝트 파일을 수정한 에피소드가 올바르게 회상됨."""

    def test_recall_by_project_file(self, store):
        """fintech 파일, devops 파일, Go 파일 각각 에피소드 조회."""
        episodes = [
            ("Fix Sharpe ratio calculation", '["fintech/portfolio_analyzer.py"]'),
            ("Update nginx config template", '["devops/config_manager.py"]'),
            ("Add rate limiting middleware", '["gateway-go/middleware/rate_limit.go"]'),
            ("Fix WebSocket reconnection", '["AetherArc/src/gateway.rs"]'),
            ("Improve portfolio analyzer performance", '["fintech/portfolio_analyzer.py"]'),
        ]
        for summary, files in episodes:
            store.save_episode(Episode(
                task_summary=summary,
                files_touched=files,
                utility=1,
                importance=0.7,
                timestamp=datetime.now(UTC).isoformat(),
            ))

        # fintech 파일 관련 에피소드
        results = store.get_episodes_by_file("portfolio_analyzer.py")
        assert len(results) == 2
        summaries = {ep.task_summary for ep in results}
        assert "Fix Sharpe ratio calculation" in summaries
        assert "Improve portfolio analyzer performance" in summaries

        # Go gateway 파일 관련 에피소드
        results = store.get_episodes_by_file("rate_limit.go")
        assert len(results) == 1

    def test_episode_utility_tracks_project_difficulty(self, store):
        """프로젝트별 성공/실패 비율이 에피소드에 반영됨."""
        # Rust 프로젝트: 어려움 (실패 많음)
        for i in range(5):
            store.save_episode(Episode(
                task_summary=f"Rust task {i}",
                intent="code_modify",
                utility=-1 if i < 3 else 1,  # 60% 실패
                importance=0.6,
                timestamp=datetime.now(UTC).isoformat(),
            ))

        # Python 프로젝트: 쉬움 (성공 많음)
        for i in range(5):
            store.save_episode(Episode(
                task_summary=f"Python task {i}",
                intent="code_modify",
                utility=1 if i < 4 else -1,  # 80% 성공
                importance=0.5,
                timestamp=datetime.now(UTC).isoformat(),
            ))

        all_eps = store.get_recent_episodes(20)
        rust_eps = [ep for ep in all_eps if "Rust" in ep.task_summary]
        python_eps = [ep for ep in all_eps if "Python" in ep.task_summary]

        rust_success = sum(1 for ep in rust_eps if ep.utility > 0) / len(rust_eps)
        python_success = sum(1 for ep in python_eps if ep.utility > 0) / len(python_eps)

        assert python_success > rust_success
