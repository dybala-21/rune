"""End-to-end scenario tests: simulate real developer workflows.

Each test simulates a week of actual AI agent usage and checks
observable outcomes a user would notice.
"""

from __future__ import annotations

import json

import pytest

from rune.memory.rule_learner import (
    _GC_THRESHOLD,
    _INITIAL_CONFIDENCE,
    _INJECTION_THRESHOLD,
    _error_signature,
    get_rules_for_domain,
    update_rules_from_outcome,
)
from rune.memory.store import MemoryStore

# ---------------------------------------------------------------------------
# Fixtures — a fully wired local environment
# ---------------------------------------------------------------------------


@pytest.fixture
def env(tmp_dir, monkeypatch):
    """Full local environment: DB + state dir + learned.md."""
    state_dir = tmp_dir / "memory" / ".state"
    state_dir.mkdir(parents=True)
    memory_dir = tmp_dir / "memory"

    # Wire state.py to temp
    monkeypatch.setattr("rune.memory.state._state_dir", lambda: state_dir)

    # Wire rule_learner's state imports to temp
    def _load():
        p = state_dir / "fact-meta.json"
        if not p.exists():
            return {}
        return json.loads(p.read_text())

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

    # Wire markdown_store for get_rules_for_domain
    learned_md = memory_dir / "learned.md"
    learned_md.write_text("# Auto-learned facts\n\n")

    def _parse_learned():
        from rune.memory.markdown_store import _LEARNED_RE
        facts = []
        for _i, line in enumerate(learned_md.read_text().splitlines()):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            m = _LEARNED_RE.match(stripped)
            if m:
                conf_str = m.group("confidence")
                facts.append({
                    "category": m.group("category").strip(),
                    "key": m.group("key").strip(),
                    "value": m.group("value").strip(),
                    "confidence": float(conf_str) if conf_str else 0.5,
                })
        return facts

    def _save_learned(category, key, value, confidence=0.5, path=None):
        line = f"- [{category}] {key}: {value} ({confidence:.2f})\n"
        text = learned_md.read_text()
        # Update existing or append
        import re
        pattern = rf"- \[{re.escape(category)}\] {re.escape(key)}:.*\n?"
        if re.search(pattern, text):
            text = re.sub(pattern, line, text)
        else:
            text += line
        learned_md.write_text(text)

    monkeypatch.setattr("rune.memory.rule_learner.save_learned_fact", _save_learned)
    monkeypatch.setattr("rune.memory.markdown_store.parse_learned_md", _parse_learned)

    # Create DB
    store = MemoryStore(db_path=":memory:")

    class Env:
        def __init__(self):
            self.store = store
            self.state_dir = state_dir
            self.memory_dir = memory_dir
            self.learned_md = learned_md
            self.load_meta = _load
            self.save_meta = _save

        def add_rule(self, domain, human_key, value, sig=None):
            """Manually add a rule (simulating learn_from_failures)."""
            sig = sig or _error_signature("bash", human_key)
            meta_key = f"rule:{domain}:{sig}"
            # Save to learned.md
            _save_learned(f"rule:{domain}", human_key, value, _INITIAL_CONFIDENCE)
            # Save to meta
            meta = _load()
            meta[meta_key] = {
                "confidence": _INITIAL_CONFIDENCE,
                "hit_count": 0,
                "eval_count": 0,
                "source": "rule_learner",
                "human_key": human_key,
                "category": f"rule:{domain}",
                "created_at": "2026-03-20T00:00:00+00:00",
            }
            _save(meta)
            return meta_key

        def run_task(self, domain, success, goal, error=""):
            """Simulate a task completion and feed outcome to rules."""
            return update_rules_from_outcome(domain, success, goal=goal, error_message=error)

        def get_injected_rules(self, domain):
            """What rules would actually appear in the agent's prompt?"""
            return get_rules_for_domain(domain)

        def get_confidence(self, meta_key):
            """Get current confidence of a rule."""
            return self.load_meta().get(meta_key, {}).get("confidence", 0)

        def get_eval_count(self, meta_key):
            """How many tasks have evaluated this rule."""
            return self.load_meta().get(meta_key, {}).get("eval_count", 0)

    yield Env()
    store.close()


# ===========================================================================
# Scenario A: "파일 수정 전 항상 re-read" 규칙이 실제로 도움이 되는 경우
# ===========================================================================


class TestScenarioGoodRule:
    """사용자가 에이전트에게 여러 파일 수정을 요청.
    에이전트가 stale content 에러를 반복 → 규칙 생성.
    이후 edit 관련 태스크에서 대부분 성공 → 규칙이 프롬프트에 등장."""

    def test_rule_not_injected_immediately(self, env):
        """규칙 생성 직후에는 프롬프트에 들어가지 않아야 함."""
        env.add_rule("code_modify", "verify_before_edit",
                     "re-read file before file_edit to avoid stale content")

        rules = env.get_injected_rules("code_modify")
        assert len(rules) == 0, "New rule should NOT be in prompt yet"

    def test_rule_promoted_after_consistent_success(self, env):
        """80% 성공률로 15개 태스크 → 규칙이 프롬프트에 등장."""
        key = env.add_rule("code_modify", "verify_before_edit",
                           "re-read file before file_edit to avoid stale content")

        # 15 edit tasks, 80% success
        tasks = [
            ("edit auth.py to add verify logic", True),
            ("edit user.py to fix verify bug", True),
            ("edit config.py to verify settings", True),
            ("edit handler.py for verify endpoint", False),
            ("edit routes.py with verify middleware", True),
            ("edit test_auth.py to verify assertions", True),
            ("edit models.py for verify field", True),
            ("edit views.py to add verify check", True),
            ("edit serializers.py for verify method", False),
            ("edit permissions.py to verify access", True),
            ("edit middleware.py with verify header", True),
            ("edit signals.py for verify event", True),
            ("edit admin.py to verify config", False),
            ("edit forms.py with verify validator", True),
            ("edit utils.py for verify helper", True),
        ]

        for goal, success in tasks:
            env.run_task("code_modify", success, goal)

        # Check rule is now injected
        rules = env.get_injected_rules("code_modify")
        assert len(rules) == 1
        assert rules[0]["key"] == "verify_before_edit"
        assert rules[0]["confidence"] >= _INJECTION_THRESHOLD

        # Check metadata
        assert env.get_eval_count(key) == 15

    def test_visible_confidence_progression(self, env):
        """사용자가 확인할 수 있는 신뢰도 변화 추이."""
        key = env.add_rule("code_modify", "verify_before_edit",
                           "re-read file before file_edit")

        progression = [env.get_confidence(key)]

        # 10 tasks: 8 success, 2 failure
        outcomes = [True, True, True, False, True, True, True, True, False, True]
        for success in outcomes:
            env.run_task("code_modify", success, "edit file with verify check")
            progression.append(round(env.get_confidence(key), 3))

        # Confidence should generally trend upward
        assert progression[-1] > progression[0], (
            f"Expected upward trend, got: {progression}"
        )
        # Should not be flat
        assert len(set(progression)) > 3, (
            f"Expected variation, got: {progression}"
        )


# ===========================================================================
# Scenario B: 잘못된 규칙이 빠르게 제거되는 경우
# ===========================================================================


class TestScenarioHarmfulRule:
    """LLM이 잘못된 규칙을 생성. 적용된 태스크들이 오히려 실패.
    비대칭 페널티로 빠르게 GC 임계값 아래로 떨어져야 함."""

    def test_harmful_rule_never_reaches_prompt(self, env):
        """50% 성공률 → 규칙이 프롬프트에 절대 도달하지 않음."""
        key = env.add_rule("code_modify", "always_restart_server",
                           "restart dev server before every file edit")

        # 20 tasks, 50% success (bad rule doesn't help)
        for i in range(20):
            env.run_task("code_modify", i % 2 == 0,
                        "restart server and edit config")

        rules = env.get_injected_rules("code_modify")
        assert len(rules) == 0, "Harmful rule should never reach prompt"

        # Should have decayed below GC
        conf = env.get_confidence(key)
        assert conf < _GC_THRESHOLD, f"Expected < {_GC_THRESHOLD}, got {conf}"

    def test_harmful_rule_faster_removal_than_good_promotion(self, env):
        """나쁜 규칙 제거(~10 태스크)가 좋은 규칙 승격(~15 태스크)보다 빠름."""
        good_key = env.add_rule("code_modify", "verify_before_edit",
                                "re-read file before edit")
        bad_key = env.add_rule("code_modify", "always_rebuild",
                               "run full rebuild before every change")

        tasks_to_gc = 0
        tasks_to_inject = 0

        for i in range(30):
            # Good rule: 80% success on edit tasks
            env.run_task("code_modify", i % 5 != 0, "edit file to verify fix")
            # Bad rule: 40% success on rebuild tasks
            env.run_task("code_modify", i % 5 < 2, "rebuild and deploy changes")

            if tasks_to_gc == 0 and env.get_confidence(bad_key) < _GC_THRESHOLD:
                tasks_to_gc = i + 1
            if tasks_to_inject == 0 and env.get_confidence(good_key) >= _INJECTION_THRESHOLD:
                tasks_to_inject = i + 1

        assert tasks_to_gc > 0, "Bad rule should have been GC'd"
        assert tasks_to_inject > 0, "Good rule should have been promoted"
        assert tasks_to_gc < tasks_to_inject, (
            f"Bad rule removal ({tasks_to_gc}) should be faster than "
            f"good rule promotion ({tasks_to_inject})"
        )


# ===========================================================================
# Scenario C: 도메인 격리 — research 규칙이 code_modify에 영향 안 줌
# ===========================================================================


class TestScenarioDomainIsolation:
    """연구 도메인과 코드 수정 도메인의 규칙이 서로 간섭하지 않음."""

    def test_cross_domain_no_interference(self, env):
        """research 규칙의 성공이 code_modify 규칙에 영향 없음."""
        code_key = env.add_rule("code_modify", "check_syntax",
                                "validate syntax before save")
        research_key = env.add_rule("research", "verify_source",
                                    "check source credibility before citing")

        # 10 successful research tasks
        for _ in range(10):
            env.run_task("research", True, "verify source and cite paper")

        # code_modify rule should be unchanged
        assert env.get_confidence(code_key) == _INITIAL_CONFIDENCE
        assert env.get_eval_count(code_key) == 0

        # research rule should have been updated
        assert env.get_eval_count(research_key) == 10
        assert env.get_confidence(research_key) > _INITIAL_CONFIDENCE


# ===========================================================================
# Scenario D: 키워드 관련성 필터링이 실제로 노이즈를 줄이는지
# ===========================================================================


class TestScenarioRelevanceFiltering:
    """3개 규칙이 있을 때, 관련 없는 태스크가 무관한 규칙을 건드리지 않는지."""

    def test_only_relevant_rules_updated(self, env):
        key_edit = env.add_rule("code_modify", "verify_before_edit",
                                "re-read file before edit")
        key_import = env.add_rule("code_modify", "check_import_paths",
                                  "verify import paths before test run")
        key_timeout = env.add_rule("code_modify", "reduce_timeout_scope",
                                   "narrow scope when timeout occurs")

        # Task about editing → should only affect "verify_before_edit"
        env.run_task("code_modify", True, "edit auth.py to verify token logic")

        assert env.get_eval_count(key_edit) == 1, "verify rule should be updated"
        assert env.get_eval_count(key_import) == 0, "import rule should NOT be updated"
        assert env.get_eval_count(key_timeout) == 0, "timeout rule should NOT be updated"

    def test_error_message_also_matches(self, env):
        """에러 메시지 내 키워드로도 매칭."""
        key_timeout = env.add_rule("code_modify", "reduce_timeout_scope",
                                   "narrow scope when timeout occurs")

        # Task about CSS (no timeout in goal) but error mentions timeout
        env.run_task("code_modify", False,
                    goal="fix CSS styling",
                    error="TimeoutError: command exceeded timeout limit")

        assert env.get_eval_count(key_timeout) == 1


# ===========================================================================
# Scenario E: 주간 사용 시뮬레이션 — 50개 태스크
# ===========================================================================


class TestScenarioWeeklyUsage:
    """월요일부터 금요일까지 50개 태스크를 처리하는 현실적 시나리오."""

    def test_weekly_simulation(self, env):
        # 규칙 3개 생성 (월요일에 실패 패턴 감지됨)
        rule_edit = env.add_rule("code_modify", "verify_before_edit",
                                 "re-read before edit")          # 실제로 유용
        rule_test = env.add_rule("code_modify", "check_test_output",
                                 "read test output before fixing") # 보통
        rule_bad = env.add_rule("code_modify", "always_restart",
                                "restart IDE before coding")       # 쓸모없음

        weekly_tasks = [
            # 월요일 — edit 작업 위주
            ("edit auth.py to verify token", True),
            ("edit user.py to verify email", True),
            ("edit handler.py to verify request", False),
            ("edit config.py to verify settings", True),
            ("fix test output for login", True),
            ("debug test output mismatch", False),
            ("restart and fix build error", False),
            ("edit routes.py to verify paths", True),
            ("fix failing test output", True),
            ("edit middleware.py with verify logic", True),
            # 화요일 — 테스트 작업 위주
            ("run test and check output", True),
            ("fix test output format", True),
            ("edit views.py to verify response", True),
            ("debug test output parsing", False),
            ("restart service after deploy", False),
            ("edit serializers.py for verify", True),
            ("check test output for API", True),
            ("edit admin.py to verify perms", True),
            ("fix test output encoding", True),
            ("edit forms.py with verify rule", False),
            # 수요일 — 혼합
            ("edit models.py to verify FK", True),
            ("fix test output comparison", True),
            ("restart and rebuild assets", False),
            ("edit signals.py with verify event", True),
            ("check test output regression", True),
            ("edit tasks.py to verify queue", True),
            ("fix flaky test output", False),
            ("edit cache.py with verify TTL", True),
            ("restart dev server for debug", False),
            ("edit logger.py to verify format", True),
            # 목요일 — 주로 edit
            ("edit deploy.py to verify config", True),
            ("edit health.py with verify check", True),
            ("fix test output assertion", True),
            ("edit monitor.py to verify alert", False),
            ("restart service and check", False),
            ("edit backup.py with verify hash", True),
            ("edit migration.py to verify schema", True),
            ("check test output coverage", True),
            ("edit cleanup.py for verify job", True),
            ("restart and run full suite", False),
            # 금요일 — 마무리
            ("edit changelog.py to verify entry", True),
            ("fix test output summary", True),
            ("edit release.py with verify tag", True),
            ("check test output final", True),
            ("edit docs.py to verify links", True),
            ("restart for final deploy", False),
            ("edit notify.py with verify msg", True),
            ("fix test output report", True),
            ("edit version.py to verify bump", True),
            ("final check and verify release", True),
        ]

        for goal, success in weekly_tasks:
            env.run_task("code_modify", success, goal)

        # ---- 결과 확인 ----

        # 1. verify_before_edit: 높은 관련성 + 높은 성공률 → 승격됨
        edit_conf = env.get_confidence(rule_edit)
        assert edit_conf >= _INJECTION_THRESHOLD, (
            f"Good rule should be promoted: {edit_conf:.3f}"
        )

        # 2. check_test_output: 중간 관련성 + 중간 성공률 → 대기 중
        test_conf = env.get_confidence(rule_test)
        # 정확한 값은 예측 어렵지만, GC되지는 않아야 함
        assert test_conf > _GC_THRESHOLD, (
            f"Neutral rule should survive: {test_conf:.3f}"
        )

        # 3. always_restart: 관련 태스크에서 실패율 높음 → GC 근처
        bad_conf = env.get_confidence(rule_bad)
        # restart 관련 태스크가 대부분 실패 → 빠르게 하락
        assert bad_conf < edit_conf, (
            f"Bad rule ({bad_conf:.3f}) should be lower than good ({edit_conf:.3f})"
        )

        # 4. 프롬프트에는 좋은 규칙만 들어감
        injected = env.get_injected_rules("code_modify")
        injected_keys = {r["key"] for r in injected}
        assert "verify_before_edit" in injected_keys, (
            f"Good rule should be in prompt. Injected: {injected_keys}"
        )
        assert "always_restart" not in injected_keys, (
            "Bad rule should NOT be in prompt"
        )

        # 5. 전체 통계 출력 (디버깅용)
        print("\n=== Weekly Simulation Results ===")
        for name, key in [
            ("verify_before_edit (good)", rule_edit),
            ("check_test_output (neutral)", rule_test),
            ("always_restart (bad)", rule_bad),
        ]:
            meta = env.load_meta().get(key, {})
            print(f"  {name}:")
            print(f"    confidence: {meta.get('confidence', 0):.3f}")
            print(f"    eval_count: {meta.get('eval_count', 0)}")
            print(f"    in prompt:  {'YES' if meta.get('confidence', 0) >= _INJECTION_THRESHOLD else 'no'}")
        print(f"\n  Injected into prompt: {[r['key'] for r in injected]}")
        print(f"  Total tasks: {len(weekly_tasks)}")
