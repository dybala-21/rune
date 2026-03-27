"""Tests for the self-improving loop — realistic AI agent learning scenarios.

These tests simulate real-world agent workflows:
1. Agent fails at a task → failure pattern detected → rule generated
2. Agent completes tasks → episodes recorded with utility scores
3. Agent gets rejected → adapts proactive strategy
4. Episodes accumulate → ranked retrieval surfaces relevant past experience
5. Rules decay over time → unused knowledge garbage-collected
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from rune.memory.rule_learner import (
    _DECAY_FACTOR,
    _GC_THRESHOLD,
    _INITIAL_CONFIDENCE,
    _error_signature,
    decay_unused_rules,
    find_repeated_failures,
)
from rune.memory.store import MemoryStore
from rune.memory.types import Episode
from rune.proactive.reflexion import ReflexionLearner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store():
    """In-memory SQLite store for isolated testing."""
    s = MemoryStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def learner():
    """Fresh ReflexionLearner for each test."""
    return ReflexionLearner()


def _make_episode(
    *,
    task_summary: str = "",
    intent: str = "code_modify",
    result: str = "success",
    lessons: str = "[]",
    importance: float = 0.5,
    utility: int = 1,
    files_touched: str = "[]",
    entities: str = "",
    commitments: str = "",
    conversation_id: str = "sess-1",
    duration_ms: float = 5000.0,
    timestamp: str | None = None,
) -> Episode:
    """Helper to build episodes with defaults."""
    return Episode(
        timestamp=timestamp or datetime.now(UTC).isoformat(),
        task_summary=task_summary,
        intent=intent,
        result=result,
        lessons=lessons,
        importance=importance,
        utility=utility,
        files_touched=files_touched,
        entities=entities,
        commitments=commitments,
        conversation_id=conversation_id,
        duration_ms=duration_ms,
    )


# ===========================================================================
# Scenario 1: Agent repeatedly fails file_edit on stale content
# ===========================================================================


class TestStaleFileEditPattern:
    """Real scenario: Agent reads a file, user edits it externally, agent's
    file_edit fails because content doesn't match.  Happens 3 times → rule
    should be detected."""

    def test_stale_content_pattern_detected(self, store):
        """Three stale-content failures across different files should
        produce one unified failure pattern."""
        errors = [
            "file_edit failed: content mismatch in /Users/a/project/routes.py at line 42",
            "file_edit failed: content mismatch in /Users/b/project/user.py at line 18",
            "file_edit failed: content mismatch in /Users/c/project/auth.py at line 7",
        ]
        for i, err in enumerate(errors):
            store.log_tool_call(
                f"sess-{i}",
                "file_edit",
                result_success=False,
                error_message=err,
                duration_ms=120.0,
            )

        patterns = find_repeated_failures(store)
        assert len(patterns) == 1
        assert patterns[0]["tool_name"] == "file_edit"
        assert patterns[0]["count"] == 3

    def test_stale_content_signature_is_stable(self):
        """Different file paths and line numbers should normalize to the same
        signature, because the root cause is identical."""
        sig1 = _error_signature(
            "file_edit", "content mismatch in /home/alice/project/routes.py at line 42"
        )
        sig2 = _error_signature(
            "file_edit", "content mismatch in /home/bob/project/user.py at line 18"
        )
        assert sig1 == sig2

    def test_mixed_tools_not_conflated(self, store):
        """file_edit stale errors and bash permission errors should be
        separate patterns, even if both fail."""
        for i in range(2):
            store.log_tool_call(
                f"s{i}", "file_edit",
                result_success=False,
                error_message=f"content mismatch in /home/user/file{i}.py at line {i+1}",
            )
        for i in range(2):
            store.log_tool_call(
                f"s{i+10}", "bash",
                result_success=False,
                error_message=f"Permission denied: /etc/config{i}.yaml",
            )

        patterns = find_repeated_failures(store)
        assert len(patterns) == 2
        tools = {p["tool_name"] for p in patterns}
        assert tools == {"file_edit", "bash"}


# ===========================================================================
# Scenario 2: Agent deploys code, CI fails, same error repeats
# ===========================================================================


class TestCIFailurePatternLearning:
    """Real scenario: Agent runs `pytest` in bash, gets ImportError multiple
    times because it forgot to install a dependency."""

    def test_import_error_pattern(self, store):
        """Two test failures from different files → one pattern.

        The normalizer strips absolute paths, filenames (word.ext), and
        numbers.  We craft errors where only those parts differ."""
        store.log_tool_call(
            "deploy-1", "bash",
            params={"command": "pytest tests/"},
            result_success=False,
            error_message="FAILED /Users/alice/project/tests/test_auth.py::test_login - AssertionError: assert 200 == 401",
            duration_ms=3200.0,
        )
        store.log_tool_call(
            "deploy-2", "bash",
            params={"command": "pytest tests/unit/"},
            result_success=False,
            error_message="FAILED /Users/bob/project/tests/test_user.py::test_login - AssertionError: assert 200 == 403",
            duration_ms=2800.0,
        )

        patterns = find_repeated_failures(store)
        assert len(patterns) == 1
        assert patterns[0]["count"] == 2

    def test_success_between_failures_still_detects(self, store):
        """A success between two failures should not mask the pattern.

        Uses errors with absolute paths so the normalizer strips them
        to the same signature."""
        store.log_tool_call(
            "s1", "bash", result_success=False,
            error_message="FileNotFoundError: /home/alice/project/config.yaml not found",
        )
        store.log_tool_call(
            "s2", "bash", result_success=True,
            error_message="",
        )
        store.log_tool_call(
            "s3", "bash", result_success=False,
            error_message="FileNotFoundError: /home/bob/project/settings.yaml not found",
        )

        patterns = find_repeated_failures(store)
        assert len(patterns) >= 1

    def test_old_failures_outside_lookback_ignored(self, store):
        """Failures older than 7 days should not count."""
        # Insert old failure directly with a timestamp > 7 days ago
        old_ts = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        store.conn.execute(
            """INSERT INTO tool_call_log
               (session_id, tool_name, params, result_success, error_message, duration_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("old-sess", "bash", "{}", 0, "SyntaxError in main.py", 100.0, old_ts),
        )
        # Insert one recent failure (not enough alone)
        store.log_tool_call(
            "new-sess", "bash", result_success=False,
            error_message="SyntaxError in utils.py",
        )

        patterns = find_repeated_failures(store)
        assert len(patterns) == 0  # Only 1 recent, need 2


# ===========================================================================
# Scenario 3: Episode utility scoring for experience replay
# ===========================================================================


class TestEpisodeUtilityScoring:
    """Real scenario: Agent completes several tasks with varying outcomes.
    The memory system must accurately store and retrieve episodes by utility
    for future decision-making."""

    def test_successful_refactor_episode(self, store):
        """A successful multi-file refactor should be stored with utility=+1."""
        ep = _make_episode(
            task_summary="Refactor auth middleware to use async/await",
            intent="code_modify",
            result="success: 5 files modified, all tests pass",
            lessons=json.dumps(["async middleware needs explicit error boundaries"]),
            importance=0.8,
            utility=1,
            files_touched=json.dumps([
                "src/middleware/auth.py",
                "src/middleware/rate_limit.py",
                "tests/test_auth.py",
            ]),
            duration_ms=45000.0,
        )
        store.save_episode(ep)

        retrieved = store.get_recent_episodes(1)
        assert len(retrieved) == 1
        assert retrieved[0].utility == 1
        assert retrieved[0].importance == 0.8
        assert "auth.py" in retrieved[0].files_touched

    def test_failed_migration_episode(self, store):
        """A failed DB migration should be stored with utility=-1 and lesson."""
        ep = _make_episode(
            task_summary="Run database migration for user_profiles table",
            intent="code_modify",
            result="failure: column 'email' already exists",
            lessons=json.dumps([
                "Always check existing schema before ALTER TABLE",
                "Use IF NOT EXISTS for idempotent migrations",
            ]),
            importance=0.6,
            utility=-1,
            duration_ms=8000.0,
        )
        store.save_episode(ep)

        retrieved = store.get_recent_episodes(1)
        assert retrieved[0].utility == -1
        lessons = json.loads(retrieved[0].lessons)
        assert len(lessons) == 2

    def test_neutral_chat_episode(self, store):
        """A simple chat/question should be stored with utility=0."""
        ep = _make_episode(
            task_summary="Explain how the authentication flow works",
            intent="chat",
            result="explained auth flow with JWT + refresh tokens",
            utility=0,
            importance=0.3,
            duration_ms=2000.0,
        )
        store.save_episode(ep)

        retrieved = store.get_recent_episodes(1)
        assert retrieved[0].utility == 0

    def test_utility_distribution_across_session(self, store):
        """A real session has a mix of outcomes — verify all are persisted."""
        episodes = [
            _make_episode(task_summary="Read API docs", utility=0, importance=0.2),
            _make_episode(task_summary="Fix login bug", utility=1, importance=0.7),
            _make_episode(task_summary="Deploy to staging", utility=-1, importance=0.9),
            _make_episode(task_summary="Add unit tests", utility=1, importance=0.6),
            _make_episode(task_summary="Refactor CSS", utility=1, importance=0.4),
        ]
        for ep in episodes:
            store.save_episode(ep)

        all_eps = store.get_recent_episodes(10)
        utilities = [ep.utility for ep in all_eps]
        assert utilities.count(1) == 3
        assert utilities.count(-1) == 1
        assert utilities.count(0) == 1


# ===========================================================================
# Scenario 4: Episode-based file recall
# ===========================================================================


class TestFileBasedRecall:
    """Real scenario: Agent is asked to modify auth.py. The memory system
    should surface past episodes that touched the same file."""

    def test_recall_episodes_for_repeatedly_modified_file(self, store):
        """Three episodes touching auth.py → all three returned."""
        for i, summary in enumerate([
            "Fix JWT expiry check in auth.py",
            "Add rate limiting to auth.py endpoints",
            "Refactor auth.py to use dependency injection",
        ]):
            ep = _make_episode(
                task_summary=summary,
                files_touched=json.dumps(["src/middleware/auth.py"]),
                utility=1,
                conversation_id=f"sess-{i}",
            )
            store.save_episode(ep)

        # Unrelated episode
        store.save_episode(_make_episode(
            task_summary="Update README",
            files_touched=json.dumps(["README.md"]),
        ))

        results = store.get_episodes_by_file("auth.py")
        assert len(results) == 3
        assert all("auth.py" in ep.files_touched for ep in results)

    def test_recall_by_entity(self, store):
        """Episodes mentioning a specific entity (e.g. 'payment_gateway')
        should be retrievable."""
        store.save_episode(_make_episode(
            task_summary="Debug payment_gateway timeout",
            entities="payment_gateway,stripe",
            utility=-1,
        ))
        store.save_episode(_make_episode(
            task_summary="Add retry logic to payment_gateway",
            entities="payment_gateway",
            utility=1,
        ))
        store.save_episode(_make_episode(
            task_summary="Update user profile endpoint",
            entities="user_profile",
            utility=1,
        ))

        results = store.get_episodes_by_entity("payment_gateway")
        assert len(results) == 2


# ===========================================================================
# Scenario 5: Commitment tracking across sessions
# ===========================================================================


class TestCommitmentTracking:
    """Real scenario: Agent promises to add tests after a refactor.
    The system should track and surface unfulfilled commitments."""

    def test_commitment_lifecycle(self, store):
        """Create → open → resolve lifecycle."""
        ep = _make_episode(
            task_summary="Refactor payment module",
            commitments=json.dumps(["Add integration tests for payment module"]),
        )
        store.save_episode(ep)
        store.save_commitment(
            ep.id,
            "Add integration tests for payment module",
            deadline="2026-04-01",
        )

        open_commitments = store.get_open_commitments()
        assert len(open_commitments) == 1
        assert "integration tests" in open_commitments[0]["text"]
        assert open_commitments[0]["deadline"] == "2026-04-01"

        # Resolve
        store.resolve_commitment(open_commitments[0]["id"])
        assert len(store.get_open_commitments()) == 0

    def test_multiple_commitments_across_episodes(self, store):
        """Multiple commitments from different tasks should all be tracked."""
        for i, (summary, commitment) in enumerate([
            ("Add new API endpoint", "Write API documentation"),
            ("Fix memory leak", "Add memory profiling test"),
            ("Update dependencies", "Verify backward compatibility"),
        ]):
            ep = _make_episode(task_summary=summary, conversation_id=f"s{i}")
            store.save_episode(ep)
            store.save_commitment(ep.id, commitment)

        open_c = store.get_open_commitments()
        assert len(open_c) == 3


# ===========================================================================
# Scenario 6: Reflexion learning from real task patterns
# ===========================================================================


class TestReflexionRealPatterns:
    """Real scenario: Agent learns from successes and failures across
    different domains."""

    def test_timeout_lesson_extraction(self, learner):
        """Timeout failure should produce a meaningful lesson."""
        learner.record_task_outcome({
            "domain": "code_modify",
            "success": False,
            "goal": "Run full test suite with coverage",
            "error": "TimeoutError: command exceeded 120s limit",
            "steps_taken": 3,
            "duration_ms": 120000,
        })

        lessons = learner.get_domain_lessons("code_modify")
        assert len(lessons) == 1
        assert "timeout" in lessons[0].lower() or "limit" in lessons[0].lower()

    def test_permission_denied_lesson(self, learner):
        """Permission failure should extract access-related lesson."""
        learner.record_task_outcome({
            "domain": "code_modify",
            "success": False,
            "goal": "Modify system configuration",
            "error": "PermissionError: [Errno 13] Permission denied: '/etc/nginx/nginx.conf'",
            "steps_taken": 2,
        })

        lessons = learner.get_domain_lessons("code_modify")
        assert len(lessons) == 1
        assert "permission" in lessons[0].lower() or "access" in lessons[0].lower()

    def test_inefficient_completion_lesson(self, learner):
        """Task that succeeds but takes too many steps should suggest
        a more direct approach."""
        learner.record_task_outcome({
            "domain": "code_modify",
            "success": True,
            "goal": "Fix typo in error message",
            "steps_taken": 15,  # Way too many for a typo fix
            "duration_ms": 60000,
        })

        lessons = learner.get_domain_lessons("code_modify")
        assert len(lessons) == 1
        assert "direct" in lessons[0].lower() or "15 steps" in lessons[0]

    def test_efficient_completion_lesson(self, learner):
        """Quick 1-step success should be noted as efficient."""
        learner.record_task_outcome({
            "domain": "quick_fix",
            "success": True,
            "goal": "Update version number in pyproject.toml",
            "steps_taken": 1,
            "duration_ms": 800,
        })

        lessons = learner.get_domain_lessons("quick_fix")
        assert len(lessons) == 1
        assert "efficient" in lessons[0].lower()

    def test_cross_domain_lesson_isolation(self, learner):
        """Lessons from different domains should not leak into each other."""
        learner.record_task_outcome({
            "domain": "code_modify",
            "success": False,
            "goal": "Refactor auth",
            "error": "TimeoutError: exceeded limit",
            "steps_taken": 5,
        })
        learner.record_task_outcome({
            "domain": "research",
            "success": False,
            "goal": "Find API documentation",
            "error": "Resource not found: /api/v2/docs",
            "steps_taken": 8,
        })

        code_lessons = learner.get_domain_lessons("code_modify")
        research_lessons = learner.get_domain_lessons("research")
        assert len(code_lessons) == 1
        assert len(research_lessons) == 1
        assert "timeout" in code_lessons[0].lower()
        assert "not found" in research_lessons[0].lower()

    def test_success_rate_tracking(self, learner):
        """Domain success rate should reflect actual outcomes."""
        # 3 successes, 2 failures in code_modify
        for success in [True, True, False, True, False]:
            learner.record_task_outcome({
                "domain": "code_modify",
                "success": success,
                "goal": "task",
                "steps_taken": 3,
            })

        rate = learner.get_domain_success_rate("code_modify")
        assert rate == pytest.approx(0.6, abs=0.01)

    def test_unknown_domain_defaults_to_100_percent(self, learner):
        """Domain with no data should return 1.0 (optimistic default)."""
        assert learner.get_domain_success_rate("never_seen") == 1.0

    def test_duplicate_lessons_not_stored(self, learner):
        """Same lesson should not be stored twice."""
        for _ in range(3):
            learner.record_task_outcome({
                "domain": "code_modify",
                "success": False,
                "goal": "Run tests",
                "error": "TimeoutError: exceeded 120s",
                "steps_taken": 1,
            })

        lessons = learner.get_domain_lessons("code_modify")
        assert len(lessons) == 1  # Deduplicated


# ===========================================================================
# Scenario 7: Proactive suggestion rejection → strategy adaptation
# ===========================================================================


class TestRejectionAdaptation:
    """Real scenario: User is deep in coding flow. Agent keeps suggesting
    things. User rejects 5 times → agent backs off automatically."""

    def test_consecutive_rejections_raise_threshold(self, learner):
        """5 rejections in 30 minutes should increase min_score_threshold."""
        initial_threshold = learner.get_score_threshold()

        for i in range(5):
            learner.record_rejection(
                event_type="proactive_reminder",
                suggestion_type="optimization",
                score=0.6,
                reason="too_frequent",
                context_summary=f"user editing file iteration {i}",
            )

        new_threshold = learner.get_score_threshold()
        assert new_threshold is not None
        assert new_threshold > (initial_threshold or 0.55)

    def test_consecutive_rejections_increase_interval(self, learner):
        """5 rejections should also increase the min intervention interval."""
        initial_interval = learner.get_min_interval()

        for _ in range(5):
            learner.record_rejection(
                event_type="proactive_reminder",
                suggestion_type="test_suggestion",
                score=0.5,
                reason="user_busy",
            )

        new_interval = learner.get_min_interval()
        assert new_interval > initial_interval

    def test_varied_rejections_below_threshold_no_change(self, learner):
        """4 rejections (below threshold of 5) should NOT trigger adaptation."""
        for _ in range(4):
            learner.record_rejection(
                event_type="proactive_reminder",
                suggestion_type="optimization",
                score=0.6,
            )

        # Strategy should remain at defaults
        assert learner.get_score_threshold() is None

    def test_rejection_stats_aggregate_correctly(self, learner):
        """Stats should accurately reflect rejection history."""
        learner.record_rejection("reminder", "opt", 0.6, reason="too_frequent")
        learner.record_rejection("reminder", "opt", 0.5, reason="bad_timing")
        learner.record_rejection("code_review", "quality", 0.7, reason="irrelevant")

        stats = learner.get_stats()
        assert stats["total_rejections"] == 3
        assert any(r["reason"] == "too_frequent" for r in stats["top_reasons"])
        assert any(r["reason"] == "bad_timing" for r in stats["top_reasons"])

    def test_rule_based_rejection_analysis(self, learner):
        """Rule-based fallback should correctly classify rejection text."""
        result = learner._rule_based_analysis(
            "I'm busy, not now please",
            {"domain": "code_modify", "event_type": "reminder"},
        )
        assert result["reason"] == "bad_timing"

        result2 = learner._rule_based_analysis(
            "Stop suggesting things, too many interruptions",
            {"domain": "code_modify"},
        )
        assert result2["reason"] == "too_frequent"

        result3 = learner._rule_based_analysis(
            "This suggestion is useless and obvious",
            {},
        )
        assert result3["reason"] == "not_helpful"

    def test_reset_clears_all_state(self, learner):
        """Reset should return the learner to a clean slate."""
        for _ in range(5):
            learner.record_rejection("x", "y", 0.5, reason="too_frequent")
        learner.record_task_outcome({
            "domain": "d", "success": True, "goal": "g", "steps_taken": 1,
        })

        learner.reset()

        assert learner.get_score_threshold() is None
        assert learner.get_min_interval() == 120.0
        assert learner.get_domain_lessons("d") == []
        assert learner.get_domain_success_rate("d") == 1.0
        assert learner.get_stats()["total_rejections"] == 0


# ===========================================================================
# Scenario 8: Error signature normalization for real-world errors
# ===========================================================================


class TestErrorSignatureRealErrors:
    """Validate signature normalization against realistic error messages
    that an AI agent would encounter in production."""

    def test_python_traceback_normalization(self):
        """Different Python tracebacks for the same error type should match."""
        sig1 = _error_signature("bash", (
            "Traceback (most recent call last):\n"
            "  File '/Users/alice/project/src/api/server.py', line 42\n"
            "    raise ValueError('invalid token')\n"
            "ValueError: invalid token"
        ))
        sig2 = _error_signature("bash", (
            "Traceback (most recent call last):\n"
            "  File '/Users/bob/work/src/api/server.py', line 88\n"
            "    raise ValueError('invalid token')\n"
            "ValueError: invalid token"
        ))
        assert sig1 == sig2

    def test_npm_error_normalization(self):
        """npm errors with different package versions should normalize."""
        sig1 = _error_signature("bash",
            "npm ERR! 404 Not Found - GET https://registry.npmjs.org/@scope/pkg-1.2.3"
        )
        sig2 = _error_signature("bash",
            "npm ERR! 404 Not Found - GET https://registry.npmjs.org/@scope/pkg-4.5.6"
        )
        assert sig1 == sig2

    def test_numeric_id_normalization(self):
        """Errors with different numeric IDs should normalize to the same sig."""
        sig1 = _error_signature("bash",
            "Error: process 12345 exited with code 1"
        )
        sig2 = _error_signature("bash",
            "Error: process 67890 exited with code 2"
        )
        assert sig1 == sig2

    def test_completely_different_errors_diverge(self):
        """Fundamentally different errors should NOT normalize to the same sig."""
        sig_syntax = _error_signature("bash", "SyntaxError: unexpected EOF")
        sig_perm = _error_signature("bash", "PermissionError: access denied")
        sig_conn = _error_signature("bash", "ConnectionRefusedError: port 5432")
        assert len({sig_syntax, sig_perm, sig_conn}) == 3


# ===========================================================================
# Scenario 9: Episode ranked retrieval for context injection
# ===========================================================================


class TestEpisodeRankedRetrieval:
    """Real scenario: Agent is about to work on a task. The system retrieves
    the most relevant past episodes to inject into the prompt."""

    def test_recent_high_importance_ranked_first(self, store):
        """Recent + important episodes should outrank old ones."""
        # Old episode (2 weeks ago)
        old_ts = (datetime.now(UTC) - timedelta(days=14)).isoformat()
        store.save_episode(_make_episode(
            task_summary="Fix authentication bug",
            importance=0.9,
            utility=1,
            timestamp=old_ts,
        ))

        # Recent episode (1 hour ago)
        recent_ts = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        store.save_episode(_make_episode(
            task_summary="Fix authentication token refresh",
            importance=0.7,
            utility=1,
            timestamp=recent_ts,
        ))

        results = store.get_ranked_episodes(query="authentication", limit=2)
        assert len(results) == 2
        # Recent + relevant should rank higher despite lower importance
        assert "refresh" in results[0].task_summary

    def test_query_relevance_dominates_ranking(self, store):
        """When query is specific, relevance should outweigh recency."""
        now = datetime.now(UTC)
        # Recent but irrelevant
        store.save_episode(_make_episode(
            task_summary="Update CSS styling for header",
            importance=0.5,
            timestamp=(now - timedelta(minutes=30)).isoformat(),
        ))
        # Older but highly relevant
        store.save_episode(_make_episode(
            task_summary="Fix Redis connection pool memory leak",
            importance=0.8,
            timestamp=(now - timedelta(days=3)).isoformat(),
        ))

        results = store.get_ranked_episodes(query="Redis connection pool", limit=2)
        assert "Redis" in results[0].task_summary

    def test_empty_query_uses_recency_and_importance(self, store):
        """Without a query, ranking should favor recency + importance."""
        now = datetime.now(UTC)
        store.save_episode(_make_episode(
            task_summary="Low importance old task",
            importance=0.1,
            timestamp=(now - timedelta(days=7)).isoformat(),
        ))
        store.save_episode(_make_episode(
            task_summary="High importance recent task",
            importance=0.9,
            timestamp=(now - timedelta(minutes=5)).isoformat(),
        ))

        results = store.get_ranked_episodes(query="", limit=2)
        assert results[0].importance == 0.9

    def test_failed_episodes_not_excluded(self, store):
        """Failed episodes (utility=-1) should still be retrievable —
        they contain valuable lessons."""
        store.save_episode(_make_episode(
            task_summary="Deploy to production failed",
            utility=-1,
            importance=0.9,
            lessons=json.dumps(["Always run smoke tests before deploy"]),
        ))

        results = store.get_ranked_episodes(query="deploy production", limit=5)
        assert len(results) == 1
        assert results[0].utility == -1


# ===========================================================================
# Scenario 10: Full learning pipeline (tool calls → patterns → episodes)
# ===========================================================================


class TestFullLearningPipeline:
    """Integration test: simulate a multi-step agent session where
    failures occur, episodes are recorded, and patterns are detected."""

    def test_debugging_session_workflow(self, store):
        """Simulate: Agent tries to fix a bug, fails twice with same error
        pattern, then succeeds on third attempt."""

        # Attempt 1: Read file, edit fails
        store.log_tool_call("debug-1", "file_read", result_success=True,
                           params={"path": "src/api/handler.py"})
        store.log_tool_call("debug-1", "file_edit", result_success=False,
                           error_message="SyntaxError: unexpected indent at line 45",
                           duration_ms=150.0)

        # Attempt 2: Same error, different location
        store.log_tool_call("debug-1", "file_read", result_success=True,
                           params={"path": "src/api/handler.py"})
        store.log_tool_call("debug-1", "file_edit", result_success=False,
                           error_message="SyntaxError: unexpected indent at line 52",
                           duration_ms=130.0)

        # Attempt 3: Success after rethinking
        store.log_tool_call("debug-1", "file_read", result_success=True)
        store.log_tool_call("debug-1", "file_edit", result_success=True,
                           duration_ms=200.0)

        # Failure pattern should be detected
        patterns = find_repeated_failures(store)
        assert len(patterns) == 1
        assert patterns[0]["tool_name"] == "file_edit"
        assert "SyntaxError" in patterns[0]["error_sample"]

        # Save episode with lesson learned
        ep = _make_episode(
            task_summary="Fix handler.py indentation bug",
            result="success after 3 attempts",
            lessons=json.dumps([
                "Verify indentation matches surrounding code before file_edit",
                "Use AST validation before applying edits",
            ]),
            utility=1,
            importance=0.7,
            files_touched=json.dumps(["src/api/handler.py"]),
            duration_ms=15000.0,
        )
        store.save_episode(ep)

        # Verify the episode is queryable
        file_eps = store.get_episodes_by_file("handler.py")
        assert len(file_eps) == 1
        assert file_eps[0].utility == 1

    def test_multi_tool_session_with_mixed_outcomes(self, store):
        """Simulate a realistic session: research → code → test → deploy."""

        # Research phase
        store.log_tool_call("sess-1", "web_search", result_success=True,
                           params={"query": "FastAPI middleware best practices"})
        store.log_tool_call("sess-1", "file_read", result_success=True,
                           params={"path": "src/middleware/__init__.py"})

        # Code phase
        store.log_tool_call("sess-1", "file_edit", result_success=True,
                           params={"path": "src/middleware/cors.py"})
        store.log_tool_call("sess-1", "file_edit", result_success=True,
                           params={"path": "src/middleware/logging.py"})

        # Test phase — first run fails
        store.log_tool_call("sess-1", "bash", result_success=False,
                           error_message="FAILED tests/test_middleware.py::test_cors - AssertionError",
                           params={"command": "pytest tests/test_middleware.py"})

        # Fix and re-test
        store.log_tool_call("sess-1", "file_edit", result_success=True,
                           params={"path": "src/middleware/cors.py"})
        store.log_tool_call("sess-1", "bash", result_success=True,
                           params={"command": "pytest tests/test_middleware.py"})

        # Verify tool call log integrity
        recent = store.get_recent_tool_calls(20)
        assert len(recent) == 7
        fail_calls = [c for c in recent if not c["result_success"]]
        assert len(fail_calls) == 1

        # Only 1 failure → no pattern detected (needs 2)
        patterns = find_repeated_failures(store)
        assert len(patterns) == 0

    def test_tool_call_params_preserved(self, store):
        """Tool call parameters should be stored and retrievable for
        debugging and replay."""
        store.log_tool_call(
            "sess-1", "bash",
            params={"command": "git diff --cached", "timeout": 30000},
            result_success=True,
            duration_ms=450.0,
        )

        calls = store.get_recent_tool_calls(1)
        assert len(calls) == 1
        # get_recent_tool_calls returns params already deserialized as dict
        params = calls[0]["params"]
        assert isinstance(params, dict)
        assert params["command"] == "git diff --cached"
        assert params["timeout"] == 30000


# ===========================================================================
# Scenario 11: Reflexion learner stats over realistic workload
# ===========================================================================


class TestReflexionWorkloadStats:
    """Simulate a realistic day's worth of agent interactions and verify
    the statistical integrity of the reflexion system."""

    def test_full_day_workload(self, learner):
        """A day with 20 tasks across 3 domains."""
        # Morning: code tasks
        for i in range(8):
            learner.record_task_outcome({
                "domain": "code_modify",
                "success": i % 3 != 0,  # 66% success
                "goal": f"Code task {i}",
                "error": "TimeoutError: slow" if i % 3 == 0 else None,
                "steps_taken": i + 2,
            })

        # Afternoon: research tasks
        for i in range(6):
            learner.record_task_outcome({
                "domain": "research",
                "success": True,  # 100% success
                "goal": f"Research task {i}",
                "steps_taken": i + 1,
            })

        # Evening: debugging
        for i in range(6):
            learner.record_task_outcome({
                "domain": "debug",
                "success": i > 2,  # 50% success
                "goal": f"Debug task {i}",
                "error": "PermissionError: /var/log" if i <= 2 else None,
                "steps_taken": i + 3,
            })

        # Verify rates
        assert learner.get_domain_success_rate("research") == 1.0
        assert 0.5 < learner.get_domain_success_rate("code_modify") < 0.8
        assert learner.get_domain_success_rate("debug") == 0.5

        # Verify lessons extracted
        code_lessons = learner.get_domain_lessons("code_modify")
        debug_lessons = learner.get_domain_lessons("debug")
        assert len(code_lessons) > 0  # Timeout lessons
        assert len(debug_lessons) > 0  # Permission lessons

        # Verify stats integrity
        stats = learner.get_stats()
        total = sum(
            s["success"] + s["failure"]
            for s in stats["domain_stats"].values()
        )
        assert total == 20


# ===========================================================================
# Scenario 12: Episode time-range queries
# ===========================================================================


class TestEpisodeTimeRange:
    """Real scenario: 'What did we work on last week?' queries."""

    def test_query_episodes_by_date_range(self, store):
        """Episodes within a date range should be returned."""
        now = datetime.now(UTC)

        # Last week's episodes
        for i in range(3):
            ts = (now - timedelta(days=5, hours=i)).isoformat()
            store.save_episode(_make_episode(
                task_summary=f"Last week task {i}",
                timestamp=ts,
            ))

        # This week's episodes
        for i in range(2):
            ts = (now - timedelta(hours=i + 1)).isoformat()
            store.save_episode(_make_episode(
                task_summary=f"This week task {i}",
                timestamp=ts,
            ))

        # Query last week only
        start = (now - timedelta(days=7)).isoformat()
        end = (now - timedelta(days=3)).isoformat()
        results = store.get_episodes_by_timerange(start, end)
        assert len(results) == 3
        assert all("Last week" in ep.task_summary for ep in results)
