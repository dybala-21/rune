#!/usr/bin/env python3
"""RUNE E2E Test Harness — zero side-effect, full pipeline testing.

Runs multi-turn scenarios through the full RUNE pipeline including
self-improving subsystems (rule learner, memory promotion, reflexion,
autonomy). All state is isolated to a temp directory.

Usage:
    # Inline turns
    uv run python scripts/e2e_test.py \\
        --provider ollama --model gemma4:26b \\
        --turn "List all files in the current directory" \\
        --turn "Filter only .py files from the previous result"

    # YAML scenario
    uv run python scripts/e2e_test.py \\
        --scenario tests/e2e/scenarios/self_improving.yaml

    # All scenarios in a directory
    uv run python scripts/e2e_test.py \\
        --suite tests/e2e/scenarios/

    # JSON output for automation
    uv run python scripts/e2e_test.py --scenario ... --format json
"""

from __future__ import annotations

# ISOLATION: must happen before any rune import
import os
import shutil
import sys
import tempfile
from pathlib import Path

TMPDIR = tempfile.mkdtemp(prefix="rune-e2e-")
REAL_HOME = str(Path.home())

# Workspace must be under $HOME, not /var/folders (Guardian blocks /var writes).
_WORKSPACE = Path(REAL_HOME) / ".rune-e2e-workspace"
_WORKSPACE.mkdir(parents=True, exist_ok=True)

# Symlink embedding models to avoid re-download
_real_models = Path(REAL_HOME) / ".rune" / "data" / "models"
if _real_models.exists():
    _dst = Path(TMPDIR) / ".rune" / "data" / "models"
    _dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(_real_models, _dst)

# Copy config.yaml and .env for LLM provider settings + API keys
_cfg_dst = Path(TMPDIR) / ".rune"
_cfg_dst.mkdir(parents=True, exist_ok=True)
_real_config = Path(REAL_HOME) / ".rune" / "config.yaml"
if _real_config.exists():
    shutil.copy2(_real_config, _cfg_dst / "config.yaml")
_real_env = Path(REAL_HOME) / ".rune" / ".env"
if _real_env.exists():
    shutil.copy2(_real_env, _cfg_dst / ".env")

# Redirect all rune paths to tmpdir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rune.utils.paths as _paths  # noqa: E402

_paths._HOME = Path(TMPDIR)
# END ISOLATION

import argparse  # noqa: E402
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from typing import Any  # noqa: E402

# Result types

@dataclass
class VerifyResult:
    type: str
    passed: bool
    expected: Any = None
    actual: Any = None
    detail: str = ""


@dataclass
class TurnResult:
    turn: int
    goal: str
    trace_reason: str = ""
    steps: int = 0
    tokens: int = 0
    duration_ms: float = 0.0
    answer_preview: str = ""
    verifications: list[VerifyResult] = field(default_factory=list)


@dataclass
class SelfImprovingReport:
    episodes_saved: int = 0
    episode_utilities: list[int] = field(default_factory=list)
    rules_generated: int = 0
    rule_candidates: list[str] = field(default_factory=list)
    memory_promoted: bool = False
    daily_summary_exists: bool = False
    reflexion_lessons: dict[str, list[str]] = field(default_factory=dict)
    reflexion_domain_rates: dict[str, float] = field(default_factory=dict)
    autonomy_patterns: dict[str, dict[str, Any]] = field(default_factory=dict)
    conversation_turns: int = 0


@dataclass
class E2EReport:
    scenario: str = ""
    model: str = ""
    provider: str = ""
    isolated_dir: str = ""
    turns: list[TurnResult] = field(default_factory=list)
    self_improving: SelfImprovingReport = field(default_factory=SelfImprovingReport)
    after_all: list[VerifyResult] = field(default_factory=list)
    error: str | None = None
    duration_s: float = 0.0


# Harness

class E2EHarness:
    """Full-pipeline E2E test harness with complete isolation."""

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        self.tmpdir = TMPDIR
        self.workspace = str(_WORKSPACE)
        self._loop: Any = None
        self._conv_manager: Any = None
        self._conv_id: str = ""
        self._collected_text: list[str] = []
        self._last_answer: str = ""
        self._reflexion: Any = None
        self._autonomy: Any = None

    async def setup(self) -> None:
        """Initialize all subsystems pointing at tmpdir."""
        from rune.agent.loop import NativeAgentLoop
        from rune.conversation.manager import ConversationManager
        from rune.conversation.store import ConversationStore
        from rune.types import AgentConfig

        # Override config to use the specified provider/model for ALL tiers
        # (goal classifier, rule learner, etc. — not just the main loop)
        try:
            from rune.config import get_config

            cfg = get_config()
            cfg.llm.active_provider = self.provider
            cfg.llm.active_model = self.model
            cfg.llm.default_provider = self.provider
        except Exception:
            pass

        # Agent loop
        config = AgentConfig()
        config.model = self.model
        config.provider = self.provider
        self._loop = NativeAgentLoop(config=config)

        # Wire text collection
        async def _on_text(delta: str) -> None:
            self._collected_text.append(delta)

        self._loop.on("text_delta", _on_text)

        # Auto-approve all tool calls (non-interactive)
        async def _auto_approve(_cap: str, _reason: str) -> bool:
            return True

        self._loop.set_approval_callback(_auto_approve)

        # Conversation manager (tmpdir DB)
        conv_db = os.path.join(self.tmpdir, ".rune", "data", "conversations.db")
        os.makedirs(os.path.dirname(conv_db), exist_ok=True)
        conv_store = ConversationStore(conv_db)
        self._conv_manager = ConversationManager(conv_store)
        conv = self._conv_manager.start_conversation(user_id="e2e-test")
        self._conv_id = conv.id

        # Reflexion learner
        from rune.proactive.reflexion import ReflexionLearner

        self._reflexion = ReflexionLearner()

        # Autonomy executor
        from rune.agent.autonomous import AutonomousExecutor

        self._autonomy = AutonomousExecutor()

        # Ensure memory store is initialized
        from rune.memory.store import get_memory_store

        get_memory_store()

    async def run_turn(self, goal: str, turn_num: int) -> TurnResult:
        """Run a single turn through the full pipeline."""
        from rune.agent.agent_context import (
            PostProcessInput,
            PrepareContextOptions,
            post_process_agent_result,
            prepare_agent_context,
        )

        self._collected_text.clear()
        t0 = time.monotonic()

        # Resolve {workspace} template in goal
        goal = goal.replace("{workspace}", self.workspace)
        result = TurnResult(turn=turn_num, goal=goal)

        try:
            # Record user turn
            with contextlib.suppress(Exception):
                self._conv_manager.add_turn(self._conv_id, "user", goal)

            # Prepare context (memory search, conversation history)
            ctx = await prepare_agent_context(
                PrepareContextOptions(
                    goal=goal,
                    channel="e2e-test",
                    conversation_id=self._conv_id,
                    cwd=self.workspace,
                ),
                conversation_manager=self._conv_manager,
            )

            # Build memory context
            run_context: dict[str, Any] = {
                "workspace_root": ctx.workspace_root or self.workspace,
            }
            try:
                from rune.memory.manager import get_memory_manager

                mgr = get_memory_manager()
                mem_ctx = await mgr.build_memory_context(goal)
                if mem_ctx:
                    run_context["memory_context"] = mem_ctx
            except Exception:
                pass

            # Run agent loop with multi-turn history
            trace = await self._loop.run(
                ctx.goal,
                context=run_context,
                message_history=ctx.messages if ctx.messages else None,
            )

            answer = "".join(self._collected_text)
            self._last_answer = answer
            duration_ms = (time.monotonic() - t0) * 1000

            result.trace_reason = trace.reason or "unknown"
            result.steps = trace.final_step or 0
            result.tokens = getattr(trace, "total_tokens_used", 0)
            result.duration_ms = round(duration_ms, 1)
            result.answer_preview = answer[:200]

            # Record assistant turn
            with contextlib.suppress(Exception):
                self._conv_manager.add_turn(
                    self._conv_id,
                    "assistant",
                    answer,
                    goal_type=getattr(self._loop, "_last_goal_type", ""),
                )

            # Post-process: episode saving, memory bridge
            with contextlib.suppress(Exception):
                await post_process_agent_result(
                    PostProcessInput(
                        context=ctx,
                        success=(trace.reason == "completed"),
                        answer=answer,
                        duration_ms=duration_ms,
                    )
                )

            # Reflexion: record task outcome
            with contextlib.suppress(Exception):
                self._reflexion.record_task_outcome(
                    {
                        "domain": getattr(self._loop, "_last_goal_type", "general"),
                        "success": trace.reason == "completed",
                        "goal": goal,
                        "error": trace.reason if trace.reason != "completed" else None,
                        "steps_taken": trace.final_step or 0,
                        "duration_ms": duration_ms,
                    }
                )

            # Autonomy: simulate feedback for file operations
            with contextlib.suppress(Exception):
                decision = self._autonomy.decide(goal)
                self._autonomy.record_feedback(
                    decision.pattern_key,
                    "approved" if trace.reason == "completed" else "manual_correction",
                    risk_score=decision.risk_score,
                )

        except Exception as exc:
            result.trace_reason = f"error: {exc}"

        return result

    async def run_self_improving_cycle(self) -> SelfImprovingReport:
        """Run all self-improving subsystems and collect state."""
        report = SelfImprovingReport()

        # 1. Episode count and utilities
        try:
            from rune.memory.store import get_memory_store

            store = get_memory_store()
            episodes = store.get_recent_episodes(limit=100)
            report.episodes_saved = len(episodes)
            report.episode_utilities = [
                getattr(ep, "utility", 0) for ep in episodes
            ]
        except Exception:
            pass

        # 2. Rule Learner: find repeated failures
        try:
            from rune.memory.rule_learner import find_repeated_failures
            from rune.memory.store import get_memory_store

            store = get_memory_store()
            repeated = find_repeated_failures(store)
            report.rules_generated = len(repeated)
            report.rule_candidates = [
                f"{r['tool_name']}: {r['error_sample'][:80]}" for r in repeated
            ]
        except Exception:
            pass

        # 3. Memory promotion
        try:
            from rune.memory.manager import get_memory_manager

            mgr = get_memory_manager()
            await mgr.promote_memories()
            report.memory_promoted = True

            # Check daily summary
            from datetime import UTC, datetime

            today = datetime.now(UTC).strftime("%Y-%m-%d")
            daily = mgr._tiered.get_daily_summary(today)
            report.daily_summary_exists = daily is not None
        except Exception:
            pass

        # 4. Reflexion state
        if self._reflexion:
            for domain in list(self._reflexion._domain_stats.keys()):
                report.reflexion_domain_rates[domain] = (
                    self._reflexion.get_domain_success_rate(domain)
                )
                report.reflexion_lessons[domain] = (
                    self._reflexion.get_domain_lessons(domain)
                )

        # 5. Autonomy patterns
        if self._autonomy:
            for key, stats in self._autonomy.pattern_stats.items():
                report.autonomy_patterns[key] = {
                    "total": stats.total_executions,
                    "approved": stats.approved,
                    "reverted": stats.reverted,
                    "avg_risk": round(stats.avg_risk, 3),
                }

        # 6. Conversation turns (in-memory, not persisted to DB)
        try:
            conv = self._conv_manager._active.get(self._conv_id)
            report.conversation_turns = len(conv.turns) if conv else 0
        except Exception:
            pass

        return report

    def verify(self, checks: list[dict[str, Any]]) -> list[VerifyResult]:
        """Run verification checks against current state."""
        results: list[VerifyResult] = []

        for check in checks:
            vtype = check.get("type", "")
            try:
                if vtype == "episode_count":
                    from rune.memory.store import get_memory_store

                    store = get_memory_store()
                    actual = len(store.get_recent_episodes(limit=1000))
                    expected = check.get("expect_min", 1)
                    results.append(
                        VerifyResult(
                            type=vtype,
                            passed=actual >= expected,
                            expected=expected,
                            actual=actual,
                        )
                    )

                elif vtype == "conversation_turns":
                    conv = self._conv_manager._active.get(self._conv_id)
                    actual = len(conv.turns) if conv else 0
                    expected = check.get("expect_min", 1)
                    results.append(
                        VerifyResult(
                            type=vtype,
                            passed=actual >= expected,
                            expected=expected,
                            actual=actual,
                        )
                    )

                elif vtype == "reflexion_domain_rate":
                    domain = check.get("domain", "general")
                    actual = self._reflexion.get_domain_success_rate(domain)
                    expect_min = check.get("expect_min", 0.0)
                    expect_max = check.get("expect_max", 1.0)
                    results.append(
                        VerifyResult(
                            type=vtype,
                            passed=expect_min <= actual <= expect_max,
                            expected=f"[{expect_min}, {expect_max}]",
                            actual=round(actual, 3),
                            detail=f"domain={domain}",
                        )
                    )

                elif vtype == "autonomy_promoted":
                    pattern = check.get("pattern", "")
                    stats = self._autonomy.pattern_stats.get(pattern)
                    if stats:
                        results.append(
                            VerifyResult(
                                type=vtype,
                                passed=stats.approved >= check.get("expect_min", 1),
                                expected=check.get("expect_min", 1),
                                actual=stats.approved,
                            )
                        )
                    else:
                        results.append(
                            VerifyResult(
                                type=vtype,
                                passed=False,
                                detail=f"pattern '{pattern}' not found",
                            )
                        )

                elif vtype == "fact_meta_exists":
                    from rune.memory.state import load_fact_meta

                    meta = load_fact_meta()
                    results.append(
                        VerifyResult(
                            type=vtype,
                            passed=len(meta) > 0,
                            actual=len(meta),
                        )
                    )

                elif vtype == "file_exists":
                    path = check.get("path", "")
                    full = os.path.join(self.tmpdir, "workspace", path)
                    exists = os.path.exists(full)
                    results.append(
                        VerifyResult(
                            type=vtype, passed=exists, detail=path
                        )
                    )

                elif vtype == "answer_contains":
                    expect = check.get("expect", "")
                    found = expect in self._last_answer
                    results.append(
                        VerifyResult(
                            type=vtype,
                            passed=found,
                            expected=expect,
                            actual=self._last_answer[:100],
                        )
                    )

                elif vtype == "answer_not_contains":
                    expect = check.get("expect", "")
                    found = expect in self._last_answer
                    results.append(
                        VerifyResult(
                            type=vtype,
                            passed=not found,
                            expected=f"NOT '{expect}'",
                            actual=self._last_answer[:100],
                        )
                    )

                elif vtype == "file_contains":
                    path = check.get("path", "").replace("{workspace}", self.workspace)
                    expect = check.get("expect", "")
                    content = ""
                    try:
                        content = open(path, encoding="utf-8").read()
                    except FileNotFoundError:
                        pass
                    found = expect in content
                    results.append(
                        VerifyResult(
                            type=vtype,
                            passed=found,
                            expected=expect,
                            actual=content[:100],
                            detail=path,
                        )
                    )

                else:
                    results.append(
                        VerifyResult(
                            type=vtype,
                            passed=False,
                            detail=f"unknown verify type: {vtype}",
                        )
                    )
            except Exception as exc:
                results.append(
                    VerifyResult(
                        type=vtype, passed=False, detail=f"error: {exc}"
                    )
                )

        return results

    def cleanup(self) -> None:
        """Remove all temp data."""
        # Reset singletons so they don't leak into subsequent runs
        import rune.memory.manager as _mm
        import rune.memory.store as _ms

        _mm._manager = None
        _ms._store = None

        shutil.rmtree(self.tmpdir, ignore_errors=True)
        # Clean workspace (under $HOME, separate from tmpdir)
        shutil.rmtree(self.workspace, ignore_errors=True)


# Scenario loading

def load_scenario(path: str) -> dict[str, Any]:
    """Load a YAML scenario file."""
    import yaml  # type: ignore[import-untyped]

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_scenario_json(path: str) -> dict[str, Any]:
    """Load a JSON scenario file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Main runner

async def run_scenario(
    scenario: dict[str, Any],
    provider: str,
    model: str,
) -> E2EReport:
    """Execute a complete E2E scenario."""
    t0 = time.monotonic()
    harness = E2EHarness(
        provider=provider or scenario.get("provider", "ollama"),
        model=model or scenario.get("model", "gemma4:26b"),
    )
    report = E2EReport(
        scenario=scenario.get("name", "inline"),
        model=harness.model,
        provider=harness.provider,
        isolated_dir=harness.tmpdir,
    )

    # Create workspace dir for file operations (under $HOME to avoid Guardian blocks)
    os.makedirs(harness.workspace, exist_ok=True)

    try:
        await harness.setup()

        # Run turns
        turns = scenario.get("turns", [])
        for i, turn_spec in enumerate(turns, 1):
            goal = turn_spec if isinstance(turn_spec, str) else turn_spec.get("goal", "")
            turn_id = turn_spec.get("id", f"turn_{i}") if isinstance(turn_spec, dict) else f"turn_{i}"
            print(f"  [{i}/{len(turns)}] ({turn_id}) {goal[:60]}...", file=sys.stderr, flush=True)

            turn_result = await harness.run_turn(goal, i)

            # Per-turn verification
            if isinstance(turn_spec, dict) and "verify" in turn_spec:
                turn_result.verifications = harness.verify(turn_spec["verify"])

            report.turns.append(turn_result)
            print(
                f"         → {turn_result.trace_reason} "
                f"(steps={turn_result.steps}, tokens={turn_result.tokens})",
                file=sys.stderr,
                flush=True,
            )

        # Self-improving cycle
        report.self_improving = await harness.run_self_improving_cycle()

        # After-all verification
        if "after_all" in scenario:
            report.after_all = harness.verify(scenario["after_all"])

    except Exception as exc:
        report.error = str(exc)
    finally:
        report.duration_s = round(time.monotonic() - t0, 2)
        harness.cleanup()

    return report


def format_report(report: E2EReport, fmt: str) -> str:
    """Format the report for output."""
    if fmt == "json":
        return json.dumps(asdict(report), indent=2, ensure_ascii=False, default=str)

    # Human-readable
    lines: list[str] = []
    lines.append(f"═══ E2E Report: {report.scenario} ═══")
    lines.append(f"Model: {report.provider}/{report.model}")
    lines.append(f"Duration: {report.duration_s}s")
    lines.append(f"Isolated dir: {report.isolated_dir}")
    if report.error:
        lines.append(f"ERROR: {report.error}")
    lines.append("")

    # Turns
    lines.append("── Turns ──")
    for t in report.turns:
        status = "✓" if t.trace_reason == "completed" else "✗"
        lines.append(
            f"  {status} Turn {t.turn}: {t.goal[:50]}... "
            f"→ {t.trace_reason} (steps={t.steps}, tokens={t.tokens})"
        )
        for v in t.verifications:
            vs = "✓" if v.passed else "✗"
            lines.append(f"    {vs} {v.type}: expected={v.expected}, actual={v.actual}")

    # Self-improving
    si = report.self_improving
    lines.append("")
    lines.append("── Self-Improving ──")
    lines.append(f"  Episodes saved: {si.episodes_saved}")
    lines.append(f"  Utilities: {si.episode_utilities}")
    lines.append(f"  Rule candidates: {si.rules_generated}")
    for r in si.rule_candidates:
        lines.append(f"    - {r}")
    lines.append(f"  Memory promoted: {si.memory_promoted}")
    lines.append(f"  Daily summary: {si.daily_summary_exists}")
    lines.append(f"  Conversation turns: {si.conversation_turns}")
    if si.reflexion_domain_rates:
        lines.append(f"  Reflexion rates: {si.reflexion_domain_rates}")
    if si.reflexion_lessons:
        for domain, lessons in si.reflexion_lessons.items():
            for lesson in lessons:
                lines.append(f"    [{domain}] {lesson}")
    if si.autonomy_patterns:
        lines.append(f"  Autonomy patterns: {json.dumps(si.autonomy_patterns, default=str)}")

    # After-all
    if report.after_all:
        lines.append("")
        lines.append("── After-All Checks ──")
        for v in report.after_all:
            vs = "✓" if v.passed else "✗"
            lines.append(f"  {vs} {v.type}: expected={v.expected}, actual={v.actual}")

    # Summary
    all_v = [v for t in report.turns for v in t.verifications] + report.after_all
    passed = sum(1 for v in all_v if v.passed)
    total = len(all_v)
    completed = sum(1 for t in report.turns if t.trace_reason == "completed")
    lines.append("")
    lines.append("── Summary ──")
    lines.append(f"  Turns: {completed}/{len(report.turns)} completed")
    if total > 0:
        lines.append(f"  Checks: {passed}/{total} passed")
    lines.append(f"  Total tokens: {sum(t.tokens for t in report.turns)}")

    return "\n".join(lines)


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="RUNE E2E Test Harness")
    parser.add_argument("--provider", default="ollama", help="LLM provider")
    parser.add_argument("--model", default="gemma4:26b", help="Model name")
    parser.add_argument("--turn", action="append", help="Add a turn (repeatable)")
    parser.add_argument("--scenario", help="YAML/JSON scenario file")
    parser.add_argument("--suite", help="Directory of scenario files")
    parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    args = parser.parse_args()

    if args.suite:
        # Run all scenarios in directory
        suite_dir = Path(args.suite)
        scenarios = sorted(suite_dir.glob("*.yaml")) + sorted(suite_dir.glob("*.json"))
        if not scenarios:
            print(f"No scenarios found in {suite_dir}", file=sys.stderr)
            sys.exit(1)

        reports: list[E2EReport] = []
        for spath in scenarios:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Running: {spath.name}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            if spath.suffix == ".json":
                scenario = load_scenario_json(str(spath))
            else:
                scenario = load_scenario(str(spath))
            report = asyncio.run(run_scenario(scenario, args.provider, args.model))
            reports.append(report)
            print(format_report(report, args.format))

        if args.format == "json":
            print(json.dumps([asdict(r) for r in reports], indent=2, ensure_ascii=False, default=str))

    elif args.scenario:
        # Single scenario
        if args.scenario.endswith(".json"):
            scenario = load_scenario_json(args.scenario)
        else:
            scenario = load_scenario(args.scenario)
        report = asyncio.run(run_scenario(scenario, args.provider, args.model))
        print(format_report(report, args.format))

    elif args.turn:
        # Inline turns
        scenario = {
            "name": "inline",
            "turns": args.turn,
        }
        report = asyncio.run(run_scenario(scenario, args.provider, args.model))
        print(format_report(report, args.format))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
