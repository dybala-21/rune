"""RUNE agent hooks - pre/post tool use and task completion gates.

Ported from src/agent/hooks/index.ts.
"""

from rune.agent.hooks.runner import HookRunner
from rune.agent.hooks.skill_security_gate import (
    DEFAULT_SKILL_SECURITY_GATE_CONFIG,
    SkillGateMode,
    SkillSecurityGateConfig,
    create_skill_security_gate_hook,
    resolve_skill_security_gate_config_for_goal,
)
from rune.agent.hooks.test_gate import (
    DEFAULT_TEST_GATE_CONFIG,
    TestGateConfig,
    TestGateMode,
    create_test_gate_hook,
)

__all__ = [
    # Runner
    "HookRunner",
    # Test gate
    "TestGateConfig",
    "TestGateMode",
    "DEFAULT_TEST_GATE_CONFIG",
    "create_test_gate_hook",
    # Skill security gate
    "SkillSecurityGateConfig",
    "SkillGateMode",
    "DEFAULT_SKILL_SECURITY_GATE_CONFIG",
    "create_skill_security_gate_hook",
    "resolve_skill_security_gate_config_for_goal",
]
