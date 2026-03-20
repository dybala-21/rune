"""Agent Roles - capability-scoped role definitions for multi-agent orchestration.

Ported from src/agent/roles.ts (260 lines).
Each role defines a constrained tool set, iteration limits, and system prompt
so that sub-agents operate with least-privilege.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AgentRoleId = Literal["researcher", "planner", "executor", "communicator"]


@dataclass(slots=True)
class AgentRole:
    """Defines the capabilities and constraints for a specialised sub-agent."""

    id: AgentRoleId
    name: str
    description: str
    capabilities: list[str] = field(default_factory=list)
    system_prompt: str = ""
    max_iterations: int = 50
    timeout_seconds: int = 600
    risk_level: str = "low"


# Built-in role catalogue

AGENT_ROLES: dict[AgentRoleId, AgentRole] = {
    "researcher": AgentRole(
        id="researcher",
        name="Researcher",
        description=(
            "Read-only investigator. Gathers information from the filesystem, "
            "web, and code analysis tools without making any modifications."
        ),
        capabilities=[
            "file_read",
            "file_list",
            "file_search",
            "web_search",
            "web_fetch",
            "code_analyze",
            "memory_search",
            "think",
            "grep",
            "glob",
        ],
        system_prompt=(
            "You are the Researcher agent. Your job is to gather information, "
            "read files, search the web, and analyse code. You must NOT modify "
            "any files or execute commands that change state. Summarise your "
            "findings clearly so other agents can act on them."
        ),
        max_iterations=15,
        timeout_seconds=600,
        risk_level="low",
    ),
    "planner": AgentRole(
        id="planner",
        name="Planner",
        description=(
            "Analysis-only strategist. Formulates plans by reasoning about "
            "the goal and available context. Does not access the filesystem "
            "or execute any tools beyond thinking and memory search."
        ),
        capabilities=[
            "think",
            "memory_search",
        ],
        system_prompt=(
            "You are the Planner agent. Your job is to break down goals into "
            "concrete, ordered steps. Consider risks, dependencies, and edge "
            "cases. Output a structured plan with clear success criteria for "
            "each step. Do NOT execute any actions — only plan."
        ),
        max_iterations=10,
        timeout_seconds=300,
        risk_level="low",
    ),
    "executor": AgentRole(
        id="executor",
        name="Executor",
        description=(
            "Full-capability agent. Carries out file edits, shell commands, "
            "git operations, and any other actions required to implement the "
            "plan produced by the Planner."
        ),
        capabilities=[
            "file_read",
            "file_write",
            "file_edit",
            "file_list",
            "file_search",
            "bash_execute",
            "git_commit",
            "git_diff",
            "git_log",
            "git_status",
            "web_search",
            "web_fetch",
            "code_analyze",
            "memory_search",
            "memory_store",
            "think",
            "ask_user",
            "grep",
            "glob",
        ],
        system_prompt=(
            "You are the Executor agent. Implement changes precisely "
            "according to the plan. Verify each step before moving on. "
            "If you encounter unexpected issues, report them clearly rather "
            "than guessing. Prefer minimal, targeted changes."
        ),
        max_iterations=20,
        timeout_seconds=1800,
        risk_level="high",
    ),
    "communicator": AgentRole(
        id="communicator",
        name="Communicator",
        description=(
            "User-facing formatting agent. Takes raw results from other "
            "agents and produces well-structured, human-readable output. "
            "May ask the user clarifying questions."
        ),
        capabilities=[
            "think",
            "ask_user",
        ],
        system_prompt=(
            "You are the Communicator agent. Your job is to take raw results "
            "from other agents and present them to the user in a clear, "
            "well-formatted manner. Highlight key findings, risks, and "
            "next steps. Ask the user for clarification when needed."
        ),
        max_iterations=20,
        timeout_seconds=120,
        risk_level="low",
    ),
}


# Public helpers

def get_role(role_id: AgentRoleId) -> AgentRole:
    """Return the :class:`AgentRole` for *role_id*.

    Raises:
        KeyError: If *role_id* is not a recognised role.
    """
    try:
        return AGENT_ROLES[role_id]
    except KeyError:
        raise KeyError(f"Unknown agent role: {role_id!r}") from None


def suggest_role(intent: str) -> AgentRoleId:
    """Heuristically suggest the best role for a natural-language *intent*.

    This is a lightweight keyword classifier; the orchestrator may override
    the suggestion with LLM-based classification.
    """
    lower = intent.lower()

    # Executor keywords - check first since they are the most specific
    executor_keywords = [
        "edit", "write", "create", "delete", "remove", "run", "execute",
        "build", "compile", "install", "commit", "push", "deploy", "fix",
        "refactor", "implement", "change", "modify", "update", "apply",
    ]
    if any(kw in lower for kw in executor_keywords):
        return "executor"

    # Planner keywords
    planner_keywords = [
        "plan", "design", "architect", "strategy", "outline", "propose",
        "break down", "steps", "approach", "roadmap",
    ]
    if any(kw in lower for kw in planner_keywords):
        return "planner"

    # Communicator keywords
    communicator_keywords = [
        "explain", "summarise", "summarize", "describe", "format",
        "present", "tell me", "what is", "help me understand",
    ]
    if any(kw in lower for kw in communicator_keywords):
        return "communicator"

    # Default: researcher
    return "researcher"


def list_role_capabilities(role_id: AgentRoleId) -> list[str]:
    """Return the list of capability names allowed for *role_id*."""
    return list(get_role(role_id).capabilities)
