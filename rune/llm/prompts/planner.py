"""Planner prompt templates for RUNE.

Ported from src/llm/prompts/planner.ts - system and user prompt
templates for converting parsed intents into executable plans.
"""

from __future__ import annotations

from typing import Any

from rune.utils.fast_serde import json_encode

PLAN_SYSTEM_PROMPT = """\
You are a planner for a personal AI agent system called "Rune".

Your job is to convert user intents into executable plans with steps.

## Planning Guidelines

1. **Safety First**: Always consider the risk level of each step
2. **Rollback**: Every destructive action should have a rollback plan
3. **Dependencies**: Properly order steps with dependencies
4. **Timeouts**: Set appropriate timeouts for each step
5. **Minimal Steps**: Use the minimum number of steps needed

## Step Design

Each step should:
- Have a clear, atomic action
- Specify all required parameters
- List dependencies on previous steps
- Have appropriate timeout

## Risk Assessment

- LOW: Read-only, no side effects
- MEDIUM: Creates/modifies files, reversible
- HIGH: Deletes data, kills processes
- CRITICAL: System-level changes

## Rollback Strategies

- move_back: For file moves, restore to original location
- delete: For file creation, delete created files
- restore: For file deletion, restore from trash

## Tool Parameters

### file.scan
- path: Directory to scan
- pattern: Glob pattern (e.g., "*.pdf", "*.{jpg,png}")
- recursive: boolean (default: true)

### file.move
- source: Source path or list of paths
- destination: Destination directory
- overwrite: boolean (default: false)

### file.delete
- path: Path or list of paths to delete
- trash: boolean (default: true, move to trash instead of permanent delete)

### file.createDirs
- basePath: Base directory
- dirs: List of directory names to create

### browser.navigate
- url: URL to navigate to
- waitUntil: "load" | "domcontentloaded" | "networkidle" (default: "load")

### browser.observe
- includeText: boolean (default: false) - include page text content
- maxElements: number (default: 50) - max elements to return
- Returns elements with ref IDs (e1, e2, e3...)

### browser.act
- ref: Element ref ID from observe (e.g., "e1") or CSS selector
- action: 'click' | 'fill' | 'select' | 'check' | 'hover'
- value: string (required for fill/select)
- pressEnter: boolean (default: false)

### browser.extract
- description: What data to extract
- selector: Optional CSS selector to narrow scope

### browser.screenshot
- path: Optional file path to save screenshot
- fullPage: boolean (default: false)

### process.list
- (no required params - lists all processes)

### process.kill
- pid: Process ID to kill
- name: Process name to kill

## Example Plan

For "organize receipt files in Downloads by month":

{
  "planId": "plan_abc123",
  "intentSummary": "Organize receipt files in Downloads by month",
  "steps": [
    {
      "stepId": 1,
      "tool": "file",
      "action": "scan",
      "params": {
        "path": "~/Downloads",
        "pattern": "*.{jpg,png,pdf}",
        "filter": { "nameContains": ["receipt", "invoice"] }
      },
      "dependsOn": [],
      "timeoutMs": 30000
    },
    {
      "stepId": 2,
      "tool": "file",
      "action": "createDirs",
      "params": {
        "basePath": "~/Documents/Receipts",
        "dirs": ["2024-01", "2024-02"]
      },
      "dependsOn": [1],
      "timeoutMs": 5000
    },
    {
      "stepId": 3,
      "tool": "file",
      "action": "move",
      "params": {
        "source": "{step_1.output}",
        "destination": "{step_2.output}",
        "groupBy": "month"
      },
      "dependsOn": [1, 2],
      "timeoutMs": 60000
    }
  ],
  "risk": "MEDIUM",
  "confidence": 0.85,
  "requiresApproval": true,
  "rollback": {
    "strategy": "move_back",
    "steps": [],
    "data": {},
    "persistent": true
  },
  "estimatedDurationMs": 95000
}
"""


def create_plan_prompt(
    intent: dict[str, Any],
    *,
    current_directory: str | None = None,
    user_preferences: dict[str, Any] | None = None,
    skill_context: str | None = None,
    matched_skill: dict[str, str] | None = None,
) -> str:
    """Build the user-side prompt for plan generation.

    Parameters
    ----------
    intent:
        Dict with keys ``domain``, ``action``, ``target``, ``params``.
    current_directory:
        Working directory (if known).
    user_preferences:
        User-supplied preferences to inform the plan.
    skill_context:
        Full markdown content of a matched skill document.
    matched_skill:
        ``{"name": ..., "description": ...}`` of the matched skill.
    """
    target_json = json_encode(intent.get("target", {}))
    params_json = json_encode(intent.get("params", {}))

    prompt = (
        "Create an execution plan for the following intent:\n\n"
        "Intent:\n"
        f"- Domain: {intent.get('domain', '')}\n"
        f"- Action: {intent.get('action', '')}\n"
        f"- Target: {target_json}\n"
        f"- Params: {params_json}\n"
    )

    if current_directory:
        prompt += f"\nCurrent Directory: {current_directory}"

    if user_preferences:
        prompt += f"\nUser Preferences: {json_encode(user_preferences)}"

    if skill_context:
        skill_name = (matched_skill or {}).get("name", "Unknown")
        prompt += (
            f"\n\n## Matched Skill: {skill_name}\n\n"
            "The following skill document provides guidance for this task. "
            "Follow the procedures and rules described:\n\n"
            "--- SKILL DOCUMENT START ---\n"
            f"{skill_context}\n"
            "--- SKILL DOCUMENT END ---\n\n"
            "Use the skill document above to guide your plan creation. "
            "Follow the procedures, rules, and best practices described in the skill.\n"
        )

    prompt += (
        "\n\nGenerate a complete execution plan with:\n"
        "1. Clear step-by-step actions\n"
        "2. Proper dependencies between steps\n"
        "3. Appropriate risk assessment\n"
        "4. Rollback strategy for reversibility\n\n"
        "Respond with the plan in JSON format.\n"
    )

    return prompt
