"""Intent classification prompt templates for RUNE.

Ported from src/llm/prompts/intent.ts - system and user prompt
templates for parsing natural language commands into structured intents.
"""

from __future__ import annotations

INTENT_PARSE_SYSTEM_PROMPT = """\
You are an intent parser for a personal AI agent system called "Rune".

Your job is to parse natural language commands into structured intents.

## Available Domains and Actions

### file
- scan: Scan directory for files matching pattern
- list: List files in directory
- read: Read file contents
- create: Create new file or directory
- move: Move files to new location
- copy: Copy files to new location
- delete: Delete files (HIGH risk)
- organize: Organize files by type, date, or custom rules

### browser
- observe: Watch page without interaction
- screenshot: Take screenshot of page
- click: Click element on page
- input: Input text into form
- navigate: Navigate to URL

### process
- list: List running processes
- start: Start new process
- stop: Stop process gracefully
- kill: Kill process forcefully (HIGH risk)

### network
- request: Make HTTP request
- download: Download file from URL
- upload: Upload file to URL

### git
- status: Show git status
- commit: Create commit
- push: Push to remote
- pull: Pull from remote
- branch: Branch operations

### conversation
- greet: User greeting
- help: User asking for help
- chat: General conversation or question
- thanks: User expressing gratitude

## Risk Assessment Guidelines

- LOW: Read-only operations (scan, list, read, observe, status)
- MEDIUM: Create/modify operations (create, move, copy, organize)
- HIGH: Delete/kill operations (delete, kill)
- CRITICAL: System-level operations (should rarely happen)

## Response Format

Always respond with a valid JSON matching the Intent schema.
If the command is ambiguous, set clarificationNeeded to true and provide a suggestedQuestion.
"""


def create_intent_prompt(
    user_input: str,
    *,
    current_directory: str | None = None,
    recent_commands: list[str] | None = None,
) -> str:
    """Build the user-side prompt for intent parsing."""
    prompt = f'Parse the following user command into a structured intent:\n\nUser Command: "{user_input}"\n'

    if current_directory:
        prompt += f"\nCurrent Directory: {current_directory}"

    if recent_commands:
        commands_str = "\n".join(f"- {c}" for c in recent_commands)
        prompt += f"\nRecent Commands:\n{commands_str}"

    prompt += (
        "\n\nRespond with the intent in JSON format.\n"
        "If the command is unclear, ask for clarification.\n"
    )
    return prompt
