"""Default configuration constants for RUNE.

Ported from src/config/defaults.ts - all default values centralized.
"""

from __future__ import annotations

# LLM Defaults

DEFAULT_OPENAI_MODELS = {
    "best": "gpt-5.4",
    "coding": "gpt-5.3-codex",
    "fast": "gpt-5-mini",
}

DEFAULT_ANTHROPIC_MODELS = {
    "best": "claude-sonnet-4-5-20250929",
    "coding": "claude-sonnet-4-5-20250929",
    "fast": "claude-haiku-4-5-20251001",
}

DEFAULT_GEMINI_MODELS = {
    "best": "gemini-2.5-flash",
    "coding": "gemini-2.5-flash",
    "fast": "gemini-2.5-flash",
}

DEFAULT_AZURE_MODELS = {
    "best": "gpt-5.4",
    "coding": "gpt-5.3-codex",
    "fast": "gpt-5-mini",
}

DEFAULT_OLLAMA_MODELS = {
    "best": "llama3.2",
    "coding": "codellama",
    "fast": "llama3.2",
}

# Hardcoded Anthropic model list (no official list API)
ANTHROPIC_MODELS = [
    "claude-opus-4-6",
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-20250514",
    "claude-haiku-4-5-20251001",
]

# Safety Defaults

DEFAULT_APPROVAL_TIMEOUT = 300  # seconds
DEFAULT_SESSION_CACHE_MAX = 200

# Autonomy promotion thresholds
AUTONOMY_L1_ACCEPTS = 3
AUTONOMY_L1_CONFIDENCE = 0.7
AUTONOMY_L2_SUCCESSES = 10
AUTONOMY_L2_CONFIDENCE = 0.9

# Agent Defaults

DEFAULT_MAX_ITERATIONS = 200
DEFAULT_AGENT_TIMEOUT = 1800  # 30 min
DEFAULT_MAX_TOKENS = 16_384

# Token budget phases (monotonic)
TOKEN_BUDGET_PHASES = {
    "phase_1": 0.40,
    "phase_2": 0.60,
    "phase_3": 0.75,
    "phase_4": 0.85,
}

# 4-Phase Rollover thresholds
ROLLOVER_THRESHOLDS = {
    "phase_1": 0.70,
    "phase_2": 0.80,
    "phase_3": 0.90,
    "phase_4": 0.97,
}

# Observation masking windows
FULL_WINDOW_MIN = 2
FULL_WINDOW_MAX = 6
TRUNCATE_WINDOW_MIN = 4
TRUNCATE_WINDOW_MAX = 10

# Active Tools Reduction: after step 6, only recent + base tools
ACTIVE_TOOLS_REDUCTION_STEP = 6

# Delayed commit streaming: 500ms buffer
DELAYED_COMMIT_MS = 500

# Cognitive cache: LRU 50 entries
COGNITIVE_CACHE_MAX = 50

# Stale output compression: 13 tools
STALE_OUTPUT_TOOLS_COUNT = 13

# Token Optimization A/B Flag
# Set to True to enable token cost optimizations (chat prompt separation,
# budget right-sizing, output truncation tightening).
# Set to False to use the original behavior for comparison.
# Env override: RUNE_TOKEN_OPT=1 or RUNE_TOKEN_OPT=0
import os as _os

TOKEN_OPTIMIZATION_ENABLED: bool = _os.environ.get("RUNE_TOKEN_OPT", "1") == "1"

# Filesystem Defaults

DEFAULT_MAX_FILE_SIZE = 10_485_760  # 10MB
DEFAULT_MAX_LINE_COUNT = 2000
DEFAULT_MAX_DIRECTORY_DEPTH = 10
DEFAULT_MAX_FILES_PER_LIST = 1000

# Process Defaults

DEFAULT_BASH_TIMEOUT_MS = 60_000  # 60s
DEFAULT_READINESS_TIMEOUT_MS = 15_000
DEFAULT_READINESS_INTERVAL_MS = 1_000
DEFAULT_SMOKE_TIMEOUT_MS = 10_000
DEFAULT_TEARDOWN_TIMEOUT_MS = 10_000
DEFAULT_OUTPUT_BUFFER_LIMIT = 10_485_760  # 10MB

# Health Check Defaults

HEALTH_CHECK_TIMEOUT_MS = 1500
HEALTH_CHECK_CACHE_TTL_MS = 15_000

# Tool Subsets by Goal Type

TOOLS_CHAT = [
    "think", "memory_search", "memory_save", "ask_user",
    "web_search", "web_fetch", "file_read", "file_write", "file_edit",
]

TOOLS_WEB = [
    "think", "memory_search", "memory_save",
    "web_search", "web_fetch",
    "browser_navigate", "browser_observe", "browser_act",
    "browser_batch", "browser_extract", "browser_find",
    "browser_screenshot", "file_read",
]

TOOLS_RESEARCH = [
    "think", "memory_search", "memory_save",
    "web_search", "web_fetch",
    "file_read", "file_list", "file_search",
    "code_analyze", "code_find_def", "code_find_refs",
    "code_impact", "project_map",
]
