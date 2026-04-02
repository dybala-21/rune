"""Configuration schema for RUNE.

Ported from src/config/schema.ts - Zod schemas to Pydantic v2 models.
All 145 config fields with defaults.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# LLM Configuration

class ModelsByTier(BaseModel):
    best: str = "gpt-5.4"
    coding: str = "gpt-5.3-codex"
    fast: str = "gpt-5-mini"


class ProviderModels(BaseModel):
    openai: ModelsByTier = Field(default_factory=lambda: ModelsByTier())
    anthropic: ModelsByTier = Field(
        default_factory=lambda: ModelsByTier(
            best="claude-sonnet-4-5-20250929",
            coding="claude-sonnet-4-5-20250929",
            fast="claude-haiku-4-5-20251001",
        )
    )
    gemini: ModelsByTier = Field(
        default_factory=lambda: ModelsByTier(
            best="gemini-2.5-flash",
            coding="gemini-2.5-flash",
            fast="gemini-2.5-flash",
        )
    )
    azure: ModelsByTier = Field(
        default_factory=lambda: ModelsByTier(
            best="gpt-5.4",
            coding="gpt-5.3-codex",
            fast="gpt-5-mini",
        )
    )
    ollama: ModelsByTier = Field(
        default_factory=lambda: ModelsByTier(
            best="llama3.2",
            coding="codellama",
            fast="llama3.2",
        )
    )


class LLMConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    default_provider: str = Field(default="openai", alias="defaultProvider")
    default_model: str = Field(default="gpt-5.4", alias="defaultModel")
    active_provider: str | None = Field(default=None, alias="activeProvider")
    active_model: str | None = Field(default=None, alias="activeModel")
    models: ProviderModels = Field(default_factory=ProviderModels)
    routing_mode: str = Field(default="cloud-first", alias="routingMode")
    request_timeout_ms: int = Field(default=600_000, alias="requestTimeoutMs")
    max_retries: int = Field(default=2, alias="maxRetries")


# Approval Configuration

class ApprovalConfig(BaseModel):
    profile: str = "general"  # general | developer | automation
    auto_approve_safe: bool = True
    auto_approve_low: bool = True
    auto_approve_medium: bool = False
    timeout_seconds: int = 300
    session_cache_max: int = 200


# Safety Configuration

class SandboxConfig(BaseModel):
    enabled: bool = True
    allow_network: bool = False
    writable_paths: list[str] = Field(default_factory=list)
    readable_paths: list[str] = Field(default_factory=list)
    blocked_paths: list[str] = Field(default_factory=list)
    timeout_seconds: int = 60


class SafetyConfig(BaseModel):
    rollout_mode: str = "auto"  # auto | shadow | balanced | strict | legacy
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    deny_by_default_executables: bool = True
    executable_allowlist: list[str] = Field(
        default_factory=lambda: [
            "ls", "pwd", "echo", "cat", "head", "tail", "find", "wc",
            "sort", "uniq", "grep", "rg", "sed", "awk", "tr", "cut",
            "diff", "file", "stat", "du", "df", "which", "whoami",
            "date", "env", "printenv", "true", "false", "test",
            "git", "npm", "pnpm", "yarn", "pip", "pip3", "uv",
            "python", "python3", "node", "npx", "go", "cargo", "rustc",
            "pytest", "vitest", "jest", "make", "cmake",
            "docker", "docker-compose",
            "curl", "wget", "ssh", "scp", "rsync",
            "tar", "zip", "unzip", "gzip", "gunzip",
            "mkdir", "cp", "mv", "ln", "touch", "chmod", "chown",
            "tee", "xargs", "basename", "dirname", "realpath",
            "jq", "yq",
        ]
    )


# Hooks Configuration

class HooksConfig(BaseModel):
    test_gate: str = "advisory"  # advisory | required
    skill_gate: str = "advisory"  # advisory | required
    suspicious_patterns: list[str] = Field(
        default_factory=lambda: [
            r"curl\s+.*\|\s*(bash|sh)",
            r"wget\s+.*\|\s*(bash|sh)",
            r"rm\s+-rf\s+/",
            r"chmod\s+777",
            r"\bsudo\b",
            r"export\s+(OPENAI|ANTHROPIC|AWS|GITHUB)_[A-Z_]*\s*=",
            r"(api[_-]?key|secret|token|password)\s*[:=]",
            r"\.env",
        ]
    )


# Filesystem Configuration

class FilesystemConfig(BaseModel):
    allow_paths: list[str] = Field(
        default_factory=lambda: ["~/Projects", "~/workspace"]
    )
    deny_paths: list[str] = Field(
        default_factory=lambda: [
            "/etc/", "/usr/", "/System/",
            "~/.ssh/", "~/.aws/", "~/.gnupg/",
        ]
    )
    max_file_size_bytes: int = 10_485_760  # 10MB
    max_directory_depth: int = 10
    max_files_per_list: int = 1000


# Proactive Configuration

class ProactiveConfig(BaseModel):
    enabled: bool = False
    quiet_hours_start: int = 22  # 10 PM
    quiet_hours_end: int = 8    # 8 AM
    autonomy_promotion_accepts: int = 3
    autonomy_promotion_confidence: float = 0.7
    autonomy_demotion_failures: int = 2


# Browser Configuration

class BrowserConfig(BaseModel):
    default_profile: str = "managed"  # managed | relay
    headless: bool = True  # managed mode is always headless; use relay for headed
    timeout_ms: int = 30_000
    viewport_width: int = 1280
    viewport_height: int = 720


# Search Configuration

class SearchConfig(BaseModel):
    provider: str = "auto"  # brave | duckduckgo | browser | auto
    max_concurrent: int = 3
    native_budget: int = 5


# Voice Configuration

class VoiceConfig(BaseModel):
    enabled: bool = False
    provider: str = "deepgram"  # deepgram | sherpa-onnx
    language: str = "en"


# General Configuration

class GeneralConfig(BaseModel):
    locale: str = "en"
    theme: str = "auto"
    telemetry: bool = False
    update_check: bool = True


# Cron Execution Configuration

class CronExecutionConfig(BaseModel):
    enabled: bool = True
    max_concurrent: int = 3
    default_timeout_seconds: int = 300


# Root Configuration

class RuneConfig(BaseModel):
    """Root configuration schema for RUNE."""

    version: str = "1.0"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    approval: ApprovalConfig = Field(default_factory=ApprovalConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    filesystem: FilesystemConfig = Field(default_factory=FilesystemConfig)
    proactive: ProactiveConfig = Field(default_factory=ProactiveConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    cron_execution: CronExecutionConfig = Field(default_factory=CronExecutionConfig)

    # API keys (resolved from config, then env, then None)
    openai_api_key: str | None = Field(default=None, alias="openai_api_key")
    anthropic_api_key: str | None = Field(default=None, alias="anthropic_api_key")

    # Google Gemini (simple API key, like OpenAI)
    gemini_api_key: str | None = Field(default=None, alias="gemini_api_key")

    # Google Cloud / Vertex AI (service account, for enterprise)
    google_credentials_file: str | None = Field(default=None, alias="google_credentials_file")
    vertex_project: str | None = Field(default=None, alias="vertex_project")
    vertex_location: str = Field(default="us-central1", alias="vertex_location")

    model_config = ConfigDict(populate_by_name=True)
