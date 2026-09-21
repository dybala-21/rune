"""Rune configuration models, defaults, and compatibility aliases."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rune.llm.reasoning import ReasoningEffort, reasoning_model_key

# LLM Configuration

class ModelsByTier(BaseModel):
    best: str = "gpt-6-astra"
    coding: str = "gpt-5.3-codex"
    fast: str = "gpt-5.4-mini"


class ProviderModels(BaseModel):
    openai: ModelsByTier = Field(default_factory=lambda: ModelsByTier())
    xai: ModelsByTier = Field(default_factory=lambda: ModelsByTier(
        best="grok-4.6", coding="grok-4.6", fast="grok-4.3",
    ))
    anthropic: ModelsByTier = Field(
        default_factory=lambda: ModelsByTier(
            best="claude-opus-5",
            coding="claude-sonnet-5",
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
            best="gpt-6-astra",
            coding="gpt-5.3-codex",
            fast="gpt-5.4-mini",
        )
    )
    # Installed local models take precedence over these fallback names.
    ollama: ModelsByTier = Field(
        default_factory=lambda: ModelsByTier(
            best="qwen3-coder:30b",
            coding="qwen3-coder:30b",
            fast="qwen3-coder:30b",
        )
    )


class DecisionRoutingConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    backend: Literal["connected", "jev"] = "connected"
    timeout_ms: int = Field(default=1500, ge=250, le=5000, alias="timeoutMs")


class LLMConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    default_provider: str = Field(default="openai", alias="defaultProvider")
    default_model: str = Field(default="gpt-6-astra", alias="defaultModel")
    active_provider: str | None = Field(default=None, alias="activeProvider")
    active_model: str | None = Field(default=None, alias="activeModel")
    # Migrate the legacy global preference to the previously selected model.
    reasoning_effort: ReasoningEffort | None = Field(default=None, alias="reasoningEffort")
    reasoning_efforts: dict[str, ReasoningEffort | None] = Field(default_factory=dict, alias="reasoningEfforts")
    # Escalation uses the provider's best tier when no model is specified.
    escalation_provider: str | None = Field(default=None, alias="escalationProvider")
    escalation_model: str | None = Field(default=None, alias="escalationModel")
    models: ProviderModels = Field(default_factory=ProviderModels)
    decision_routing: DecisionRoutingConfig = Field(
        default_factory=DecisionRoutingConfig, alias="decisionRouting",
    )
    # Simple-query fast lane: high-confidence chat/web goals run on the
    # provider's fast tier (docs/design/simple-query-fast-path.md).
    route_simple_queries: bool = Field(default=True, alias="routeSimpleQueries")
    simple_query_tier: str = Field(
        default="fast", pattern="^(best|coding|fast)$", alias="simpleQueryTier"
    )
    simple_query_confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, alias="simpleQueryConfidence"
    )
    routing_mode: str = Field(default="cloud-first", alias="routingMode")
    request_timeout_ms: int = Field(default=600_000, alias="requestTimeoutMs")
    max_retries: int = Field(default=2, alias="maxRetries")

    @model_validator(mode="after")
    def migrate_reasoning_preference(self):
        if self.reasoning_effort is not None:
            provider = self.active_provider if self.active_provider and self.active_model else self.default_provider
            tiers = getattr(self.models, provider, None)
            model = self.active_model if self.active_provider and self.active_model else getattr(tiers, "best", "unknown")
            key = reasoning_model_key(model if provider == "openai" else f"{provider}/{model}")
            self.reasoning_efforts.setdefault(key, self.reasoning_effort)
            self.reasoning_effort = None
        return self


# Approval Configuration

class ApprovalConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    profile: str = "general"  # general | developer | automation
    auto_approve_safe: bool = True
    auto_approve_low: bool = True
    auto_approve_medium: bool = False
    timeout_seconds: int = 300
    session_cache_max: int = 200
    # Require approval regardless of risk score, unless mode is bypass.
    # Names accept globs; dots, underscores, and hyphens are equivalent.
    require_explicit_for: list[str] = Field(
        default_factory=list, alias="requireExplicitFor"
    )
    # RUNE_APPROVAL_MODE overrides this value for a run.
    # bypass: skip approvals; standard: gate risky commands and external writes;
    # strict: also gate network reads and browser interactions.
    mode: str = "standard"


# Safety Configuration

class SandboxConfig(BaseModel):
    enabled: bool = True
    allow_network: bool = False
    writable_paths: list[str] = Field(default_factory=list)
    readable_paths: list[str] = Field(default_factory=list)
    blocked_paths: list[str] = Field(default_factory=list)
    timeout_seconds: int = 60


class DenyByDefaultConfig(BaseModel):
    """Allowlist-only execution policy, read by the shell gate."""

    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = True
    allowed_executables: list[str] = Field(
        default_factory=list, alias="allowedExecutables"
    )


class SafetyConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # An empty allowlist uses the shipped defaults; a non-empty list replaces them.
    deny_by_default: DenyByDefaultConfig = Field(
        default_factory=DenyByDefaultConfig, alias="denyByDefault"
    )

    # Read by the shell gate (rune/capabilities/bash.py). "auto" means the
    # shipped default; the other values map to execution-policy branches.
    rollout_mode: str = Field(
        default="auto", alias="rolloutMode"
    )  # auto | shadow | balanced | strict | legacy
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

class SkillGateConfig(BaseModel):
    """Settings for checking generated skills before saving them."""

    model_config = ConfigDict(populate_by_name=True)

    # "required" rejects a flagged skill; "advisory" writes it and logs.
    mode: str = "advisory"
    auto_harden_on_code_tasks: bool = Field(default=True, alias="autoHardenOnCodeTasks")
    allowed_authors: list[str] = Field(
        default_factory=lambda: ["rune-agent"], alias="allowedAuthors"
    )
    require_signature: bool = Field(default=False, alias="requireSignature")
    allow_auto_sign_when_missing: bool = Field(
        default=True, alias="allowAutoSignWhenMissing"
    )
    signature_secret_env: str = Field(
        default="RUNE_SKILL_SIGNING_KEY", alias="signatureSecretEnv"
    )
    block_project_scope: bool = Field(default=True, alias="blockProjectScope")
    project_scope_allowed_name_prefixes: list[str] = Field(
        default_factory=list, alias="projectScopeAllowedNamePrefixes"
    )
    max_body_chars: int = Field(default=12_000, alias="maxBodyChars")
    suspicious_patterns: list[str] = Field(
        alias="suspiciousPatterns",
        default_factory=lambda: [
            r"curl\s+.*\|\s*(bash|sh)",
            r"wget\s+.*\|\s*(bash|sh)",
            r"rm\s+-rf\s+/",
            r"chmod\s+777",
            r"\bsudo\b",
            r"export\s+(OPENAI|ANTHROPIC|AWS|GITHUB)_[A-Z_]*\s*=",
            r"(api[_-]?key|secret|token|password)\s*[:=]",
            r"\.env",
        ],
    )


class HooksConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    skill_gate: SkillGateConfig = Field(
        default_factory=SkillGateConfig, alias="skillGate"
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
    # Preserve the daemon default when older configs omit this section.
    enabled: bool = True
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
    model_config = ConfigDict(populate_by_name=True)

    provider: str = "auto"  # brave | duckduckgo | browser | auto
    # Caps concurrent browser pages when search falls back to the browser
    # (rune/daemon/main.py builds the page pool from this).
    max_concurrent: int = Field(default=3, alias="maxConcurrent")
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


# Goal Loop Configuration (the /goal autonomous loop)

class GoalLoopConfig(BaseModel):
    enabled: bool = True
    max_iterations: int = 10
    max_total_tokens: int = 2_000_000  # cost-runaway cap
    stagnation_window: int = 3  # identical outcomes in a row -> stop
    evidence_threshold: float = 0.8  # inner-loop evidence_score floor
    validation_timeout_seconds: int = 600  # per SPEC validation command
    adversarial_review: bool = True  # run the allow/block gate before accepting
    ssc_interval: int = 0  # self-critique every N iterations (0 = off, opt-in)
    inner_token_budget: int = 1_000_000  # per-iteration NativeAgentLoop budget
    # When stuck (stagnation/max_iterations/budget), run one final attempt on the
    # escalation profile before giving up. Opt-in: that attempt goes to the cloud
    # escalation model, so default off to preserve local-only operation.
    escalate_on_stuck: bool = False


# Root Configuration

class SkillsConfig(BaseModel):
    """Opt-in learning from completed or verified runs and reuse in later tasks.

    ``auto_skill`` controls both generation and reuse of learned skills.
    """

    model_config = ConfigDict(populate_by_name=True)

    auto_skill: bool = Field(default=False, alias="autoSkill")

    # Evaluate candidates before reuse and deprecate skills that regress.
    gated_learning: bool = Field(default=False, alias="gatedLearning")
    # Evidence thresholds for promoting a candidate skill.
    eval_delta_min: float = Field(default=0.05, alias="evalDeltaMin")
    eval_prob_threshold: float = Field(default=0.95, alias="evalProbThreshold")
    eval_min_samples_paired: int = Field(default=12, alias="evalMinSamplesPaired")
    eval_min_samples_online: int = Field(default=40, alias="evalMinSamplesOnline")
    # Capture reproducible tasks (workspace snapshot + check) for offline paired
    # replay. Default off — snapshotting has storage cost.
    capture_replay: bool = Field(default=False, alias="captureReplay")
    eval_max_pairs: int = Field(default=20, alias="evalMaxPairs")


class RuneConfig(BaseModel):
    """Root configuration schema for RUNE."""

    version: str = "1.0"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
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
    goal_loop: GoalLoopConfig = Field(default_factory=GoalLoopConfig)

    # API keys (resolved from config, then env, then None)
    openai_api_key: str | None = Field(default=None, alias="openai_api_key")
    anthropic_api_key: str | None = Field(default=None, alias="anthropic_api_key")

    # Google Gemini API key
    gemini_api_key: str | None = Field(default=None, alias="gemini_api_key")

    # Google Cloud / Vertex AI service account
    google_credentials_file: str | None = Field(default=None, alias="google_credentials_file")
    vertex_project: str | None = Field(default=None, alias="vertex_project")
    vertex_location: str = Field(default="us-central1", alias="vertex_location")

    model_config = ConfigDict(populate_by_name=True)
