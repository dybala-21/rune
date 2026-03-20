"""Structured error hierarchy for RUNE.

Ported from src/utils/errors.ts - error codes E1xx-E6xx with typed subclasses.
"""

from __future__ import annotations


class RuneError(Exception):
    """Base error for all RUNE operations."""

    code: str = "E000"

    def __init__(self, message: str, *, code: str | None = None, cause: Exception | None = None):
        self.message = message
        if code is not None:
            self.code = code
        self.__cause__ = cause
        super().__init__(f"[{self.code}] {message}")


# E1xx - Parse / Validation

class ParseError(RuneError):
    """Invalid input or schema validation failure."""

    code = "E100"


class ConfigError(ParseError):
    """Configuration file is invalid or missing."""

    code = "E101"


class SchemaError(ParseError):
    """Data does not match expected schema."""

    code = "E102"


# E2xx - Policy / Safety

class PolicyError(RuneError):
    """Action blocked by safety policy."""

    code = "E200"


class ApprovalTimeoutError(PolicyError):
    """User did not respond to approval prompt in time."""

    code = "E201"


class SandboxError(PolicyError):
    """Sandbox execution failed or is unavailable."""

    code = "E202"


class GuardianBlockError(PolicyError):
    """Guardian blocked a dangerous command."""

    code = "E203"


# E3xx - Execution

class ExecutionError(RuneError):
    """Tool or command execution failed."""

    code = "E300"


class TimeoutError_(ExecutionError):
    """Operation exceeded its time budget."""

    code = "E301"


class SubprocessError(ExecutionError):
    """Subprocess returned non-zero exit code."""

    code = "E302"


class FileOperationError(ExecutionError):
    """File read/write/delete failed."""

    code = "E303"


# E4xx - LLM / Provider

class LLMError(RuneError):
    """LLM API call failed."""

    code = "E400"


class ProviderUnavailableError(LLMError):
    """No LLM provider is reachable."""

    code = "E401"


class TokenBudgetExceededError(LLMError):
    """Token budget exhausted."""

    code = "E402"


class RateLimitError(LLMError):
    """Provider returned rate limit error."""

    code = "E403"


class ModelNotFoundError(LLMError):
    """Requested model is not available."""

    code = "E404"


# E5xx - Memory / Storage

class StorageError(RuneError):
    """Database or storage operation failed."""

    code = "E500"


class VectorIndexError(StorageError):
    """FAISS vector index operation failed."""

    code = "E501"


class EmbeddingError(StorageError):
    """Embedding generation failed."""

    code = "E502"


# E6xx - Network / Channel

class NetworkError(RuneError):
    """Network request failed."""

    code = "E600"


class ChannelError(NetworkError):
    """Messaging channel (Telegram/Discord/Slack) error."""

    code = "E601"


class MCPError(NetworkError):
    """MCP protocol error."""

    code = "E602"
