"""Structured logging setup for RUNE.

Ported from src/utils/logger.ts - uses structlog with JSON output,
context binding, and storm-guard (repeated log suppression).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import structlog

_configured = False


def configure_logging(
    *,
    level: str | None = None,
    json_output: bool | None = None,
    log_file: Path | None = None,
) -> None:
    """Configure structured logging for the entire application.

    Call once at startup. Subsequent calls are no-ops.
    """
    global _configured
    if _configured:
        return
    _configured = True

    resolved_level = (
        level
        or os.environ.get("RUNE_LOG_LEVEL")
        or os.environ.get("LOG_LEVEL")
        or "warning"
    ).upper()

    use_json = json_output if json_output is not None else (
        os.environ.get("RUNE_LOG_JSON", "").lower() in ("1", "true", "yes")
    )

    # Shared processors for both structlog and stdlib loggers
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if use_json:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # stderr handler (main output)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(stderr_handler)
    root.setLevel(getattr(logging, resolved_level, logging.WARNING))

    # Optional file handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    for name in ("httpx", "httpcore", "uvicorn.access", "watchfiles"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a named logger. Auto-configures if not yet done."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)


def bind_context(**kwargs: object) -> None:
    """Bind key-value pairs to the current async context (task-local)."""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all context-bound variables."""
    structlog.contextvars.clear_contextvars()
