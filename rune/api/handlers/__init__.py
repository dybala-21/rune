"""API route handlers.

Each module exposes a FastAPI ``APIRouter`` that is included
by the main application via ``app.include_router(...)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI


def register_all_handlers(app: FastAPI) -> None:
    """Include every handler router in *app*."""
    from rune.api.handlers.agent import router as agent_router
    from rune.api.handlers.channels import router as channels_router
    from rune.api.handlers.config import router as config_router
    from rune.api.handlers.cron import router as cron_router
    from rune.api.handlers.env import router as env_router
    from rune.api.handlers.health import router as health_router
    from rune.api.handlers.proactive import router as proactive_router
    from rune.api.handlers.runs import router as runs_router
    from rune.api.handlers.sessions import router as sessions_router
    from rune.api.handlers.skills import router as skills_router
    from rune.api.handlers.tokens import router as tokens_router

    app.include_router(health_router)
    app.include_router(agent_router)
    app.include_router(channels_router)
    app.include_router(config_router)
    app.include_router(cron_router)
    app.include_router(env_router)
    app.include_router(proactive_router)
    app.include_router(runs_router)
    app.include_router(sessions_router)
    app.include_router(skills_router)
    app.include_router(tokens_router)
