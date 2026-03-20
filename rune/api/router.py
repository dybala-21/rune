"""API Router - RPC method dispatch.

Ported from src/api/router.ts - type-safe method dispatch for
POST /api/v1/rpc. Dispatches to registered handlers based on method name
and wraps results in ApiResponse envelope.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from rune.api.protocol import ApiError, ApiResponse, Permission
from rune.utils.logger import get_logger

log = get_logger(__name__)


class RequestContext:
    """Context passed to each RPC handler."""

    __slots__ = ("client_id", "is_localhost", "auth_token", "permissions", "is_admin")

    def __init__(
        self,
        *,
        client_id: str = "",
        is_localhost: bool = False,
        auth_token: str | None = None,
        permissions: list[Permission] | None = None,
        is_admin: bool = False,
    ) -> None:
        self.client_id = client_id
        self.is_localhost = is_localhost
        self.auth_token = auth_token
        self.permissions = permissions or []
        self.is_admin = is_admin


# Handler type: (params, ctx) -> result
MethodHandler = Callable[..., Awaitable[Any]]


class ApiRouter:
    """RPC router that maps method names to async handler functions."""

    def __init__(self) -> None:
        self._handlers: dict[str, MethodHandler] = {}
        self._permissions: dict[str, Permission] = {}

    def register(
        self,
        method: str,
        handler: MethodHandler,
        required_permission: Permission | None = None,
    ) -> None:
        """Register an RPC method handler.

        Args:
            method: Dot-separated method name (e.g. ``"agent.request"``).
            handler: Async callable ``(params, ctx) -> result``.
            required_permission: Optional permission required to invoke.
        """
        self._handlers[method] = handler
        if required_permission is not None:
            self._permissions[method] = required_permission

    async def dispatch(
        self,
        method: str,
        params: Any,
        ctx: RequestContext,
    ) -> ApiResponse:
        """Dispatch an RPC call to its handler.

        Returns an :class:`ApiResponse` envelope - never raises.
        """
        handler = self._handlers.get(method)
        if handler is None:
            return self._error_response("METHOD_NOT_FOUND", f"Unknown method: {method}")

        # Permission check
        perm = self._permissions.get(method)
        if perm and not ctx.is_admin and perm not in ctx.permissions:
            return self._error_response("FORBIDDEN", f"Permission '{perm}' required")

        try:
            result = await handler(params, ctx)
            return ApiResponse(
                success=True,
                data=result,
                timestamp=datetime.now(UTC).isoformat(),
            )
        except Exception as exc:
            log.warning("rpc_handler_error", method=method, error=str(exc))
            return self._error_response("INTERNAL_ERROR", str(exc))

    def has_method(self, method: str) -> bool:
        return method in self._handlers

    def get_registered_methods(self) -> list[str]:
        return list(self._handlers.keys())


    @staticmethod
    def _error_response(
        code: str,
        message: str,
        details: Any | None = None,
    ) -> ApiResponse:
        error = ApiError(code=code, message=message)
        if details is not None:
            error.details = details
        return ApiResponse(
            success=False,
            error=error,
            timestamp=datetime.now(UTC).isoformat(),
        )
