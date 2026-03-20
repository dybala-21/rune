"""Python SDK for remote RUNE clients.

Ported from src/api/client-sdk.ts - httpx-based client supporting
RPC calls over POST /api/v1/rpc and SSE event subscriptions.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from rune.api.protocol import (
    API_VERSION,
)
from rune.utils.fast_serde import json_decode
from rune.utils.logger import get_logger

log = get_logger(__name__)


class RuneApiError(Exception):
    """Error from the RUNE API."""

    def __init__(self, code: str, message: str, http_status: int = 0) -> None:
        self.code = code
        self.http_status = http_status
        super().__init__(f"[{code}] {message}")


class RuneClient:
    """Async HTTP client for the RUNE API.

    Uses ``httpx`` for HTTP transport. Supports RPC calls, SSE subscriptions,
    and high-level convenience methods.

    Example::

        client = RuneClient("http://127.0.0.1:18789", token="rune_...")
        result = await client.request("Write a test")
        print(result)
        await client.close()
    """

    def __init__(self, base_url: str, *, token: str | None = None) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for the RUNE client SDK. "
                "Install with: pip install httpx"
            ) from exc

        self._base_url = base_url.rstrip("/")
        self._token = token
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> RuneClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # Internal RPC

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def rpc(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Make a raw RPC call.

        Args:
            method: RPC method name (e.g. ``"agent.request"``).
            params: Method parameters.

        Returns:
            The ``data`` field from the response envelope.

        Raises:
            RuneApiError: If the API returns an error.
        """
        url = f"/api/{API_VERSION}/rpc"
        body = {"method": method, "params": params or {}}

        resp = await self._client.post(url, json=body, headers=self._auth_headers())
        data = resp.json()

        if not data.get("success"):
            err = data.get("error", {})
            raise RuneApiError(
                code=err.get("code", "UNKNOWN"),
                message=err.get("message", "Unknown error"),
                http_status=resp.status_code,
            )

        return data.get("data")

    # Agent

    async def request(
        self,
        goal: str,
        *,
        session_id: str | None = None,
        cwd: str | None = None,
        sender_name: str | None = None,
    ) -> dict[str, Any]:
        """Submit an agent execution request. Returns ``{runId, sessionId}``."""
        return await self.rpc(
            "agent.request",
            {
                "goal": goal,
                "sessionId": session_id,
                "cwd": cwd,
                "senderName": sender_name,
            },
        )

    async def abort(self, run_id: str) -> None:
        """Abort a running agent execution."""
        await self.rpc("agent.abort", {"runId": run_id})

    async def status(self, run_id: str) -> dict[str, Any]:
        """Get the status of an agent run."""
        return await self.rpc("agent.status", {"runId": run_id})

    async def approve(
        self,
        approval_id: str,
        decision: str,
        guidance: str | None = None,
    ) -> None:
        """Respond to an approval request."""
        await self.rpc(
            "agent.approval",
            {
                "approvalId": approval_id,
                "decision": decision,
                "userGuidance": guidance,
            },
        )

    async def answer(
        self,
        question_id: str,
        answer: str,
        selected_index: int | None = None,
    ) -> None:
        """Respond to a question from the agent."""
        await self.rpc(
            "agent.question",
            {
                "questionId": question_id,
                "answer": answer,
                "selectedIndex": selected_index,
            },
        )

    # Sessions

    async def list_sessions(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return await self.rpc("sessions.list", params)

    async def get_session(
        self,
        session_id: str,
        *,
        include_turns: bool = False,
        max_turns: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "sessionId": session_id,
            "includeTurns": include_turns,
        }
        if max_turns is not None:
            params["maxTurns"] = max_turns
        return await self.rpc("sessions.get", params)

    async def delete_session(self, session_id: str) -> None:
        await self.rpc("sessions.delete", {"sessionId": session_id})

    async def archive_session(self, session_id: str) -> None:
        await self.rpc("sessions.archive", {"sessionId": session_id})

    # Channels & Config

    async def list_channels(self) -> dict[str, Any]:
        return await self.rpc("channels.list")

    async def get_config(self) -> dict[str, Any]:
        return await self.rpc("config.get")

    async def patch_config(self, **kwargs: Any) -> None:
        await self.rpc("config.patch", kwargs)

    # Health

    async def health(self) -> dict[str, Any]:
        return await self.rpc("health")

    # Tokens

    async def create_token(
        self,
        label: str,
        permissions: list[str],
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"label": label, "permissions": permissions}
        if expires_at:
            params["expiresAt"] = expires_at
        return await self.rpc("tokens.create", params)

    async def list_tokens(self) -> dict[str, Any]:
        return await self.rpc("tokens.list")

    async def revoke_token(self, token_id: str) -> None:
        await self.rpc("tokens.revoke", {"id": token_id})

    # Skills

    async def list_skills(self, scope: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if scope:
            params["scope"] = scope
        return await self.rpc("skills.list", params)

    async def get_skill(self, name: str) -> dict[str, Any]:
        return await self.rpc("skills.get", {"name": name})

    # Runs

    async def list_runs(self, **kwargs: Any) -> dict[str, Any]:
        return await self.rpc("runs.list", kwargs)

    async def get_run(self, run_id: str) -> dict[str, Any]:
        return await self.rpc("runs.get", {"runId": run_id})

    # SSE subscription

    async def subscribe_events(
        self,
        session_id: str,
        *,
        run_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Subscribe to SSE events for a session/run.

        Yields parsed event dictionaries. The caller should iterate
        asynchronously::

            async for event in client.subscribe_events(session_id):
                print(event)
        """
        import httpx

        params: dict[str, str] = {"sessionId": session_id}
        if run_id:
            params["runId"] = run_id

        url = f"/api/{API_VERSION}/events"

        async with self._client.stream(
            "GET",
            url,
            params=params,
            headers=self._auth_headers(),
            timeout=httpx.Timeout(None, connect=10.0),
        ) as resp:
            resp.raise_for_status()
            buffer = ""
            async for chunk in resp.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    raw_event, buffer = buffer.split("\n\n", 1)
                    for line in raw_event.split("\n"):
                        if line.startswith("data: "):
                            data_str = line[6:]
                            try:
                                yield json_decode(data_str)
                            except json.JSONDecodeError:
                                continue
