"""Resolve once, then pin the connection while retaining TLS hostname checks."""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import socket
from urllib.parse import quote, urlencode, urlsplit

import httpx

from rune.connectors.models import ConnectorPolicy, ConnectorRequest
from rune.types import CapabilityResult

MAX_RESPONSE = 2 * 1024 * 1024


async def public_address(host: str) -> str:
    records = await asyncio.get_running_loop().getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    addresses = [ipaddress.ip_address(record[4][0]) for record in records]
    if not addresses or any(not address.is_global or address.is_multicast for address in addresses):
        raise ValueError("Connector destination must resolve only to public addresses")
    return str(addresses[0])


async def send_request(policy: ConnectorPolicy, secret: str, request: ConnectorRequest) -> CapabilityResult:
    if not policy.permits(request):
        return CapabilityResult(success=False, error="Request is outside the connector policy",
                                metadata={"action_status": "not_executed"})
    started = False
    try:
        async with asyncio.timeout(30):
            host = urlsplit(policy.origin).hostname
            address = await public_address(host)
            authority = f"[{address}]" if ":" in address else address
            url = f"https://{authority}{request.path}"
            if request.query:
                url += "?" + urlencode(request.query)
            authorization = f"Bearer {secret}" if policy.scheme == "Bearer" else secret
            headers = {"Host": host, policy.header: authorization,
                       "Content-Type": request.content_type, "Accept-Encoding": "identity"}
            async with httpx.AsyncClient(trust_env=False, timeout=20, follow_redirects=False) as client:
                started = True
                async with client.stream(request.method, url, headers=headers, content=request.body.encode(),
                                         extensions={"sni_hostname": host}) as response:
                    # Do not forward credentials to a redirect, even on the same origin.
                    if response.is_redirect:
                        return CapabilityResult(success=False, error="Connector redirects are not followed. Check the original action before resubmitting a write.",
                                                metadata={"status_code": response.status_code,
                                                          **({"action_status": "unknown"} if request.method != "GET" else {})})
                    if response.headers.get("content-encoding", "identity") != "identity":
                        raise ValueError("Compressed connector response refused")
                    content = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        content.extend(chunk)
                        if len(content) > MAX_RESPONSE:
                            raise ValueError("Connector response exceeds 2 MiB")
                    output = content.decode("utf-8", errors="replace")
                    for value in {secret, quote(secret, safe=""), base64.b64encode(secret.encode()).decode()}:
                        output = output.replace(value, "[redacted]")
                    truncated = len(output) > request.max_length
                    output = output[:request.max_length]
                    if truncated:
                        output += "\n[Truncated; request a smaller result or increase max_length.]"
                    if response.status_code == 202:
                        return CapabilityResult(success=False, output=output,
                                                error="API accepted the request but has not confirmed completion. Check its status; do not resubmit.",
                                                metadata={"status_code": 202, "action_status": "unknown"})
                    uncertain = not response.is_success and request.method != "GET"
                    return CapabilityResult(success=response.is_success, output=output,
                                            error=None if response.is_success else f"HTTP {response.status_code}" + ("; the action may have started. Check its outcome before resubmitting." if uncertain else ""),
                                            metadata={"status_code": response.status_code,
                                                      "connector": policy.name, "truncated": truncated,
                                                      **({"action_status": "unknown"} if uncertain else {})})
    except Exception as exc:
        # Transport errors may contain request headers or credentials. Return only the class.
        return CapabilityResult(success=False, error=f"Connector request failed ({type(exc).__name__})",
                                metadata={"action_status": "unknown" if started else "not_executed"})
