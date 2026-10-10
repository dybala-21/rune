"""Route channel messages, notifications, questions and approvals to the agent."""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from rune.agent.run_outcome import run_outcome
from rune.api.event_logger import append_event
from rune.channels.registry import ChannelRegistry
from rune.channels.types import (
    IncomingMessage,
    OutgoingMessage,
)
from rune.proactive.channel_delivery import ChannelDeliveryManager
from rune.utils.logger import get_logger

log = get_logger(__name__)

# Type aliases

NotificationPriority = Literal["urgent", "high", "medium", "low"]
ApprovalDecision = Literal["approve_once", "approve_always", "deny"]


# Internal data structures


@dataclass(slots=True)
class GatewayNotification:
    """A notification to be routed through the gateway."""

    title: str
    body: str
    priority: NotificationPriority = "medium"
    source: str = ""
    channel: str = ""


@dataclass(slots=True)
class ChannelRoutingRule:
    """Maps a notification priority to a list of preferred channels."""

    priority: NotificationPriority
    channels: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ApprovalResponse:
    """Result of an approval request."""

    decision: ApprovalDecision
    approved: bool
    timed_out: bool = False
    user_guidance: str | None = None


@dataclass(slots=True)
class _PendingAskQuestion:
    """State for a pending ask_user question awaiting channel response."""

    question: str
    options: list[str] | None
    future: asyncio.Future[str]
    expires_at: float


@dataclass(slots=True)
class _PendingApproval:
    """State for a pending approval request awaiting channel response."""

    command: str
    future: asyncio.Future[ApprovalResponse]
    expires_at: float


@dataclass(slots=True)
class _SenderQueue:
    """Per-sender task queue to ensure sequential processing."""

    queue: asyncio.Queue[IncomingMessage] = field(
        default_factory=lambda: asyncio.Queue()
    )
    worker: asyncio.Task[None] | None = None


# Default routing rules

_REALTIME_CHANNELS = [
    "slack",
    "mattermost",
    "telegram",
    "discord",
    "whatsapp",
    "line",
    "googlechat",
    "web",
]

DEFAULT_ROUTING: list[ChannelRoutingRule] = [
    ChannelRoutingRule(priority="urgent", channels=list(_REALTIME_CHANNELS)),
    ChannelRoutingRule(priority="high", channels=list(_REALTIME_CHANNELS)),
    ChannelRoutingRule(
        priority="medium",
        channels=["slack", "mattermost", "telegram", "discord", "whatsapp", "line", "web"],
    ),
    ChannelRoutingRule(priority="low", channels=[]),  # TUI only
]

# Timeouts (seconds)

_ASK_USER_TIMEOUT = 300.0  # 5 minutes
_APPROVAL_TIMEOUT = 60.0  # 1 minute


# ChannelGateway


class ChannelGateway:
    """Serialize each sender's messages while routing channels and proactive notifications."""

    __slots__ = (
        "_registry",
        "_scheduler",
        "_sender_queues",
        "_running",
        "_routing",
        "_default_recipients",
        "_proactive_unsubscribe",
        "_pending_questions",
        "_pending_approvals",
        "_listeners",
        "_delivery_manager",
        "_conv_manager",
    )

    def __init__(
        self,
        channel_registry: ChannelRegistry,
        agent_scheduler: Any = None,
        *,
        routing: list[ChannelRoutingRule] | None = None,
        default_recipients: dict[str, str] | None = None,
    ) -> None:
        self._registry = channel_registry
        self._scheduler = agent_scheduler
        self._sender_queues: dict[str, _SenderQueue] = {}
        self._running = False
        self._routing = routing or list(DEFAULT_ROUTING)
        self._default_recipients: dict[str, str] = dict(default_recipients or {})
        self._proactive_unsubscribe: Callable[[], None] | None = None
        self._pending_questions: dict[str, _PendingAskQuestion] = {}
        self._pending_approvals: dict[str, _PendingApproval] = {}
        self._listeners: dict[str, list[Callable[..., Any]]] = {}
        self._delivery_manager = ChannelDeliveryManager()
        # Reuse one manager to avoid leaked connections and competing in-memory conversations.
        self._conv_manager: Any = None

    # Event emitter (lightweight, ported from TS EventEmitter pattern)

    def on(self, event: str, callback: Callable[..., Any]) -> None:
        """Register a listener for a gateway event."""
        self._listeners.setdefault(event, []).append(callback)

    def off(self, event: str, callback: Callable[..., Any]) -> None:
        """Remove a listener for a gateway event."""
        if event in self._listeners:
            with contextlib.suppress(ValueError):
                self._listeners[event].remove(callback)

    def _emit(self, event: str, *args: Any) -> None:
        """Emit an event to all registered listeners."""
        for cb in self._listeners.get(event, []):
            try:
                cb(*args)
            except Exception as exc:
                log.warning("gateway_event_listener_error", event_name=event, error=str(exc))

    # Lifecycle

    async def start(self) -> None:
        """Subscribe to all registered channels and start processing."""
        if self._running:
            return

        self._running = True

        for name in self._registry.list():
            adapter = self._registry.get(name)
            if adapter is not None:
                adapter.on_message = self._handle_incoming
                # Register channel with delivery manager for smart routing
                self._delivery_manager.register_channel(
                    name, lambda suggestions, a=adapter, n=name: a.send_notification(
                        self._default_recipients.get(n, ""),
                        "\n".join(s.description for s in suggestions),
                    )
                )

        # Subscribe to proactive engine suggestions
        self._subscribe_to_proactive()

        log.info(
            "gateway_started",
            channels=self._registry.list(),
        )
        self._emit("started")

    async def stop(self) -> None:
        """Stop the gateway and cancel all sender workers."""
        if not self._running:
            return

        self._running = False

        # Unsubscribe from proactive engine
        if self._proactive_unsubscribe is not None:
            self._proactive_unsubscribe()
            self._proactive_unsubscribe = None

        # Unsubscribe from channels
        for name in self._registry.list():
            adapter = self._registry.get(name)
            if adapter is not None:
                adapter.on_message = None

        # Cancel all sender queue workers
        for _sender_id, sq in self._sender_queues.items():
            if sq.worker is not None and not sq.worker.done():
                sq.worker.cancel()
        if self._sender_queues:
            await asyncio.gather(
                *(
                    sq.worker
                    for sq in self._sender_queues.values()
                    if sq.worker is not None and not sq.worker.done()
                ),
                return_exceptions=True,
            )
        self._sender_queues.clear()

        # Reject all pending questions and approvals
        for pending in self._pending_questions.values():
            if not pending.future.done():
                pending.future.set_exception(
                    RuntimeError("Gateway stopped while waiting for user response")
                )
        self._pending_questions.clear()

        for pending in self._pending_approvals.values():
            if not pending.future.done():
                pending.future.set_exception(
                    RuntimeError("Gateway stopped while waiting for approval response")
                )
        self._pending_approvals.clear()

        log.info("gateway_stopped")
        self._emit("stopped")

    # Proactive engine integration

    def _subscribe_to_proactive(self) -> None:
        """Forward proactive suggestion events through notification routing."""
        try:
            from rune.proactive.engine import get_proactive_engine
            from rune.proactive.types import Suggestion

            engine = get_proactive_engine()

            def on_suggestion(suggestions: list[Suggestion]) -> None:
                """Handle new suggestions from the proactive engine."""
                from rune.proactive.formatter import format_for_channel

                for suggestion in suggestions:
                    formatted_body = format_for_channel(suggestion, "gateway")
                    notification = GatewayNotification(
                        title="💬 rune",
                        body=formatted_body,
                        priority=_confidence_to_priority(suggestion.confidence),
                        source=suggestion.id,
                    )
                    asyncio.create_task(
                        self._route_notification_safe(notification)
                    )

            def on_intervention(suggestions: list[Suggestion]) -> None:
                """Handle high-confidence intervention suggestions."""
                for suggestion in suggestions:
                    notification = GatewayNotification(
                        title=suggestion.title,
                        body=suggestion.description,
                        priority="urgent",
                        source=suggestion.id,
                    )
                    asyncio.create_task(
                        self._route_notification_safe(notification)
                    )

            engine.on("suggestion", on_suggestion)
            engine.on("intervention", on_intervention)

            self._proactive_unsubscribe = lambda: (
                engine.off("suggestion", on_suggestion),
                engine.off("intervention", on_intervention),
            )

            log.info("gateway_proactive_subscribed")
        except Exception as exc:
            log.warning("gateway_proactive_subscribe_failed", error=str(exc))

    async def _route_notification_safe(self, notification: GatewayNotification) -> None:
        """Route a notification, catching and logging any errors."""
        try:
            await self.route_notification(notification)
        except Exception as exc:
            log.error("gateway_notification_routing_failed", error=str(exc))

    async def handle_suggestion_response(
        self,
        suggestion_id: str,
        response: Literal["accept", "dismiss", "defer", "annoyed"],
    ) -> None:
        """Pass channel feedback to the proactive engine."""
        try:
            from rune.proactive.engine import get_proactive_engine

            engine = get_proactive_engine()
            accepted = response == "accept"
            engine.handle_response(suggestion_id, accepted=accepted)
            log.info(
                "gateway_suggestion_response_forwarded",
                suggestion_id=suggestion_id,
                response=response,
            )
        except Exception as exc:
            log.warning(
                "gateway_suggestion_response_failed",
                suggestion_id=suggestion_id,
                error=str(exc),
            )

    async def route_notification(self, notification: GatewayNotification) -> None:
        """Try preferred channels by priority, then emit a local notification if none delivers."""
        rule = next(
            (r for r in self._routing if r.priority == notification.priority),
            None,
        )
        channels = [notification.channel] if notification.channel else (rule.channels if rule else [])
        if not channels:
            log.debug(
                "gateway_no_route_for_priority",
                priority=notification.priority,
            )
            # TUI-only: emit event for local consumption
            self._emit("notification", notification)
            return

        delivered = False

        for channel_name in channels:
            adapter = self._registry.get(channel_name)
            if adapter is None:
                continue

            recipient_id = self._default_recipients.get(channel_name)
            if not recipient_id:
                log.warning(
                    "gateway_no_default_recipient",
                    channel=channel_name,
                )
                continue

            try:
                text = f"{notification.title}\n{notification.body}" if notification.body else notification.title
                await adapter.send_notification(recipient_id, text)
                delivered = True
                log.info(
                    "gateway_notification_delivered",
                    channel=channel_name,
                    priority=notification.priority,
                )
                break  # Delivered to first available channel
            except Exception as exc:
                log.error(
                    "gateway_notification_delivery_failed",
                    channel=channel_name,
                    error=str(exc),
                )

        if not delivered:
            # No channel accepted the notification - emit for TUI fallback
            self._emit("notification", notification)

    # Orchestrator event relay

    def wire_orchestrator(self, orchestrator: Any) -> None:
        """Forward orchestrator events through notification routing."""

        async def _on_plan(plan: Any) -> None:
            tc = len(plan.tasks) if hasattr(plan, "tasks") else 0
            self._emit("orchestration_started", tc, getattr(plan, "description", ""))

        async def _on_progress(
            completed: int, total: int, task_id: str, success: bool,
        ) -> None:
            self._emit("orchestration_progress", completed, total, task_id, success)

        async def _on_completed(result: Any) -> None:
            results = getattr(result, "results", [])
            ok = sum(1 for r in results if getattr(r, "success", False))
            fail = len(results) - ok
            duration_ms = getattr(result, "duration_ms", 0.0)
            success = getattr(result, "success", False)

            # Route summary as notification on external channels
            status = "completed" if success else "failed"
            await self.route_notification(GatewayNotification(
                title=f"Orchestration {status}",
                body=f"{ok} tasks succeeded, {fail} failed ({duration_ms / 1000:.1f}s)",
                priority="medium" if success else "high",
                source="orchestrator",
            ))

        orchestrator.on("plan_ready", _on_plan)
        orchestrator.on("progress", _on_progress)
        orchestrator.on("completed", _on_completed)

    # Ask-user pipeline

    async def ask_user(
        self,
        session_key: str,
        channel_name: str,
        recipient_id: str,
        question: str,
        options: list[str] | None = None,
        timeout: float = _ASK_USER_TIMEOUT,
    ) -> str:
        """Ask through a channel and consume its next reply; raise TimeoutError if unanswered."""
        adapter = self._registry.get(channel_name)
        if adapter is None:
            raise RuntimeError(f"Channel adapter not available: {channel_name}")

        # If the adapter natively supports ask_question, use it directly
        try:
            answer = await asyncio.wait_for(
                adapter.ask_question(recipient_id, question, options, timeout),
                timeout=timeout,
            )
            return answer
        except NotImplementedError:
            pass  # Fall through to text-based fallback

        # Text-based fallback: send prompt and wait for next message
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()

        # Replace any existing pending question for this session
        prev = self._pending_questions.pop(session_key, None)
        if prev is not None and not prev.future.done():
            prev.future.set_exception(
                RuntimeError("Replaced by a newer ask_user request")
            )

        self._pending_questions[session_key] = _PendingAskQuestion(
            question=question,
            options=options,
            future=future,
            expires_at=time.monotonic() + timeout,
        )

        # Build and send the text prompt
        prompt = _build_ask_user_prompt(question, options)
        try:
            await adapter.send(recipient_id, OutgoingMessage(text=prompt))
        except Exception as exc:
            self._pending_questions.pop(session_key, None)
            raise RuntimeError(f"Failed to send ask_user prompt: {exc}") from exc

        # Wait for the user's reply (or timeout)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            self._pending_questions.pop(session_key, None)
            raise TimeoutError("ask_user timed out waiting for channel response") from None

    def _resolve_pending_question(self, session_key: str, text: str) -> bool:
        """Return True when user text resolves a pending question."""
        pending = self._pending_questions.get(session_key)
        if pending is None:
            return False

        # Check expiry
        if time.monotonic() > pending.expires_at:
            self._pending_questions.pop(session_key, None)
            if not pending.future.done():
                pending.future.set_exception(
                    TimeoutError("User response timeout exceeded")
                )
            return False

        answer = text.strip()
        if not answer:
            # Ignore blank messages while waiting for an answer
            return True

        self._pending_questions.pop(session_key, None)
        if not pending.future.done():
            pending.future.set_result(answer)
        log.info(
            "gateway_ask_user_resolved",
            session_key=session_key,
        )
        return True

    # Approval pipeline

    async def request_approval(
        self,
        session_key: str,
        channel_name: str,
        recipient_id: str,
        *,
        command: str,
        risk_level: str = "medium",
        reason: str = "",
        suggestions: list[str] | None = None,
        timeout: float = _APPROVAL_TIMEOUT,
    ) -> ApprovalResponse:
        """Request approval through the native channel UI, falling back to numbered text choices."""
        adapter = self._registry.get(channel_name)
        if adapter is None:
            return ApprovalResponse(decision="deny", approved=False, timed_out=False)

        # Try native adapter approval first
        try:
            approval_id = f"approval:{session_key}:{int(time.monotonic() * 1000)}"
            await adapter.send_approval(recipient_id, command, approval_id)
            # Native approval still resolves through the pending-response pipeline.
        except NotImplementedError:
            pass  # Fall through to text prompt

        # Text-based approval fallback
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ApprovalResponse] = loop.create_future()

        prev = self._pending_approvals.pop(session_key, None)
        if prev is not None and not prev.future.done():
            prev.future.set_exception(
                RuntimeError("Replaced by a newer approval request")
            )

        self._pending_approvals[session_key] = _PendingApproval(
            command=command,
            future=future,
            expires_at=time.monotonic() + timeout,
        )

        prompt = _build_approval_prompt(
            command=command,
            risk_level=risk_level,
            reason=reason,
            suggestions=suggestions,
            timeout_secs=int(timeout),
        )

        try:
            await adapter.send(recipient_id, OutgoingMessage(text=prompt))
        except Exception as exc:
            self._pending_approvals.pop(session_key, None)
            log.warning("gateway_approval_prompt_failed", error=str(exc))
            return ApprovalResponse(decision="deny", approved=False, timed_out=False)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            self._pending_approvals.pop(session_key, None)
            return ApprovalResponse(decision="deny", approved=False, timed_out=True)

    def _resolve_pending_approval(self, session_key: str, text: str) -> bool:
        """Return True when user input resolves a pending approval."""
        pending = self._pending_approvals.get(session_key)
        if pending is None:
            return False

        if time.monotonic() > pending.expires_at:
            self._pending_approvals.pop(session_key, None)
            if not pending.future.done():
                pending.future.set_result(
                    ApprovalResponse(decision="deny", approved=False, timed_out=True)
                )
            return True

        answer = text.strip()
        if not answer:
            return True

        parsed = _parse_approval_response(answer)
        if parsed is None:
            # Leave unrecognized replies unconsumed so the user can try again.
            log.debug("gateway_approval_parse_failed", input=answer[:80])
            return True  # Still consumed - don't route to agent

        self._pending_approvals.pop(session_key, None)
        if not pending.future.done():
            pending.future.set_result(parsed)
        log.info(
            "gateway_approval_resolved",
            session_key=session_key,
            decision=parsed.decision,
        )
        return True

    # Connected channels / recipients

    def get_connected_channels(self) -> list[str]:
        """Return the names of all currently registered channels."""
        return self._registry.list()

    def set_default_recipient(self, channel_name: str, recipient_id: str) -> None:
        """Set the default recipient for proactive notifications on a channel."""
        self._default_recipients[channel_name] = recipient_id
        log.info(
            "gateway_default_recipient_set",
            channel=channel_name,
            recipient_id=recipient_id,
        )

    async def send_to_channel(
        self, channel_name: str, recipient_id: str, content: str
    ) -> None:
        """Send a message directly to a channel."""
        adapter = self._registry.get(channel_name)
        if adapter is None:
            raise RuntimeError(f"Channel not available: {channel_name}")
        await adapter.send(recipient_id, OutgoingMessage(text=content))

    # Message handling

    async def _handle_incoming(self, message: IncomingMessage) -> None:
        """Route an incoming message to the appropriate sender queue."""
        if not self._running:
            return

        sender_key = f"{message.channel_id}:{message.sender_id}"

        # Auto-register default recipient for proactive notifications
        channel_name = message.metadata.get("channel_name", message.channel_id)
        if channel_name not in self._default_recipients:
            default_recipient = (
                message.metadata.get("default_recipient")
                or message.metadata.get("reply_to")
                or message.sender_id
            )
            self._default_recipients[channel_name] = str(default_recipient)
            log.info(
                "gateway_auto_registered_recipient",
                channel=channel_name,
                recipient_id=str(default_recipient),
            )

        # Check if this message resolves a pending approval
        if self._resolve_pending_approval(sender_key, message.text):
            return

        # Check if this message resolves a pending ask_user question
        if self._resolve_pending_question(sender_key, message.text):
            return

        # Event logging - message received
        _log_event(
            message.metadata.get("conversation_id", sender_key),
            "message_received",
            {
                "channel": message.channel_id,
                "sender": message.sender_id,
                "text_length": len(message.text),
                "has_attachments": bool(message.attachments),
            },
        )

        # Normal message - enqueue for agent processing
        if sender_key not in self._sender_queues:
            sq = _SenderQueue()
            sq.worker = asyncio.create_task(
                self._sender_worker(sender_key, sq.queue)
            )
            self._sender_queues[sender_key] = sq

        await self._sender_queues[sender_key].queue.put(message)
        log.debug(
            "message_enqueued",
            sender=sender_key,
            queue_size=self._sender_queues[sender_key].queue.qsize(),
        )

    async def _sender_worker(
        self, sender_key: str, queue: asyncio.Queue[IncomingMessage]
    ) -> None:
        """Process messages for a single sender sequentially."""
        try:
            while self._running:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=300.0)
                except TimeoutError:
                    # No messages for 5 minutes - clean up this worker
                    break

                channel_name = message.metadata.get("channel_name", "")
                try:
                    response_text = await self._execute_for_message(message)
                    await self._send_response(
                        message.channel_id,
                        message.sender_id,
                        response_text,
                        reply_to=message.metadata.get("message_id")
                        or message.metadata.get("ts"),
                        channel_name=channel_name,
                    )
                except Exception as exc:
                    log.error(
                        "message_processing_failed",
                        sender=sender_key,
                        error=str(exc),
                    )
                    await self._send_response(
                        message.channel_id,
                        message.sender_id,
                        "An error occurred while processing your request. Please try again.",
                        channel_name=channel_name,
                    )
        except asyncio.CancelledError:
            pass
        finally:
            self._sender_queues.pop(sender_key, None)

    async def _execute_for_message(self, message: IncomingMessage) -> str:
        """Run the message with attachments and memory, then format the reply."""
        sender_key = f"{message.channel_id}:{message.sender_id}"
        conv_id = message.metadata.get("conversation_id", sender_key)
        run_id = f"{int(time.time() * 1000)}-{uuid4().hex[:6]}"

        if self._scheduler is not None:
            _log_event(conv_id, "agent_start", {"goal": message.text}, run_id=run_id)
            try:
                result = await self._scheduler.execute(
                    goal=message.text,
                    sender_id=message.sender_id,
                )
                response = result if isinstance(result, str) else str(result)
            except Exception as exc:
                log.error("scheduler_execute_failed", error=str(exc))
                _log_event(conv_id, "agent_error", {"error": str(exc)[:200]}, run_id=run_id)
                return "An error occurred while processing your request. Please try again."
            response = _truncate_for_channel(response, message.metadata.get("channel_name", ""))
            _log_event(conv_id, "agent_complete", {"success": True, "answer_length": len(response)}, run_id=run_id)
            return response

        try:
            from rune.agent.agent_context import (
                PostProcessInput,
                PrepareContextOptions,
                post_process_agent_result,
                prepare_agent_context,
                resolve_assistant_answer,
            )
            from rune.agent.loop import NativeAgentLoop

            # 1. Check for proactive suggestion feedback (approve:/deny:)
            text = message.text.strip()
            if text.startswith("approve:") or text.startswith("deny:"):
                return await self._handle_proactive_feedback(text)

            _log_event(conv_id, "agent_start", {"goal": text}, run_id=run_id)
            start_time = time.monotonic()

            # 2. Download attachments to temp files
            attachment_paths: list[str] = []
            if message.attachments:
                # Resolve adapter for file_id downloads (e.g. Telegram)
                _adapter = None
                channel_name = message.metadata.get("channel_name", "")
                if channel_name:
                    _adapter = self._registry.get(channel_name)
                attachment_paths = await _download_attachments(
                    message.attachments,
                    message.channel_id,
                    message.metadata,
                    adapter=_adapter,
                )
                if attachment_paths:
                    log.info(
                        "gateway_attachments_downloaded",
                        count=len(attachment_paths),
                    )

            # 3. Resolve the multi-turn conversation for this sender so channel
            conv_manager = await self._get_conv_manager()
            if conv_manager is not None:
                try:
                    from rune.conversation.types import Conversation
                    if conv_id not in conv_manager._active:
                        existing = await conv_manager._store.load(conv_id)
                        conv_manager._active[conv_id] = existing or Conversation(
                            id=conv_id, user_id=message.sender_id or "channel")
                except Exception as exc:
                    log.debug("gateway_conv_load_failed", error=str(exc)[:100])
                    conv_manager = None

            # Save this turn; history omits it because the goal is passed separately.
            if conv_manager is not None:
                with contextlib.suppress(Exception):
                    conv_manager.add_turn(conv_id, "user", text)

            # 4. Prepare agent context with identity + conversation history
            context_opts = PrepareContextOptions(
                goal=text,
                channel=message.metadata.get("channel_name", "remote"),
                sender_id=message.sender_id,
                conversation_id=conv_id,
            )
            agent_ctx = await prepare_agent_context(
                context_opts, conversation_manager=conv_manager)
            conv_id = getattr(agent_ctx, "conversation_id", conv_id) or conv_id

            # 4. Build memory context (best-effort)
            memory_context = ""
            try:
                from rune.memory.manager import get_memory_manager
                manager = get_memory_manager()
                memory_context = await manager.build_memory_context(text)
            except Exception as exc:
                log.debug("gateway_memory_context_failed", error=str(exc)[:100])

            # 5. Run agent with context + callbacks
            loop = NativeAgentLoop()

            # Wire approval callback via gateway's channel pipeline
            channel_name = message.metadata.get("channel_name", "")
            session_key = f"{message.channel_id}:{message.sender_id}"

            async def _gw_approval_cb(command: str, risk_level: str) -> bool:
                try:
                    result = await self.request_approval(
                        session_key,
                        channel_name,
                        message.sender_id,
                        command=command,
                        risk_level=risk_level,
                    )
                    return result.approved
                except Exception as exc:
                    log.warning("gateway_approval_cb_error", error=str(exc)[:100])
                    return False

            loop.set_approval_callback(_gw_approval_cb)

            # Wire ask_user callback via gateway's channel pipeline
            from rune.capabilities.ask_user import AskUserParams, UserResponse, user_response

            async def _gw_ask_user_cb(
                params: AskUserParams,
            ) -> UserResponse:
                labels = [option.label for option in params.options or []]
                answer = await self.ask_user(
                    session_key, channel_name, message.sender_id,
                    params.question, options=labels or None,
                )
                index = labels.index(answer) if answer in labels else None
                return user_response(params, answer, index)

            loop.set_ask_user_callback(_gw_ask_user_cb)

            # Keep the last step's text so the reply does not repeat intermediate narration.
            collected: list[str] = []
            _prev_steps_text: list[str] = []

            async def _collect_text(delta: str) -> None:
                collected.append(delta)

            async def _on_new_step(step: int) -> None:
                if collected:
                    _prev_steps_text.clear()
                    _prev_steps_text.extend(collected)
                    collected.clear()

            loop.on("text_delta", _collect_text)
            loop.on("step", _on_new_step)

            context_dict: dict[str, Any] = {}
            if agent_ctx.workspace_root:
                context_dict["cwd"] = agent_ctx.workspace_root
            if memory_context:
                context_dict["memory_context"] = memory_context
            if message.sender_id:
                context_dict["sender_id"] = message.sender_id
            if attachment_paths:
                context_dict["attachment_paths"] = attachment_paths

            trace = await loop.run(
                text,
                context=context_dict if context_dict else None,
                message_history=agent_ctx.messages if agent_ctx.messages else None,
            )

            # Use the final step's text instead of repeating earlier narration.
            last_text = "".join(collected)
            if last_text.strip():
                response = last_text
            elif _prev_steps_text:
                response = "".join(_prev_steps_text)
            else:
                reason = trace.reason or ""
                if reason.startswith("error:"):
                    response = "An error occurred while processing your request."
                else:
                    response = _USER_FRIENDLY_REASON.get(
                        reason, reason or "Task completed."
                    )

            duration_ms = int((time.monotonic() - start_time) * 1000)

            # Persist the final answer without channel-specific notes for the next turn.
            if conv_manager is not None:
                with contextlib.suppress(Exception):
                    assistant_answer = resolve_assistant_answer(
                        getattr(loop, "_last_answer_text", ""), response)
                    if not assistant_answer:
                        # Save a placeholder so the next turn does not repeat this work.
                        assistant_answer = (
                            "(the run ended without a textual answer: "
                            f"{trace.reason or 'unknown'})"
                        )
                    conv_manager.add_turn(
                        conv_id, "assistant", assistant_answer,
                        goal_type=getattr(loop, "_last_goal_type", ""),
                    )
                    await conv_manager._store.save(
                        conv_manager._active[conv_id])

            # 6. Post-process (memory persistence)
            try:
                learned = await post_process_agent_result(PostProcessInput(
                    classification_hint=getattr(loop, "_last_goal_type", "") or None,
                    verification=getattr(trace, "verification", None),
                    reason=getattr(trace, "reason", ""),
                    mech_check=getattr(trace, "mech_check", ""),
                    evidence_gate=getattr(trace, "evidence_gate", None),
                    context=agent_ctx,
                    success=run_outcome(trace).success,
                    answer=response,
                    duration_ms=duration_ms,
                ))
                from rune.agent.memory_bridge import format_learned_rules_note
                _ln = format_learned_rules_note(learned)
                if _ln:
                    response = f"{_ln}\n\n{response}"
            except Exception as exc:
                log.debug("gateway_post_process_failed", error=str(exc)[:100])

            # 7. Truncate response per channel max length
            response = _truncate_for_channel(
                response,
                message.metadata.get("channel_name", ""),
            )

            _log_event(
                conv_id,
                "agent_complete",
                {"success": run_outcome(trace).success, "answer_length": len(response), "duration_ms": duration_ms},
                run_id=run_id,
            )

            return response

        except Exception as exc:
            log.error("gateway_agent_error", error=str(exc))
            _log_event(conv_id, "agent_error", {"error": str(exc)[:200]}, run_id=run_id)
            # Sanitize error - don't expose stack traces to end users
            return "An error occurred while processing your request. Please try again."

    async def _get_conv_manager(self) -> Any:
        """Reuse a conversation manager, or continue without memory if the store cannot open."""
        if self._conv_manager is not None:
            return self._conv_manager
        try:

            from rune.conversation.manager import ConversationManager
            from rune.conversation.store import ConversationStore
            from rune.utils.paths import conversations_db_path

            store = ConversationStore(conversations_db_path())
            self._conv_manager = ConversationManager(store)
        except Exception as exc:
            log.debug("gateway_conv_manager_init_failed", error=str(exc)[:100])
            self._conv_manager = None
        return self._conv_manager

    async def _handle_proactive_feedback(self, text: str) -> str:
        """Handle approve:/deny: prefix messages for proactive suggestions."""
        try:
            from rune.memory.store import get_memory_store
            from rune.proactive.engine import get_proactive_engine
            engine = get_proactive_engine()
            engine.load_persisted_suggestions(get_memory_store())

            if text.startswith("approve:"):
                suggestion_id = text[len("approve:"):].strip()
                if engine.handle_response(suggestion_id, accepted=True):
                    return "Suggestion approved and queued."
                return "Suggestion not found."
            elif text.startswith("deny:"):
                suggestion_id = text[len("deny:"):].strip()
                if engine.handle_response(suggestion_id, accepted=False):
                    return "Suggestion dismissed."
                return "Suggestion not found."
        except Exception as exc:
            log.warning("proactive_feedback_error", error=str(exc))
        return "The response could not be recorded. Refresh the suggestion and try again."

    async def _send_response(
        self,
        channel_id: str,
        sender_id: str,
        response: str,
        *,
        reply_to: str | None = None,
        channel_name: str = "",
    ) -> None:
        """Reply on the originating channel, trying other adapters if it is absent or fails."""
        outgoing = OutgoingMessage(
            text=response,
            reply_to=reply_to,
        )

        # Try the originating adapter first
        if channel_name:
            adapter = self._registry.get(channel_name)
            if adapter is not None:
                try:
                    await adapter.send(channel_id, outgoing)
                    _log_event(
                        f"{channel_id}:{sender_id}",
                        "response_sent",
                        {
                            "channel": channel_name,
                            "response_length": len(response),
                        },
                    )
                    return
                except Exception:
                    log.warning(
                        "primary_adapter_send_failed",
                        channel=channel_name,
                        channel_id=channel_id,
                    )

        # Fallback: try remaining adapters
        for name in self._registry.list():
            if name == channel_name:
                continue  # already tried
            adapter = self._registry.get(name)
            if adapter is not None:
                try:
                    await adapter.send(channel_id, outgoing)
                    _log_event(
                        f"{channel_id}:{sender_id}",
                        "response_sent",
                        {
                            "channel": name,
                            "response_length": len(response),
                        },
                    )
                    return
                except Exception:
                    continue

        log.warning(
            "no_adapter_for_channel",
            channel_id=channel_id,
            sender_id=sender_id,
        )


# Module-level helpers


def _confidence_to_priority(confidence: float) -> NotificationPriority:
    """Map a suggestion confidence score to a notification priority."""
    if confidence >= 0.8:
        return "urgent"
    if confidence >= 0.6:
        return "high"
    if confidence >= 0.4:
        return "medium"
    return "low"


def _build_ask_user_prompt(question: str, options: list[str] | None) -> str:
    """Build a text prompt for the ask_user fallback path."""
    lines: list[str] = [question]
    if options:
        lines.append("")
        for i, option in enumerate(options, 1):
            lines.append(f"{i}) {option}")
        lines.append("")
        lines.append(
            "Reply with your answer (e.g. the number, option text, or a custom response)."
        )
        lines.append("To abort the current task, send /abort.")
    return "\n".join(lines)


def _build_approval_prompt(
    *,
    command: str,
    risk_level: str,
    reason: str,
    suggestions: list[str] | None,
    timeout_secs: int,
) -> str:
    """Build the text fallback for an approval request."""
    cmd_display = command if len(command) <= 220 else f"{command[:220]}..."
    lines: list[str] = [
        "Approval Required",
        "",
        f"Command: {cmd_display}",
        f"Risk level: {risk_level}",
    ]
    if reason:
        lines.append(f"Reason: {reason}")
    if suggestions:
        lines.append("")
        lines.append("Notes:")
        for s in suggestions[:3]:
            lines.append(f"- {s}")
    lines.append("")
    lines.append("Reply:")
    lines.append("1) Approve once")
    lines.append("2) Always approve")
    lines.append("3) Deny")
    lines.append("4) Deny + guidance (e.g. '4 use a safer alternative')")
    lines.append("")
    lines.append(
        f"Time limit: {timeout_secs}s. Auto-denied if no response."
    )
    lines.append("Send /abort to cancel the current task.")
    return "\n".join(lines)


def _parse_approval_response(text: str) -> ApprovalResponse | None:
    """Parse a text reply into an ApprovalResponse."""
    trimmed = text.strip()
    normalized = trimmed.lower()

    if normalized in {"1", "y", "yes", "allow", "approve"}:
        return ApprovalResponse(decision="approve_once", approved=True, timed_out=False)

    if normalized in {"2", "a", "always", "always allow"}:
        return ApprovalResponse(decision="approve_always", approved=True, timed_out=False)

    if normalized in {"3", "n", "no", "deny"}:
        return ApprovalResponse(decision="deny", approved=False, timed_out=False)

    if normalized == "4" or normalized.startswith("4 "):
        guidance = trimmed[1:].strip() or None
        return ApprovalResponse(
            decision="deny",
            approved=False,
            timed_out=False,
            user_guidance=guidance,
        )

    return None


# Channel-aware response truncation

# Maximum message lengths per channel
_CHANNEL_MAX_LENGTH: dict[str, int] = {
    "telegram": 4096,
    "discord": 2000,
    "slack": 4000,
    "googlechat": 8000,
    "whatsapp": 4096,
    "mattermost": 16383,
    "line": 5000,
}

_USER_FRIENDLY_REASON: dict[str, str] = {
    "completed": "Task completed.",
    "cancelled": "Request cancelled.",
    "stalled": "I couldn't make further progress. Please try rephrasing your request.",
    "max_iterations": "I've reached my step limit. Here's what I found so far.",
    "token_budget_exhausted": "I ran out of processing budget. Please try a simpler request.",
    "request_budget_exhausted": "The configured model budget stopped further requests. Some work may be unfinished.",
}


def _log_event(
    conversation_id: str,
    event: str,
    data: dict[str, Any],
    *,
    run_id: str = "",
) -> None:
    """Append an event to the JSONL event logger (best-effort, never raises)."""
    try:
        append_event(
            conversation_id,
            run_id or "gateway",
            {
                "event": event,
                "data": data,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
    except Exception:
        pass  # Logging failures must not affect message processing


async def _download_attachments(
    attachments: list[dict[str, Any]],
    channel_id: str,
    metadata: dict[str, Any],
    adapter: Any = None,
) -> list[str]:
    """Save base64, URL or adapter-downloaded attachments and return their local paths."""
    import base64

    paths: list[str] = []
    tmp_dir = os.path.join(tempfile.gettempdir(), "rune-attachments")
    os.makedirs(tmp_dir, exist_ok=True)

    raw_attachments = metadata.get("raw_attachments")

    for idx, att in enumerate(attachments):
        try:
            filename = att.get("filename", f"attachment_{idx}")
            dest = os.path.join(tmp_dir, f"{uuid4().hex[:8]}_{filename}")

            # Strategy 1: base64 data from API channel
            if raw_attachments and idx < len(raw_attachments):
                raw = raw_attachments[idx]
                b64_data = raw.get("data") if isinstance(raw, dict) else None
                if b64_data and isinstance(b64_data, str):
                    data = base64.b64decode(b64_data)
                    await asyncio.to_thread(_write_bytes, dest, data)
                    paths.append(dest)
                    continue

            # Strategy 2: URL-based download (Discord, Slack, etc.)
            url = att.get("url")
            if url and isinstance(url, str):
                data = await _fetch_url_bytes(url)
                if data:
                    await asyncio.to_thread(_write_bytes, dest, data)
                    paths.append(dest)
                    continue

            # Strategy 3: file_id via adapter.download_file() (Telegram, etc.)
            file_id = att.get("file_id")
            if file_id:
                if adapter is not None and hasattr(adapter, "download_file"):
                    data = await adapter.download_file(file_id)
                    if data:
                        await asyncio.to_thread(_write_bytes, dest, data)
                        paths.append(dest)
                        continue
                log.debug(
                    "gateway_attachment_file_id_no_adapter",
                    file_id=file_id,
                    channel=channel_id,
                )
        except Exception as exc:
            log.warning(
                "gateway_attachment_download_failed",
                index=idx,
                error=str(exc)[:100],
            )

    return paths


def _write_bytes(path: str, data: bytes) -> None:
    """Write bytes to a file (sync helper for asyncio.to_thread)."""
    with open(path, "wb") as f:
        f.write(data)


async def _fetch_url_bytes(url: str) -> bytes | None:
    """Fetch a URL and return its content as bytes (best-effort)."""
    try:
        import urllib.request

        def _do_fetch() -> bytes:
            req = urllib.request.Request(url, headers={"User-Agent": "RUNE/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()

        return await asyncio.to_thread(_do_fetch)
    except Exception as exc:
        log.warning("gateway_url_fetch_failed", url=url[:120], error=str(exc)[:80])
        return None


def _truncate_for_channel(text: str, channel_name: str) -> str:
    """Truncate response text to the channel's maximum message length."""
    max_len = _CHANNEL_MAX_LENGTH.get(channel_name.lower(), 8000)
    if len(text) <= max_len:
        return text
    # Keep first portion and add truncation notice
    suffix = "\n\n... (response truncated)"
    cut = max(1, max_len - len(suffix))
    return text[:cut] + suffix


# Module-level singleton for gateway access from other subsystems (e.g. cron)

_gateway_instance: ChannelGateway | None = None


def set_gateway(gw: ChannelGateway) -> None:
    """Set the module-level gateway singleton (called during daemon init)."""
    global _gateway_instance  # noqa: PLW0603
    _gateway_instance = gw


def get_gateway() -> ChannelGateway | None:
    """Return the gateway singleton, or None if not initialized."""
    return _gateway_instance
