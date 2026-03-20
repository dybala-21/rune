"""Main daemon server process for RUNE.

Ported from src/daemon/daemon.ts - background service that hosts the
proactive engine, pattern learner, memory/conversation stores, and
an agent loop for background tasks.  Communicates over a Unix domain
socket.

Startup order (matches TS):
 1. Process lock acquisition
 2. Memory store init
 3. Conversation store init
 4. Orphaned run cleanup
 5. Reflexion learner init
 6. Pattern learner init
 7. Proactive engine start
 7a. Heartbeat scheduler start (early - needed by proactive subsystems)
 7b. Proactive subsystems (EngagementTracker, ConversationInitiator,
     EnvironmentSensor, Heartbeat-to-Engine wiring)
 8. MCP bridge init
 9. Autonomous executor init
10. Channel adapters start
11. API server start (non-blocking)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

from rune.utils.fast_serde import json_decode, json_encode
from rune.utils.logger import get_logger

log = get_logger(__name__)

VERSION = "0.1.0"

_HEARTBEAT_INTERVAL_S = 30.0
_EXECUTOR_FLUSH_INTERVAL_S = 300.0  # 5 minutes


# Default configuration

def _default_rune_dir() -> Path:
    """Resolve ``~/.rune/data``."""
    return Path.home() / ".rune" / "data"


def _default_config() -> dict[str, Any]:
    rune_dir = _default_rune_dir()
    return {
        "socket_path": str(rune_dir / "rune.sock"),
        "pid_file": str(rune_dir / "rune.pid"),
        "log_file": str(rune_dir / "daemon.log"),
        "heartbeat_file": str(rune_dir / "heartbeat"),
        "proactive_enabled": True,
        "check_interval_ms": 60_000,  # 1 minute
        "api_host": os.environ.get("RUNE_API_HOST", "127.0.0.1"),
        "api_port": int(os.environ.get("RUNE_API_PORT", "18789")),
        "api_enabled": os.environ.get("RUNE_API_ENABLED", "").lower()
        in ("1", "true", "yes"),
    }


# EngagementStore adapter (bridges MemoryStore to EngagementTracker)

class _EngagementStoreAdapter:
    """Thin adapter mapping ``MemoryStore`` methods to the
    ``EngagementStore`` protocol expected by ``EngagementTracker``.

    Mirrors the TS ``engagementStoreAdapter`` object created in daemon.ts
    (lines 321-341).
    """

    __slots__ = ("_store",)

    def __init__(self, store: Any) -> None:
        self._store = store

    # -- EngagementStore protocol methods -----------------------------------

    def get_engagement_metrics(self, user_id: str) -> Any:
        from rune.proactive.engagement_tracker import _metrics_from_dict
        raw = self._store.get_user_engagement_metrics(user_id)
        if raw is None:
            return None
        return _metrics_from_dict(raw)

    def store_engagement_metrics(self, user_id: str, data: Any) -> None:
        from rune.proactive.engagement_tracker import _metrics_to_dict
        self._store.store_user_engagement_metrics(user_id, _metrics_to_dict(data))

    def get_channel_preferences(self, user_id: str) -> Any:
        from rune.proactive.engagement_tracker import _channel_pref_from_dict
        raw = self._store.get_user_channel_preferences(user_id)
        if raw is None:
            return None
        return _channel_pref_from_dict(raw)

    def store_channel_preferences(self, user_id: str, data: Any) -> None:
        from rune.proactive.engagement_tracker import _channel_pref_to_dict
        self._store.store_user_channel_preferences(user_id, _channel_pref_to_dict(data))

    def store_conversation_record(self, record: Any) -> None:
        from dataclasses import asdict
        rec_dict = asdict(record) if hasattr(record, "__dataclass_fields__") else record
        self._store.store_proactive_conversation_record(rec_dict)


# Daemon server

class RuneDaemon:
    """Unix-domain-socket daemon that orchestrates background RUNE services.

    Lifecycle:
    1. ``start()`` - load env, acquire lock, initialise subsystems, open socket.
    2. Handle incoming JSON commands (execute, status, stop, ...).
    3. ``stop()`` - graceful shutdown of all subsystems.
    """

    def __init__(self, config: dict[str, Any] | None = None, install_signal_handlers: bool = True) -> None:
        self._config = config or _default_config()
        self._install_signal_handlers = install_signal_handlers
        self._server: asyncio.AbstractServer | None = None
        self._running = False
        self._start_time: float | None = None

        # Subsystem references (initialised in start())
        self._proactive_engine: Any = None
        self._pattern_learner: Any = None
        self._memory_store: Any = None
        self._conversation_store: Any = None
        self._reflexion_learner: Any = None
        self._mcp_bridge: Any = None
        self._mcp_clients: list[Any] = []
        self._autonomous_executor: Any = None
        self._channel_registry: Any = None
        self._heartbeat_scheduler: Any = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._api_server_task: asyncio.Task[None] | None = None
        self._executor_flush_task: asyncio.Task[None] | None = None

        # Proactive subsystem references (initialised in _init_proactive_subsystems)
        self._engagement_tracker: Any = None
        self._conversation_initiator: Any = None
        self._environment_sensor: Any = None

        # Browser subsystem references
        self._relay_server: Any = None
        self._browser_page_pool: Any = None

        # Process lock path
        self._lock_path: Path | None = None

    # -- public API ---------------------------------------------------------

    async def start(self) -> None:
        """Initialise subsystems and start the Unix socket server."""
        log.info("daemon_starting", version=VERSION)

        # Load environment (.rune/.env etc.)
        self._load_env()

        # Ensure data directory exists
        rune_dir = _default_rune_dir()
        rune_dir.mkdir(parents=True, exist_ok=True)

        # 1. Process lock acquisition
        await self._acquire_process_lock()

        # Write PID file
        pid_path = Path(self._config["pid_file"])
        pid_path.write_text(str(os.getpid()))

        # Initialise subsystems (2-12)
        await self._init_subsystems()

        # Start Unix socket server
        socket_path = self._config["socket_path"]
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=socket_path,
        )
        self._running = True
        self._start_time = time.monotonic()
        log.info("daemon_started", socket=socket_path, pid=os.getpid())

        # Install signal handlers (skip when caller manages signals, e.g. rune web)
        if self._install_signal_handlers:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.stop(s)))

    async def stop(self, sig: signal.Signals | None = None) -> None:
        """Gracefully shut down all subsystems."""
        if not self._running:
            return
        self._running = False
        log.info("daemon_stopping", signal=sig.name if sig else "manual")

        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Shutdown subsystems in reverse-dependency order
        await self._shutdown_subsystems()

        # Remove socket and PID file
        for path_key in ("socket_path", "pid_file"):
            p = Path(self._config[path_key])
            if p.exists():
                p.unlink(missing_ok=True)

        # Release process lock
        await self._release_process_lock()

        log.info("daemon_stopped")

    async def serve_forever(self) -> None:
        """Convenience: start and run until stopped."""
        await self.start()
        if self._server:
            async with self._server:
                await self._server.serve_forever()

    # -- process lock -------------------------------------------------------

    async def _acquire_process_lock(self) -> None:
        """Acquire the daemon PID-based process lock."""
        try:
            from rune.daemon.process_lock import acquire_lock
            self._lock_path = acquire_lock()
            log.info("process_lock_acquired", path=str(self._lock_path))
        except RuntimeError as exc:
            log.error("process_lock_failed", error=str(exc))
            raise
        except Exception as exc:
            log.warning("process_lock_unavailable", error=str(exc))

    async def _release_process_lock(self) -> None:
        """Release the daemon process lock."""
        try:
            from rune.daemon.process_lock import release_lock
            release_lock(self._lock_path)
            log.info("process_lock_released")
        except Exception as exc:
            log.warning("process_lock_release_failed", error=str(exc))

    # -- subsystem init / shutdown ------------------------------------------

    def _load_env(self) -> None:
        """Load .rune/.env file if present."""
        try:
            from rune.utils.env import load_env
            load_env()
        except ImportError:
            log.debug("env_loader_not_available")

    async def _init_subsystems(self) -> None:
        """Initialise all subsystems in the correct dependency order.

        Order:
        2. Memory store
        3. Conversation store
        4. Orphaned run cleanup
        5. Reflexion learner
        6. Pattern learner
        7. Proactive engine
        7a. Heartbeat scheduler
        7b. Proactive subsystems (EngagementTracker, ConversationInitiator,
            EnvironmentSensor, Heartbeat-to-Engine wiring)
        8. MCP bridge
        9. Autonomous executor
        10. Channel adapters
        11. API server
        """

        # 2. Memory store init
        try:
            from rune.memory.store import get_memory_store
            self._memory_store = get_memory_store()
            if hasattr(self._memory_store, "initialize"):
                await self._memory_store.initialize()
            log.info("subsystem_initialised", name="memory_store")
        except Exception as exc:
            log.warning("memory_store_init_failed", error=str(exc))

        # 3. Conversation store init
        try:
            from rune.conversation.store import ConversationStore
            db_path = _default_rune_dir() / "conversations.db"
            self._conversation_store = ConversationStore(db_path)
            log.info("subsystem_initialised", name="conversation_store")
        except Exception as exc:
            log.warning("conversation_store_init_failed", error=str(exc))

        # 3a. Trash cleanup (enforce retention TTL on startup)
        try:
            from rune.tools.file import cleanup_trash
            removed = cleanup_trash()
            if removed:
                log.info("subsystem_initialised", name="trash_cleanup", removed=removed)
        except Exception as exc:
            log.debug("trash_cleanup_failed", error=str(exc))

        # 4. Orphaned run cleanup
        await self._cleanup_orphaned_runs()

        # 5. Reflexion learner init
        try:
            from rune.proactive.reflexion import get_reflexion_learner
            self._reflexion_learner = get_reflexion_learner()
            log.info("subsystem_initialised", name="reflexion_learner")
        except Exception as exc:
            log.warning("reflexion_learner_init_failed", error=str(exc))

        # 6. Pattern learner init
        try:
            from rune.proactive.patterns import PatternLearner
            self._pattern_learner = PatternLearner.load_from_db()
            log.info("subsystem_initialised", name="pattern_learner")
        except Exception as exc:
            log.warning("pattern_learner_init_failed", error=str(exc))

        # 7. Proactive engine start
        if self._config.get("proactive_enabled", True):
            try:
                from rune.proactive.engine import get_proactive_engine
                self._proactive_engine = get_proactive_engine()
                if hasattr(self._proactive_engine, "initialize"):
                    await self._proactive_engine.initialize()
                elif hasattr(self._proactive_engine, "start"):
                    await self._proactive_engine.start()
                log.info("subsystem_initialised", name="proactive_engine")
            except Exception as exc:
                log.warning("proactive_engine_init_failed", error=str(exc))

        # 7a. Heartbeat scheduler (moved early so proactive subsystems can
        #     register cron tasks on it - matches TS where heartbeat.start()
        #     happens inside the proactive-enabled block before subsystem wiring)
        await self._start_heartbeat()

        # 7b. Proactive subsystems (EngagementTracker, ConversationInitiator,
        #     EnvironmentSensor, Heartbeat-to-Engine wiring)
        if self._proactive_engine is not None:
            await self._init_proactive_subsystems()

        # 8. MCP bridge init
        await self._initialize_mcp_bridge()

        # 9. Autonomous executor init
        try:
            from rune.agent.autonomous import get_autonomous_executor
            self._autonomous_executor = get_autonomous_executor()
            log.info("subsystem_initialised", name="autonomous_executor")

            # Start periodic executor state flush
            self._executor_flush_task = asyncio.create_task(
                self._executor_flush_loop()
            )
        except Exception as exc:
            log.warning("autonomous_executor_init_failed", error=str(exc))

        # 9b. Browser relay + page pool
        await self._init_browser_subsystems()

        # 10. Channel adapters start
        await self._start_channel_adapters()

        # 11. API server start (non-blocking)
        await self._start_api_server()

    async def _shutdown_subsystems(self) -> None:
        """Gracefully shut down subsystems in reverse order."""

        # 11. Stop API server
        if self._api_server_task is not None:
            self._api_server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._api_server_task
            self._api_server_task = None
            log.info("subsystem_shutdown", name="api_server")

        # 10. Stop channel adapters
        if self._channel_registry is not None:
            try:
                await self._channel_registry.stop_all()
                log.info("subsystem_shutdown", name="channel_adapters")
            except Exception as exc:
                log.warning("channel_adapters_shutdown_failed", error=str(exc))

        # 9. Stop autonomous executor flush
        if self._executor_flush_task is not None:
            self._executor_flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._executor_flush_task
            self._executor_flush_task = None
            # Persist final state
            if self._autonomous_executor is not None:
                with contextlib.suppress(Exception):
                    await self._persist_executor_state()
            log.info("subsystem_shutdown", name="autonomous_executor")

        # 9b. Shutdown browser subsystems
        await self._shutdown_browser_subsystems()

        # 8. Shutdown MCP bridge
        await self._shutdown_mcp_bridge()

        # 7b. Stop proactive subsystems (reverse of init order)
        # EnvironmentSensor
        if self._environment_sensor is not None:
            try:
                await self._environment_sensor.stop()
                log.info("subsystem_shutdown", name="environment_sensor")
            except Exception as exc:
                log.warning("environment_sensor_shutdown_failed", error=str(exc))
            self._environment_sensor = None

        # ConversationInitiator (remove cron task)
        if self._conversation_initiator is not None:
            try:
                if self._heartbeat_scheduler is not None:
                    self._heartbeat_scheduler.remove_task("conversation_initiator")
                log.info("subsystem_shutdown", name="conversation_initiator")
            except Exception as exc:
                log.warning("conversation_initiator_shutdown_failed", error=str(exc))
            self._conversation_initiator = None

        # EngagementTracker - persist suggestions before shutting down
        if self._engagement_tracker is not None:
            try:
                if self._proactive_engine is not None and self._memory_store is not None:
                    saved = self._proactive_engine.save_suggestions(self._memory_store)
                    if saved > 0:
                        log.info("suggestions_persisted_on_shutdown", count=saved)
                log.info("subsystem_shutdown", name="engagement_tracker")
            except Exception as exc:
                log.warning("engagement_tracker_shutdown_failed", error=str(exc))
            self._engagement_tracker = None

        # Remove heartbeat-to-engine cron task
        if self._heartbeat_scheduler is not None:
            with contextlib.suppress(Exception):
                self._heartbeat_scheduler.remove_task("proactive_engine_tick")

        # 7a. Stop heartbeat scheduler (after cron tasks removed)
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._heartbeat_task
            self._heartbeat_task = None
        if self._heartbeat_scheduler is not None:
            with contextlib.suppress(Exception):
                await self._heartbeat_scheduler.stop()
            log.info("subsystem_shutdown", name="heartbeat")

        # 7. Stop proactive engine
        if self._proactive_engine is not None:
            try:
                if hasattr(self._proactive_engine, "shutdown"):
                    await self._proactive_engine.shutdown()
                elif hasattr(self._proactive_engine, "stop"):
                    await self._proactive_engine.stop()
                log.info("subsystem_shutdown", name="proactive_engine")
            except Exception as exc:
                log.warning("proactive_engine_shutdown_failed", error=str(exc))

        # 6. Pattern learner shutdown
        if self._pattern_learner is not None and hasattr(self._pattern_learner, "shutdown"):
            try:
                await self._pattern_learner.shutdown()
                log.info("subsystem_shutdown", name="pattern_learner")
            except Exception as exc:
                log.warning("pattern_learner_shutdown_failed", error=str(exc))

        # 5. Reflexion learner - stateless singleton, no shutdown needed

        # 3. Close memory store (sync method)
        if self._memory_store is not None:
            try:
                if hasattr(self._memory_store, "close"):
                    self._memory_store.close()
                log.info("subsystem_shutdown", name="memory_store")
            except Exception as exc:
                log.warning("memory_store_shutdown_failed", error=str(exc))

    # -- browser subsystem init/shutdown ------------------------------------

    async def _init_browser_subsystems(self) -> None:
        """Start the CDP relay server and browser page pool."""
        # Relay server (for Chrome Extension integration)
        try:
            from rune.browser.relay_server import RelayServer
            self._relay_server = RelayServer()
            await self._relay_server.start()
            log.info("subsystem_initialised", name="relay_server", port=self._relay_server.port)
        except Exception as exc:
            log.warning("relay_server_init_failed", error=str(exc))

        # Browser page pool (for browser-based search fallback)
        try:
            from rune.capabilities.search.browser_page_pool import BrowserPagePool
            from rune.capabilities.web import build_search_provider, set_search_provider
            from rune.config.loader import get_config

            search_cfg = get_config().search
            self._browser_page_pool = BrowserPagePool(max_pages=search_cfg.max_concurrent)

            # Rebuild search provider with the page pool for browser fallback.
            provider = build_search_provider(
                provider_config=search_cfg.provider,
                page_pool=self._browser_page_pool,
            )
            set_search_provider(provider)
            log.info("subsystem_initialised", name="browser_page_pool")
        except Exception as exc:
            log.warning("browser_page_pool_init_failed", error=str(exc))

    async def _shutdown_browser_subsystems(self) -> None:
        """Stop relay server and destroy page pool."""
        if self._browser_page_pool is not None:
            try:
                await self._browser_page_pool.destroy()
                log.info("subsystem_shutdown", name="browser_page_pool")
            except Exception as exc:
                log.warning("browser_page_pool_shutdown_failed", error=str(exc))
            self._browser_page_pool = None

        if self._relay_server is not None:
            try:
                await self._relay_server.stop()
                log.info("subsystem_shutdown", name="relay_server")
            except Exception as exc:
                log.warning("relay_server_shutdown_failed", error=str(exc))
            self._relay_server = None

    # -- orphaned run cleanup -----------------------------------------------

    async def _cleanup_orphaned_runs(self) -> None:
        """Mark runs with status 'running' or 'queued' from a previous daemon
        instance as 'interrupted'.

        Matches TS: ``store.listRuns({status})`` then ``updateRunStatus('aborted')``.
        """
        if self._memory_store is None:
            return

        for status in ("running", "queued"):
            try:
                if hasattr(self._memory_store, "list_runs"):
                    orphans = self._memory_store.list_runs(status=status)
                else:
                    log.debug(
                        "orphan_cleanup_skipped",
                        reason="memory_store has no list_runs method",
                    )
                    return

                for orphan in orphans:
                    run_id = (
                        orphan.get("run_id")
                        or orphan.get("runId")
                    )
                    if run_id is None:
                        continue

                    if hasattr(self._memory_store, "update_run_status"):
                        self._memory_store.update_run_status(
                            run_id,
                            "interrupted",
                            error="Daemon restarted — run orphaned",
                        )

                if orphans:
                    log.info(
                        "orphaned_runs_cleaned",
                        status=status,
                        count=len(orphans),
                    )
            except Exception as exc:
                log.warning(
                    "orphan_cleanup_failed",
                    status=status,
                    error=str(exc),
                )

    # -- proactive subsystems -----------------------------------------------

    async def _init_proactive_subsystems(self) -> None:
        """Wire EngagementTracker, ConversationInitiator, EnvironmentSensor,
        and the Heartbeat-to-Engine bridge.

        Matches TS ``initializeProactive()`` (lines 318-392 of daemon.ts).
        Each subsystem is optional - failures are logged as warnings but
        do not prevent other subsystems from starting.
        """
        engine = self._proactive_engine

        # --- EngagementTracker ---
        try:
            from rune.proactive.engagement_tracker import EngagementTracker

            self._engagement_tracker = EngagementTracker()

            # Wire MemoryStore to EngagementStore adapter (mirrors TS engagementStoreAdapter)
            if self._memory_store is not None:
                store = self._memory_store
                engagement_store_adapter = _EngagementStoreAdapter(store)
                self._engagement_tracker.initialize(engagement_store_adapter)

                # Connect persist store to engine and restore persisted suggestions
                try:
                    loaded = engine.load_persisted_suggestions(store)
                    if loaded > 0:
                        log.info("persisted_suggestions_restored", count=loaded)
                except Exception as exc:
                    log.warning("persisted_suggestions_restore_failed", error=str(exc))

            log.info("subsystem_initialised", name="engagement_tracker")
        except Exception as exc:
            log.warning("engagement_tracker_init_failed", error=str(exc))

        # --- ConversationInitiator ---
        try:
            from rune.proactive.conversation_initiator import ConversationInitiator

            self._conversation_initiator = ConversationInitiator()

            # Wire heartbeat-driven conversation checks: register a cron task
            # that evaluates whether to initiate a conversation every 5 minutes.
            if self._heartbeat_scheduler is not None:
                async def _conversation_check() -> None:
                    """Periodic check for conversation opportunities."""
                    try:
                        from rune.proactive.context import ContextGatherer
                        gatherer = ContextGatherer()
                        ctx = await gatherer.gather()
                        metrics = (
                            self._engagement_tracker.get_metrics()
                            if self._engagement_tracker is not None
                            else None
                        )
                        if metrics is None:
                            from rune.proactive.types import EngagementMetrics
                            metrics = EngagementMetrics()

                        if self._conversation_initiator.should_initiate(ctx, metrics):
                            suggestion = self._conversation_initiator.create_initiative(
                                "followup", ctx,
                            )
                            engine.add_suggestion(suggestion)
                            log.info(
                                "conversation_opportunity_delivered",
                                title=suggestion.title,
                            )
                    except Exception as exc:
                        log.debug("conversation_check_failed", error=str(exc))

                self._heartbeat_scheduler.add_task(
                    "conversation_initiator",
                    "*/5 * * * *",  # every 5 minutes
                    _conversation_check,
                )

            log.info("subsystem_initialised", name="conversation_initiator")
        except Exception as exc:
            log.warning("conversation_initiator_init_failed", error=str(exc))

        # --- EnvironmentSensor ---
        try:
            from rune.proactive.sensor import EnvironmentSensor

            workspace = os.environ.get("RUNE_WORKSPACE", os.getcwd())
            self._environment_sensor = EnvironmentSensor(workspace_root=workspace)
            await self._environment_sensor.start()

            # Forward sensor events to the proactive engine as suggestions
            async def _on_sensor_event(event: Any) -> None:
                try:
                    from rune.proactive.types import Suggestion

                    suggestion = Suggestion(
                        type="insight",
                        title=f"Environment: {event.type}",
                        description=str(event.data),
                        confidence=0.3,
                        source="environment_sensor",
                    )
                    engine.add_suggestion(suggestion)
                except Exception as exc:
                    log.debug("sensor_event_forward_failed", error=str(exc))

            self._environment_sensor.on("sensor_event", _on_sensor_event)
            log.info("subsystem_initialised", name="environment_sensor", workspace=workspace)
        except Exception as exc:
            log.warning("environment_sensor_init_failed", error=str(exc))

        # --- ProactiveAgentBridge (autonomous execution + learning feedback) ---
        # Matches TS daemon.ts lines 299-306: initializeProactiveAgentBridge({...})
        try:
            from rune.agent.loop import NativeAgentLoop
            from rune.proactive.bridge import (
                BridgeConfig,
                initialize_proactive_bridge,
            )

            bridge_config = BridgeConfig(
                poll_interval_seconds=60.0,
                max_retries=2,
                max_executions_per_hour=5,
                min_confidence=0.5,
                max_steps=50,
                timeout_ms=180_000,
            )

            async def _proactive_agent_factory(goal: str) -> dict[str, Any]:
                """Agent factory for proactive bridge that creates a NativeAgentLoop
                per suggestion and runs it to completion."""
                from rune.types import AgentConfig
                cfg = AgentConfig(
                    max_iterations=bridge_config.max_steps,
                    timeout_seconds=bridge_config.timeout_ms / 1000,
                )
                loop = NativeAgentLoop(config=cfg)
                result = await asyncio.wait_for(
                    loop.run(goal),
                    timeout=bridge_config.timeout_ms / 1000,
                )
                return {
                    "success": getattr(result, "success", True),
                    "output": getattr(result, "answer", str(result)),
                    "error": getattr(result, "error", None),
                }

            bridge = initialize_proactive_bridge(
                engine=engine,
                agent_factory=_proactive_agent_factory,
                config=bridge_config,
                context={"policy_profile": "rune"},
                autonomous_executor=self._autonomous_executor,
            )
            bridge.start()
            log.info("subsystem_initialised", name="proactive_agent_bridge")
        except Exception as exc:
            log.warning("proactive_agent_bridge_init_failed", error=str(exc))

        # --- Heartbeat to Engine wiring ---
        # Register a cron task on the heartbeat scheduler that triggers
        # the proactive engine's evaluation pipeline each tick.
        if self._heartbeat_scheduler is not None:
            try:
                async def _heartbeat_engine_tick() -> None:
                    """Heartbeat tick: trigger proactive evaluation."""
                    try:
                        from dataclasses import asdict

                        from rune.proactive.context import ContextGatherer
                        gatherer = ContextGatherer()
                        awareness = await gatherer.gather()
                        ctx = asdict(awareness)

                        # Enrich context with prediction engine data for
                        # frustration detection and need inference.
                        try:
                            from rune.proactive.prediction.engine import get_prediction_engine
                            pred = get_prediction_engine()
                            # Use _recent_actions (recorded by agent loop
                            # _on_tool_end with actual success/failure data).
                            recent_actions = getattr(pred, '_recent_actions', [])
                            if recent_actions:
                                ctx["recent_actions"] = recent_actions[-20:]
                                # Count errors for frustration detection
                                ctx["error_count"] = sum(
                                    1 for a in recent_actions[-20:]
                                    if not a.get("success", True)
                                )
                                # Count repeated consecutive commands
                                repeated = 0
                                history = pred.behavior_predictor._history
                                if len(history) >= 2:
                                    last = history[-1]
                                    for t in reversed(history[:-1]):
                                        if t == last:
                                            repeated += 1
                                        else:
                                            break
                                ctx["repeated_commands"] = repeated
                        except Exception:
                            pass

                        suggestions = await engine.evaluate(ctx)
                        if suggestions:
                            log.info(
                                "heartbeat_proactive_suggestions",
                                count=len(suggestions),
                            )
                    except Exception as exc:
                        log.debug("heartbeat_engine_tick_failed", error=str(exc))

                self._heartbeat_scheduler.add_task(
                    "proactive_engine_tick",
                    "* * * * *",  # every minute
                    _heartbeat_engine_tick,
                )

                # --- Cron goal execution ---
                # Check cron jobs every minute and execute any that match
                # the current time.  Goal-based jobs run asynchronously
                # so they don't block the heartbeat.
                async def _cron_goal_tick() -> None:
                    try:
                        from datetime import datetime

                        from rune.capabilities.cron import execute_cron_job, get_active_cron_jobs
                        from rune.daemon.heartbeat import HeartbeatScheduler

                        now = datetime.now()
                        for job in get_active_cron_jobs():
                            if HeartbeatScheduler._matches_cron(job.schedule, now):
                                # Run in background to avoid blocking heartbeat
                                asyncio.create_task(execute_cron_job(job))
                    except Exception as exc:
                        log.debug("cron_goal_tick_failed", error=str(exc))

                self._heartbeat_scheduler.add_task(
                    "cron_goal_tick",
                    "* * * * *",
                    _cron_goal_tick,
                )

                log.info("heartbeat_engine_wiring_complete")
            except Exception as exc:
                log.warning("heartbeat_engine_wiring_failed", error=str(exc))

    # -- MCP bridge ---------------------------------------------------------

    async def _initialize_mcp_bridge(self) -> None:
        """Connect configured MCP servers and register their tools."""
        try:
            from rune.mcp.bridge import initialize_mcp_bridge
            from rune.mcp.config import MCPServerConfig

            # Load MCP server configurations
            mcp_servers = self._get_mcp_server_configs()
            if not mcp_servers:
                log.debug("mcp_bridge_skipped", reason="no servers configured")
                return

            # Convert raw dicts to MCPServerConfig if needed
            configs: dict[str, MCPServerConfig] = {}
            for server_cfg in mcp_servers:
                name = server_cfg.get("name", "unknown")
                try:
                    configs[name] = MCPServerConfig(**server_cfg)
                except Exception:
                    configs[name] = MCPServerConfig(
                        name=name,
                        command=server_cfg.get("command", server_cfg.get("url", "")),
                        transport=server_cfg.get("transport", "stdio"),
                    )

            result = await initialize_mcp_bridge(configs=configs)
            self._mcp_bridge = True

            if result.connected_servers > 0:
                log.info(
                    "mcp_bridge_initialised",
                    servers=result.connected_servers,
                    tools=result.registered_count,
                )
            if result.failed_servers:
                for name in result.failed_servers:
                    log.warning("mcp_server_connect_failed", name=name)
        except Exception as exc:
            log.warning("mcp_bridge_init_failed", error=str(exc))

    def _get_mcp_server_configs(self) -> list[dict[str, Any]]:
        """Load MCP server configurations from RUNE config or environment."""
        servers: list[dict[str, Any]] = []

        # Try loading from config
        try:
            from rune.config import get_config
            config = get_config()
            mcp_cfg = getattr(config, "mcp", None)
            if mcp_cfg is not None:
                raw_servers = getattr(mcp_cfg, "servers", None)
                if isinstance(raw_servers, list):
                    servers.extend(raw_servers)
                elif isinstance(raw_servers, dict):
                    for name, cfg in raw_servers.items():
                        if isinstance(cfg, dict):
                            cfg["name"] = name
                            servers.append(cfg)
        except Exception:
            pass

        # Also check environment variables
        mcp_servers_env = os.environ.get("RUNE_MCP_SERVERS", "")
        if mcp_servers_env:
            try:
                parsed = json_decode(mcp_servers_env)
                if isinstance(parsed, list):
                    servers.extend(parsed)
            except ValueError:
                # Treat as a single server command
                servers.append({"name": "env", "url": mcp_servers_env})

        return servers

    async def _shutdown_mcp_bridge(self) -> None:
        """Disconnect all MCP clients."""
        for client in self._mcp_clients:
            try:
                await client.disconnect()
            except Exception as exc:
                log.warning("mcp_client_disconnect_failed", error=str(exc))
        self._mcp_clients.clear()
        if self._mcp_bridge:
            log.info("subsystem_shutdown", name="mcp_bridge")
        self._mcp_bridge = None

    # -- autonomous executor ------------------------------------------------

    async def _persist_executor_state(self) -> None:
        """Persist autonomous executor state to disk."""
        if self._autonomous_executor is None:
            return
        try:
            data = self._autonomous_executor.serialize()
            state_path = _default_rune_dir() / "autonomous_executor_state.json"
            state_path.write_text(json.dumps(data, default=str))
        except Exception as exc:
            log.warning("executor_state_persist_failed", error=str(exc))

    async def _executor_flush_loop(self) -> None:
        """Periodically persist autonomous executor state."""
        try:
            while True:
                await asyncio.sleep(_EXECUTOR_FLUSH_INTERVAL_S)
                await self._persist_executor_state()
        except asyncio.CancelledError:
            pass

    # -- channel adapters ---------------------------------------------------

    async def _start_channel_adapters(self) -> None:
        """Discover and start channel adapters from environment."""
        try:
            from rune.channels.registry import (
                auto_discover_channels,
                get_channel_registry,
            )

            discovered = auto_discover_channels()
            self._channel_registry = get_channel_registry()

            if discovered:
                await self._channel_registry.start_all()
                log.info(
                    "subsystem_initialised",
                    name="channel_adapters",
                    channels=discovered,
                )
            else:
                log.debug("channel_adapters_none_discovered")

            # Register gateway singleton so cron/proactive can route notifications
            try:
                from rune.daemon.gateway import ChannelGateway, set_gateway
                gw = ChannelGateway(self._channel_registry)
                set_gateway(gw)
                log.debug("gateway_singleton_registered")
            except Exception:
                pass
        except Exception as exc:
            log.warning("channel_adapters_init_failed", error=str(exc))

    # -- heartbeat ----------------------------------------------------------

    async def _start_heartbeat(self) -> None:
        """Start the heartbeat scheduler and a simple file-based heartbeat."""
        # Start the cron-style heartbeat scheduler
        try:
            from rune.daemon.heartbeat import HeartbeatScheduler
            self._heartbeat_scheduler = HeartbeatScheduler(
                interval_seconds=_HEARTBEAT_INTERVAL_S,
            )
            await self._heartbeat_scheduler.start()
            log.info("subsystem_initialised", name="heartbeat_scheduler")
        except Exception as exc:
            log.warning("heartbeat_scheduler_init_failed", error=str(exc))

        # Start a simple file-based heartbeat indicator
        self._heartbeat_task = asyncio.create_task(self._heartbeat_file_loop())

    async def _heartbeat_file_loop(self) -> None:
        """Periodically update a heartbeat file to indicate daemon is alive."""
        heartbeat_path = Path(self._config.get(
            "heartbeat_file",
            str(_default_rune_dir() / "heartbeat"),
        ))
        try:
            while True:
                try:
                    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
                    heartbeat_path.write_text(
                        json_encode({
                            "pid": os.getpid(),
                            "timestamp": time.time(),
                            "version": VERSION,
                            "running": self._running,
                        })
                    )
                except OSError:
                    pass
                await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
        except asyncio.CancelledError:
            # Clean up heartbeat file on stop
            with contextlib.suppress(OSError):
                heartbeat_path.unlink(missing_ok=True)

    # -- API server ---------------------------------------------------------

    async def _start_api_server(self) -> None:
        """Start the FastAPI-based API server in the background."""
        if not self._config.get("api_enabled", False):
            # Also check env
            env_enabled = os.environ.get("RUNE_API_ENABLED", "").lower() in (
                "1", "true", "yes",
            )
            if not env_enabled:
                log.debug("api_server_disabled")
                return

        try:
            from rune.api.server import create_app

            app = create_app()
            host = self._config.get("api_host", "127.0.0.1")
            port = self._config.get("api_port", 18789)

            self._api_server_task = asyncio.create_task(
                self._run_api_server(app, host, port)
            )
            log.info(
                "subsystem_initialised",
                name="api_server",
                host=host,
                port=port,
            )
        except ImportError as exc:
            log.warning(
                "api_server_unavailable",
                error=str(exc),
                hint="Install fastapi and uvicorn",
            )
        except Exception as exc:
            log.warning("api_server_init_failed", error=str(exc))

    @staticmethod
    async def _run_api_server(app: Any, host: str, port: int) -> None:
        """Run uvicorn in-process (non-blocking)."""
        try:
            import uvicorn

            config = uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level="warning",
            )
            server = uvicorn.Server(config)
            server.install_signal_handlers = lambda: None  # Caller manages signals
            await server.serve()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("api_server_crashed", error=str(exc))

    # -- client handling ----------------------------------------------------

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single client connection over the Unix socket."""
        try:
            data = await reader.readline()
            if not data:
                return
            request = json_decode(data.decode())
            response = await self._dispatch(request)
            writer.write(json_encode(response).encode() + b"\n")
            await writer.drain()
        except Exception as exc:
            log.warning("client_handler_error", error=str(exc))
            try:
                err_resp = {"success": False, "error": str(exc)}
                writer.write(json_encode(err_resp).encode() + b"\n")
                await writer.drain()
            except Exception:
                pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a daemon command to the appropriate handler."""
        command = request.get("command", "")
        params = request.get("params", {})

        if command == "status":
            return self._status()
        elif command == "execute":
            return await self._execute_agent(params)
        elif command == "stop":
            asyncio.create_task(self.stop())
            return {"success": True, "message": "Daemon stopping"}
        elif command == "health":
            return {"success": True, "version": VERSION, "pid": os.getpid()}
        else:
            return {"success": False, "error": f"Unknown command: {command}"}

    def _status(self) -> dict[str, Any]:
        """Return daemon status."""
        uptime = (
            time.monotonic() - self._start_time
            if self._start_time is not None
            else 0.0
        )
        return {
            "success": True,
            "version": VERSION,
            "pid": os.getpid(),
            "running": self._running,
            "uptime_seconds": uptime,
            "proactive_enabled": self._config.get("proactive_enabled", False),
            "subsystems": {
                "memory_store": self._memory_store is not None,
                "conversation_store": self._conversation_store is not None,
                "reflexion_learner": self._reflexion_learner is not None,
                "pattern_learner": self._pattern_learner is not None,
                "proactive_engine": self._proactive_engine is not None,
                "engagement_tracker": self._engagement_tracker is not None,
                "conversation_initiator": self._conversation_initiator is not None,
                "environment_sensor": self._environment_sensor is not None,
                "mcp_bridge": self._mcp_bridge is not None,
                "autonomous_executor": self._autonomous_executor is not None,
                "channel_adapters": self._channel_registry is not None,
                "heartbeat": self._heartbeat_scheduler is not None,
                "api_server": self._api_server_task is not None,
            },
        }

    async def _execute_agent(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute an agent loop for a background task."""
        goal = params.get("goal", "")
        if not goal:
            return {"success": False, "error": "Missing goal"}

        try:
            from rune.agent.llm_adapter import create_agent_model_for_goal
            from rune.agent.loop import NativeAgentLoop

            await create_agent_model_for_goal(goal)
            loop = NativeAgentLoop()
            result = await loop.run(goal)

            return {
                "success": result.success if hasattr(result, "success") else True,
                "answer": getattr(result, "answer", str(result)),
                "iterations": getattr(result, "iterations", 0),
            }
        except Exception as exc:
            log.error("agent_execute_failed", error=str(exc))
            return {"success": False, "error": str(exc)}


# Entry point

def main() -> None:
    """CLI entry point for running the daemon."""
    daemon = RuneDaemon()
    try:
        asyncio.run(daemon.serve_forever())
    except KeyboardInterrupt:
        log.info("daemon_interrupted")


if __name__ == "__main__":
    main()
