"""Host background agent, memory, proactive and channel services over a Unix socket."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

from rune.daemon.lifecycle import SHUTDOWN_TIMEOUT
from rune.utils.fast_serde import json_decode, json_encode
from rune.utils.logger import get_logger

log = get_logger(__name__)

VERSION = "0.1.0"

_HEARTBEAT_INTERVAL_S = 30.0
_EXECUTOR_FLUSH_INTERVAL_S = 300.0  # 5 minutes


# Default configuration

def _default_rune_dir() -> Path:
    from rune.utils.paths import rune_data
    return rune_data()


_proactive_cache: tuple[float, bool] | None = None


def _configured_proactive_enabled() -> bool:
    """Read the proactive toggle without reloading session-only model settings."""
    global _proactive_cache

    try:
        from rune.config.loader import _find_config_file

        path = _find_config_file()
        if path is None:
            return True

        mtime = path.stat().st_mtime
        if _proactive_cache is not None and _proactive_cache[0] == mtime:
            return _proactive_cache[1]

        from ruamel.yaml import YAML

        raw = YAML(typ="safe").load(path) or {}
        section = raw.get("proactive")
        enabled = True if not isinstance(section, dict) else bool(
            section.get("enabled", True)
        )
        _proactive_cache = (mtime, enabled)
        return enabled
    except Exception as exc:  # unreadable — keep the prior default
        log.debug("proactive_config_read_failed", error=str(exc))
        return True


def _default_config() -> dict[str, Any]:
    rune_dir = _default_rune_dir()
    return {
        "socket_path": str(rune_dir / "rune.sock"),
        "pid_file": str(rune_dir / "rune.pid"),
        "log_file": str(rune_dir / "daemon.log"),
        "heartbeat_file": str(rune_dir / "heartbeat"),
        "proactive_enabled": _configured_proactive_enabled(),
        "check_interval_ms": 60_000,  # 1 minute
        "api_host": os.environ.get("RUNE_API_HOST", "127.0.0.1"),
        "api_port": int(os.environ.get("RUNE_API_PORT", "18789")),
        "api_enabled": os.environ.get("RUNE_API_ENABLED", "").lower()
        in ("1", "true", "yes"),
    }


# EngagementStore adapter (bridges MemoryStore to EngagementTracker)

class _EngagementStoreAdapter:
    """Adapt MemoryStore methods to the EngagementStore protocol."""

    __slots__ = ("_store",)

    def __init__(self, store: Any) -> None:
        self._store = store

    # EngagementStore protocol methods

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

def _config_to_dict(cfg: Any) -> dict[str, Any]:
    """Best-effort dict view of an MCPServerConfig-like object."""
    import dataclasses

    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.asdict(cfg)
    if hasattr(cfg, "model_dump"):
        return dict(cfg.model_dump())
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    return {}


class RuneDaemon:
    """Manage background services and handle commands over a Unix socket."""

    def __init__(self, config: dict[str, Any] | None = None, install_signal_handlers: bool = True) -> None:
        self._config = config or _default_config()
        self._install_signal_handlers = install_signal_handlers
        self._server: asyncio.AbstractServer | None = None
        self._running = False
        self._start_time: float | None = None
        self._stop_requested = asyncio.Event()
        self._stop_task: asyncio.Task[None] | None = None
        self._api_server: Any = None
        self._cron_dispatcher: Any = None
        self._maintenance: Any = None
        self._registered_signals: list[signal.Signals] = []
        from rune.daemon.connections import Connections
        self._connections = Connections()
        self._startup_incomplete = False
        self._process_deadline = None

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
        # Skill evaluations can outlast a heartbeat tick; allow only one at a time.
        self._skill_eval_running: bool = False
        self._skill_eval_task: asyncio.Task | None = None
        self._proactive_bridge: Any = None
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

    # public API

    async def start(self) -> None:
        """Initialise subsystems and start the Unix socket server."""
        log.info("daemon_starting", version=VERSION)

        self._load_env()

        rune_dir = _default_rune_dir()
        rune_dir.mkdir(parents=True, exist_ok=True)

        await self._acquire_process_lock()

        pid_path = Path(self._config["pid_file"])
        pid_path.write_text(str(os.getpid()))

        await self._init_subsystems()

        socket_path = self._config["socket_path"]
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=socket_path,
        )
        Path(socket_path).chmod(0o600)
        self._running = True
        self._start_time = time.monotonic()
        log.info("daemon_started", socket=socket_path, pid=os.getpid())

    async def stop(self, sig: signal.Signals | None = None) -> None:
        """Share one cleanup task across concurrent shutdown requests."""
        self._stop_requested.set()
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(self._stop())
        await asyncio.shield(self._stop_task)

    async def _stop(self) -> None:
        if self._process_deadline is not None:
            self._process_deadline.arm(SHUTDOWN_TIMEOUT)
        self._running = False
        if self._lock_path is None:
            return
        log.info("daemon_stopping")
        cleaned = False
        try:
            async with asyncio.timeout(25):
                if self._server:
                    self._server.close()
                    await self._server.wait_closed()
                await self._shutdown_subsystems()
                if self._startup_incomplete:
                    raise RuntimeError("Startup is still running; process ownership retained")
                cleaned = True
        finally:
            if cleaned:
                try:
                    for key in ("socket_path", "pid_file", "heartbeat_file"):
                        if path := self._config.get(key):
                            Path(path).unlink(missing_ok=True)
                finally:
                    await self._release_process_lock()
                log.info("daemon_stopped")
            else:
                # The OS releases ownership when this process exits.
                log.error("daemon_cleanup_incomplete", lock_retained=True)

    async def serve_forever(self) -> None:
        """Keep the supervised process alive until shutdown or a critical failure."""
        loop = asyncio.get_running_loop()
        if self._install_signal_handlers:
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                loop.add_signal_handler(sig, self._stop_requested.set)
                self._registered_signals.append(sig)
        try:
            from rune.daemon.lifecycle import IncompleteShutdown, start_until_stopped
            try:
                if not await start_until_stopped(self.start, self._stop_requested):
                    return
            except IncompleteShutdown:
                self._startup_incomplete = True
                raise
            if self._process_deadline is not None:
                self._process_deadline.disarm()
            api_deadline = time.monotonic() + 30
            while not self._stop_requested.is_set():
                if (self._api_server_task is not None and time.monotonic() >= api_deadline
                        and not (self._api_server and self._api_server.started)):
                    raise RuntimeError("API server did not become ready within 30 seconds")
                tasks = {"api": self._api_server_task, "heartbeat": self._heartbeat_task}
                if self._heartbeat_scheduler is not None:
                    tasks["scheduler"] = self._heartbeat_scheduler._loop_task
                for name, task in tasks.items():
                    if task is not None and task.done():
                        if not task.cancelled():
                            task.result()
                        raise RuntimeError(f"Critical service stopped: {name}")
                try:
                    await asyncio.wait_for(self._stop_requested.wait(), timeout=0.5)
                except TimeoutError:
                    pass
        finally:
            try:
                await self.stop()
            finally:
                for sig in self._registered_signals:
                    loop.remove_signal_handler(sig)
                self._registered_signals.clear()

    def health(self) -> dict[str, Any]:
        checks = {"daemon": self._running and not self._stop_requested.is_set()}
        if self._config.get("api_enabled"):
            checks["api"] = bool(self._api_server and self._api_server.started
                                 and self._api_server_task and not self._api_server_task.done())
        if self._heartbeat_scheduler is not None:
            checks["scheduler"] = self._heartbeat_scheduler.is_running
            if self._cron_dispatcher is not None:
                checks["schedules"] = self._heartbeat_scheduler.healthy("cron_goal_tick")
        if self._maintenance is not None:
            checks["storage"] = self._maintenance.healthy
        return {"status": "ok" if all(checks.values()) else "degraded", "checks": checks,
                "connections": dict(self._connections.status)}

    # process lock

    async def _acquire_process_lock(self) -> None:
        """Acquire the daemon's advisory file lock."""
        try:
            from rune.daemon.process_lock import acquire_lock
            self._lock_path = acquire_lock()
            log.info("process_lock_acquired", path=str(self._lock_path))
        except RuntimeError as exc:
            log.error("process_lock_failed", error=str(exc))
            raise
        except Exception:
            log.exception("process_lock_unavailable")
            raise

    async def _release_process_lock(self) -> None:
        """Release the daemon process lock."""
        try:
            from rune.daemon.process_lock import release_lock
            if self._lock_path is not None:
                release_lock(self._lock_path)
                self._lock_path = None
            log.info("process_lock_released")
        except Exception as exc:
            log.warning("process_lock_release_failed", error=str(exc))

    # subsystem init / shutdown

    def _load_env(self) -> None:
        """Load .rune/.env file if present."""
        try:
            from rune.utils.env import load_env
            load_env()
        except ImportError:
            log.debug("env_loader_not_available")

    async def _init_subsystems(self) -> None:
        """Initialize daemon services in dependency order."""

        try:
            from rune.memory.store import get_memory_store
            self._memory_store = get_memory_store()
            if hasattr(self._memory_store, "initialize"):
                await self._memory_store.initialize()
            log.info("subsystem_initialised", name="memory_store")
        except Exception as exc:
            log.warning("memory_store_init_failed", error=str(exc))
            raise

        try:
            from rune.conversation.store import ConversationStore
            from rune.utils.paths import conversations_db_path
            self._conversation_store = ConversationStore(conversations_db_path())
            log.info("subsystem_initialised", name="conversation_store")
        except Exception as exc:
            log.warning("conversation_store_init_failed", error=str(exc))
            raise

        await self._cleanup_orphaned_runs()
        from rune.daemon.maintenance import recover_spend
        await recover_spend()

        try:
            from rune.proactive.reflexion import get_reflexion_learner
            self._reflexion_learner = get_reflexion_learner()
            log.info("subsystem_initialised", name="reflexion_learner")
        except Exception as exc:
            log.warning("reflexion_learner_init_failed", error=str(exc))

        try:
            from rune.proactive.patterns import PatternLearner
            self._pattern_learner = PatternLearner.load_from_db()
            log.info("subsystem_initialised", name="pattern_learner")
        except Exception as exc:
            log.warning("pattern_learner_init_failed", error=str(exc))

        # Cron uses this engine even when proactive suggestions are disabled.
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

        # Start the scheduler before proactive services register jobs.
        await self._start_heartbeat()
        from rune.daemon.cron_dispatcher import CronDispatcher
        self._cron_dispatcher = CronDispatcher()
        self._heartbeat_scheduler.add_task("cron_goal_tick", "* * * * *", self._cron_dispatcher.tick)
        from rune.daemon.maintenance import Maintenance
        self._maintenance = Maintenance(self._memory_store)
        self._heartbeat_scheduler.add_task("maintenance", "* * * * *", self._maintenance.tick)

        if self._proactive_engine is not None:
            await self._init_proactive_subsystems()

        try:
            from rune.agent.autonomous import get_autonomous_executor
            self._autonomous_executor = get_autonomous_executor()
            log.info("subsystem_initialised", name="autonomous_executor")

            self._executor_flush_task = asyncio.create_task(
                self._executor_flush_loop()
            )
        except Exception as exc:
            log.warning("autonomous_executor_init_failed", error=str(exc))

        await self._init_browser_subsystems()

        await self._start_api_server()
        self._connections.start("mcp", self._initialize_mcp_bridge)
        self._connections.start("channels", self._start_channel_adapters)

    async def _shutdown_subsystems(self) -> None:
        from rune.daemon.lifecycle import Cleanup

        cleanup = Cleanup()
        if self._api_server is not None:
            self._api_server.should_exit = True
        await self._connections.stop(cleanup)
        if self._heartbeat_scheduler is not None:
            await cleanup.run("scheduler", self._heartbeat_scheduler.stop)
        if self._cron_dispatcher is not None:
            await cleanup.run("cron", self._cron_dispatcher.stop)
        await cleanup.cancel("skill_eval", self._skill_eval_task)
        if self._proactive_bridge is not None:
            await cleanup.run("proactive_bridge", self._proactive_bridge.close)

        if self._api_server_task is not None:
            await cleanup.run("api", lambda: self._api_server_task, timeout=12)
            self._api_server_task = None

        async def stop_gateway():
            from rune.daemon.gateway import get_gateway
            if gateway := get_gateway():
                await gateway.stop()

        await cleanup.run("gateway", stop_gateway)
        if self._channel_registry is not None:
            await cleanup.run("channels", self._channel_registry.stop_all)
        await cleanup.cancel("executor_flush", self._executor_flush_task)
        self._executor_flush_task = None
        if self._autonomous_executor is not None:
            await cleanup.run("executor_state", self._persist_executor_state)
        await cleanup.run("browser", self._shutdown_browser_subsystems)
        await cleanup.run("mcp", self._shutdown_mcp_bridge)
        if self._environment_sensor is not None:
            await cleanup.run("environment_sensor", self._environment_sensor.stop)
            self._environment_sensor = None
        self._conversation_initiator = None

        if self._engagement_tracker is not None:
            if self._proactive_engine is not None and self._memory_store is not None:
                await cleanup.run("suggestions", lambda: self._proactive_engine.save_suggestions(self._memory_store))
            self._engagement_tracker = None
        await cleanup.cancel("heartbeat", self._heartbeat_task)
        self._heartbeat_task = None
        if self._proactive_engine is not None:
            stop_engine = getattr(self._proactive_engine, "shutdown", None) or getattr(self._proactive_engine, "stop", None)
            if stop_engine is not None:
                await cleanup.run("proactive_engine", stop_engine)
        if self._pattern_learner is not None and hasattr(self._pattern_learner, "shutdown"):
            await cleanup.run("pattern_learner", self._pattern_learner.shutdown)

        def persist_vectors():
            from rune.memory import manager
            if manager._manager is not None:
                manager._manager.persist_vectors()

        await cleanup.run("vector_index", persist_vectors)
        if self._memory_store is not None and hasattr(self._memory_store, "close"):
            await cleanup.run("memory_store", self._memory_store.close)
        cleanup.check()

    # browser subsystem init/shutdown

    async def _init_browser_subsystems(self) -> None:
        """Start the browser search pool; agent runs own their browser sessions."""
        # Leave the unauthenticated legacy relay disabled; start only the search page pool.
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

    # orphaned run cleanup

    async def _cleanup_orphaned_runs(self) -> None:
        """Mark unfinished runs from the previous daemon instance as interrupted."""
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

    # proactive subsystems

    async def _init_proactive_subsystems(self) -> None:
        """Wire optional proactive services independently so one failure does not block the rest."""
        engine = self._proactive_engine

        # EngagementTracker
        try:
            from rune.proactive.engagement_tracker import EngagementTracker

            self._engagement_tracker = EngagementTracker()

            if self._memory_store is not None:
                store = self._memory_store
                engagement_store_adapter = _EngagementStoreAdapter(store)
                self._engagement_tracker.initialize(engagement_store_adapter)

                try:
                    loaded = engine.load_persisted_suggestions(store)
                    if loaded > 0:
                        log.info("persisted_suggestions_restored", count=loaded)
                except Exception as exc:
                    log.warning("persisted_suggestions_restore_failed", error=str(exc))

            log.info("subsystem_initialised", name="engagement_tracker")
        except Exception as exc:
            log.warning("engagement_tracker_init_failed", error=str(exc))

        # ConversationInitiator
        try:
            from rune.proactive.conversation_initiator import ConversationInitiator

            self._conversation_initiator = ConversationInitiator()

            # Schedule conversation-initiation checks every five minutes.
            if self._heartbeat_scheduler is not None:
                async def _conversation_check() -> None:
                    """Periodic check for conversation opportunities."""
                    if not _configured_proactive_enabled():
                        return
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
                    "conversation_initiator",  # gated inside _conversation_check
                    "*/5 * * * *",  # every 5 minutes
                    _conversation_check,
                )

            log.info("subsystem_initialised", name="conversation_initiator")
        except Exception as exc:
            log.warning("conversation_initiator_init_failed", error=str(exc))

        # EnvironmentSensor
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

        # Connect proactive execution and feedback.
        try:
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

            async def _proactive_agent_factory(
                goal: str, *, verification: list[str] | None = None
            ) -> dict[str, Any]:
                from rune.agent.background import BackgroundTask, run_background

                return await run_background(BackgroundTask(
                    goal=goal, source="proactive", verification=list(verification or []),
                    max_steps=bridge_config.max_steps,
                    timeout_seconds=bridge_config.timeout_ms / 1000,
                ))

            from rune.proactive.execution_store import ExecutionStore
            from rune.utils.paths import rune_data

            bridge = initialize_proactive_bridge(
                engine=engine,
                agent_factory=_proactive_agent_factory,
                config=bridge_config,
                context={"policy_profile": "rune"},
                autonomous_executor=self._autonomous_executor,
                execution_store=ExecutionStore(rune_data() / "proactive-executions.db"),
            )
            self._proactive_bridge = bridge
            bridge.start()
            log.info("subsystem_initialised", name="proactive_agent_bridge")
        except Exception as exc:
            log.warning("proactive_agent_bridge_init_failed", error=str(exc))

        # Evaluate proactive context on scheduler ticks.
        if self._heartbeat_scheduler is not None:
            try:
                async def _heartbeat_engine_tick() -> None:
                    """Heartbeat tick: trigger proactive evaluation."""
                    # Reload each tick to pick up toggle changes from other API processes.
                    if not _configured_proactive_enabled():
                        return
                    try:
                        from dataclasses import asdict

                        from rune.proactive.context import ContextGatherer
                        gatherer = ContextGatherer()
                        awareness = await gatherer.gather()
                        ctx = asdict(awareness)

                        # Add prediction signals for frustration and inferred needs.
                        try:
                            from rune.proactive.prediction.engine import get_prediction_engine
                            pred = get_prediction_engine()
                            # Use actions recorded with their actual execution outcomes.
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

                # Consolidate the backlog left by short-lived CLI runs.
                async def _consolidation_tick() -> None:
                    try:
                        from rune.memory.consolidation import consolidate_recent

                        done = await consolidate_recent()
                        if done:
                            log.info("memory_consolidated", episodes=done)
                    except Exception as exc:
                        log.debug("consolidation_tick_failed", error=str(exc))

                self._heartbeat_scheduler.add_task(
                    "memory_consolidation",
                    "*/10 * * * *",  # every 10 minutes
                    _consolidation_tick,
                )

                log.info("heartbeat_engine_wiring_complete")
            except Exception as exc:
                log.warning("heartbeat_engine_wiring_failed", error=str(exc))

    # MCP bridge

    async def _initialize_mcp_bridge(self) -> str:
        """Connect configured MCP servers and register their tools."""
        try:
            from rune.mcp.bridge import initialize_mcp_bridge
            from rune.mcp.config import MCPServerConfig

            # Load MCP server configurations
            mcp_servers = self._get_mcp_server_configs()
            if not mcp_servers:
                log.debug("mcp_bridge_skipped", reason="no servers configured")
                return "disabled"

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
            return "degraded" if result.failed_servers else "ready"
        except Exception as exc:
            log.warning("mcp_bridge_init_failed", error=str(exc))
            return "failed"

    def _get_mcp_server_configs(self) -> list[dict[str, Any]]:
        """Load MCP server settings, starting with the shared ~/.rune/mcp_servers.json file."""
        servers: list[dict[str, Any]] = []

        try:
            from rune.mcp.config import load_mcp_config

            for name, cfg in (load_mcp_config() or {}).items():
                entry = cfg if isinstance(cfg, dict) else _config_to_dict(cfg)
                entry.setdefault("name", name)
                servers.append(entry)
        except Exception as exc:
            log.debug("mcp_config_file_read_failed", error=str(exc))

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
        from rune.mcp.bridge import shutdown_mcp_bridge
        await shutdown_mcp_bridge()
        for client in self._mcp_clients:
            try:
                await client.disconnect()
            except Exception as exc:
                log.warning("mcp_client_disconnect_failed", error=str(exc))
        self._mcp_clients.clear()
        if self._mcp_bridge:
            log.info("subsystem_shutdown", name="mcp_bridge")
        self._mcp_bridge = None

    # autonomous executor

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

    # channel adapters

    async def _start_channel_adapters(self) -> str:
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

            # Connect adapter message callbacks to the gateway after opening channels.
            try:
                from rune.daemon.gateway import ChannelGateway, set_gateway
                gw = ChannelGateway(self._channel_registry)
                set_gateway(gw)
                if discovered:
                    await gw.start()
                log.debug("gateway_singleton_registered", started=bool(discovered))
            except Exception as exc:
                log.warning("gateway_start_failed", error=str(exc))
                return "failed"
            if not discovered:
                return "disabled"
            return "ready" if all(self._channel_registry.get(name).status == "connected"
                                  for name in discovered) else "degraded"
        except Exception as exc:
            log.warning("channel_adapters_init_failed", error=str(exc))
            return "failed"

    # heartbeat

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

            # Register HEARTBEAT.md checker as a cron task
            try:
                from rune.daemon.heartbeat_checker import (
                    create_default_heartbeat,
                    run_heartbeat_checks,
                )
                create_default_heartbeat()

                async def _heartbeat_md_check() -> None:
                    results = await run_heartbeat_checks()
                    for r in results:
                        if r["needs_attention"]:
                            log.info(
                                "heartbeat_alert",
                                item=r["item"][:80],
                                output=r["output"][:200],
                            )
                            # Broadcast to connected channels
                            try:
                                msg = f"⚡ {r['item']}\n{r['output'][:300]}"
                                await self._broadcast_notification(msg)
                            except Exception:
                                pass

                self._heartbeat_scheduler.add_task(
                    "heartbeat_md_checker",
                    "* * * * *",  # every minute (checks are internally rate-limited)
                    _heartbeat_md_check,
                )
                log.info("heartbeat_md_checker_registered")
            except Exception as exc:
                log.debug("heartbeat_md_checker_failed", error=str(exc)[:200])

            # Schedule opt-in skill replay off the heartbeat path with overlap protection.
            try:
                self._heartbeat_scheduler.add_task(
                    "skill_eval_cycle",
                    "*/30 * * * *",  # every 30 min
                    self._skill_eval_tick,
                )
                log.info("skill_eval_cycle_registered")
            except Exception as exc:
                log.debug("skill_eval_cycle_register_failed", error=str(exc)[:200])

        except Exception as exc:
            log.warning("heartbeat_scheduler_init_failed", error=str(exc))

        # Start a simple file-based heartbeat indicator
        self._heartbeat_task = asyncio.create_task(self._heartbeat_file_loop())

    async def _broadcast_notification(self, message: str) -> None:
        """Deliver a notification through the gateway singleton, best-effort."""
        try:
            from rune.daemon.gateway import get_gateway

            gateway = get_gateway()
            if gateway is not None:
                await gateway.broadcast_text(message)
        except Exception as exc:
            log.debug("notification_broadcast_failed", error=str(exc)[:120])

    async def _skill_eval_tick(self) -> None:
        """Start an opt-in skill evaluation task without blocking or overlapping prior work."""
        try:
            from rune.config import get_config
            if not getattr(get_config().skills, "gated_learning", False):
                return
        except Exception:
            return
        if self._skill_eval_running:
            return
        self._skill_eval_running = True
        self._skill_eval_task = asyncio.create_task(self._run_skill_eval_cycle_bg())

    async def _run_skill_eval_cycle_bg(self) -> None:
        """Run one bounded gated-skill eval cycle in the background."""
        try:
            from rune.config import get_config
            from rune.memory.store import get_memory_store
            from rune.skills.eval_cycle import run_skill_eval_cycle
            from rune.skills.registry import get_skill_registry
            from rune.skills.replay import AgentLoopTaskRunner

            llm = get_config().llm
            runner = AgentLoopTaskRunner(
                provider=llm.active_provider or llm.default_provider,
                model=llm.active_model or llm.default_model,
                max_iterations=30,
                timeout_seconds=300,
            )
            reports = await run_skill_eval_cycle(
                store=get_memory_store(),
                registry=get_skill_registry(),
                runner=runner,
                max_skills=1,
            )
            if reports:
                log.info("skill_eval_cycle_completed", count=len(reports))
        except Exception as exc:
            log.debug("skill_eval_cycle_bg_failed", error=str(exc)[:200])
        finally:
            self._skill_eval_running = False

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

    # API server

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

        from rune.api.server import create_app

        app = create_app()
        app.state.runtime_health = self.health
        host = self._config.get("api_host", "127.0.0.1")
        port = self._config.get("api_port", 18789)
        self._api_server_task = asyncio.create_task(self._run_api_server(app, host, port))

    async def _run_api_server(self, app: Any, host: str, port: int) -> None:
        import uvicorn

        config = uvicorn.Config(app, host=host, port=port, log_level="warning", log_config=None,
                                timeout_graceful_shutdown=10)
        self._api_server = uvicorn.Server(config)
        self._api_server.capture_signals = contextlib.nullcontext
        try:
            await self._api_server.serve()
        except SystemExit as exc:
            raise RuntimeError("API server could not start") from exc

    # client handling

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
            if self._stop_requested.is_set():
                return {"success": False, "error": "Rune is shutting down"}
            return await self._execute_agent(params)
        elif command == "stop":
            if params.get("expected_pid", os.getpid()) != os.getpid():
                return {"success": False, "error": "The Rune instance changed"}
            self._stop_requested.set()
            return {"success": True, "pid": os.getpid(), "message": "Daemon stopping"}
        elif command == "health":
            return {"success": True, "version": VERSION, "pid": os.getpid(), **self.health()}
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
                "heartbeat": bool(self._heartbeat_scheduler and self._heartbeat_scheduler.is_running),
                "api_server": bool(self._api_server_task and not self._api_server_task.done()),
            },
        }

    async def _execute_agent(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute an agent loop for a background task."""
        goal = params.get("goal", "")
        if not goal:
            return {"success": False, "error": "Missing goal"}

        try:
            from rune.agent.background import BackgroundTask, run_background

            result = await run_background(BackgroundTask(
                goal=goal, source="daemon", workspace=params.get("cwd") or "",
            ))
            return {**result, "answer": result["output"]}
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
