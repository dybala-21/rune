"""RUNE TUI application using Rich + prompt_toolkit.

Two-zone rendering matching TS RUNE (Ink.js):
- Scrollback: Rich Console.print() - messages persist in terminal buffer
- Dynamic footer: Rich Live during agent runs (streaming + status)
- Input: prompt_toolkit PromptSession with bottom toolbar (idle status)
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console

from rune.ui.clipboard import copy_to_clipboard
from rune.ui.commands import _ALIAS_MAP, COMMANDS, parse_slash_command, suggest_command
from rune.ui.cost import estimate_cost, format_cost
from rune.ui.renderer import Renderer, format_idle_status, make_safe_console
from rune.ui.utils.output_styles import OutputStyleName, cycle_style
from rune.utils.logger import get_logger

__version__ = "0.1.0"
log = get_logger(__name__)


# prompt_toolkit style

_PT_STYLE = PTStyle.from_dict({
    "prompt": "#D4A017 bold",
    "bottom-toolbar": "bg:#0A0A0A #444444",
    "bottom-toolbar.text": "#444444",
    # Completion menu - transparent bg (inherits terminal), subtle highlight
    "completion-menu": "bg:default #888888",
    "completion-menu.completion": "bg:default #999999",
    "completion-menu.completion.current": "bg:#1A1F2E #E0E0F0",
    "completion-menu.meta.completion": "bg:default #555555",
    "completion-menu.meta.completion.current": "bg:#1A1F2E #8888CC",
    "scrollbar.background": "bg:default",
    "scrollbar.button": "bg:#333333",
})


# Tab completer: slash commands + @file references

def _build_known_models() -> list[tuple[str, str]]:
    """Build model list from the central registry (models.py)."""
    from rune.llm.models import (
        ANTHROPIC_MODELS,
        AZURE_MODELS,
        COHERE_MODELS,
        DEEPSEEK_MODELS,
        FALLBACK_OPENAI_MODELS,
        GEMINI_MODELS,
        MISTRAL_MODELS,
        XAI_MODELS,
    )
    result: list[tuple[str, str]] = []
    for m in FALLBACK_OPENAI_MODELS:
        result.append((m.provider, m.id))
    for m in ANTHROPIC_MODELS:
        result.append((m.provider, m.id))
    for m in GEMINI_MODELS:
        result.append((m.provider, m.id))
    for m in XAI_MODELS:
        result.append((m.provider, m.id))
    for m in AZURE_MODELS:
        result.append((m.provider, m.id))
    for m in MISTRAL_MODELS:
        result.append((m.provider, m.id))
    for m in DEEPSEEK_MODELS:
        result.append((m.provider, m.id))
    for m in COHERE_MODELS:
        result.append((m.provider, m.id))
    return result


_KNOWN_MODELS: list[tuple[str, str]] = _build_known_models()

_KNOWN_THEMES = ["dark", "light", "minimal"]

_EXPORT_FORMATS = ["markdown", "json", "html"]

_MEMORY_SUBS = ["show", "add", "clear"]

_STYLE_NAMES = ["compact", "normal", "verbose"]


class _RuneCompleter(Completer):
    """Tab completion for slash commands, arguments, and @file references."""

    def __init__(self) -> None:
        # Primary commands only (no aliases, no hidden)
        self._commands: list[tuple[str, str]] = []
        for cmd in COMMANDS.values():
            if not cmd.hidden:
                self._commands.append((cmd.name, cmd.description))
        # Sort by name for consistent display
        self._commands.sort(key=lambda x: x[0])

    def get_completions(self, document: Document, complete_event: Any) -> Any:
        text = document.text_before_cursor

        if text.startswith("/"):
            # Check if we're completing a command argument (has a space)
            parts = text.split(None, 1)
            cmd_part = parts[0].lower()

            if len(parts) == 2 or text.endswith(" "):
                # Completing argument for a known command
                arg_partial = parts[1].lower() if len(parts) == 2 else ""
                canonical = _ALIAS_MAP.get(cmd_part, cmd_part)
                yield from self._arg_completions(canonical, arg_partial, text)
                return

            # Slash command name completion - clean, no aliases
            partial = text.lower()
            for name, desc in self._commands:
                if name.startswith(partial):
                    yield Completion(
                        name,
                        start_position=-len(text),
                        display_meta=desc,
                    )
            return

        # @file reference completion
        word = document.get_word_before_cursor(WORD=True)
        if word.startswith("@") and len(word) > 1:
            from rune.ui.file_autocomplete import complete_at_reference
            items = complete_at_reference(word, cwd=Path.cwd())
            for item in items:
                yield Completion(
                    f"@{item.text}",
                    start_position=-len(word),
                    display=item.display,
                )

    def _arg_completions(
        self, cmd: str, partial: str, full_text: str
    ) -> Any:
        """Yield completions for command arguments."""
        len(full_text) - len(partial)

        if cmd == "/model" or cmd == "/models":
            if ":" not in partial:
                # Step 1: complete provider name
                seen_providers: list[str] = []
                for prov, _ in _KNOWN_MODELS:
                    if prov not in seen_providers:
                        seen_providers.append(prov)
                for prov in seen_providers:
                    if prov.startswith(partial):
                        count = sum(1 for p, _ in _KNOWN_MODELS if p == prov)
                        yield Completion(
                            f"{prov}:",
                            start_position=-len(partial),
                            display=prov,
                            display_meta=f"{count} models",
                        )
            else:
                # Step 2: complete model within provider
                prov_prefix, model_partial = partial.split(":", 1)
                for prov, model in _KNOWN_MODELS:
                    if prov == prov_prefix and model.startswith(model_partial):
                        yield Completion(
                            f"{prov}:{model}",
                            start_position=-len(partial),
                            display=model,
                            display_meta=prov,
                        )
        elif cmd == "/theme":
            for t in _KNOWN_THEMES:
                if t.startswith(partial):
                    yield Completion(t, start_position=-len(partial))
        elif cmd == "/export":
            for f in _EXPORT_FORMATS:
                if f.startswith(partial):
                    yield Completion(f, start_position=-len(partial))
        elif cmd == "/memory":
            for s in _MEMORY_SUBS:
                if s.startswith(partial):
                    yield Completion(s, start_position=-len(partial))
        elif cmd == "/style":
            for s in _STYLE_NAMES:
                if s.startswith(partial):
                    yield Completion(s, start_position=-len(partial))
        elif cmd == "/load":
            try:
                import asyncio

                from rune.ui.sessions import list_sessions
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return  # Can't await in sync context
                items = loop.run_until_complete(list_sessions())
                for item in items:
                    if item.id.startswith(partial) or item.name.startswith(partial):
                        yield Completion(
                            item.id,
                            start_position=-len(partial),
                            display_meta=item.name or item.preview[:40],
                        )
            except Exception:
                pass


# RuneApp

class RuneApp:
    """Main RUNE application: Rich Console + prompt_toolkit."""

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        # Load model: active (from /model) > CLI arg > config default
        from rune.config import get_config
        cfg = get_config()
        self._provider = (
            provider
            or cfg.llm.active_provider
            or cfg.llm.default_provider
            or "openai"
        )
        self._model = (
            model
            or cfg.llm.active_model
            or cfg.llm.default_model
            or "gpt-5.4"
        )

        # Rich console for scrollback output (blocking-IO safe for Live threads)
        self.console = make_safe_console()
        self.renderer = Renderer(self.console)

        # prompt_toolkit session for input with persistent history
        self._kb = self._build_key_bindings()
        from rune.utils.paths import rune_data_dir
        history_path = rune_data_dir() / "prompt_history"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        self._completer = _RuneCompleter()

        # Auto-complete only when input starts with / or @
        @Condition
        def _should_complete() -> bool:
            buf = self._session.default_buffer
            text = buf.text
            return text.startswith("/") or ("@" in text)

        self._session = PromptSession(
            key_bindings=self._kb,
            history=FileHistory(str(history_path)),
            style=_PT_STYLE,
            bottom_toolbar=self._bottom_toolbar,
            completer=self._completer,
            complete_while_typing=_should_complete,
            complete_style=CompleteStyle.MULTI_COLUMN,
            multiline=False,
        )
        self._multiline = False

        # Agent integration
        self._on_user_message: Any | None = None
        self._agent_controller: Any | None = None
        self._file_tracker: Any | None = None
        self._agent_running: bool = False
        self._shutting_down: bool = False
        self._shutdown_reason: str | None = None
        self._loop_exception_handler: Any | None = None
        self._SIGINT_EXIT_GRACE_S: float = 0.35
        self._main_sigint_handler_installed: bool = False

        # State tracking
        self._last_abort_time: float = 0.0
        self._user_message_history: list[str] = []
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._session_start: float = time.monotonic()
        self._output_style: OutputStyleName = "normal"
        self._undo_count: int = 0
        self._last_response_text: str = ""

    # ===================================================================
    # Main loop
    # ===================================================================

    def run(self) -> None:
        """Main entry point - runs the async main loop."""
        original_policy = asyncio.get_event_loop_policy()
        restore_policy = False

        # prompt_toolkit shutdown is more reliable on the stdlib event loop.
        if type(original_policy).__module__.startswith("uvloop"):
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
            restore_policy = True

        try:
            asyncio.run(self._main_loop())
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            if restore_policy:
                asyncio.set_event_loop_policy(original_policy)
            self._safe_goodbye()

    def _safe_goodbye(self) -> None:
        """Print goodbye message, handling terminal state errors gracefully."""
        import fcntl
        import termios

        # 1. Restore blocking mode on stdout
        for stream in (sys.stdout, sys.stderr):
            try:
                fd = stream.fileno()
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                if flags & os.O_NONBLOCK:
                    fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
            except (OSError, ValueError):
                pass

        # 2. Restore terminal to sane cooked mode (in case raw mode leaked)
        try:
            fd = sys.stdin.fileno()
            termios.tcsetattr(fd, termios.TCSADRAIN, termios.tcgetattr(fd))
        except (OSError, ValueError, termios.error):
            pass

        # 3. Reset terminal state: show cursor, clear attributes
        try:
            sys.stdout.write("\033[?25h\033[0m")
            sys.stdout.flush()
        except Exception:
            pass

        # 4. Print goodbye
        try:
            Console(stderr=True).print("\n[dim]Goodbye.[/dim]")
        except Exception:
            try:
                sys.stderr.write("\nGoodbye.\n")
                sys.stderr.flush()
            except Exception:
                pass

    async def _start_channel_adapters(self) -> None:
        """Auto-discover and start channel adapters (Telegram, Discord, etc.).

        Runs as a background task alongside the TUI prompt loop.
        If no channel tokens are configured, this is a no-op.
        """
        try:
            from rune.channels.registry import auto_discover_channels, get_channel_registry

            discovered = auto_discover_channels()
            if not discovered:
                return

            registry = get_channel_registry()
            self._channel_registry = registry

            # Limit concurrent agent runs from channels to prevent resource exhaustion
            _channel_semaphore = asyncio.Semaphore(2)

            # Wire on_message: channel message → agent loop → channel reply
            async def _handle_channel_message(msg: Any) -> None:
                from rune.agent.agent_context import PrepareContextOptions, prepare_agent_context
                from rune.agent.loop import NativeAgentLoop

                if _channel_semaphore.locked():
                    # All slots busy - notify user
                    channel_name = msg.metadata.get("channel_name", "")
                    adapter = registry.get(channel_name)
                    if adapter:
                        from rune.channels.types import OutgoingMessage
                        await adapter.send(msg.channel_id, OutgoingMessage(
                            text="⏳ Processing previous request. Please wait.",
                        ))

                async with _channel_semaphore:
                    try:
                        channel_name = msg.metadata.get("channel_name", "")
                        ctx = await prepare_agent_context(
                            PrepareContextOptions(
                                goal=msg.text,
                                channel=channel_name,
                                sender_id=msg.sender_id,
                            )
                        )
                        agent_loop = NativeAgentLoop()

                        # Collect streamed text via event listener
                        collected: list[str] = []
                        agent_loop.on("text_delta", lambda delta: collected.append(delta))

                        await agent_loop.run(
                            ctx.goal,
                            context={"workspace_root": ctx.workspace_root},
                        )

                        answer = "".join(collected).strip()
                        adapter = registry.get(channel_name)
                        if adapter and answer:
                            from rune.channels.types import OutgoingMessage
                            await adapter.send(msg.channel_id, OutgoingMessage(
                                text=answer,
                                reply_to=msg.metadata.get("message_id"),
                            ))
                    except Exception as exc:
                        from rune.utils.logger import get_logger
                        get_logger(__name__).warning(
                            "channel_message_failed",
                            channel=getattr(msg, "channel_id", ""),
                            error=str(exc)[:200],
                        )

            for name in discovered:
                adapter = registry.get(name)
                if adapter is not None:
                    adapter.on_message = _handle_channel_message

            await registry.start_all()
            channels_str = ", ".join(discovered)
            self.console.print(f"[dim]Channels active: {channels_str}[/dim]")
        except Exception as exc:
            from rune.utils.logger import get_logger
            get_logger(__name__).debug("channel_autostart_failed", error=str(exc)[:200])

    async def _stop_channel_adapters(self) -> None:
        """Stop channel adapters if running."""
        if hasattr(self, "_channel_registry") and self._channel_registry is not None:
            try:
                await self._channel_registry.stop_all()
            except Exception:
                pass

    async def _start_mcp_bridge(self) -> None:
        """Connect configured MCP servers in background.

        Loads ~/.rune/mcp_servers.json, connects to each server,
        and registers their tools in the global CapabilityRegistry.
        No-op if no servers are configured.
        """
        try:
            from rune.mcp.config import load_mcp_config

            configs = load_mcp_config()
            if not configs:
                return

            from rune.mcp.bridge import initialize_mcp_bridge

            result = await initialize_mcp_bridge(configs=configs)
            if result.connected_servers > 0:
                self.console.print(
                    f"[dim]MCP: {result.connected_servers} server(s), "
                    f"{result.registered_count} tools[/dim]"
                )
        except Exception as exc:
            from rune.utils.logger import get_logger
            get_logger(__name__).debug(
                "mcp_bridge_autostart_failed", error=str(exc)[:200]
            )

    async def _main_loop(self) -> None:
        """Async main loop: print welcome, then read-eval-agent loop."""
        self._shutting_down = False
        self._channel_registry = None
        loop = asyncio.get_running_loop()
        self._install_loop_exception_handler(loop)
        original_sigint = signal.getsignal(signal.SIGINT)
        self._install_main_sigint_handler(loop)
        self._print_welcome()
        self._show_learning_notice()

        # Start channel adapters in background (Telegram, Discord, etc.)
        asyncio.create_task(self._start_channel_adapters())

        # Start MCP bridge in background (connects configured MCP servers)
        asyncio.create_task(self._start_mcp_bridge())

        try:
            while True:
                try:
                    prompt_msg = "❯ " if not self._multiline else "❯❯ "
                    text = await self._session.prompt_async(
                        [("class:prompt", prompt_msg)],
                        multiline=self._multiline,
                    )
                except EOFError:
                    self._request_shutdown("eof")
                    break
                except KeyboardInterrupt:
                    # Single Ctrl+C at prompt - check for double-press quit
                    now = time.monotonic()
                    if now - self._last_abort_time < 2.0:
                        self._request_shutdown("sigint")
                        break
                    self._last_abort_time = now
                    self.console.print("[dim]Press Ctrl+C again to exit.[/dim]")
                    continue
                except Exception as exc:
                    if self._is_benign_shutdown_exception(exc):
                        self._request_shutdown("driver_closed")
                        break
                    raise

                text = text.strip()
                if not text:
                    continue

                # Slash commands
                if text.startswith("/"):
                    await self._handle_slash_command_text(text)
                    continue

                # Regular message -> agent
                self._user_message_history.append(text)

                if self._agent_controller is not None:
                    await self._run_agent(text)
                else:
                    self.renderer.print_system_message("Agent not connected. No controller set.")
        finally:
            self._request_shutdown(self._shutdown_reason or "cleanup")
            # Stop channel adapters (Telegram, Discord, etc.)
            await self._stop_channel_adapters()
            # Close browser to prevent "Target page has been closed" errors
            try:
                from rune.capabilities.browser import _close_browser
                await _close_browser()
            except Exception:
                pass
            self._close_prompt_session()
            if self._requires_sigint_exit_grace():
                await asyncio.sleep(self._SIGINT_EXIT_GRACE_S)
            signal.signal(signal.SIGINT, original_sigint)
            self._main_sigint_handler_installed = False
            self._restore_loop_exception_handler(loop)

    # ===================================================================
    # Agent run with Live display
    # ===================================================================

    async def _run_agent(self, text: str) -> None:
        """Run the agent loop with Rich Live footer for streaming/status."""
        self._agent_running = True
        live = self.renderer.start_live()

        # Set up Ctrl+C to cancel agent during run
        asyncio.get_running_loop()
        original_sigint = signal.getsignal(signal.SIGINT)

        cancel_requested = False

        def _sigint_handler(signum: int, frame: Any) -> None:
            nonlocal cancel_requested
            cancel_requested = True
            if self._agent_controller is not None:
                asyncio.ensure_future(self._agent_controller.cancel())

        signal.signal(signal.SIGINT, _sigint_handler)

        try:
            with live:
                self.renderer._live = live  # noqa: SLF001

                # Start the agent loop
                if self._agent_controller is not None:
                    await self._agent_controller.start(text)

                    # Wait for the task to complete
                    task = self._agent_controller._task  # noqa: SLF001
                    if task is not None:
                        # Poll while waiting, to allow spinner ticks
                        while not task.done():
                            await asyncio.sleep(_SPINNER_INTERVAL)
                            self.renderer.tick_spinner()
                        # Get result (controller already handles errors internally)
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        except Exception as exc:
                            # Controller should have shown this, but log as fallback
                            self.renderer.print_system_message(f"Error: {exc}")

        except (KeyboardInterrupt, asyncio.CancelledError):
            if self._agent_controller is not None:
                with contextlib.suppress(Exception):
                    await self._agent_controller.cancel()
            self.renderer.print_system_message("Cancelled.")
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            self.renderer.stop_live()
            self._agent_running = False

        if cancel_requested:
            self.renderer.print_system_message("Agent cancelled.")

    # ===================================================================
    # Welcome
    # ===================================================================

    def _print_welcome(self) -> None:
        """Print the RUNE welcome banner."""
        width = min(self.console.width or 80, 80)
        rule_w = width - 4
        label = " rune "
        left = 3
        right = max(0, rule_w - left - len(label))
        self.console.print()
        self.console.print(
            f"  [#333333]{'─' * left}[/#333333]"
            f"[bold #D4A017]{label}[/bold #D4A017]"
            f"[#333333]{'─' * right}[/#333333]"
        )
        self.console.print()
        self.console.print(
            f"  [#666666]Terminal Agent[/#666666]  "
            f"[#D4A017]·[/#D4A017]  "
            f"[#AAAAAA]{self._provider}:{self._model}[/#AAAAAA]  "
            f"[#333333]·[/#333333]  "
            f"[#555555]/help for commands[/#555555]"
        )
        self.console.print()

    def _handle_learning_command(self, arg: str) -> None:
        """Handle /learning on|off slash command."""
        try:
            from rune.memory.consolidation import is_consolidation_enabled, set_learning_enabled

            lower = arg.strip().lower()
            if lower == "on":
                set_learning_enabled(True)
                self.console.print(
                    "  [#56B6C2]📝 Learning enabled.[/#56B6C2]"
                )
            elif lower == "off":
                set_learning_enabled(False)
                self.console.print(
                    "  [#888888]📝 Learning disabled.[/#888888]"
                )
            else:
                status = "on" if is_consolidation_enabled() else "off"
                self.console.print(
                    f"  [#888888]📝 Learning is {status}. Usage: /learning on | /learning off[/#888888]"
                )
        except Exception:
            self.console.print("  [#888888]📝 Learning setting unavailable.[/#888888]")

    def _show_learning_notice(self) -> None:
        """Show first-run learning notice if not yet configured."""
        try:
            from rune.memory.consolidation import is_first_run, set_learning_enabled
            if not is_first_run():
                return

            self.console.print()
            self.console.print(
                "  [#56B6C2]📝 Learning enabled — RUNE will remember past work.[/#56B6C2]"
            )
            self.console.print(
                "  [#888888]Small cost per conversation (~$0.003).[/#888888]"
            )
            self.console.print(
                "  [#888888]Disable: /learning off[/#888888]"
            )
            self.console.print()

            # Mark as configured (default: on)
            set_learning_enabled(True)
        except Exception:
            pass

    # ===================================================================
    # Bottom toolbar (idle status bar)
    # ===================================================================

    def _bottom_toolbar(self) -> str:
        """Return idle status bar text for prompt_toolkit."""
        return format_idle_status(
            model=self._model,
            provider=self._provider,
            output_style=self._output_style,
            undo_count=self._undo_count,
            multiline=self._multiline,
        )

    # ===================================================================
    # Key bindings
    # ===================================================================

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("c-s")
        def _cycle_style(event: Any) -> None:
            self._output_style = cycle_style(self._output_style)
            self.renderer.set_output_style(self._output_style)
            self.console.print(f"[dim]Output style: {self._output_style}[/dim]")

        @kb.add("c-l")
        def _clear_view(event: Any) -> None:
            self.renderer.clear_messages()
            self.console.clear()
            self._print_welcome()

        @kb.add("c-y")
        def _copy_response(event: Any) -> None:
            if self._last_response_text:
                if copy_to_clipboard(self._last_response_text):
                    self.console.print("[dim]Response copied to clipboard.[/dim]")
                else:
                    self.console.print("[dim]Clipboard not available.[/dim]")

        @kb.add("c-j")
        def _toggle_multiline(event: Any) -> None:
            self._multiline = not self._multiline
            mode = "multi-line (Enter=newline, Esc+Enter=submit)" if self._multiline else "single-line"
            self.console.print(f"[dim]Input mode: {mode}[/dim]")

        return kb

    # ===================================================================
    # Slash commands
    # ===================================================================

    async def _handle_slash_command_text(self, text: str) -> None:
        """Parse and dispatch a slash command."""
        parsed = parse_slash_command(text)
        if parsed is None:
            cmd_text = text.split()[0]
            suggestion = suggest_command(text)
            if suggestion:
                self.console.print(
                    f"[yellow]Unknown command: {cmd_text}[/yellow] "
                    f"[dim]— did you mean[/dim] [bold]{suggestion}[/bold][dim]?[/dim]"
                )
            else:
                self.console.print(f"[yellow]Unknown command: {cmd_text}[/yellow] [dim]— /help for list[/dim]")
            return

        cmd_name, cmd_args = parsed

        # App-level commands
        if cmd_name == "/exit":
            self._request_shutdown("slash_exit")
            raise EOFError

        if cmd_name == "/clear":
            self.console.clear()
            self._print_welcome()
            self.renderer.clear_messages()
            return

        if cmd_name == "/help":
            await self._show_help()
            return

        if cmd_name == "/model":
            if cmd_args.strip():
                self._apply_model(cmd_args.strip())
            else:
                await self._interactive_model_select()
            return

        if cmd_name == "/learning":
            self._handle_learning_command(cmd_args)
            return

        # Run the command handler
        cmd = COMMANDS.get(cmd_name)
        if cmd is not None:
            result = await cmd.handler(cmd_args)
            if result:
                await self._handle_action_result(result)

    async def _handle_action_result(self, result: str) -> None:
        """Dispatch __ACTION__ markers returned by command handlers."""
        if not result.startswith("__ACTION__:"):
            self.renderer.print_system_message(result)
            return

        action = result[len("__ACTION__:"):]

        if action == "undo":
            self._do_undo()
        elif action == "retry":
            await self._do_retry()
        elif action == "cycle_style":
            self._output_style = cycle_style(self._output_style)
            self.renderer.set_output_style(self._output_style)
            self.console.print(f"[dim]Output style: {self._output_style}[/dim]")
        elif action.startswith("set_style:"):
            style_name = action.split(":", 1)[1]
            if style_name in ("compact", "normal", "verbose"):
                self._output_style = style_name  # type: ignore[assignment]
                self.renderer.set_output_style(self._output_style)
                self.console.print(f"[dim]Output style: {style_name}[/dim]")
        elif action == "toggle_files":
            self._show_file_changes()
        elif action == "toggle_git_diff":
            self._show_git_diff()
        elif action == "copy_response":
            if self._last_response_text:
                if copy_to_clipboard(self._last_response_text):
                    self.console.print("[dim]Response copied to clipboard.[/dim]")
            else:
                self.console.print("[dim]No assistant message to copy.[/dim]")
        elif action.startswith("save"):
            parts = action.split(":", 1)
            name = parts[1] if len(parts) > 1 else None
            await self._do_save_session(name)
        elif action.startswith("load:"):
            session_id = action.split(":", 1)[1]
            await self._do_load_session(session_id)
        elif action.startswith("search:"):
            query = action.split(":", 1)[1]
            self._do_search(query)
        elif action.startswith("export:"):
            fmt = action.split(":", 1)[1]
            await self._do_export(fmt)
        elif action == "show_session":
            self._show_session_info()
        elif action == "show_sessions":
            await self._show_sessions_list()
        elif action == "interactive_load":
            await self._interactive_load_session()
        elif action == "show_status":
            self._show_agent_status()
        elif action == "show_config":
            self._show_config()
        elif action == "interactive_theme":
            await self._interactive_theme_select()
        elif action == "interactive_export":
            await self._interactive_export()
        elif action.startswith("set_theme:"):
            theme_name = action.split(":", 1)[1]
            self._do_set_theme(theme_name)
        elif action == "show_cost":
            self._show_cost()
        elif action == "show_stats":
            self._show_stats()
        elif action.startswith("memory:"):
            sub = action.split(":", 1)[1]
            await self._do_memory_operation(sub)
        else:
            self.renderer.print_system_message(result)

    # ===================================================================
    # Action implementations
    # ===================================================================

    async def _show_help(self) -> None:
        """Display available commands grouped by category."""
        from rich.text import Text as RText

        _cat_map = {
            "/help": "General", "/exit": "General", "/clear": "General",
            "/model": "General", "/theme": "General", "/config": "General",
            "/retry": "Agent", "/undo": "Agent", "/memory": "Agent",
            "/style": "Output", "/copy": "Output",
            "/files": "Files & Git", "/diff": "Files & Git",
            "/save": "Session", "/load": "Session", "/sessions": "Session",
            "/export": "Session", "/search": "Session",
            "/cost": "Info", "/stats": "Info", "/status": "Info",
            "/session": "Info",
        }
        groups: dict[str, list[tuple[str, str, str, str]]] = {
            "General": [], "Agent": [], "Output": [],
            "Files & Git": [], "Session": [], "Info": [],
        }
        for cmd in COMMANDS.values():
            if cmd.hidden:
                continue
            cat = _cat_map.get(cmd.name, "General")
            alias_str = ", ".join(cmd.aliases) if cmd.aliases else ""
            usage_str = cmd.usage or ""
            groups.setdefault(cat, []).append((cmd.name, cmd.description, alias_str, usage_str))

        self.console.print()
        for group_name, items in groups.items():
            if not items:
                continue
            # Section header
            self.console.print(f"  [bold #D4A017]{group_name}[/bold #D4A017]")
            for name, desc, alias, _usage in items:
                line = RText()
                line.append(f"    {name:<14}", style="bold #00CED1")
                line.append(desc, style="#999999")
                if alias:
                    line.append(f"  ({alias})", style="#555555")
                self.console.print(line)
            self.console.print()

        # Keybindings
        keys = RText()
        keys.append("  Shortcuts  ", style="#555555")
        for key, label in [("^C", "exit"), ("^S", "style"), ("^L", "clear"), ("^Y", "copy"), ("^J", "multi")]:
            keys.append(key, style="bold #E0E0E0")
            keys.append(f" {label}  ", style="#555555")
        self.console.print(keys)
        self.console.print()

    def _apply_model(self, value: str) -> None:
        """Apply a model selection string (provider:model or just model)."""
        if ":" in value:
            self._provider, self._model = value.split(":", 1)
        else:
            # Try to find the provider
            for prov, model in _KNOWN_MODELS:
                if model == value:
                    self._provider = prov
                    self._model = model
                    break
            else:
                self._model = value
        # Persist to config (in-memory)
        from rune.config import get_config
        cfg = get_config()
        cfg.llm.default_provider = self._provider
        cfg.llm.default_model = self._model
        cfg.llm.active_provider = self._provider
        cfg.llm.active_model = self._model

        # Persist to config.yaml (disk)
        self._save_active_model_to_yaml(self._provider, self._model)

        # Update agent config if controller is connected
        if self._agent_controller and hasattr(self._agent_controller, "_loop"):
            loop_obj = self._agent_controller._loop
            if hasattr(loop_obj, "_config"):
                loop_obj._config.model = self._model
                loop_obj._config.provider = self._provider
                loop_obj._config._overridden = True

            # Rebuild failover profiles with new model
            from rune.agent.failover import build_profiles_from_config
            if hasattr(loop_obj, "_failover"):
                loop_obj._failover._profiles = build_profiles_from_config()

        self.console.print(f"[bold #D4A017]Model:[/bold #D4A017] {self._provider}:{self._model}")

    @staticmethod
    def _save_active_model_to_yaml(provider: str, model: str) -> None:
        """Write activeProvider/activeModel to ~/.rune/config.yaml."""
        from rune.utils.paths import rune_home

        cfg_path = rune_home() / "config.yaml"
        if not cfg_path.exists():
            return
        try:
            from ruamel.yaml import YAML
            yaml = YAML()
            yaml.preserve_quotes = True
            data = yaml.load(cfg_path) or {}
            llm = data.setdefault("llm", {})
            llm["activeProvider"] = provider
            llm["activeModel"] = model
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f)
        except Exception:
            pass  # best-effort, don't crash on save failure

    async def _interactive_model_select(self) -> None:
        """Two-step model selection: provider → model."""
        from rune.ui.inline_select import inline_select

        # Step 1: Select provider
        providers: list[str] = []
        for prov, _ in _KNOWN_MODELS:
            if prov not in providers:
                providers.append(prov)

        provider_items = [(p, p) for p in providers]
        default_prov_idx = 0
        for i, p in enumerate(providers):
            if p == self._provider:
                default_prov_idx = i

        prov_idx = inline_select(
            provider_items,
            title="Select provider",
            default_index=default_prov_idx,
        )
        if prov_idx is None:
            return

        selected_provider = providers[prov_idx]

        # Step 2: Select model within provider
        models = [(m, m) for p, m in _KNOWN_MODELS if p == selected_provider]
        if not models:
            return

        default_model_idx = 0
        for i, (m, _) in enumerate(models):
            if m == self._model:
                default_model_idx = i

        model_idx = inline_select(
            models,
            title=f"{selected_provider} models",
            default_index=default_model_idx,
        )
        if model_idx is None:
            return

        self._apply_model(f"{selected_provider}:{models[model_idx][0]}")

    def _do_undo(self) -> None:
        """Undo the last file change."""
        if self._file_tracker is None:
            self.console.print("[yellow]File tracker not connected.[/yellow]")
            return

        changed = self._file_tracker.get_changed_files()
        if not changed:
            self.console.print("[dim]Nothing to undo.[/dim]")
            return

        last_file = changed[-1]
        snapshot = self._file_tracker.get_snapshot(last_file)
        if snapshot is None:
            self.console.print(f"[yellow]No snapshot for {last_file}.[/yellow]")
            return

        try:
            workspace = self._file_tracker._workspace
            target = workspace / last_file
            target.write_text(snapshot, encoding="utf-8")
            self._file_tracker._changed_files.discard(last_file)
            if last_file in self._file_tracker._snapshots:
                del self._file_tracker._snapshots[last_file]
            self.console.print(f"[dim]Reverted: {last_file}[/dim]")
            self._undo_count = max(0, self._undo_count - 1)
        except OSError as exc:
            self.console.print(f"[red]Undo failed: {exc}[/red]")

    async def _do_retry(self) -> None:
        """Re-run the last user message."""
        if not self._user_message_history:
            self.console.print("[yellow]No previous message to retry.[/yellow]")
            return
        last_msg = self._user_message_history[-1]
        self.renderer.print_system_message(f"Retrying: {last_msg[:80]}...")
        if self._agent_controller is not None:
            await self._run_agent(last_msg)

    def _show_file_changes(self) -> None:
        """Show file changes."""
        if self._file_tracker is None:
            self.console.print("[yellow]File tracker not connected.[/yellow]")
            return
        changed = self._file_tracker.get_changed_files()
        if not changed:
            self.console.print("[dim]No file changes.[/dim]")
            return
        self.console.print("\n[bold]File changes:[/bold]")
        for f in changed:
            self.console.print(f"  [#D4A017]◆[/#D4A017] {f}")
        self.console.print()

    def _show_git_diff(self) -> None:
        """Show git diff."""
        try:
            result = subprocess.run(
                ["git", "diff"],
                capture_output=True, text=True, timeout=10,
                cwd=str(Path.cwd()),
            )
            if result.stdout.strip():
                self.console.print("\n[bold]Git diff:[/bold]\n")
                for line in result.stdout.split("\n")[:100]:
                    if line.startswith("+") and not line.startswith("+++"):
                        self.console.print(f"[green]{line}[/green]")
                    elif line.startswith("-") and not line.startswith("---"):
                        self.console.print(f"[red]{line}[/red]")
                    else:
                        self.console.print(f"[dim]{line}[/dim]")
                self.console.print()
            else:
                self.console.print("[dim]No git diff available.[/dim]")
        except (subprocess.SubprocessError, OSError) as exc:
            self.console.print(f"[red]Failed to get git diff: {exc}[/red]")

    def _show_cost(self) -> None:
        """Show estimated API cost."""
        from rich.panel import Panel
        from rich.table import Table
        cost = estimate_cost(self._model, self._total_input_tokens, self._total_output_tokens)
        cost_str = format_cost(cost)
        from rune.ui.theme import format_tokens
        table = Table(show_header=False, show_edge=False, padding=(0, 1), expand=True)
        table.add_column("key", style="#888888", width=16)
        table.add_column("val", style="bold #E0E0E0")
        table.add_row("Model", f"[#00CED1]{self._model}[/#00CED1]")
        table.add_row("Input tokens", format_tokens(self._total_input_tokens))
        table.add_row("Output tokens", format_tokens(self._total_output_tokens))
        table.add_row("Estimated", f"[bold #D4A017]{cost_str}[/bold #D4A017]")
        self.console.print(Panel(table, title="[bold #D4A017]💰 Cost[/bold #D4A017]", title_align="left", border_style="#333333", padding=(0, 1)))

    def _show_stats(self) -> None:
        """Show session stats."""
        from rich.panel import Panel
        from rich.table import Table

        from rune.ui.theme import format_duration, format_tokens
        elapsed = (time.monotonic() - self._session_start) * 1000
        table = Table(show_header=False, show_edge=False, padding=(0, 1), expand=True)
        table.add_column("key", style="#888888", width=16)
        table.add_column("val", style="#E0E0E0")
        table.add_row("Duration", f"[bold]{format_duration(elapsed)}[/bold]")
        table.add_row("Messages", f"{len(self._user_message_history)}")
        table.add_row("Input tokens", format_tokens(self._total_input_tokens))
        table.add_row("Output tokens", format_tokens(self._total_output_tokens))
        table.add_row("Style", self._output_style)
        table.add_row("Model", f"[#00CED1]{self._provider}:{self._model}[/#00CED1]")
        if self._file_tracker:
            changed = self._file_tracker.get_changed_files()
            table.add_row("Files changed", f"{len(changed)}")
        self.console.print(Panel(table, title="[bold #D4A017]📊 Stats[/bold #D4A017]", title_align="left", border_style="#333333", padding=(0, 1)))

    async def _do_save_session(self, name: str | None = None) -> None:
        """Save the current session."""
        from rune.ui.sessions import SerializedMessage, save_session
        session_id = str(uuid.uuid4())[:8]
        session_name = name or f"session-{session_id}"
        serialized = [
            SerializedMessage(id=f"msg-{i}", role="user", content=msg, timestamp="")
            for i, msg in enumerate(self._user_message_history)
        ]
        try:
            fp = await save_session(session_id, session_name, serialized, [])
            self.console.print(f"[dim]Session saved: {session_name} ({fp.name})[/dim]")
        except Exception as exc:
            self.console.print(f"[red]Save failed: {exc}[/red]")

    async def _do_load_session(self, session_id: str) -> None:
        """Load a saved session."""
        from rune.ui.sessions import load_session
        session = await load_session(session_id)
        if session is None:
            self.console.print(f"[yellow]Session '{session_id}' not found.[/yellow]")
            return
        self._user_message_history.clear()
        for msg in session.messages:
            if msg.role == "user":
                self.renderer.print_user_message(msg.content)
                self._user_message_history.append(msg.content)
            elif msg.role == "assistant":
                self.renderer.print_assistant_response(msg.content)
            else:
                self.renderer.print_system_message(msg.content)
        self.console.print(f"[dim]Loaded session: {session.name}[/dim]")

    def _do_search(self, query: str) -> None:
        """Search through message history."""
        query_lower = query.lower()
        matches: list[tuple[int, str]] = []
        for i, msg in enumerate(self._user_message_history):
            if query_lower in msg.lower():
                matches.append((i + 1, msg))

        if not matches:
            self.console.print(f"[dim]  No matches for '{query}'.[/dim]")
            return
        from rich.panel import Panel
        from rich.text import Text
        results = Text()
        for idx, msg in matches[-20:]:
            preview = msg[:100] + ("…" if len(msg) > 100 else "")
            results.append(f"  #{idx}", style="bold #D4A017")
            results.append(f"  {preview}\n", style="#CCCCCC")
        self.console.print(Panel(results, title=f"[bold #D4A017]🔍 '{query}' ({len(matches)} matches)[/bold #D4A017]", title_align="left", border_style="#333333", padding=(0, 1)))

    async def _do_export(self, fmt: str) -> None:
        """Export conversation to file."""
        from rune.ui.export import ExportMessage, export_conversation
        messages = [
            ExportMessage(role="user", content=msg)
            for msg in self._user_message_history
        ]
        if self._last_response_text:
            messages.append(ExportMessage(role="assistant", content=self._last_response_text))
        try:
            fp = await export_conversation(messages, [], fmt=fmt)
            self.console.print(f"[green]Exported to:[/green] {fp}")
        except Exception as exc:
            self.console.print(f"[red]Export failed: {exc}[/red]")

    def _show_session_info(self) -> None:
        """Show current session information."""
        from rich.panel import Panel
        from rich.table import Table

        from rune.ui.theme import format_duration
        elapsed = (time.monotonic() - self._session_start) * 1000
        table = Table(show_header=False, show_edge=False, padding=(0, 1), expand=True)
        table.add_column("key", style="#888888", width=16)
        table.add_column("val", style="#E0E0E0")
        table.add_row("Model", f"[bold #00CED1]{self._provider}:{self._model}[/bold #00CED1]")
        table.add_row("Style", self._output_style)
        table.add_row("Uptime", format_duration(elapsed))
        table.add_row("Messages", str(len(self._user_message_history)))
        agent_status = "[#98C379]connected[/#98C379]" if self._agent_controller else "[dim]disconnected[/dim]"
        table.add_row("Agent", agent_status)
        self.console.print(Panel(table, title="[bold #D4A017]📋 Session[/bold #D4A017]", title_align="left", border_style="#333333", padding=(0, 1)))

    def _show_agent_status(self) -> None:
        """Show agent status."""
        from rich.panel import Panel
        from rich.table import Table
        status = "running" if self._agent_running else "idle"
        color = "#D4A017" if self._agent_running else "#98C379"
        table = Table(show_header=False, show_edge=False, padding=(0, 1), expand=True)
        table.add_column("key", style="#888888", width=16)
        table.add_column("val", style="#E0E0E0")
        table.add_row("Status", f"[bold {color}]{status}[/bold {color}]")
        if self._agent_controller:
            ctrl = self._agent_controller
            if hasattr(ctrl, "_step_count") and ctrl._step_count > 0:
                table.add_row("Steps", str(ctrl._step_count))
            if hasattr(ctrl, "_tool_count") and ctrl._tool_count > 0:
                table.add_row("Tools", str(ctrl._tool_count))
        self.console.print(Panel(table, title="[bold #D4A017]🤖 Agent[/bold #D4A017]", title_align="left", border_style="#333333", padding=(0, 1)))

    def _show_config(self) -> None:
        """Show current RUNE configuration."""
        from rich.panel import Panel
        from rich.table import Table
        try:
            from rune.config import get_config
            config = get_config()
            table = Table(show_header=False, show_edge=False, padding=(0, 1), expand=True)
            table.add_column("key", style="#888888", width=16)
            table.add_column("val", style="#E0E0E0")
            table.add_row("Model", f"[bold #00CED1]{self._provider}:{self._model}[/bold #00CED1]")
            table.add_row("Style", self._output_style)
            if hasattr(config, "openai_api_key") and config.openai_api_key:
                key = config.openai_api_key
                table.add_row("OpenAI key", f"[dim]{key[:8]}…{key[-4:]}[/dim]")
            if hasattr(config, "anthropic_api_key") and config.anthropic_api_key:
                key = config.anthropic_api_key
                table.add_row("Anthropic", f"[dim]{key[:8]}…{key[-4:]}[/dim]")
            if hasattr(config, "llm") and hasattr(config.llm, "max_tokens"):
                table.add_row("Max tokens", str(config.llm.max_tokens))
            self.console.print(Panel(table, title="[bold #D4A017]⚙ Config[/bold #D4A017]", title_align="left", border_style="#333333", padding=(0, 1)))
        except Exception as exc:
            self.console.print(f"[yellow]Config error: {exc}[/yellow]")

    async def _show_sessions_list(self) -> None:
        """List saved sessions with clean formatting."""
        from rune.ui.sessions import list_sessions
        items = await list_sessions()
        if not items:
            self.console.print("[dim]No saved sessions.[/dim]")
            return
        from rich.panel import Panel
        from rich.table import Table
        table = Table(show_header=True, show_edge=False, padding=(0, 1), expand=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("ID", style="bold #00CED1", width=10)
        table.add_column("Name", style="#E0E0E0")
        table.add_column("Msgs", style="dim", width=6, justify="right")
        for i, item in enumerate(items, 1):
            name_display = item.name or "(unnamed)"
            if item.preview:
                name_display += f"  [dim]{item.preview[:40]}[/dim]"
            table.add_row(str(i), item.id, name_display, str(item.message_count))
        self.console.print(Panel(table, title="[bold #D4A017]💾 Sessions[/bold #D4A017]", title_align="left", border_style="#333333", padding=(0, 1)))
        self.console.print("[dim]  Use /load <id> to restore a session.[/dim]")

    async def _interactive_load_session(self) -> None:
        """Interactive session loader with ↑↓ arrow keys."""
        from rune.ui.inline_select import inline_select
        from rune.ui.sessions import list_sessions

        sessions = await list_sessions()
        if not sessions:
            self.console.print("[dim]No saved sessions.[/dim]")
            return

        items = [
            (s.id, f"{s.id}  {s.name or '(unnamed)'}  ({s.message_count} msgs)")
            for s in sessions
        ]

        idx = inline_select(
            items,

            title="Load session",
        )
        if idx is not None:
            await self._do_load_session(items[idx][0])

    async def _interactive_export(self) -> None:
        """Interactive export format selection with ↑↓ arrow keys."""
        from rune.ui.inline_select import inline_select

        formats = _EXPORT_FORMATS
        items = [(f, f) for f in formats]

        idx = inline_select(
            items,

            title="Export format",
        )
        if idx is not None:
            await self._do_export(items[idx][0])

    async def _interactive_theme_select(self) -> None:
        """Interactive theme selection with ↑↓ arrow keys."""
        from rune.ui.inline_select import inline_select
        from rune.ui.theme import THEMES, current_theme_name

        current = current_theme_name()
        theme_names = sorted(THEMES)
        items = [(name, name) for name in theme_names]
        default_idx = theme_names.index(current) if current in theme_names else 0

        idx = inline_select(
            items,

            title="Select theme",
            default_index=default_idx,
        )
        if idx is not None:
            self._do_set_theme(items[idx][0])

    def _do_set_theme(self, theme_name: str) -> None:
        """Switch UI theme."""
        from rune.ui.theme import THEMES, set_theme
        if theme_name not in THEMES:
            self.console.print(f"[yellow]Unknown theme '{theme_name}'. Available: {', '.join(sorted(THEMES))}[/yellow]")
            return
        set_theme(theme_name)
        self.console.print(f"[#D4A017]Theme set to: {theme_name}[/#D4A017]")

    async def _do_memory_operation(self, sub: str) -> None:
        """Handle /memory show|add|clear."""
        try:
            from rune.memory.project_memory import (
                find_project_memory_file,
                read_project_memory_head,
            )
        except ImportError:
            self.console.print("[yellow]Memory module not available.[/yellow]")
            return
        workspace = Path.cwd()
        if sub == "show":
            content = read_project_memory_head(workspace)
            if content:
                self.console.print(f"\n[bold]Project Memory:[/bold]\n\n{content}\n")
            else:
                self.console.print("[dim]No project memory found.[/dim]")
        elif sub == "add":
            try:
                text = await self._session.prompt_async(
                    [("class:prompt", "memory> ")],
                    multiline=True,
                )
                text = text.strip()
                if not text:
                    self.console.print("[dim]Cancelled.[/dim]")
                    return
                mem_file = find_project_memory_file(workspace)
                if mem_file is None:
                    mem_file = workspace / ".rune" / "memory.md"
                    mem_file.parent.mkdir(parents=True, exist_ok=True)
                existing = mem_file.read_text(encoding="utf-8") if mem_file.exists() else ""
                mem_file.write_text(
                    existing.rstrip() + "\n\n" + text + "\n" if existing.strip() else text + "\n",
                    encoding="utf-8",
                )
                self.console.print("[dim]Added to project memory.[/dim]")
            except (EOFError, KeyboardInterrupt):
                self.console.print("[dim]Cancelled.[/dim]")
        elif sub == "clear":
            mem_file = find_project_memory_file(workspace)
            if mem_file and mem_file.exists():
                try:
                    mem_file.write_text("", encoding="utf-8")
                    self.console.print("[dim]Project memory cleared.[/dim]")
                except OSError as exc:
                    self.console.print(f"[red]Failed to clear memory: {exc}[/red]")
            else:
                self.console.print("[dim]No project memory file found.[/dim]")

    # ===================================================================
    # Agent integration hooks (called from cli.py)
    # ===================================================================

    def set_agent_handler(self, handler: Any) -> None:
        """Set the async handler for user messages."""
        self._on_user_message = handler

    def set_agent_controller(self, controller: Any) -> None:
        """Set the agent loop controller."""
        self._agent_controller = controller

    def _request_shutdown(self, reason: str | None = None) -> None:
        """Mark the TUI as shutting down so benign teardown noise can be suppressed."""
        self._shutting_down = True
        if reason and self._shutdown_reason is None:
            self._shutdown_reason = reason

    def _requires_sigint_exit_grace(self) -> bool:
        """Return True when exiting via Ctrl+C and we should delay process teardown."""
        return self._shutdown_reason == "sigint"

    def _install_main_sigint_handler(self, loop: asyncio.AbstractEventLoop) -> None:
        """Install a SIGINT handler for the idle TUI prompt."""
        if self._main_sigint_handler_installed:
            return

        def _sigint_handler(signum: int, frame: Any) -> None:
            if self._agent_running:
                return
            loop.call_soon_threadsafe(self._on_main_sigint)

        signal.signal(signal.SIGINT, _sigint_handler)
        self._main_sigint_handler_installed = True

    def _on_main_sigint(self) -> None:
        """Handle Ctrl+C while the idle prompt is active."""
        if self._agent_running:
            return

        now = time.monotonic()
        if now - self._last_abort_time < 2.0:
            self._request_shutdown("sigint")
            self._interrupt_prompt(EOFError())
            return

        self._last_abort_time = now
        self.console.print("[dim]Press Ctrl+C again to exit.[/dim]")
        self._interrupt_prompt(KeyboardInterrupt())

    def _interrupt_prompt(self, exc: BaseException) -> None:
        """Exit the active prompt_toolkit application with *exc* if it is running."""
        app = getattr(self._session, "app", None)
        if app is None:
            return
        try:
            if getattr(app, "is_running", False):
                try:
                    app.exit(exception=exc)
                except TypeError:
                    app.exit()
        except Exception:
            pass

    def _install_loop_exception_handler(self, loop: asyncio.AbstractEventLoop) -> None:
        """Install a loop exception handler that suppresses known TUI shutdown noise."""
        self._loop_exception_handler = loop.get_exception_handler()

        def _handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
            if self._should_suppress_loop_exception(context):
                exc = context.get("exception")
                log.debug(
                    "suppressed_tui_shutdown_exception",
                    message=context.get("message", ""),
                    error=str(exc or "")[:200],
                )
                return
            if self._loop_exception_handler is not None:
                self._loop_exception_handler(loop, context)
            else:
                loop.default_exception_handler(context)

        loop.set_exception_handler(_handler)

    def _restore_loop_exception_handler(self, loop: asyncio.AbstractEventLoop) -> None:
        """Restore the loop exception handler after the TUI exits."""
        loop.set_exception_handler(self._loop_exception_handler)
        self._loop_exception_handler = None

    def _should_suppress_loop_exception(self, context: dict[str, Any]) -> bool:
        """Return True for known benign exceptions raised during TUI shutdown."""
        message = str(context.get("message", ""))
        exc = context.get("exception")
        text = " ".join(
            part for part in (message, str(exc or ""), str(context.get("future", ""))) if part
        )

        if "Connection closed while reading from the driver" in text:
            return True

        if not self._shutting_down:
            return False

        if "Failed to get PID of child process" in text:
            return True
        if "ESRCH: No such process" in text:
            return True
        if isinstance(exc, ProcessLookupError):
            return True
        return bool(isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ESRCH)

    def _is_benign_shutdown_exception(self, exc: BaseException) -> bool:
        """Classify prompt/driver shutdown errors that should not reach the user."""
        if isinstance(exc, (EOFError, KeyboardInterrupt)):
            return True
        text = str(exc)
        if "Connection closed while reading from the driver" in text:
            return True
        if self._shutting_down and (
            "Failed to get PID of child process" in text
            or "ESRCH: No such process" in text
        ):
            return True
        if self._shutting_down and isinstance(exc, ProcessLookupError):
            return True
        if self._shutting_down and isinstance(exc, OSError):
            return getattr(exc, "errno", None) == errno.ESRCH
        return False

    def _close_prompt_session(self) -> None:
        """Best-effort prompt_toolkit app shutdown to avoid teardown noise."""
        app = getattr(self._session, "app", None)
        if app is None:
            return
        try:
            if getattr(app, "is_running", False):
                try:
                    app.exit(exception=EOFError())
                except TypeError:
                    app.exit()
        except Exception:
            pass

    def set_file_tracker(self, tracker: Any) -> None:
        """Set the file tracker for undo support."""
        self._file_tracker = tracker

    # -- Methods called by AgentLoopController --------------------------------

    def update_status(
        self,
        *,
        status: str | None = None,
        tokens_used: int | None = None,
        token_budget: int | None = None,
        current_step: int | None = None,
        total_steps: int | None = None,
        status_text: str | None = None,
    ) -> None:
        """Update status (delegates to renderer for live display)."""
        if status is not None:
            self._agent_running = status not in ("idle", "")
        self.renderer.update_status(
            status_text=status_text,
            current_step=current_step,
            tokens_used=tokens_used,
            token_budget=token_budget,
        )

    def update_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Accumulate token usage for cost estimation."""
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

    def add_message(
        self,
        role: str,
        content: str,
        *,
        use_markdown: bool = False,
        tool_category: str = "default",
    ) -> None:
        """Add a message to scrollback (callable from agent loop)."""
        if role == "user":
            self.renderer.print_user_message(content)
        elif role == "assistant":
            if use_markdown:
                self.renderer.print_assistant_response(content)
                self._last_response_text = content
            else:
                self.renderer.print_system_message(content)
        elif role == "system":
            self.renderer.print_completion_summary(content)
        elif role == "thinking":
            self.renderer.print_thinking(content)
        elif role == "tool":
            self.renderer.print_tool_call(content, tool_category)
        else:
            self.console.print(f"[dim]{content}[/dim]")

    def update_phase(self, phase: str, **kwargs: Any) -> None:
        """Update agent phase (no-op in terminal mode, logged only)."""
        pass

    def show_pending_input(self, mode: str, **kwargs: Any) -> None:
        """Show pending input banner (prints to scrollback)."""
        title = kwargs.get("title", "")
        headline = kwargs.get("headline", "")
        if headline:
            self.console.print(f"[bold #D4A017]{title}:[/bold #D4A017] {headline}")

    def hide_pending_input(self) -> None:
        """Hide pending input banner (no-op in terminal mode)."""
        pass

    def push_toast(self, message: str, *, toast_type: str = "info", timeout: float = 3.0) -> None:
        """Show toast notification (prints to scrollback)."""
        if toast_type == "error":
            self.console.print(f"[red]{message}[/red]")
        elif toast_type == "warning":
            self.console.print(f"[yellow]{message}[/yellow]")
        else:
            self.console.print(f"[dim]{message}[/dim]")

    @staticmethod
    def _ring_bell() -> None:
        """Ring terminal bell on agent completion."""
        sys.stdout.write("\x07")
        sys.stdout.flush()


# Spinner interval constant
_SPINNER_INTERVAL = 0.25
