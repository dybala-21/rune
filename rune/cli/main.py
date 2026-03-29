"""CLI entry point for RUNE (package form).

Phase 9 adds uvloop installation at the very top so every ``asyncio.run()``
call in the CLI benefits automatically.

Can be invoked via::

    python -m rune.cli.main
    rune --message "..."
    rune  (interactive REPL)
"""

from __future__ import annotations

import os
import sys
from typing import Annotated, Any

# Phase 9: Install uvloop before any asyncio usage.
from rune.utils.loop_setup import setup_event_loop

setup_event_loop()

import typer
from rich.console import Console

from rune import __version__

# App

app = typer.Typer(
    name="rune",
    help="RUNE — AI Development Environment",
    no_args_is_help=False,
    add_completion=False,
)

console = Console(stderr=True)

# Subcommand groups
env_app = typer.Typer(help="Environment variable management")
token_app = typer.Typer(help="API token management")
daemon_app = typer.Typer(help="Background daemon control")
db_app = typer.Typer(help="Database operations")

browser_app = typer.Typer(help="Browser relay and extension management")

import contextlib

from rune.cli.memory_cmd import memory_app
from rune.cli.self_cmd import self_app

app.add_typer(env_app, name="env")
app.add_typer(token_app, name="token")
app.add_typer(daemon_app, name="daemon")
app.add_typer(db_app, name="db")
app.add_typer(browser_app, name="browser")
app.add_typer(self_app, name="self")
app.add_typer(memory_app, name="memory")


# Global callback

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool, typer.Option("--version", "-v", help="Show version")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Enable verbose logging")
    ] = False,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Model to use")
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", "-p", help="LLM provider")
    ] = None,
    message: Annotated[
        str | None, typer.Option("--message", help="Non-interactive message")
    ] = None,
    voice: Annotated[
        bool, typer.Option("--voice", help="Voice input mode (speak instead of type)")
    ] = False,
) -> None:
    """RUNE - AI Development Environment.

    Run without arguments to start the interactive REPL.
    Use --message for non-interactive one-shot mode.
    """
    log_level = "debug" if verbose else os.environ.get("RUNE_LOG_LEVEL", "warning")
    from rune.utils.logger import configure_logging
    configure_logging(level=log_level)

    if version:
        from rune import __codename__
        console.print(f"rune v{__version__} ({__codename__})")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        if voice:
            _handle_voice_mode(model=model, provider=provider)
        elif message:
            _handle_non_interactive(message, model=model, provider=provider)
        else:
            _start_interactive(model=model, provider=provider)


# Voice mode handling

def _handle_voice_mode(
    model: str | None = None,
    provider: str | None = None,
) -> None:
    """Voice input mode — speak instead of type."""
    import asyncio

    from rune.agent.loop import NativeAgentLoop
    from rune.types import AgentConfig

    if not _ensure_llm_key():
        console.print("[red]No API key configured.[/red]")
        raise typer.Exit(1)

    async def _voice_loop() -> None:
        from rune.voice.service import get_voice_service

        voice_svc = get_voice_service()
        if not voice_svc.has_stt:
            console.print(
                "[red]No STT provider available.[/red]\n"
                "  Set DEEPGRAM_API_KEY for cloud STT, or install sherpa-onnx for local."
            )
            raise typer.Exit(1)

        agent_config = AgentConfig()
        if model or provider:
            from rune.config import get_config
            cfg = get_config()
            if model:
                agent_config.model = model
            if provider:
                agent_config.provider = provider

        loop = NativeAgentLoop(config=agent_config)

        stt_name = type(voice_svc._stt).__name__ if voice_svc._stt else "none"
        tts_name = type(voice_svc._tts).__name__ if voice_svc._tts else "text only"
        console.print(
            f"[bold green]🎤 Voice mode active.[/bold green]\n"
            f"  STT: {stt_name}  |  TTS: {tts_name}\n"
            f"  Speak to interact. Ctrl+C to exit.\n"
        )

        while True:
            try:
                console.print("[dim]🎤 Listening...[/dim]")
                text = await voice_svc.listen_and_transcribe()

                if not text:
                    console.print("[dim]  (no speech detected)[/dim]")
                    continue

                console.print(f"[bold]❯[/bold] {text}\n")

                # Build memory context
                run_context: dict = {}
                try:
                    from rune.memory.manager import get_memory_manager
                    mgr = get_memory_manager()
                    mem_ctx = await mgr.build_memory_context(text)
                    if mem_ctx:
                        run_context["memory_context"] = mem_ctx
                except Exception:
                    pass

                trace = await loop.run(text, context=run_context)

                # TTS output (if available)
                if voice_svc.has_tts and trace.reason == "completed":
                    output = getattr(trace, "final_output", "") or ""
                    if output:
                        await voice_svc.speak_and_play(output[:500])

            except KeyboardInterrupt:
                console.print("\n[dim]Voice mode ended.[/dim]")
                break

    asyncio.run(_voice_loop())


# Non-interactive message handling

def _handle_non_interactive(
    message: str,
    model: str | None = None,
    provider: str | None = None,
) -> None:
    """Handle a single non-interactive message."""
    import asyncio

    from rune.agent.loop import NativeAgentLoop
    from rune.types import AgentConfig

    if not _ensure_llm_key():
        console.print("[red]No API key configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.[/red]")
        raise typer.Exit(1)

    agent_config = AgentConfig()
    if model or provider:
        from rune.config import get_config
        cfg = get_config()
        if model:
            agent_config.model = model
            cfg.llm.active_model = model
        if provider:
            agent_config.provider = provider
            cfg.llm.active_provider = provider

    loop = NativeAgentLoop(config=agent_config)
    output_parts: list[str] = []

    # Wire a simple stdin approval callback for non-interactive
    _wire_cli_approval(loop)
    # ask_user left unset intentionally (agent proceeds autonomously)

    async def _on_text_delta(delta: str) -> None:
        output_parts.append(delta)
        print(delta, end="", flush=True)

    async def _on_tool_call(info: dict) -> None:
        from rune.ui.message_format import _extract_tool_target
        name = info.get("name", "?")
        target = _extract_tool_target(name, info.get("params", {}))
        label = f"{name} {target}" if target else name
        console.print(f"\n[dim]  -> {label}[/dim]", highlight=False)

    loop.on("text_delta", _on_text_delta)
    loop.on("tool_call", _on_tool_call)

    async def _run() -> None:
        from rune.agent.agent_context import (
            PostProcessInput,
            PrepareContextOptions,
            post_process_agent_result,
            prepare_agent_context,
        )

        # Initialize MCP servers (best-effort, non-blocking)
        try:
            from rune.mcp.config import load_mcp_config
            configs = load_mcp_config()
            if configs:
                from rune.mcp.bridge import initialize_mcp_bridge
                result = await initialize_mcp_bridge(configs=configs)
                if result.connected_servers > 0:
                    console.print(
                        f"[dim]MCP: {result.connected_servers} server(s), "
                        f"{result.registered_count} tools[/dim]"
                    )
        except Exception:
            pass  # MCP init is best-effort

        ctx = await prepare_agent_context(PrepareContextOptions(
            goal=message, channel="cli",
        ))

        # Build memory context for self-improving
        run_context: dict = {"workspace_root": ctx.workspace_root}
        try:
            from rune.memory.manager import get_memory_manager
            mgr = get_memory_manager()
            mem_ctx = await mgr.build_memory_context(message)
            if mem_ctx:
                run_context["memory_context"] = mem_ctx
        except Exception:
            pass

        trace = await loop.run(ctx.goal, context=run_context)

        if output_parts:
            print()
        if trace.reason != "completed":
            console.print(f"[dim]({trace.reason})[/dim]")

        try:
            await post_process_agent_result(PostProcessInput(
                context=ctx,
                success=(trace.reason == "completed"),
                answer="".join(output_parts),
            ))
        except Exception:
            pass  # best-effort memory save

        # Cleanup MCP servers
        try:
            from rune.mcp.client import get_mcp_client_manager
            await get_mcp_client_manager().disconnect_all()
        except Exception:
            pass

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")


# Interactive REPL

def _start_interactive(
    model: str | None = None,
    provider: str | None = None,
) -> None:
    """Launch the interactive REPL."""
    # Banner is printed by RuneApp._print_welcome() - skip duplicate here.

    if not _ensure_llm_key():
        console.print("[red]No API key configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.[/red]")
        raise typer.Exit(1)

    # Rich + prompt_toolkit TUI with agent loop controller
    try:
        from rune.agent.loop import NativeAgentLoop
        from rune.types import AgentConfig
        from rune.ui.app import RuneApp
        from rune.ui.controllers.agent_loop_controller import AgentLoopController

        agent_config = AgentConfig()
        if model or provider:
            if model:
                agent_config.model = model
            if provider:
                agent_config.provider = provider
            agent_config._overridden = True

        agent_loop = NativeAgentLoop(config=agent_config)
        app_instance = RuneApp(model=model, provider=provider)

        # Single controller bridges ALL loop events to UI
        controller = AgentLoopController(app=app_instance, loop=agent_loop)
        app_instance.set_agent_controller(controller)

        # Start proactive + cron loop eagerly (don't wait for first user message)
        controller._ensure_proactive_started()

        # Wire approval callback: controller -> loop
        _wire_controller_approval(agent_loop, controller)

        # Wire ask_user callback: controller -> global
        _wire_controller_ask_user(agent_loop, controller)

        app_instance.run()
    except ImportError as exc:
        console.print(f"[dim]TUI import error: {exc}[/dim]")
        _simple_repl(model=model, provider=provider)


def _simple_repl(model: str | None = None, provider: str | None = None) -> None:
    """Simple REPL fallback when TUI is not available."""
    import asyncio

    from rune.agent.loop import NativeAgentLoop
    from rune.config import get_config
    from rune.types import AgentConfig

    config = get_config()
    display_model = model or config.llm.models.openai.best

    agent_config = AgentConfig()
    if model:
        agent_config.model = model

    # --- Conversation manager for multi-turn context ---
    conv_manager = None
    conversation_id = ""
    try:
        import os
        import tempfile

        from rune.conversation.manager import ConversationManager
        from rune.conversation.store import ConversationStore
        db_path = os.path.join(tempfile.gettempdir(), "rune_repl_conversations.db")
        conv_store = ConversationStore(db_path)
        conv_manager = ConversationManager(conv_store)
        conv = conv_manager.start_conversation(user_id="repl:local")
        conversation_id = conv.id
    except Exception:
        pass

    def _make_loop() -> NativeAgentLoop:
        lp = NativeAgentLoop(config=agent_config)
        lp.on("text_delta", _on_text_delta)
        lp.on("step", _on_step)
        lp.on("tool_call", _on_tool_call)
        _wire_cli_approval(lp)
        _wire_cli_ask_user(lp)
        return lp

    collected_text: list[str] = []

    async def _on_text_delta(delta: str) -> None:
        collected_text.append(delta)
        print(delta, end="", flush=True)

    async def _on_step(step: int) -> None:
        pass

    async def _on_tool_call(info: dict) -> None:
        from rune.ui.message_format import _extract_tool_target
        name = info.get("name", "?")
        target = _extract_tool_target(name, info.get("params", {}))
        label = f"{name} {target}" if target else name
        console.print(f"\n[dim]  -> {label}[/dim]", highlight=False)

    loop = _make_loop()

    async def _run_goal(goal: str) -> None:
        from rune.agent.agent_context import (
            PostProcessInput,
            PrepareContextOptions,
            post_process_agent_result,
            prepare_agent_context,
        )

        collected_text.clear()

        # Record user turn in conversation manager
        if conv_manager and conversation_id:
            with contextlib.suppress(Exception):
                conv_manager.add_turn(conversation_id, "user", goal)

        ctx = await prepare_agent_context(
            PrepareContextOptions(
                goal=goal,
                channel="tui",
                conversation_id=conversation_id,
            ),
            conversation_manager=conv_manager,
        )

        # Pass conversation messages as message_history to the agent loop
        trace = await loop.run(
            ctx.goal,
            context={"workspace_root": ctx.workspace_root},
            message_history=ctx.messages if ctx.messages else None,
        )

        answer = "".join(collected_text)

        # Record assistant turn in conversation manager
        if conv_manager and conversation_id and answer:
            with contextlib.suppress(Exception):
                conv_manager.add_turn(conversation_id, "assistant", answer)

        if collected_text:
            print()
        console.print(f"[dim]({trace.reason}, step {trace.final_step or '?'})[/dim]\n")

        with contextlib.suppress(Exception):
            await post_process_agent_result(PostProcessInput(
                context=ctx,
                success=(trace.reason == "completed"),
                answer=answer,
            ))

    console.print("[dim]Type your message. /exit to quit, /help for commands.[/dim]\n")

    while True:
        try:
            user_input = console.input(f"[bold cyan]rune ({display_model})>[/bold cyan] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input.strip():
            continue

        if user_input.strip() in ("/exit", "/quit", "/q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input.strip() == "/help":
            console.print("[bold]Commands:[/bold]")
            console.print("  /exit, /quit, /q  — Exit RUNE")
            console.print("  /help             — Show this help")
            console.print("  /model <name>     — Switch model")
            continue

        if user_input.strip().startswith("/model "):
            new_model = user_input.strip().split(None, 1)[1]
            display_model = new_model
            agent_config.model = new_model
            loop = _make_loop()
            console.print(f"[green]Switched to {new_model}[/green]\n")
            continue

        try:
            asyncio.run(_run_goal(user_input))
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]\n")
            loop = _make_loop()


# Env subcommands

@env_app.command("list")
def env_list() -> None:
    """List RUNE-related environment variables."""
    from rich.table import Table

    table = Table(title="RUNE Environment Variables")
    table.add_column("Variable", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Source", style="dim")

    rune_vars = {
        k: v for k, v in os.environ.items()
        if k.startswith("RUNE_") or k in (
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OLLAMA_HOST",
        )
    }

    for key, value in sorted(rune_vars.items()):
        display_value = f"{value[:8]}...{value[-4:]}" if "KEY" in key and len(value) > 12 else value
        table.add_row(key, display_value, "env")

    console.print(table)


@env_app.command("set")
def env_set(
    key: str = typer.Argument(help="Variable name"),
    value: str = typer.Argument(help="Variable value"),
) -> None:
    """Set a RUNE environment variable in ~/.rune/.env."""
    from rune.utils.paths import rune_home

    env_file = rune_home() / ".env"
    lines: list[str] = []

    if env_file.is_file():
        lines = env_file.read_text().splitlines()

    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            found = True
            break

    if not found:
        lines.append(f"{key}={value}")

    env_file.write_text("\n".join(lines) + "\n")
    os.chmod(str(env_file), 0o600)
    console.print(f"[green]Set {key} in {env_file}[/green]")


# Token subcommands

@token_app.command("check")
def token_check() -> None:
    """Check API token validity."""
    import asyncio

    from rune.llm.client import get_llm_client

    async def _check() -> None:
        client = get_llm_client()
        # Quick ping - ask for 1 token
        try:
            await client.completion(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                timeout=10.0,
            )
            console.print("[green]LLM connection OK[/green]")
        except Exception as exc:
            console.print(f"[red]LLM connection failed: {exc}[/red]")

    asyncio.run(_check())


# Web UI command

@app.command()
def web(
    host: Annotated[
        str, typer.Option("--host", "-H", help="Host to bind")
    ] = "127.0.0.1",
    port: Annotated[
        int, typer.Option("--port", "-P", help="Port to bind")
    ] = 18789,
    no_open: Annotated[
        bool, typer.Option("--no-open", help="Don't open browser automatically")
    ] = False,
) -> None:
    """Start the RUNE web UI.

    Launches the API server with the web frontend and opens the browser.
    """
    import asyncio
    from pathlib import Path

    # Locate the bundled web/dist directory.
    # 1. Check RUNE_WEB_STATIC_DIR env override
    env_dist = os.environ.get("RUNE_WEB_STATIC_DIR", "")
    # 2. Relative to package (for pipx / pip install)
    pkg_dist = Path(__file__).parent.parent.parent / "web" / "dist"
    # 3. CWD fallback for development
    cwd_dist = Path.cwd() / "web" / "dist"

    if env_dist:
        dist_dir = Path(env_dist)
    elif pkg_dist.is_dir():
        dist_dir = pkg_dist
    elif cwd_dist.is_dir():
        dist_dir = cwd_dist
    else:
        console.print(
            "[red]Web UI not found.[/red]\n"
            "[dim]Build it first: cd web && npm install && npm run build[/dim]\n"
            f"[dim]Searched: {pkg_dist}, {cwd_dist}[/dim]"
        )
        raise typer.Exit(1)

    index_html = dist_dir / "index.html"
    if not index_html.is_file():
        console.print(f"[red]index.html not found in {dist_dir}[/red]")
        raise typer.Exit(1)

    # Set env vars for the daemon / API server
    os.environ["RUNE_API_ENABLED"] = "true"
    os.environ["RUNE_WEB_STATIC_DIR"] = str(dist_dir.resolve())

    url = f"http://{host}:{port}"
    console.print("\n[bold cyan]RUNE Web UI[/bold cyan]")
    console.print(f"[dim]Static dir:[/dim] {dist_dir.resolve()}")
    console.print(f"[dim]Server:[/dim]     {url}")
    console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

    # Open browser (with short delay so server can start)
    if not no_open:
        import threading
        import webbrowser

        def _open_browser() -> None:
            import time as _time
            _time.sleep(1.0)
            webbrowser.open(url)

        threading.Thread(target=_open_browser, daemon=True).start()

    # Start daemon in foreground (runs API server as a subsystem)
    from rune.daemon.main import RuneDaemon, _default_config

    cfg = _default_config()
    cfg.update({
        "api_enabled": True,
        "api_host": host,
        "api_port": port,
    })
    daemon = RuneDaemon(config=cfg, install_signal_handlers=False)

    import signal

    def _force_exit() -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(0)

    async def _serve() -> None:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, _force_exit)
        loop.add_signal_handler(signal.SIGTERM, _force_exit)
        await daemon.serve_forever()

    try:
        asyncio.run(_serve())
    except (KeyboardInterrupt, SystemExit):
        _force_exit()


# DB subcommands

@db_app.command("status")
def db_status_cmd() -> None:
    """Show database status."""
    from rune.memory.store import get_memory_store
    store = get_memory_store()
    episodes = len(store.get_recent_episodes(limit=999_999))
    facts = len(store.search_facts("", limit=999_999))
    console.print("[green]Database OK[/green]")
    console.print(f"  Episodes: {episodes}")
    console.print(f"  Facts: {facts}")


# Callback wiring helpers


def _wire_cli_approval(loop: Any) -> None:
    """Wire a simple stdin-based approval callback into the loop."""

    async def _cli_approval(capability: str, reason: str) -> bool:
        console.print(f"\n[bold #D4A017]Approval required:[/bold #D4A017] {capability}")
        console.print(f"[dim]{reason}[/dim]")
        console.print("[dim](y)es | (n)o[/dim]")
        try:
            response = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return response in ("y", "yes")

    loop.set_approval_callback(_cli_approval)


def _wire_cli_ask_user(loop: Any) -> None:
    """Wire a simple stdin-based ask_user callback into the loop."""
    from rune.capabilities.ask_user import AskUserParams, UserResponse

    async def _cli_ask_user(params: AskUserParams) -> UserResponse:
        console.print(f"\n[bold #D4A017]Agent question:[/bold #D4A017] {params.question}")
        if params.options:
            for i, opt in enumerate(params.options, 1):
                desc = f" — {opt.description}" if opt.description else ""
                console.print(f"  {i}. {opt.label}{desc}")
        try:
            response = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            response = ""

        if params.options and response.isdigit():
            idx = int(response) - 1
            if 0 <= idx < len(params.options):
                return UserResponse(
                    selected_index=idx,
                    answer=params.options[idx].label,
                    raw_input=response,
                )
        return UserResponse(selected_index=-1, answer=response, raw_input=response, free_text=True)

    loop.set_ask_user_callback(_cli_ask_user)


def _wire_controller_approval(loop: Any, controller: Any) -> None:
    """Wire the TUI controller's approval prompt into the loop."""
    from rune.ui.approval_selection import ApprovalDecision

    async def _approval_bridge(capability: str, reason: str) -> bool:
        decision = await controller.request_approval(f"{capability}: {reason}")
        return decision in (ApprovalDecision.APPROVE_ONCE, ApprovalDecision.APPROVE_ALWAYS)

    loop.set_approval_callback(_approval_bridge)


def _wire_controller_ask_user(loop: Any, controller: Any) -> None:
    """Wire the TUI controller's question prompt into the loop as ask_user callback."""
    from rune.capabilities.ask_user import AskUserParams, UserResponse

    async def _ask_user_bridge(params: AskUserParams) -> UserResponse:
        options_list = [o.label for o in params.options] if params.options else None
        answer = await controller.request_question(
            params.question,
            options=options_list,
            urgency=params.urgency,
        )
        if params.options and answer.isdigit():
            idx = int(answer) - 1
            if 0 <= idx < len(params.options):
                return UserResponse(
                    selected_index=idx,
                    answer=params.options[idx].label,
                    raw_input=answer,
                )
        return UserResponse(selected_index=-1, answer=answer, raw_input=answer, free_text=True)

    loop.set_ask_user_callback(_ask_user_bridge)


# Helpers

def _print_banner() -> None:
    banner = rf"""
[bold cyan]
  ██████╗ ██╗   ██╗███╗   ██╗███████╗
  ██╔══██╗██║   ██║████╗  ██║██╔════╝
  ██████╔╝██║   ██║██╔██╗ ██║█████╗
  ██╔══██╗██║   ██║██║╚██╗██║██╔══╝
  ██║  ██║╚██████╔╝██║ ╚████║███████╗
  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
[/bold cyan]
[dim]AI Development Environment v{__version__}[/dim]
"""
    console.print(banner)


def _ensure_llm_key() -> bool:
    """Check if at least one API key is available."""
    from rune.config import get_config
    config = get_config()
    return bool(
        config.openai_api_key
        or config.anthropic_api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )


# Browser commands

@browser_app.command("connect")
def browser_connect(
    no_open: Annotated[
        bool, typer.Option("--no-open", help="Don't open chrome://extensions automatically")
    ] = False,
) -> None:
    """Set up the Chrome Extension and start the relay server."""
    import shutil
    import subprocess
    from pathlib import Path

    # 1. Locate extension directory
    ext_dir = Path(__file__).resolve().parent.parent.parent / "extension" / "rune-browser-bridge"
    if not (ext_dir / "manifest.json").exists():
        console.print(f"[red]Extension not found at {ext_dir}[/red]")
        raise typer.Exit(1)

    console.print("\n[bold]RUNE Browser Bridge[/bold] — Chrome Extension Setup\n")

    # 2. Copy path to clipboard
    ext_path_str = str(ext_dir)
    try:
        if sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=ext_path_str.encode(), check=True)
            console.print("[green]Extension path copied to clipboard[/green]")
        elif shutil.which("xclip"):
            subprocess.run(["xclip", "-selection", "clipboard"], input=ext_path_str.encode(), check=True)
            console.print("[green]Extension path copied to clipboard[/green]")
    except Exception:
        pass

    console.print(f"  Path: [cyan]{ext_path_str}[/cyan]\n")

    # 3. Instructions
    console.print("[bold]Installation steps:[/bold]")
    console.print("  1. Open Chrome → [cyan]chrome://extensions[/cyan]")
    console.print("  2. Enable [bold]Developer mode[/bold] (top right toggle)")
    console.print("  3. Click [bold]Load unpacked[/bold]")
    console.print("  4. Paste the path (already copied) and select the folder\n")

    # 4. Open chrome://extensions
    if not no_open:
        try:
            import webbrowser
            webbrowser.open("chrome://extensions")
        except Exception:
            pass

    # 5. Start relay server and wait for extension
    console.print("[bold]Starting relay server...[/bold]")

    import asyncio

    async def _run_relay() -> None:
        from rune.browser.relay_server import RelayServer
        server = RelayServer()
        await server.start()
        console.print(f"  Listening on port [cyan]{server.port}[/cyan]")
        console.print(f"  Extension endpoint: [cyan]{server.extension_endpoint}[/cyan]\n")
        console.print("[dim]Waiting for extension connection (up to 2 minutes)...[/dim]")

        connected = await server.wait_for_extension(timeout=120.0)
        if connected:
            console.print("\n[green bold]Extension connected![/green bold] RUNE can now control your browser.")
            console.print("[dim]Press Ctrl+C to stop the relay server.[/dim]\n")
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
        else:
            console.print("\n[yellow]Extension did not connect within 2 minutes.[/yellow]")
            console.print("Make sure the extension is installed and enabled in Chrome.")

        await server.stop()

    try:
        asyncio.run(_run_relay())
    except KeyboardInterrupt:
        console.print("\n[dim]Relay server stopped.[/dim]")


@browser_app.command("status")
def browser_status() -> None:
    """Check the relay server and extension connection status."""
    import httpx

    from rune.browser.relay_server import DISCOVERY_PORT_END, DISCOVERY_PORT_START

    found = False
    for port in range(DISCOVERY_PORT_START, DISCOVERY_PORT_END + 1):
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
            if resp.status_code == 200:
                data = resp.json()
                ext_status = (
                    "[green]Connected[/green]"
                    if data.get("extensionConnected")
                    else "[yellow]Not connected[/yellow]"
                )
                console.print(f"  Port [cyan]{port}[/cyan]: Relay running — Extension: {ext_status}")
                found = True
        except Exception:
            continue

    if not found:
        console.print("[dim]No relay server found on ports 19222-19231.[/dim]")
        console.print("Run [cyan]rune browser connect[/cyan] to start one.")


# Entry point

if __name__ == "__main__":
    app()
