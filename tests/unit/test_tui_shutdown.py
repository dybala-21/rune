from __future__ import annotations

import errno

from rune.ui.app import RuneApp


def _make_app(shutting_down: bool = False) -> RuneApp:
    app = RuneApp.__new__(RuneApp)
    app._shutting_down = shutting_down
    app._shutdown_reason = None
    app._loop_exception_handler = None
    app._SIGINT_EXIT_GRACE_S = 0.35
    app._agent_running = False
    app._last_abort_time = 0.0
    app._main_sigint_handler_installed = False
    return app


def test_suppresses_prompt_driver_close_during_shutdown() -> None:
    app = _make_app()
    context = {
        "message": "Future exception was never retrieved",
        "exception": Exception("Connection closed while reading from the driver"),
    }
    assert app._should_suppress_loop_exception(context) is True


def test_suppresses_esrch_only_while_shutting_down() -> None:
    shutting_down = _make_app(shutting_down=True)
    running = _make_app(shutting_down=False)
    exc = ProcessLookupError(errno.ESRCH, "No such process")
    context = {"message": "Future exception was never retrieved", "exception": exc}

    assert shutting_down._should_suppress_loop_exception(context) is True
    assert running._should_suppress_loop_exception(context) is False


def test_does_not_suppress_unrelated_exception() -> None:
    app = _make_app(shutting_down=True)
    context = {
        "message": "Future exception was never retrieved",
        "exception": RuntimeError("boom"),
    }
    assert app._should_suppress_loop_exception(context) is False


def test_close_prompt_session_swallow_errors() -> None:
    app = _make_app(shutting_down=True)

    class DummyPromptApp:
        is_running = True

        def exit(self, exception: BaseException | None = None) -> None:
            raise RuntimeError("already closed")

    class DummySession:
        app = DummyPromptApp()

    app._session = DummySession()
    app._close_prompt_session()


def test_sigint_exit_grace_only_for_sigint_shutdown() -> None:
    app = _make_app()
    assert app._requires_sigint_exit_grace() is False

    app._request_shutdown("sigint")
    assert app._requires_sigint_exit_grace() is True

    other = _make_app()
    other._request_shutdown("slash_exit")
    assert other._requires_sigint_exit_grace() is False


def test_main_sigint_first_press_interrupts_prompt(monkeypatch) -> None:
    app = _make_app()
    seen: list[BaseException] = []
    printed: list[str] = []

    class DummyConsole:
        def print(self, msg: str) -> None:
            printed.append(msg)

    app.console = DummyConsole()
    app._interrupt_prompt = lambda exc: seen.append(exc)

    monkeypatch.setattr("rune.ui.app.time.monotonic", lambda: 10.0)
    app._on_main_sigint()

    assert printed == ["[dim]Press Ctrl+C again to exit.[/dim]"]
    assert len(seen) == 1
    assert isinstance(seen[0], KeyboardInterrupt)
    assert app._shutdown_reason is None


def test_main_sigint_second_press_requests_shutdown(monkeypatch) -> None:
    app = _make_app()
    seen: list[BaseException] = []

    class DummyConsole:
        def print(self, msg: str) -> None:
            raise AssertionError("second Ctrl+C should not print notice")

    app.console = DummyConsole()
    app._interrupt_prompt = lambda exc: seen.append(exc)
    app._last_abort_time = 10.0

    monkeypatch.setattr("rune.ui.app.time.monotonic", lambda: 10.5)
    app._on_main_sigint()

    assert app._shutdown_reason == "sigint"
    assert len(seen) == 1
    assert isinstance(seen[0], EOFError)
