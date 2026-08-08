"""Environment-runner prefixes must not hide the command behind them."""

import pytest

from rune.agent.bash_parsing import is_verification_command, strip_runner_prefix


@pytest.mark.parametrize("command", [
    # The measured failure: RUNE's own project runs its tests this way, and the
    # completion guard saw no passing test, so it refused to finish green work.
    "uv run pytest --tb=short",
    "cd /tmp/project && uv run pytest",
    "poetry run pytest -q",
    "pdm run pytest",
    "pipenv run pytest tests/",
    "hatch run pytest",
    "rye run pytest",
    "uv run python -m pytest",
    "npx jest",
    "bunx vitest run",
    "pnpm exec vitest",
    "pnpm run test",
    "yarn run build",
])
def test_wrapped_test_commands_are_recognized(command):
    assert is_verification_command(command) is True


@pytest.mark.parametrize("command", [
    "uv run ls",
    "poetry run python manage.py runserver",
    "npx serve",
    "uv run",          # nothing after the wrapper
    "ls -la",
    "cd /tmp && echo hi",
])
def test_wrappers_do_not_make_everything_a_test(command):
    assert is_verification_command(command) is False


def test_unwrapped_commands_still_work():
    assert is_verification_command("pytest -q") is True
    assert is_verification_command("npm test") is True
    assert is_verification_command("cargo test") is True


def test_strip_runner_prefix_behavior():
    assert strip_runner_prefix(("uv", "run", "pytest")) == ("pytest",)
    assert strip_runner_prefix(("npx", "jest", "--ci")) == ("jest", "--ci")
    # Nested wrappers collapse.
    assert strip_runner_prefix(("uv", "run", "npx", "jest")) == ("jest",)
    # A bare wrapper keeps its tokens rather than becoming empty.
    assert strip_runner_prefix(("uv", "run")) == ("uv", "run")
    assert strip_runner_prefix(()) == ()
    # Non-wrappers are untouched.
    assert strip_runner_prefix(("pytest", "-q")) == ("pytest", "-q")
