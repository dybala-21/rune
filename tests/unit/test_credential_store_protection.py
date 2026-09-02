"""RUNE's own credential store was readable by any bash command.

``~/.rune/.env`` holds live provider keys. Every equivalent file — ~/.ssh,
~/.aws, ~/.netrc — is read-blocked; this one was not, and neither were the
``.env.bak.<timestamp>`` copies beside it.

Substring matching is what protects ~/.ssh, and it works there only because
the directory name survives every rewrite of the path. ~/.rune must stay
readable (config.yaml lives there), so these paths are resolved instead —
which is what the bypass cases below check.
"""

from __future__ import annotations

import pytest

from rune.safety.guardian import get_guardian


@pytest.fixture
def guardian():
    return get_guardian()


@pytest.fixture
def store():
    """Where the credential store actually lives.

    Hardcoding ~/.rune couples these to the default location; RUNE_HOME can
    point elsewhere, and then ~/.rune is correctly *not* the store.
    """
    from rune.utils.paths import rune_home

    return rune_home()


@pytest.mark.parametrize(
    "template",
    [
        "cat {env}",
        "head -1 {env}",
        "grep OPENAI {env}",
        "rg sk- {env}",
        "awk '{{print}}' {env}",
        "sed -n 1p {env}",
        "cp {env} /tmp/stolen",
        "cat {env} | base64",
    ],
)
def test_reading_the_credential_store_is_refused(guardian, store, template):
    command = template.format(env=store / ".env")
    assert guardian.validate(command).allowed is False, command


@pytest.mark.parametrize(
    "template",
    [
        "cat {home}/../{name}/.env",     # parent-then-back
        "cat {home}/./.env",             # dot segment
        "cd {home} && cat .env",         # relative after cd
        "cat {home}/.en\'\'v",             # quote splitting
        'cat "{home}/.env"',             # quoted
    ],
)
def test_spelling_the_path_differently_does_not_help(guardian, store, template):
    command = template.format(home=store, name=store.name)
    assert guardian.validate(command).allowed is False, command


def test_rotating_backup_names_are_covered(guardian, store):
    """Backups carry the same keys under a name no fixed list can hold."""
    for name in (".env.bak.1778997275", ".env.bak.1", ".env.old", ".env.local"):
        command = f"cat {store / name}"
        assert guardian.validate(command).allowed is False, name


@pytest.mark.parametrize(
    "command",
    [
        "cat ~/.rune/config.yaml",   # a sibling file must stay readable
        "cat ~/.rune/audit.jsonl",
        "ls ~/.rune",
        "cat .env.example",          # a template holds no secret
        "cat ./.env",                # a project's own env is the developer's
        "cd ~/project && cat .env",
        "cat README.md",
        "grep -r TODO .",
    ],
)
def test_ordinary_reads_are_untouched(guardian, command):
    assert guardian.validate(command).allowed is True, command


@pytest.mark.parametrize("name", [".env", ".env.bak.1778997275"])
def test_file_read_capability_is_blocked_too(guardian, store, name):
    """The bash gate and the file capability must agree."""
    assert guardian.validate_file_read_path(str(store / name)).allowed is False


@pytest.mark.parametrize("name", ["config.yaml", "audit.jsonl"])
def test_file_read_capability_allows_siblings(guardian, store, name):
    assert guardian.validate_file_read_path(str(store / name)).allowed is True


@pytest.mark.parametrize("path", ["./.env", "./.env.example"])
def test_a_project_env_is_still_the_developers(guardian, path):
    assert guardian.validate_file_read_path(path).allowed is True


def test_validation_stays_on_the_fast_path(guardian):
    """This runs on every shell command; CLAUDE.md wants hot paths sub-ms."""
    import time

    commands = ["ls -la", "git status", "npm run build", "grep -rn TODO ."]
    for command in commands:
        guardian.validate(command)

    start = time.perf_counter()
    for _ in range(200):
        for command in commands:
            guardian.validate(command)
    per_call_ms = (time.perf_counter() - start) / (200 * len(commands)) * 1000

    assert per_call_ms < 1.0, f"{per_call_ms:.3f} ms per validate()"
