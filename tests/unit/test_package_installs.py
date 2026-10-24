"""Installing software outside the workspace needs approval; project-local installs do not."""

import pytest

from rune.safety.guardian import Guardian
from rune.safety.package_installs import install_scope

ROOT = "/w/proj"


@pytest.mark.parametrize("command,outside", [
    ("python3 -m pip install --user pytest", True),
    ("pip install pytest", True),                                   # PATH decides the interpreter
    ("/usr/bin/python3 -m pip install x", True),
    ("pip install --target /tmp/x y", True),
    ("uv pip install --system x", True),
    ("npm install -g typescript", True),
    ("yarn global add x", True),
    ("sudo apt-get install -y jq", True),
    ("brew install jq", True),
    ("cargo install ripgrep", True),
    ("go install golang.org/x/tools/gopls@latest", True),
    ("pipx install black", True),
    (". /other/venv/bin/activate && pip install x", True),
    (".venv/bin/pip install pytest", False),
    (".venv/bin/python -m pip install -e .", False),
    ("source .venv/bin/activate && pip install pytest", False),
    ("pip install --target ./vendor y", False),
    ("uv pip install pytest", False),
    ("uv add requests", False),
    ("poetry add httpx", False),
    ("npm install", False),
    ("npm i -D vitest", False),
])
def test_install_scope(command, outside):
    found = install_scope(command, ROOT)
    assert found is not None and found.outside_workspace is outside


@pytest.mark.parametrize("command", ["pip list", "npm test", "python3 -m pytest -q", 'git commit -m "pip install x"', "ls"])
def test_commands_that_install_nothing(command):
    assert install_scope(command, ROOT) is None


def test_guardian_asks_before_installing_outside_the_workspace():
    guard = Guardian()
    asked = guard.validate("python3 -m pip install --user pytest", cwd=ROOT)
    assert asked.allowed and asked.requires_approval and "user site-packages" in asked.reason
    local = guard.validate(".venv/bin/pip install pytest", cwd=ROOT)
    assert local.allowed and not local.requires_approval


def test_strict_mode_asks_before_any_install(monkeypatch):
    monkeypatch.setenv("RUNE_APPROVAL_MODE", "strict")
    result = Guardian().validate("npm install", cwd=ROOT)
    assert result.requires_approval and "Strict approval" in result.reason
