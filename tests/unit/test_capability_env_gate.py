"""RUNE_DISABLED_CAPABILITIES removes matching capabilities at registry init."""

from __future__ import annotations

from rune.capabilities.registry import (
    CapabilityRegistry,
    _apply_disabled_capabilities,
    _register_all_capabilities,
)


def _fresh_registry() -> CapabilityRegistry:
    reg = CapabilityRegistry()
    _register_all_capabilities(reg)
    return reg


class TestDisabledCapabilitiesEnv:
    def test_unset_keeps_everything(self, monkeypatch):
        monkeypatch.delenv("RUNE_DISABLED_CAPABILITIES", raising=False)
        reg = _fresh_registry()
        before = set(reg.list_names())
        _apply_disabled_capabilities(reg)
        assert set(reg.list_names()) == before

    def test_prefix_pattern_removes_family(self, monkeypatch):
        monkeypatch.setenv("RUNE_DISABLED_CAPABILITIES", "browser_*,web_*")
        reg = _fresh_registry()
        assert any(n.startswith("browser_") for n in reg.list_names())
        _apply_disabled_capabilities(reg)
        left = reg.list_names()
        assert not any(n.startswith(("browser_", "web_")) for n in left)
        assert any(n.startswith("file") or "bash" in n for n in left)  # rest intact

    def test_exact_name_removes_only_that(self, monkeypatch):
        reg = _fresh_registry()
        target = next(n for n in reg.list_names() if n.startswith("web_"))
        monkeypatch.setenv("RUNE_DISABLED_CAPABILITIES", target)
        before = set(reg.list_names())
        _apply_disabled_capabilities(reg)
        assert set(reg.list_names()) == before - {target}
