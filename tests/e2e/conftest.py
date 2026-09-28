"""Explicitly enabled tests against the configured model providers."""

from pathlib import Path

import pytest

from tests.e2e.live_config import configure_model, select_model


def pytest_collection_modifyitems(config, items):
    live = [item for item in items if Path(__file__).parent in item.path.parents]
    if not live:
        return
    if not config.getoption("--run-live"):
        for item in live:
            item.add_marker(pytest.mark.skip(reason="Use --run-live to enable model API calls"))
        return
    from rune.config import get_config

    try:
        selection = select_model(get_config(), config.getoption("--live-provider"), config.getoption("--live-model"))
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc
    config._rune_live_selection = selection


@pytest.fixture(autouse=True)
def live_model(request, monkeypatch):
    from rune.config import loader

    selection = request.config._rune_live_selection
    cfg = loader.get_config().model_copy(deep=True)
    configure_model(cfg, *selection)
    backend = request.config.getoption("--decision-backend")
    if backend:
        cfg.llm.decision_routing.backend = backend
    monkeypatch.setattr(loader, "_config", cfg)
    return selection
