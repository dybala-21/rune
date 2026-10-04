"""Explicitly enabled tests against the configured model providers."""

from pathlib import Path

import pytest

from tests.e2e.live_config import configure_model, select_model


def pytest_generate_tests(metafunc):
    if "live_trial" in metafunc.fixturenames:
        count = metafunc.config.getoption("--live-repeat")
        if not 1 <= count <= 10:
            raise pytest.UsageError("--live-repeat must be between 1 and 10")
        if metafunc.config.getoption("--decision-backend") == "paired":
            trials = [(index, backend) for index in range(count)
                      for backend in (("connected", "jev") if index % 2 == 0 else ("jev", "connected"))]
            metafunc.parametrize("live_trial", trials, indirect=True,
                                 ids=lambda trial: f"trial{trial[0] + 1}-{trial[1]}")
        else:
            metafunc.parametrize("live_trial", range(count), indirect=True, ids=lambda n: f"trial{n + 1}")


@pytest.fixture(autouse=True)
def live_trial(request):
    return request.param


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
def live_model(request, monkeypatch, live_trial):
    from rune.config import loader

    selection = request.config._rune_live_selection
    cfg = loader.get_config().model_copy(deep=True)
    configure_model(cfg, *selection)
    backend = request.config.getoption("--decision-backend")
    if backend == "paired":
        backend = live_trial[1]
    if backend:
        cfg.llm.decision_routing.backend = backend
    monkeypatch.setattr(loader, "_config", cfg)
    return selection
