"""Keep model settings stable for the lifetime of a run."""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

_settings = ContextVar("llm_settings", default=None)
_environment = ContextVar("llm_environment", default=None)


def llm_settings():
    from rune.config import get_config

    return _settings.get() or get_config().llm


def model_environment(name: str, default: str = "") -> str:
    environment = _environment.get()
    return (environment if environment is not None else os.environ).get(name, default)


@contextmanager
def model_scope():
    if _settings.get() is not None:
        yield
        return
    settings = _settings.set(deepcopy(llm_settings()))
    environment = _environment.set(dict(os.environ))
    try:
        yield
    finally:
        _environment.reset(environment)
        _settings.reset(settings)
