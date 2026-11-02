"""Hosted workers send browser traffic through the forced egress proxy; local runs are unchanged."""

import pytest

from rune.cloud.boundary import browser_proxy


@pytest.fixture
def clean_proxy_env(monkeypatch):
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)


def test_local_runs_never_get_a_proxy(clean_proxy_env, monkeypatch):
    monkeypatch.delenv("RUNE_CLOUD_WORKER", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://10.207.0.1:3128")
    assert browser_proxy() is None


def test_hosted_without_proxy_env(clean_proxy_env, monkeypatch):
    monkeypatch.setenv("RUNE_CLOUD_WORKER", "1")
    assert browser_proxy() is None


def test_hosted_proxy_with_credentials_and_bypass(clean_proxy_env, monkeypatch):
    monkeypatch.setenv("RUNE_CLOUD_WORKER", "1")
    monkeypatch.setenv("https_proxy", "http://vm%2D7:p%40ss@10.207.0.1:3128")
    monkeypatch.setenv("NO_PROXY", "10.207.0.1,localhost")
    assert browser_proxy() == {"server": "http://10.207.0.1:3128", "bypass": "10.207.0.1,localhost",
                               "username": "vm-7", "password": "p@ss"}


def test_hosted_proxy_without_scheme_and_ipv6(clean_proxy_env, monkeypatch):
    monkeypatch.setenv("RUNE_CLOUD_WORKER", "1")
    monkeypatch.setenv("HTTP_PROXY", "proxy.internal:8080")
    assert browser_proxy() == {"server": "http://proxy.internal:8080"}
    monkeypatch.setenv("HTTP_PROXY", "http://[fd00::1]:3128")
    assert browser_proxy() == {"server": "http://[fd00::1]:3128"}


def test_hosted_rejects_unusable_proxy(clean_proxy_env, monkeypatch):
    monkeypatch.setenv("RUNE_CLOUD_WORKER", "1")
    monkeypatch.setenv("HTTPS_PROXY", "ftp://10.0.0.1")
    with pytest.raises(ValueError):
        browser_proxy()


def test_hosted_skill_key_lives_under_rune_home(monkeypatch, tmp_path):
    from rune.skills.key_manager import SKILL_SIGNING_KEY_FILE_ENV, _resolve_key_file_path

    monkeypatch.delenv(SKILL_SIGNING_KEY_FILE_ENV, raising=False)
    monkeypatch.setenv("RUNE_HOME", str(tmp_path))
    monkeypatch.setenv("RUNE_CLOUD_WORKER", "1")
    assert _resolve_key_file_path().startswith(str(tmp_path))
    monkeypatch.delenv("RUNE_CLOUD_WORKER")
    assert not _resolve_key_file_path().startswith(str(tmp_path))


def test_socks5_with_credentials_is_refused_with_the_reason(clean_proxy_env, monkeypatch):
    monkeypatch.setenv("RUNE_CLOUD_WORKER", "1")
    monkeypatch.setenv("HTTPS_PROXY", "socks5://user:pw@10.207.0.1:1080")
    with pytest.raises(ValueError, match="SOCKS5"):
        browser_proxy()
    monkeypatch.setenv("HTTPS_PROXY", "socks5://10.207.0.1:1080")
    assert browser_proxy() == {"server": "socks5://10.207.0.1:1080"}
