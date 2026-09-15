"""Keep the installed app and its signing identity intact during rebuilds."""

import json
import plistlib
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from rune.computer.build import build_bundle


@pytest.fixture
def builder(tmp_path, monkeypatch):
    bundle = tmp_path / "Rune Computer.app"
    executable = bundle / "Contents/MacOS/RuneComputer"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"installed app")
    (bundle / "Contents/Info.plist").write_bytes(plistlib.dumps({"CFBundleIdentifier": "dev.rune.computer"}))
    monkeypatch.setattr("rune.computer.build._build_inputs", lambda *_: "source-v1")
    monkeypatch.setattr("rune.computer.installation.installation_info", lambda *_: {"signing": "ad_hoc"})

    def run(args, **kwargs):
        if args[:2] == ["xcrun", "swiftc"]:
            Path(args[args.index("-o") + 1]).write_bytes(b"rebuilt app")
        return subprocess.CompletedProcess(args, 0)

    process = Mock(side_effect=run)
    monkeypatch.setattr("rune.computer.build.subprocess.run", process)
    return bundle, executable, process


def test_unchanged_build_keeps_signed_executable_and_forced_rebuild_preserves_identity(builder):
    bundle, executable, process = builder
    assert build_bundle(bundle, identity="Test certificate")
    stamp = executable.stat().st_mtime_ns
    process.reset_mock()
    assert not build_bundle(bundle)
    assert executable.stat().st_mtime_ns == stamp
    assert not any(call.args[0][0] == "xcrun" or "--sign" in call.args[0] for call in process.call_args_list)
    assert build_bundle(bundle, force=True)
    sign = next(call.args[0] for call in process.call_args_list if "--sign" in call.args[0])
    assert sign[sign.index("--sign") + 1] == "Test certificate"


def test_ad_hoc_update_requires_explicit_identity_replacement(builder):
    bundle, executable, process = builder
    with pytest.raises(ValueError, match="replace-adhoc"):
        build_bundle(bundle)
    assert executable.read_bytes() == b"installed app"
    assert not any(call.args[0][0] == "xcrun" for call in process.call_args_list)
    assert build_bundle(bundle, replace_adhoc=True)
    assert not build_bundle(bundle)


@pytest.mark.parametrize("failure", ["compile", "sign", "verify"])
def test_failed_build_leaves_installed_app_untouched(builder, failure):
    bundle, executable, process = builder
    original = process.side_effect

    def fail(args, **kwargs):
        if ((failure == "compile" and args[0] == "xcrun")
                or (failure == "sign" and "--sign" in args)
                or (failure == "verify" and "--verify" in args)):
            raise subprocess.CalledProcessError(1, args)
        return original(args, **kwargs)

    process.side_effect = fail
    with pytest.raises(subprocess.CalledProcessError):
        build_bundle(bundle, replace_adhoc=True)
    assert executable.read_bytes() == b"installed app"


def test_certificate_is_not_silently_replaced_by_ad_hoc_signature(builder, monkeypatch):
    bundle, executable, process = builder
    (bundle / "Contents/Resources").mkdir()
    (bundle / "Contents/Resources/build.json").write_text(json.dumps({"signingIdentity": "-"}))
    monkeypatch.setattr("rune.computer.installation.installation_info", lambda *_: {"signing": "certificate"})
    with pytest.raises(ValueError, match="will not be downgraded"):
        build_bundle(bundle, replace_adhoc=True)
    assert executable.read_bytes() == b"installed app"
    process.assert_not_called()


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_install_failure_restores_or_preserves_previous_app(builder, monkeypatch, rollback_fails):
    bundle, executable, _ = builder
    rename = Path.rename

    def fail_install(path, target):
        if target == bundle and (path.parent.name.startswith(".rune-build-")
                                 or rollback_fails and ".previous-" in path.name):
            raise OSError("install interrupted")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_install)
    with pytest.raises(RuntimeError if rollback_fails else OSError):
        build_bundle(bundle, replace_adhoc=True)
    if rollback_fails:
        backup, = bundle.parent.glob("*.previous-*")
        assert (backup / "Contents/MacOS/RuneComputer").read_bytes() == b"installed app"
    else:
        assert executable.read_bytes() == b"installed app"
