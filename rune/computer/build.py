"""Build the macOS helper without requesting screen or input permissions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import plistlib
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from rune.utils.logger import get_logger

log = get_logger(__name__)


def _build_inputs(source: Path, identity: str) -> str:
    digest = hashlib.sha256()
    for path in [Path(__file__), *sorted(source.glob("*.swift"))]:
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    toolchain = subprocess.run(["xcrun", "swiftc", "--version"], check=True, capture_output=True).stdout
    digest.update(toolchain + platform.machine().encode() + identity.encode())
    return digest.hexdigest()


def _saved_build(bundle: Path) -> dict:
    try:
        value = json.loads((bundle / "Contents/Resources/build.json").read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError) as exc:
        log.debug("native_build_manifest_unavailable", error=str(exc))
        return {}


def build_bundle(bundle: Path, *, identity: str | None = None, force: bool = False, replace_adhoc: bool = False) -> bool:
    """Return False when the verified installed build already matches the inputs."""
    import fcntl

    if bundle.is_symlink() or bundle.suffix != ".app":
        raise ValueError("Choose a real .app bundle directory")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with (bundle.parent / f".{bundle.name}.build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _build_bundle(bundle, identity=identity, force=force, replace_adhoc=replace_adhoc)


def _build_bundle(bundle: Path, *, identity: str | None, force: bool, replace_adhoc: bool) -> bool:
    if bundle.is_symlink():
        raise ValueError("Choose a real .app bundle directory")
    if bundle.exists():
        info = plistlib.loads((bundle / "Contents/Info.plist").read_bytes())
        if info.get("CFBundleIdentifier") != "dev.rune.computer":
            raise ValueError("Refusing to replace an app other than Rune Computer")
    saved = _saved_build(bundle)
    if identity is None:
        identity = saved.get("signingIdentity") or "-"
        if bundle.exists() and identity == "-":
            from rune.computer.installation import installation_info

            if installation_info(bundle / "Contents/MacOS/RuneComputer")["signing"] != "ad_hoc":
                raise ValueError("Supply --sign with the existing app's signing identity; its signature will not be downgraded")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("A nonempty code signing identity is required")
    source = Path(__file__).with_name("native")
    inputs = _build_inputs(source, identity)
    if not force and saved.get("inputs") == inputs:
        verified = subprocess.run(["codesign", "--verify", "--strict", str(bundle)], capture_output=True)
        if verified.returncode == 0:
            return False

    if bundle.exists() and identity == "-" and not replace_adhoc:
        raise ValueError(
            "An ad-hoc rebuild can invalidate this app's Accessibility and Screen Recording approvals. "
            "Use --sign with a stable signing identity, or --replace-adhoc and re-register the updated app in System Settings."
        )

    # A compiler or signing failure must leave the working app intact.
    with tempfile.TemporaryDirectory(prefix=".rune-build-", dir=bundle.parent) as staging_dir:
        staging = Path(staging_dir)
        candidate = staging / bundle.name
        contents = candidate / "Contents"
        executable = contents / "MacOS/RuneComputer"
        executable.parent.mkdir(parents=True)
        subprocess.run(["xcrun", "swiftc", "-parse-as-library", "-O", "-target",
                        f"{platform.machine()}-apple-macosx14.0",
                        "-module-cache-path", str(bundle.parent / ".swift-cache"),
                        *(str(path) for path in sorted(source.glob("*.swift"))),
                        "-o", str(executable)], check=True)
        (contents / "Info.plist").write_bytes(plistlib.dumps({
            "CFBundleIdentifier": "dev.rune.computer", "CFBundleName": "Rune Computer",
            "CFBundleDisplayName": "Rune Computer", "CFBundleExecutable": "RuneComputer",
            "CFBundlePackageType": "APPL", "CFBundleVersion": "1", "LSMinimumSystemVersion": "14.0",
            "LSUIElement": True,
            "NSScreenCaptureUsageDescription": "Show selected app windows to the model used by your Rune conversation.",
        }))
        (contents / "Resources").mkdir()
        (contents / "Resources/build.json").write_text(json.dumps({"inputs": inputs, "signingIdentity": identity}))
        subprocess.run(["codesign", "--force", "--sign", identity, str(candidate)], check=True)
        subprocess.run(["codesign", "--verify", "--strict", str(candidate)], check=True)
        previous = bundle.with_name(f".{bundle.name}.previous-{uuid.uuid4().hex}")
        if bundle.exists():
            bundle.rename(previous)
        try:
            candidate.rename(bundle)
        except BaseException:
            if previous.exists():
                try:
                    previous.rename(bundle)
                except OSError as exc:
                    raise RuntimeError(f"Could not restore Rune Computer; the previous app is preserved at {previous}") from exc
            raise
        if previous.exists():
            try:
                shutil.rmtree(previous)
            except OSError as exc:
                log.warning("native_build_backup_cleanup_failed", path=str(previous), error=str(exc))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="App bundle output directory")
    parser.add_argument("--sign", default=os.environ.get("RUNE_COMPUTER_SIGN_IDENTITY"),
                        help="Code signing identity; preserves the previous identity when omitted")
    parser.add_argument("--force", action="store_true", help="Rebuild even when the inputs are unchanged")
    parser.add_argument("--replace-adhoc", action="store_true",
                        help="Replace an ad-hoc build even though its macOS permission identity may change")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("The native host requires macOS 14 or later.")
    from rune.computer.macos import host_path

    bundle = args.output or host_path().parents[2]
    changed = build_bundle(bundle, identity=args.sign, force=args.force, replace_adhoc=args.replace_adhoc)
    print(f"{'Built' if changed else 'Already current:'} {bundle}.")
    if changed and _saved_build(bundle).get("signingIdentity") == "-":
        print("This local build uses an ad-hoc signature. After code changes, macOS may require you to remove "
              "the previous Rune Computer entry and add this app again. Use --sign for a code signing identity.")


if __name__ == "__main__":
    main()
