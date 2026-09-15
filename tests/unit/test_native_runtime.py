"""Compile the native input reader and exercise its real event loop."""

import hashlib
import json
import select
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def native_runtime(tmp_path_factory):
    if sys.platform != "darwin" or not shutil.which("swiftc"):
        pytest.skip("Requires macOS and the Swift toolchain")
    root = tmp_path_factory.mktemp("native-runtime")
    source = Path(__file__).parents[2] / "rune/computer/native"
    harness = root / "Harness.swift"
    harness.write_text('''
import AppKit
import Foundation

@main struct Harness {
    @MainActor static func main() {
        if CommandLine.arguments.dropFirst().first == "text" {
            let text = String(repeating: "한", count: 500) + "문서의 끝"
            var fields = ObservedText.fields("value", text.decomposedStringWithCanonicalMapping)
            fields["missing"] = ObservedText.fields("value", nil)
            fields["empty"] = ObservedText.fields("value", "")
            fields["differentTail"] = ObservedText.fingerprint(String(repeating: "한", count: 500) + "다른 끝")
            FileHandle.standardOutput.write(try! JSONSerialization.data(withJSONObject: fields))
            return
        }
        if CommandLine.arguments.count > 1 {
            let base = CGRect(x: -1200, y: 30, width: 800, height: 600)
            precondition(WindowGeometry.matches(base, base.offsetBy(dx: 0.5, dy: -0.5)))
            precondition(!WindowGeometry.matches(base, base.offsetBy(dx: 2, dy: 0)))
            precondition(!WindowGeometry.matches(base, CGRect(x: -1200, y: 30, width: 810, height: 600)))
            precondition(!WindowGeometry.matches(base, .null))
            precondition(!WindowGeometry.matches(base, .infinite))
            precondition(!WindowGeometry.matches(.zero, .zero))
            return
        }
        NSApplication.shared.setActivationPolicy(.prohibited)
        Timer.scheduledTimer(withTimeInterval: 0.1, repeats: false) { _ in
            FileHandle.standardOutput.write(Data("responsive\\n".utf8))
        }
        Task { @MainActor in
            do {
                try await CommandInput.consume(.standardInput) { data in
                    FileHandle.standardOutput.write(data + Data([10]))
                }
                exit(0)
            } catch {
                FileHandle.standardError.write(Data("invalid command\\n".utf8))
                exit(2)
            }
        }
        NSApplication.shared.run()
    }
}
''')
    executable = root / "harness"
    built = subprocess.run(["swiftc", "-parse-as-library", "-module-cache-path", str(root / "cache"),
        str(source / "CommandInput.swift"), str(source / "WindowGeometry.swift"), str(source / "ObservedText.swift"), str(harness),
        "-o", str(executable)], capture_output=True, text=True, timeout=120)
    assert built.returncode == 0, built.stderr
    return executable


def test_native_reader_keeps_appkit_responsive_and_preserves_fragmented_utf8(native_runtime):
    process = subprocess.Popen([native_runtime], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert select.select([process.stdout], [], [], 5)[0], "AppKit stalled while awaiting a command"
        assert process.stdout.readline() == b"responsive\n"
        command = '{"text":"한글 문서"}'.encode()
        process.stdin.write(command[:10])
        process.stdin.flush()
        output, error = process.communicate(command[10:] + b'\n{"second":true}\n', timeout=5)
        assert process.returncode == 0, error
        assert output == command + b'\n{"second":true}\n'
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


@pytest.mark.parametrize("data", [b"unfinished", b"x" * 32769 + b"\n"])
def test_native_reader_rejects_truncated_and_oversized_commands(native_runtime, data):
    result = subprocess.run([native_runtime], input=data, capture_output=True, timeout=5)
    assert result.returncode == 2 and b"invalid command" in result.stderr
    assert b"x" not in result.stdout and b"unfinished" not in result.stdout


def test_native_window_geometry_tolerates_rounding_but_rejects_movement(native_runtime):
    result = subprocess.run([native_runtime, "geometry"], capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr


def test_native_text_preserves_missing_values_and_checks_beyond_the_preview(native_runtime):
    from rune.computer.observation import match_condition
    from rune.computer.protocol import DesktopCondition

    result = subprocess.run([native_runtime, "text"], capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr
    row = json.loads(result.stdout)
    text = "한" * 500 + "문서의 끝"
    assert row["value"] == text[:500] and row["valueTruncated"] is True
    assert row["valueSHA256"] == hashlib.sha256(text.encode()).hexdigest()
    assert row["differentTail"] != row["valueSHA256"]
    assert row["missing"] == {} and row["empty"] == {"value": ""}
    row["role"] = "AXTextArea"
    condition = DesktopCondition(kind="control", role="AXTextArea", text=text)
    assert match_condition(condition, {"controls": [row]})["matchedBy"] == "full_value_sha256"
    assert match_condition(condition.model_copy(update={"text": text[:500]}), {"controls": [row]}) is None
