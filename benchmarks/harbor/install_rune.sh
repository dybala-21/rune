#!/usr/bin/env sh
set -eu

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

RUNE_VENV="${RUNE_VENV:-/logs/agent/rune_venv}"
RUNE_INSTALL_FINGERPRINT="${RUNE_INSTALL_FINGERPRINT:-/logs/agent/rune_install_fingerprint.json}"
RUNE_WHEELHOUSE="${RUNE_HARBOR_WHEELHOUSE:-/rune-wheelhouse}"
RUNE_INSTALL_MODE="${RUNE_HARBOR_INSTALL_MODE:-auto}"

mkdir -p "$(dirname "$RUNE_VENV")" "$(dirname "$RUNE_INSTALL_FINGERPRINT")"

if [ -z "${UV_CACHE_DIR:-}" ] && [ -d /uv-cache ]; then
  export UV_CACHE_DIR=/uv-cache
fi
if [ -d /uv-cache ]; then
  export UV_INSTALL_DIR="${UV_INSTALL_DIR:-/uv-cache/bin}"
  mkdir -p "$UV_INSTALL_DIR"
  export PATH="$UV_INSTALL_DIR:$PATH"
fi
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_PYTHON="${UV_PYTHON:-3.13}"

install_fetcher() {
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    apt-get install -y curl ca-certificates
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache curl ca-certificates
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y curl ca-certificates
  elif command -v yum >/dev/null 2>&1; then
    yum install -y curl ca-certificates
  elif command -v microdnf >/dev/null 2>&1; then
    microdnf install -y curl ca-certificates
  fi
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    install_fetcher
  fi

  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  elif command -v python3 >/dev/null 2>&1; then
    python3 -m pip install --user uv
  else
    echo "Unable to install uv: no curl, wget, python3, or supported package manager found" >&2
    exit 127
  fi
}

wheelhouse_has_wheels() {
  [ -d "$RUNE_WHEELHOUSE" ] && find "$RUNE_WHEELHOUSE" -maxdepth 1 -name "*.whl" -type f | grep -q .
}

install_into_venv() {
  rm -rf "$RUNE_VENV"
  uv venv --python "$UV_PYTHON" "$RUNE_VENV"

  case "$1" in
    wheelhouse)
      uv pip install \
        --python "$RUNE_VENV/bin/python" \
        --no-index \
        --find-links "$RUNE_WHEELHOUSE" \
        rune-ai
      ;;
    source)
      uv pip install \
        --python "$RUNE_VENV/bin/python" \
        --reinstall-package rune-ai \
        --refresh-package rune-ai \
        /rune-src
      ;;
    pip)
      uv pip install --python "$RUNE_VENV/bin/python" rune-ai
      ;;
    *)
      echo "Unknown RUNE install mode: $1" >&2
      exit 2
      ;;
  esac
}

resolve_install_mode() {
  case "$RUNE_INSTALL_MODE" in
    auto)
      if wheelhouse_has_wheels; then
        echo "wheelhouse"
      elif [ -d /rune-src ]; then
        echo "source"
      else
        echo "pip"
      fi
      ;;
    wheelhouse)
      if ! wheelhouse_has_wheels; then
        echo "RUNE_HARBOR_INSTALL_MODE=wheelhouse requires wheels in $RUNE_WHEELHOUSE" >&2
        exit 2
      fi
      echo "wheelhouse"
      ;;
    source)
      if [ ! -d /rune-src ]; then
        echo "RUNE_HARBOR_INSTALL_MODE=source requires /rune-src" >&2
        exit 2
      fi
      echo "source"
      ;;
    pip)
      echo "pip"
      ;;
    *)
      echo "Unknown RUNE_HARBOR_INSTALL_MODE: $RUNE_INSTALL_MODE" >&2
      exit 2
      ;;
  esac
}

write_install_fingerprint() {
  "$RUNE_VENV/bin/python" - "$1" "$RUNE_WHEELHOUSE" "$RUNE_INSTALL_FINGERPRINT" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

install_mode = sys.argv[1]
wheelhouse = Path(sys.argv[2])
output = Path(sys.argv[3])


def run(command: list[str], cwd: str | None = None) -> dict[str, object]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str | None:
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for file_path in sorted(path.glob("*.whl")):
        digest.update(file_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(file_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def source_git_sha() -> str | None:
    env_value = os.environ.get("RUNE_BENCH_SOURCE_GIT_SHA")
    if env_value:
        return env_value
    result = run(["git", "rev-parse", "HEAD"], cwd="/rune-src")
    if result.get("ok"):
        return str(result.get("stdout") or "")
    return None


try:
    import rune
except Exception as exc:
    rune_file = None
    rune_import_error = str(exc)
else:
    rune_file = getattr(rune, "__file__", None)
    rune_import_error = None

payload = {
    "install_mode": install_mode,
    "rune_venv": os.environ.get("RUNE_VENV", "/logs/agent/rune_venv"),
    "python_executable": sys.executable,
    "rune_module_file": rune_file,
    "rune_import_error": rune_import_error,
    "rune_cli_version": run(["rune", "--version"]),
    "source_git_sha": source_git_sha(),
    "source_diff_sha256": os.environ.get("RUNE_BENCH_SOURCE_DIFF_SHA256"),
    "wheelhouse": str(wheelhouse) if wheelhouse.exists() else None,
    "wheelhouse_sha256": (
        os.environ.get("RUNE_BENCH_WHEELHOUSE_SHA256") or directory_sha256(wheelhouse)
    ),
}

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

ensure_uv
RESOLVED_INSTALL_MODE="$(resolve_install_mode)"
install_into_venv "$RESOLVED_INSTALL_MODE"
export PATH="$RUNE_VENV/bin:$PATH"
write_install_fingerprint "$RESOLVED_INSTALL_MODE"
rune --version
