#!/usr/bin/env sh
set -eu

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
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

if ! command -v uv >/dev/null 2>&1; then
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
fi

uv tool install --force --python "$UV_PYTHON" /rune-src
rune --version
