#!/bin/sh
# Launch RUNE as a desktop app: build the UI if needed, ensure the engine is up,
# then open the native window on it. `rune` must be on your PATH.
set -eu

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
port="${RUNE_API_PORT:-18789}"

# 1. Web UI (served by the daemon) — build once if missing.
if [ ! -f "$root/web/dist/index.html" ]; then
  echo "==> building web UI…"
  ( cd "$root/web" && npm install && npm run build )
fi

# 2. Electron deps — install once if missing.
if [ ! -x "$here/node_modules/.bin/electron" ]; then
  echo "==> installing Electron…"
  ( cd "$here" && npm install )
fi

# 3. Start the engine (idempotent + detached; stays up as a local service).
echo "==> starting RUNE engine on :$port …"
rune daemon start --port "$port"

# 4. Open the window pointed at the running engine.
echo "==> opening RUNE…"
cd "$here"
RUNE_UI_URL="http://127.0.0.1:$port" exec ./node_modules/.bin/electron .
