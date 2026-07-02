# RUNE desktop shell (Electron)

The web UI **is** the app. This shell wraps it — it does not reimplement it.

```
Electron main (this dir)          Python daemon (`rune web`)         Renderer
──────────────────────            ──────────────────────────        ────────────────
spawn / supervise daemon   ─────▶ serves API + web/dist on   ◀────  hardened window
open hardened window              127.0.0.1:18789                    loads that UI
(no fs/shell surface)             (all privilege, Guardian-gated)     (sandboxed)
```

A renderer compromise can't reach the OS through Electron — the Node side is
deliberately minimal and the dangerous capability all lives in the daemon.
See `docs/design/desktop-app.md` §10 for the full design & threat model.

## Run it (dev)

Two panes — the shell loads the live Vite dev server, you run the engine:

```bash
# 1. deps (one-time; downloads Electron)
cd desktop && npm install

# 2. web dev server (new terminal)
cd web && npm run dev            # http://localhost:5173

# 3. RUNE engine (new terminal) — the /api the UI talks to
rune web --no-open               # 127.0.0.1:18789

# 4. the desktop window, pointed at the dev server
cd desktop && npm run dev        # RUNE_UI_URL=http://localhost:5173
```

## Run it (prod-like)

The shell brings the daemon up itself (`rune daemon start`, idempotent + detached)
and loads the daemon-served UI:

```bash
cd web && npm run build          # produce web/dist (served by the daemon)
cd desktop && npm install
npm start                        # runs `rune daemon start`, loads :18789
```

`rune` must be on PATH (the installed CLI). `RUNE_API_PORT` overrides the port.
The daemon stays up when the app quits (local-service model); set
`RUNE_STOP_DAEMON_ON_QUIT=1` to stop it with the window.

## Security baseline (enforced in `main.js`)

- `contextIsolation: true`, `sandbox: true`, `nodeIntegration: false`, `webviewTag: false`
- no `@electron/remote`; preload exposes only inert metadata (`window.rune`)
- navigation off the trusted origin is blocked; external links open in the OS browser
- `window.open` denied; permission requests denied by default
- CSP header injected for the localhost UI (defense-in-depth)

## Not done yet (later phases)

- **Fuses** (`@electron/fuses` at package time): RunAsNode off, ASAR integrity on, etc.
- **Packaging**: `electron-builder` installers + code-sign / notarize (mac) / Authenticode (win).
- **Self-contained daemon**: PyInstaller-bundle so users need no Python/`uv`.
- **Close-to-tray + a tray menu**: today the daemon already stays alive when the
  window closes (headless); a tray icon to reopen/quit is still to do.

Done: **detached `rune daemon start/stop/status`** (in the `rune` CLI) — the shell
uses it for idempotent attach-or-spawn.
