// RUNE desktop shell: window + daemon supervision only.
//
// Security invariant: all privileged capability (files, shell, browser, MCP)
// lives in the Python daemon behind Guardian. This process must expose no
// fs/shell surface to the renderer, so a renderer compromise cannot pivot to
// the OS. Threat model: docs/design/desktop-app.md §10.

'use strict';

const { app, BrowserWindow, shell, session } = require('electron');
const { spawn } = require('node:child_process');
const http = require('node:http');
const path = require('node:path');

const PORT = Number(process.env.RUNE_API_PORT || 18789);
const ORIGIN = `http://127.0.0.1:${PORT}`;
// In dev, point at the Vite dev server (RUNE_UI_URL=http://localhost:5173) and
// run the daemon yourself; in prod the shell spawns the daemon and loads ORIGIN.
const DEV_URL = process.env.RUNE_UI_URL || null;
const UI_URL = DEV_URL || ORIGIN;

let mainWindow = null;

/** Poll `url` until it responds, or reject after `timeoutMs`. */
function waitForBackend(url, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const tick = () => {
      const req = http.get(url, (res) => {
        res.resume();
        resolve(true);
      });
      req.on('error', () => {
        if (Date.now() > deadline) reject(new Error('backend did not come up'));
        else setTimeout(tick, 400);
      });
      req.setTimeout(2000, () => req.destroy());
    };
    tick();
  });
}

/**
 * `rune daemon start` is idempotent and detached: it attaches to a running
 * engine, and the engine outlives this app. In dev (RUNE_UI_URL set) the
 * user runs the engine themselves.
 */
function ensureDaemon() {
  if (DEV_URL) return Promise.resolve();
  return new Promise((resolve) => {
    const p = spawn('rune', ['daemon', 'start', '--port', String(PORT)], {
      stdio: 'inherit',
      env: { ...process.env },
    });
    p.on('error', (err) => {
      console.error('[rune-desktop] failed to start daemon:', err.message);
      resolve();
    });
    p.on('exit', () => resolve());
  });
}

/** The daemon stays up on quit unless RUNE_STOP_DAEMON_ON_QUIT=1. */
function maybeStopDaemon() {
  if (DEV_URL) return;
  if (process.env.RUNE_STOP_DAEMON_ON_QUIT !== '1') return;
  spawn('rune', ['daemon', 'stop'], { stdio: 'inherit' });
}

function hardenSession() {
  session.defaultSession.setPermissionRequestHandler((_wc, _perm, cb) => cb(false));

  // Content-Security-Policy for our first-party, localhost-only UI.
  // NOTE: the React app uses inline style attributes, hence 'unsafe-inline' for
  // styles only. Scripts stay 'self'. The daemon should also send this header;
  // this is defense-in-depth. Tune if the app needs more.
  session.defaultSession.webRequest.onHeadersReceived((details, cb) => {
    cb({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [
          "default-src 'self'; " +
            `connect-src 'self' ${ORIGIN} ws://127.0.0.1:${PORT}; ` +
            "img-src 'self' data:; " +
            "style-src 'self' 'unsafe-inline'; " +
            "script-src 'self'; " +
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none';",
        ],
      },
    });
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 820,
    minWidth: 720,
    minHeight: 480,
    backgroundColor: '#0E1116', // basalt, so no white flash
    title: 'RUNE',
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webviewTag: false,
      nodeIntegrationInWorker: false,
      nodeIntegrationInSubFrames: false,
      spellcheck: false,
    },
  });

  // Block navigation away from our trusted origin; open external links in the OS browser.
  const trusted = new URL(UI_URL).origin;
  mainWindow.webContents.on('will-navigate', (e, url) => {
    if (new URL(url).origin !== trusted) e.preventDefault();
  });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/.test(url)) shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.loadURL(UI_URL);

  mainWindow.webContents.on('did-finish-load', () => {
    console.log(`[rune-desktop] window loaded ${mainWindow.webContents.getURL()}`);
  });

  // RUNE_DESKTOP_SMOKE=1: confirm the UI loads, then exit (headless/CI check).
  if (process.env.RUNE_DESKTOP_SMOKE === '1') {
    mainWindow.webContents.on('did-finish-load', () => {
      console.log(`[smoke] loaded ${mainWindow.webContents.getURL()} · title="${mainWindow.getTitle()}"`);
      setTimeout(() => app.exit(0), 400);
    });
    mainWindow.webContents.on('did-fail-load', (_e, code, desc, url) => {
      console.error(`[smoke] FAILED ${code} ${desc} ${url}`);
      app.exit(1);
    });
    setTimeout(() => { console.error('[smoke] timeout'); app.exit(2); }, 20000);
  }

  mainWindow.on('closed', () => { mainWindow = null; });
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(async () => {
    hardenSession();
    await ensureDaemon();
    try {
      await waitForBackend(UI_URL);
    } catch (err) {
      console.error('[rune-desktop]', err.message);
    }
    createWindow();

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  });

  app.on('window-all-closed', () => {
    maybeStopDaemon();
    if (process.platform !== 'darwin') app.quit();
  });

  app.on('before-quit', maybeStopDaemon);
}
