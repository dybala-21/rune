// The daemon owns execution and input leases; the shell hosts browser tabs.
// Remote pages have no preload, filesystem bridge, or shell access.

'use strict';

const { app, BrowserWindow, dialog, shell, session, ipcMain, powerMonitor } = require('electron');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const { waitForBackend } = require('./backend');
const { readEndpoint } = require('./endpoint');
const os = require('node:os');
const path = require('node:path');

const ICON_PATH = path.join(__dirname, 'build', 'icon.png');
// Use the app name in the menu bar and dock during development too.
app.setName('RUNE');

let PORT = 18789;
let ORIGIN = `http://127.0.0.1:${PORT}`;
// RUNE_UI_URL uses a dev server and leaves daemon startup to the developer.
const DEV_URL = process.env.RUNE_UI_URL || null;
let UI_URL = DEV_URL || ORIGIN;

let mainWindow = null;
let browserHost = null;
let openingWindow = null;

ipcMain.handle('rune:retry-backend', async event => {
  if (!mainWindow || event.sender !== mainWindow.webContents
      || event.senderFrame !== mainWindow.webContents.mainFrame) throw new Error('Unknown app window');
  try {
    await ensureDaemon();
    await waitForBackend(ORIGIN);
    startBrowserHost();
    await mainWindow.loadURL(UI_URL);
    return { connected: true };
  } catch (error) { return { error: error.message }; }
});

ipcMain.handle('rune:browser-layout', (event, payload) => {
  if (!mainWindow || event.sender !== mainWindow.webContents
      || event.senderFrame !== mainWindow.webContents.mainFrame) throw new Error('Unknown browser surface');
  if (!browserHost) return false;
  if (!payload) { browserHost.hide(); return true; }
  const { sessionId, bounds } = payload;
  const [width, height] = mainWindow.getContentSize();
  if (typeof sessionId !== 'string' || !bounds || !['x', 'y', 'width', 'height'].every(k => Number.isFinite(bounds[k]))) throw new Error('Invalid browser layout');
  const zoom = mainWindow.webContents.getZoomFactor();
  const x = Math.max(0, Math.round(bounds.x * zoom)), y = Math.max(0, Math.round(bounds.y * zoom));
  if (x >= width || y >= height || bounds.width <= 0 || bounds.height <= 0) { browserHost.hide(); return false; }
  const box = { x, y, width: Math.max(1, Math.min(width - x, Math.round(bounds.width * zoom))),
    height: Math.max(1, Math.min(height - y, Math.round(bounds.height * zoom))) };
  return browserHost.mount(sessionId, box);
});
ipcMain.on('rune:browser-takeover', event => {
  if (browserHost && event.sender === browserHost.shield.webContents) void browserHost.takeover();
});

// Check common install paths too; Finder launches with a minimal PATH.
function resolveRuneBinary() {
  const override = process.env.RUNE_BIN;
  if (override && fs.existsSync(override)) return override;

  const exe = process.platform === 'win32' ? 'rune.exe' : 'rune';
  for (const dir of (process.env.PATH || '').split(path.delimiter)) {
    if (dir && fs.existsSync(path.join(dir, exe))) return path.join(dir, exe);
  }

  const home = os.homedir();
  const candidates = [
    path.join(home, '.local', 'bin', exe), // uv tool / pipx default
    '/opt/homebrew/bin/' + exe,
    '/usr/local/bin/' + exe,
    path.join(home, '.cargo', 'bin', exe),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return null;
}

// Reuse a ready daemon or start one that outlives the window.
function ensureDaemon() {
  let endpoint;
  try {
    endpoint = readEndpoint();
    PORT = endpoint.port;
    ORIGIN = endpoint.origin;
    UI_URL = DEV_URL || ORIGIN;
  } catch (error) { return Promise.reject(error); }
  if (DEV_URL) return Promise.resolve();
  const bin = resolveRuneBinary();
  if (!bin) {
    console.error('[rune-desktop] rune CLI not found (RUNE_BIN unset, not on PATH)');
    if (process.env.RUNE_DESKTOP_SMOKE === '1') return Promise.resolve();
    dialog.showErrorBox(
      'RUNE engine not found',
      'The rune CLI is not installed (or not in a known location).\n\n' +
        'Install it first:\n  uv tool install rune-agent\n\n' +
        'Or point the app at your binary:\n  RUNE_BIN=/path/to/rune\n\n' +
        'The window will still open and connect if an engine is already ' +
        `running on port ${PORT}.`,
    );
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const p = spawn(bin, ['daemon', 'start', '--host', endpoint.host, '--port', String(PORT)], {
      stdio: 'inherit',
      env: { ...process.env },
    });
    p.on('error', (err) => {
      console.error('[rune-desktop] failed to start daemon:', err.message);
      reject(err);
    });
    p.on('exit', code => code === 0 ? resolve() : reject(new Error('Rune could not start. Check the daemon address and logs.')));
  });
}

/** The daemon stays up on quit unless RUNE_STOP_DAEMON_ON_QUIT=1. */
function maybeStopDaemon() {
  if (DEV_URL) return;
  if (process.env.RUNE_STOP_DAEMON_ON_QUIT !== '1') return;
  const bin = resolveRuneBinary();
  if (bin) spawn(bin, ['daemon', 'stop'], { stdio: 'inherit' });
}

function hardenSession() {
  session.defaultSession.setPermissionRequestHandler((_wc, _perm, cb) => cb(false));

  // React needs inline styles; scripts remain restricted to the UI origin.
  session.defaultSession.webRequest.onHeadersReceived((details, cb) => {
    cb({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [
          "default-src 'self'; " +
            `connect-src 'self' ${ORIGIN} ws://127.0.0.1:${PORT}; ` +
            // blob: is needed for attachment thumbnails (URL.createObjectURL).
            "img-src 'self' data: blob:; " +
            "style-src 'self' 'unsafe-inline'; " +
            "script-src 'self'; " +
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none';",
        ],
      },
    });
  });
}

function createWindow(connectionError = null) {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 820,
    minWidth: 720,
    minHeight: 480,
    backgroundColor: '#0E1116', // basalt, so no white flash
    title: 'RUNE',
    icon: ICON_PATH,
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

  if (connectionError) mainWindow.loadFile(path.join(__dirname, 'connection.html'));
  else mainWindow.loadURL(UI_URL);
  startBrowserHost();

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

  mainWindow.on('close', () => { browserHost?.close(); browserHost = null; });
  mainWindow.on('closed', () => { mainWindow = null; });

  const stopWatching = watchForReload();
  mainWindow.once('closed', () => stopWatching?.());
}

function startBrowserHost() {
  if (browserHost?.origin === ORIGIN) return;
  browserHost?.close();
  const { BrowserHost } = require('./browser-host');
  browserHost = new BrowserHost(mainWindow, ORIGIN);
  browserHost.start().catch(error => console.error('[browser-host] startup:', error.message));
}

// Reload after local web builds; packaged assets have no watchable dist directory.
function watchForReload() {
  const distDir = path.join(__dirname, '..', 'web', 'dist');
  if (!fs.existsSync(distDir)) return;
  let timer = null;
  try {
    const watcher = fs.watch(distDir, { recursive: true }, () => {
      if (timer) clearTimeout(timer);
      // Wait for the build to finish writing before reloading.
      timer = setTimeout(() => {
        if (mainWindow && !mainWindow.isDestroyed()) {
          console.log('[rune-desktop] web build changed — reloading');
          mainWindow.webContents.reloadIgnoringCache();
        }
      }, 400);
    });
    return () => { watcher.close(); if (timer) clearTimeout(timer); };
  } catch (err) {
    console.error('[rune-desktop] dist watch unavailable:', err.message);
  }
}

async function openWindow() {
  if (openingWindow) return openingWindow;
  openingWindow = (async () => {
    let connectionError = null;
    try {
      await ensureDaemon();
      await waitForBackend(ORIGIN);
    } catch (error) {
      connectionError = error;
      console.error('[rune-desktop]', error.message);
    }
    createWindow(connectionError);
  })();
  try { await openingWindow; }
  finally { openingWindow = null; }
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
    // Dev run: set the dock icon explicitly (packaging bakes it from build/).
    if (process.platform === 'darwin' && app.dock) {
      try { app.dock.setIcon(ICON_PATH); } catch { /* non-fatal */ }
    }
    hardenSession();
    await openWindow();
    powerMonitor.on('resume', () => {
      if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send('rune:resume');
    });

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) void openWindow();
    });
  });

  app.on('window-all-closed', () => {
    maybeStopDaemon();
    if (process.platform !== 'darwin') app.quit();
  });

  app.on('before-quit', maybeStopDaemon);
}
