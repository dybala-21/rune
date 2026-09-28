const { app, BrowserWindow } = require('electron');
const { BrowserHost } = require('../../desktop/browser-host');

app.setPath('userData', process.env.RUNE_TEST_PROFILE);
app.whenReady().then(async () => {
  const window = new BrowserWindow({ show: false, webPreferences: { sandbox: true, contextIsolation: true } });
  await window.loadURL('about:blank');
  const host = new BrowserHost(window, process.env.RUNE_TEST_ORIGIN);
  await host.start();
  app.on('before-quit', () => host.close());
});
