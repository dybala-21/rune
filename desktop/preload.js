// Exposes only inert metadata — never ipcRenderer or Node primitives. The
// renderer must gain no power beyond a browser tab's; anything privileged
// goes through the daemon's localhost API (docs/design/desktop-app.md §10.3).

'use strict';

const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('rune', {
  desktop: true,
  platform: process.platform,
  versions: {
    electron: process.versions.electron,
    chrome: process.versions.chrome,
  },
});
