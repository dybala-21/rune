// The UI can position its browser surface; page actions still belong to the daemon.

'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('rune', {
  desktop: true,
  platform: process.platform,
  versions: {
    electron: process.versions.electron,
    chrome: process.versions.chrome,
  },
  browserLayout: payload => ipcRenderer.invoke('rune:browser-layout', payload),
  retryConnection: () => ipcRenderer.invoke('rune:retry-backend'),
  onResume: callback => {
    const listener = () => callback();
    ipcRenderer.on('rune:resume', listener);
    return () => ipcRenderer.removeListener('rune:resume', listener);
  },
});
