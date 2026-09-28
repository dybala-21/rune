'use strict';

const { ipcRenderer } = require('electron');

window.addEventListener('DOMContentLoaded', () => {
  document.documentElement.style.cssText = 'background:transparent;cursor:pointer;height:100%;overflow:hidden';
  document.body.style.cssText = 'margin:0;background:transparent;height:100%';
  const pointer = document.createElement('div');
  pointer.style.cssText = 'position:fixed;pointer-events:none;display:none;z-index:1;filter:drop-shadow(0 2px 3px #0008)';
  pointer.innerHTML = '<svg width="28" height="32" viewBox="0 0 28 32"><path d="M3 2L3 25L9 19L14 29L19 27L14 17L24 17Z" fill="#b39aff" stroke="white" stroke-width="2"/></svg>';
  const ring = document.createElement('div');
  ring.style.cssText = 'position:fixed;pointer-events:none;width:28px;height:28px;border:2px solid #b39aff;border-radius:50%;display:none;transform:translate(-50%,-50%)';
  document.body.append(pointer, ring);
  let timer;
  ipcRenderer.on('rune:browser-pointer', (_event, point) => {
    pointer.style.display = 'block';
    pointer.style.left = `${point.x}px`; pointer.style.top = `${point.y}px`;
    if (point.click) {
      ring.style.left = `${point.x}px`; ring.style.top = `${point.y}px`; ring.style.display = 'block';
      clearTimeout(timer); timer = setTimeout(() => { ring.style.display = 'none'; }, 400);
    }
  });
  document.addEventListener('pointerdown', () => ipcRenderer.send('rune:browser-takeover'));
});
