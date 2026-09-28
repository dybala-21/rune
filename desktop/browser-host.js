'use strict';

const http = require('node:http');
const { randomBytes, randomUUID, timingSafeEqual } = require('node:crypto');
const { WebSocketServer, WebSocket } = require('ws');
const { WebContentsView, session, nativeImage } = require('electron');
const path = require('node:path');

function webURL(value) {
  const url = new URL(value);
  if (!['http:', 'https:', 'about:'].includes(url.protocol)
      || (url.protocol === 'about:' && value !== 'about:blank')) throw new Error('Use an HTTP or HTTPS address.');
  return url.href;
}

// Each connection exposes only the tabs belonging to one Rune conversation.
class BrowserHost {
  constructor(window, origin) {
    this.window = window;
    this.origin = origin;
    this.token = randomBytes(32).toString('hex');
    this.groups = new Map();
    this.visible = null;
    this.shield = new WebContentsView({ webPreferences: {
      preload: path.join(__dirname, 'browser-shield.js'), sandbox: true, contextIsolation: true, nodeIntegration: false,
    } });
    this.shield.setBackgroundColor('#00000000');
    this.shield.webContents.loadURL('about:blank');
    this.shield.setVisible(false);
    this.window.contentView.addChildView(this.shield);
    this.server = http.createServer((req, res) => this.handle(req, res));
    this.sockets = new WebSocketServer({ noServer: true, maxPayload: 16 * 1024 * 1024 });
    this.server.on('upgrade', (req, socket, head) => {
      try {
        const url = new URL(req.url, 'http://localhost');
        if (req.headers.origin || !this.authorized(req.headers.authorization?.replace(/^Bearer /, ''))) return socket.destroy();
        const group = this.groups.get(decodeURIComponent(url.pathname.slice('/cdp/'.length)));
        if (!url.pathname.startsWith('/cdp/') || !group || group.socket) return socket.destroy();
        this.sockets.handleUpgrade(req, socket, head, ws => this.connect(group, ws));
      } catch { socket.destroy(); }
    });
  }

  authorized(value) {
    return typeof value === 'string' && /^[a-f0-9]{64}$/.test(value)
      && timingSafeEqual(Buffer.from(value), Buffer.from(this.token));
  }

  async start() {
    await new Promise((resolve, reject) => { this.server.once('error', reject); this.server.listen(0, '127.0.0.1', resolve); });
    this.address = `http://127.0.0.1:${this.server.address().port}`;
    const register = async () => {
      if (this.closed) return;
      try {
        await this.api('/api/computer/native/host', { address: this.address, token: this.token });
      } catch (error) { console.warn('[browser-host] registration:', error.message); }
    };
    await register();
    this.heartbeat = setInterval(register, 5000);
  }

  async api(path, body) {
    const response = await fetch(this.origin + path, {
      method: body ? 'POST' : 'GET', headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined, signal: AbortSignal.timeout(10000),
    });
    if (!response.ok) throw new Error(`Rune returned ${response.status}`);
    return response.json();
  }

  async handle(req, res) {
    if (req.headers.origin || !this.authorized(req.headers.authorization?.replace(/^Bearer /, ''))) {
      res.writeHead(403).end(); return;
    }
    try {
      let body = '';
      for await (const chunk of req) {
        body += chunk;
        if (body.length > 65536) throw new Error('Request too large');
      }
      const { sessionId, operation, ...args } = JSON.parse(body);
      if (operation === 'health') { res.writeHead(200, { 'Content-Type': 'application/json' }).end('{"ok":true}'); return; }
      if (typeof sessionId !== 'string' || !sessionId || sessionId.length > 200) throw new Error('Invalid conversation');
      let group = this.groups.get(sessionId);
      if (operation === 'open' && !group) {
        if (this.groups.size >= 8) throw new Error('Close an idle browser first.');
        group = { id: sessionId, tabs: new Map(), selected: '', manual: false, bounds: null, socket: null, attached: false, browserSessions: new Set() };
        this.groups.set(sessionId, group);
        try { await this.createTab(group); }
        catch (error) { this.closeGroup(group); throw error; }
      }
      if (!group) throw new Error('The browser has closed.');
      if (operation === 'control') {
        group.manual = args.manual === true;
        this.layout(group);
        if (!group.manual && [...group.tabs.values()].some(tab => tab.view.webContents.isFocused())) this.shield.webContents.focus();
      } else if (operation === 'close') {
        this.closeGroup(group);
      } else if (operation === 'navigate') {
        if (!group.manual) throw new Error('Take control before changing tabs.');
        if (args.action === 'new_tab') await this.createTab(group, webURL(args.url || 'about:blank'));
        else if (args.action === 'select_tab' && group.tabs.has(args.tabId)) group.selected = args.tabId;
        else if (args.action === 'close_tab' && group.tabs.has(args.tabId)) {
          const tab = group.tabs.get(args.tabId);
          this.window.contentView.removeChildView(tab.view);
          tab.view.webContents.close(); group.tabs.delete(args.tabId);
          if (group.selected === args.tabId) group.selected = group.tabs.keys().next().value || '';
          if (!group.selected) await this.createTab(group);
        } else throw new Error('Unknown tab');
        this.layout(group);
      } else if (operation !== 'open' && operation !== 'status') throw new Error('Unknown operation');
      res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify({
        endpoint: this.address.replace('http:', 'ws:') + `/cdp/${encodeURIComponent(sessionId)}`,
        targetId: group.selected, tabs: this.tabInfo(group),
      }));
    } catch (error) {
      res.writeHead(400, { 'Content-Type': 'application/json' }).end(JSON.stringify({ error: error.message }));
    }
  }

  tabInfo(group) {
    return [...group.tabs.values()].map(tab => ({ id: tab.id, title: tab.view.webContents.getTitle(), url: tab.view.webContents.getURL() }));
  }

  makeView(group, options = {}) {
    if (group.tabs.size >= 16) throw new Error('Close a tab before opening another.');
    const partition = `rune-browser-${group.id}`;
    const isolated = session.fromPartition(partition);
    isolated.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
    isolated.setPermissionCheckHandler(() => false);
    if (!group.downloadListener) {
      group.partition = isolated;
      group.downloadListener = (event, item, contents) => {
        const policy = group.downloads;
        if (policy?.behavior === 'deny') { event.preventDefault(); return; }
        if (!policy || policy.behavior === 'default') return;
        const guid = randomUUID();
        item.setSavePath(path.join(policy.downloadPath, policy.behavior === 'allowAndName' ? guid : path.basename(item.getFilename())));
        const tab = [...group.tabs.values()].find(tab => tab.view.webContents === contents);
        if (policy.eventsEnabled && tab) {
          this.send(group, { method: 'Browser.downloadWillBegin', params: {
            frameId: tab.id, guid, url: item.getURL(), suggestedFilename: item.getFilename(),
          } });
          const progress = state => this.send(group, { method: 'Browser.downloadProgress', params: {
            guid, state, receivedBytes: item.getReceivedBytes(), totalBytes: item.getTotalBytes(),
          } });
          item.on('updated', () => progress('inProgress'));
          item.once('done', (_event, state) => progress(state === 'completed' ? 'completed' : 'canceled'));
        }
      };
      isolated.on('will-download', group.downloadListener);
    }
    const view = new WebContentsView({ ...(options.webContents ? { webContents: options.webContents } : {}), webPreferences: {
      ...options.webPreferences, partition, sandbox: true, contextIsolation: true,
      nodeIntegration: false, nodeIntegrationInSubFrames: false, webviewTag: false, preload: undefined,
      nodeIntegrationInWorker: false, webSecurity: true, allowRunningInsecureContent: false, backgroundThrottling: false,
    } });
    this.window.contentView.addChildView(view);
    view.setBounds({ x: 0, y: 0, width: 1280, height: 720 });
    view.setVisible(false);
    const tab = { id: '', view, sessions: new Set(), roots: new Map(), primary: null };
    view.webContents.on('will-navigate', (event, destination) => {
      try { webURL(destination); } catch { event.preventDefault(); }
    });
    view.webContents.on('will-redirect', (event, destination) => {
      try { webURL(destination); } catch { event.preventDefault(); }
    });
    view.webContents.setWindowOpenHandler(details => {
      try { webURL(details.url); } catch { return { action: 'deny' }; }
      if (group.tabs.size >= 16) return { action: 'deny' };
      return { action: 'allow', createWindow: options => {
        const popup = this.makeView(group, options);
        void this.registerTab(group, popup, details.disposition !== 'background-tab').then(() => {
          if (details.disposition === 'background-tab' && !options.webContents) return popup.view.webContents.loadURL(details.url);
        }).catch(error => {
          console.warn('[browser-host] popup:', error.message);
          popup.view.webContents.close();
        });
        return popup.view.webContents;
      } };
    });
    return tab;
  }

  async createTab(group, url = 'about:blank') {
    const destination = webURL(url);
    const tab = this.makeView(group);
    try {
      await tab.view.webContents.loadURL(destination);
      return await this.registerTab(group, tab);
    } catch (error) {
      this.window.contentView.removeChildView(tab.view);
      tab.view.webContents.close();
      throw error;
    }
  }

  async registerTab(group, tab, select = true) {
    const { view } = tab;
    if (!this.groups.has(group.id)) throw new Error('The conversation browser closed.');
    view.webContents.debugger.attach('1.3');
    const info = await view.webContents.debugger.sendCommand('Target.getTargetInfo');
    tab.info = info.targetInfo;
    tab.id = info.targetInfo.targetId;
    group.tabs.set(tab.id, tab);
    if (select || !group.selected) group.selected = tab.id;
    view.webContents.debugger.on('message', (_event, method, params, sessionId) => {
      if (method === 'Target.attachedToTarget') tab.sessions.add(params.sessionId);
      if (method === 'Target.detachedFromTarget') tab.sessions.delete(params.sessionId);
      for (const targetSession of sessionId ? [sessionId] : [...tab.roots.keys()]) {
        this.send(group, { method, params, sessionId: targetSession });
      }
    });
    view.webContents.on('destroyed', () => {
      group.tabs.delete(tab.id);
      if (group.selected === tab.id) group.selected = group.tabs.keys().next().value || '';
      for (const [sessionId, parent] of tab.roots) this.send(group, { sessionId: parent, method: 'Target.detachedFromTarget', params: { sessionId, targetId: tab.id } });
      if (this.groups.has(group.id)) this.layout(group);
    });
    if (group.attached) this.attach(group, tab);
    this.layout(group);
    return tab;
  }

  targetInfo(group, tab) {
    return { ...tab.info, targetId: tab.id, type: 'page', title: tab.view.webContents.getTitle(),
      url: tab.view.webContents.getURL(), attached: true, browserContextId: group.id };
  }

  attach(group, tab, parent) {
    const sessionId = randomUUID();
    tab.sessions.add(sessionId);
    tab.roots.set(sessionId, parent);
    tab.primary ||= sessionId;
    this.send(group, { sessionId: parent, method: 'Target.attachedToTarget', params: {
      sessionId, targetInfo: this.targetInfo(group, tab), waitingForDebugger: false,
    } });
    return { sessionId };
  }

  send(group, value) {
    if (group.socket?.readyState === WebSocket.OPEN) group.socket.send(JSON.stringify(value));
  }

  connect(group, socket) {
    group.socket = socket;
    socket.on('error', error => console.warn('[browser-host] connection:', error.message));
    socket.on('message', async raw => {
      let message;
      try {
        message = JSON.parse(raw.toString());
        const result = await this.command(group, message);
        this.send(group, { id: message.id, sessionId: message.sessionId, result });
      } catch (error) {
        this.send(group, { id: message?.id, sessionId: message?.sessionId, error: { code: -32000, message: error.message } });
      }
    });
    socket.on('close', () => {
      group.socket = null; group.attached = false; group.manual = false; group.browserSessions.clear();
      for (const tab of group.tabs.values()) { tab.sessions.clear(); tab.roots.clear(); tab.primary = null; }
      this.layout(group);
    });
  }

  async command(group, { method, params = {}, sessionId }) {
    let tab = [...group.tabs.values()].find(item => item.sessions.has(sessionId));
    if (sessionId && !group.browserSessions.has(sessionId)) {
      if (!tab) throw new Error('Unknown page session');
      if (method === 'Target.getTargetInfo') return { targetInfo: this.targetInfo(group, tab) };
      if (method.startsWith('Browser.') || method.startsWith('Target.')
          && !['Target.setAutoAttach', 'Target.detachFromTarget'].includes(method)) {
        throw new Error('This connection is restricted to conversation tabs.');
      }
      if (method === 'Input.dispatchMouseEvent' && group.id === this.visible && tab.id === group.selected) {
        this.shield.webContents.send('rune:browser-pointer', { x: params.x, y: params.y, click: params.type === 'mousePressed' });
      }
      if (method === 'Page.captureScreenshot') {
        return this.screenshot(tab, params);
      }
      // Synthetic page sessions route to this WebContents; child frames keep their real CDP session.
      return tab.view.webContents.debugger.sendCommand(method, params,
        tab.roots.has(sessionId) ? undefined : sessionId);
    }
    tab = group.tabs.get(params.targetId || group.selected);
    if (method === 'Target.attachToBrowserTarget') {
      const id = randomUUID(); group.browserSessions.add(id); return { sessionId: id };
    }
    if (method === 'Browser.getVersion') return tab.view.webContents.debugger.sendCommand(method);
    if (method === 'Target.setAutoAttach') {
      group.attached = params.autoAttach;
      if (group.attached) for (const page of group.tabs.values()) this.attach(group, page);
      return {};
    }
    if (method === 'Target.getTargetInfo' && tab) return { targetInfo: this.targetInfo(group, tab) };
    if (method === 'Target.getTargets') return { targetInfos: [...group.tabs.values()].map(page => this.targetInfo(group, page)) };
    if (method === 'Target.attachToTarget' && tab) {
      const result = this.attach(group, tab, sessionId);
      return result;
    }
    if (method === 'Target.detachFromTarget') {
      for (const page of group.tabs.values()) { page.sessions.delete(params.sessionId); page.roots.delete(params.sessionId); }
      return {};
    }
    if (method === 'Target.createTarget') {
      const page = await this.createTab(group, params.url);
      return { targetId: page.id };
    }
    if (method === 'Target.closeTarget' && tab) {
      tab.view.webContents.close(); return { success: true };
    }
    if (method === 'Target.activateTarget' && tab) { group.selected = tab.id; this.layout(group); return {}; }
    if (method === 'Browser.setDownloadBehavior') {
      if (!['deny', 'default', 'allowAndName', 'allow'].includes(params.behavior)
          || params.behavior.startsWith('allow') && !path.isAbsolute(params.downloadPath || '')) throw new Error('Invalid download policy');
      group.downloads = params;
      return {};
    }
    if (method === 'Target.setDiscoverTargets') return {};
    throw new Error(`Unsupported browser command: ${method}`);
  }

  async screenshot(tab, params) {
    const contents = tab.view.webContents;
    const metrics = await contents.debugger.sendCommand('Page.getLayoutMetrics');
    const viewport = metrics.cssLayoutViewport;
    const clip = params.clip || { x: viewport.pageX, y: viewport.pageY, width: viewport.clientWidth, height: viewport.clientHeight, scale: 1 };
    if (clip.width <= 0 || clip.height <= 0 || clip.width * clip.height > 16_000_000) throw new Error('Capture a smaller area of the page.');
    const extended = clip.x < viewport.pageX || clip.y < viewport.pageY
      || clip.x + clip.width > viewport.pageX + viewport.clientWidth || clip.y + clip.height > viewport.pageY + viewport.clientHeight;
    try {
      if (extended) {
        const ratio = await contents.debugger.sendCommand('Runtime.evaluate', { expression: 'devicePixelRatio', returnByValue: true });
        // Change the capture area without changing viewport units or the page's scroll position.
        await contents.debugger.sendCommand('Emulation.setDeviceMetricsOverride', {
          width: viewport.clientWidth, height: viewport.clientHeight, deviceScaleFactor: ratio.result.value, mobile: false,
          viewport: { x: clip.x, y: clip.y, width: clip.width, height: clip.height, scale: 1 },
        });
      }
      // CDP captures stall on hidden Electron views. The native capture API wakes their compositor.
      const captured = await contents.capturePage();
      let frame = nativeImage.createFromBuffer(captured.toPNG({ scaleFactor: 1 }));
      const size = frame.getSize();
      if (size.width < clip.width || size.height < clip.height) throw new Error(`Capture is ${size.width}x${size.height}, expected ${clip.width}x${clip.height}`);
      if (!extended) frame = frame.crop({ x: Math.round(clip.x - viewport.pageX), y: Math.round(clip.y - viewport.pageY), width: Math.round(clip.width), height: Math.round(clip.height) });
      frame = frame.resize({ width: Math.max(1, Math.round(clip.width * clip.scale)), height: Math.max(1, Math.round(clip.height * clip.scale)) });
      return { data: (params.format === 'jpeg' ? frame.toJPEG(params.quality ?? 80) : frame.toPNG()).toString('base64') };
    } finally {
      if (extended) await contents.debugger.sendCommand('Emulation.clearDeviceMetricsOverride');
    }
  }

  mount(sessionId, bounds) {
    const group = this.groups.get(sessionId);
    if (!group) return false;
    this.visible = sessionId;
    group.bounds = bounds;
    for (const entry of this.groups.values()) this.layout(entry);
    return true;
  }

  layout(group) {
    if (this.closed || this.window.isDestroyed() || !this.groups.has(group.id)) return;
    for (const tab of group.tabs.values()) {
      const visible = this.visible === group.id && tab.id === group.selected && group.bounds;
      if (!this.window.contentView.children.includes(tab.view)) this.window.contentView.addChildView(tab.view);
      tab.view.setVisible(Boolean(visible));
      tab.view.setBounds(group.bounds || { x: 0, y: 0, width: 1280, height: 720 });
    }
    this.window.contentView.addChildView(this.shield);
    const active = this.groups.get(this.visible);
    this.shield.setVisible(Boolean(active?.bounds && !active.manual));
    if (active?.bounds) this.shield.setBounds(active.bounds);
  }

  async takeover() {
    if (this.takingControl || !this.visible) return;
    this.takingControl = true;
    const id = this.visible;
    try {
      let state = await this.api(`/api/computer/state?sessionId=${encodeURIComponent(id)}&preview=false`);
      if (state.state === 'running') state = await this.api('/api/computer/control', { sessionId: id, lease: state.lease, action: 'pause' });
      const deadline = Date.now() + 15000;
      while (state.state === 'pausing' && Date.now() < deadline) {
        await new Promise(resolve => setTimeout(resolve, 100));
        state = await this.api(`/api/computer/state?sessionId=${encodeURIComponent(id)}&preview=false`);
      }
      if (!['paused', 'idle', 'manual'].includes(state.state)) throw new Error('Wait for the current action to finish.');
      if (state.state !== 'manual') await this.api('/api/computer/control', { sessionId: id, lease: state.lease, action: 'takeover' });
    } catch (error) { console.warn('[browser-host] takeover:', error.message); }
    finally { this.takingControl = false; }
  }

  hide() { this.visible = null; for (const group of this.groups.values()) this.layout(group); }

  closeGroup(group) {
    this.groups.delete(group.id);
    group.partition?.removeListener('will-download', group.downloadListener);
    if (this.visible === group.id) this.visible = null;
    group.socket?.close();
    for (const tab of group.tabs.values()) {
      this.window.contentView.removeChildView(tab.view);
      tab.view.webContents.close();
    }
    if (!this.closed && !this.window.isDestroyed()) {
      const active = this.groups.get(this.visible);
      if (active) this.layout(active);
      else this.shield.setVisible(false);
    }
  }

  close() {
    if (this.closed) return;
    this.closed = true;
    clearInterval(this.heartbeat);
    for (const group of this.groups.values()) this.closeGroup(group);
    this.sockets.close(); this.server.close();
    this.shield.webContents.close();
  }
}

module.exports = { BrowserHost, webURL };
