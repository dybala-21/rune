/**
 * RUNE Browser Bridge — Background Service Worker
 *
 * 자동 연결: relay 서버를 주기적으로 탐색하여 발견 즉시 WebSocket 자동 연결.
 * 사용자가 Connect 버튼을 누를 필요 없음.
 *
 * 프로토콜:
 *   Relay → Extension: { id, method: "attachToTab"|"forwardCDPCommand", params }
 *   Extension → Relay: { id, result } 또는 { id, error }
 *   Extension → Relay: { method: "forwardCDPEvent", params: { method, params, sessionId } }
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DISCOVERY_PORT_START = 19222;
const DISCOVERY_PORT_END = 19231;
const POLL_ALARM_NAME = 'rune-relay-poll';
const POLL_INTERVAL_MINUTES = 0.5; // 30초

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let ws = null;
let attachedTabId = null;
let isConnected = false;
let lastError = '';
let autoDiscoveryEnabled = true;

// ---------------------------------------------------------------------------
// Auto-discovery: relay 서버를 자동으로 찾아 연결
// ---------------------------------------------------------------------------

async function discoverAndConnect() {
  if (isConnected) return;

  // 마지막으로 성공한 포트부터 시도 (대부분 같은 포트 재사용)
  const { lastRelayPort } = await chrome.storage.local.get('lastRelayPort');
  const portsToTry = [];
  if (lastRelayPort >= DISCOVERY_PORT_START && lastRelayPort <= DISCOVERY_PORT_END) {
    portsToTry.push(lastRelayPort);
  }
  for (let port = DISCOVERY_PORT_START; port <= DISCOVERY_PORT_END; port++) {
    if (port !== lastRelayPort) portsToTry.push(port);
  }

  for (const port of portsToTry) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 500);
      const resp = await fetch(`http://127.0.0.1:${port}/discover`, {
        signal: controller.signal,
      });
      clearTimeout(timer);

      if (resp.ok) {
        const data = await resp.json();
        if (data.extensionEndpoint) {
          console.log(`[RUNE Bridge] Relay 서버 발견: port ${port}`);
          connect(data.extensionEndpoint, port);
          return;
        }
      }
    } catch {
      // 포트 미사용 — 무시
    }
  }
}

// Service Worker 시작 시 즉시 탐색
discoverAndConnect();

// 빠른 폴링: 2초 간격으로 relay 서버 탐색 (최대 60초, 연결 후 자동 중단)
// chrome.alarms 최소 간격이 30초라 setTimeout 사용
let rapidRetryCount = 0;
const RAPID_RETRY_MAX = 30; // 30 * 2s = 60초
const RAPID_RETRY_INTERVAL = 2000;
let rapidRetryTimer = null;

function startRapidRetry() {
  rapidRetryCount = 0;
  if (rapidRetryTimer) clearTimeout(rapidRetryTimer);
  rapidRetryTimer = setTimeout(rapidRetry, RAPID_RETRY_INTERVAL);
}

function rapidRetry() {
  rapidRetryTimer = null;
  if (isConnected || rapidRetryCount >= RAPID_RETRY_MAX) return;
  rapidRetryCount++;
  discoverAndConnect().then(() => {
    if (!isConnected) {
      rapidRetryTimer = setTimeout(rapidRetry, RAPID_RETRY_INTERVAL);
    }
  });
}
startRapidRetry();

// 장기 폴링: 30초마다 (초기 빠른 폴링 종료 후 백업)
chrome.alarms.create(POLL_ALARM_NAME, { periodInMinutes: POLL_INTERVAL_MINUTES });

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === POLL_ALARM_NAME && autoDiscoveryEnabled) {
    discoverAndConnect();
  }
});

// 탭 활성화 시 즉시 탐색 (RUNE 실행 직후 탭 전환하면 빠르게 감지)
chrome.tabs.onActivated.addListener(() => {
  if (!isConnected && autoDiscoveryEnabled) {
    discoverAndConnect();
  }
});

// ---------------------------------------------------------------------------
// WebSocket connection
// ---------------------------------------------------------------------------

function connect(relayUrl, port) {
  console.log('[RUNE Bridge] Connecting to:', relayUrl);
  lastError = '';

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
  }

  try {
    ws = new WebSocket(relayUrl);
  } catch (err) {
    lastError = `WebSocket 생성 실패: ${err.message}`;
    console.error('[RUNE Bridge]', lastError);
    notifyPopup({ type: 'status', connected: false, error: lastError });
    return;
  }

  ws.onopen = () => {
    console.log('[RUNE Bridge] WebSocket connected');
    isConnected = true;
    lastError = '';
    if (port) {
      chrome.storage.local.set({ lastRelayPort: port });
    }
    updateBadge('ON', '#4CAF50');
    notifyPopup({ type: 'status', connected: true });
  };

  ws.onclose = (event) => {
    console.log('[RUNE Bridge] WebSocket closed:', event.code, event.reason);
    const wasConnected = isConnected;
    isConnected = false;
    if (!lastError && event.code !== 1000) {
      lastError = `연결 끊김 (code: ${event.code})`;
    }
    detachDebugger();
    updateBadge('', '');
    notifyPopup({ type: 'status', connected: false, error: lastError });
    ws = null;

    // 연결이 끊기면 빠른 재탐색 시작 (2초 간격)
    if (wasConnected && autoDiscoveryEnabled) {
      startRapidRetry();
    }
  };

  ws.onerror = () => {
    lastError = 'WebSocket 연결 실패';
    console.error('[RUNE Bridge]', lastError);
  };

  ws.onmessage = (event) => {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch {
      return;
    }
    handleRelayMessage(msg);
  };
}

function disconnect() {
  if (ws) {
    ws.close();
    ws = null;
  }
  detachDebugger();
  isConnected = false;
  lastError = '';
  updateBadge('', '');
  notifyPopup({ type: 'status', connected: false });
}

// ---------------------------------------------------------------------------
// Relay message handlers
// ---------------------------------------------------------------------------

async function handleRelayMessage(msg) {
  const { id, method, params } = msg;

  try {
    let result;
    switch (method) {
      case 'attachToTab':
        result = await handleAttachToTab();
        break;
      case 'forwardCDPCommand':
        result = await handleForwardCDP(params);
        break;
      default:
        throw new Error(`Unknown method: ${method}`);
    }
    sendToRelay({ id, result });
  } catch (err) {
    console.error('[RUNE Bridge] Handler error:', method, err);
    sendToRelay({ id, error: err.message || String(err) });
  }
}

async function handleAttachToTab() {
  // chrome://, edge://, about: 등 디버거 접근 불가 URL은 건너뛰기
  const RESTRICTED_PREFIXES = ['chrome://', 'chrome-extension://', 'edge://', 'about:', 'devtools://'];

  function isRestricted(url) {
    return !url || RESTRICTED_PREFIXES.some(prefix => url.startsWith(prefix));
  }

  // 1) 활성 탭 시도
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  // 2) 활성 탭이 제한 URL이면 일반 웹 탭 찾기
  if (!tab || !tab.id || isRestricted(tab.url)) {
    console.log('[RUNE Bridge] Active tab is restricted:', tab?.url, '— finding a web tab');
    const allTabs = await chrome.tabs.query({ currentWindow: true });
    tab = allTabs.find(t => t.id && !isRestricted(t.url));
  }

  // 3) 일반 웹 탭도 없으면 새 탭 생성
  if (!tab || !tab.id) {
    console.log('[RUNE Bridge] No suitable tab found — creating new tab');
    tab = await chrome.tabs.create({ url: 'about:blank', active: true });
    // about:blank은 제한 URL이지만 디버거 attach는 가능
  }

  if (attachedTabId && attachedTabId !== tab.id) {
    await detachDebugger();
  }

  if (attachedTabId !== tab.id) {
    console.log('[RUNE Bridge] Attaching debugger to tab:', tab.id, tab.url);
    await chrome.debugger.attach({ tabId: tab.id }, '1.3');
    attachedTabId = tab.id;

    chrome.debugger.onEvent.removeListener(onDebuggerEvent);
    chrome.debugger.onEvent.addListener(onDebuggerEvent);
    chrome.debugger.onDetach.removeListener(onDebuggerDetach);
    chrome.debugger.onDetach.addListener(onDebuggerDetach);
  }

  // 실제 CDP 프레임 ID 조회 — Playwright가 targetId === mainFrame.id를 기대
  let frameId = `tab-${tab.id}`;
  try {
    const frameTree = await chrome.debugger.sendCommand(
      { tabId: tab.id },
      'Page.getFrameTree',
      {}
    );
    if (frameTree?.frameTree?.frame?.id) {
      frameId = frameTree.frameTree.frame.id;
      console.log('[RUNE Bridge] Main frame ID:', frameId);
    }
  } catch (err) {
    console.warn('[RUNE Bridge] Page.getFrameTree failed, using fallback targetId:', err.message);
  }

  return {
    targetInfo: {
      targetId: frameId,
      type: 'page',
      title: tab.title || '',
      url: tab.url || '',
      browserContextId: 'default',
    },
  };
}

async function handleForwardCDP(params) {
  if (!attachedTabId) {
    throw new Error('No tab attached');
  }

  const { sessionId, method, params: cdpParams } = params;

  // Include sessionId for child sessions (iframes, workers, etc.)
  // chrome.debugger API supports flat sessions since Chrome 125+
  const target = { tabId: attachedTabId };
  if (sessionId) {
    target.sessionId = sessionId;
  }

  const result = await chrome.debugger.sendCommand(
    target,
    method,
    cdpParams || {}
  );
  return result;
}

// ---------------------------------------------------------------------------
// CDP event forwarding
// ---------------------------------------------------------------------------

function onDebuggerEvent(source, method, params) {
  if (!source.tabId || source.tabId !== attachedTabId) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  sendToRelay({
    method: 'forwardCDPEvent',
    params: {
      method,
      params,
      // Forward session ID for child session events (iframes, workers)
      sessionId: source.sessionId || undefined,
    },
  });
}

function onDebuggerDetach(source, reason) {
  if (source.tabId === attachedTabId) {
    console.log('[RUNE Bridge] Debugger detached:', reason);
    attachedTabId = null;
    if (reason === 'canceled_by_user') {
      disconnect();
    }
  }
}

// ---------------------------------------------------------------------------
// Debugger cleanup
// ---------------------------------------------------------------------------

async function detachDebugger() {
  if (attachedTabId) {
    try {
      await chrome.debugger.detach({ tabId: attachedTabId });
    } catch {
      // 이미 detach된 경우 무시
    }
    attachedTabId = null;
  }
}

// ---------------------------------------------------------------------------
// Badge & popup communication
// ---------------------------------------------------------------------------

function updateBadge(text, color) {
  chrome.action.setBadgeText({ text });
  if (color) {
    chrome.action.setBadgeBackgroundColor({ color });
  }
}

function notifyPopup(message) {
  chrome.runtime.sendMessage(message).catch(() => {});
}

function sendToRelay(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}

// ---------------------------------------------------------------------------
// Message listener (from popup)
// ---------------------------------------------------------------------------

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  switch (message.type) {
    case 'connect':
      connect(message.relayUrl, message.port);
      sendResponse({ ok: true });
      break;
    case 'disconnect':
      disconnect();
      sendResponse({ ok: true });
      break;
    case 'getStatus':
      sendResponse({
        connected: isConnected,
        tabId: attachedTabId,
        error: lastError,
        autoDiscovery: autoDiscoveryEnabled,
      });
      break;
    case 'setAutoDiscovery':
      autoDiscoveryEnabled = message.enabled;
      sendResponse({ ok: true });
      break;
    default:
      sendResponse({ error: 'Unknown message type' });
  }
  return true;
});
