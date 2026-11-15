import assert from 'node:assert/strict';
import { after, before, beforeEach, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

const originalStorage = globalThis.sessionStorage;
let server, runActivity, ChatPanel, StatusBar, api;
before(async () => {
  globalThis.sessionStorage = { getItem: () => 'activity-test', setItem() {} };
  server = await createServer({
    configFile: false, esbuild: { jsx: 'automatic' },
    root: fileURLToPath(new URL('..', import.meta.url)),
    server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom',
    optimizeDeps: { noDiscovery: true, include: [] },
  });
  ({ runActivity } = await server.ssrLoadModule('/src/utils/runActivity.ts'));
  ({ ChatPanel } = await server.ssrLoadModule('/src/components/ChatPanel.tsx'));
  ({ StatusBar } = await server.ssrLoadModule('/src/components/StatusBar.tsx'));
  api = await server.ssrLoadModule('/src/api.ts');
  api.setLiveSessionId('activity-test');
});
after(async () => {
  await server?.close();
  if (originalStorage === undefined) delete globalThis.sessionStorage;
  else globalThis.sessionStorage = originalStorage;
});
beforeEach(() => { api.setCurrentRunId('current'); });

const call = (extra = {}) => ({ id: 'tool', toolName: 'web_search', args: { query: 'Release notes' },
  runId: 'current', timestamp: 2, ...extra });
const waiting = { state: 'thinking', label: 'Preparing a response…' };
const render = (component, props) => renderToStaticMarkup(createElement(component, props));
const chat = props => render(ChatPanel, {
  conversationKey: 'live', connected: true, isRunning: true,
  messages: [{ id: 'request', role: 'user', content: 'Check the release notes.', timestamp: 1 }],
  toolCalls: [], thinkingBlocks: [], activitySummary: null, delegateEvents: [], compactionEvents: [],
  pendingApproval: null, pendingQuestion: null, onRespondApproval: async () => {}, onRespondQuestion: async () => {},
  ...props,
});

function presence(html) {
  return html.match(/<div class="run-presence"[\s\S]*?<\/div>/)?.[0] ?? '';
}

test('a new run ignores unresolved tools from an earlier run or an unknown run', () => {
  const old = call({ runId: 'previous' });
  assert.deepEqual(runActivity([old], 'current'), waiting);
  assert.deepEqual(runActivity([call()], ''), waiting);
  assert.deepEqual(runActivity([call({ runId: undefined })], 'current'), waiting);
  assert.match(presence(chat({ toolCalls: [old] })), /data-state="thinking"/);
  assert.doesNotMatch(presence(chat({ toolCalls: [old] })), /Searching the web/);
});

test('completed calls do not keep the wolf working when their result text is absent', () => {
  for (const completion of [{ result: '' }, { completedAt: 3 }, { completedAt: 0 }]) {
    assert.deepEqual(runActivity([call(completion)], 'current'), waiting);
  }
  const calls = [call({ id: 'slow', toolName: 'file_read' }), call({ id: 'done', completedAt: 3 })];
  assert.deepEqual(runActivity(calls, 'current'), { state: 'working', label: 'Reading files…' });
});

test('current tool execution and response preparation have distinct, readable states', () => {
  assert.deepEqual(runActivity([call()], 'current'), { state: 'working', label: 'Searching the web…' });
  assert.deepEqual(runActivity([call({ toolName: 'think' })], 'current'), { state: 'thinking', label: 'Planning the next step…' });
  assert.deepEqual(runActivity([call({ toolName: 'plugin_private_method', args: { token: 'PRIVATE' } })], 'current'),
    { state: 'working', label: 'Working on your request…' });
  const active = presence(chat({ toolCalls: [call()] }));
  assert.match(active, /role="status" aria-live="polite"/);
  assert.match(active, /data-state="working"/);
  assert.match(active, /Searching the web…/);
  assert.match(presence(chat({})), /data-state="thinking"/);
});

test('idle, disconnected, historical, and waiting-for-user views do not animate an active wolf', () => {
  const approval = { id: 'approval', command: 'Save changes', riskLevel: 'medium', timeoutMs: 30000, receivedAt: 1 };
  const question = { id: 'question', question: 'Which folder?' };
  for (const props of [
    { isRunning: false }, { connected: false },
    { conversationKey: 'history', isRunning: false },
    { pendingApproval: approval }, { pendingQuestion: question },
  ]) {
    const html = chat({ toolCalls: [call()], ...props });
    assert.equal(presence(html), '', JSON.stringify(props));
    assert.doesNotMatch(html, /class="rune-mark" data-state="(?:thinking|working)"/);
  }
});

test('the header keeps a static identity and preserves attention messages and controls', () => {
  for (const connected of [false, true]) {
    for (const state of ['idle', 'submitting', 'running', 'stopping', 'waiting_approval', 'waiting_question']) {
      const html = render(StatusBar, { connected, state, showRunActivity: true, currentActivity: 'Reading files…' });
      assert.match(html, /class="rune-mark" data-state="idle"/);
      assert.doesNotMatch(html, /data-state="(?:thinking|working|passed)"|status-dot--pulse/);
    }
  }
  const approval = render(StatusBar, { connected: true, state: 'waiting_approval', approvalMode: 'bypass',
    onToggleWorkbench() {}, onOpenSettings() {}, tokenUsage: { total: 123, cost: { usd: 0.02 } } });
  assert.match(approval, /Waiting for approval/);
  assert.match(approval, /Approvals off/);
  assert.match(approval, /aria-label="Show work panel"/);
  assert.match(approval, /aria-label="Open settings"/);
  assert.match(approval, /aria-label="Usage and cost"/);
});

test('the header repeats running activity only when the chat activity is out of view', () => {
  const props = { connected: true, state: 'running', currentActivity: 'Reading files…' };
  assert.doesNotMatch(render(StatusBar, props), /Reading files…|class="toolbar-status"/);
  assert.match(render(StatusBar, { ...props, showRunActivity: true }), /Reading files…/);
  assert.match(render(StatusBar, { ...props, attentionLabel: 'Waiting for app access' }), /Waiting for app access/);
});


test('completed page reads and searches share a compact summary without hiding failures', () => {
  const tools = [
    call({ id: 'empty', result: 'No search results found.', success: true }),
    call({ id: 'read', toolName: 'web_fetch', args: { url: 'https://example.com/news' }, timestamp: 3, result: 'Page content', success: true }),
    call({ id: 'failed', toolName: 'web_fetch', args: { url: 'https://example.com/missing' }, timestamp: 4, result: 'Not found', success: false }),
  ];
  const html = chat({ toolCalls: tools, isRunning: false });
  assert.equal((html.match(/class="tool-activity web-activity/g) ?? []).length, 1);
  assert.match(html, /Web activity · 1 search · 2 page requests · 1 request failed · 1 search without results/);
  assert.doesNotMatch(html, /3 succeeded|Page content|Not found/);
});

test('pending searches and mutating requests remain outside a completed web group', () => {
  const read = call({ id: 'read', toolName: 'web_fetch', args: { url: 'https://example.com/read' }, result: 'Read', success: true });
  for (const separate of [call({ id: 'pending', timestamp: 3 }),
    call({ id: 'post', toolName: 'web_fetch', args: { url: 'https://example.com/save', method: 'POST' }, timestamp: 3, result: 'Accepted', success: true })]) {
    const html = chat({ toolCalls: [read, separate] });
    assert.doesNotMatch(html, /class="tool-activity web-activity/);
    assert.match(html, /Read web page/);
    if (separate.id === 'pending') assert.match(html, /Searching the web/);
  }
});

test('web groups preserve run and message boundaries and unconfirmed checks', () => {
  const read = call({ id: 'read', toolName: 'web_fetch', args: { url: 'https://example.com/read' }, result: 'Read', success: true });
  const checked = call({ id: 'check', timestamp: 4, result: 'Search response', success: true, checkStatus: 'inconclusive' });
  assert.match(chat({ toolCalls: [read, checked], isRunning: false }), /1 request has unconfirmed checks/);
  assert.doesNotMatch(chat({ toolCalls: [read, { ...checked, runId: 'another' }] }), /class="tool-activity web-activity/);
  const messages = [{ id: 'followup', role: 'user', content: 'Check something else', timestamp: 3 }];
  assert.doesNotMatch(chat({ toolCalls: [read, checked], messages }), /class="tool-activity web-activity/);
});

test('web groups do not absorb questions or code edits', () => {
  const read = call({ id: 'read', toolName: 'web_fetch', args: { url: 'https://example.com/read' }, result: 'Read', success: true });
  for (const middle of [
    call({ id: 'question', toolName: 'ask_user', args: { question: 'Which folder?' }, timestamp: 3 }),
    call({ id: 'edit', toolName: 'file_edit', args: { path: 'src/main.ts' }, result: 'Saved', success: true, timestamp: 3 }),
  ]) {
    const html = chat({ toolCalls: [read, middle, { ...read, id: 'last', timestamp: 4 }] });
    assert.doesNotMatch(html, /class="tool-activity web-activity/);
    assert.match(html, middle.id === 'question' ? /Which folder/ : /Edit file/);
  }
});


test('completed search groups keep the web disclosure as page reads arrive', () => {
  const searches = [1, 2].map(i => call({ id: `search-${i}`, timestamp: i + 1, result: 'No search results found.', success: true }));
  const read = call({ id: 'read', toolName: 'web_fetch', args: { url: 'https://example.com' }, timestamp: 4, result: 'Read', success: true });
  for (const tools of [searches, [...searches, read]]) {
    const html = chat({ toolCalls: tools });
    assert.equal((html.match(/class="tool-activity web-activity/g) ?? []).length, 1);
    assert.match(html, /2 searches without results/);
  }
  const pending = searches.map(tool => ({ ...tool, result: undefined, success: undefined }));
  const html = chat({ toolCalls: pending });
  assert.equal((html.match(/class="search-activity"/g) ?? []).length, 1);
  assert.match(html, /Searching the web/);
  assert.doesNotMatch(html, /class="tool-activity web-activity/);
});
