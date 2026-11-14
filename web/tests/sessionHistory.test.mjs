import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server, SessionListLoader, SessionSidebar, WorkbenchPanel;
before(async () => {
  server = await createServer({
    root: fileURLToPath(new URL('..', import.meta.url)),
    server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom',
    optimizeDeps: { noDiscovery: true, include: [] },
  });
  ({ SessionListLoader } = await server.ssrLoadModule('/src/utils/sessionList.ts'));
  ({ SessionSidebar } = await server.ssrLoadModule('/src/components/SessionSidebar.tsx'));
  ({ WorkbenchPanel } = await server.ssrLoadModule('/src/components/WorkbenchPanel.tsx'));
});
after(async () => { await server?.close(); });
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
const page = (start, count, total = 75) => ({
  sessions: Array.from({ length: count }, (_, index) => ({ id: String(start + index), title: `Chat ${start + index}` })), total,
});
function mount() {
  const calls = [];
  let state;
  const loader = new SessionListLoader(params => {
    const result = deferred();
    calls.push({ params, ...result });
    return result.promise;
  }, next => { state = next; });
  loader.reset('');
  return { loader, calls, get state() { return state; } };
}

test('older pages append without losing history and periodic requests stay bounded', async () => {
  const app = mount();
  let loading = app.loader.load();
  app.calls[0].resolve(page(0, 50));
  await loading;
  assert.equal(app.state.hasMore, true);
  loading = app.loader.loadMore();
  await app.loader.refresh();
  assert.equal(app.calls.length, 2);
  assert.deepEqual(app.calls[1].params, { limit: 50, offset: 50, query: '' });
  app.calls[1].resolve(page(49, 26));
  await loading;
  assert.equal(app.state.sessions.length, 75);
  assert.equal(new Set(app.state.sessions.map(session => session.id)).size, 75);
  assert.equal(app.state.hasMore, false);
  await app.loader.refresh();
  assert.equal(app.calls.length, 2);
});

test('stale searches and cancelled loads cannot overwrite the current query', async () => {
  const app = mount();
  const old = app.loader.load();
  app.loader.reset('older title');
  const current = app.loader.load();
  app.calls[0].resolve(page(0, 50));
  await old;
  assert.equal(app.state.loading, true);
  assert.deepEqual(app.state.sessions, []);
  app.calls[1].resolve(page(60, 1, 1));
  await current;
  assert.equal(app.state.sessions[0].id, '60');
  await app.loader.refresh();
  assert.equal(app.calls.length, 2);
  app.loader.reset('next');
  const cancelled = app.loader.load();
  app.loader.cancel();
  app.calls[2].reject(new Error('late failure'));
  await cancelled;
  assert.equal(app.state.error, null);
});

test('failed pagination retries the same offset while retaining loaded chats', async () => {
  const app = mount();
  let loading = app.loader.load();
  app.calls[0].resolve(page(0, 50));
  await loading;
  loading = app.loader.loadMore();
  app.calls[1].reject(new Error('offline'));
  await loading;
  assert.equal(app.state.error, 'more');
  assert.equal(app.state.sessions.length, 50);
  loading = app.loader.retry();
  assert.equal(app.calls[2].params.offset, 50);
  app.calls[2].resolve(page(50, 25));
  await loading;
  assert.equal(app.state.sessions.length, 75);
  assert.equal(app.state.error, null);
});

test('clearing search restores the recent first page and refresh never overlaps', async () => {
  const app = mount();
  app.loader.reset('archive');
  const search = app.loader.load();
  app.loader.reset('');
  const recent = app.loader.load();
  app.calls[0].resolve(page(70, 1, 1));
  await search;
  app.calls[1].resolve(page(0, 50));
  await recent;
  const refresh = app.loader.refresh();
  await app.loader.refresh();
  assert.equal(app.calls.length, 3);
  assert.deepEqual(app.calls[2].params, { limit: 50, offset: 0, query: '' });
  app.calls[2].reject(new Error('offline'));
  await refresh;
  assert.equal(app.state.sessions.length, 50);
  assert.equal(app.state.error, 'refresh');
});

test('chat search keeps current navigation and saved tasks outside the results', () => {
  const html = renderToStaticMarkup(createElement(SessionSidebar, {
    currentSessionId: 'older', liveTitle: 'Current request', onSelectSession() {}, onSelectTask() {},
  }));
  assert.match(html, /aria-label="Search chats by title"/);
  assert.match(html, /Current request/);
  assert.match(html, /Saved work/);
  assert.match(html, /New chat/);
});

test('workspace shortcuts follow file activity without crowding research or history', () => {
  const render = props => renderToStaticMarkup(createElement(WorkbenchPanel, {
    sessionId: 'live', toolCalls: [], isRunning: false, activitySummary: null, onClose() {}, ...props,
  }));
  const workspacePath = '/projects/project-name';
  const call = toolName => ({ id: toolName, toolName, args: { path: 'notes.txt' }, timestamp: 1 });
  const selected = render({ workspacePath, toolCalls: [call('file_read')] });
  assert.match(selected, /aria-label="Workspace tools"/);
  assert.match(selected, />project-name</);
  assert.match(selected, />Open file</);
  assert.match(selected, />Changes</);
  assert.doesNotMatch(selected, /role="tab"[^>]*>Terminal/);
  for (const toolCalls of [[], [call('web_search')], [call('browser_navigate')]]) {
    const quiet = render({ workspacePath, toolCalls });
    assert.doesNotMatch(quiet, /aria-label="Workspace tools"/);
    assert.match(quiet, /aria-label="Panel tools"/);
  }
  assert.match(render({ workspacePath, fileChanges: [{ id: 'saved', path: 'notes.txt', kind: 'modified', patch: '-old\n+new' }] }), /aria-label="Workspace tools"/);
  assert.doesNotMatch(render({}), /aria-label="Workspace tools"/);
  assert.doesNotMatch(render({ workspacePath, toolCalls: [call('file_read')], historical: true }), /aria-label="Workspace tools"/);
});
