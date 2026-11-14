import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server, mergeConversationList, rememberLiveConversation, SessionSidebar, useAgent, api,
  shouldPreserveLocalRequest, restoreRunMessages, latestTurn;
before(async () => {
  server = await createServer({
    // SSR tests need JSX compilation, not the dev plugin's dependency optimizer.
    configFile: false, esbuild: { jsx: 'automatic' },
    root: fileURLToPath(new URL('..', import.meta.url)),
    server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom',
    optimizeDeps: { noDiscovery: true, include: [] },
  });
  ({ mergeConversationList, rememberLiveConversation } = await server.ssrLoadModule('/src/utils/conversationList.ts'));
  ({ SessionSidebar } = await server.ssrLoadModule('/src/components/SessionSidebar.tsx'));
  ({ useAgent } = await server.ssrLoadModule('/src/hooks/useAgent.ts'));
  ({ shouldPreserveLocalRequest, restoreRunMessages } = await server.ssrLoadModule('/src/utils/runSnapshot.ts'));
  ({ latestTurn } = await server.ssrLoadModule('/src/utils/retry.ts'));
  api = await server.ssrLoadModule('/src/api.ts');
});
after(async () => { await server?.close(); });

const chat = (id, title, turnCount = 2, updatedAt = '2026-11-14T10:00:00Z') => ({ id, title, turnCount, updatedAt });

test('a new live chat appears once before history polling catches up', () => {
  const live = chat('live', 'Plan the launch', 3);
  assert.deepEqual(mergeConversationList([], [], live, ''), [live]);
  const saved = chat('live', 'Old title', 2, '2026-11-14T09:00:00Z');
  assert.deepEqual(mergeConversationList([saved], [live], live, ''), [live]);
});

test('starting another chat retains the previous conversation immediately', () => {
  const previous = chat('previous', 'Completed request');
  const remembered = rememberLiveConversation([], previous);
  const blank = chat('new', 'New chat', 0, '2026-11-14T11:00:00Z');
  assert.deepEqual(rememberLiveConversation(remembered, blank), remembered);
  assert.deepEqual(mergeConversationList([], remembered, blank, '').map(item => item.id), ['new', 'previous']);
  assert.deepEqual(mergeConversationList([previous], remembered, blank, '').map(item => item.id), ['new', 'previous']);
});

test('unconfirmed delivery cannot leave a phantom history row, while persisted turns remain', () => {
  const failed = chat('failed', 'Unconfirmed first request', 1);
  assert.deepEqual(rememberLiveConversation([], failed, false), []);
  const confirmed = chat('existing', 'Previously saved request');
  const remembered = rememberLiveConversation([], confirmed);
  const pendingFollowup = { ...confirmed, turnCount: 3 };
  const retained = rememberLiveConversation(remembered, pendingFollowup, false);
  assert.deepEqual(retained, []);
  const blank = chat('new', 'New chat', 0, '2026-11-14T11:00:00Z');
  assert.deepEqual(mergeConversationList([], retained, blank, '').map(item => item.id), ['new']);
  assert.deepEqual(mergeConversationList([confirmed], retained, blank, '').map(item => item.id), ['new', 'existing']);
});

test('optimistic conversation rows obey title search and later server titles win', () => {
  const local = chat('past', 'Launch notes');
  const live = chat('live', 'Another request');
  assert.deepEqual(mergeConversationList([], [local], live, '  LAUNCH  ').map(item => item.id), ['past']);
  assert.deepEqual(mergeConversationList([], [local], live, 'no match'), []);
  const renamed = chat('past', 'Project plan', 4, '2026-11-14T12:00:00Z');
  assert.deepEqual(mergeConversationList([renamed], [local], live, 'Launch'), []);
  assert.deepEqual(mergeConversationList([renamed], [local], live, 'plan'), [renamed]);
});

test('an earlier stopped run cannot replace a later rejected or uncertain request after reload', () => {
  const oldRun = { runId: 'old-run', requestId: 'old-request', status: 'cancelled',
    goal: 'Previous task', text: '', startedAt: 100, updatedAt: 200, trust: null };
  for (const delivery of ['pending', 'unknown', 'rejected']) {
    const saved = JSON.parse(JSON.stringify([
      { id: 'old-run:user', role: 'user', content: oldRun.goal, timestamp: 100, requestId: 'old-request', delivery: 'accepted' },
      { id: 'new-user', role: 'user', content: 'Retry this request', timestamp: 300, requestId: 'new-request', delivery },
      { id: 'delivery-new-request', role: 'system', content: 'Failed to send', timestamp: 301, level: 'error' },
    ]));
    // Reload has no visible messages yet; the saved copy must protect Restore.
    assert.equal(shouldPreserveLocalRequest([], oldRun), false);
    assert.equal(shouldPreserveLocalRequest(saved, oldRun), true);
    const restored = restoreRunMessages(saved, oldRun);
    assert.deepEqual(restored, saved);
    assert.equal(latestTurn(restored).user.requestId, 'new-request');
    assert.equal(latestTurn(restored).retryAnchor.id, 'delivery-new-request');

    const receipt = { ...oldRun, runId: 'new-run', requestId: 'new-request', goal: 'Retry this request', startedAt: 300 };
    assert.equal(shouldPreserveLocalRequest(saved, receipt), false);
    const acknowledged = restoreRunMessages(saved, receipt);
    assert.equal(latestTurn(acknowledged).user.delivery, 'accepted');
    assert.equal(latestTurn(acknowledged).user.requestId, 'new-request');
    assert.equal(acknowledged.some(message => message.id === 'delivery-new-request'), false);
  }
});

test('snapshot protection does not prevent a newer run or explicit conversation load', async t => {
  const rejected = [{ id: 'local', role: 'user', content: 'Earlier rejected request', timestamp: 10,
    requestId: 'rejected-request', delivery: 'rejected' }];
  const run = { runId: 'newer-run', sessionId: 'another-chat', requestId: 'newer-request',
    status: 'running', goal: 'Newer server task', startedAt: 20, text: '', stepNumber: 1, toolCalls: [],
    question: null, approval: null, trust: null };
  assert.equal(shouldPreserveLocalRequest(rejected, run), false);
  assert.equal(shouldPreserveLocalRequest(rejected, { ...run, status: 'completed' }), false);
  const { agent } = hookFixture(t, () => response({ success: true, data: { turns: [], run } }),
    [{ ...rejected[0], delivery: 'unknown' }]);
  agent.restoreSavedDraft();
  assert.equal(await agent.loadConversation('another-chat'), true);
  assert.equal(api.getCurrentRunId(), 'newer-run');
});

test('live navigation remains a selected list row while history initially loads', () => {
  const html = renderToStaticMarkup(createElement(SessionSidebar, {
    currentSessionId: null, liveTitle: 'Current request', liveState: 'running', liveMessageCount: 1,
    newChatDisabled: true, newChatDisabledReason: 'Finish or stop this chat before starting another.',
    suggestionCount: 3, onOpenSuggestions() {}, onSelectSession() {}, onSelectTask() {},
  }));
  assert.doesNotMatch(html, /session-current/);
  assert.match(html, /class="session-list"[\s\S]*aria-current="page"[\s\S]*Current request/);
  assert.match(html, /class="new-task-button" disabled="" aria-describedby=/);
  assert.match(html, /Finish or stop this chat/);
  assert.match(html, /Working/);
  assert.match(html, /Saved work/);
  assert.match(html, /Suggestions <span>3<\/span>/);
  const history = renderToStaticMarkup(createElement(SessionSidebar, {
    currentSessionId: 'past', liveTitle: 'Current request', liveState: 'running', onSelectSession() {},
  }));
  assert.match(history, /Current request/);
  assert.doesNotMatch(history, /aria-current="page"/);
});

const deferred = () => {
  let resolve;
  const promise = new Promise(yes => { resolve = yes; });
  return { promise, resolve };
};
const response = data => new Response(JSON.stringify(data), { headers: { 'Content-Type': 'application/json' } });
const tick = () => new Promise(resolve => setImmediate(resolve));

function hookFixture(t, request, savedMessages = [{ id: 'saved', role: 'user', content: 'Saved draft', timestamp: 1 }]) {
  const session = new Map([['rune.live.sessionId', 'original-chat']]);
  const draft = new Map([['rune:web:live-state:v1', JSON.stringify({ version: 1, messages: savedMessages })]]);
  t.mock.method(globalThis, 'fetch', async (url, init) => url.endsWith('/auth/bootstrap') ? response({}) : request(url, init));
  const originalWindow = globalThis.window, originalStorage = globalThis.sessionStorage;
  globalThis.sessionStorage = { getItem: key => session.get(key) ?? null, setItem: (key, value) => session.set(key, value) };
  globalThis.window = { localStorage: { getItem: key => draft.get(key) ?? null, setItem: (key, value) => draft.set(key, value) }, dispatchEvent() {} };
  t.after(() => {
    if (originalWindow === undefined) delete globalThis.window; else globalThis.window = originalWindow;
    if (originalStorage === undefined) delete globalThis.sessionStorage; else globalThis.sessionStorage = originalStorage;
    api.resetWebAuth(); api.setCurrentRunId('');
  });
  api.resetWebAuth();
  let agent;
  function Harness() { agent = useAgent(); return null; }
  renderToStaticMarkup(createElement(Harness));
  return { agent, draft, savedDraft: draft.get('rune:web:live-state:v1') };
}

test('reset refuses in-flight submission, running and stopping without losing run identity or saved draft', async t => {
  const submitted = deferred(), stopped = deferred();
  const { agent, draft, savedDraft } = hookFixture(t, url => {
    if (url === '/api/message') return submitted.promise;
    if (url === '/api/abort') return stopped.promise;
    assert.fail(`Unexpected request: ${url}`);
  });
  const sending = agent.sendMessage('Continue working');
  assert.equal(agent.resetLiveConversation(), false);
  assert.equal(api.getLiveSessionId(), 'original-chat');
  assert.equal(draft.get('rune:web:live-state:v1'), savedDraft);
  submitted.resolve(response({ ok: true, runId: 'active-run' }));
  assert.equal(await sending, true);
  assert.equal(agent.resetLiveConversation(), false);
  assert.equal(api.getCurrentRunId(), 'active-run');
  agent.abort();
  assert.equal(agent.resetLiveConversation(), false);
  assert.equal(api.getCurrentRunId(), 'active-run');
  assert.equal(draft.get('rune:web:live-state:v1'), savedDraft);
  stopped.resolve(response({ stopped: true }));
  await tick();
  assert.equal(agent.resetLiveConversation(), true);
  assert.notEqual(api.getLiveSessionId(), 'original-chat');
  assert.equal(api.getCurrentRunId(), '');
});

test('reset preserves a chat waiting for approval or an answer', async t => {
  for (const waiting of ['approval', 'question']) {
    await t.test(waiting, async t => {
      const run = { runId: 'waiting-run', sessionId: 'waiting-chat', startedAt: 1, goal: 'Keep waiting',
        status: waiting === 'approval' ? 'waiting_approval' : 'waiting_input', text: '', stepNumber: 1, toolCalls: [], trust: null,
        approval: waiting === 'approval' ? { id: 'approval', expiresAt: Date.now() + 60_000 } : null,
        question: waiting === 'question' ? { id: 'question', expiresAt: Date.now() + 60_000 } : null };
      const { agent, draft } = hookFixture(t, url => {
        assert.equal(url, '/api/v1/rpc');
        return response({ success: true, data: { turns: [], run } });
      });
      assert.equal(await agent.loadConversation('waiting-chat'), true);
      const beforeReset = draft.get('rune:web:live-state:v1');
      assert.equal(agent.resetLiveConversation(), false);
      assert.equal(api.getLiveSessionId(), 'waiting-chat');
      assert.equal(api.getCurrentRunId(), 'waiting-run');
      assert.equal(draft.get('rune:web:live-state:v1'), beforeReset);
    });
  }
});
