import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

let server, RunRecovery, restoreRunMessages, abortedMessage, upsertRunMessage, belongsToConversation, api;
before(async () => {
  server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ RunRecovery } = await server.ssrLoadModule('/src/utils/runRecovery.ts'));
  ({ restoreRunMessages } = await server.ssrLoadModule('/src/utils/runSnapshot.ts'));
  ({ abortedMessage, upsertRunMessage, belongsToConversation } = await server.ssrLoadModule('/src/utils/runEvents.ts'));
  api = await server.ssrLoadModule('/src/api.ts');
});
after(async () => { await server?.close(); });

test('snapshot and in-flight events converge without duplicate text or stale questions', () => {
  const received = [];
  const recovery = new RunRecovery((type, data) => received.push([type, data]));
  recovery.begin();
  recovery.receive('text_delta', { runId: 'r1', seq: 2, delta: 'old' });
  recovery.receive('question_closed', { runId: 'r1', seq: 4, id: 'q1' });
  const run = { runId: 'r1', seq: 3, question: { id: 'q1' } };
  assert.equal(recovery.finish(run), true);
  recovery.receive('question', { runId: 'r1', seq: 3, id: 'q1' });
  recovery.receive('question_closed', { runId: 'r1', seq: 4, id: 'q1' });
  assert.deepEqual(received.map(([type]) => type), ['run_snapshot', 'question_closed']);
});

test('overflow requires a newer snapshot before buffered events are applied', () => {
  const received = [];
  const recovery = new RunRecovery((...event) => received.push(event));
  recovery.begin();
  for (let seq = 1; seq <= 513; seq++) recovery.receive('text_delta', { runId: 'r1', seq, delta: 'x' });
  assert.equal(recovery.finish({ runId: 'r1', seq: 2 }), false);
  assert.equal(received.length, 0);
  recovery.begin();
  assert.equal(recovery.finish({ runId: 'r1', seq: 513 }), true);
  assert.equal(received.length, 1);
});

test('an older daemon keeps delivering events without an empty snapshot reset', () => {
  const received = [];
  const recovery = new RunRecovery((...event) => received.push(event));
  recovery.begin();
  recovery.receive('question', { id: 'q1' });
  assert.equal(recovery.finish(undefined), true);
  assert.deepEqual(received, [['question', { id: 'q1' }]]);
});

test('restoring a finished run replaces its partial answer and preserves earlier turns', () => {
  const earlier = { id: 'old', role: 'assistant', content: 'previous answer', timestamp: 0 };
  const messages = [earlier, { id: 'user', role: 'user', content: 'report', timestamp: 1 },
    { id: 'partial', role: 'assistant', content: 'unfinished', timestamp: 2 }];
  const run = { runId: 'r1', goal: 'report', text: 'unfinished', answer: 'final report',
    startedAt: 1, updatedAt: 3, trust: null };
  const restored = restoreRunMessages(messages, run);
  assert.deepEqual(restored.map(message => message.content), ['previous answer', 'report', 'final report']);
  assert.deepEqual(restoreRunMessages(restored, run), restored);
  assert.deepEqual(restored[0], earlier);
  assert.deepEqual(restoreRunMessages([], { ...run, history: messages }), restored);
});

test('a late cancellation receipt updates the restored stop card', () => {
  const run = { runId: 'r1', status: 'cancelled', goal: 'report', text: '', startedAt: 1,
    trust: { reason: 'cancelled', completionStatus: 'cancelled', verified: false } };
  const restored = restoreRunMessages([], run);
  const trust = { ...run.trust, artifactReceipts: [{ revision: 'r2' }] };
  const updated = upsertRunMessage(restored, abortedMessage({ runId: 'r1', trust }, '', 2));
  assert.equal(updated.length, restored.length);
  assert.deepEqual(updated[1].trust, trust);
});

test('interrupted recovery preserves recorded input and adds one stop card', () => {
  const run = { runId: 'r1', status: 'interrupted', goal: 'report', text: 'Saved progress', startedAt: 1,
    trust: { reason: 'server_restart', completionStatus: 'interrupted', verified: false },
    interactions: [{ id: 'q1', kind: 'question', request: { question: '거래 범위?' },
      status: 'answered', response: { answer: '확정 거래만 🧾' } }], question: null, approval: null };
  const restored = restoreRunMessages([], run);
  assert.deepEqual(restoreRunMessages(restored, run), restored);
  assert.equal(restored.filter(message => message.trust).length, 1);
  assert.ok(restored.some(message => message.content.includes('확정 거래만 🧾')));
  assert.ok(restored.some(message => message.content === 'Saved progress'));
});

test('a stale run ID cannot admit events from a different conversation', () => {
  assert.equal(belongsToConversation({ sessionId: 'old', runId: 'r1' }, 'new', 'r1'), false);
  assert.equal(belongsToConversation({ sessionId: 'new', runId: 'r2' }, 'new', ''), true);
  assert.equal(belongsToConversation({ runId: 'r1' }, 'new', ''), false);
  assert.equal(belongsToConversation({ runId: 'r2' }, 'new', 'r2'), true);
  assert.equal(belongsToConversation({}, 'new', ''), true);
});

test('a delayed message acknowledgement cannot reattach a cleared conversation', async t => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'sessionStorage');
  const storage = new Map();
  Object.defineProperty(globalThis, 'sessionStorage', {
    configurable: true, value: { getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) },
  });
  t.after(() => {
    if (descriptor) Object.defineProperty(globalThis, 'sessionStorage', descriptor);
    else delete globalThis.sessionStorage;
  });
  let accept;
  const pending = new Promise(resolve => { accept = resolve; });
  let sent;
  api.resetWebAuth();
  t.mock.method(globalThis, 'fetch', async (path, options) => {
    if (path === '/api/message') {
      sent = JSON.parse(options.body);
      return pending;
    }
    return Response.json({});
  });
  await api.ensureWebAuth();
  api.setLiveSessionId('old');
  api.setCurrentRunId('previous');
  const request = api.sendMessage('report');
  await Promise.resolve();
  assert.equal(sent.sessionId, 'old');
  api.rotateLiveSessionId();
  assert.equal(api.getCurrentRunId(), '');
  accept(Response.json({ ok: true, runId: 'old-run' }));
  await request;
  assert.equal(api.getCurrentRunId(), '');
  api.setCurrentRunId('another-run');
  api.setLiveSessionId('loaded-conversation');
  assert.equal(api.getCurrentRunId(), '');
});

test('question and approval retries retain the accepted response ID', async t => {
  api.resetWebAuth();
  for (const [path, send] of [
    ['/api/question', () => api.sendQuestion('lost-question', '확정 거래만 & 환불 제외', 0)],
    ['/api/approval', () => api.sendApproval('lost-approval', 'approve_once')],
  ]) {
    const bodies = [];
    const mock = t.mock.method(globalThis, 'fetch', async (url, options) => {
      if (url === '/api/v1/auth/bootstrap') return Response.json({});
      assert.equal(url, path);
      bodies.push(JSON.parse(options.body));
      return bodies.length === 1
        ? Response.json({ error: 'response lost' }, { status: 503 })
        : Response.json({ ok: true });
    });
    await assert.rejects(send(), /response lost/);
    await send();
    assert.ok(bodies[0].responseId);
    assert.deepEqual(bodies[0], bodies[1]);
    mock.mock.restore();
  }
});
