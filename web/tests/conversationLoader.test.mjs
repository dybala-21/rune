import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

let server, ConversationLoader;
before(async () => {
  server = await createServer({
    root: fileURLToPath(new URL('..', import.meta.url)),
    server: { middlewareMode: true, hmr: false, ws: false, watch: null }, appType: 'custom',
    optimizeDeps: { noDiscovery: true, include: [] },
  });
  ({ ConversationLoader } = await server.ssrLoadModule('/src/utils/conversationLoader.ts'));
});
after(async () => { await server?.close(); });
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
const contents = { turns: [{ role: 'user', content: 'Earlier request', timestamp: '2026-09-26' }], run: null };
function fixture() {
  const loader = new ConversationLoader();
  const response = deferred();
  let session = 'live', busy = false, signal;
  const activations = [];
  const options = {
    read: (_, s) => { signal = s; return response.promise; },
    currentSession: () => session,
    isBusy: () => busy,
    activate: (id, data) => { session = id; activations.push({ id, data }); },
  };
  return { loader, response, options, activations, getSession: () => session,
    setSession: id => { session = id; }, setBusy: value => { busy = value; }, getSignal: () => signal };
}

test('a continuation cannot activate before its history response arrives', async () => {
  const f = fixture();
  const result = f.loader.load('past', f.options);
  await Promise.resolve();
  assert.equal(f.getSession(), 'live');
  assert.equal(f.loader.pending, true);
  assert.deepEqual(f.activations, []);
  f.response.resolve(contents);
  assert.equal(await result, true);
  assert.deepEqual(f.activations, [{ id: 'past', data: contents }]);
  assert.equal(f.loader.pending, false);
});

test('leaving history cancels an in-flight continuation even if fetch ignores abort', async () => {
  const f = fixture();
  const result = f.loader.load('past', f.options);
  f.loader.cancel();
  assert.equal(f.getSignal().aborted, true);
  f.response.resolve(contents);
  assert.equal(await result, false);
  assert.deepEqual(f.activations, []);
  assert.equal(f.getSession(), 'live');
});

test('a slow older response cannot replace the later-selected conversation', async () => {
  const f = fixture();
  const first = f.loader.load('A', f.options);
  const next = deferred();
  const second = f.loader.load('B', { ...f.options, read: () => next.promise });
  f.response.resolve(contents);
  assert.equal(await first, false);
  assert.equal(f.loader.pending, true);
  next.resolve(contents);
  assert.equal(await second, true);
  assert.deepEqual(f.activations.map(a => a.id), ['B']);
});

test('failure leaves the current conversation unchanged and permits retry', async () => {
  const f = fixture();
  const first = f.loader.load('past', f.options);
  f.response.reject(new Error('Session not found'));
  await assert.rejects(first, /Session not found/);
  assert.equal(f.getSession(), 'live');
  assert.equal(f.loader.pending, false);
  assert.equal(await f.loader.load('past', { ...f.options, read: async () => contents }), true);
});

test('a new live conversation invalidates pending activation', async () => {
  const f = fixture();
  const result = f.loader.load('past', f.options);
  f.setSession('new-chat');
  f.response.resolve(contents);
  assert.equal(await result, false);
  assert.deepEqual(f.activations, []);
});

test('a run that starts during loading prevents switching away', async () => {
  const f = fixture();
  const result = f.loader.load('past', f.options);
  f.setBusy(true);
  f.response.resolve(contents);
  await assert.rejects(result, /busy/);
  assert.equal(f.getSession(), 'live');
});

test('a busy live conversation is rejected before fetching history', async () => {
  const f = fixture();
  f.setBusy(true);
  await assert.rejects(f.loader.load('past', f.options), /current task/);
  assert.equal(f.getSignal(), undefined);
});

test('an active destination may be opened but cannot receive another task', async () => {
  for (const status of ['queued', 'running', 'waiting_input', 'waiting_approval']) {
    const f = fixture();
    const active = { ...contents, run: { status } };
    await assert.rejects(f.loader.load('past', { ...f.options, read: async () => active, forMessage: true }), /already has a running task/);
    assert.deepEqual(f.activations, []);
    assert.equal(await f.loader.load('past', { ...f.options, read: async () => active }), true);
  }
});
