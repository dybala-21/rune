import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { test } from 'node:test';

const compile = name => ts.transpileModule(fs.readFileSync(new URL(`../src/${name}.ts`, import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const apiSource = compile('api');
const scopeSource = compile('utils/sessionContext');

function client(send, saved = new Map()) {
  const scope = {};
  vm.runInNewContext(scopeSource, { exports: scope });
  const window = new EventTarget();
  const navigations = [];
  window.location = { assign: value => navigations.push(value) };
  const exports = {};
  vm.runInNewContext(apiSource, {
    exports, require: name => name === './utils/sessionContext' ? scope : { presenceHeaders: () => ({}), noteResponse() {} },
    window, fetch: send, Headers, AbortSignal, DOMException, CustomEvent, crypto,
    sessionStorage: { getItem: key => saved.get(key) ?? null, setItem: (key, value) => saved.set(key, value) },
  });
  return { api: exports, window, navigations, saved };
}
const ok = () => Response.json({ storageScope: 'a'.repeat(64) });

test('expired auth is reported once and polling cannot keep requesting credentials', async () => {
  let calls = 0;
  const { api, window } = client(() => { calls++; return Promise.resolve(Response.json({ detail: 'Sign in' }, { status: 401 })); });
  const events = [];
  window.addEventListener(api.AUTH_EVENT, event => events.push(event.detail));
  await assert.rejects(api.ensureWebAuth(), error => error.status === 401);
  for (let i = 0; i < 20; i++) await assert.rejects(api.ensureWebAuth());
  assert.equal(calls, 1);
  assert.deepEqual(events, ['required']);
});

test('forbidden mutation is never replayed or reclassified as an expired login', async () => {
  const calls = [];
  const { api } = client(async url => {
    calls.push(url);
    return url.endsWith('/bootstrap') ? ok() : Response.json({ detail: 'Policy denied' }, { status: 403 });
  });
  await assert.rejects(api.post('/api/message', { text: 'send' }), error => error.status === 403);
  assert.deepEqual(calls, ['/api/v1/auth/bootstrap', '/api/message']);
});

test('interactive sign-in never replays a rejected message', async () => {
  let bootstrap = 0, messages = 0, signedIn = false;
  const { api } = client(async url => {
    if (url.endsWith('/bootstrap')) return ++bootstrap === 1 || signedIn ? ok() : Response.json({}, { status: 401 });
    messages++;
    return Response.json({}, { status: 401 });
  });
  await assert.rejects(api.post('/api/message', { text: 'only once' }));
  assert.equal(messages, 1);
  signedIn = true;
  api.resetWebAuth();
  await api.ensureWebAuth();
  assert.equal(messages, 1);
});

test('an unauthenticated message is retried once after a successful legacy bootstrap', async () => {
  let messages = 0, bootstrap = 0;
  const { api } = client(async url => {
    if (url.endsWith('/bootstrap')) { bootstrap++; return new Response('', { status: 404 }); }
    return ++messages === 1 ? Response.json({}, { status: 401 }) : Response.json({ ok: true });
  });
  assert.equal((await api.post('/api/message', { requestId: 'stable' })).ok, true);
  assert.equal(messages, 2);
  assert.equal(bootstrap, 2);
});

test('an older bootstrap cannot overwrite a newer account context', async () => {
  let release, calls = 0;
  const { api } = client(async () => ++calls === 1 ? new Promise(resolve => { release = resolve; }) : ok());
  const old = api.ensureWebAuth();
  api.resetWebAuth();
  await api.ensureWebAuth();
  release(Response.json({ storageScope: 'b'.repeat(64) }));
  await assert.rejects(old, error => error.name === 'AbortError');
  await api.ensureWebAuth();
  assert.equal(calls, 2);
});

test('same-account reload keeps the conversation; another account starts separately', async () => {
  const saved = new Map();
  const first = client(async () => ok(), saved);
  await first.api.ensureWebAuth();
  first.api.setLiveSessionId('conversation-a');
  const same = client(async () => ok(), saved);
  await same.api.ensureWebAuth();
  assert.equal(same.api.getLiveSessionId(), 'conversation-a');
  const other = client(async () => Response.json({ storageScope: 'b'.repeat(64) }), saved);
  await other.api.ensureWebAuth();
  assert.notEqual(other.api.getLiveSessionId(), 'conversation-a');
});

test('an in-page account change reloads before a new mutation can leave the browser', async () => {
  let account = 'a', messages = 0;
  const { api, navigations } = client(async url => {
    if (url.endsWith('/bootstrap')) return Response.json({ storageScope: account.repeat(64) });
    messages++;
    return Response.json({});
  });
  await api.ensureWebAuth();
  account = 'b';
  api.resetWebAuth();
  await assert.rejects(api.post('/api/message', { text: 'old account draft' }));
  assert.equal(messages, 0);
  assert.deepEqual(navigations, ['/']);
});

test('successful auth is shared and cached across all subsequent API calls', async () => {
  let calls = 0;
  const { api } = client(async () => { calls++; return ok(); });
  await Promise.all(Array.from({ length: 100 }, () => api.ensureWebAuth()));
  for (let i = 0; i < 1000; i++) await api.ensureWebAuth();
  assert.equal(calls, 1);
});


test('API requests and event streams carry the bound account without another bootstrap', async () => {
  const seen = [];
  const { api } = client(async (url, options) => {
    seen.push([url, options.headers.get('X-Rune-Account')]);
    return url.endsWith('/bootstrap') ? ok() : Response.json({});
  });
  await api.ensureWebAuth();
  await api.post('/api/message', {});
  assert.equal(seen.length, 2);
  assert.equal(seen[0][1], null);
  assert.equal(seen[1][1], 'a'.repeat(64));
  assert.equal(api.eventStreamUrl(), `/api/events?accountScope=${'a'.repeat(64)}`);
});


test('a protected-read authentication failure also stops background API polling', async () => {
  let calls = 0;
  const { api, window } = client(async () => { calls++; return ok(); });
  await api.ensureWebAuth();
  const events = [];
  window.addEventListener(api.AUTH_EVENT, event => events.push(event.detail));
  api.requireWebAuth();
  api.requireWebAuth();
  for (let i = 0; i < 10; i++) await assert.rejects(api.post('/api/v1/rpc', {}), error => error.status === 401);
  assert.equal(calls, 1);
  assert.deepEqual(events, ['required']);
  api.resetWebAuth();
  await api.ensureWebAuth();
  assert.equal(calls, 2);
});

test('an authentication rejection invalidates a bootstrap still in flight', async () => {
  let resolve, calls = 0;
  const { api } = client(() => { calls++; return new Promise(yes => { resolve = yes; }); });
  const pending = api.ensureWebAuth();
  api.requireWebAuth();
  resolve(ok());
  await assert.rejects(pending, error => error.name === 'AbortError');
  await assert.rejects(api.ensureWebAuth(), error => error.status === 401);
  assert.equal(calls, 1);
});

for (const status of [404, 405]) {
  test(`a local daemon bootstrap ${status} preserves its legacy conversation`, async () => {
    const saved = new Map([['rune.live.sessionId', 'local-chat']]);
    const { api } = client(async () => new Response('', { status }), saved);
    await api.ensureWebAuth();
    assert.equal(api.getLiveSessionId(), 'local-chat');
    assert.equal(api.eventStreamUrl(), '/api/events');
  });

  test(`a hosted bootstrap ${status} cannot downgrade the account to local storage`, async () => {
    let missing = false;
    const { api, navigations } = client(async () => missing ? new Response('', { status }) : ok());
    await api.ensureWebAuth();
    api.setLiveSessionId('hosted-chat');
    missing = true;
    api.resetWebAuth();
    await assert.rejects(api.ensureWebAuth(), error => error.status === 503);
    assert.deepEqual(navigations, []);
    assert.equal(api.getLiveSessionId(), 'hosted-chat');
    missing = false;
    await api.ensureWebAuth();
    assert.equal(api.getLiveSessionId(), 'hosted-chat');
  });
}

test('a missing bootstrap tagged by a hosted gateway never restores local history', async () => {
  const { api } = client(async () => new Response('', { status: 404, headers: { 'X-Rune-Machine': 'running' } }));
  await assert.rejects(api.ensureWebAuth(), error => error.status === 503);
});


for (const refreshed of [true, false]) {
  test(`concurrent unauthorized requests share one ${refreshed ? 'successful' : 'rejected'} refresh`, async () => {
    let bootstraps = 0;
    const requests = new Map();
    const { api } = client(async url => {
      if (url.endsWith('/bootstrap')) {
        bootstraps++;
        return bootstraps === 1 || refreshed ? ok() : Response.json({}, { status: 401 });
      }
      const count = (requests.get(url) ?? 0) + 1;
      requests.set(url, count);
      return count === 1 ? Response.json({}, { status: 401 }) : Response.json({ ok: true });
    });
    const outcomes = await Promise.allSettled(Array.from({ length: 5 }, (_, index) => api.post(`/api/check/${index}`, {})));
    assert.equal(bootstraps, 2);
    assert.ok(outcomes.every(outcome => outcome.status === (refreshed ? 'fulfilled' : 'rejected')));
    assert.ok([...requests.values()].every(count => count === (refreshed ? 2 : 1)));
    if (!refreshed) {
      for (const outcome of outcomes) assert.equal(outcome.reason.status, 401);
      await assert.rejects(api.ensureWebAuth(), error => error.status === 401);
      assert.equal(bootstraps, 2);
    }
  });
}


const streamSource = compile('hooks/useSSE');
const flush = () => new Promise(resolve => setImmediate(resolve));

function mountStream(api, window) {
  const states = [], streams = [], timers = new Map();
  let stateIndex = 0, cleanup;
  class Stream {
    static CLOSED = 2;
    constructor() { this.listeners = new Map(); streams.push(this); }
    addEventListener(type, listener) { this.listeners.set(type, listener); }
    close() { this.closed = true; }
    emit(type, data) { this.listeners.get(type)?.({ data: JSON.stringify(data) }); }
  }
  const modules = {
    react: { useState(value) { const index = stateIndex++; states[index] = value; return [value, next => { states[index] = next; }]; },
      useRef: current => ({ current }), useEffect: effect => { cleanup = effect(); }, useCallback: fn => fn, useMemo: fn => fn() },
    '../api': api, '../types': { SSE_EVENT_TYPES: ['connected'] },
    '../utils/presence': { MACHINE_EVENT: 'rune:machine' },
    '../utils/runRecovery': { RunRecovery: class { begin() {} finish() { return true; } receive() {} } },
  };
  const exports = {};
  vm.runInNewContext(streamSource, { exports, require: name => modules[name], window, EventSource: Stream,
    setTimeout(callback) { const id = Symbol(); timers.set(id, callback); return id; }, clearTimeout: id => timers.delete(id) });
  exports.useSSE();
  return { states, streams, timers, tick() { const callbacks = [...timers.values()]; timers.clear(); callbacks.forEach(callback => callback()); },
    stop() { cleanup(); } };
}

for (const path of ['restore', 'rejected-stream']) {
  for (const status of [401, 403]) {
    test(`a stale ${path} ${status} cannot undo a parallel successful authentication refresh`, async () => {
      let bootstraps = 0, messages = 0, snapshots = 0, finishRead;
      const { api, window } = client(async url => {
        if (url.endsWith('/bootstrap')) { bootstraps++; return ok(); }
        if (url.startsWith('/api/runs/snapshot')) {
          return ++snapshots === 1 ? new Promise(resolve => { finishRead = resolve; }) : Response.json({ run: null });
        }
        if (url === '/api/message') return ++messages === 1 ? Response.json({}, { status: 401 }) : Response.json({ ok: true });
        throw new Error(`Unexpected request: ${url}`);
      });
      const app = mountStream(api, window);
      try {
        await flush();
        if (path === 'restore') app.streams[0].emit('connected', { clientId: 'current' });
        else { app.streams[0].readyState = 2; app.streams[0].onerror(); }
        await flush();
        assert.equal(snapshots, 1);
        assert.equal((await api.post('/api/message', {})).ok, true);
        assert.equal(bootstraps, 2);
        finishRead(Response.json({}, { status }));
        await flush();
        assert.equal(app.states[0], null);
        await api.ensureWebAuth();
        assert.equal(bootstraps, 2, 'the late denial must not discard the refreshed authentication');
        if (path === 'rejected-stream') {
          assert.equal(app.timers.size, 1);
          app.tick();
          await flush();
          app.streams[1].emit('connected', { clientId: 'reconnected' });
          await flush();
        } else assert.equal(app.streams[0].closed, undefined);
        assert.equal(snapshots, 2);
        assert.equal(bootstraps, 2);
        assert.equal(app.states[0], null);
      } finally { app.stop(); }
    });
  }
}

for (const status of [401, 403]) {
  test(`a current rejected-stream ${status} still enforces its authentication boundary`, async () => {
    let bootstraps = 0;
    const { api, window } = client(async url => {
      if (url.endsWith('/bootstrap')) { bootstraps++; return ok(); }
      return Response.json({}, { status });
    });
    const app = mountStream(api, window);
    try {
      await flush();
      app.streams[0].readyState = 2;
      app.streams[0].onerror();
      await flush();
      assert.equal(app.states[0], status === 401 ? 'required' : 'forbidden');
      if (status === 401) await assert.rejects(api.ensureWebAuth(), error => error.status === 401);
      else await api.ensureWebAuth();
      assert.equal(bootstraps, 1);
      assert.equal(app.timers.size, 0);
    } finally { app.stop(); }
  });
}
