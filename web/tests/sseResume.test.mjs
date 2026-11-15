import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { test } from 'node:test';

const source = ts.transpileModule(fs.readFileSync(new URL('../src/hooks/useSSE.ts', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;

function mount(options = {}) {
  let cleanup, resume;
  const streams = [], calls = [], states = [];
  const timers = new Map();
  let stateIndex = 0, timerId = 0;
  const window = new EventTarget();
  window.rune = { onResume(callback) { resume = callback; return () => { resume = null; }; } };
  class Stream {
    static CLOSED = 2;
    constructor() { this.listeners = new Map(); streams.push(this); }
    addEventListener(type, callback) { this.listeners.set(type, callback); }
    close() { this.closed = true; }
    emit(type, value) { this.listeners.get(type)?.({ data: JSON.stringify(value) }); }
  }
  const api = {
    async ensureWebAuth() { calls.push('auth'); return options.authenticate?.(); },
    AUTH_EVENT: 'rune:auth-required', eventStreamUrl: () => '/api/events',
    isStaleAuthFailure: () => false,
    authFailure: error => error?.status === 401 ? 'required' : error?.status === 403 ? 'forbidden' : null,
    resetWebAuth() { calls.push('reset-auth'); },
    requireWebAuth() { calls.push('require-auth'); },
    setClientId() {}, getLiveSessionId: () => 'chat',
    async fetchRunSnapshot() { calls.push('snapshot'); return options.snapshot ? options.snapshot() : { run: { runId: 'r1' } }; },
  };
  const modules = {
    react: { useState: value => { const index = stateIndex++; states[index] = value; return [value, next => { states[index] = next; }]; }, useRef: value => ({ current: value }),
      useEffect: effect => { cleanup = effect(); }, useCallback: fn => fn, useMemo: fn => fn() },
    '../api': api, '../types': { SSE_EVENT_TYPES: ['connected', 'text_delta'] },
    '../utils/presence': { MACHINE_EVENT: 'rune:machine' },
    '../utils/runRecovery': { RunRecovery: class {
      constructor(emit) { this.emit = emit; }
      begin() {} finish() { return true; }
      receive(type, data) { this.emit(type, data); }
    } },
  };
  const exports = {};
  vm.runInNewContext(source, { exports, require: name => modules[name], window,
    EventSource: Stream, setTimeout: fn => { timers.set(++timerId, fn); return timerId; }, clearTimeout: id => timers.delete(id) });
  const hook = exports.useSSE();
  return { hook, streams, calls, window, states, timers, tick: () => { const pending = [...timers.values()]; timers.clear(); pending.forEach(fn => fn()); }, resume: () => resume?.(), unmount: () => cleanup() };
}

const flush = () => new Promise(resolve => setImmediate(resolve));

test('resume reconnects and restores once without accepting stale stream events', async () => {
  const app = mount();
  try {
    await flush();
    const received = [];
    app.hook.addEventListener('text_delta', value => received.push(value));
    app.streams[0].emit('connected', { clientId: 'old' });
    await flush();
    app.resume();
    await flush();
    assert.equal(app.streams.length, 2);
    assert.equal(app.streams[0].closed, true);
    app.streams[0].emit('connected', { clientId: 'old' });
    app.streams[0].emit('text_delta', { delta: 'stale' });
    app.streams[1].emit('connected', { clientId: 'new' });
    app.streams[1].emit('text_delta', { delta: 'current' });
    await flush();
    assert.deepEqual(received.map(value => value.delta), ['current']);
    assert.equal(app.calls.filter(call => call === 'snapshot').length, 2);
    assert.ok(app.calls.every(call => ['auth', 'reset-auth', 'snapshot'].includes(call)));
  } finally { app.unmount(); }
});

test('overlapping wake and network events create one stream and detach on unmount', async () => {
  const app = mount();
  await flush();
  app.resume();
  app.window.dispatchEvent(new Event('online'));
  await flush();
  assert.equal(app.streams.length, 2);
  app.unmount();
  app.resume();
  app.window.dispatchEvent(new Event('online'));
  await flush();
  assert.equal(app.streams.length, 2);
  assert.equal(app.streams[1].closed, true);
});

test('a hosted machine waking up reconnects at once, and only a dropped stream', async () => {
  const app = mount();
  try {
    await flush();
    app.window.dispatchEvent(new CustomEvent('rune:machine', { detail: 'running' }));
    await flush();
    assert.equal(app.streams.length, 1, 'an open stream is left alone');
    app.streams[0].readyState = 2;
    app.streams[0].onerror();
    app.window.dispatchEvent(new CustomEvent('rune:machine', { detail: 'suspended' }));
    await flush();
    assert.equal(app.streams.length, 1, 'still asleep: wait for the retry');
    app.window.dispatchEvent(new CustomEvent('rune:machine', { detail: 'running' }));
    await flush();
    assert.equal(app.streams.length, 2);
  } finally { app.unmount(); }
});

for (const status of [401, 403]) {
  test(`bootstrap ${status} stops retries until an explicit reconnect`, async () => {
    let expired = true;
    const app = mount({ authenticate() { if (expired) throw { status }; } });
    try {
      await flush();
      assert.equal(app.states[0], status === 401 ? 'required' : 'forbidden');
      assert.equal(app.streams.length, 0);
      assert.equal(app.timers.size, 0);
      app.resume();
      app.window.dispatchEvent(new Event('online'));
      app.tick();
      await flush();
      assert.equal(app.calls.filter(call => call === 'auth').length, 1);
      expired = false;
      app.hook.reconnect();
      await flush();
      app.streams[0].emit('connected', { clientId: 'signed-in' });
      await flush();
      assert.equal(app.states[0], null);
      assert.equal(app.calls.filter(call => call === 'snapshot').length, 1);
    } finally { app.unmount(); }
  });
}

test('transient failures still reconnect while auth events stop an existing stream', async () => {
  let offline = true;
  const app = mount({ authenticate() { if (offline) throw new Error('offline'); } });
  try {
    await flush();
    assert.equal(app.timers.size, 1);
    offline = false;
    app.tick();
    await flush();
    assert.equal(app.streams.length, 1);
    app.window.dispatchEvent(new CustomEvent('rune:auth-required', { detail: 'required' }));
    assert.equal(app.streams[0].closed, true);
    assert.equal(app.timers.size, 0);
    assert.equal(app.states[0], 'required');
  } finally { app.unmount(); }
});

test('failed snapshot detaches the stream so a machine wake can recover immediately', async () => {
  let fail = true;
  const app = mount({ snapshot() { if (fail) throw new Error('asleep'); return { run: null }; } });
  try {
    await flush();
    app.streams[0].emit('connected', { clientId: 'old' });
    await flush();
    assert.equal(app.streams[0].closed, true);
    fail = false;
    app.window.dispatchEvent(new CustomEvent('rune:machine', { detail: 'running' }));
    await flush();
    assert.equal(app.streams.length, 2);
  } finally { app.unmount(); }
});


for (const status of [401, 403]) {
  test(`snapshot ${status} stops after at most one authentication refresh`, async () => {
    const app = mount({ snapshot() { throw { status }; } });
    try {
      await flush();
      app.streams[0].emit('connected', { clientId: 'first' });
      await flush();
      if (status === 401) {
        app.tick();
        await flush();
        app.streams[1].emit('connected', { clientId: 'second' });
        await flush();
      }
      assert.equal(app.states[0], status === 401 ? 'required' : 'forbidden');
      assert.equal(app.timers.size, 0);
      assert.equal(app.streams.length, status === 401 ? 2 : 1);
      const requests = app.calls.length;
      app.hook.refresh();
      app.resume();
      app.tick();
      await flush();
      assert.equal(app.calls.length, requests);
    } finally { app.unmount(); }
  });
}


for (const status of [401, 403]) {
  test(`a stream rejected before connected stops on a protected read ${status}`, async () => {
    const app = mount({ snapshot() { throw { status }; } });
    try {
      await flush();
      app.streams[0].readyState = 2;
      app.streams[0].onerror();
      app.streams[0].onerror();
      await flush();
      assert.equal(app.calls.filter(call => call === 'snapshot').length, 1);
      assert.equal(app.calls.filter(call => call === 'require-auth').length, status === 401 ? 1 : 0);
      assert.equal(app.states[0], status === 401 ? 'required' : 'forbidden');
      assert.equal(app.timers.size, 0);
      app.resume();
      app.tick();
      await flush();
      assert.equal(app.streams.length, 1);
    } finally { app.unmount(); }
  });
}

test('a healthy stream does not add an authentication probe', async () => {
  const app = mount();
  try {
    await flush();
    assert.deepEqual(app.calls, ['auth']);
    app.streams[0].emit('connected', { clientId: 'healthy' });
    await flush();
    assert.deepEqual(app.calls, ['auth', 'snapshot']);
    app.streams[0].readyState = 2;
    app.streams[0].onerror();
    await flush();
    assert.equal(app.calls.filter(call => call === 'snapshot').length, 1);
    assert.equal(app.timers.size, 1);
  } finally { app.unmount(); }
});

test('a temporary rejected stream still retries', async () => {
  const app = mount({ snapshot() { throw { status: 503 }; } });
  try {
    await flush();
    app.streams[0].readyState = 2;
    app.streams[0].onerror();
    await flush();
    assert.equal(app.states[0], null);
    assert.equal(app.timers.size, 1);
    app.tick();
    await flush();
    assert.equal(app.streams.length, 2);
    assert.equal(app.calls.filter(call => call === 'require-auth').length, 0);
  } finally { app.unmount(); }
});

test('a slow rejected-stream probe cannot block a newer working connection', async () => {
  let reject, reads = 0;
  const app = mount({ snapshot() {
    return ++reads === 1 ? new Promise((_, fail) => { reject = fail; }) : { run: null };
  } });
  try {
    await flush();
    app.streams[0].readyState = 2;
    app.streams[0].onerror();
    app.hook.reconnect();
    await flush();
    app.streams[1].emit('connected', { clientId: 'new' });
    await flush();
    reject({ status: 401 });
    await flush();
    assert.equal(app.states[0], null);
    assert.equal(app.streams[1].closed, undefined);
    assert.equal(app.calls.filter(call => call === 'require-auth').length, 0);
    assert.equal(app.timers.size, 0);
  } finally { app.unmount(); }
});
