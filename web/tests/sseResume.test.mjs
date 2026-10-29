import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { test } from 'node:test';

const source = ts.transpileModule(fs.readFileSync(new URL('../src/hooks/useSSE.ts', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;

function mount() {
  let cleanup, resume;
  const streams = [], calls = [];
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
    async ensureWebAuth() { calls.push('auth'); },
    resetWebAuth() { calls.push('reset-auth'); },
    setClientId() {}, getLiveSessionId: () => 'chat',
    async fetchRunSnapshot() { calls.push('snapshot'); return { run: { runId: 'r1' } }; },
  };
  const modules = {
    react: { useState: () => [null, () => {}], useRef: value => ({ current: value }),
      useEffect: effect => { cleanup = effect(); }, useCallback: fn => fn, useMemo: fn => fn() },
    '../api': api, '../types': { SSE_EVENT_TYPES: ['connected', 'text_delta'] },
    '../utils/runRecovery': { RunRecovery: class {
      constructor(emit) { this.emit = emit; }
      begin() {} finish() { return true; }
      receive(type, data) { this.emit(type, data); }
    } },
  };
  const exports = {};
  vm.runInNewContext(source, { exports, require: name => modules[name], window,
    EventSource: Stream, setTimeout, clearTimeout });
  const hook = exports.useSSE();
  return { hook, streams, calls, window, resume: () => resume?.(), unmount: () => cleanup() };
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
