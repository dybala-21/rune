import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { test } from 'node:test';

const source = ts.transpileModule(fs.readFileSync(new URL('../src/components/SessionBoundary.tsx', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX },
}).outputText;
const flush = () => new Promise(resolve => setImmediate(resolve));
const deferred = () => {
  let resolve;
  const promise = new Promise(yes => { resolve = yes; });
  return { promise, resolve };
};
function mount(outcomes) {
  const slots = [], timers = new Map();
  let cursor = 0, calls = 0, resets = 0, output, stopped = false;
  const children = { type: 'private-app' };
  const effects = [];
  const modules = {
    react: {
      useState(initial) {
        const i = cursor++;
        if (!(i in slots)) slots[i] = initial;
        return [slots[i], value => { slots[i] = typeof value === 'function' ? value(slots[i]) : value; render(); }];
      },
      useRef(initial) { const i = cursor++; return slots[i] ??= { current: initial }; },
      useEffect(effect, dependencies) {
        const i = cursor++, old = slots[i];
        if (old && old.dependencies.every((value, n) => value === dependencies[n])) return;
        effects.push(() => { old?.cleanup?.(); slots[i] = { dependencies, cleanup: effect() }; });
      },
    },
    'react/jsx-runtime': { jsx: (type, props) => ({ type, props }) },
    '../api': {
      ensureWebAuth() { calls++; return Promise.resolve().then(() => outcomes.shift()()); },
      resetWebAuth() { resets++; },
      authFailure(error) { return error?.status === 401 ? 'required' : error?.status === 403 ? 'forbidden' : null; },
    },
    './ConnectionBanner': { ConnectionBanner: 'connection-banner' },
  };
  const exports = {};
  vm.runInNewContext(source, { exports, require: name => modules[name],
    setTimeout(callback, delay) { const id = Symbol(); timers.set(id, { callback, delay }); return id; },
    clearTimeout(id) { timers.delete(id); } });
  function render() {
    if (stopped) return;
    cursor = 0;
    output = exports.SessionBoundary({ children });
    while (effects.length) effects.shift()();
  }
  render();
  return { get output() { return output; }, get calls() { return calls; }, get resets() { return resets; }, children, timers,
    tick() { const [id, timer] = timers.entries().next().value; timers.delete(id); timer.callback(); },
    stop() { stopped = true; for (const slot of slots) slot?.cleanup?.(); } };
}

test('private content mounts only after bootstrap; the pending screen is neutral', async () => {
  const pending = deferred();
  const app = mount([() => pending.promise]);
  assert.equal(app.output.props.role, 'status');
  assert.equal(app.output.props.children, 'Connecting to Rune…');
  await flush();
  assert.notEqual(app.output, app.children);
  pending.resolve();
  await flush();
  assert.equal(app.output, app.children);
  assert.equal(app.calls, 1);
  app.stop();
});

test('authentication failures stop retries until the user explicitly reconnects', async () => {
  for (const status of [401, 403]) {
    const app = mount([() => { throw { status }; }, () => undefined]);
    await flush();
    assert.equal(app.output.props.authFailure, status === 401 ? 'required' : 'forbidden');
    assert.equal(app.timers.size, 0);
    app.output.props.onRetry();
    await flush();
    assert.equal(app.calls, 2);
    assert.equal(app.resets, 1);
    assert.equal(app.output, app.children);
    app.stop();
  }
});

test('temporary failures back off and cleanup prevents a stale bootstrap from mounting', async () => {
  const pending = deferred();
  const app = mount([() => { throw new Error('offline'); }, () => { throw new Error('offline'); }, () => pending.promise]);
  await flush();
  assert.equal([...app.timers.values()][0].delay, 2000);
  app.tick();
  await flush();
  assert.equal([...app.timers.values()][0].delay, 4000);
  app.tick();
  await flush();
  app.stop();
  pending.resolve();
  await flush();
  assert.notEqual(app.output, app.children);
  assert.equal(app.timers.size, 0);
});
