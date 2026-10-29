import { strict as assert } from 'node:assert';
import { EventEmitter } from 'node:events';
import { test } from 'node:test';
import backend from '../../desktop/backend.js';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import endpoint from '../../desktop/endpoint.js';

function stub(responses) {
  let calls = 0;
  const get = (url, callback) => {
    assert.equal(url.pathname, '/ready');
    const req = new EventEmitter();
    req.setTimeout = () => {};
    req.destroy = error => req.emit('error', error);
    const value = responses[Math.min(calls++, responses.length - 1)];
    queueMicrotask(() => {
      const res = new EventEmitter();
      res.statusCode = value.status;
      res.setEncoding = () => {};
      callback(res);
      res.emit('data', JSON.stringify(value.body));
      res.emit('end');
    });
    return req;
  };
  return { get, calls: () => calls };
}

const ready = { status: 200, body: { ready: true, pid: 123 } };

test('waits for readiness after a 503 without treating it as success', async () => {
  const source = stub([{ status: 503, body: { ready: false, pid: 123 } }, ready]);
  const result = await backend.waitForBackend('http://localhost', { get: source.get, intervalMs: 1, timeoutMs: 1000 });
  assert.equal(result.pid, 123);
  assert.equal(source.calls(), 2);
});

test('rejects a different HTTP app and a permanently unready engine', async () => {
  for (const response of [{ status: 200, body: { message: 'hello' } }, { status: 503, body: { ready: true, pid: 123 } }]) {
    const source = stub([response]);
    await assert.rejects(backend.waitForBackend('http://localhost', { get: source.get, intervalMs: 1, timeoutMs: 10 }));
  }
});

test('desktop uses the installed address, including wildcard and IPv6 listeners', () => {
  const home = mkdtempSync(join(tmpdir(), 'rune-endpoint-'));
  try {
    const env = { RUNE_HOME: home };
    assert.equal(endpoint.readEndpoint(env).origin, 'http://127.0.0.1:18789');
    writeFileSync(join(home, 'daemon.json'), JSON.stringify({ host: '0.0.0.0', port: 19876 }));
    assert.equal(endpoint.readEndpoint(env).origin, 'http://127.0.0.1:19876');
    assert.equal(endpoint.readEndpoint({ ...env, RUNE_API_HOST: '::' }).origin, 'http://[::1]:19876');
    assert.equal(endpoint.readEndpoint({ ...env, RUNE_API_PORT: '19877' }).port, 19877);
    writeFileSync(join(home, 'daemon.json'), '{broken');
    assert.throws(() => endpoint.readEndpoint(env));
    writeFileSync(join(home, 'daemon.json'), JSON.stringify({ host: 'localhost', port: true }));
    assert.throws(() => endpoint.readEndpoint(env));
  } finally { rmSync(home, { recursive: true, force: true }); }
});
