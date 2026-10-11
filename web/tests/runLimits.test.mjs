import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server, RunLimitSettings, parseRunLimits;
before(async () => {
  server = await createServer({ server: { middlewareMode: true, hmr: false, ws: false, watch: null },
    appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ RunLimitSettings, parseRunLimits } = await server.ssrLoadModule('/src/components/RunLimitSettings.tsx'));
});
after(async () => { await server?.close(); });

test('blank cost removes the limit while zero blocks model requests', () => {
  assert.deepEqual(parseRunLimits('100', ''), { modelRequests: 100, costUsd: null });
  assert.deepEqual(parseRunLimits('0', '0'), { modelRequests: 0, costUsd: 0 });
  assert.deepEqual(parseRunLimits('12', '0.025'), { modelRequests: 12, costUsd: .025 });
  for (const args of [['', ''], ['1.5', '1'], ['-1', ''], ['1', '-0.1'], ['1', 'Infinity'], ['1', 'NaN']]) {
    assert.throws(() => parseRunLimits(...args));
  }
});

test('the form distinguishes an estimate from an account spending cap', () => {
  const html = renderToStaticMarkup(createElement(RunLimitSettings, {
    limits: { modelRequests: 12, costUsd: null }, onSaved: async () => {},
  }));
  assert.ok(html.includes('value="12"'));
  assert.ok(html.includes('No cost limit'));
  assert.ok(html.includes('in-flight request can'));
  assert.ok(html.includes('not an account spending cap'));
});
