import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server, TokenUsage;
before(async () => {
  server = await createServer({ server: { middlewareMode: true, hmr: false, ws: false, watch: null },
    appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ TokenUsage } = await server.ssrLoadModule('/src/components/TokenUsage.tsx'));
});
after(async () => { await server?.close(); });

const usage = { total: 1100, input: 1000, output: 100,
  cost: { usd: .0026, knownUsd: .0026, unpricedCalls: 0, scope: 'model_tokens' } };

test('changing the selected provider does not reprice completed work', () => {
  const render = provider => renderToStaticMarkup(createElement(TokenUsage, { usage, provider }));
  assert.equal(render('xai'), render('anthropic'));
  assert.ok(render('xai').includes('~$0.0026'));
});

test('missing prices and old usage display unavailable, not zero', () => {
  for (const cost of [undefined, { usd: null, knownUsd: .0026, unpricedCalls: 1 }]) {
    const html = renderToStaticMarkup(createElement(TokenUsage, { usage: { ...usage, cost } }));
    assert.ok(html.includes('Unavailable'));
    assert.ok(!html.includes('$0.0000'));
  }
});

test('background accounting is marked pending until the total is known', () => {
  const html = renderToStaticMarkup(createElement(TokenUsage, {
    usage: { ...usage, cost: { ...usage.cost, usd: null, pending: true } },
  }));
  assert.ok(html.includes('Updating'));
  assert.ok(!html.includes('$0.0000'));
});
