import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let server, TokenUsage, usageSummary;
before(async () => {
  server = await createServer({ server: { middlewareMode: true, hmr: false, ws: false, watch: null },
    appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ TokenUsage } = await server.ssrLoadModule('/src/components/TokenUsage.tsx'));
  ({ usageSummary } = await server.ssrLoadModule('/src/utils/usageSummary.ts'));
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

test('partial usage shows the known portion without calling it the total', () => {
  const html = renderToStaticMarkup(createElement(TokenUsage, {
    usage: { ...usage, modelCalls: 3, unreportedCalls: 1,
      cost: { ...usage.cost, usd: null, incomplete: true, unpricedCalls: 1 } },
  }));
  assert.ok(html.includes('Reported tokens'));
  assert.ok(html.includes('Model requests'));
  assert.ok(html.includes('Known portion'));
  assert.ok(html.includes('~$0.0026'));
  assert.ok(html.includes('Unavailable'));
  assert.ok(html.includes('Some usage was not fully reported'));
});

test('/cost uses recorded prices and does not report unknown usage as free', () => {
  assert.match(usageSummary(usage), /Estimated token cost: ~\$0.0026/);
  const partial = usageSummary({ ...usage, modelCalls: 3, unreportedCalls: 1,
    cost: { ...usage.cost, usd: null, incomplete: true, unpricedCalls: 1 } });
  assert.match(partial, /Reported tokens/);
  assert.match(partial, /Model requests: 3/);
  assert.match(partial, /Estimated token cost: unavailable/);
  assert.match(partial, /Known portion: ~\$0.0026/);
  assert.match(partial, /Some usage was not fully reported/);
  assert.match(usageSummary({ ...usage, cost: undefined }), /unavailable/);
  assert.match(usageSummary({ ...usage, cost: { ...usage.cost, usd: null, pending: true } }), /updating/);
});
